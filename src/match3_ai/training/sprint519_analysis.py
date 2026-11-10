"""Joint board-paired Sprint 5.19 analysis, checkpoint auditing and reporting."""

import argparse
import hashlib
import json
from copy import deepcopy
from datetime import UTC, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import torch

from match3_ai.agents.dqn_agent import DQNAgent
from match3_ai.agents.ppo_agent import PPOAgent
from match3_ai.evaluation.analysis import wilson_interval
from match3_ai.evaluation.behavior_statistics import holm_adjust
from match3_ai.evaluation.fresh_protocol import sha256
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.training.overnight import late_learning_rate
from match3_ai.training.overnight_analysis import paired_statistics
from match3_ai.training.ppo_analysis import read_csv
from match3_ai.training.ppo_stability import distribution, trailing_mean, update_summary
from match3_ai.training.sprint519 import DQN_SEEDS, PPO_SEEDS, ROOT, verify


def label(spec):
    if spec["kind"] == "dqn":
        return "Dueling DDQN"
    return (
        "PPO LR decay"
        if spec["config"]["learning_rate_schedule"] == "late_linear"
        else "PPO constant"
    )


def group_records(rows, steps, boards, name):
    rows = [r for r in rows if r["agent_name"] == name]
    groups = {}
    for step in steps:
        selected = [r for r in rows if int(r["episode_id"].split(":")[-3]) == step]
        group = {int(r["seed"]): r for r in selected}
        if len(group) != len(selected) or set(group) != set(boards):
            raise ValueError("evaluation coverage differs from frozen protocol")
        if any(
            r["outcome"] not in ("WON", "LOST")
            or int(r["invalid_actions"])
            or r["error_type"]
            for r in selected
        ):
            raise ValueError("invalid or unfinished evaluation")
        groups[step] = group
    if len(rows) != len(steps) * len(boards):
        raise ValueError("unexpected evaluation checkpoint records")
    return groups


def metrics(group):
    rows = list(group.values())
    wins = sum(r["outcome"] == "WON" for r in rows)
    return {
        "wins": wins,
        "losses": len(rows) - wins,
        "win_rate": wins / len(rows),
        "wilson_ci95": wilson_interval(wins, len(rows)),
        **{
            k: float(np.mean([float(r[k]) for r in rows]))
            for k in (
                "objective_completion_ratio",
                "blocker_hp_remaining",
                "moves_used",
                "total_reward",
            )
        },
    }


def compare_groups(models, ids, steps, boards, settings):
    matrices = []
    for names, step in zip(ids, steps):
        matrices.append(
            np.array(
                [
                    [
                        models[name]["groups"][step][b]["outcome"] == "WON"
                        for b in boards
                    ]
                    for name in names
                ]
            )
        )
    return paired_statistics(
        *matrices, samples=settings["samples"], seed=settings["seed"]
    )


def completed_model(root, spec, boards):
    path = root / "runs" / spec["id"]
    summary = json.loads((path / "training_summary.json").read_text())
    if (
        summary["training_config"] != spec["config"]
        or summary["environment_steps"] != 100000
    ):
        raise ValueError("completed run differs from frozen budget or configuration")
    expected = 6252 if spec["kind"] == "ppo" else 24751
    if summary["optimizer_updates"] != expected:
        raise ValueError("optimizer budget differs")
    if any(
        summary.get(k, 0)
        for k in (
            "invalid_actions",
            "evaluation_invalid_actions",
            "evaluation_errors",
            "evaluation_time_limits",
        )
    ):
        raise ValueError("invalid completed run summary")
    steps = (
        [0, 50000, 75000, 100000]
        if spec["kind"] == "ppo"
        else [0, 25000, 50000, 75000, 100000]
    )
    groups = group_records(
        read_csv(path / "evaluation/episodes.csv"),
        steps,
        boards,
        "PPOAgent" if spec["kind"] == "ppo" else "DQNAgent",
    )
    model = {
        "id": spec["id"],
        "label": label(spec),
        "seed": spec["config"]["seed"],
        "curve": [{"step": step, **metrics(groups[step])} for step in steps],
        "groups": groups,
        "training_seconds": summary["training_seconds"],
        "evaluation_seconds": summary["evaluation_seconds"],
        "active_training_seconds": summary["collection_optimization_seconds"],
        "optimizer_updates": expected,
        "seed_metadata": summary["seed_metadata"],
        "checkpoint_hashes": {},
        "checkpoint_inference_checks": [],
    }
    if spec["kind"] == "ppo":
        updates = read_csv(path / "optimization.csv")
        rollouts = read_csv(path / "rollouts.csv")
        if len(updates) != 6252 or len(rollouts) != 391:
            raise ValueError("PPO log accounting differs")
        for i, row in enumerate(updates):
            wanted = (
                late_learning_rate(int(row["environment_step"]))
                if spec["config"]["learning_rate_schedule"] == "late_linear"
                else 0.0003
            )
            if int(row["optimizer_update"]) != i + 1 or not np.isclose(
                float(row["learning_rate"]), wanted, rtol=0, atol=1e-15
            ):
                raise ValueError(
                    "PPO optimizer index or learning-rate schedule differs"
                )
        with (path / "transitions.csv").open() as f:
            if sum(1 for _ in f) != 100001:
                raise ValueError("PPO transitions missing")
        intervals = []
        for low, high in ((0, 50000), (50000, 75000), (75000, 100000)):
            us = [r for r in updates if low < int(r["environment_step"]) <= high]
            rs = [r for r in rollouts if low < int(r["environment_step"]) <= high]
            intervals.append(
                {
                    "start": low,
                    "end": high,
                    "optimization": update_summary(us, threshold=0.5, coefficient=0.01),
                    "explained_variance": distribution(
                        r["explained_variance"] for r in rs
                    ),
                    "advantage_std": distribution(r["advantage_std"] for r in rs),
                }
            )
        model["late_diagnostics"] = intervals
        model["plot_updates"] = updates
        model["plot_rollouts"] = rollouts
        model["late_stability"] = {
            "signed_75k_to_100k": model["curve"][-1]["win_rate"]
            - model["curve"][-2]["win_rate"],
            "negative_part_regression": max(
                0.0, model["curve"][-2]["win_rate"] - model["curve"][-1]["win_rate"]
            ),
            "absolute_change": abs(
                model["curve"][-1]["win_rate"] - model["curve"][-2]["win_rate"]
            ),
        }
    else:
        records = read_csv(path / "training_metrics.csv")
        if (
            len(records) != 100000
            or summary["target_synchronizations"] != 24
            or not summary["finite_losses"]
        ):
            raise ValueError("DQN log or synchronization accounting differs")
        model["plot_updates"] = records
        if sum(r["td_loss"] != "" for r in records) != 24751 or any(
            r["valid_action"] != "True" or float(r["shaping_bonus"]) != 0
            for r in records
        ):
            raise ValueError("DQN raw updates, valid actions or sparse reward differ")
        model["target_synchronizations"] = 24
        model["process_seconds_at_50k"] = float(records[49999]["training_seconds"])
        model["dqn_diagnostics"] = {
            k: distribution(r[k] for r in records)
            for k in (
                "td_loss",
                "q_mean",
                "q_min",
                "q_max",
                "gradient_norm",
                "epsilon",
                "replay_size",
                "optimizer_updates",
                "target_synchronizations",
            )
        }
        if any(
            not np.isclose(float(r["epsilon"]), 0.05, rtol=0, atol=1e-14)
            for r in records[50000:]
        ):
            raise ValueError("exploration schedule changed beyond50k")
    if len(list(path.glob("*.pt"))) != len(steps) + 1:
        raise ValueError("checkpoint inventory differs from frozen schedule")
    return model


def audit_checkpoints(root, spec, model, observations):
    path = root / "runs" / spec["id"]
    moves = read_csv(path / "evaluation/moves.csv")
    steps = [r["step"] for r in model["curve"]]
    for step in steps:
        loader = PPOAgent if spec["kind"] == "ppo" else DQNAgent
        checkpoint = path / f"policy_step_{step}.pt"
        agent, payload = loader.load_checkpoint(checkpoint)
        network = agent.network if spec["kind"] == "ppo" else agent.online
        if json.loads(json.dumps(payload["training_config"])) != spec["config"]:
            raise ValueError("checkpoint config differs from protocol")
        if not all(torch.isfinite(p).all() for p in network.parameters()):
            raise ValueError("checkpoint weights nonfinite")
        before = [p.detach().clone() for p in network.parameters()]
        rng = (
            agent.policy_rng.get_state().clone()
            if spec["kind"] == "ppo"
            else deepcopy(agent.rng.bit_generator.state)
        )
        first = {
            int(r["episode_id"].split(":")[-1]): int(r["action_id"])
            for r in moves
            if int(r["episode_id"].split(":")[-3]) == step
            and int(r["move_index"]) == 1
            and r["episode_id"].split(":")[-2]
            == ("PPOAgent" if spec["kind"] == "ppo" else "DQNAgent")
        }
        for i, observation in enumerate(observations):
            action = agent.select_action(observation)
            if action != first[i] or not observation["action_mask"][action]:
                raise ValueError(
                    "initial-state deterministic checkpoint action differs"
                )
        if not all(torch.equal(p, q) for p, q in zip(network.parameters(), before)):
            raise ValueError("inference modified weights")
        if spec["kind"] == "ppo":
            if not torch.equal(rng, agent.policy_rng.get_state()):
                raise ValueError("inference modified sampling RNG")
        elif rng != agent.rng.bit_generator.state:
            raise ValueError("inference modified exploration RNG")
        model["checkpoint_hashes"][checkpoint.name] = sha256(checkpoint)
        model["checkpoint_inference_checks"].append(
            {
                "step": step,
                "initial_actions_checked": len(observations),
                "rng_and_weights_preserved": True,
            }
        )
    last = network
    final, payload = loader.load_checkpoint(path / "checkpoint_final.pt")
    final_network = final.network if spec["kind"] == "ppo" else final.online
    if not all(
        torch.equal(p, q) for p, q in zip(last.parameters(), final_network.parameters())
    ):
        raise ValueError("final checkpoint does not match100k milestone")
    if {int(v["step"]) for v in final.optimizer.state.values()} != {
        model["optimizer_updates"]
    }:
        raise ValueError("final Adam counter differs")
    model["checkpoint_hashes"]["checkpoint_final.pt"] = sha256(
        path / "checkpoint_final.pt"
    )


def analyze(root):
    root = Path(root)
    protocol = verify(root)
    output = root / "analysis"
    output.mkdir(exist_ok=False)
    execution = json.loads((root / "execution_complete.json").read_text())
    boards = protocol["validation_seeds"]
    models = {}
    inventory = []
    torch.set_num_threads(1)
    env = Match3LevelEnv(
        "blocker-mayhem", max_cascades=200, max_initialization_attempts=1000
    )
    try:
        observations = [env.reset(seed=b)[0] for b in boards]
    finally:
        env.close()
    for spec in protocol["runs"]:
        status = execution["statuses"].get(spec["id"], "not_started")
        inventory.append(
            {
                "id": spec["id"],
                "experiment": spec["experiment"],
                "status": status,
                "config": spec["config"],
            }
        )
        if status != "completed":
            continue
        model = completed_model(root, spec, boards)
        audit_checkpoints(root, spec, model, observations)
        models[spec["id"]] = model
    for board in boards:
        ids = {
            (
                group[board]["board_seed"],
                group[board]["engine_seed"],
                group[board]["level_config_id"],
            )
            for model in models.values()
            for group in model["groups"].values()
        }
        if len(ids) > 1:
            raise ValueError(
                "initial board identities differ across models/checkpoints"
            )
    prefixes = {}
    for seed in PPO_SEEDS:
        names = [f"a_constant_{seed}", f"a_late_linear_{seed}"]
        if not all(n in models for n in names):
            continue
        for step in (0, 50000):
            agents = [
                PPOAgent.load_checkpoint(
                    root / "runs" / name / f"policy_step_{step}.pt"
                )[0]
                for name in names
            ]
            if not all(
                torch.equal(a, b)
                for a, b in zip(
                    agents[0].network.parameters(), agents[1].network.parameters()
                )
            ):
                raise ValueError("PPO paired pre-intervention weights differ")
        traces = []
        for name in names:
            with (root / "runs" / name / "transitions.csv").open() as f:
                traces.append([next(f) for _ in range(50001)])
        if traces[0] != traces[1]:
            raise ValueError("PPO paired pre-intervention transitions differ")
        if models[names[0]]["seed_metadata"] != models[names[1]]["seed_metadata"]:
            raise ValueError("paired RNG streams differ")
        prefixes[str(seed)] = {
            "initial_and_50k_weights_equal": True,
            "first50000_transition_records_equal": True,
            "seed_streams_equal": True,
        }
    comparisons = {}
    specifications = [
        (
            "A",
            [f"a_late_linear_{s}" for s in PPO_SEEDS],
            [f"a_constant_{s}" for s in PPO_SEEDS],
            100000,
            100000,
        ),
        (
            "B",
            [f"b_dueling_{s}" for s in DQN_SEEDS],
            [f"b_dueling_{s}" for s in DQN_SEEDS],
            100000,
            50000,
        ),
    ]
    primary_p = []
    individual = []
    settings = protocol["primary_statistics"]
    for name, left, right, lstep, rstep in specifications:
        if all(n in models for n in left + right):
            stat = compare_groups(
                models, (left, right), (lstep, rstep), boards, settings
            )
            stat["status"] = "estimated"
            primary_p.append(stat["conditional_board_p"])
            individual.extend(stat["individual"])
        else:
            stat = {
                "status": "not_estimable",
                "reason": "incomplete planned cohort; no exclusion or substitution",
            }
            primary_p.append(1.0)
            individual.extend([None] * 3)
        comparisons[name] = stat
    for stat, p in zip(comparisons.values(), holm_adjust(primary_p)):
        stat["holm_p_primary_two"] = p if stat["status"] == "estimated" else None
    adjusted = holm_adjust(
        [r["mcnemar_p"] if r is not None else 1.0 for r in individual]
    )
    for row, p in zip(individual, adjusted):
        if row is not None:
            row["holm_p_exploratory_six"] = p
    rng = np.random.default_rng(settings["seed"])
    digest = hashlib.sha256()
    for start in range(0, settings["samples"], 250):
        indices = rng.integers(
            len(boards), size=(min(250, settings["samples"] - start), len(boards))
        )
        digest.update(indices.astype(np.int64).tobytes())
    aggregates = {}
    for name in ("PPO constant", "PPO LR decay", "Dueling DDQN"):
        selected = [m for m in models.values() if m["label"] == name]
        if not selected:
            continue
        rates = [m["curve"][-1]["win_rate"] for m in selected]
        aggregates[name] = {
            "n_completed_models": len(selected),
            "mean_final_win_rate": float(np.mean(rates)),
            "training_seed_sample_std": float(np.std(rates, ddof=1))
            if len(rates) > 1
            else None,
            "mean_process_seconds": float(
                np.mean([m["training_seconds"] for m in selected])
            ),
            "mean_objective_completion": float(
                np.mean(
                    [m["curve"][-1]["objective_completion_ratio"] for m in selected]
                )
            ),
            "mean_remaining_hp": float(
                np.mean([m["curve"][-1]["blocker_hp_remaining"] for m in selected])
            ),
        }
        if name != "Dueling DDQN":
            aggregates[name]["late_stability"] = {
                k: distribution(m["late_stability"][k] for m in selected)
                for k in selected[0]["late_stability"]
            }
    result = {
        "study": "sprint5_19_rl_optimization",
        "manifest_sha256": sha256(root / "manifest.json"),
        "inventory": inventory,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "ppo_prefix_checks": prefixes,
        "joint_board_index_sha256": digest.hexdigest(),
        "statistical_scope": "conditional board uncertainty for three observed pairs; training-seed variability separate; no robust population superiority or equivalence claim",
        "historical_reference": json.loads(
            Path(
                "experiments/results/overnight_rl_research/final_summary.json"
            ).read_text()
        )["aggregates"],
        "final_test_evaluation": False,
        "reset_only_integrity_boards": len(boards),
        "additional_full_evaluation_episodes": 0,
    }
    result["figures"] = figures(models, output / "figures")
    result["models"] = {
        name: {
            k: v
            for k, v in model.items()
            if k not in ("groups", "plot_updates", "plot_rollouts")
        }
        for name, model in models.items()
    }
    result["highest_fixed_endpoint"] = max(
        (
            {
                "id": name,
                "win_rate": m["curve"][-1]["win_rate"],
                "checkpoint": str(root / "runs" / name / "checkpoint_final.pt"),
            }
            for name, m in models.items()
        ),
        key=lambda r: r["win_rate"],
        default=None,
    )
    result["highest_scheduled_checkpoint"] = max(
        (
            {"id": name, "step": row["step"], "win_rate": row["win_rate"]}
            for name, m in models.items()
            for row in m["curve"]
        ),
        key=lambda r: r["win_rate"],
        default=None,
    )
    history = json.loads((root / "history_inventory.json").read_text())
    if any(sha256(n) != d for n, d in history["files_sha256"].items()):
        raise ValueError("historical artifact changed")
    result["historical_artifacts_verified"] = len(history["files_sha256"])
    verify(root)
    with (output / "summary.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    with (output / "per_seed_results.json").open("x") as f:
        json.dump(result["models"], f, indent=2, allow_nan=False)
    with (output / "statistics.json").open("x") as f:
        json.dump(comparisons, f, indent=2, allow_nan=False)
    with (output / "checkpoint_integrity.json").open("x") as f:
        json.dump(
            {
                k: {
                    "hashes": m["checkpoint_hashes"],
                    "checks": m["checkpoint_inference_checks"],
                }
                for k, m in models.items()
            },
            f,
            indent=2,
        )
    return result


def figures(models, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(exist_ok=False)
    files = []

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            fig.savefig(output / filename, dpi=150)
            files.append(filename)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, seed in zip(axes, PPO_SEEDS):
        for mode in ("constant", "late_linear"):
            name = f"a_{mode}_{seed}"
            if name in models:
                curve = models[name]["curve"]
                ax.plot(
                    [r["step"] for r in curve],
                    [r["win_rate"] for r in curve],
                    "o-",
                    label=mode,
                )
        ax.set(
            title=f"PPO pair{seed}", xlabel="Collected transitions", ylabel="Win rate"
        )
        if ax.lines:
            ax.legend()
    save(fig, "ppo_learning_by_pair")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for seed in PPO_SEEDS:
        a = models.get(f"a_constant_{seed}")
        b = models.get(f"a_late_linear_{seed}")
        if not a or not b:
            continue
        axes[0].plot(
            [0, 1],
            [a["curve"][-1]["win_rate"], b["curve"][-1]["win_rate"]],
            "o-",
            label=str(seed),
        )
        axes[1].plot(
            [0, 1],
            [
                a["late_stability"]["signed_75k_to_100k"],
                b["late_stability"]["signed_75k_to_100k"],
            ],
            "o-",
            label=str(seed),
        )
        axes[2].plot(
            [0, 1],
            [
                a["late_stability"]["negative_part_regression"],
                b["late_stability"]["negative_part_regression"],
            ],
            "o-",
            label=str(seed),
        )
    for ax, title in zip(
        axes,
        ("Fixed100k endpoint", "Signed75k→100k change", "Negative-part regression"),
    ):
        ax.set_xticks([0, 1], ["constant", "LR decay"])
        ax.set_title(title)
        if ax.lines:
            ax.legend()
    save(fig, "ppo_endpoint_and_late_stability")
    fig, axes = plt.subplots(6, 1, figsize=(12, 13), sharex=True)
    for name, m in models.items():
        if not name.startswith("a_"):
            continue
        for ax, key in zip(
            axes,
            (
                "approximate_kl",
                "clip_fraction",
                "entropy",
                "value_loss",
                "gradient_norm",
                "learning_rate",
            ),
        ):
            rows = m["plot_updates"]
            x = [int(r["environment_step"]) for r in rows]
            y = np.array([float(r[key]) for r in rows])
            ax.plot(
                x, y if key == "learning_rate" else trailing_mean(y), lw=0.8, label=name
            )
            ax.set_ylabel(key)
            if key == "learning_rate":
                ax.set_ylim(0, 0.00033)
    if axes[0].lines:
        axes[0].legend(fontsize=6)
    axes[-1].set_xlabel(
        "Collected transitions; trailing50 optimizer steps except raw LR"
    )
    save(fig, "ppo_optimization_diagnostics")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for seed in DQN_SEEDS:
        name = f"b_dueling_{seed}"
        if name not in models:
            continue
        curve = models[name]["curve"]
        axes[0].plot(
            [r["step"] for r in curve],
            [r["win_rate"] for r in curve],
            "o-",
            label=str(seed),
        )
        axes[1].plot(
            [50000, 100000],
            [curve[2]["win_rate"], curve[-1]["win_rate"]],
            "o-",
            label=str(seed),
        )
        axes[2].plot(
            [r["step"] for r in curve],
            [r["blocker_hp_remaining"] for r in curve],
            "o-",
            label=str(seed),
        )
    for ax, title in zip(
        axes,
        ("Dueling learning curves", "Paired50k versus100k", "Remaining blocker HP"),
    ):
        ax.set(title=title, xlabel="Collected transitions")
        if ax.lines:
            ax.legend()
    save(fig, "dueling_budget_by_seed")
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
    for name, m in models.items():
        if not name.startswith("b_"):
            continue
        for ax, key in zip(axes, ("td_loss", "q_mean", "epsilon", "replay_size")):
            rows = [r for r in m["plot_updates"] if r[key] != ""]
            ax.plot(
                [int(r["environment_step"]) for r in rows],
                trailing_mean([float(r[key]) for r in rows]),
                label=name,
                lw=0.8,
            )
            ax.set_ylabel(key)
    if axes[0].lines:
        axes[0].legend(fontsize=6)
    axes[-1].set_xlabel("Collected transitions; trailing50 recorded values")
    save(fig, "dueling_diagnostics")
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, m in models.items():
        axes[0].scatter(m["label"], m["curve"][-1]["win_rate"])
        axes[1].scatter(
            m["training_seconds"] / 60, m["curve"][-1]["win_rate"], label=name
        )
        if name.startswith("b_"):
            axes[1].scatter(
                m["process_seconds_at_50k"] / 60, m["curve"][2]["win_rate"], marker="x"
            )
    axes[0].set(title="Per-seed100k endpoints", ylabel="Win rate")
    axes[0].tick_params(axis="x", rotation=20)
    axes[1].set(
        xlabel="Process minutes including validation",
        ylabel="Win rate",
        title="Compute trade-off; x marks within-run dueling50k",
    )
    if axes[1].collections:
        axes[1].legend(fontsize=6)
    save(fig, "endpoint_and_compute")
    return files


def finalize(root):
    root = Path(root)
    protocol = verify(root)
    result = json.loads((root / "analysis/summary.json").read_text())
    validation = json.loads((root / "validation.json").read_text())
    before = json.loads((root / "workspace_before.json").read_text())
    changed = [n for n, d in before.items() if not Path(n).is_file() or sha256(n) != d]
    allowed = {
        "src/match3_ai/training/ppo.py",
        "src/match3_ai/training/dqn.py",
        "src/match3_ai/training/seed_policy.py",
    }
    if set(changed) - allowed:
        raise ValueError(f"unrelated pre-existing work changed: {changed}")
    result.update(
        validation=validation,
        preexisting_file_integrity={
            "files_checked": len(before),
            "only_intended_changes": changed,
        },
        finalized_utc=datetime.now(UTC).isoformat(),
        within_deadline=datetime.now(UTC).timestamp()
        <= datetime.fromisoformat(protocol["deadline_utc"]).timestamp(),
    )
    result["retention"] = (
        "retain every checkpoint including failures; highest fixed endpoint is a development-selected candidate, not held-out evidence"
    )
    with (root / "final_summary.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    report = report_text(root, protocol, result)
    with (root / "final_report.md").open("x") as f:
        f.write(report)
    return result


def report_text(root, protocol, result):
    local_start = datetime.fromisoformat(protocol["actual_start_utc"]).astimezone(
        ZoneInfo("Europe/Istanbul")
    )
    local_deadline = datetime.fromisoformat(protocol["deadline_utc"]).astimezone(
        ZoneInfo("Europe/Istanbul")
    )
    lines = [
        "# Sprint 5.19 — Controlled PPO optimization and dueling DDQN budget",
        "",
        f"Frozen manifest SHA-256: `{result['manifest_sha256']}`. Actual start {protocol['actual_start_utc']}; deadline {protocol['deadline_utc']}. Finalized {result['finalized_utc']}; within deadline: {result['within_deadline']}.",
        f"Istanbul execution window: {local_start.isoformat()} to {local_deadline.isoformat()}.",
        "",
        "The study uses 200 fresh development boards, 310000–310199, disjoint from all previously recorded seeds. This explicit study block leaves the legacy training/validation/final-test contracts unchanged. No reserved final-test board was evaluated.",
        "",
        "## Findings",
        "",
        "| Variant | Completed seeds | Mean 100k win rate | Training-seed SD (pp) | Mean process minutes | Objective completion | Remaining HP |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, r in result["aggregates"].items():
        sd = (
            "n/a"
            if r["training_seed_sample_std"] is None
            else f"{100 * r['training_seed_sample_std']:.2f}"
        )
        lines.append(
            f"| {name} | {r['n_completed_models']} | {r['mean_final_win_rate']:.2%} | {sd} | {r['mean_process_seconds'] / 60:.2f} | {r['mean_objective_completion']:.2%} | {r['mean_remaining_hp']:.3f} |"
        )
    lines += [
        "",
        "| Primary hypothesis | Difference(pp) | Pointwise conditional95% CI(pp) | Raw p | Holm(two) p |",
        "|---|---:|---|---:|---:|",
    ]
    for name, stat in result["comparisons"].items():
        if stat["status"] != "estimated":
            lines.append(f"| {name} | unavailable: incomplete cohort | — | — | — |")
            continue
        lo, hi = stat["conditional_board_ci95"]
        lines.append(
            f"| {name} | {100 * stat['difference']:+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] | {stat['conditional_board_p']:.6f} | {stat['holm_p_primary_two']:.6f} |"
        )
    lines += [
        "",
        "A compares PPO LR decay minus constant at fixed100k. B compares the same new dueling runs at100k minus50k. Both were preregistered; one Holm correction covers both. Pointwise CIs are not multiplicity-adjusted. Bootstrap uses20000 replicates and seed519519 with common board draws across all policies/checkpoints; its index digest is saved. Raw p-values are approximate null-centered paired board-bootstrap values, not exact permutation tests.",
        "",
        "These intervals condition on the three observed training pairs. Per-seed variation and sign-flip sensitivity do not establish population-level superiority; with three pairs the minimum two-sided sign-flip p is.25. Nonsignificance is not equivalence. A development score above70% is not held-out generalization.",
        "",
        "## Per-seed outcomes",
        "",
        "| Run | Checkpoint wins/200 in collected-transition order | Total process seconds | Validation seconds |",
        "|---|---|---:|---:|",
    ]
    for name, m in result["models"].items():
        curve = "; ".join(
            f"{r['step']}: {r['wins']} ({r['win_rate']:.1%})" for r in m["curve"]
        )
        lines.append(
            f"| {name} | {curve} | {m['training_seconds']:.2f} | {m['evaluation_seconds']:.2f} |"
        )
    lines += ["", "## Interpretation and late stability", ""]
    for name, stat in result["comparisons"].items():
        if stat["status"] == "estimated":
            conclusion = (
                "positive conditional evidence after correction"
                if stat["difference"] > 0 and stat["holm_p_primary_two"] < 0.05
                else "no positive effect established after correction"
            )
            lines.append(
                f"Primary{name}: {conclusion}. Individual effects(pp): {[round(100 * x, 2) for x in stat['seed_pair_effects']]}; pair-effect SD {100 * stat['training_pair_sample_std']:.2f}pp; training-pair sign-flip sensitivity p={stat['training_pair_signflip_sensitivity_p']:.3f}."
            )
    for name, aggregate in result["aggregates"].items():
        if "late_stability" in aggregate:
            s = aggregate["late_stability"]
            lines.append(
                f"{name}: mean75k→100k signed change {100 * s['signed_75k_to_100k']['mean']:+.2f}pp; negative-part regression {100 * s['negative_part_regression']['mean']:.2f}pp; absolute change {100 * s['absolute_change']['mean']:.2f}pp. These are exploratory descriptive summaries."
            )
    if all(name in result["aggregates"] for name in ("PPO constant", "PPO LR decay")):
        a = result["aggregates"]["PPO constant"]["late_stability"][
            "negative_part_regression"
        ]["mean"]
        b = result["aggregates"]["PPO LR decay"]["late_stability"][
            "negative_part_regression"
        ]["mean"]
        lines.append(
            f"Decay-minus-control mean negative-part regression is {100 * (b - a):+.2f} pp: {'lower observed regression' if b < a else 'no observed regression reduction'}. This descriptive result from three pairs is not a confirmatory stability test."
        )
    lines += [
        "",
        "| PPO run / interval | KL | Clip fraction | Entropy | Value loss | Explained variance | Gradient clipping frequency |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, model in result["models"].items():
        for interval in model.get("late_diagnostics", [])[1:]:
            opt = interval["optimization"]
            ev = interval["explained_variance"].get("mean")
            ev_text = "undefined" if ev is None else f"{ev:.4f}"
            lines.append(
                f"| {name}: {interval['start']}–{interval['end']} | {opt['approximate_kl']['mean']:.5f} | {opt['clip_fraction']['mean']:.4f} | {opt['entropy']['mean']:.4f} | {opt['value_loss']['mean']:.4f} | {ev_text} | {opt['gradient_clipping_frequency']:.2%} |"
            )
    lines += [
        "",
        "Lower KL or smaller gradients alone do not establish better performance or stability. KL, clip fraction, entropy, value loss, pre-update trajectory-dependent EV and raw advantage summaries are retained by interval. Gradient norms are the total L2 norms returned before clipping; clipping frequency is count(norm>.5)/optimizer steps. The entropy coefficient remains.01. Learning-rate trajectories verify the sole PPO intervention. Associations across three pairs and sparse checkpoints cannot identify a causal diagnostic mechanism.",
        "",
        "Dueling TD loss/Q statistics, epsilon, replay size, updates, target synchronizations and gradients remain in raw CSVs and summary distributions. Epsilon stays.05 after50k, replay capacity remains50000, and target synchronization occurs every1000 optimizer updates. Different on-policy or replay states make loss magnitudes imperfect quality comparisons.",
        "",
        "## Strongest checkpoint and compute trade-off",
        "",
    ]
    best = result["highest_fixed_endpoint"]
    if best:
        lines.append(
            f"Highest fixed100k endpoint: {best['id']}, {best['win_rate']:.2%}, `{best['checkpoint']}`. Keep all final and intermediate checkpoints. This candidate is development-selected."
        )
    peak = result["highest_scheduled_checkpoint"]
    if peak:
        lines.append(
            f"Highest scheduled checkpoint, descriptive only: {peak['id']} at{peak['step']}, {peak['win_rate']:.2%}. Aspirational70% reached by a measured checkpoint: {peak['win_rate'] >= 0.70}."
        )
    if result["aggregates"]:
        observed = max(
            result["aggregates"],
            key=lambda name: (
                result["aggregates"][name]["mean_final_win_rate"]
                / (result["aggregates"][name]["mean_process_seconds"] / 60)
            ),
        )
        lines.append(
            f"Largest observed endpoint win-rate/process-minute ratio among100k cohorts: {observed}. This simple descriptive ratio is not a statistically controlled compute-efficiency claim; budgets, optimizer counts and initial models differ. Within-run dueling50k cumulative time is also plotted. Process time includes evaluation/checkpoint work and contention; sums are not elapsed batch time or measured CPU time."
        )
    lines += [
        "",
        "## Protocol and reproducibility",
        "",
        "Three new PPO master-seed pairs5970–5972 use equivalent initial parameters and controlled environment/model/policy/shuffle streams. Identical0/50k weights and the first50000 transition records are audited. LR is.0003 through50k, then linearly declines by collected rollout completion to.00003 at100k; each rollout shares its rate across all epochs. PPO intermediate policies optimize through49920/74752 at50k/75k, retaining80/248 pending samples. Final100k includes the160-sample tail:391 rollouts and6252 Adam steps/run. Scheduled PPO checkpoints are exactly0,50k,75k,100k.",
        "",
        "Dueling masters5980–5982 initialize new networks/replay. Old50k checkpoints include online/target weights, optimizer, replay/sampling RNG, exploration RNG and counters, but omit the active environment and in-progress episode. Their budget-truncated final sample can also differ from a continuous100k run. Exact continuation is unsupported; no model was resumed. The new within-run50k checkpoint is the matched control. Each100k run has24751 updates and24 target synchronizations, with checkpoints0/25k/50k/75k/100k.",
        "",
        "Dueling runs use private model/exploration/replay streams under a common deterministic episode-seed curriculum starting200000. Reported training variability is conditional on that curriculum; shared environment seeds do not make the three initial models identical. Training and validation seeds are disjoint. Sparse rewards, legal masks, architecture, replay/target settings, game rules and guards are unchanged.",
        "",
        "The first smoke attempt failed configuration validation before any transitions because default baseline agent seeds overlapped the training partition. It is preserved. A separate successful attempt used explicit independent baseline seeds. Both PPO and dueling smoke runs collected512 transitions, optimized32/113 steps, exported actual evaluation metadata and statuses, saved/reloaded checkpoints and wrote summaries. Smoke outcomes are excluded from research estimates and used legacy validation boards, not the fresh production set.",
        "",
        "## Historical context, disclosure and limitations",
        "",
        "The overnight199-board means were dueling63.65%, standardDDQN56.95%, historicalPPO52.60%, Greedy52.26%, Random46.73%, and historicalVanillaDQN46.23%. Those checkpoints/boards/training cohorts are different. These values are context only, not paired controls for this sprint. No historical result was overwritten or substituted.",
        "",
        "The six individual endpoint tests have a separate exploratory six-slot Holm family; raw/adjusted McNemar p-values and discordant tables are in statistics.json. Missing planned comparisons retain non-rejection correction slots and are never silently replaced. Other checkpoints and diagnostic correlations are descriptive. Three training seeds, reused development boards and candidate selection limit generalization claims. No reserved final-test evaluation was performed.",
        "",
        "Publish a concise protocol/results/limitations report, source changes and reproducibility commands after a separate user-approved Git review. Include negative results, per-seed tables, confidence intervals and failures; do not publish a70%/superiority claim from a selected development score. Do not commit large checkpoint/replay artifacts blindly; preserve them locally and describe their provenance. No Git publication operation is performed here.",
        "",
        "The experiment and artifact pipeline can support a Human Gameplay/Data Science Analytics handoff, but mature product infrastructure is distinct from verified RL superiority. Broader seed replication and genuinely independent evaluation would be needed for strong performance claims. Do not automatically expand RL experiments to chase70%, and do not begin the next phase in this sprint.",
        "",
        "## Inventory, validation and artifacts",
        "",
        "| Job | Status |",
        "|---|---|",
    ]
    lines.extend(f"| {row['id']} | {row['status']} |" for row in result["inventory"])
    lines += [
        "",
        f"Historical artifacts verified unchanged: {result['historical_artifacts_verified']}. Only intended existing files changed: {result['preexisting_file_integrity']['only_intended_changes']}. All completed policy checkpoints load, match frozen config, retain finite parameters, reproduce200 recorded initial decisions, preserve inference RNG/weights and match final Adam counters. These are reset-only/initial-decision checks, not additional full evaluation episodes.",
        "",
        "Validation statuses: " + json.dumps(result["validation"], sort_keys=True),
        "",
        f"Machine-readable data: `{root}/final_summary.json`, `analysis/per_seed_results.json`, `analysis/statistics.json`, `analysis/checkpoint_integrity.json`, frozen manifest/inventory, actual smoke outputs, per-run append-only events/status/config/provenance, raw training/evaluation CSVs and source archive. All PNG/PDF figures are reproducible from `python -m match3_ai.training.sprint519_analysis analyze --root <fresh-root-with-manifest-and-runs>`; existing output directories are rejected. Original raw optimizer data remain available; plots use trailing50 recorded values except unsmoothed LR, with no smoothing across runs.",
        "",
        "No staging, committing, pushing, resetting, rebasing, stashing or discarding work occurred. Failed and partial attempts remain preserved. No Human Gameplay or Data Science Analytics implementation was started.",
    ]
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("analyze", "finalize"))
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args()
    (analyze if args.mode == "analyze" else finalize)(args.root)


if __name__ == "__main__":
    main()
