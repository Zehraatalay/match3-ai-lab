"""Paired board analysis, checkpoint audits and figures for the overnight manifest."""

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import torch

from match3_ai.agents.dqn_agent import DQNAgent
from match3_ai.agents.ppo_agent import PPOAgent
from match3_ai.evaluation.analysis import exact_mcnemar, wilson_interval
from match3_ai.evaluation.behavior_statistics import holm_adjust
from match3_ai.evaluation.fresh_protocol import sha256
from match3_ai.training.overnight import ROOT, verify_manifest
from match3_ai.training.ppo_analysis import read_csv
from match3_ai.training.ppo_stability import trailing_mean


def paired_statistics(left, right, *, samples=20000, seed=51919):
    left, right = np.asarray(left, float), np.asarray(right, float)
    if left.shape != right.shape or left.ndim != 2 or not left.size:
        raise ValueError("paired policies require matching seed-pair by board matrices")
    if not np.isin(left, [0, 1]).all() or not np.isin(right, [0, 1]).all():
        raise ValueError("paired outcomes must be binary")
    differences = left - right
    board_mean = differences.mean(0)
    estimate = float(board_mean.mean())
    rng = np.random.default_rng(seed)
    draws = []
    for start in range(0, samples, 250):
        indices = rng.integers(
            left.shape[1], size=(min(250, samples - start), left.shape[1])
        )
        draws.extend(board_mean[indices].mean(-1).tolist())
    draws = np.asarray(draws)
    pairs = differences.mean(1)
    signs = np.array(list(itertools.product((-1, 1), repeat=len(pairs))))
    p = (1 + int((np.abs(draws - estimate) >= abs(estimate) - 1e-12).sum())) / (
        samples + 1
    )
    individual = []
    for a, b in zip(left.astype(bool), right.astype(bool)):
        later, earlier = int((a & ~b).sum()), int((~a & b).sum())
        individual.append(
            {
                "difference": float((a.astype(float) - b).mean()),
                "counts": {
                    "both_win": int((a & b).sum()),
                    "treatment_only": later,
                    "control_only": earlier,
                    "both_lose": int((~a & ~b).sum()),
                },
                "mcnemar_p": exact_mcnemar(later, earlier),
            }
        )
    return {
        "difference": estimate,
        "conditional_board_ci95": np.quantile(draws, [0.025, 0.975]).tolist(),
        "conditional_board_p": p,
        "seed_pair_effects": pairs.tolist(),
        "training_pair_sample_std": float(pairs.std(ddof=1))
        if len(pairs) > 1
        else None,
        "training_pair_signflip_sensitivity_p": float(
            (np.abs((signs * pairs).mean(-1)) >= abs(estimate) - 1e-12).mean()
        ),
        "signflip_assumption": "pairwise sign symmetry/exchangeability under null; only 3 independent pairs, minimum attainable p=.25",
        "individual": individual,
        "n_boards": left.shape[1],
        "n_training_pairs": len(pairs),
        "scope": "board uncertainty conditional on observed models, not population uncertainty over training seeds",
    }


def label(spec):
    if spec["kind"] == "ppo":
        return (
            "PPO LR decay"
            if spec["config"]["learning_rate_schedule"] == "late_linear"
            else "PPO constant"
        )
    if spec["kind"] == "dqn":
        return (
            "Dueling DDQN"
            if spec["config"]["architecture"] == "dueling"
            else "Double DQN"
        )
    return {
        "vanilla": "Vanilla DQN (historically trained)",
        "historical_ppo": "PPO (historically trained)",
        "random": "Random",
        "greedy": "Greedy",
    }[spec["kind"]]


def episode_summary(group):
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
        "invalid_actions": sum(int(r["invalid_actions"]) for r in rows),
    }


def analyze(root=ROOT):
    root = Path(root)
    manifest = verify_manifest(root)
    output = root / "analysis"
    output.mkdir(exist_ok=False)
    completed = json.loads((root / "execution_complete.json").read_text())
    inventory, models, cubes, training = [], {}, {}, {}
    for spec in manifest["runs"]:
        state = completed["statuses"].get(spec["id"], "not_started")
        inventory.append({"id": spec["id"], "stage": spec["stage"], "status": state})
        if state != "completed" or "smoke" in spec["stage"]:
            continue
        path = root / "runs" / spec["id"]
        rows = read_csv(path / "evaluation/episodes.csv")
        is_training = spec["kind"] in ("ppo", "dqn")
        if is_training:
            name = "PPOAgent" if spec["kind"] == "ppo" else "DQNAgent"
            rows = [r for r in rows if r["agent_name"] == name]
        milestones = (
            sorted({int(r["episode_id"].split(":")[-3]) for r in rows})
            if is_training
            else [0]
        )
        groups = {}
        for step in milestones:
            subset = (
                [r for r in rows if int(r["episode_id"].split(":")[-3]) == step]
                if is_training
                else rows
            )
            group = {int(r["seed"]): r for r in subset}
            if len(group) != len(subset) or set(group) != set(
                manifest["validation_seeds"]
            ):
                raise ValueError("evaluation coverage differs from manifest")
            if any(
                r["outcome"] not in ("WON", "LOST") or int(r["invalid_actions"])
                for r in subset
            ):
                raise ValueError("incomplete or invalid evaluation")
            groups[step] = group
        if is_training and milestones != [
            0,
            spec["config"]["environment_steps"] // 2,
            spec["config"]["environment_steps"],
        ]:
            raise ValueError("checkpoint schedule differs from manifest")
        final = groups[max(groups)]
        models[spec["id"]] = {
            "label": label(spec),
            "seed": spec["config"].get("seed"),
            "curve": [
                {"step": step, **episode_summary(g)} for step, g in groups.items()
            ],
            "final": episode_summary(final),
        }
        cubes[spec["id"]] = final
        if is_training:
            summary = json.loads((path / "training_summary.json").read_text())
            if summary["environment_steps"] != spec["config"]["environment_steps"]:
                raise ValueError("training budget incomplete")
            models[spec["id"]]["training_seconds"] = summary["training_seconds"]
            models[spec["id"]]["evaluation_seconds"] = summary["evaluation_seconds"]
            models[spec["id"]]["optimizer_updates"] = summary["optimizer_updates"]
            checkpoints = {}
            for checkpoint in sorted(path.glob("*.pt")):
                loader = PPOAgent if spec["kind"] == "ppo" else DQNAgent
                agent, payload = loader.load_checkpoint(checkpoint)
                network = agent.network if spec["kind"] == "ppo" else agent.online
                if not all(torch.isfinite(p).all() for p in network.parameters()):
                    raise ValueError("nonfinite checkpoint weights")
                if json.loads(json.dumps(payload["training_config"])) != spec["config"]:
                    raise ValueError("checkpoint config differs from frozen manifest")
                checkpoints[checkpoint.name] = sha256(checkpoint)
            if len(checkpoints) != 4:
                raise ValueError(
                    "expected three milestone policies and one final checkpoint"
                )
            models[spec["id"]]["checkpoint_hashes"] = checkpoints
            training[spec["id"]] = (path, spec)
        else:
            models[spec["id"]]["evaluation_seconds"] = json.loads(
                (path / "baseline_summary.json").read_text()
            )["evaluation_seconds"]
    for board in manifest["validation_seeds"]:
        identities = {
            (
                g[board]["board_seed"],
                g[board]["engine_seed"],
                g[board]["level_config_id"],
            )
            for g in cubes.values()
        }
        if len(identities) > 1:
            raise ValueError("paired initial boards differ across algorithms")
    comparisons = {}
    primary_left = [f"a_ppo_late_linear_{s}" for s in (5930, 5931, 5932)]
    primary_right = [f"a_ppo_constant_{s}" for s in (5930, 5931, 5932)]
    duel_left = [f"c_dueling_{s}" for s in (5950, 5951, 5952)]
    duel_right = [f"b_mlp_{s}" for s in (5950, 5951, 5952)]
    exploratory = []
    for name, left, right in (
        ("primary_ppo_lr", primary_left, primary_right),
        ("exploratory_dueling", duel_left, duel_right),
    ):
        if not all(n in cubes for n in left + right):
            comparisons[name] = {
                "status": "not_estimable",
                "reason": "incomplete planned cohort; no seed substitution",
            }
            continue
        arrays = [
            np.array(
                [
                    [
                        cubes[n][b]["outcome"] == "WON"
                        for b in manifest["validation_seeds"]
                    ]
                    for n in names
                ]
            )
            for names in (left, right)
        ]
        result = paired_statistics(
            *arrays,
            samples=manifest["bootstrap"]["samples"],
            seed=manifest["bootstrap"]["seed"],
        )
        comparisons[name] = result
        if name == "primary_ppo_lr":
            result["holm_p_primary_singleton"] = result["conditional_board_p"]
        else:
            exploratory.append((result, "conditional_board_p"))
        exploratory.extend((row, "mcnemar_p") for row in result["individual"])
    if len(exploratory) == 7:
        for (row, _), p in zip(
            exploratory, holm_adjust([r[k] for r, k in exploratory])
        ):
            row["holm_p_exploratory_seven"] = p
    aggregates = {}
    for name in sorted({r["label"] for r in models.values()}):
        selected = [r for r in models.values() if r["label"] == name]
        rates = [r["final"]["win_rate"] for r in selected]
        aggregates[name] = {
            "mean_win_rate": float(np.mean(rates)),
            "training_seed_sample_std": float(np.std(rates, ddof=1))
            if len(rates) > 1
            else None,
            "n_models": len(rates),
            "max_observed": max(rates),
            "objective_completion_ratio": float(
                np.mean([r["final"]["objective_completion_ratio"] for r in selected])
            ),
            "blocker_hp_remaining": float(
                np.mean([r["final"]["blocker_hp_remaining"] for r in selected])
            ),
        }
    checks = {}
    for seed in (5930, 5931, 5932):
        control, treatment = f"a_ppo_constant_{seed}", f"a_ppo_late_linear_{seed}"
        if control in training and treatment in training:
            paths = [training[k][0] for k in (control, treatment)]
            equal = []
            for step in (0, 50000):
                agents = [
                    PPOAgent.load_checkpoint(p / f"policy_step_{step}.pt")[0]
                    for p in paths
                ]
                equal.append(
                    all(
                        torch.equal(a, b)
                        for a, b in zip(
                            agents[0].network.parameters(),
                            agents[1].network.parameters(),
                        )
                    )
                )
            if not all(equal):
                raise ValueError(
                    "paired PPO initialization or pre-intervention prefix differs"
                )
            prefixes = []
            for path in paths:
                with (path / "transitions.csv").open() as handle:
                    prefixes.append([next(handle) for _ in range(50001)])
            if prefixes[0] != prefixes[1]:
                raise ValueError(
                    "paired PPO collected transitions differ before intervention"
                )
            checks[str(seed)] = {
                "initial_and_50k_weights_identical": True,
                "first_50000_transitions_identical": True,
            }
    result = {
        "manifest_sha256": sha256(root / "manifest.json"),
        "inventory": inventory,
        "models": models,
        "aggregates": aggregates,
        "comparisons": comparisons,
        "ppo_prefix_checks": checks,
        "final_test_used": False,
        "board_scope": "199 common development boards; finite cohort of three training seeds, not held-out generalization",
        "highest_observed_model": max(
            models, key=lambda n: models[n]["final"]["win_rate"]
        )
        if models
        else None,
    }
    result["highest_scheduled_checkpoint"] = max(
        (
            {"id": name, "step": row["step"], "win_rate": row["win_rate"]}
            for name, model in models.items()
            for row in model["curve"]
        ),
        key=lambda row: row["win_rate"],
        default=None,
    )
    result["figures"] = plot(models, aggregates, training, output / "figures")
    history = json.loads((root / "history_inventory.json").read_text())
    if any(sha256(n) != d for n, d in history["files_sha256"].items()):
        raise ValueError("historical artifacts changed")
    result["historical_files_verified"] = len(history["files_sha256"])
    verify_manifest(root)
    with (output / "summary.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    lines = [
        "# Overnight RL research results",
        "",
        "All rates use the frozen shared development boards. Historical training cohorts are descriptive.",
        "",
        "| Variant | Models | Mean win rate | Training-seed SD (pp) | Objective completion | Remaining HP |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in aggregates.items():
        sd = (
            "n/a"
            if r["training_seed_sample_std"] is None
            else f"{100 * r['training_seed_sample_std']:.2f}"
        )
        lines.append(
            f"| {name} | {r['n_models']} | {r['mean_win_rate']:.2%} | {sd} | {r['objective_completion_ratio']:.2%} | {r['blocker_hp_remaining']:.3f} |"
        )
    lines += [
        "",
        "Full inventory, paired outcome tables, effect sizes, conditional CIs, raw/adjusted p-values, checkpoint hashes and costs are in summary.json.",
        "",
        "No algorithm-level superiority or equivalence follows from three training seeds. Best observed checkpoints are development-selected. No reserved final-test boards were used.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    return result


def plot(models, aggregates, training, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    files = []

    def save(fig, name):
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            file = f"{name}.{suffix}"
            fig.savefig(output / file, dpi=150)
            files.append(file)
        plt.close(fig)

    names = list(aggregates)
    fig, axes = plt.subplots(3, 1, figsize=(12, 11))
    for ax, key, title in zip(
        axes,
        ("mean_win_rate", "objective_completion_ratio", "blocker_hp_remaining"),
        ("Mean win rate", "Objective completion", "Remaining blocker HP"),
    ):
        ax.barh(names, [aggregates[n][key] for n in names])
        ax.set_title(title)
    save(fig, "algorithm_comparison")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ident, r in models.items():
        if len(r["curve"]) > 1:
            ax = axes[0] if "PPO" in r["label"] else axes[1]
            ax.plot(
                [c["step"] for c in r["curve"]],
                [c["win_rate"] for c in r["curve"]],
                "o-",
                label=ident,
            )
    for ax in axes:
        ax.set(xlabel="Collected transitions", ylabel="Win rate")
        ax.legend(fontsize=6)
    save(fig, "learning_curves")
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for name in names:
        rows = [r for r in models.values() if r["label"] == name]
        axes[0].scatter([name] * len(rows), [r["final"]["win_rate"] for r in rows])
        active = [r for r in rows if "training_seconds" in r]
        if active:
            axes[1].scatter(
                [r["training_seconds"] / 60 for r in active],
                [r["final"]["win_rate"] for r in active],
                label=name,
            )
    axes[0].tick_params(axis="x", rotation=60)
    axes[0].set_ylabel("Per-model endpoint win rate")
    axes[1].set(
        xlabel="Total process minutes (includes validation)", ylabel="Endpoint win rate"
    )
    axes[1].legend(fontsize=7)
    save(fig, "seed_variability_and_cost")
    for family, keys in (
        (
            "ppo",
            (
                "approximate_kl",
                "clip_fraction",
                "entropy",
                "value_loss",
                "gradient_norm",
                "learning_rate",
            ),
        ),
        ("dqn", ("td_loss", "q_mean", "gradient_norm", "epsilon")),
    ):
        fig, axes = plt.subplots(
            len(keys), 1, figsize=(12, len(keys) * 2.4), sharex=True
        )
        for ident, (path, spec) in training.items():
            if spec["kind"] != family:
                continue
            rows = read_csv(
                path
                / ("optimization.csv" if family == "ppo" else "training_metrics.csv")
            )
            for ax, key in zip(axes, keys):
                valid = [r for r in rows if r.get(key) not in (None, "")]
                x = [int(r["environment_step"]) for r in valid]
                y = [float(r[key]) for r in valid]
                ax.plot(
                    x,
                    trailing_mean(y) if key != "learning_rate" else y,
                    label=ident,
                    lw=0.8,
                )
                ax.set_ylabel(key)
                if key == "learning_rate":
                    ax.set_ylim(0, 0.00033)
        axes[0].legend(fontsize=6)
        axes[-1].set_xlabel(
            "Collected transitions; trailing 50 recorded values except LR"
        )
        save(fig, f"{family}_diagnostics")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    analyze(parser.parse_args().root)


if __name__ == "__main__":
    main()
