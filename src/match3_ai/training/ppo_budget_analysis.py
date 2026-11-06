"""Joint board-paired inference and audits for the preregistered PPO budget study."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from match3_ai.evaluation.analysis import exact_mcnemar
from match3_ai.evaluation.behavior_statistics import holm_adjust
from match3_ai.evaluation.fresh_protocol import sha256
from match3_ai.training.ppo_analysis import read_csv
from match3_ai.training.ppo_budget import (
    MASTER_SEEDS,
    MILESTONES,
    experiment_config,
    load_protocol,
    optimized_progress,
)


def joint_budget_statistics(outcomes, *, samples=20000, seed=51717, metrics=None):
    """One board draw per replicate for every policy, seed, checkpoint and metric."""
    cube = np.asarray(outcomes, dtype=float)
    if (
        cube.ndim != 3
        or cube.shape[:2] != (3, 5)
        or not cube.shape[2]
        or not np.isin(cube, [0, 1]).all()
    ):
        raise ValueError("outcomes must be binary (three runs,five checkpoints,boards)")
    if type(samples) is not int or samples <= 0:
        raise ValueError("bootstrap samples must be positive")
    n = cube.shape[2]
    differences = np.stack(
        [
            cube[:, 3].mean(0) - cube[:, 2].mean(0),
            cube[:, 4].mean(0) - cube[:, 2].mean(0),
            *[cube[i, 3] - cube[i, 2] for i in range(3)],
            *[cube[i, 4] - cube[i, 2] for i in range(3)],
        ]
    )
    metric_cubes = {"win_rate": cube}
    for name, value in (metrics or {}).items():
        value = np.asarray(value, dtype=float)
        if value.shape != cube.shape or not np.isfinite(value).all():
            raise ValueError("descriptive metric cubes must be finite and aligned")
        metric_cubes[name] = value
    matrix = np.concatenate(
        [differences, *[value.reshape(15, n) for value in metric_cubes.values()]]
    )
    rng = np.random.default_rng(seed)
    index_digest = hashlib.sha256()
    parts = []
    for start in range(0, samples, 250):
        indices = rng.integers(n, size=(min(250, samples - start), n))
        index_digest.update(indices.astype(np.int64).tobytes())
        parts.append(matrix[:, indices].mean(-1))
    boot = np.concatenate(parts, axis=1)
    observed = differences.mean(1)
    cis = np.quantile(boot[:8], [0.025, 0.975], axis=1).T
    comparisons = []
    labels = [
        "mean_75k_minus_50k",
        "mean_100k_minus_50k",
        *[f"seed{s}_75k_minus_50k" for s in MASTER_SEEDS],
        *[f"seed{s}_100k_minus_50k" for s in MASTER_SEEDS],
    ]
    for index, name in enumerate(labels):
        p = (
            1
            + int(
                np.sum(
                    np.abs(boot[index] - observed[index])
                    >= abs(observed[index]) - 1e-12
                )
            )
        ) / (samples + 1)
        item = {
            "name": name,
            "n_boards": n,
            "difference": float(observed[index]),
            "ci95": cis[index].tolist(),
            "p_value": p,
            "test": "approximate null-centered joint board bootstrap",
        }
        if index >= 2:
            run = (index - 2) % 3
            checkpoint = 3 if index < 5 else 4
            left, right = cube[run, checkpoint], cube[run, 2]
            counts = {
                "both_win": int(np.sum((left == 1) & (right == 1))),
                "later_only_win": int(np.sum((left == 1) & (right == 0))),
                "50k_only_win": int(np.sum((left == 0) & (right == 1))),
                "both_lose": int(np.sum((left == 0) & (right == 0))),
            }
            item.update(
                counts=counts,
                p_value=exact_mcnemar(counts["later_only_win"], counts["50k_only_win"]),
                test="exact two-sided McNemar",
            )
        comparisons.append(item)
    for item, p in zip(comparisons, holm_adjust([r["p_value"] for r in comparisons])):
        item["holm_p"] = p
    descriptive = {}
    for index, (name, value) in enumerate(metric_cubes.items()):
        draws = boot[8 + index * 15 : 8 + (index + 1) * 15].reshape(3, 5, samples)
        descriptive[name] = {
            "per_seed": value.mean(-1).tolist(),
            "mean": value.mean((0, 2)).tolist(),
            "training_seed_sample_std": value.mean(-1).std(0, ddof=1).tolist(),
            "conditional_mean_ci95": np.quantile(
                draws.mean(0), [0.025, 0.975], axis=1
            ).T.tolist(),
        }
    return {
        "comparisons": comparisons,
        "metrics": descriptive,
        "n_boards": n,
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "joint_resampled_indices_sha256": index_digest.hexdigest(),
        "scope": "conditional board uncertainty, fixed three training runs; sample SD separately measures observed training-run variation",
    }


def audit_run(root, protocol):
    import torch

    from match3_ai.agents.ppo_agent import PPOAgent, masked_distribution
    from match3_ai.training.ppo import environment, fingerprint

    root = Path(root)
    summary = json.loads((root / "training_summary.json").read_text())
    seed = summary["training_config"]["seed"]
    expected = protocol["configurations"][str(seed)]
    if (
        summary["training_config"] != expected
        or summary["level_config_id"] != protocol["level_config_id"]
        or summary["initial_weights"]
        != protocol["streams"][str(seed)]["initial_weights"]
    ):
        raise ValueError(
            "run configuration, level or initialization differs from protocol"
        )
    paths = (
        [
            root / name
            for name in (
                "training_summary.json",
                "training_episodes.json",
                "transitions.csv",
                "optimization.csv",
                "rollouts.csv",
                "evaluation/episodes.csv",
                "evaluation/moves.csv",
            )
        ]
        + sorted(root.glob("policy_step_*.pt"))
        + [root / "checkpoint_final.pt"]
    )
    hashes = {str(path): sha256(path) for path in paths}
    transitions, updates, rollouts, validation, moves = [
        read_csv(root / name)
        for name in (
            "transitions.csv",
            "optimization.csv",
            "rollouts.csv",
            "evaluation/episodes.csv",
            "evaluation/moves.csv",
        )
    ]
    episodes = json.loads((root / "training_episodes.json").read_text())
    progress = optimized_progress(100000)
    if (
        len(transitions) != 100000
        or sum(e["steps"] for e in episodes) != 100000
        or len(rollouts) != progress["rollouts"]
        or len(updates) != progress["optimizer_updates"]
    ):
        raise ValueError("collected transitions, rollouts or updates do not reconcile")
    if np.diff([0, *[int(r["environment_step"]) for r in rollouts]]).tolist() != [
        256
    ] * 390 + [160]:
        raise ValueError("rollout boundaries differ from the approved fixed schedule")
    if any(
        summary["seed_metadata"][name] != protocol["streams"][str(seed)][name]
        for name in ("agent_seed", "model", "policy", "shuffle")
    ):
        raise ValueError("recorded independent RNG streams differ from preregistration")
    if any(
        int(r["environment_step"]) != i + 1
        or not 200000 <= int(r["episode_seed"]) <= 299999
        or r["valid_action"] != "True"
        or float(r["reward"]) not in (-1, 0, 1)
        for i, r in enumerate(transitions)
    ):
        raise ValueError("training seed, sparse reward or transition sequence invalid")
    if any(
        float(r["next_value"]) != 0 for r in transitions if r["terminated"] == "True"
    ) or any(
        float(r["reward"]) != 0 for r in transitions if r["terminated"] == "False"
    ):
        raise ValueError("termination bootstrap or nonterminal sparse reward differs")
    if sum(int(r["minibatch_size"]) for r in updates) != 400000 or any(
        float(r["learning_rate"]) != 0.0003
        or not np.isfinite([float(v) for v in r.values()]).all()
        for r in updates
    ):
        raise ValueError("optimizer sample budget, learning rate or finiteness invalid")
    if any(
        not all(v == "" or np.isfinite(float(v)) for v in r.values()) for r in rollouts
    ):
        raise ValueError("nonfinite rollout diagnostic")
    if [r["environment_step"] for r in summary["evaluation_rounds"]] != list(
        MILESTONES
    ):
        raise ValueError("milestone schedule differs from protocol")
    if len(validation) != 1000 or any(
        r["outcome"] not in ("WON", "LOST") or int(r["invalid_actions"])
        for r in validation
    ):
        raise ValueError("validation count or completed outcomes invalid")
    config = experiment_config(seed)
    env = environment(config)
    try:
        observations = {s: env.reset(seed=s)[0] for s in protocol["validation_seeds"]}
    finally:
        env.close()
    groups, checks = [], []
    for milestone in summary["evaluation_rounds"]:
        step = milestone["environment_step"]
        group = {
            int(r["seed"]): r
            for r in validation
            if int(r["episode_id"].split(":")[-3]) == step
        }
        if (
            len(group) != 200
            or set(group) != set(protocol["validation_seeds"])
            or sum(r["outcome"] == "WON" for r in group.values()) != milestone["wins"]
        ):
            raise ValueError("validation seed coverage or summary mismatched")
        agent, payload = PPOAgent.load_checkpoint(root / f"policy_step_{step}.pt")
        actual = payload["training_progress"]
        wanted = optimized_progress(step)
        for key, field in (
            ("environment_step", "collected"),
            ("last_optimized_step", "optimized"),
            ("pending_rollout_steps", "pending"),
            ("rollouts", "rollouts"),
            ("optimizer_updates", "optimizer_updates"),
        ):
            if actual[key] != wanted[field]:
                raise ValueError(
                    "checkpoint collected/optimized budget metadata differs"
                )
        if (
            payload["validation_metrics"]["wins"] != milestone["wins"]
            or json.loads(json.dumps(payload["training_config"])) != expected
        ):
            raise ValueError("checkpoint metrics or configuration differs")
        rng = agent.policy_rng.get_state().clone()
        before = fingerprint(agent)
        first_actions = {
            int(r["episode_id"].split(":")[-1]): int(r["action_id"])
            for r in moves
            if int(r["episode_id"].split(":")[-3]) == step and int(r["move_index"]) == 1
        }
        maximum_error = 0.0
        for index, s in enumerate(protocol["validation_seeds"]):
            observation = observations[s]
            features, mask = agent.encoder.encode(observation)
            with torch.inference_mode():
                logits, values = agent.network(torch.from_numpy(features[None]))
                distribution = masked_distribution(logits, torch.from_numpy(mask[None]))
            maximum_error = max(maximum_error, abs(float(distribution.probs.sum()) - 1))
            if (
                agent.select_action(observation) != first_actions[index]
                or distribution.probs[0, ~torch.from_numpy(mask)].any()
                or not torch.isfinite(values).all()
            ):
                raise ValueError("checkpoint action/probability replay differs")
        if (
            maximum_error > 1e-6
            or fingerprint(agent) != before
            or not torch.equal(rng, agent.policy_rng.get_state())
        ):
            raise ValueError(
                "checkpoint inference modified state or invalid normalization"
            )
        groups.append(group)
        checks.append(
            {
                **wanted,
                "states_checked": 200,
                "max_probability_sum_error": maximum_error,
                "checkpoint_sha256": hashes[str(root / f"policy_step_{step}.pt")],
            }
        )
    final, final_payload = PPOAgent.load_checkpoint(root / "checkpoint_final.pt")
    if (
        fingerprint(final) != summary["final_weights"]
        or fingerprint(final) != before
        or {int(state["step"]) for state in final.optimizer.state.values()} != {6252}
    ):
        raise ValueError("final checkpoint differs or Adam count invalid")
    if any(sha256(path) != digest for path, digest in hashes.items()):
        raise RuntimeError("run artifact changed during audit")
    return {
        "summary": summary,
        "rollouts": rollouts,
        "updates": updates,
        "groups": groups,
        "checkpoint_checks": checks,
        "source_sha256": hashes,
        "final_checkpoint_progress": final_payload["training_progress"],
    }


def create_figures(runs, statistics, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    files = []

    def finish(fig, name):
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            fig.savefig(output / filename, dpi=180)
            files.append(filename)
        plt.close(fig)

    for name, key, label in (
        ("win_rate", "win_rate", "Development-validation win rate"),
        ("objective_completion", "objective_completion", "Objective completion"),
        ("remaining_blocker_hp", "remaining_blocker_hp", "Remaining blocker HP"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for index, seed in enumerate(runs):
            ax.plot(
                MILESTONES,
                statistics["metrics"][key]["per_seed"][index],
                marker="o",
                label=str(seed),
            )
        ax.set(xlabel="Collected environment transitions", ylabel=label)
        ax.grid(alpha=0.25)
        ax.legend(title="Training seed")
        finish(fig, name)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    metric = statistics["metrics"]["win_rate"]
    mean = np.array(metric["mean"])
    sd = np.array(metric["training_seed_sample_std"])
    ax.plot(MILESTONES, mean, marker="o", label="Mean of three runs")
    ax.fill_between(
        MILESTONES, mean - sd, mean + sd, alpha=0.2, label="± sample SD, not CI"
    )
    ax.set(
        xlabel="Collected environment transitions",
        ylabel="Mean development-validation win rate",
    )
    ax.legend()
    ax.grid(alpha=0.25)
    finish(fig, "mean_win_rate")
    fig, axes = plt.subplots(
        1, 3, figsize=(13, 4), constrained_layout=True, sharey=True
    )
    for ax, (seed, run) in zip(axes, runs.items()):
        rows = run["summary"]["evaluation_rounds"]
        ax.plot(MILESTONES, [r["win_rate"] for r in rows], marker="o")
        ax.set(title=f"Seed {seed}", xlabel="Collected transitions", ylabel="Win rate")
        ax.grid(alpha=0.25)
    finish(fig, "per_seed_learning_curves")
    for key, label, source in (
        ("explained_variance", "Pre-update explained variance", "rollouts"),
        ("entropy", "Masked entropy, nats", "updates"),
        ("approximate_kl", "Approximate KL", "updates"),
        ("clip_fraction", "Clip fraction", "updates"),
        ("value_loss", "Value loss", "updates"),
        ("policy_loss", "Policy loss", "updates"),
        ("gradient_norm", "Gradient norm before clipping", "updates"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for seed, run in runs.items():
            rows = run[source]
            x = [int(r["environment_step"]) for r in rows]
            y = np.array([float(r[key]) if r[key] != "" else np.nan for r in rows])
            if source == "updates":
                ends = np.arange(1, len(y) + 1)
                starts = np.maximum(ends - 50, 0)
                sums = np.r_[0.0, y.cumsum()]
                y = (sums[ends] - sums[starts]) / (ends - starts)
            ax.plot(x, y, label=str(seed), linewidth=1.1)
        ax.set(
            xlabel="Collected environment transitions",
            ylabel=label + (" (rolling 50 updates)" if source == "updates" else ""),
        )
        ax.legend()
        ax.grid(alpha=0.25)
        finish(fig, key)
    fig, ax = plt.subplots(figsize=(8, 5), constrained_layout=True)
    labels = list(map(str, runs))
    active = [r["summary"]["collection_optimization_seconds"] for r in runs.values()]
    validation = [r["summary"]["evaluation_seconds"] for r in runs.values()]
    ax.bar(labels, active, label="Active training/checkpoints")
    ax.bar(labels, validation, bottom=active, label="Validation")
    ax.set(xlabel="Master seed", ylabel="Process seconds")
    ax.legend()
    finish(fig, "training_runtime")
    return files


def analyze(protocol_path, run_dirs, output_dir):
    import torch

    protocol = load_protocol(protocol_path)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    if len(run_dirs) != 3:
        raise ValueError("three independent run directories required")
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        runs = {}
        for directory in run_dirs:
            run = audit_run(directory, protocol)
            seed = run["summary"]["training_config"]["seed"]
            if seed in runs:
                raise ValueError("duplicate training seed")
            runs[seed] = run
        if set(runs) != set(MASTER_SEEDS):
            raise ValueError("training seed set differs")
        runs = dict(sorted(runs.items()))
        for step_index in range(5):
            for board in protocol["validation_seeds"]:
                identifiers = {
                    (
                        run["groups"][step_index][board]["board_seed"],
                        run["groups"][step_index][board]["engine_seed"],
                        run["groups"][step_index][board]["level_config_id"],
                    )
                    for run in runs.values()
                }
                if len(identifiers) != 1:
                    raise ValueError("shared-board reset identifiers differ")
        for board in protocol["validation_seeds"]:
            identifiers = {
                (
                    group[board]["board_seed"],
                    group[board]["engine_seed"],
                    group[board]["level_config_id"],
                )
                for run in runs.values()
                for group in run["groups"]
            }
            if len(identifiers) != 1:
                raise ValueError(
                    "checkpoint dependence broken by different initial boards"
                )
        cubes = {
            key: np.array(
                [
                    [
                        [float(group[b][key]) for b in protocol["validation_seeds"]]
                        for group in run["groups"]
                    ]
                    for run in runs.values()
                ]
            )
            for key in ("objective_completion_ratio", "blocker_hp_remaining")
        }
        outcomes = np.array(
            [
                [
                    [group[b]["outcome"] == "WON" for b in protocol["validation_seeds"]]
                    for group in run["groups"]
                ]
                for run in runs.values()
            ]
        )
        statistics = joint_budget_statistics(
            outcomes,
            samples=protocol["statistics"]["bootstrap_samples"],
            seed=protocol["statistics"]["bootstrap_seed"],
            metrics={
                "objective_completion": cubes["objective_completion_ratio"],
                "remaining_blocker_hp": cubes["blocker_hp_remaining"],
            },
        )
        if [r["name"] for r in statistics["comparisons"]] != protocol["hypotheses"]:
            raise ValueError("Holm family differs from frozen protocol")
        historical = json.loads(
            Path(protocol_path).with_name("history_inventory.json").read_text()
        )
        if any(
            sha256(name) != digest
            for name, digest in historical["files_sha256"].items()
        ):
            raise RuntimeError("historical artifact changed")
        result = {
            "protocol_sha256": sha256(protocol_path),
            "budget_axis": protocol["axis"],
            "statistics": statistics,
            "runs": {
                str(seed): {
                    k: run[k]
                    for k in (
                        "summary",
                        "checkpoint_checks",
                        "source_sha256",
                        "final_checkpoint_progress",
                    )
                }
                for seed, run in runs.items()
            },
            "historical_files_preserved": len(historical["files_sha256"]),
            "final_test_used": False,
        }
        output.mkdir(parents=True, exist_ok=False)
        result["figures"] = create_figures(runs, statistics, output / "figures")
        write_report(result, runs, output / "report.md")
        with (output / "analysis.json").open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        with (output / "paired_outcomes.json").open("x") as handle:
            json.dump(
                {
                    "master_seeds": list(runs),
                    "milestones": list(MILESTONES),
                    "validation_seeds": protocol["validation_seeds"],
                    "wins": outcomes.astype(int).tolist(),
                },
                handle,
            )
        for run in runs.values():
            if any(
                sha256(name) != digest for name, digest in run["source_sha256"].items()
            ):
                raise RuntimeError("run artifact changed after analysis")
        load_protocol(protocol_path)
        return result
    finally:
        torch.set_num_threads(old_threads)


def write_report(result, runs, path):
    statistics = result["statistics"]
    lines = [
        "# Sprint 5.17: PPO training budget study",
        "",
        f"Frozen protocol SHA-256: `{result['protocol_sha256']}`. Budget axis is collected single-environment transitions. Optimized and pending transitions are reported separately; intermediate evaluations do not split 256-step rollouts. No exact continuation, tuning, PBRS, CNN-PPO or final-test evaluation.",
        "",
        "| Seed | Collected | Optimized | Pending | Wins / 200 [Wilson 95% CI] | Completion | HP | Moves | Reward |",
        "|---:|---:|---:|---:|---|---:|---:|---:|---:|",
    ]
    for seed, run in runs.items():
        for row, check in zip(
            run["summary"]["evaluation_rounds"], run["checkpoint_checks"]
        ):
            lo, hi = row["win_rate_ci95"]
            lines.append(
                f"| {seed} | {check['collected']} | {check['optimized']} | {check['pending']} | {row['wins']} [{lo:.2%}, {hi:.2%}] | {row['mean_objective_completion_ratio']:.2%} | {row['mean_blocker_hp_remaining']:.3f} | {row['mean_moves_used']:.3f} | {row['mean_total_reward']:.3f} |"
            )
    lines += [
        "",
        "## Cohort budget curves",
        "",
        "| Collected | Optimized | Mean win rate | Training-seed sample SD | Conditional board-bootstrap 95% CI |",
        "|---:|---:|---:|---:|---|",
    ]
    metric = statistics["metrics"]["win_rate"]
    for index, step in enumerate(MILESTONES):
        lo, hi = metric["conditional_mean_ci95"][index]
        lines.append(
            f"| {step} | {optimized_progress(step)['optimized']} | {metric['mean'][index]:.2%} | {100 * metric['training_seed_sample_std'][index]:.3f} pp | [{lo:.2%}, {hi:.2%}] |"
        )
    lines += [
        "",
        "## Eight preregistered paired hypotheses",
        "",
        "| Contrast | Difference pp [pointwise 95% CI] | Test | Raw p | Holm p |",
        "|---|---|---|---:|---:|",
    ]
    for r in statistics["comparisons"]:
        lines.append(
            f"| {r['name']} | {100 * r['difference']:+.3f} [{100 * r['ci95'][0]:+.3f}, {100 * r['ci95'][1]:+.3f}] | {r['test']} | {r['p_value']:.6f} | {r['holm_p']:.6f} |"
        )
    lines += [
        "",
        "One shared board-index vector is applied jointly to the entire seed × checkpoint × board cube in every bootstrap replicate. Repeated checkpoints, policies and training seeds are never resampled as independent board observations. These are conditional board intervals for these three training runs; training-seed sample SD is separate, not a population confidence interval. The exact common index-stream SHA-256 is saved in analysis.json. All eight p values receive one Holm correction; intervals remain pointwise. Other checkpoints and objective/HP changes are descriptive, not extra uncorrected significance tests. Nonsignificance is not equivalence. Aggregate p values use approximate null-centered paired bootstrap, individual binary p values exact McNemar.",
        "",
        "## Final versus development-validation selected checkpoints",
        "",
        "| Seed | Final wins | Best wins / collected step |",
        "|---:|---:|---|",
    ]
    for seed, run in runs.items():
        rows = run["summary"]["evaluation_rounds"]
        best = max(rows, key=lambda r: r["win_rate"])
        lines.append(
            f"| {seed} | {rows[-1]['wins']} | {best['wins']} / {best['environment_step']} |"
        )
    lines += [
        "",
        "Best checkpoints use these same development boards and earliest-tie selection; they are not unbiased performance estimates. Earlier 50k cohorts and Sprint 5.16 boards are not the paired controls in this study. The within-run 50k checkpoint has optimized 49920 transitions, unlike the separately terminated historical 50k runs.",
        "",
        "## Training diagnostics and costs",
        "",
        "| Seed | Rollouts / updates | Training terminal episodes | Training win rate / return | Explained variance mean / last | Active / validation / wall seconds |",
        "|---:|---|---:|---|---|---|",
    ]
    for seed, run in runs.items():
        s = run["summary"]
        last = run["rollouts"][-1]
        ev = [
            float(r["explained_variance"])
            for r in run["rollouts"]
            if r["explained_variance"] != ""
        ]
        lines.append(
            f"| {seed} | {s['rollouts']} / {s['optimizer_updates']} | {last['completed_episodes']} | {float(last['training_win_rate']):.2%} / {float(last['mean_training_episode_reward']):.4f} | {np.mean(ev):.4f} / {ev[-1]:.4f} | {s['collection_optimization_seconds']:.2f} / {s['evaluation_seconds']:.2f} / {s['training_seconds']:.2f} |"
        )
    lines += [
        "",
        "| Seed / metric | Minimum | Maximum | Mean |",
        "|---|---:|---:|---:|",
    ]
    for seed, run in runs.items():
        for key, values in run["summary"]["optimization_statistics"].items():
            lines.append(
                f"| {seed} {key} | {values['min']:.6f} | {values['max']:.6f} | {values['mean']:.6f} |"
            )
    lines += [
        "",
        "All recorded losses/gradients/rollout metrics were checked finite; rewards remain sparse and every training seed remains in 200000–299999. Explained variance uses pre-update on-policy GAE targets on different trajectories and is not a common held-out critic benchmark. Undefined constant-target statistics remain blank. Final diagnostics have only 160 samples versus 256 in full rollouts. Value loss, entropy and finite gradients do not themselves establish useful learning. Processes ran concurrently with one Torch CPU thread each: wall timers include validation and contention, exclude final exports, and cannot be summed as elapsed batch time.",
        "",
        "## Provenance, checkpoints and limits",
        "",
        f"All {result['historical_files_preserved']} historical artifact hashes remain unchanged. Every milestone policy was loaded and checked against its 200 recorded initial-state decisions, finite values, legal probabilities and unchanged sampling RNG. Final files match the 100k milestone weights and Adam step count 6252. Exact active-engine/unfinished-rollout continuation is not supported. Fresh master seeds 5918–5920 initialize new models. Three training seeds cannot establish algorithm superiority or precise population-level budget effects.",
        "",
        "## Recommendation",
        "",
        "Review the predefined 75k−50k and 100k−50k effects, their paired intervals and variation across seeds before approving Sprint 5.18. Do not select a favorable checkpoint to claim a budget benefit, interpret nonsignificance as equivalence, or automatically change hyperparameters. No Sprint 5.18 work has started.",
        "",
        "## Figures",
        "",
    ]
    for name in result["figures"]:
        if name.endswith(".png"):
            lines.extend([f"![{name}](figures/{name})", ""])
    with path.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.protocol, args.run_dir, args.output_dir)


if __name__ == "__main__":
    main()
