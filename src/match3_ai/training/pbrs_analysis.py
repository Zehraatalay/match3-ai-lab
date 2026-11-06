"""Controlled sparse/PBRS pilot audit, paired validation comparisons and curves."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from match3_ai.evaluation.behavior_statistics import paired_difference, paired_outcomes
from match3_ai.training.reward_shaping import training_reward
from match3_ai.training.seed_policy import validate_partition


def _csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _fingerprint(agent):
    return hashlib.sha256(
        b"".join(p.detach().numpy().tobytes() for p in agent.online.parameters())
    ).hexdigest()


def load_run(directory):
    from match3_ai.agents.dqn_agent import DQNAgent

    root = Path(directory)
    names = (
        "training_summary.json",
        "training_metrics.csv",
        "training_episodes.json",
        "checkpoint_final.pt",
        "evaluation/episodes.csv",
        "evaluation/moves.csv",
        "policy_step_0.pt",
    )
    hashes = {
        str(root / name): hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in names
    }
    summary = json.loads((root / names[0]).read_text())
    rows = _csv(root / names[1])
    episodes = json.loads((root / names[2]).read_text())
    config = summary["training_config"]
    validate_partition(
        "training", config["train_seed_start"], config["environment_steps"]
    )
    validate_partition(
        "validation", config["evaluation_seed_start"], config["evaluation_episodes"]
    )
    if len(rows) != config["environment_steps"] or sum(
        e["steps"] for e in episodes
    ) != len(rows):
        raise ValueError("training transition counts do not reconcile")
    agent, payload = DQNAgent.load_checkpoint(root / "checkpoint_final.pt")
    initial, initial_payload = DQNAgent.load_checkpoint(root / "policy_step_0.pt")
    if (
        initial_payload["training_step"] != 0
        or initial.optimizer.state
        or _fingerprint(initial) != summary["initial_weights"]
    ):
        raise ValueError("step-zero model differs from initialization metadata")
    replay = payload["training_state"]["replay"]
    if replay["size"] != len(rows) or replay["size"] > config["replay_capacity"]:
        raise ValueError("pilot replay does not preserve every recorded transition")
    if _fingerprint(agent) != summary["final_weights"]:
        raise ValueError("checkpoint and summary weights differ")
    for index, row in enumerate(rows):
        observation = {
            key: array[index].numpy() for key, array in replay["observations"].items()
        }
        after = {
            key: array[index].numpy()
            for key, array in replay["next_observations"].items()
        }
        components = training_reward(
            observation,
            after,
            float(row["sparse_reward"]),
            terminated=bool(replay["terminated"][index]),
            truncated=bool(replay["truncated"][index]),
            mode=config["reward_mode"],
            beta=config["shaping_beta"],
            gamma=config["gamma"],
        )
        for key in (
            "sparse_reward",
            "shaping_bonus",
            "shaped_reward",
            "potential_current",
            "potential_next",
        ):
            if not np.isclose(
                float(row[key]), getattr(components, key), rtol=0, atol=1e-12
            ):
                raise ValueError(
                    "recorded shaping components differ from replay observations"
                )
        if not np.isclose(
            float(replay["rewards"][index]),
            components.shaped_reward,
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError("replay reward differs from configured training reward")
        if float(row["sparse_reward"]) not in (-1, 0, 1):
            raise ValueError("original sparse reward was modified")
    start = 0
    max_telescoping_error = 0.0
    beta = config["shaping_beta"] if config["reward_mode"] == "pbrs" else 0.0
    for episode in episodes:
        subset = rows[start : start + episode["steps"]]
        if not np.isclose(
            sum(float(r["sparse_reward"]) for r in subset), episode["sparse_return"]
        ):
            raise ValueError("sparse episode return differs")
        if not np.isclose(
            sum(float(r["shaped_reward"]) for r in subset), episode["shaped_return"]
        ):
            raise ValueError("shaped episode return differs")
        boundary = beta * (
            config["gamma"] ** episode["steps"] * episode["final_potential"]
            - float(subset[0]["potential_current"])
        )
        error = abs(
            episode["discounted_shaped_return"]
            - episode["discounted_sparse_return"]
            - boundary
        )
        max_telescoping_error = max(max_telescoping_error, error)
        start += episode["steps"]
    if max_telescoping_error > 1e-10:
        raise ValueError("discounted shaping does not telescope")
    validation = _csv(root / "evaluation/episodes.csv")
    for episode in validation:
        validate_partition("validation", int(episode["seed"]), 1)
        if (
            episode["outcome"] not in ("WON", "LOST")
            or float(episode["total_reward"]) not in (-1, 1)
            or int(episode["invalid_actions"])
        ):
            raise ValueError("invalid or non-sparse validation episode")
    if any(
        float(row["reward"]) not in (-1, 0, 1)
        for row in _csv(root / "evaluation/moves.csv")
    ):
        raise ValueError("evaluation move reward is shaped")
    updates = [row for row in rows if row["td_loss"]]
    stats = {
        key: {
            "min": min(float(row[key]) for row in updates),
            "max": max(float(row[key]) for row in updates),
            "mean": float(np.mean([float(row[key]) for row in updates])),
        }
        for key in ("td_loss", "q_mean", "q_min", "q_max", "gradient_norm")
    }
    if not all(
        np.isfinite(value) for values in stats.values() for value in values.values()
    ):
        raise ValueError("nonfinite training metrics")
    rounds = [
        {
            "step": item["environment_step"],
            **next(
                group for group in item["summary"] if group["agent_name"] == "DQNAgent"
            ),
        }
        for item in summary["evaluation_rounds"]
    ]
    expected_seeds = set(
        range(
            config["evaluation_seed_start"],
            config["evaluation_seed_start"] + config["evaluation_episodes"],
        )
    )
    for item in rounds:
        selected = [
            e
            for e in validation
            if e["agent_name"] == "DQNAgent"
            and int(e["episode_id"].split(":")[-3]) == item["step"]
        ]
        if (
            len(selected) != config["evaluation_episodes"]
            or {int(e["seed"]) for e in selected} != expected_seeds
        ):
            raise ValueError("validation checkpoint seed coverage differs")
        if sum(e["outcome"] == "WON" for e in selected) != item["wins"]:
            raise ValueError("validation summary differs from episode records")
    return {
        "root": str(root),
        "summary": summary,
        "config": config,
        "rows": rows,
        "episodes": episodes,
        "validation": validation,
        "statistics": stats,
        "rounds": rounds,
        "best": max(rounds, key=lambda item: item["win_rate"]),
        "max_telescoping_error": max_telescoping_error,
        "source_sha256": hashes,
    }


def create_figures(runs, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    files = []

    def save(name, xlabel, ylabel, series, *, proportion=False):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for x, y, label in series:
            ax.plot(x, y, label=label, linewidth=1.3)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if proportion:
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        ax.legend()
        finish(fig, name)

    def finish(fig, name):
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            fig.savefig(output / filename, dpi=180)
            files.append(filename)
        plt.close(fig)

    def rolling(values, window=50):
        values = np.asarray(values, dtype=float)
        ends = np.arange(1, len(values) + 1)
        starts = np.maximum(ends - window, 0)
        totals = np.r_[0.0, values.cumsum()]
        return (totals[ends] - totals[starts]) / (ends - starts)

    for name, key, ylabel in (
        ("validation_win_rate", "win_rate", "Sparse validation win rate"),
        (
            "objective_completion",
            "mean_objective_completion_ratio",
            "Mean capped objective completion",
        ),
        (
            "remaining_blocker_hp",
            "mean_blocker_hp_remaining",
            "Mean remaining blocker HP",
        ),
    ):
        save(
            name,
            "Training environment steps",
            ylabel,
            [
                (
                    [r["step"] for r in run["rounds"]],
                    [r[key] for r in run["rounds"]],
                    label,
                )
                for label, run in runs.items()
            ],
            proportion=key != "mean_blocker_hp_remaining",
        )
    series = []
    for label, run in runs.items():
        episodes = [e for e in run["episodes"] if e["terminated"]]
        for key in (
            ("sparse_return",)
            if label == "sparse"
            else ("sparse_return", "shaped_return")
        ):
            series.append(
                (
                    [e["environment_step"] for e in episodes],
                    rolling([e[key] for e in episodes]),
                    f"{label}: {key}",
                )
            )
    save(
        "training_returns",
        "Environment steps",
        "Undiscounted return (rolling 50 terminal episodes)",
        series,
    )
    save(
        "td_loss",
        "Optimizer updates",
        "Huber loss (rolling 50 updates)",
        [
            (
                [
                    int(row["optimizer_updates"])
                    for row in run["rows"]
                    if row["td_loss"]
                ],
                rolling(
                    [float(row["td_loss"]) for row in run["rows"] if row["td_loss"]]
                ),
                label,
            )
            for label, run in runs.items()
        ],
    )
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    for label, run in runs.items():
        ax.hist(
            [float(row["shaping_bonus"]) for row in run["rows"]],
            bins=np.linspace(-0.5, 0.5, 81),
            alpha=0.5,
            label=label,
        )
    ax.set(xlabel="Training shaping bonus", ylabel="Transitions", yscale="log")
    ax.legend()
    finish(fig, "shaping_bonus")
    save(
        "potential_progression",
        "Environment steps",
        "Current objective potential (rolling 50 transitions)",
        [
            (
                [int(row["environment_step"]) for row in run["rows"]],
                rolling([float(row["potential_current"]) for row in run["rows"]]),
                label,
            )
            for label, run in runs.items()
        ],
        proportion=True,
    )
    save(
        "q_values",
        "Environment steps",
        "Valid-action Q statistics (rolling 50 updates)",
        [
            (
                [int(row["environment_step"]) for row in run["rows"] if row["td_loss"]],
                rolling([float(row[key]) for row in run["rows"] if row["td_loss"]]),
                f"{label} {key}",
            )
            for label, run in runs.items()
            for key in ("q_mean", "q_max")
        ],
    )
    return files


def analyze_pilot(sparse_dir, pbrs_dir, output_dir, *, historical_checkpoint=None):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    runs = {"sparse": load_run(sparse_dir), "pbrs": load_run(pbrs_dir)}
    a, b = runs["sparse"], runs["pbrs"]
    for key in a["config"]:
        if key != "reward_mode" and a["config"][key] != b["config"][key]:
            raise ValueError("pilot arms differ beyond reward mode")
    if a["config"]["reward_mode"] != "sparse" or b["config"]["reward_mode"] != "pbrs":
        raise ValueError("incorrect arm reward modes")
    if (
        a["summary"]["initial_weights"] != b["summary"]["initial_weights"]
        or a["summary"]["seed_metadata"] != b["summary"]["seed_metadata"]
    ):
        raise ValueError("initial models or RNG streams differ")
    if a["summary"]["level_config_id"] != b["summary"]["level_config_id"]:
        raise ValueError("level configuration differs")
    paired_initial_seeds = {}
    for run in runs.values():
        for episode in run["validation"]:
            paired_initial_seeds.setdefault(int(episode["seed"]), set()).add(
                (
                    episode["board_seed"],
                    episode["engine_seed"],
                    episode["level_config_id"],
                )
            )
    if any(len(values) != 1 for values in paired_initial_seeds.values()):
        raise ValueError("validation initial board or RNG seeds differ")
    if [r["step"] for r in a["rounds"]] != [r["step"] for r in b["rounds"]]:
        raise ValueError("validation milestones differ")
    initial_groups = []
    for run in runs.values():
        initial_groups.append(
            {
                int(row["seed"]): row
                for row in run["validation"]
                if row["agent_name"] == "DQNAgent"
                and int(row["episode_id"].split(":")[-3]) == 0
            }
        )
    if set(initial_groups[0]) != set(initial_groups[1]):
        raise ValueError("step-zero validation seeds differ")
    for seed in initial_groups[0]:
        for key in (
            "outcome",
            "total_cleared",
            "total_reward",
            "total_cascades",
            "blocker_hp_remaining",
        ):
            if initial_groups[0][seed][key] != initial_groups[1][seed][key]:
                raise ValueError("step-zero behavior differs")
    final_step = a["config"]["environment_steps"]
    paired = []
    for run in (b, a):
        paired.append(
            {
                int(row["seed"]): row
                for row in run["validation"]
                if row["agent_name"] == "DQNAgent"
                and int(row["episode_id"].split(":")[-3]) == final_step
            }
        )
    if set(paired[0]) != set(paired[1]):
        raise ValueError("final validation seeds differ")
    comparison = paired_outcomes(
        {key: row["outcome"] for key, row in paired[0].items()},
        {key: row["outcome"] for key, row in paired[1].items()},
        seed=51111,
    )
    continuous = {
        key: paired_difference(
            [float(paired[0][seed][key]) for seed in sorted(paired[0])],
            [float(paired[1][seed][key]) for seed in sorted(paired[0])],
            seed=51111,
        )
        for key in (
            "objective_completion_ratio",
            "blocker_hp_remaining",
            "moves_used",
            "total_reward",
        )
    }
    historical_match = None
    historical_hash = None
    if historical_checkpoint is not None:
        from match3_ai.agents.dqn_agent import DQNAgent

        agent, _ = DQNAgent.load_checkpoint(historical_checkpoint)
        historical_match = _fingerprint(agent) == a["summary"]["final_weights"]
        historical_hash = hashlib.sha256(
            Path(historical_checkpoint).read_bytes()
        ).hexdigest()
        if not historical_match:
            raise ValueError("sparse control differs from historical same-step model")
    result = {
        "configuration": a["config"],
        "pbrs_configuration": b["config"],
        "initial_weights_identical": True,
        "rng_streams_identical": True,
        "historical_sparse_weights_match": historical_match,
        "historical_checkpoint_sha256": historical_hash,
        "paired_final_outcomes": comparison,
        "continuous_final_differences": continuous,
        "runs": {},
        "final_test_used": False,
    }
    for label, run in runs.items():
        rows = run["rows"]
        bonuses = np.array([float(row["shaping_bonus"]) for row in rows])
        result["runs"][label] = {
            "directory": run["root"],
            "summary": run["summary"],
            "rounds": run["rounds"],
            "best": run["best"],
            "optimization_statistics": run["statistics"],
            "max_telescoping_error": run["max_telescoping_error"],
            "bonus_distribution": {
                "min": float(bonuses.min()),
                "max": float(bonuses.max()),
                "mean": float(bonuses.mean()),
                "positive_fraction": float(np.mean(bonuses > 0)),
                "negative_fraction": float(np.mean(bonuses < 0)),
                "zero_fraction": float(np.mean(bonuses == 0)),
            },
            "zero_training_reward_fraction": float(
                np.mean([float(row["shaped_reward"]) == 0 for row in rows])
            ),
            "source_sha256": run["source_sha256"],
        }
    output.mkdir(parents=True, exist_ok=False)
    result["figures"] = create_figures(runs, output / "figures")
    lines = [
        "# Sprint 5.11: controlled potential-based reward shaping pilot",
        "",
        "## Reward design and policy-invariance scope",
        "",
        "The environment and evaluation remain sparse. Training optionally uses r' = r + beta (gamma Phi(next) - Phi(current)), with beta .5 and gamma .99. Phi is the equally weighted mean of active observable objective progress/target, clipped to [0,1]. CLEAR_ALL_BLOCKERS uses removed-blocker counts, not damage HP or unrelated clears; COLLECT_TILES uses its recorded objective count. Invalid/nonpositive targets or nonfinite progress contribute zero while retaining their objective weight; unsupported active types fail explicitly. No active objectives yields potential zero.",
        "",
        "True terminal states, including wins and move-budget losses, have Phi=0. Artificial truncations retain Phi(next) and use the existing next-observation bootstrap. No reward clipping, adaptive beta or extra bonus is applied. The same config.gamma is used for both shaping and DQN targets.",
        "",
        "For any T-step prefix, G' - G = beta[-Phi(s0) + gamma^T Phi(sT)]. Completed true-terminal trajectories have zero final potential, leaving an action-independent initial-state offset. Conditions: fixed bounded state potential, fixed beta, matching discount, policy-independent start distribution, unchanged transitions/actions, correct terminal boundary and consistently shaped continuation values at artificial truncations. Raw truncated prefixes retain a boundary term; discarding it or zeroing truncation potential would invalidate the guarantee. The theorem concerns exact returns/optimal-policy sets, not finite-budget neural DQN convergence or identical learned actions. Float32 replay also introduces ordinary numerical approximation. Undiscounted episode reward sums are not covered by the discounted telescoping statement.",
        "",
        "Reference: [Ng, Harada and Russell (1999)](https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf). The finite-prefix identity above is directly verified by synthetic tests and every recorded pilot episode.",
        "",
        "## Controlled configuration",
        "",
        "```json",
        json.dumps(a["config"], indent=2),
        "```",
        "",
        "The PBRS arm differs only in reward_mode=pbrs. Both start from master seed 5903, training episode seeds at 240000, validation 300000–300099 and baseline agent seeds 600000–600099. Network, optimizer, replay, exploration, gamma and guards are unchanged. Model initialization and local RNG stream seeds match. Trajectories may diverge after learned action differences; equal seed initialization does not force identical subsequent RNG consumption. Reserved final-test seeds were not used.",
        "",
        f"Sparse control matches saved Sprint 5.9 seed-5903 step-5000 weights: **{historical_match}**. This tests backward-compatible sparse behavior despite the added logging and more frequent isolated evaluations.",
        "",
        "## Sparse validation results",
        "",
        "| Step | Arm | Wins / losses / errors | Win rate [Wilson 95% CI] | Completion | Remaining HP | Mean sparse reward |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for index in range(len(a["rounds"])):
        for label, run in runs.items():
            row = run["rounds"][index]
            low, high = row["win_rate_ci95"]
            lines.append(
                f"| {row['step']} | {label} | {row['wins']} / {row['losses']} / {row['errors']} | {row['win_rate']:.1%} [{low:.1%}, {high:.1%}] | {row['mean_objective_completion_ratio']:.2%} | {row['mean_blocker_hp_remaining']:.3f} | {row['mean_total_reward']:.3f} |"
            )
    diff = comparison["difference"]
    lines += [
        "",
        f"Final PBRS-minus-sparse win difference: {100 * diff['absolute_difference']:+.2f} pp, pointwise paired bootstrap 95% CI [{100 * diff['ci95'][0]:+.2f}, {100 * diff['ci95'][1]:+.2f}] pp; exact exploratory McNemar p={comparison['mcnemar_exact_p']:.5f}. Paired table: {comparison['counts']}.",
        "",
        "Only the predeclared final comparison receives an exploratory outcome test; the reused checkpoint curves are correlated and not independent replications. One short training seed cannot establish superiority, convergence or a causal explanation for weak learning. Nonsignificance is not equivalence. Best validation checkpoints are selected on validation, not final-test estimates.",
        "",
        "| Arm | Best validation / step | Completed episodes | Training sparse return | Training shaped return | Updates / target syncs | Wall / validation seconds |",
        "|---|---|---:|---:|---:|---|---|",
    ]
    for label, run in runs.items():
        s = run["summary"]
        lines.append(
            f"| {label} | {run['best']['win_rate']:.1%} / {run['best']['step']} | {s['completed_training_episodes']} | {s['mean_training_episode_reward']:.5f} | {s['mean_training_episode_shaped_return']:.5f} | {s['optimizer_updates']} / {s['target_synchronizations']} | {s['training_seconds']:.2f} / {s['evaluation_seconds']:.2f} |"
        )
    lines += [
        "",
        "Training returns are undiscounted natural-terminal episode means; budget truncations are separately recorded. Sparse returns are retained unchanged in old reward fields and gameplay analytics. Discounted episode returns are additionally logged for the identity audit. Replay stores exactly the configured training reward. All validation moves/episodes were audited to have original sparse rewards.",
        "",
        "## Continuous paired final effects",
        "",
        "| Metric | PBRS minus sparse | Bootstrap 95% CI |",
        "|---|---:|---|",
    ]
    for key, stats in continuous.items():
        lines.append(
            f"| {key} | {stats['absolute_difference']:+.5f} | [{stats['ci95'][0]:+.5f}, {stats['ci95'][1]:+.5f}] |"
        )
    lines += [
        "",
        "## Numerical stability and reward density",
        "",
        "| Arm / metric | Minimum | Maximum | Mean |",
        "|---|---:|---:|---:|",
    ]
    for label, run in runs.items():
        for key, stats in run["statistics"].items():
            lines.append(
                f"| {label} {key} | {stats['min']:.6f} | {stats['max']:.6f} | {stats['mean']:.6f} |"
            )
    for label, run in result["runs"].items():
        lines += [
            "",
            f"{label}: zero training reward fraction {run['zero_training_reward_fraction']:.2%}; shaping bonus distribution {run['bonus_distribution']}; maximum discounted telescoping error {run['max_telescoping_error']:.3g}.",
        ]
    active_ratio = (
        b["summary"]["collection_optimization_seconds"]
        / a["summary"]["collection_optimization_seconds"]
        - 1
    )
    lines += [
        "",
        "## Runtime and interpretation",
        "",
        f"Observed PBRS-minus-sparse active-time difference: {100 * active_ratio:+.2f}%. Both processes ran concurrently with one Torch thread each. This timing includes differing trajectories, checkpoint work and system load, not an isolated causal measurement of reward-function overhead. Both arms compute potential for logging. Total timers exclude final gameplay CSV export and plots.",
        "",
        "The unchanged 50000-step epsilon schedule leaves epsilon .905 at step 5000; each arm has only 1001 optimizer updates and one late target copy. The pilot tests plumbing and an initial learning signal, not the eventual value of PBRS. Q-value and loss scales differ between rewards and cannot be compared as direct policy-quality metrics.",
        "",
        "Do not automatically increase beta or extend training in response to noisy pilot outcomes. Review the paired validation effects and learning curves, then seek approval for a preregistered multi-seed, single-factor experiment if warranted. Preserve the same architecture/hyperparameters and sparse outcome evaluation. No PPO, new architecture or longer training was executed.",
        "",
        "## Figures",
        "",
    ]
    for filename in result["figures"]:
        if filename.endswith(".png"):
            lines += [f"![{filename}](figures/{filename})", ""]
    with (output / "report.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")
    with (output / "analysis.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    for run in runs.values():
        for path, digest in run["source_sha256"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise RuntimeError("pilot source artifacts changed during analysis")
    if (
        historical_checkpoint is not None
        and hashlib.sha256(Path(historical_checkpoint).read_bytes()).hexdigest()
        != historical_hash
    ):
        raise RuntimeError("historical checkpoint changed")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sparse-dir", type=Path, required=True)
    parser.add_argument("--pbrs-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path)
    args = parser.parse_args()
    analyze_pilot(
        args.sparse_dir,
        args.pbrs_dir,
        args.output_dir,
        historical_checkpoint=args.historical_checkpoint,
    )


if __name__ == "__main__":
    main()
