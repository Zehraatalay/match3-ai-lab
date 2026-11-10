"""Read-only PPO diagnostics audit, paired baseline evaluation and ten figures."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np

from match3_ai.evaluation.behavior_statistics import paired_difference, paired_outcomes


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def paired_comparison(left, right):
    """Compare final PPO (left) against one policy on identical completed seeds."""
    if set(left) != set(right):
        raise ValueError("comparison policies must cover identical validation seeds")
    return {
        "outcomes": paired_outcomes(
            {s: r["outcome"] for s, r in left.items()},
            {s: r["outcome"] for s, r in right.items()},
            seed=51414,
        ),
        "continuous": {
            key: paired_difference(
                [float(left[s][key]) for s in sorted(left)],
                [float(right[s][key]) for s in sorted(left)],
                seed=51414,
            )
            for key in (
                "objective_completion_ratio",
                "blocker_hp_remaining",
                "moves_used",
                "total_reward",
            )
        },
    }


def create_figures(rounds, updates, rollouts, baselines, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    files = []
    specifications = [
        ("validation_win_rate", "win_rate", "Validation win rate", rounds),
        (
            "objective_completion",
            "mean_objective_completion_ratio",
            "Objective completion",
            rounds,
        ),
        (
            "remaining_blocker_hp",
            "mean_blocker_hp_remaining",
            "Remaining blocker HP",
            rounds,
        ),
        *[
            (key, key, label, updates)
            for key, label in (
                ("policy_loss", "Policy loss"),
                ("value_loss", "Half squared value loss"),
                ("entropy", "Masked entropy (nats)"),
                ("approximate_kl", "Approximate KL"),
                ("clip_fraction", "Clip fraction"),
                ("gradient_norm", "Gradient norm before clipping"),
            )
        ],
        (
            "explained_variance",
            "explained_variance",
            "Pre-update explained variance",
            rollouts,
        ),
    ]
    for name, key, label, rows in specifications:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        x = [int(r["environment_step"]) for r in rows]
        y = np.array(
            [float(r[key]) if r[key] not in (None, "") else np.nan for r in rows]
        )
        if rows is updates:
            window = 20
            sums = np.r_[0.0, y.cumsum()]
            ends = np.arange(1, len(y) + 1)
            starts = np.maximum(0, ends - window)
            y = (sums[ends] - sums[starts]) / (ends - starts)
        ax.plot(x, y, label="PPO", linewidth=1.4)
        if key == "win_rate":
            ax.fill_between(
                x,
                [r["win_rate_ci95"][0] for r in rows],
                [r["win_rate_ci95"][1] for r in rows],
                alpha=0.15,
                label="Pointwise Wilson 95%",
            )
        if rows is rounds:
            for index, baseline in enumerate(baselines):
                ax.axhline(
                    baseline[key],
                    color=f"C{index + 1}",
                    linestyle="--",
                    alpha=0.65,
                    label=baseline["agent_name"],
                )
        ax.set(
            xlabel="PPO environment steps",
            ylabel=label + (" (rolling 20 updates)" if rows is updates else ""),
        )
        ax.grid(alpha=0.25)
        ax.legend(fontsize="small")
        for extension in ("png", "pdf"):
            filename = f"{name}.{extension}"
            fig.savefig(output / filename, dpi=180)
            files.append(filename)
        plt.close(fig)
    return files


def analyze_pilot(run_dir, output_dir, dqn_checkpoints):
    import torch

    from match3_ai.agents.dqn_agent import DQNAgent
    from match3_ai.agents.greedy_agent import GreedyAgent
    from match3_ai.agents.ppo_agent import PPOAgent, masked_distribution
    from match3_ai.agents.random_agent import RandomAgent
    from match3_ai.evaluation.analytics import AnalyticsCollector, export_evaluation
    from match3_ai.evaluation.experiments import summarize_level_episodes
    from match3_ai.evaluation.runner import run_episode
    from match3_ai.training.ppo import PPOTrainingConfig, environment, fingerprint

    root, output = Path(run_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    names = (
        "training_summary.json",
        "training_episodes.json",
        "optimization.csv",
        "rollouts.csv",
        "transitions.csv",
        "evaluation/episodes.csv",
        "evaluation/moves.csv",
    )
    sources = (
        [root / name for name in names]
        + sorted(root.glob("policy_step_*.pt"))
        + [root / "checkpoint_final.pt"]
        + list(map(Path, dqn_checkpoints))
    )
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}
    summary = json.loads((root / names[0]).read_text())
    config_data = dict(summary["training_config"])
    config_data["hidden_dims"] = tuple(config_data["hidden_dims"])
    config = PPOTrainingConfig(**config_data)
    transitions, updates, rollouts = [
        read_csv(root / name)
        for name in ("transitions.csv", "optimization.csv", "rollouts.csv")
    ]
    episodes = json.loads((root / "training_episodes.json").read_text())
    validation = read_csv(root / "evaluation/episodes.csv")
    if len(transitions) != config.environment_steps or sum(
        e["steps"] for e in episodes
    ) != len(transitions):
        raise ValueError("PPO transition/episode counts differ")
    if (
        len(updates) != summary["optimizer_updates"]
        or len(rollouts) != summary["rollouts"]
    ):
        raise ValueError("PPO optimization counts differ")
    if any(
        not 200000 <= int(r["episode_seed"]) <= 299999
        or r["valid_action"] != "True"
        or float(r["reward"]) not in (-1, 0, 1)
        for r in transitions
    ):
        raise ValueError("training seed, sparse reward or validity audit failed")
    if any(
        float(r["next_value"]) != 0 for r in transitions if r["terminated"] == "True"
    ):
        raise ValueError("true terminal state was bootstrapped")
    for row in updates:
        if not np.isfinite([float(v) for v in row.values()]).all():
            raise ValueError("nonfinite optimization diagnostic")
    for row in rollouts:
        if not all(v == "" or np.isfinite(float(v)) for v in row.values()):
            raise ValueError("nonfinite rollout diagnostic")
    if any(
        e["outcome"] not in ("WON", "LOST") or int(e["invalid_actions"])
        for e in validation
    ):
        raise ValueError("invalid PPO validation episode")
    expected_seeds = set(
        range(
            config.evaluation_seed_start,
            config.evaluation_seed_start + config.evaluation_episodes,
        )
    )
    checkpoint_checks = []
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = perf_counter()
    try:
        env = environment(config)
        observations = {
            seed: env.reset(seed=seed)[0] for seed in sorted(expected_seeds)
        }
        env.close()
        moves = read_csv(root / "evaluation/moves.csv")
        for milestone in summary["evaluation_rounds"]:
            step = milestone["environment_step"]
            agent, payload = PPOAgent.load_checkpoint(root / f"policy_step_{step}.pt")
            group = {
                e["episode_id"]: e
                for e in validation
                if int(e["episode_id"].split(":")[-3]) == step
            }
            if (
                len(group) != config.evaluation_episodes
                or {int(e["seed"]) for e in group.values()} != expected_seeds
            ):
                raise ValueError("checkpoint validation seed coverage differs")
            first_actions = {
                int(group[m["episode_id"]]["seed"]): int(m["action_id"])
                for m in moves
                if m["episode_id"] in group and int(m["move_index"]) == 1
            }
            state = agent.policy_rng.get_state().clone()
            maximum_error = 0.0
            for seed, obs in observations.items():
                features, mask = agent.encoder.encode(obs)
                with torch.inference_mode():
                    logits, value = agent.network(torch.from_numpy(features[None]))
                    distribution = masked_distribution(
                        logits, torch.from_numpy(mask[None])
                    )
                maximum_error = max(
                    maximum_error, abs(float(distribution.probs.sum()) - 1)
                )
                if (
                    agent.select_action(obs) != first_actions[seed]
                    or distribution.probs[0, ~torch.from_numpy(mask)].any()
                    or not torch.isfinite(value).all()
                ):
                    raise ValueError("checkpoint inference/probability audit failed")
            if maximum_error > 1e-6 or not torch.equal(
                state, agent.policy_rng.get_state()
            ):
                raise ValueError("checkpoint probability normalization or RNG changed")
            checkpoint_checks.append(
                {
                    "step": step,
                    "progress": payload["training_progress"],
                    "states_checked": len(observations),
                    "max_probability_sum_error": maximum_error,
                }
            )
        final_agent, _ = PPOAgent.load_checkpoint(root / "checkpoint_final.pt")
        if fingerprint(final_agent) != summary["final_weights"]:
            raise ValueError("final PPO checkpoint differs from summary")
        collector = AnalyticsCollector()
        baseline_metadata = []
        policies = [("RandomAgent", None), ("GreedyAgent", GreedyAgent())]
        for path in map(Path, dqn_checkpoints):
            agent, payload = DQNAgent.load_checkpoint(path)
            if agent.architecture != "mlp":
                raise ValueError("DQN references must be MLP checkpoints")
            if payload["training_config"].get("level_id") != config.level_id:
                raise ValueError("DQN reference was trained on a different level")
            step = payload["training_step"]
            training_seed = payload["training_config"].get("seed", agent.seed)
            name = f"MLP-DQN seed {training_seed} saved {step}"
            policies.append((name, agent))
            historical_file = (
                path.parent / "validation" / f"step_{step}" / "experiment_metadata.json"
            )
            historical = (
                json.loads(historical_file.read_text())
                if historical_file.exists()
                else {}
            )
            if historical:
                if historical["level_config_id"] != summary["level_config_id"]:
                    raise ValueError("DQN historical level configuration differs")
                hashes[str(historical_file)] = hashlib.sha256(
                    historical_file.read_bytes()
                ).hexdigest()
                sources.append(historical_file)
            baseline_metadata.append(
                {
                    "name": name,
                    "checkpoint": str(path),
                    "training_step": step,
                    "training_seed": training_seed,
                    "training_config": payload["training_config"],
                    "optimizer_updates": max(
                        (
                            int(state["step"])
                            for state in agent.optimizer.state.values()
                        ),
                        default=0,
                    ),
                    "historical_wall_seconds": historical.get(
                        "training_seconds_cumulative"
                    ),
                    "historical_validation_seconds": historical.get(
                        "evaluation_seconds_cumulative"
                    ),
                    "note": "existing checkpoint; no DQN training executed",
                }
            )
        for name, agent in policies:
            for index, seed in enumerate(sorted(expected_seeds)):
                env = environment(config)
                policy = RandomAgent(600000 + index) if agent is None else agent
                try:
                    record = run_episode(
                        env,
                        policy,
                        seed=seed,
                        max_steps=config.max_episode_steps,
                        experiment_id=summary["experiment_id"],
                        episode_id=f"{summary['experiment_id']}:baseline:{name}:{seed}",
                        agent_name=name,
                        agent_seed=600000 + index
                        if agent is None
                        else getattr(agent, "seed", None),
                        collector=collector,
                    )
                    if record.outcome not in ("WON", "LOST") or record.invalid_actions:
                        raise RuntimeError("baseline evaluation failed")
                finally:
                    env.close()
        baselines = summarize_level_episodes(collector.episodes)
        final = {
            int(e["seed"]): e
            for e in validation
            if int(e["episode_id"].split(":")[-3]) == config.environment_steps
        }
        comparisons = {}
        for name, _ in policies:
            reference = {
                r.seed: asdict(r) for r in collector.episodes if r.agent_name == name
            }
            for seed in final:
                if any(
                    str(final[seed][key]) != str(reference[seed][key])
                    for key in ("board_seed", "engine_seed", "level_config_id")
                ):
                    raise ValueError("baseline initial boards or engine seeds differ")
            comparisons[name] = paired_comparison(final, reference)
        initial = {
            int(e["seed"]): e
            for e in validation
            if int(e["episode_id"].split(":")[-3]) == 0
        }
        comparisons["PPO step zero"] = paired_comparison(final, initial)
        result = {
            "summary": summary,
            "baseline_summary": baselines,
            "baseline_metadata": baseline_metadata,
            "comparisons_final_ppo_minus_reference": comparisons,
            "checkpoint_checks": checkpoint_checks,
            "source_sha256": hashes,
            "analysis_and_baseline_seconds": perf_counter() - started,
            "final_test_used": False,
        }
        output.mkdir(parents=True, exist_ok=False)
        export_evaluation(
            collector,
            {
                "experiment_id": summary["experiment_id"],
                "reference_checkpoints": baseline_metadata,
            },
            output / "baselines",
        )
        result["figures"] = create_figures(
            summary["evaluation_rounds"],
            updates,
            rollouts,
            baselines,
            output / "figures",
        )
        write_report(result, rollouts, output / "report.md")
        with (output / "analysis.json").open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        if hashes != {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources
        }:
            raise RuntimeError("source artifacts changed during analysis")
        return result
    finally:
        torch.set_num_threads(previous_threads)


def write_report(result, rollouts, path):
    s = result["summary"]
    lines = [
        f"# PPO learning diagnostics: master seed {s['training_config']['seed']}",
        "",
        f"Evaluation methods and limitations: docs/RESULTS.md. Sparse rewards, unchanged architecture and optimizer settings; one master seed {s['training_config']['seed']}, {s['environment_steps']} training transitions. No DQN retraining or final-test seeds.",
        "",
        "| Step | Wins / losses / errors | Win rate [Wilson 95% CI] | Completion | Remaining HP | Moves | Reward |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for row in s["evaluation_rounds"]:
        lo, hi = row["win_rate_ci95"]
        lines.append(
            f"| {row['environment_step']} | {row['wins']}/{row['losses']}/{row['errors']} | {row['win_rate']:.1%} [{lo:.1%}, {hi:.1%}] | {row['mean_objective_completion_ratio']:.2%} | {row['mean_blocker_hp_remaining']:.3f} | {row['mean_moves_used']:.2f} | {row['mean_total_reward']:.3f} |"
        )
    lines += [
        "",
        "## Frozen baseline comparisons",
        "",
        "All policies use identical validation initial boards/engine seeds. DQN labels refer to existing checkpoints, not newly trained models. Checkpoint steps and sample budgets are listed below. Even equal transition counts differ in optimizer-update count, initialization, collection policy and wall budget; this is not a controlled sample-efficiency comparison.",
        "",
        "| Baseline | Wins / losses / errors | Win rate [Wilson 95% CI] | Completion | HP | Moves | Reward |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for row in result["baseline_summary"]:
        lo, hi = row["win_rate_ci95"]
        lines.append(
            f"| {row['agent_name']} | {row['wins']}/{row['losses']}/{row['errors']} | {row['win_rate']:.1%} [{lo:.1%}, {hi:.1%}] | {row['mean_objective_completion_ratio']:.2%} | {row['mean_blocker_hp_remaining']:.3f} | {row['mean_moves_used']:.2f} | {row['mean_total_reward']:.3f} |"
        )
    lines += [
        "",
        "| Reference | Both win / PPO only / reference only / both lose | PPO minus reference (pp), paired 95% CI |",
        "|---|---|---|",
    ]
    for name, item in result["comparisons_final_ppo_minus_reference"].items():
        counts, difference = item["outcomes"]["counts"], item["outcomes"]["difference"]
        lines.append(
            f"| {name} | {counts['both_win']}/{counts['left_only_win']}/{counts['right_only_win']}/{counts['both_lose']} | {100 * difference['absolute_difference']:+.1f} [{100 * difference['ci95'][0]:+.1f}, {100 * difference['ci95'][1]:+.1f}] |"
        )
    lines += [
        "",
        "| Existing DQN reference | Training transitions | Optimizer updates | Historical wall / validation seconds |",
        "|---|---:|---:|---|",
    ]
    for reference in result["baseline_metadata"]:
        lines.append(
            f"| {reference['name']} | {reference['training_step']} | {reference['optimizer_updates']} | {reference['historical_wall_seconds']} / {reference['historical_validation_seconds']} |"
        )
    lines += [
        "",
        "Paired bootstrap uses 5000 seed resamples, analysis seed 51414. Intervals are pointwise/exploratory, without multiplicity correction; reused validation seeds/checkpoints are correlated. Nonsignificance is not equivalence. Continuous paired effects for completion, HP, moves and reward are in analysis.json. These intervals condition on this single training seed and do not measure training-seed variability.",
        "",
        "## Diagnostics",
        "",
        "| Metric | Minimum | Maximum | Mean |",
        "|---|---:|---:|---:|",
    ]
    for key, values in s["optimization_statistics"].items():
        lines.append(
            f"| {key} | {values['min']:.6f} | {values['max']:.6f} | {values['mean']:.6f} |"
        )
    for key in (
        "explained_variance",
        "advantage_mean",
        "advantage_std",
        "training_win_rate",
        "mean_training_episode_reward",
        "learning_rate",
    ):
        values = [float(r[key]) for r in rollouts if r[key] != ""]
        lines.append(
            f"| rollout {key} | {min(values):.6f} | {max(values):.6f} | {np.mean(values):.6f} |"
            if values
            else f"| rollout {key} | undefined | undefined | undefined |"
        )
    lines += [
        "",
        f"{s['environment_steps']} transitions, {s['rollouts']} rollouts, {s['optimizer_updates']} optimizer updates. PPO wall timer {s['training_seconds']:.2f}s; validation {s['evaluation_seconds']:.2f}s; active collection/optimization/checkpoint work {s['collection_optimization_seconds']:.2f}s. Additional baseline/audit time {result['analysis_and_baseline_seconds']:.2f}s. Timers exclude final exports/plots and are machine/load dependent.",
        "",
        "All milestone policies were loaded and checked on every validation initial observation. Recorded first actions, finite values, probability normalization, zero invalid probabilities and unchanged sampling RNG were verified. Source/checkpoint hashes were checked before and after analysis. Intermediate checkpoints are taken while rollout samples remain unoptimized; progress records last_optimized_step and pending_rollout_steps. Exact training continuation is not supported: engine state and unfinished rollouts are not serialized.",
        "",
        "Finite losses demonstrate numerical health, not useful learning. Inspect pre-update explained variance, advantage scale, KL and clipping together with sparse outcome curves. Negative explained variance means the critic's residual variance exceeds target variance. Undefined constant-target cases remain gaps in the plot. Best validation checkpoints are selection-biased and are not final-test estimates. This one-seed pilot cannot establish PPO superiority or inferiority. No hyperparameter change, PBRS, CNN-PPO or multi-seed training was performed. Review the results before approving another preregistered experiment.",
        "",
        "## Figures",
        "",
    ]
    for name in result["figures"]:
        if name.endswith(".png"):
            lines += [f"![{name}](figures/{name})", ""]
    with path.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dqn-checkpoint", action="append", type=Path, required=True)
    args = parser.parse_args()
    analyze_pilot(args.run_dir, args.output_dir, args.dqn_checkpoint)


if __name__ == "__main__":
    main()
