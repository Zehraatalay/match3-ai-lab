"""Learning curves and descriptive analysis of an existing DQN pilot."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from match3_ai.training.seed_policy import validate_partition


def _number(row, key):
    return float(row[key]) if row.get(key) not in (None, "") else np.nan


def create_learning_curves(rows, summary, output_dir):
    """Write seven standalone PNG/PDF figures into a new directory."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=False)
    steps = [_number(row, "environment_step") for row in rows]
    updated = [row for row in rows if row.get("td_loss") not in (None, "")]
    figures = []

    def save(name, xlabel, ylabel, series):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for x, y, label in series:
            ax.plot(x, y, label=label, linewidth=1.2)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        if name in ("training_win_rate", "validation_win_rate", "epsilon"):
            ax.set_ylim(0, 1)
        ax.grid(alpha=0.25)
        if len(series) > 1:
            ax.legend()
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            fig.savefig(destination / filename, dpi=180)
            figures.append(filename)
        plt.close(fig)

    finished = [row for row in rows if row.get("episode_reward") not in (None, "")]
    save(
        "episode_rewards",
        "Environment steps",
        "Episode reward (including truncations)",
        [
            (
                [_number(row, "environment_step") for row in finished],
                [_number(row, "episode_reward") for row in finished],
                "Training",
            )
        ],
    )
    save(
        "training_win_rate",
        "Environment steps",
        "Cumulative training win rate (terminal episodes)",
        [(steps, [_number(row, "training_win_rate") for row in rows], "Training")],
    )
    rounds = summary["evaluation_rounds"]
    series = []
    for name in ("DQNAgent", "RandomAgent", "GreedyAgent"):
        series.append(
            (
                [item["environment_step"] for item in rounds],
                [
                    next(
                        group["win_rate"]
                        for group in item["summary"]
                        if group["agent_name"] == name
                    )
                    for item in rounds
                ],
                name,
            )
        )
    save(
        "validation_win_rate",
        "Environment steps",
        "Validation win rate (fixed seeds)",
        series,
    )
    save(
        "td_loss",
        "Optimizer updates",
        "TD loss (Huber)",
        [
            (
                [_number(row, "optimizer_updates") for row in updated],
                [_number(row, "td_loss") for row in updated],
                "Training",
            )
        ],
    )
    save(
        "epsilon",
        "Environment steps",
        "Collection epsilon",
        [(steps, [_number(row, "epsilon") for row in rows], "Training")],
    )
    save(
        "q_values",
        "Environment steps",
        "Valid-action Q-values (minibatches)",
        [
            (
                [_number(row, "environment_step") for row in updated],
                [_number(row, key) for row in updated],
                key,
            )
            for key in ("q_mean", "q_max", "q_min")
        ],
    )
    save(
        "gradient_norm",
        "Optimizer updates",
        "Global L2 gradient norm before clipping",
        [
            (
                [_number(row, "optimizer_updates") for row in updated],
                [_number(row, "gradient_norm") for row in updated],
                "Training",
            )
        ],
    )
    return figures


def analyze_pilot(input_dir, output_dir):
    """Analyze recorded transitions only; never run an environment or select a model."""
    source, output = Path(input_dir), Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    filenames = (
        "training_summary.json",
        "training_metrics.csv",
        "training_episodes.json",
        "evaluation/episodes.csv",
    )
    hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest()
        for name in filenames
    }
    summary = json.loads((source / filenames[0]).read_text())
    episodes = json.loads((source / filenames[2]).read_text())
    with (source / filenames[1]).open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    config = summary["training_config"]
    if config.get("seed_policy") != "sprint5_8":
        raise ValueError("pilot analysis requires the Sprint 5.8 seed policy")
    validate_partition(
        "training", config["train_seed_start"], config["environment_steps"]
    )
    validate_partition(
        "validation", config["evaluation_seed_start"], config["evaluation_episodes"]
    )
    for episode in episodes:
        validate_partition("training", episode["environment_seed"], 1)
    with (source / "evaluation/episodes.csv").open(newline="") as handle:
        validation = list(csv.DictReader(handle))
    expected_seeds = set(
        range(
            config["evaluation_seed_start"],
            config["evaluation_seed_start"] + config["evaluation_episodes"],
        )
    )
    if len(validation) != len(summary["evaluation_rounds"]) * 3 * len(expected_seeds):
        raise ValueError("validation episode counts do not reconcile")
    if any(int(episode["seed"]) not in expected_seeds for episode in validation):
        raise ValueError("unexpected validation seed or final-test leakage")
    if len(rows) != summary["environment_steps"] - summary["training_start_step"]:
        raise ValueError("training step records do not reconcile")
    if any(row["valid_action"] != "True" for row in rows):
        raise ValueError("invalid training action recorded")
    updated = [row for row in rows if row["td_loss"]]
    stats = {}
    for key in ("td_loss", "q_mean", "q_max", "q_min", "gradient_norm"):
        values = np.array([float(row[key]) for row in updated])
        if not len(values) or not np.isfinite(values).all():
            raise ValueError(f"missing or nonfinite optimization metrics: {key}")
        stats[key] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
        }
    terminal = [episode for episode in episodes if episode["terminated"]]
    if len(terminal) != summary["completed_training_episodes"]:
        raise ValueError("completed episode records do not reconcile")
    if sum(episode["steps"] for episode in episodes) != len(rows):
        raise ValueError("episode step totals do not reconcile")
    duration, validation_time = (
        summary["training_seconds"],
        summary["evaluation_seconds"],
    )
    active_time = duration - validation_time
    per_1000 = 1000 * active_time / len(rows)
    evaluation_round_seconds = validation_time / len(summary["evaluation_rounds"])
    projected_run = 50 * per_1000 + 10 * evaluation_round_seconds
    analysis = {
        "experiment_id": summary["experiment_id"],
        "source_sha256": hashes,
        "optimization_statistics": stats,
        "environment_steps": len(rows),
        "optimizer_updates_recorded": len(updated),
        "completed_training_episodes": len(terminal),
        "truncated_training_episodes": len(episodes) - len(terminal),
        "training_win_rate": summary["training_win_rate"],
        "mean_training_episode_reward": summary["mean_training_episode_reward"],
        "wall_clock_seconds": duration,
        "evaluation_seconds": validation_time,
        "seconds_per_1000_collection_optimization_steps": per_1000,
        "seconds_per_1000_steps_including_pilot_evaluation": 1000
        * duration
        / len(rows),
        "seconds_per_300_episode_validation_round": evaluation_round_seconds,
        "projected_50000_step_run_seconds": projected_run,
        "projected_three_run_seconds": 3 * projected_run,
        "projection_limits": "linear pilot extrapolation; excludes optional initial evaluation, analysis, repeated validation and system load variation",
        "epsilon_first": float(rows[0]["epsilon"]),
        "epsilon_last": float(rows[-1]["epsilon"]),
        "evaluation_rounds": summary["evaluation_rounds"],
        "final_test_evaluated": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    analysis["figures"] = create_learning_curves(rows, summary, output / "figures")
    rounds = summary["evaluation_rounds"]
    first_dqn = next(
        group for group in rounds[0]["summary"] if group["agent_name"] == "DQNAgent"
    )
    last_dqn = next(
        group for group in rounds[-1]["summary"] if group["agent_name"] == "DQNAgent"
    )
    lines = [
        "# Sprint 5.8 DQN pilot report",
        "",
        "## Configuration and review",
        "",
        "Vanilla CPU DQN; sparse rewards; 1467 → 256 → 128 → 144 MLP. No algorithm or observation normalization changes were made.",
        "Categorical one-hot planes, HP/3, moves/15, dimensions, log-scaled targets and capped progress remain as reviewed. Masks stay outside model features and restrict exploration and Bellman maximization. Uniform raw replay, frozen targets, Adam/Huber and gradient clipping are unchanged.",
        "```json",
        json.dumps(config, indent=2),
        "```",
        "",
        "## Seed partitions and separation",
        "",
        "Environment seeds: training 200000–299999; validation 300000–300999; reserved final test 400000–400999. The pilot uses fixed validation seeds 300000–300099 at every checkpoint. Baseline agent seeds start at 600000; network/exploration/replay use independently spawned streams from the fixed training seed.",
        "Evaluation uses a separate environment per episode, epsilon zero, eval mode and no replay insertion or optimization. Policy weights and exploration RNG are checked for mutation, and the previous model mode is restored. Checkpoint restoration includes model/target/optimizer/replay/local RNG; resume starts a new episode as documented in Sprint 5.7.",
        "Final-test performance: **NOT EVALUATED**. Validation is reused for development and is not an independent final test. Training is an evolving exploratory policy, not a held-out score.",
        "",
        "## Training and numerical stability",
        "",
        f"Total wall clock: {duration:.2f} s; validation: {validation_time:.2f} s; collection/optimization and other overhead: {active_time:.2f} s.",
        f"Completed terminal episodes: {len(terminal)}; truncations: {len(episodes) - len(terminal)}. Training win rate: {summary['training_win_rate']:.2%}; mean terminal-episode reward: {summary['mean_training_episode_reward']:.4f}.",
        f"Replay occupancy: {summary['replay_size']} / {config['replay_capacity']}; optimizer updates: {summary['optimizer_updates']}; target synchronizations: {summary['target_synchronizations']}.",
        f"Collection epsilon: {analysis['epsilon_first']:.5f} → {analysis['epsilon_last']:.5f}; scheduled epsilon at step 5000 is 0.905. Warm-up is 1000, minibatch size 64, updates every four steps, hard target copy every 1000 updates.",
        f"Invalid training actions: {summary['invalid_actions']}; invalid validation actions: {summary['evaluation_invalid_actions']}; validation errors: {summary['evaluation_errors']}; validation time limits: {summary['evaluation_time_limits']}.",
        f"Parameters changed: {summary['parameters_changed']}; checkpoint inference round trip: {summary['checkpoint_inference_valid']}.",
        "",
        "| Optimization metric | Minimum | Maximum | Mean |",
        "|---|---:|---:|---:|",
    ]
    for key, values in stats.items():
        lines.append(
            f"| {key} | {values['min']:.6f} | {values['max']:.6f} | {values['mean']:.6f} |"
        )
    lines += [
        "",
        "## Validation comparisons",
        "",
        "Win denominators include every attempted validation episode; intervals are Wilson 95%. Completion is the mean capped objective fraction; HP is total remaining blocker HP.",
        "",
        "| Step | Agent | W / L / Error / Time limit | Win rate [95% CI] | Objective completion | Blocker HP | Mean reward |",
        "|---:|---|---|---|---:|---:|---:|",
    ]
    for item in rounds:
        for group in item["summary"]:
            low, high = group["win_rate_ci95"]
            lines.append(
                f"| {item['environment_step']} | {group['agent_name']} | {group['wins']} / {group['losses']} / {group['errors']} / {group['time_limits']} | "
                f"{group['win_rate']:.1%} [{low:.1%}, {high:.1%}] | {group['mean_objective_completion_ratio']:.2%} | {group['mean_blocker_hp_remaining']:.3f} | {group['mean_total_reward']:.3f} |"
            )
    lines += [
        "",
        "## Interpretation and limitations",
        "",
        f"DQN validation win rate changed from {first_dqn['win_rate']:.1%} at the first checkpoint to {last_dqn['win_rate']:.1%} at the last. Intermediate values are shown above; endpoint movement alone does not establish learning or convergence.",
        "Finite losses, Q-values and gradients establish numerical health over this run, not policy quality. Inspect nonmonotonic validation and the Q-value plot before scaling. The pilot has one training seed and only one late target synchronization; epsilon remains high and replay contains only 5000 transitions, although sufficient for 64-entry batches. Sparse reward, vanilla max-target overestimation, one-hot spatial generalization and the final surviving blocker are potential limitations, not demonstrated defects.",
        "Deterministic short-run training, fixed-seed evaluation, evaluation/checkpoint equality and isolation are covered by fast tests. The 5000-step pilot was not repeated, so this report does not claim a full-run bitwise replication. Reused validation seeds and checkpoint selection prevent an unbiased final-performance claim. Timing is hardware/load dependent.",
        "",
        "## Learning curves",
        "",
    ]
    final_groups = {group["agent_name"]: group for group in rounds[-1]["summary"]}
    random_rate = final_groups["RandomAgent"]["win_rate"]
    greedy_rate = final_groups["GreedyAgent"]["win_rate"]
    lines += [
        f"Final-checkpoint DQN-minus-Random win difference: {100 * (last_dqn['win_rate'] - random_rate):+.1f} percentage points; DQN-minus-Greedy: {100 * (last_dqn['win_rate'] - greedy_rate):+.1f} points. These descriptive differences do not establish superiority; they come from one training seed and reused validation boards.",
        f"Observed Q extrema were {stats['q_min']['min']:.3f} and {stats['q_max']['max']:.3f}. Sparse discounted true returns lie in [-1, 1], but the unrestricted function approximator may overshoot. The observed modest overshoot merits monitoring; no numerical divergence or sustained zero-win policy collapse was observed in this pilot.",
        "The first evaluation occurs after warm-up and one optimizer update; an untrained step-zero policy was not evaluated. The rising endpoint and improving objective completion are compatible with early learning, but intermediate reversals and the single seed limit that conclusion. Training and validation use different boards and exploration settings, so their win rates must not be compared as matched samples.",
        "",
    ]
    for name in analysis["figures"]:
        if name.endswith(".png"):
            lines += [f"![{name}](figures/{name})", ""]
    lines += [
        "## Proposed next experiment — approval required",
        "",
        "Candidate: 50000 environment steps per run, three independent master seeds, same vanilla DQN defaults, validation every 5000 steps with 100 paired episodes per agent. Keep training episode seeds in the training partition and baseline seeds fixed. Do not use final-test seeds for tuning; final evaluation requires a separately approved frozen-model protocol.",
        "Suggested master seeds: 5801, 5802 and 5803. Reuse the validation seed set for development only; report every run, the final checkpoint and any validation-selected checkpoint separately. Expect 12251 optimizer updates and 12 target synchronizations per uninterrupted run with these defaults.",
        f"Measured collection/optimization throughput: {per_1000:.2f} seconds per 1000 steps; including the pilot's unusually frequent evaluation: {analysis['seconds_per_1000_steps_including_pilot_evaluation']:.2f} seconds per 1000 steps. A 300-episode validation round averages {evaluation_round_seconds:.2f} seconds.",
        f"Linear projection: {projected_run / 60:.2f} minutes per 50000-step run, {3 * projected_run / 60:.2f} minutes for three runs; allow additional headroom for system load and longer episodes. This is a planning estimate, not a runtime guarantee.",
        "Do not start that experiment automatically. If the pilot shows worsening validation, review its paired outcomes and seed variability before adjusting hyperparameters; no algorithm, reward or gameplay change is justified solely by this short run.",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    (output / "analysis.json").write_text(
        json.dumps(analysis, indent=2, allow_nan=False) + "\n"
    )
    for name, digest in hashes.items():
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != digest:
            raise RuntimeError("input artifacts changed during analysis")
    return analysis


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze_pilot(args.input_dir, args.output_dir)


if __name__ == "__main__":
    main()
