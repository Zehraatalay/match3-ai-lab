"""Descriptive multi-seed DQN analysis without environment access or model selection."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from match3_ai.training.seed_policy import validate_partition


def _read_csv(path):
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _stats(values):
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.isfinite(values).all():
        raise ValueError("statistics require finite nonempty data")
    return {
        "mean": float(values.mean()),
        "sample_std": float(values.std(ddof=1)) if len(values) > 1 else None,
        "min": float(values.min()),
        "max": float(values.max()),
    }


def _rolling(values, window=100):
    values = np.asarray(values, dtype=float)
    totals = np.r_[0.0, np.cumsum(values)]
    ends = np.arange(1, len(values) + 1)
    starts = np.maximum(ends - window, 0)
    return (totals[ends] - totals[starts]) / (ends - starts)


def _load_run(directory):
    directory = Path(directory)
    names = (
        "training_summary.json",
        "training_metrics.csv",
        "training_episodes.json",
        "evaluation/episodes.csv",
    )
    hashes = {
        name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
        for name in names
    }
    summary = json.loads((directory / names[0]).read_text())
    rows = _read_csv(directory / names[1])
    episodes = json.loads((directory / names[2]).read_text())
    validation = _read_csv(directory / names[3])
    config = summary["training_config"]
    if config.get("seed_policy") != "sprint5_8" or not config.get("evaluate_step_zero"):
        raise ValueError(
            "analysis requires reserved seed partitions and step-zero evaluation"
        )
    validate_partition(
        "training", config["train_seed_start"], config["environment_steps"]
    )
    validate_partition(
        "validation", config["evaluation_seed_start"], config["evaluation_episodes"]
    )
    if len(rows) != config["environment_steps"] or [
        int(row["environment_step"]) for row in rows
    ] != list(range(1, len(rows) + 1)):
        raise ValueError("incomplete or unordered training metrics")
    if sum(episode["steps"] for episode in episodes) != len(rows):
        raise ValueError("episode steps do not reconcile")
    for episode in episodes:
        validate_partition("training", episode["environment_seed"], 1)
    if any(row["valid_action"] != "True" for row in rows):
        raise ValueError("invalid training action")
    updates = [row for row in rows if row["td_loss"]]
    expected_updates = (config["environment_steps"] - config["warmup_steps"]) // config[
        "train_every"
    ] + 1
    if len(updates) != summary["optimizer_updates"] or len(updates) != expected_updates:
        raise ValueError("optimizer updates do not reconcile")
    if (
        summary["target_synchronizations"]
        != len(updates) // config["target_sync_updates"]
    ):
        raise ValueError("target synchronization count does not reconcile")
    optimization = {
        key: _stats([float(row[key]) for row in updates])
        for key in ("td_loss", "q_mean", "q_max", "q_min", "gradient_norm")
    }
    expected_steps = [
        0,
        *range(
            config["evaluation_every"],
            config["environment_steps"] + 1,
            config["evaluation_every"],
        ),
    ]
    if expected_steps[-1] != config["environment_steps"]:
        expected_steps.append(config["environment_steps"])
    if [
        item["environment_step"] for item in summary["evaluation_rounds"]
    ] != expected_steps:
        raise ValueError("missing or unordered validation checkpoints")
    curves = []
    for item in summary["evaluation_rounds"]:
        dqn = [group for group in item["summary"] if group["agent_name"] == "DQNAgent"]
        if len(dqn) != 1 or dqn[0]["attempted"] != config["evaluation_episodes"]:
            raise ValueError("invalid validation summary")
        curves.append({"step": item["environment_step"], **dqn[0]})
    baselines = {
        group["agent_name"]: group
        for group in summary["evaluation_rounds"][0]["summary"]
        if group["agent_name"] in ("RandomAgent", "GreedyAgent")
    }
    if set(baselines) != {"RandomAgent", "GreedyAgent"}:
        raise ValueError("missing step-zero baselines")
    seen, pair_seeds = set(), {}
    for episode in validation:
        seed = int(episode["seed"])
        if (
            not config["evaluation_seed_start"]
            <= seed
            < config["evaluation_seed_start"] + config["evaluation_episodes"]
        ):
            raise ValueError("unexpected validation seed or final-test leakage")
        step = int(episode["episode_id"].split(":")[-3])
        name = episode["agent_name"]
        if step not in expected_steps or name not in {
            "DQNAgent",
            "RandomAgent",
            "GreedyAgent",
        }:
            raise ValueError("unexpected validation record")
        if name != "DQNAgent" and step != 0 and not config["baseline_every_checkpoint"]:
            raise ValueError("unexpected baseline repetition")
        key = step, name, seed
        if key in seen:
            raise ValueError("duplicate validation episode")
        seen.add(key)
        if episode["board_seed"] and episode["engine_seed"]:
            pair_seeds.setdefault(seed, set()).add(
                (
                    episode["board_seed"],
                    episode["engine_seed"],
                    episode["level_config_id"],
                )
            )
    expected = {
        (step, name, seed)
        for step in expected_steps
        for name in (
            ("DQNAgent", "RandomAgent", "GreedyAgent")
            if step == 0 or config["baseline_every_checkpoint"]
            else ("DQNAgent",)
        )
        for seed in range(
            config["evaluation_seed_start"],
            config["evaluation_seed_start"] + config["evaluation_episodes"],
        )
    }
    if seen != expected or any(len(values) != 1 for values in pair_seeds.values()):
        raise ValueError("validation coverage or paired board seeds differ")
    initial, final = curves[0], curves[-1]
    best = max(curves, key=lambda item: item["win_rate"])
    report = {
        "master_seed": config["seed"],
        "experiment_id": summary["experiment_id"],
        "directory": str(directory),
        "source_sha256": hashes,
        "config": config,
        "seed_metadata": summary["seed_metadata"],
        "level_config_id": summary["level_config_id"],
        "step_zero": initial,
        "final": final,
        "best_validation": best,
        "best_tie_policy": "earliest checkpoint, including step zero",
        "improvement_from_step_zero": final["win_rate"] - initial["win_rate"],
        "improvement_over_random": final["win_rate"]
        - baselines["RandomAgent"]["win_rate"],
        "improvement_over_greedy": final["win_rate"]
        - baselines["GreedyAgent"]["win_rate"],
        "training_win_rate": summary["training_win_rate"],
        "training_mean_reward": summary["mean_training_episode_reward"],
        "completed_training_episodes": summary["completed_training_episodes"],
        "truncated_training_episodes": sum(e["truncated"] for e in episodes),
        "wall_seconds": summary["training_seconds"],
        "validation_seconds": summary["evaluation_seconds"],
        "active_seconds": summary["collection_optimization_seconds"],
        "optimizer_updates": summary["optimizer_updates"],
        "target_synchronizations": summary["target_synchronizations"],
        "replay_size": summary["replay_size"],
        "optimization": optimization,
        "initial_weights": summary["initial_weights"],
        "final_weights": summary["final_weights"],
        "parameters_changed": summary["parameters_changed"],
        "checkpoint_inference_valid": summary["checkpoint_inference_valid"],
        "invalid_actions": summary["invalid_actions"]
        + summary["evaluation_invalid_actions"],
        "validation_errors": summary["evaluation_errors"],
        "validation_time_limits": summary["evaluation_time_limits"],
        "epsilon_last": float(rows[-1]["epsilon"]),
        "curves": curves,
        "baselines": baselines,
        "actual_training_seed_min": min(e["environment_seed"] for e in episodes),
        "actual_training_seed_max": max(e["environment_seed"] for e in episodes),
    }
    return report, rows, episodes


def create_figures(runs, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    files = []

    def save(name, xlabel, ylabel, series, *, proportion=False, band=None):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for x, y, label in series:
            ax.plot(x, y, label=label, linewidth=1.3)
        if band is not None:
            x, mean, sd = band
            ax.fill_between(
                x,
                np.maximum(0, mean - sd),
                np.minimum(1, mean + sd),
                alpha=0.2,
                label="±1 sample SD across seeds",
            )
        if proportion:
            ax.set_ylim(0, 1)
        ax.set(xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            fig.savefig(output / filename, dpi=180)
            files.append(filename)
        plt.close(fig)

    summaries = [run[0] for run in runs]
    steps = [item["step"] for item in summaries[0]["curves"]]
    rates = np.array(
        [[item["win_rate"] for item in report["curves"]] for report in summaries]
    )
    baselines = summaries[0]["baselines"]
    baseline_series = [
        (steps, [group["win_rate"]] * len(steps), name)
        for name, group in baselines.items()
    ]
    save(
        "validation_by_seed",
        "Environment steps",
        "Validation win rate",
        [
            (steps, rates[index], str(report["master_seed"]))
            for index, report in enumerate(summaries)
        ]
        + baseline_series,
        proportion=True,
    )
    mean, sd = rates.mean(axis=0), rates.std(axis=0, ddof=1)
    save(
        "mean_validation",
        "Environment steps",
        "Mean validation win rate (descriptive)",
        [(steps, mean, "DQN mean")] + baseline_series,
        proportion=True,
        band=(steps, mean, sd),
    )
    save(
        "episode_rewards",
        "Environment steps",
        "Training reward (rolling 100 terminal episodes)",
        [
            (
                [e["environment_step"] for e in episodes if e["terminated"]],
                _rolling([e["reward"] for e in episodes if e["terminated"]]),
                str(report["master_seed"]),
            )
            for report, rows, episodes in runs
        ],
    )
    save(
        "training_win_rate",
        "Environment steps",
        "Cumulative terminal training win rate",
        [
            (
                [int(row["environment_step"]) for row in rows],
                [
                    float(row["training_win_rate"])
                    if row["training_win_rate"]
                    else np.nan
                    for row in rows
                ],
                str(report["master_seed"]),
            )
            for report, rows, episodes in runs
        ],
        proportion=True,
    )
    for name, key, ylabel in (
        ("td_loss", "td_loss", "Huber loss (rolling 100 updates)"),
        (
            "gradient_norm",
            "gradient_norm",
            "Pre-clip L2 gradient norm (rolling 100 updates)",
        ),
    ):
        series = []
        for report, rows, episodes in runs:
            updated = [row for row in rows if row["td_loss"]]
            series.append(
                (
                    [int(row["optimizer_updates"]) for row in updated],
                    _rolling([float(row[key]) for row in updated]),
                    str(report["master_seed"]),
                )
            )
        save(name, "Optimizer updates", ylabel, series)
    save(
        "epsilon",
        "Environment steps",
        "Collection epsilon (identical schedule)",
        [
            (
                [int(row["environment_step"]) for row in rows],
                [float(row["epsilon"]) for row in rows],
                str(report["master_seed"]),
            )
            for report, rows, episodes in runs
        ],
        proportion=True,
    )
    series = []
    for report, rows, episodes in runs:
        updated = [row for row in rows if row["td_loss"]]
        for key in ("q_mean", "q_max"):
            series.append(
                (
                    [int(row["environment_step"]) for row in updated],
                    _rolling([float(row[key]) for row in updated]),
                    f"{report['master_seed']} {key}",
                )
            )
    save(
        "q_values",
        "Environment steps",
        "Valid-action Q statistics (rolling 100 updates)",
        series,
    )
    for name, key, ylabel in (
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
        series = [
            (
                steps,
                [item[key] for item in report["curves"]],
                str(report["master_seed"]),
            )
            for report in summaries
        ]
        series += [
            (steps, [group[key]] * len(steps), name)
            for name, group in baselines.items()
        ]
        save(
            name,
            "Environment steps",
            ylabel,
            series,
            proportion=name == "objective_completion",
        )
    return files


def analyze_runs(input_dirs, output_dir):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    if len(input_dirs) != 3:
        raise ValueError("exactly three independent run directories are required")
    runs = [_load_run(directory) for directory in input_dirs]
    reports = [run[0] for run in runs]
    seeds = [report["master_seed"] for report in reports]
    if len(set(seeds)) != 3 or len({r["initial_weights"] for r in reports}) != 3:
        raise ValueError("runs must have independent seeds and initial weights")
    reference = reports[0]
    for report in reports[1:]:
        for key in ("level_config_id",):
            if report[key] != reference[key]:
                raise ValueError("level configurations differ")
        left, right = dict(reference["config"]), dict(report["config"])
        for key in ("seed", "train_seed_start"):
            left.pop(key)
            right.pop(key)
        if left != right:
            raise ValueError("training or evaluation configurations differ")
        for name, group in reference["baselines"].items():
            for key in (
                "wins",
                "losses",
                "errors",
                "time_limits",
                "mean_objective_completion_ratio",
                "mean_blocker_hp_remaining",
                "mean_total_reward",
            ):
                if group[key] != report["baselines"][name][key]:
                    raise ValueError("baseline gameplay results differ across runs")
    steps = [item["step"] for item in reference["curves"]]
    mean_curves = [
        {
            "step": step,
            **_stats([report["curves"][index]["win_rate"] for report in reports]),
        }
        for index, step in enumerate(steps)
    ]
    result = {
        "runs": reports,
        "mean_validation_curve": mean_curves,
        "aggregate": {
            key: _stats(
                [
                    report[key] if key in report else report["final"][key]
                    for report in reports
                ]
            )
            for key in (
                "improvement_from_step_zero",
                "improvement_over_random",
                "improvement_over_greedy",
                "win_rate",
                "training_win_rate",
                "wall_seconds",
            )
        },
        "final_test_evaluated": False,
        "uncertainty": "Wilson intervals condition on each fixed policy; sample SD describes only three training seeds; reused validation boards/checkpoints are correlated",
    }
    result["aggregate"]["step_zero_win_rate"] = _stats(
        [r["step_zero"]["win_rate"] for r in reports]
    )
    result["aggregate"]["best_validation_win_rate"] = _stats(
        [r["best_validation"]["win_rate"] for r in reports]
    )
    output.mkdir(parents=True, exist_ok=False)
    result["figures"] = create_figures(runs, output / "figures")
    lines = [
        "# Sprint 5.9: multi-seed DQN learning evaluation",
        "",
        "## Protocol and preflight",
        "",
        "Three fresh vanilla DQN models, replay buffers and local exploration/replay RNG streams. CPU MLP 1467 → 256 → 128 → 144; raw categorical observations, masked actions/targets, sparse rewards, Adam/Huber and gradient clipping are unchanged. Training episode seed starts differ across runs; validation always uses seeds 300000–300099 with epsilon zero.",
        "",
        "Training partition: 200000–299999; validation: 300000–300999; final test: 400000–400999, **never accessed**. Baseline agent seeds are 600000–600099. Baselines were freshly evaluated at step zero in each run and agreed; later comparisons reuse those same results rather than repeating them as independent samples.",
        "",
        "Epsilon reaches .05 at 50000 steps; the last action uses .050019. Warm-up 1000, one update every four steps, batch 64, replay 50000, gamma .99, Adam 3e-4, clipping 10, hard target sync every 1000 optimizer updates. Expected updates/syncs per run: 12251/12.",
        "",
        "Milestone checkpoints are atomically published and never overwritten. Each retains model/target/optimizer, encoder/configuration, master/derived seeds, exploration RNG and validation metrics. Milestone validation CSV/JSON and episode histories survive interruption. Periodic checkpoints support policy/optimizer recovery; they omit full replay and in-flight engine state and are not exact mid-episode resume checkpoints. Final checkpoints retain the existing full replay/RNG boundary-resume support. Interruption and non-overwrite behavior are regression-tested.",
        "",
        "## Exact configuration",
        "",
        "```json",
        json.dumps(reference["config"], indent=2),
        "```",
        "",
        "Only the master seed and training episode-seed start differ across runs. No algorithm or hyperparameter tuning was performed.",
        "",
        "## Per-seed validation results",
        "",
        "| Seed | Step zero [Wilson 95% CI] | Final [Wilson 95% CI] | Best validation / step | Final − zero | Final − Random | Final − Greedy |",
        "|---:|---|---|---|---:|---:|---:|",
    ]

    def interval(item):
        low, high = item["win_rate_ci95"]
        return f"{item['win_rate']:.1%} [{low:.1%}, {high:.1%}]"

    for r in reports:
        lines.append(
            f"| {r['master_seed']} | {interval(r['step_zero'])} | {interval(r['final'])} | {r['best_validation']['win_rate']:.1%} / {r['best_validation']['step']} | {100 * r['improvement_from_step_zero']:+.1f} pp | {100 * r['improvement_over_random']:+.1f} pp | {100 * r['improvement_over_greedy']:+.1f} pp |"
        )
    final_stats = result["aggregate"]["win_rate"]
    lines += [
        "",
        f"Final validation mean ± sample SD across the three training seeds: **{final_stats['mean']:.2%} ± {final_stats['sample_std']:.2%}**. This spread is not a confidence interval or three independent sets of validation boards.",
        f"Step-zero mean ± sample SD: {result['aggregate']['step_zero_win_rate']['mean']:.2%} ± {result['aggregate']['step_zero_win_rate']['sample_std']:.2%}. Best-validation mean ± sample SD: {result['aggregate']['best_validation_win_rate']['mean']:.2%} ± {result['aggregate']['best_validation_win_rate']['sample_std']:.2%}.",
        "",
        "Best validation includes step zero and breaks ties toward the earliest checkpoint. It is selected on validation and must not be reported as unbiased final-test performance.",
        "",
        "## Baselines and objective/blocker comparisons",
        "",
        "| Policy / seed | Win rate [Wilson 95% CI] | Completion | Remaining HP | Mean reward |",
        "|---|---|---:|---:|---:|",
    ]
    for name, group in reference["baselines"].items():
        lines.append(
            f"| {name} | {interval(group)} | {group['mean_objective_completion_ratio']:.2%} | {group['mean_blocker_hp_remaining']:.3f} | {group['mean_total_reward']:.3f} |"
        )
    for r in reports:
        for label in ("final", "best_validation"):
            group = r[label]
            lines.append(
                f"| {r['master_seed']} {label} | {interval(group)} | {group['mean_objective_completion_ratio']:.2%} | {group['mean_blocker_hp_remaining']:.3f} | {group['mean_total_reward']:.3f} |"
            )
    lines += [
        "",
        "## All validation checkpoints",
        "",
        "| Seed | Steps | Wins / losses / errors / limits | Win rate [Wilson 95% CI] | Completion | HP | Mean reward |",
        "|---:|---:|---|---|---:|---:|---:|",
    ]
    for r in reports:
        for c in r["curves"]:
            lines.append(
                f"| {r['master_seed']} | {c['step']} | {c['wins']} / {c['losses']} / {c['errors']} / {c['time_limits']} | {interval(c)} | {c['mean_objective_completion_ratio']:.2%} | {c['mean_blocker_hp_remaining']:.3f} | {c['mean_total_reward']:.3f} |"
            )
    lines += [
        "",
        "## Training stability and runtime",
        "",
        "Training rates concern exploratory policies on training boards, not the greedy validation policy. Natural terminal episodes define the training win/reward denominator; budget truncations are separate. Timers include validation and checkpoint overhead but exclude final CSV export and plotting. Runs executed concurrently in separate processes, so summed per-run wall times are not elapsed experiment time.",
        "",
        "| Seed | Episodes | Train win rate | Mean reward | Updates / syncs | Replay | Wall / validation seconds | Last epsilon | Invalid / errors / limits |",
        "|---:|---:|---:|---:|---|---:|---|---:|---|",
    ]
    for r in reports:
        lines.append(
            f"| {r['master_seed']} | {r['completed_training_episodes']} | {r['training_win_rate']:.2%} | {r['training_mean_reward']:.4f} | {r['optimizer_updates']} / {r['target_synchronizations']} | {r['replay_size']} | {r['wall_seconds']:.2f} / {r['validation_seconds']:.2f} | {r['epsilon_last']:.6f} | {r['invalid_actions']} / {r['validation_errors']} / {r['validation_time_limits']} |"
        )
    lines += [
        "",
        "| Seed / metric | Mean | Minimum | Maximum |",
        "|---|---:|---:|---:|",
    ]
    for r in reports:
        for name, stats in r["optimization"].items():
            lines.append(
                f"| {r['master_seed']} {name} | {stats['mean']:.6f} | {stats['min']:.6f} | {stats['max']:.6f} |"
            )
    gains = [r["improvement_from_step_zero"] for r in reports]
    lines += [
        "",
        "## Interpretation",
        "",
        f"Final validation improved over step zero in {sum(gain > 0 for gain in gains)} / 3 runs. Mean gain: {result['aggregate']['improvement_from_step_zero']['mean'] * 100:.2f} percentage points. Final DQN-minus-Random mean: {result['aggregate']['improvement_over_random']['mean'] * 100:+.2f} pp; DQN-minus-Greedy mean: {result['aggregate']['improvement_over_greedy']['mean'] * 100:+.2f} pp.",
        f"Mean validation in the last four milestones: {', '.join(str(item['step']) + ': ' + format(item['mean'], '.1%') for item in mean_curves[-4:])}. These late observations describe a trend, not independent evidence of convergence.",
        f"Observed maximum predicted Q across runs was {max(r['optimization']['q_max']['max'] for r in reports):.3f}; minimum was {min(r['optimization']['q_min']['min'] for r in reports):.3f}. True discounted sparse returns are bounded by [-1,1], so overshoot indicates value miscalibration despite finite losses. Monitor this separately from action validity and outcome quality.",
        "",
        "These are descriptive conditional validation results. The same boards are reused across checkpoints and training seeds; do not pool them into an independent binomial sample or run independent-sample tests across checkpoints. Three training runs cannot establish statistical superiority or convergence. Wilson intervals describe seed-sampled episode outcomes for a fixed policy, not training-seed uncertainty. Validation checkpoint selection adds optimism.",
        "",
        "Inspect the late curves for plateau or reversals and the loss/Q/gradient curves for instability. Finite optimization metrics establish numerical health only. Sparse rewards, the final surviving blocker, max-target overestimation and MLP generalization remain possible limitations. A larger training budget, algorithm change, reward shaping, PPO or final-test evaluation needs separate approval.",
        "",
        "## Learning curves",
        "",
        "Mean-curve shading is ±1 sample SD across training seeds, not a confidence interval. Reward/loss/Q/gradient traces use trailing 100-episode/update means where labeled; raw metrics remain in each run directory. No smoothing is applied to validation checkpoints.",
        "",
    ]
    for name in result["figures"]:
        if name.endswith(".png"):
            lines += [f"![{name}](figures/{name})", ""]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    (output / "analysis.json").write_text(
        json.dumps(result, indent=2, allow_nan=False) + "\n"
    )
    for report in reports:
        for name, digest in report["source_sha256"].items():
            if (
                hashlib.sha256(
                    (Path(report["directory"]) / name).read_bytes()
                ).hexdigest()
                != digest
            ):
                raise RuntimeError("source artifacts changed during analysis")
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dirs", type=Path, nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze_runs(args.input_dirs, args.output_dir)


if __name__ == "__main__":
    main()
