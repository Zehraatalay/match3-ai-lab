"""Aggregate independently audited PPO runs without pooling shared validation boards."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from match3_ai.evaluation.behavior_statistics import holm_adjust, paired_difference
from match3_ai.training.ppo_analysis import read_csv


def shared_board_difference(policy_wins, reference_wins):
    """Conditional paired interval for the mean of fixed policies on shared boards."""
    policies = np.asarray(policy_wins, dtype=float)
    reference = np.asarray(reference_wins, dtype=float)
    if (
        policies.ndim != 2
        or policies.shape[0] < 2
        or reference.shape != (policies.shape[1],)
    ):
        raise ValueError("policy matrix and reference must share validation boards")
    if not np.isin(policies, [0, 1]).all() or not np.isin(reference, [0, 1]).all():
        raise ValueError("win indicators must be binary")
    return paired_difference(policies.mean(axis=0), reference, seed=51515)


def learning_statistics(rounds):
    """Sample SD across training runs; checkpoint rows must be aligned."""
    steps = [[r["environment_step"] for r in run] for run in rounds]
    if len(rounds) < 2 or not steps[0] or any(s != steps[0] for s in steps):
        raise ValueError("multiple runs must have identical nonempty milestones")
    rates = np.array([[r["win_rate"] for r in run] for run in rounds])
    if not np.isfinite(rates).all() or ((rates < 0) | (rates > 1)).any():
        raise ValueError("win rates must be finite proportions")
    return {
        "steps": steps[0],
        "mean": rates.mean(axis=0).tolist(),
        "sample_std": rates.std(axis=0, ddof=1).tolist(),
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

    specifications = [
        ("validation_win_rate", "win_rate", "Validation win rate", "rounds"),
        (
            "objective_completion",
            "mean_objective_completion_ratio",
            "Objective completion",
            "rounds",
        ),
        (
            "remaining_blocker_hp",
            "mean_blocker_hp_remaining",
            "Remaining blocker HP",
            "rounds",
        ),
        *[
            (key, key, label, "updates")
            for key, label in (
                ("policy_loss", "Policy loss"),
                ("value_loss", "Value loss"),
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
            "rollouts",
        ),
    ]
    for name, key, label, source in specifications:
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        for seed, run in runs.items():
            rows = run[source]
            x = [int(r["environment_step"]) for r in rows]
            y = np.array(
                [float(r[key]) if r[key] not in (None, "") else np.nan for r in rows]
            )
            if source == "updates":
                ends = np.arange(1, len(y) + 1)
                starts = np.maximum(ends - 50, 0)
                cumulative = np.r_[0.0, y.cumsum()]
                y = (cumulative[ends] - cumulative[starts]) / (ends - starts)
            ax.plot(x, y, linewidth=1.2, label=str(seed))
        ax.set(
            xlabel="PPO environment steps",
            ylabel=label + (" (rolling 50 updates)" if source == "updates" else ""),
        )
        ax.grid(alpha=0.25)
        ax.legend(title="Master seed")
        finish(fig, name)
    fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
    mean, sd = np.array(statistics["mean"]), np.array(statistics["sample_std"])
    ax.plot(statistics["steps"], mean, label="Mean across training seeds")
    ax.fill_between(
        statistics["steps"],
        mean - sd,
        mean + sd,
        alpha=0.2,
        label="± sample SD (not a CI)",
    )
    ax.set(xlabel="PPO environment steps", ylabel="Validation win rate")
    ax.grid(alpha=0.25)
    ax.legend()
    finish(fig, "mean_validation_win_rate")
    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    labels = [f"PPO {seed}" for seed in runs]
    rows = [run["rounds"][-1] for run in runs.values()]
    for baseline in next(iter(runs.values()))["analysis"]["baseline_summary"]:
        labels.append(baseline["agent_name"])
        rows.append(baseline)
    rates = np.array([r["win_rate"] for r in rows])
    intervals = np.array([r["win_rate_ci95"] for r in rows]).T
    ax.bar(labels, rates, color=[f"C{i}" for i in range(len(labels))])
    ax.errorbar(
        labels,
        rates,
        yerr=np.vstack((rates - intervals[0], intervals[1] - rates)),
        fmt="none",
        color="black",
        capsize=3,
    )
    ax.set(ylabel="Final validation win rate (Wilson 95% CI)", ylim=(0, 1))
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    finish(fig, "ppo_vs_baselines")
    return files


def aggregate_runs(run_dirs, analysis_dirs, output_dir):
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    if len(run_dirs) != 3 or len(analysis_dirs) != 3:
        raise ValueError("three independent runs and audited analyses are required")
    runs, hashes, streams, initial_weights = {}, {}, [], []
    for run_dir, analysis_dir in zip(map(Path, run_dirs), map(Path, analysis_dirs)):
        path = analysis_dir / "analysis.json"
        analysis = json.loads(path.read_text())
        summary = analysis["summary"]
        seed = summary["training_config"]["seed"]
        if (
            seed in runs
            or Path(summary["checkpoint"]).parent.resolve() != run_dir.resolve()
        ):
            raise ValueError("duplicate seed or mismatched analysis directory")
        sources = {
            **analysis["source_sha256"],
            str(path): hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for name, digest in sources.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise ValueError("audited source artifact changed")
        hashes.update(sources)
        env_stream, agent_stream = np.random.SeedSequence(seed).spawn(2)
        agent_seed = int(agent_stream.generate_state(1)[0])
        expected = {
            name: int(stream.generate_state(1)[0])
            for name, stream in zip(
                ("model", "policy", "shuffle"),
                np.random.SeedSequence(agent_seed).spawn(3),
            )
        }
        if summary["seed_metadata"]["agent_seed"] != agent_seed or any(
            summary["seed_metadata"][name] != value for name, value in expected.items()
        ):
            raise ValueError("recorded RNG streams differ from master seed derivation")
        streams.extend([int(env_stream.generate_state(1)[0]), *expected.values()])
        initial_weights.append(summary["initial_weights"])
        runs[seed] = {
            "analysis": analysis,
            "summary": summary,
            "rounds": summary["evaluation_rounds"],
            "updates": read_csv(run_dir / "optimization.csv"),
            "rollouts": read_csv(run_dir / "rollouts.csv"),
            "episodes": read_csv(run_dir / "evaluation/episodes.csv"),
            "analysis_dir": str(analysis_dir),
        }
    if (
        set(runs) != {5915, 5916, 5917}
        or len(set(streams)) != 12
        or len(set(initial_weights)) != 3
    ):
        raise ValueError("independent approved model/RNG initializations required")
    runs = dict(sorted(runs.items()))
    first = next(iter(runs.values()))
    config = {
        k: v for k, v in first["summary"]["training_config"].items() if k != "seed"
    }
    baseline_identity = [
        (
            r["agent_name"],
            r["wins"],
            r["losses"],
            r["mean_objective_completion_ratio"],
            r["mean_blocker_hp_remaining"],
        )
        for r in first["analysis"]["baseline_summary"]
    ]
    final_groups, rows = [], []
    for seed, run in runs.items():
        summary = run["summary"]
        if {
            k: v for k, v in summary["training_config"].items() if k != "seed"
        } != config or summary["level_config_id"] != first["summary"][
            "level_config_id"
        ]:
            raise ValueError("run configurations differ beyond master seed")
        if baseline_identity != [
            (
                r["agent_name"],
                r["wins"],
                r["losses"],
                r["mean_objective_completion_ratio"],
                r["mean_blocker_hp_remaining"],
            )
            for r in run["analysis"]["baseline_summary"]
        ]:
            raise ValueError("frozen baseline results differ across paired analyses")
        best = max(run["rounds"], key=lambda r: r["win_rate"])
        initial, final = run["rounds"][0], run["rounds"][-1]
        rows.append(
            {
                "seed": seed,
                "initial": initial,
                "final": final,
                "best": best,
                "change_from_initial": final["win_rate"] - initial["win_rate"],
            }
        )
        group = {
            int(e["seed"]): e
            for e in run["episodes"]
            if int(e["episode_id"].split(":")[-3]) == final["environment_step"]
        }
        final_groups.append(group)
    expected_seeds = set(
        range(
            config["evaluation_seed_start"],
            config["evaluation_seed_start"] + config["evaluation_episodes"],
        )
    )
    if any(set(group) != expected_seeds for group in final_groups):
        raise ValueError("final policies do not share validation seeds")
    for seed in expected_seeds:
        if (
            len(
                {
                    (
                        group[seed]["board_seed"],
                        group[seed]["engine_seed"],
                        group[seed]["level_config_id"],
                    )
                    for group in final_groups
                }
            )
            != 1
        ):
            raise ValueError("shared board initialization differs across runs")
    policies = np.array(
        [
            [group[seed]["outcome"] == "WON" for seed in sorted(expected_seeds)]
            for group in final_groups
        ]
    )
    baseline_episodes = read_csv(Path(first["analysis_dir"]) / "baselines/episodes.csv")
    conditional = {}
    for name in sorted({e["agent_name"] for e in baseline_episodes}):
        group = {
            int(e["seed"]): e for e in baseline_episodes if e["agent_name"] == name
        }
        if set(group) != expected_seeds:
            raise ValueError("reference validation seeds differ")
        conditional[name] = shared_board_difference(
            policies,
            [group[seed]["outcome"] == "WON" for seed in sorted(expected_seeds)],
        )
    tests = []
    for seed, run in runs.items():
        for name, comparison in run["analysis"][
            "comparisons_final_ppo_minus_reference"
        ].items():
            tests.append({"seed": seed, "reference": name, **comparison["outcomes"]})
    for row, adjusted in zip(tests, holm_adjust([t["mcnemar_exact_p"] for t in tests])):
        row["holm_adjusted_p"] = adjusted
    statistics = learning_statistics([run["rounds"] for run in runs.values()])
    result = {
        "per_seed": rows,
        "learning_statistics": statistics,
        "mean_final_win_rate": statistics["mean"][-1],
        "sample_std_final_win_rate": statistics["sample_std"][-1],
        "conditional_shared_board_differences": conditional,
        "paired_tests": tests,
        "seed_streams_unique": True,
        "source_sha256": hashes,
        "final_test_used": False,
    }
    output.mkdir(parents=True, exist_ok=False)
    result["figures"] = create_figures(runs, statistics, output / "figures")
    write_report(result, runs, output / "report.md")
    with (output / "analysis.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    if any(
        hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest
        for name, digest in hashes.items()
    ):
        raise RuntimeError("source artifact changed during aggregation")
    return result


def write_report(result, runs, destination):
    lines = [
        "# Sprint 5.15: three-seed masked PPO",
        "",
        "Implementation/preflight audit, hyperparameters and metric definitions: docs/sprint5_15_multi_seed_ppo.md. Three independent master seeds, unchanged sparse PPO, no gameplay or DQN training changes. Validation boards 300000–300099 are shared; final-test seeds were not used.",
        "",
        "| Seed | Initial wins | Final wins [Wilson 95% CI] | Best wins / step | Final minus initial (pp) | Completion | HP | Moves | Reward |",
        "|---:|---:|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["per_seed"]:
        final, best = row["final"], row["best"]
        lo, hi = final["win_rate_ci95"]
        lines.append(
            f"| {row['seed']} | {row['initial']['wins']} | {final['wins']} [{lo:.1%}, {hi:.1%}] | {best['wins']} / {best['environment_step']} | {100 * row['change_from_initial']:+.1f} | {final['mean_objective_completion_ratio']:.2%} | {final['mean_blocker_hp_remaining']:.3f} | {final['mean_moves_used']:.2f} | {final['mean_total_reward']:.3f} |"
        )
    lines += [
        "",
        f"Mean final win rate **{result['mean_final_win_rate']:.2%}**, sample SD across three training seeds **{result['sample_std_final_win_rate']:.2%}**. SD is descriptive training-run variation, not a confidence interval. Best checkpoints are validation-selected (earliest tie), not unbiased final-test estimates.",
        "",
        "## Full learning curves",
        "",
        "| PPO step | " + " | ".join(map(str, runs)) + " | Mean | Sample SD |",
        "|---:|" + "---:|" * (len(runs) + 2),
    ]
    stats = result["learning_statistics"]
    for index, step in enumerate(stats["steps"]):
        lines.append(
            f"| {step} | "
            + " | ".join(
                f"{run['rounds'][index]['win_rate']:.1%}" for run in runs.values()
            )
            + f" | {stats['mean'][index]:.2%} | {stats['sample_std'][index]:.2%} |"
        )
    first = next(iter(runs.values()))["analysis"]
    lines += [
        "",
        "## Frozen baseline references",
        "",
        "| Policy | Wins / losses / errors | Wilson 95% CI | Completion | HP | Moves | Reward |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for r in first["baseline_summary"]:
        lo, hi = r["win_rate_ci95"]
        lines.append(
            f"| {r['agent_name']} | {r['wins']}/{r['losses']}/{r['errors']} | [{lo:.1%}, {hi:.1%}] | {r['mean_objective_completion_ratio']:.2%} | {r['mean_blocker_hp_remaining']:.3f} | {r['mean_moves_used']:.2f} | {r['mean_total_reward']:.3f} |"
        )
    lines += [
        "",
        "| PPO seed / reference | Both win / PPO only / reference only / both lose | PPO difference pp [paired 95% CI] | Exact McNemar p | Holm p |",
        "|---|---|---|---:|---:|",
    ]
    for row in result["paired_tests"]:
        c, d = row["counts"], row["difference"]
        lines.append(
            f"| {row['seed']} / {row['reference']} | {c['both_win']}/{c['left_only_win']}/{c['right_only_win']}/{c['both_lose']} | {100 * d['absolute_difference']:+.1f} [{100 * d['ci95'][0]:+.1f}, {100 * d['ci95'][1]:+.1f}] | {row['mcnemar_exact_p']:.5f} | {row['holm_adjusted_p']:.5f} |"
        )
    lines += [
        "",
        "Each policy's tests are paired on 100 boards. Holm adjustment spans all displayed final-policy baseline and initial-policy tests; it accommodates dependency but does not correct prior reuse of these familiar validation boards. Intervals are pointwise exploratory. Nonsignificance is not equivalence.",
        "",
        "### Aggregate conditional board uncertainty",
        "",
        "| Reference | Mean PPO minus reference pp | Shared-board paired 95% CI |",
        "|---|---:|---|",
    ]
    for name, item in result["conditional_shared_board_differences"].items():
        lines.append(
            f"| {name} | {100 * item['absolute_difference']:+.2f} | [{100 * item['ci95'][0]:+.2f}, {100 * item['ci95'][1]:+.2f}] |"
        )
    lines += [
        "",
        "The conditional aggregate interval averages the three fixed policies within each board and resamples the 100 boards jointly, never treating 300 correlated episodes as independent. It quantifies board uncertainty conditional on these trained models, not population training-seed uncertainty. Three seeds are too few to establish algorithm superiority. Bootstrap uses 5000 resamples; per-policy paired tests reuse seed 51414, aggregate board resampling uses 51515.",
        "",
        "## Stability, critic and runtime",
        "",
        "| Seed | Updates / rollouts | Completed training episodes | Training win rate / return | Explained variance mean / last | Wall / validation / active seconds |",
        "|---:|---|---:|---|---|---|",
    ]
    for seed, run in runs.items():
        s, rows = run["summary"], run["rollouts"]
        last = rows[-1]
        ev = [float(r["explained_variance"]) for r in rows if r["explained_variance"]]
        lines.append(
            f"| {seed} | {s['optimizer_updates']} / {s['rollouts']} | {last['completed_episodes']} | {float(last['training_win_rate']):.2%} / {float(last['mean_training_episode_reward']):.4f} | {np.mean(ev):.4f} / {ev[-1]:.4f} | {s['training_seconds']:.2f} / {s['evaluation_seconds']:.2f} / {s['collection_optimization_seconds']:.2f} |"
        )
    lines += [
        "",
        "All loss, entropy, KL, clipping and gradient diagnostics were checked finite by individual audits; invalid actions/errors are explicit failures. Explained variance uses pre-update values, with constant-target variance left undefined. Finite losses are not evidence of useful learning. Individual reports retain complete metric ranges, advantage statistics, learning rates, cumulative training returns and checkpoint checks.",
        "",
        "The three training processes ran concurrently on CPU with one Torch thread each. Process wall times include validation and shared system contention, exclude final exports, and cannot be added as elapsed experiment wall time.",
        "",
        "| Saved DQN seed | Checkpoint step / sample budget | Optimizer updates | Historical wall / validation seconds |",
        "|---:|---|---:|---|",
    ]
    for row in first["baseline_metadata"]:
        lines.append(
            f"| {row['training_seed']} | {row['training_step']} / {row['training_step']} | {row['optimizer_updates']} | {row['historical_wall_seconds']} / {row['historical_validation_seconds']} |"
        )
    lines += [
        "",
        "DQN references were loaded from existing files and never retrained. Even when transition budgets match, optimizer updates, collection trajectories, initialization and timing conditions differ. These are outcome comparisons, not a controlled algorithm sample-efficiency or speed claim. Saved DQN paths/configurations/hashes and continuous paired effects are retained in the individual analyses.",
        "",
        "## Checkpoint and continuation limits",
        "",
        "All milestone and final checkpoints were loaded; recorded actions, masks, normalization, finite values and policy RNG isolation were verified on the shared initial boards. Master/environment/model/policy/shuffle streams and distinct initial hashes were verified across runs. Checkpoint payloads carry evaluation metrics, optimizer, architecture, encoder and progress/RNG metadata. Intermediate evaluations preserve full 256-step rollouts; last_optimized_step and pending_rollout_steps document in-flight samples. Exact continuation is unsupported because engine state and unfinished rollouts are not serialized. Completed run directories are independent and never overwritten.",
        "",
        "## Recommendation",
        "",
        "Review final-minus-initial changes, training-seed variation and critic fit before approving any next study. Retain all three runs, including weak runs. Do not select a favorable seed/checkpoint and claim general superiority. No hyperparameter tuning, PBRS, CNN-PPO, or final-test evaluation was started. Any next experiment needs approval and preregistered budgets, seeds and endpoints.",
        "",
        "## Figures",
        "",
    ]
    for name in result["figures"]:
        if name.endswith(".png"):
            lines.extend([f"![{name}](figures/{name})", ""])
    with destination.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", action="append", type=Path, required=True)
    parser.add_argument("--analysis-dir", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    aggregate_runs(args.run_dir, args.analysis_dir, args.output_dir)


if __name__ == "__main__":
    main()
