"""Audit the paired architecture pilot and export descriptive comparisons."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter_ns

import numpy as np

from match3_ai.evaluation.behavior_statistics import paired_difference, paired_outcomes
from match3_ai.training.pbrs_analysis import _fingerprint, load_run


def benchmark_inference(agents, observations, *, repetitions=3, warmup=20):
    """Time batch-one greedy decisions on identical states, without environment steps."""
    import torch

    if not observations or repetitions < 1 or warmup < 0:
        raise ValueError("benchmark requires observations and positive repetitions")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    before = {name: _fingerprint(agent) for name, agent in agents.items()}
    rng_before = {
        name: json.dumps(agent.rng.bit_generator.state, sort_keys=True)
        for name, agent in agents.items()
    }
    samples = {name: [] for name in agents}
    try:
        for agent in agents.values():
            for index in range(warmup):
                agent.select_action(observations[index % len(observations)])
        names = list(agents)
        for repetition in range(repetitions):
            for observation in observations:
                for name in names[:: 1 if repetition % 2 == 0 else -1]:
                    start = perf_counter_ns()
                    action = agents[name].select_action(observation)
                    samples[name].append((perf_counter_ns() - start) / 1000)
                    if not observation["action_mask"][action]:
                        raise ValueError("benchmark policy selected a masked action")
        for name, agent in agents.items():
            if (
                _fingerprint(agent) != before[name]
                or json.dumps(agent.rng.bit_generator.state, sort_keys=True)
                != rng_before[name]
            ):
                raise RuntimeError("inference benchmark mutated policy weights or RNG")
    finally:
        torch.set_num_threads(previous_threads)
    return {
        "unit": "microseconds per batch-one encode/forward/masked-argmax decision",
        "observations": len(observations),
        "repetitions": repetitions,
        "warmup_per_policy": warmup,
        "torch_threads": 1,
        "policies": {
            name: {
                "samples": len(values),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "p95": float(np.quantile(values, 0.95)),
            }
            for name, values in samples.items()
        },
    }


def create_figures(runs, output):
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

    for name, metric, ylabel in (
        ("validation_win_rate", "win_rate", "Validation win rate"),
        (
            "objective_completion",
            "mean_objective_completion_ratio",
            "Objective completion",
        ),
        ("remaining_blocker_hp", "mean_blocker_hp_remaining", "Remaining blocker HP"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for label, run in runs.items():
            rounds = run["rounds"]
            x = [r["step"] for r in rounds]
            ax.plot(x, [r[metric] for r in rounds], marker="o", label=label)
            if metric == "win_rate":
                ax.fill_between(
                    x,
                    [r["win_rate_ci95"][0] for r in rounds],
                    [r["win_rate_ci95"][1] for r in rounds],
                    alpha=0.12,
                )
        ax.set(xlabel="Training environment steps", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        finish(fig, name)

    for name, metrics, ylabel in (
        ("td_loss", ("td_loss",), "Huber loss"),
        ("q_values", ("q_min", "q_mean", "q_max"), "Valid-action Q statistics"),
        ("gradient_norm", ("gradient_norm",), "Pre-clipping gradient norm"),
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
        for label, run in runs.items():
            rows = [r for r in run["rows"] if r["td_loss"]]
            for key in metrics:
                values = np.array([float(r[key]) for r in rows])
                ends = np.arange(1, len(values) + 1)
                starts = np.maximum(ends - 50, 0)
                sums = np.r_[0.0, values.cumsum()]
                ax.plot(
                    [int(r["environment_step"]) for r in rows],
                    (sums[ends] - sums[starts]) / (ends - starts),
                    label=f"{label} {key}",
                )
        ax.set(xlabel="Environment steps (rolling 50 updates)", ylabel=ylabel)
        ax.grid(alpha=0.25)
        ax.legend()
        finish(fig, name)

    for name, ylabel in (("runtime", "Seconds"), ("parameter_count", "Parameters")):
        fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
        labels = list(runs)
        if name == "runtime":
            active = [
                runs[k]["summary"]["collection_optimization_seconds"] for k in labels
            ]
            validation = [runs[k]["summary"]["evaluation_seconds"] for k in labels]
            ax.bar(labels, active, label="Collection/optimization/checkpoints")
            ax.bar(labels, validation, bottom=active, label="Validation")
            ax.legend()
        else:
            counts = [runs[k]["summary"]["parameter_count"] for k in labels]
            ax.bar(labels, counts)
            for index, count in enumerate(counts):
                ax.text(index, count, f"{count:,}", ha="center", va="bottom")
            ax.margins(y=0.15)
        ax.set(ylabel=ylabel)
        finish(fig, name)
    return files


def analyze_pilot(mlp_dir, cnn_dir, output_dir, *, historical_checkpoint=None):
    from match3_ai.agents.dqn_agent import DQNAgent
    from match3_ai.games.match3.environment import Match3LevelEnv

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    runs = {"mlp": load_run(mlp_dir), "cnn": load_run(cnn_dir)}
    mlp, cnn = runs.values()
    if {k: v for k, v in mlp["config"].items() if k != "architecture"} != {
        k: v for k, v in cnn["config"].items() if k != "architecture"
    }:
        raise ValueError("pilot configurations differ beyond architecture")
    if any(
        r["config"]["reward_mode"] != "sparse" or r["config"]["architecture"] != name
        for name, r in runs.items()
    ):
        raise ValueError("pilot must use MLP/CNN with sparse rewards")
    for key in ("level_config_id", "seed_metadata"):
        if mlp["summary"][key] != cnn["summary"][key]:
            raise ValueError("pilot level configuration or seed streams differ")
    if [r["step"] for r in mlp["rounds"]] != [r["step"] for r in cnn["rounds"]]:
        raise ValueError("validation milestones differ")
    initial_boards = {}
    for run in runs.values():
        for row in run["validation"]:
            initial_boards.setdefault(row["seed"], set()).add(
                (row["board_seed"], row["engine_seed"], row["level_config_id"])
            )
    if any(len(values) != 1 for values in initial_boards.values()):
        raise ValueError("paired validation initial board seeds differ")
    final = mlp["config"]["environment_steps"]
    selected = {
        name: {
            int(r["seed"]): r
            for r in run["validation"]
            if r["agent_name"] == "DQNAgent"
            and int(r["episode_id"].split(":")[-3]) == final
        }
        for name, run in runs.items()
    }
    left, right = selected["cnn"], selected["mlp"]
    if set(left) != set(right):
        raise ValueError("paired final seeds differ")
    comparison = paired_outcomes(
        {s: r["outcome"] for s, r in left.items()},
        {s: r["outcome"] for s, r in right.items()},
        seed=51212,
    )
    continuous = {
        key: paired_difference(
            [float(left[s][key]) for s in sorted(left)],
            [float(right[s][key]) for s in sorted(left)],
            seed=51212,
        )
        for key in (
            "objective_completion_ratio",
            "blocker_hp_remaining",
            "moves_used",
            "total_reward",
        )
    }
    agents = {
        name: DQNAgent.load_checkpoint(Path(run["root"]) / "checkpoint_final.pt")[0]
        for name, run in runs.items()
    }
    for name, agent in agents.items():
        if (
            sum(p.numel() for p in agent.online.parameters())
            != runs[name]["summary"]["parameter_count"]
        ):
            raise ValueError("recorded parameter count differs")
    historical_hash = None
    if historical_checkpoint is not None:
        historical, _ = DQNAgent.load_checkpoint(historical_checkpoint)
        historical_hash = hashlib.sha256(
            Path(historical_checkpoint).read_bytes()
        ).hexdigest()
        if _fingerprint(historical) != _fingerprint(agents["mlp"]):
            raise ValueError("MLP control differs from historical same-step weights")
    config = mlp["config"]
    env = Match3LevelEnv(config["level_id"])
    observations = [
        env.reset(seed=seed)[0]
        for seed in range(
            config["evaluation_seed_start"],
            config["evaluation_seed_start"] + config["evaluation_episodes"],
        )
    ]
    latency = benchmark_inference(agents, observations)
    env.close()
    result = {
        "difference_direction": "CNN minus MLP",
        "paired_final_outcomes": comparison,
        "continuous_final_differences": continuous,
        "inference_benchmark": latency,
        "historical_mlp_weights_match": historical_hash is not None,
        "historical_checkpoint_sha256": historical_hash,
        "final_test_used": False,
        "runs": {},
    }
    for name, run in runs.items():
        decisions = sum(int(r["steps"]) for r in selected[name].values())
        result["runs"][name] = {
            key: run[key]
            for key in (
                "root",
                "summary",
                "rounds",
                "best",
                "statistics",
                "source_sha256",
            )
        }
        result["runs"][name]["final_rollout_decision_microseconds"] = (
            sum(float(r["inference_time_seconds"]) for r in selected[name].values())
            * 1e6
            / decisions
        )
        result["runs"][name]["final_epsilon"] = float(run["rows"][-1]["epsilon"])
    output.mkdir(parents=True, exist_ok=False)
    result["figures"] = create_figures(runs, output / "figures")
    write_report(result, output / "report.md")
    with (output / "analysis.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    for run in runs.values():
        for path, digest in run["source_sha256"].items():
            if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
                raise RuntimeError("pilot input artifact changed")
    if (
        historical_checkpoint is not None
        and hashlib.sha256(Path(historical_checkpoint).read_bytes()).hexdigest()
        != historical_hash
    ):
        raise RuntimeError("historical checkpoint changed")
    return result


def write_report(result, destination):
    lines = [
        "# Sprint 5.12: controlled MLP/CNN pilot",
        "",
        "Hypothesis: a spatial inductive bias may improve learning; it does not guarantee better performance. Both networks receive exactly the same 1467 encoded values and a separate action mask. See docs/sprint5_12_cnn_dqn.md for the channel map, checkpoint design and reproduction protocol.",
        "",
        "CNN: Conv3x3 20→8→16, stride 1/padding 1, ReLU; flatten 1152. Global branch 27→32/ReLU. Concatenate 1184→256→128→144. No pooling, dropout or batch normalization. MLP: 1467→256→128→144. The CNN has fewer parameters but extra convolution arithmetic; this is not a capacity-matched experiment.",
        "",
        "Master seed 5903; blocker-mayhem; 5000 steps; sparse rewards; PBRS disabled. The two configurations differ only in architecture. Identical RNG stream seeds do not imply identical initial weights: layer shapes, fan-in and initialization draw counts differ. Learned trajectories and subsequent environment RNG consumption can diverge. Validation uses identical initial board/engine seeds, not forced identical trajectories.",
        "",
        "## Validation learning curves",
        "",
        "| Step | Architecture | Wins / losses / errors | Win rate [Wilson 95% CI] | Completion | Remaining HP |",
        "|---:|---|---|---|---:|---:|",
    ]
    for index in range(len(result["runs"]["mlp"]["rounds"])):
        for name, run in result["runs"].items():
            r = run["rounds"][index]
            low, high = r["win_rate_ci95"]
            lines.append(
                f"| {r['step']} | {name} | {r['wins']} / {r['losses']} / {r['errors']} | {r['win_rate']:.1%} [{low:.1%}, {high:.1%}] | {r['mean_objective_completion_ratio']:.2%} | {r['mean_blocker_hp_remaining']:.3f} |"
            )
    comparison = result["paired_final_outcomes"]
    diff = comparison["difference"]
    lines += [
        "",
        f"Final CNN-minus-MLP: {100 * diff['absolute_difference']:+.2f} percentage points; paired bootstrap 95% CI [{100 * diff['ci95'][0]:+.2f}, {100 * diff['ci95'][1]:+.2f}]; exploratory exact McNemar p={comparison['mcnemar_exact_p']:.5f}. Counts: {comparison['counts']}.",
        "",
        "Bootstrap: 5000 paired seed resamples, RNG seed 51212. Wilson intervals are pointwise. The repeated checkpoint measurements are correlated, not independent experiments. Best checkpoints are validation-selected and optimistic; ties select the earliest step. These familiar validation seeds have been reused in earlier sprints; reserved final-test seeds were not used.",
        "",
        "| Continuous final metric | CNN minus MLP | Paired bootstrap 95% CI |",
        "|---|---:|---|",
    ]
    for key, stats in result["continuous_final_differences"].items():
        lines.append(
            f"| {key} | {stats['absolute_difference']:+.5f} | [{stats['ci95'][0]:+.5f}, {stats['ci95'][1]:+.5f}] |"
        )
    lines += [
        "",
        "## Compute, parameters and training",
        "",
        "| Architecture | Parameters | Best win rate / step | Training terminal return | Updates / target syncs | Final epsilon | Active / validation / wall seconds | Median inference µs |",
        "|---|---:|---|---:|---|---:|---|---:|",
    ]
    for name, run in result["runs"].items():
        s, best = run["summary"], run["best"]
        timing = result["inference_benchmark"]["policies"][name]
        lines.append(
            f"| {name} | {s['parameter_count']:,} | {best['win_rate']:.1%} / {best['step']} | {s['mean_training_episode_reward']:.5f} | {s['optimizer_updates']} / {s['target_synchronizations']} | {run['final_epsilon']:.5f} | {s['collection_optimization_seconds']:.2f} / {s['evaluation_seconds']:.2f} / {s['training_seconds']:.2f} | {timing['median']:.2f} |"
        )
    lines += [
        "",
        "Inference: batch-one complete encode/forward/masked-argmax on the same validation initial observations, one Torch CPU thread, 20 warmups per policy and three measured passes with alternating order. Timing excludes generation and environment steps. Median/p95/mean and actual rollout decision times are saved in analysis.json. Timing is system-dependent, not deterministic. Pilot processes ran concurrently; active time includes collection, differing trajectories and checkpoint work. Wall timers exclude final CSV export/plot creation, and summed process times are not elapsed batch time.",
        "",
        f"Historical MLP same-step weights match: **{result['historical_mlp_weights_match']}**. Existing checkpoints were read and hashed, never rewritten.",
        "",
        "## Numerical diagnostics",
        "",
        "| Architecture / metric | Minimum | Maximum | Mean |",
        "|---|---:|---:|---:|",
    ]
    for name, run in result["runs"].items():
        for key, values in run["statistics"].items():
            lines.append(
                f"| {name} {key} | {values['min']:.6f} | {values['max']:.6f} | {values['mean']:.6f} |"
            )
    lines += [
        "",
        "All recorded losses, valid-action Q statistics and pre-clipping gradient norms are finite; invalid actions and evaluation errors are audited. The unchanged 50000-step epsilon schedule ends this pilot near .905, with only 1001 optimizer updates and one target synchronization. Q scales or lower loss alone do not establish better decisions.",
        "",
        "## Interpretation and recommendation",
        "",
        "This one-seed, short pilot checks integration and gives an initial signal. Architecture, initialization distribution and parameter count vary together. Paired board seeds reduce evaluation noise but do not quantify training-seed variability. A confidence interval containing zero is not proof of equivalence. A shortfall cannot be attributed to a CNN implementation defect merely from win rate: exact input round trips, legal-action masking, finite gradients, deterministic updates, checkpoint round trips and the historical MLP control are separately checked.",
        "",
        "Review these curves and compute costs before approving any preregistered multi-seed comparison. Retain MLP as default; do not automatically enlarge the CNN, extend training, enable PBRS or begin PPO. More independent training seeds would be required to assess the spatial-learning hypothesis. No long training was run.",
        "",
        "## Exact shared configuration (architecture is the sole changed field)",
        "",
        "```json",
        json.dumps(result["runs"]["mlp"]["summary"]["training_config"], indent=2),
        "```",
        "",
        "## Figures",
        "",
    ]
    for filename in result["figures"]:
        if filename.endswith(".png"):
            lines += [f"![{filename}](figures/{filename})", ""]
    with destination.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mlp-dir", type=Path, required=True)
    parser.add_argument("--cnn-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--historical-checkpoint", type=Path)
    args = parser.parse_args()
    analyze_pilot(
        args.mlp_dir,
        args.cnn_dir,
        args.output_dir,
        historical_checkpoint=args.historical_checkpoint,
    )


if __name__ == "__main__":
    main()
