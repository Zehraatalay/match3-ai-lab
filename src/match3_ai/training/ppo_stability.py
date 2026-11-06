"""Read-only Sprint 5.18 diagnostics from recorded PPO data; no episode execution."""

import argparse
import csv
import inspect
import json
from itertools import pairwise
from pathlib import Path

import numpy as np

from match3_ai.evaluation.analysis import exact_mcnemar
from match3_ai.evaluation.behavior_statistics import holm_adjust
from match3_ai.evaluation.fresh_protocol import sha256
from match3_ai.training.ppo_analysis import read_csv
from match3_ai.training.ppo_budget import MASTER_SEEDS, MILESTONES, load_protocol
from match3_ai.training.ppo_update import generalized_advantages, rollout_diagnostics

OPTIMIZATION = (
    "approximate_kl",
    "clip_fraction",
    "entropy",
    "policy_loss",
    "value_loss",
    "gradient_norm",
    "learning_rate",
    "ratio_min",
    "ratio_max",
)
PERFORMANCE = (
    "objective_completion_ratio",
    "blocker_hp_remaining",
    "moves_used",
    "total_reward",
    "total_cleared",
    "total_cascades",
    "special_activations",
)


def distribution(values):
    """Summarize finite observations; empty/undefined series retain explicit counts."""
    values = list(values)
    defined = [float(v) for v in values if v not in (None, "")]
    if not np.isfinite(defined).all():
        raise ValueError("diagnostic observations must be finite")
    if not defined:
        return {"count": 0, "undefined": len(values)}
    a = np.asarray(defined)
    return {
        "count": len(a),
        "undefined": len(values) - len(a),
        "mean": float(a.mean()),
        "std": float(a.std()),
        **{
            name: float(np.quantile(a, q))
            for name, q in (
                ("min", 0),
                ("p01", 0.01),
                ("p50", 0.5),
                ("p95", 0.95),
                ("p99", 0.99),
                ("max", 1),
            )
        },
    }


def update_summary(rows, *, threshold, coefficient):
    if not rows or threshold <= 0 or coefficient < 0:
        raise ValueError("updates and valid verified coefficients are required")
    result = {k: distribution(r[k] for r in rows) for k in OPTIMIZATION}
    norms = np.array([float(r["gradient_norm"]) for r in rows])
    result.update(
        optimizer_steps=len(rows),
        sampled_presentations=sum(int(r["minibatch_size"]) for r in rows),
        gradient_clipped_count=int((norms > threshold).sum()),
        gradient_clipping_frequency=float((norms > threshold).mean()),
        gradient_scale=distribution(np.minimum(1, threshold / (norms + 1e-6))),
        entropy_bonus=distribution(coefficient * float(r["entropy"]) for r in rows),
    )
    return result


def verified_semantics(protocol):
    """Check the frozen call path before deriving clipping and entropy metrics."""
    import torch

    from match3_ai.training.ppo import train_ppo
    from match3_ai.training.ppo_update import clipped_objective, optimize_ppo

    trainer = inspect.getsource(train_ppo)
    optimizer = inspect.getsource(optimize_ppo)
    loss = inspect.getsource(clipped_objective)
    implementation = inspect.getsource(torch.nn.utils.clip_grad_norm_)
    if not all(
        (
            "norm = torch.nn.utils.clip_grad_norm_(" in optimizer,
            '"gradient_norm": float(norm)' in optimizer,
            "total_norm = _get_total_norm(" in implementation,
            "return total_norm" in implementation,
            implementation.index("total_norm = _get_total_norm(")
            < implementation.index("_clip_grads_with_norm_("),
            "entropy_coefficient=config.entropy_coefficient" in trainer,
            "gradient_clip=config.gradient_clip" in trainer,
            "- entropy_coefficient * mean_entropy" in loss,
        )
    ):
        raise ValueError("trainer or installed Torch metric semantics need review")
    configs = list(protocol["configurations"].values())
    if {c["gradient_clip"] for c in configs} != {0.5} or {
        c["entropy_coefficient"] for c in configs
    } != {0.01}:
        raise ValueError("frozen coefficients differ from reviewed configuration")
    return {
        "torch_version": torch.__version__,
        "gradient_norm": "pre-clipping total L2 norm of combined loss over all network parameters",
        "gradient_threshold": 0.5,
        "frequency_formula": "count(gradient_norm > 0.5) / optimizer_steps; unweighted per step",
        "boundary_caveat": "Torch uses min(1, 0.5/(norm+1e-6)); strict threshold ignores epsilon-only boundary scaling",
        "entropy_coefficient": 0.01,
        "constant_entropy_coefficient": True,
        "entropy_verification": "frozen config passed unchanged to optimize_ppo and clipped_objective; no assignment/schedule in reviewed trainer",
        "entropy_bonus": ".01 * mean masked categorical entropy in nats; positive bonus, negative contribution to minimized loss",
        "torch_clipping_source_sha256": sha256(
            inspect.getfile(torch.nn.utils.clip_grad_norm_)
        ),
        "separate_actor_critic_gradients": "not recorded; combined norm cannot attribute gradients to loss terms",
    }


def reconstruct_targets(transitions, rollouts, config):
    """Recompute original on-policy GAE only, without loading policies or environments."""
    advantages, targets, diagnostics = [], [], []
    for start in range(0, len(transitions), config["rollout_steps"]):
        rows = transitions[start : start + config["rollout_steps"]]
        numeric = {
            key: np.array([float(r[key]) for r in rows])
            for key in ("reward", "value", "next_value")
        }
        flags = {
            key: np.array([r[key] == "True" for r in rows], dtype=bool)
            for key in ("terminated", "truncated")
        }
        a, t = generalized_advantages(
            numeric["reward"],
            numeric["value"],
            numeric["next_value"],
            flags["terminated"],
            flags["truncated"],
            gamma=config["gamma"],
            gae_lambda=config["gae_lambda"],
        )
        d = rollout_diagnostics(numeric["value"], t, a)
        recorded = rollouts[len(diagnostics)]
        for key, value in d.items():
            if value is None:
                if recorded[key] != "":
                    raise ValueError("undefined reconstructed rollout differs")
            elif not np.isclose(value, float(recorded[key]), atol=1e-6, rtol=1e-5):
                raise ValueError("reconstructed on-policy diagnostic differs")
        advantages.extend(a.tolist())
        targets.extend(t.tolist())
        diagnostics.append(d)
    return np.asarray(advantages), np.asarray(targets)


def paired_intervals(groups, boards, *, samples=20000, seed=51818):
    """Exploratory adjacent checkpoints; one shared board draw for every contrast."""
    if samples <= 0 or not boards:
        raise ValueError("bootstrap requires boards and positive samples")
    vectors, items = [], []
    for master, checkpoints in groups.items():
        if len(checkpoints) != 5 or any(set(g) != set(boards) for g in checkpoints):
            raise ValueError("checkpoint board coverage differs")
        for i in range(4):
            left, right = checkpoints[i], checkpoints[i + 1]
            a = np.array([left[b]["outcome"] == "WON" for b in boards])
            z = np.array([right[b]["outcome"] == "WON" for b in boards])
            if any(
                g[b]["outcome"] not in ("WON", "LOST")
                for g in (left, right)
                for b in boards
            ):
                raise ValueError("paired outcomes must be completed")
            counts = {
                "both_win": int((a & z).sum()),
                "later_only_win": int((~a & z).sum()),
                "earlier_only_win": int((a & ~z).sum()),
                "both_lose": int((~a & ~z).sum()),
            }
            difference = z.astype(float) - a.astype(float)
            continuous = {
                k: np.array([float(right[b][k]) - float(left[b][k]) for b in boards])
                for k in PERFORMANCE
            }
            vectors.extend([difference, *continuous.values()])
            items.append(
                {
                    "seed": master,
                    "start": MILESTONES[i],
                    "end": MILESTONES[i + 1],
                    "exploratory": True,
                    "counts": counts,
                    "win_difference": float(difference.mean()),
                    "mcnemar_p": exact_mcnemar(
                        counts["later_only_win"], counts["earlier_only_win"]
                    ),
                    "metric_differences": {
                        k: float(v.mean()) for k, v in continuous.items()
                    },
                }
            )
    matrix = np.asarray(vectors)
    rng = np.random.default_rng(seed)
    parts = []
    for start in range(0, samples, 250):
        indices = rng.integers(
            len(boards), size=(min(250, samples - start), len(boards))
        )
        parts.append(matrix[:, indices].mean(-1))
    cis = np.quantile(np.concatenate(parts, axis=1), [0.025, 0.975], axis=1).T
    stride = 1 + len(PERFORMANCE)
    for i, (item, adjusted) in enumerate(
        zip(items, holm_adjust([r["mcnemar_p"] for r in items]))
    ):
        item["holm_p_exploratory_12"] = adjusted
        item["win_difference_ci95"] = cis[i * stride].tolist()
        item["metric_difference_ci95"] = {
            k: cis[i * stride + j + 1].tolist() for j, k in enumerate(PERFORMANCE)
        }
    return items


def trailing_mean(values, window=50):
    a = np.asarray(values, dtype=float)
    if window < 1 or a.ndim != 1 or not np.isfinite(a).all():
        raise ValueError("smoothing requires finite vectors and a positive window")
    ends = np.arange(1, len(a) + 1)
    starts = np.maximum(0, ends - window)
    sums = np.r_[0.0, a.cumsum()]
    return (sums[ends] - sums[starts]) / (ends - starts)


def figures(runs, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(exist_ok=False)
    files = []
    specifications = {
        "checkpoint_performance": (
            "checkpoints",
            [
                ("win_rate", "Win rate"),
                ("objective_completion_ratio", "Objective completion"),
                ("blocker_hp_remaining", "Remaining blocker HP"),
                ("moves_used", "Episode moves"),
                ("total_reward", "Episode reward"),
            ],
        ),
        "policy_optimization": (
            "updates",
            [
                ("approximate_kl", "Approximate KL"),
                ("clip_fraction", "Clip fraction"),
                ("entropy", "Entropy (nats)"),
                ("entropy_bonus", "Entropy bonus (+)"),
                ("policy_loss", "Policy loss"),
                ("gradient_norm", "Pre-clip L2 norm"),
                ("learning_rate", "Learning rate"),
            ],
        ),
        "critic_diagnostics": (
            "rollouts",
            [
                ("value_loss", "Half squared value loss"),
                ("explained_variance", "Pre-update explained variance"),
                ("advantage_std", "Raw advantage SD"),
                ("target_mean", "On-policy GAE target mean"),
            ],
        ),
    }
    for name, (source, specs) in specifications.items():
        fig, axes = plt.subplots(
            len(specs), 3, figsize=(15, len(specs) * 2.3), squeeze=False, sharex=True
        )
        for col, (seed, run) in enumerate(runs.items()):
            for row, (key, label) in enumerate(specs):
                ax = axes[row, col]
                records = run[source]
                x = [int(r["environment_step"]) for r in records]
                y = np.array(
                    [
                        np.nan if r.get(key) in (None, "") else float(r[key])
                        for r in records
                    ]
                )
                ax.plot(
                    x,
                    y,
                    alpha=0.25
                    if source != "checkpoints" and key != "learning_rate"
                    else 1,
                    lw=0.6,
                    marker="o" if source == "checkpoints" else None,
                )
                if source == "updates" and key != "learning_rate":
                    ax.plot(
                        x, trailing_mean(y), lw=1, label="Trailing 50 optimizer steps"
                    )
                if key == "learning_rate" and np.ptp(y) == 0 and y[0] > 0:
                    ax.set_ylim(y[0] * 0.95, y[0] * 1.05)
                    ax.ticklabel_format(
                        axis="y", style="sci", scilimits=(0, 0), useOffset=False
                    )
                if key == "gradient_norm":
                    ax.axhline(0.5, color="red", ls="--", lw=0.8)
                for milestone in MILESTONES[1:]:
                    ax.axvline(milestone, color="gray", alpha=0.2)
                ax.axvspan(75000, 100000, alpha=0.06, color="orange")
                ax.set_ylabel(label)
                ax.grid(alpha=0.15)
                if row == 0:
                    ax.set_title(f"Seed {seed}")
                if row == len(specs) - 1:
                    ax.set_xlabel("Collected transitions")
        fig.tight_layout()
        for suffix in ("png", "pdf"):
            path = output / f"{name}.{suffix}"
            fig.savefig(path, dpi=150)
            files.append(path.name)
        plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, key in zip(axes, ("entropy", "approximate_kl", "value_loss")):
        for seed, run in runs.items():
            x = [r["optimization"][key]["mean"] for r in run["intervals"]]
            y = [100 * r["paired"]["win_difference"] for r in run["intervals"]]
            ax.plot(x, y, "o-", label=str(seed), alpha=0.7)
            for i, (xx, yy) in enumerate(zip(x, y)):
                ax.annotate(f"{i + 1}", (xx, yy))
        ax.set(xlabel=f"Interval mean {key}", ylabel="Checkpoint win change (pp)")
        ax.axhline(0, color="gray", lw=0.6)
        ax.legend()
    fig.suptitle("Descriptive alignment only; interval order 1–4; no causal inference")
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        path = output / f"interval_alignment.{suffix}"
        fig.savefig(path, dpi=150)
        files.append(path.name)
    plt.close(fig)
    return files


def analyze(protocol_path, history_path, output):
    import torch

    protocol_path, output = Path(protocol_path), Path(output)
    if output.exists():
        raise FileExistsError(output)
    history = json.loads(Path(history_path).read_text())
    protected = dict(history["files_sha256"])
    protected.update(
        {str(p): sha256(p) for p in protocol_path.parent.iterdir() if p.is_file()}
    )
    protected.update(
        {
            str(p): sha256(p)
            for p in protocol_path.parent.parent.joinpath(
                "sprint5_17_ppo_analysis"
            ).rglob("*")
            if p.is_file()
        }
    )
    verify_hashes(protected)
    protocol = load_protocol(protocol_path)
    semantics = verified_semantics(protocol)
    root = protocol_path.parent.parent
    runs, groups, derived = {}, {}, []
    for seed in MASTER_SEEDS:
        path = root / f"sprint5_17_ppo_seed_{seed}"
        summary = json.loads((path / "training_summary.json").read_text())
        config = summary["training_config"]
        if config != protocol["configurations"][str(seed)]:
            raise ValueError("saved run differs from frozen protocol")
        checkpoint_configs = {}
        for checkpoint in sorted(path.glob("*.pt")):
            payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
            if json.loads(json.dumps(payload["training_config"])) != config:
                raise ValueError("checkpoint configuration differs from frozen run")
            checkpoint_configs[checkpoint.name] = {
                key: payload["training_config"][key]
                for key in ("entropy_coefficient", "gradient_clip", "learning_rate")
            }
        if len(checkpoint_configs) != 6:
            raise ValueError("expected six saved checkpoints per run")
        updates = read_csv(path / "optimization.csv")
        rollouts = read_csv(path / "rollouts.csv")
        transitions = read_csv(path / "transitions.csv")
        episodes = read_csv(path / "evaluation/episodes.csv")
        training_episodes = json.loads((path / "training_episodes.json").read_text())
        if (len(updates), len(rollouts), len(transitions), len(episodes)) != (
            6252,
            391,
            100000,
            1000,
        ):
            raise ValueError("production run accounting differs")
        for rows in (updates, rollouts):
            for r in rows:
                distribution(r.values())
        advantage, targets = reconstruct_targets(transitions, rollouts, config)
        values = np.array([float(r["value"]) for r in transitions])
        groups[seed], checkpoints = [], []
        for step in MILESTONES:
            group = {
                int(r["seed"]): r
                for r in episodes
                if int(r["episode_id"].split(":")[-3]) == step
            }
            if len(group) != 200 or set(group) != set(protocol["validation_seeds"]):
                raise ValueError("paired board identities differ")
            if any(
                int(r["invalid_actions"]) or r["error_type"] for r in group.values()
            ):
                raise ValueError("evaluation contains invalid actions or errors")
            groups[seed].append(group)
            checkpoints.append(
                {
                    "environment_step": step,
                    "win_rate": sum(r["outcome"] == "WON" for r in group.values())
                    / 200,
                    **{
                        k: float(np.mean([float(r[k]) for r in group.values()]))
                        for k in PERFORMANCE
                    },
                    "reward_counts": {
                        str(v): sum(
                            float(r["total_reward"]) == v for r in group.values()
                        )
                        for v in (-1, 0, 1)
                    },
                }
            )
        intervals = []
        for low, high in pairwise(MILESTONES):
            ur = [r for r in updates if low < int(r["environment_step"]) <= high]
            rr = [r for r in rollouts if low < int(r["environment_step"]) <= high]
            er = [r for r in training_episodes if low < r["environment_step"] <= high]
            intervals.append(
                {
                    "start": low,
                    "end": high,
                    "optimization": update_summary(
                        ur,
                        threshold=config["gradient_clip"],
                        coefficient=config["entropy_coefficient"],
                    ),
                    "rollouts": {
                        k: distribution(r[k] for r in rr)
                        for k in (
                            "explained_variance",
                            "advantage_mean",
                            "advantage_std",
                        )
                    },
                    "values": distribution(values[low:high]),
                    "gae_targets": distribution(targets[low:high]),
                    "raw_advantages": distribution(advantage[low:high]),
                    "training_episode_return": distribution(r["reward"] for r in er),
                    "training_outcomes": {
                        name: sum(r["outcome"] == name for r in er)
                        for name in ("WON", "LOST", "TIME_LIMIT")
                    },
                }
            )
        for r in updates:
            derived.append(
                {
                    "seed": seed,
                    "environment_step": r["environment_step"],
                    "optimizer_update": r["optimizer_update"],
                    "entropy_bonus": 0.01 * float(r["entropy"]),
                    "gradient_clipped": int(float(r["gradient_norm"]) > 0.5),
                }
            )
            r["entropy_bonus"] = 0.01 * float(r["entropy"])
        for i, r in enumerate(rollouts):
            batch = [u for u in updates if int(u["rollout"]) == i + 1]
            r["value_loss"] = float(np.mean([float(u["value_loss"]) for u in batch]))
            r["target_mean"] = float(targets[i * 256 : (i + 1) * 256].mean())
        runs[seed] = {
            "checkpoints": checkpoints,
            "intervals": intervals,
            "updates": updates,
            "rollouts": rollouts,
            "all_updates": update_summary(updates, threshold=0.5, coefficient=0.01),
            "training_seconds": summary["training_seconds"],
            "checkpoint_configs": checkpoint_configs,
            "local_windows": [
                {
                    "start": low,
                    "end": low + 5000,
                    "optimization": update_summary(
                        [
                            r
                            for r in updates
                            if low < int(r["environment_step"]) <= low + 5000
                        ],
                        threshold=0.5,
                        coefficient=0.01,
                    ),
                }
                for low in range(0, 100000, 5000)
            ],
            "largest_kl_updates": sorted(
                updates, key=lambda r: float(r["approximate_kl"]), reverse=True
            )[:5],
        }
    boards = protocol["validation_seeds"]
    for b in boards:
        identities = {
            (g[b]["board_seed"], g[b]["engine_seed"], g[b]["level_config_id"])
            for gs in groups.values()
            for g in gs
        }
        if len(identities) != 1:
            raise ValueError("paired reset identities differ")
    comparisons = paired_intervals(groups, boards)
    for seed, run in runs.items():
        for interval in run["intervals"]:
            interval["paired"] = next(
                c
                for c in comparisons
                if c["seed"] == seed and c["start"] == interval["start"]
            )
    unused = sorted(
        set(range(300000, 301000)) - set(history["excluded_validation_values"])
    )
    result = {
        "study": "sprint5_18_ppo_stability",
        "semantics": semantics,
        "runs": {
            s: {k: v for k, v in r.items() if k not in ("updates", "rollouts")}
            for s, r in runs.items()
        },
        "exploratory_comparisons": comparisons,
        "bootstrap": {
            "samples": 20000,
            "seed": 51818,
            "resampling": "shared board indices across all seeds/checkpoints/metrics; conditional on three training runs",
            "ci": "pointwise, not Holm adjusted",
        },
        "unexcluded_validation_seeds": unused,
        "protected_hashes": protected,
        "raw_records": {
            s: str(root / f"sprint5_17_ppo_seed_{s}") for s in MASTER_SEEDS
        },
        "training_started": False,
        "new_evaluation_episodes": 0,
        "reserved_final_test_accessed": False,
        "undefined_explained_variance": "retained as missing, never replaced with zero",
        "interval_assignment": "optimizer/rollout records assigned by completion step low < step <= high; transition distributions by collected indices; not identical sample windows at pending-rollout boundaries",
        "smoothing": "raw optimizer data plus trailing 50 optimizer steps (short prefix windows); learning-rate, critic and checkpoint figures unsmoothed",
    }
    output.mkdir(parents=True, exist_ok=False)
    with (output / "history_inventory.json").open("x") as f:
        json.dump(history, f, indent=2, allow_nan=False)
    result["figures"] = figures(runs, output / "figures")
    with (output / "derived_updates.csv").open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(derived[0]))
        writer.writeheader()
        writer.writerows(derived)
    verify_hashes(protected)
    load_protocol(protocol_path)
    result["preservation_verified"] = True
    with (output / "summary.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    return result


def verify_hashes(files):
    changed = [
        name
        for name, digest in files.items()
        if not Path(name).is_file() or sha256(name) != digest
    ]
    if changed:
        raise ValueError(f"protected artifacts changed or missing: {changed}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--history-inventory", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.protocol, args.history_inventory, args.output_dir)


if __name__ == "__main__":
    main()
