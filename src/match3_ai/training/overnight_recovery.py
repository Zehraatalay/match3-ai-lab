"""Recover failed baseline exports from existing journals, without new episodes."""

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import numpy as np

from match3_ai.evaluation.analytics import _csv_value
from match3_ai.evaluation.behavior_statistics import holm_adjust
from match3_ai.evaluation.fresh_protocol import sha256
from match3_ai.evaluation.runner import EpisodeResult
from match3_ai.training.overnight import ROOT, verify_manifest
from match3_ai.training.overnight_analysis import episode_summary, label


def recover_rows(events_path, boards):
    events = [json.loads(line) for line in Path(events_path).read_text().splitlines()]
    episodes = [e for e in events if e["event"] == "evaluation_episode"]
    if len(episodes) != len(boards) or {e["seed"] for e in episodes} != set(boards):
        raise ValueError("journal lacks complete unique board coverage")
    if any(
        e["outcome"] not in ("WON", "LOST") or e["invalid_actions"] or e["error_type"]
        for e in episodes
    ):
        raise ValueError("journal contains invalid evaluation outcomes")
    fields = list(EpisodeResult.__dataclass_fields__)
    if any(set(fields) - set(e) for e in episodes):
        raise ValueError("journal episode fields are incomplete")
    rows = [{k: e[k] for k in fields} for e in episodes]
    if any(
        not np.isfinite(
            [
                r[k]
                for k in (
                    "total_reward",
                    "objective_completion_ratio",
                    "blocker_hp_remaining",
                    "moves_used",
                )
            ]
        ).all()
        for r in rows
    ):
        raise ValueError("journal contains nonfinite metrics")
    return rows


def finalize(root=ROOT):
    root = Path(root)
    manifest = verify_manifest(root)
    original = json.loads((root / "analysis/summary.json").read_text())
    output = root / "final_analysis"
    output.mkdir(exist_ok=False)
    models = dict(original["models"])
    recovery = {}
    for spec in manifest["runs"]:
        if spec["stage"] != "baselines":
            continue
        events = root / "records" / spec["id"] / "events.jsonl"
        rows = recover_rows(events, manifest["validation_seeds"])
        group = {r["seed"]: r for r in rows}
        directory = output / "recovered_baselines" / spec["id"]
        directory.mkdir(parents=True, exist_ok=False)
        with (directory / "episodes.csv").open("x", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(EpisodeResult.__dataclass_fields__)
            )
            writer.writeheader()
            writer.writerows({k: _csv_value(v) for k, v in r.items()} for r in rows)
        state = json.loads((events.parent / "status.json").read_text())
        seconds = (
            datetime.fromisoformat(state["updated_utc"])
            - datetime.fromisoformat(state["started_utc"])
        ).total_seconds()
        model = {
            "label": label(spec),
            "final": episode_summary(group),
            "evaluation_process_seconds_upper_bound": seconds,
            "inference_seconds": sum(r["inference_time_seconds"] for r in rows),
            "recovered_from_existing_episode_journal": True,
            "original_job_status": state["status"],
            "move_trace_available": False,
            "source_journal_sha256": sha256(events),
        }
        models[spec["id"]] = model
        recovery[spec["id"]] = {
            "episode_rows": len(rows),
            "original_status": state["status"],
            "recovery_status": "complete episode table recovered; original failed status retained",
            "new_evaluation_episodes": 0,
            "missing": "baseline move trace not persisted",
        }
        with (directory / "summary.json").open("x") as f:
            json.dump(model, f, indent=2, allow_nan=False)
    aggregates = {}
    for name in sorted({r["label"] for r in models.values()}):
        selected = [r for r in models.values() if r["label"] == name]
        rates = [r["final"]["win_rate"] for r in selected]
        aggregates[name] = {
            "mean_win_rate": float(np.mean(rates)),
            "n_models": len(rates),
            "training_seed_sample_std": float(np.std(rates, ddof=1))
            if len(rates) > 1
            else None,
            "objective_completion_ratio": float(
                np.mean([r["final"]["objective_completion_ratio"] for r in selected])
            ),
            "blocker_hp_remaining": float(
                np.mean([r["final"]["blocker_hp_remaining"] for r in selected])
            ),
        }
    comparison = original["comparisons"]["exploratory_dueling"]
    tested = [
        comparison["conditional_board_p"],
        *[r["mcnemar_p"] for r in comparison["individual"]],
    ]
    adjusted = holm_adjust(tested + [1.0, 1.0, 1.0])
    comparison["holm_p_declared_seven"] = adjusted[0]
    for row, p in zip(comparison["individual"], adjusted[1:4]):
        row["holm_p_declared_seven"] = p
    comparison["incomplete_family_handling"] = (
        "three unestimable PPO hypotheses receive conservative non-rejection slots p=1 for adjustment only; no PPO test was observed or estimated"
    )
    result = {
        **original,
        "models": models,
        "aggregates": aggregates,
        "baseline_recovery": recovery,
        "recovery_new_evaluation_episodes": 0,
        "original_manifest_sha256": sha256(root / "manifest.json"),
        "operational_amendment_sha256": sha256(root / "operational_amendment_001.json"),
        "completed_production_training_transitions": 301024,
        "PPO_training_transitions": 0,
        "unanswered_primary_question": "PPO LR comparison unavailable; all six attempts failed in initial evaluation export before training",
        "recommended_retained_checkpoint": str(
            root / "runs/c_dueling_5950/checkpoint_final.pt"
        ),
        "retention_caveat": "highest observed fixed endpoint on development data, not unbiased held-out selection; retain all six DQN-family final checkpoints",
    }
    result["final_figures"] = figures(models, aggregates, output / "figures")
    with (output / "summary.json").open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
    with (output / "comparison.csv").open("x", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "variant",
                "models",
                "win_rate",
                "training_seed_sd",
                "objective_completion",
                "remaining_hp",
            ],
        )
        writer.writeheader()
        for name, r in aggregates.items():
            writer.writerow(
                dict(
                    zip(
                        writer.fieldnames,
                        [
                            name,
                            r["n_models"],
                            r["mean_win_rate"],
                            r["training_seed_sample_std"],
                            r["objective_completion_ratio"],
                            r["blocker_hp_remaining"],
                        ],
                    )
                )
            )
    return result


def figures(models, aggregates, output):
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

    names = list(aggregates)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    for ax, key in zip(
        axes, ("mean_win_rate", "objective_completion_ratio", "blocker_hp_remaining")
    ):
        ax.barh(names, [aggregates[n][key] for n in names])
        ax.set_title(key.replace("_", " "))
    save(fig, "full_algorithm_comparison")
    fig, axes = plt.subplots(1, 3, figsize=(17, 5))
    for ident, m in models.items():
        if "curve" in m:
            axes[0].plot(
                [c["step"] for c in m["curve"]],
                [c["win_rate"] for c in m["curve"]],
                "o-",
                label=ident,
            )
            axes[2].scatter(
                m["training_seconds"] / 60, m["final"]["win_rate"], label=ident
            )
        axes[1].scatter(m["label"], m["final"]["win_rate"])
    axes[0].set(
        xlabel="Collected transitions",
        ylabel="Win rate",
        title="DQN-family learning curves",
    )
    axes[0].legend(fontsize=6)
    axes[1].tick_params(axis="x", rotation=65)
    axes[1].set_title("Per-model endpoint win rates")
    axes[2].set(
        xlabel="Total process minutes incl. validation",
        ylabel="Win rate",
        title="New training cost versus endpoint",
    )
    axes[2].legend(fontsize=6)
    save(fig, "learning_variability_and_cost")
    from shutil import copyfile

    original = output.parents[1] / "analysis/figures"
    for suffix in ("png", "pdf"):
        name = f"dqn_diagnostics.{suffix}"
        copyfile(original / name, output / name)
        files.append(name)
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    finalize(parser.parse_args().root)


if __name__ == "__main__":
    main()
