"""Analyze saved checkpoint evaluations without executing policies or training."""

import argparse
import csv
import json
from itertools import combinations
from pathlib import Path

import numpy as np

from match3_ai.evaluation.behavior_statistics import (
    holm_adjust,
    paired_difference,
    paired_outcomes,
)
from match3_ai.evaluation.fresh_protocol import sha256


def load_run(path: Path, agent_name: str) -> dict:
    summary = json.loads((path / "training_summary.json").read_text())
    config = summary["training_config"]
    groups = {}
    with (path / "evaluation/episodes.csv").open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["agent_name"] != agent_name:
                continue
            step = int(row["episode_id"].split(":")[-3])
            board = int(row["seed"])
            group = groups.setdefault(step, {})
            if board in group:
                raise ValueError(f"duplicate board at checkpoint {step}: {path}")
            if row["outcome"] not in ("WON", "LOST") or int(row["invalid_actions"]):
                raise ValueError(f"incomplete or invalid evaluation: {path}")
            group[board] = row
    if not groups:
        raise ValueError(f"no evaluations for {agent_name}: {path}")
    expected = set(
        range(
            config["evaluation_seed_start"],
            config["evaluation_seed_start"] + config["evaluation_episodes"],
        )
    )
    if any(set(group) != expected for group in groups.values()):
        raise ValueError(f"evaluation board coverage differs from saved config: {path}")
    final = config["environment_steps"]
    expected_steps = set(
        range(config["evaluation_every"], final + 1, config["evaluation_every"])
    ) | {final}
    if config.get("evaluate_step_zero", False):
        expected_steps.add(0)
    if set(groups) != expected_steps or summary["environment_steps"] != final:
        raise ValueError(f"checkpoint coverage differs from saved config: {path}")
    return {
        "seed": config["seed"],
        "path": str(path),
        "groups": dict(sorted(groups.items())),
    }


def summarize(
    runs: list[dict], checkpoints: tuple[int, ...], samples=5000, seed=52020
) -> dict:
    if len(runs) < 2 or len({r["seed"] for r in runs}) != len(runs):
        raise ValueError(
            "at least two distinct training seeds are required for sample SD"
        )
    steps = list(runs[0]["groups"])
    if any(list(r["groups"]) != steps for r in runs) or not set(checkpoints) <= set(
        steps
    ):
        raise ValueError("runs must share checkpoints, including requested comparisons")
    curve = []
    for step in steps:
        rates = [
            sum(row["outcome"] == "WON" for row in r["groups"][step].values())
            / len(r["groups"][step])
            for r in runs
        ]
        curve.append(
            {
                "step": step,
                "rates": rates,
                "mean": float(np.mean(rates)),
                "sample_std": float(np.std(rates, ddof=1)),
            }
        )
    boards = sorted(runs[0]["groups"][steps[0]])
    identity_keys = ("board_seed", "engine_seed", "level_config_id")
    paired = True
    for board in boards:
        identities = []
        for run in runs:
            for group in run["groups"].values():
                if set(group) != set(boards):
                    paired = False
                    continue
                identity = tuple(group[board].get(k, "") for k in identity_keys)
                if not all(identity):
                    paired = False
                identities.append(identity)
        if len(set(identities)) != 1:
            paired = False
    comparisons, cohort = [], []
    if paired:
        for earlier, later in combinations(sorted(checkpoints), 2):
            left, right = [], []
            for run in runs:
                a = {b: run["groups"][later][b]["outcome"] for b in boards}
                z = {b: run["groups"][earlier][b]["outcome"] for b in boards}
                comparisons.append(
                    {
                        "seed": run["seed"],
                        "earlier": earlier,
                        "later": later,
                        **paired_outcomes(a, z, samples=samples, seed=seed),
                    }
                )
                left.append([a[b] == "WON" for b in boards])
                right.append([z[b] == "WON" for b in boards])
            cohort.append(
                {
                    "earlier": earlier,
                    "later": later,
                    **paired_difference(
                        np.mean(left, axis=0),
                        np.mean(right, axis=0),
                        samples=samples,
                        seed=seed,
                    ),
                }
            )
        for row, adjusted in zip(
            comparisons, holm_adjust([r["mcnemar_exact_p"] for r in comparisons])
        ):
            row["holm_p"] = adjusted
    candidates = [
        {
            "seed": r["seed"],
            "step": row["step"],
            "win_rate": row["rates"][i],
            "checkpoint": str(Path(r["path"]) / f"policy_step_{row['step']}.pt"),
        }
        for i, r in enumerate(runs)
        for row in curve
    ]
    best = lambda rows: min(rows, key=lambda r: (-r["win_rate"], r["step"], r["seed"]))
    return {
        "runs": [{"seed": r["seed"], "path": r["path"]} for r in runs],
        "curve": curve,
        "paired_identifiers_verified": paired,
        "paired_comparisons": comparisons,
        "cohort_differences": cohort,
        "paired_skip_reason": None
        if paired
        else "missing or unequal board/engine/level identifiers or board sets",
        "best_individual_checkpoint": best(candidates),
        "best_per_seed": [
            best([c for c in candidates if c["seed"] == r["seed"]]) for r in runs
        ],
        "best_cohort_checkpoint": max(curve, key=lambda r: r["mean"]),
        "final_endpoint": curve[-1],
        "comparison_checkpoints": list(checkpoints),
        "bootstrap_samples": samples,
        "bootstrap_seed": seed,
        "scope": "exploratory development analysis; conditional board CIs use common board draws across fixed trained policies; sample SD measures observed training-seed variation; no independent test performance",
    }


def write_plots(result: dict, output: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    curve = result["curve"]
    x = [r["step"] for r in curve]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), constrained_layout=True)
    for i, run in enumerate(result["runs"]):
        axes[0].plot(x, [r["rates"][i] for r in curve], "o-", label=str(run["seed"]))
    means = np.array([r["mean"] for r in curve])
    sd = np.array([r["sample_std"] for r in curve])
    axes[1].plot(x, means, "o-", label="Mean")
    axes[1].fill_between(
        x, means - sd, means + sd, alpha=0.2, label="±1 training-seed sample SD"
    )
    for ax in axes:
        ax.set(
            xlabel="Collected environment transitions", ylabel="Development win rate"
        )
        ax.legend(fontsize=8)
        ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"learning_curves.{suffix}", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    selected = [r for r in curve if r["step"] in result["comparison_checkpoints"]]
    for i, run in enumerate(result["runs"]):
        ax.plot(
            [r["step"] for r in selected],
            [r["rates"][i] for r in selected],
            "o-",
            label=str(run["seed"]),
        )
    ax.set(
        xlabel="Collected environment transitions",
        ylabel="Development win rate",
        title="Selected checkpoint comparison",
    )
    ax.legend()
    ax.grid(alpha=0.2)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"checkpoint_comparison.{suffix}", dpi=150)
    plt.close(fig)


def report(result: dict) -> str:
    seeds = [r["seed"] for r in result["runs"]]
    lines = [
        "# Saved experiment learning-curve analysis",
        "",
        "Development-validation results only; no training or evaluation was executed.",
        "",
        "| Transitions | "
        + " | ".join(str(s) for s in seeds)
        + " | Mean | Sample SD (pp) |",
        "|---:|" + "---:|" * (len(seeds) + 2),
    ]
    for row in result["curve"]:
        lines.append(
            f"| {row['step']} | "
            + " | ".join(f"{v:.1%}" for v in row["rates"])
            + f" | {row['mean']:.2%} | {100 * row['sample_std']:.2f} |"
        )
    lines += [
        "",
        "Sample SD uses ddof=1 across training seeds; it is not a confidence interval.",
        "",
        "## Selected checkpoint differences",
        "",
    ]
    if result["paired_identifiers_verified"]:
        lines += [
            "One shared board-index stream is used for all bootstrap contrasts. Cohort CIs describe board uncertainty conditional on these trained models, not population uncertainty across training runs.",
            "",
            "| Later minus earlier | Mean change (pp) | Pointwise conditional 95% CI (pp) |",
            "|---|---:|---|",
        ]
        for r in result["cohort_differences"]:
            lo, hi = r["ci95"]
            lines.append(
                f"| {r['later']} − {r['earlier']} | {100 * r['absolute_difference']:+.2f} | [{100 * lo:+.2f}, {100 * hi:+.2f}] |"
            )
        lines += [
            "",
            "Exploratory per-seed paired comparisons; one Holm family covers all listed McNemar tests. CIs remain pointwise, not multiplicity-adjusted.",
            "",
            "| Seed | Later − earlier | Later-only wins | Earlier-only wins | Change (pp) | Raw p | Holm p |",
            "|---:|---|---:|---:|---:|---:|---:|",
        ]
        for r in result["paired_comparisons"]:
            lines.append(
                f"| {r['seed']} | {r['later']} − {r['earlier']} | {r['counts']['left_only_win']} | {r['counts']['right_only_win']} | {100 * r['difference']['absolute_difference']:+.2f} | {r['mcnemar_exact_p']:.5f} | {r['holm_p']:.5f} |"
            )
    else:
        lines.append("Paired inference skipped: " + result["paired_skip_reason"])
    best = result["best_individual_checkpoint"]
    cohort = result["best_cohort_checkpoint"]
    final = result["final_endpoint"]
    lines += [
        "",
        "## Development-selected checkpoints",
        "",
        f"Best individual: seed **{best['seed']}**, **{best['step']} transitions**, **{best['win_rate']:.1%}**; `{best['checkpoint']}`.",
        f"Best cohort checkpoint: **{cohort['step']}**, mean **{cohort['mean']:.2%}**. Fixed final endpoint: **{final['step']}**, mean **{final['mean']:.2%}**.",
        "Ties select the earliest checkpoint, then the lowest seed. Selection on these same development boards is biased; these are not independent test estimates. Three training seeds cannot establish robust algorithm superiority; nonsignificance is not equivalence.",
        "",
        "## Inputs and reproduction",
        "",
    ]
    lines.extend(f"- Seed {r['seed']}: `{r['path']}`" for r in result["runs"])
    lines += [
        "",
        f"Bootstrap: {result['bootstrap_samples']} replicates, seed {result['bootstrap_seed']}. Full counts, per-seed best checkpoints, source hashes and CIs are in `analysis.json`. Input hashes were verified unchanged. Existing output directories are rejected.",
        "",
        "![Learning curves](learning_curves.png)",
        "",
        "![Checkpoint comparison](checkpoint_comparison.png)",
    ]
    return "\n".join(lines) + "\n"


def analyze(
    run_dirs: list[Path],
    output: Path,
    *,
    checkpoints=(100000, 125000, 150000),
    agent_name="DQNAgent",
) -> dict:
    if output.exists():
        raise FileExistsError(output)
    files = [
        p / name
        for p in run_dirs
        for name in ("training_summary.json", "evaluation/episodes.csv")
    ]
    hashes = {str(p): sha256(p) for p in files}
    runs = sorted((load_run(p, agent_name) for p in run_dirs), key=lambda r: r["seed"])
    result = summarize(runs, checkpoints)
    result["source_sha256"] = hashes
    output.mkdir(parents=True, exist_ok=False)
    write_plots(result, output)
    if any(sha256(name) != digest for name, digest in hashes.items()):
        raise RuntimeError("input artifacts changed during analysis")
    with (output / "analysis.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    with (output / "report.md").open("x") as handle:
        handle.write(report(result))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--checkpoints", type=int, nargs="+", default=[100000, 125000, 150000]
    )
    parser.add_argument("--agent-name", default="DQNAgent")
    args = parser.parse_args()
    analyze(
        args.run_dir,
        args.output_dir,
        checkpoints=tuple(args.checkpoints),
        agent_name=args.agent_name,
    )


if __name__ == "__main__":
    main()
