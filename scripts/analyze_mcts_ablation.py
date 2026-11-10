"""Analyze one completed four-way MCTS ablation without running gameplay."""

import argparse
from pathlib import Path

from match3_ai.evaluation.mcts_ablation import analyze_results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--bootstrap-seed", type=int, default=51000)
    args = parser.parse_args()
    if args.bootstrap_samples <= 0:
        parser.error("--bootstrap-samples must be positive")
    rows = analyze_results(
        args.input_dir,
        args.output_dir,
        samples=args.bootstrap_samples,
        seed=args.bootstrap_seed,
    )
    for row in rows:
        if row["metric"] == "win":
            print(
                f"{row['level_id']} {row['contrast']}: n={row['n_pairs']} difference={row['difference']} CI95=[{row['ci95_low']}, {row['ci95_high']}] discordant={row['left_only_wins']}/{row['right_only_wins']}"
            )
    print(f"Saved {len(rows)} exploratory comparison rows to {args.output_dir}")


if __name__ == "__main__":
    main()
