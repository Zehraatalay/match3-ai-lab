"""Command-line entry point for the paired MCTS-versus-baselines benchmark."""

import argparse
import json
from pathlib import Path

from match3_ai.evaluation.mcts_baseline_benchmark import (
    AGENT_REGISTRY,
    AGENTS,
    BenchmarkConfig,
    run_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--simulations", type=int, default=30)
    parser.add_argument("--env-seed-start", type=int, default=10_000_000)
    parser.add_argument("--agent-seed-start", type=int, default=20_000_000)
    parser.add_argument(
        "--agents", nargs="+", choices=AGENT_REGISTRY, default=tuple(AGENTS)
    )
    parser.add_argument("--reuse-from", type=Path)
    parser.add_argument("--validate-reuse-only", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/mcts_baselines")
    )
    args = parser.parse_args()
    try:
        config = BenchmarkConfig(
            args.episodes,
            args.simulations,
            args.env_seed_start,
            args.agent_seed_start,
            tuple(args.agents),
        )
    except ValueError as error:
        parser.error(str(error))
    if args.validate_reuse_only:
        if args.reuse_from is None:
            parser.error("--validate-reuse-only requires --reuse-from")
        from match3_ai.evaluation.mcts_ablation import validate_reuse

        _, provenance = validate_reuse(config, args.reuse_from)
        print(json.dumps(provenance, indent=2))
        return
    rows = run_benchmark(config, args.output_dir, reuse_from=args.reuse_from)
    print("\nSUMMARY (decision and episode times in seconds)")
    for row in rows:
        winning = (
            "NA"
            if row["mean_winning_moves"] is None
            else f"{row['mean_winning_moves']:.2f}"
        )
        decision = (
            "NA"
            if row["mean_decision_time_seconds"] is None
            else f"{row['mean_decision_time_seconds']:.6f}"
        )
        completion = (
            "NA"
            if row["mean_objective_completion"] is None
            else f"{row['mean_objective_completion']:.3f}"
        )
        print(
            f"{row['level_id']} | {row['agent_name']} | n={row['attempted']} "
            f"win={row['win_rate']:.1%} [{row['win_rate_ci95_low']:.1%}, {row['win_rate_ci95_high']:.1%}] "
            f"completion={completion} winning_moves={winning} decision={decision} "
            f"episode={row['mean_episode_time_seconds']:.3f} errors={row['errors']} time_limits={row['time_limits']}"
        )
    print(f"Saved: {args.output_dir}")


if __name__ == "__main__":
    main()
