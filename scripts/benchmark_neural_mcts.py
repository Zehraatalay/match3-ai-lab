"""Run bounded paired development evaluation without training."""

import argparse
from pathlib import Path

from match3_ai.evaluation.neural_mcts_benchmark import (
    ROUNDS,
    EvaluationConfig,
    load_policy_value_checkpoint,
    load_profiles,
    run_benchmark,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=Path("results/neural_mcts/overnight_01/checkpoints"),
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--easy-episodes", type=int, default=5)
    parser.add_argument("--control-episodes", type=int, default=5)
    parser.add_argument("--simulations", type=int, default=8)
    parser.add_argument("--env-seed-start", type=int, default=31000000)
    parser.add_argument("--agent-seed-start", type=int, default=41000000)
    args = parser.parse_args()
    checkpoints = {
        f"Neural_round_{r:04d}": args.checkpoint_dir / f"round_{r:04d}.pt"
        for r in ROUNDS
    }
    _, metadata = load_policy_value_checkpoint(checkpoints["Neural_round_0669"])
    profiles, provenance = load_profiles(metadata)
    config = EvaluationConfig(
        args.episodes,
        args.easy_episodes,
        args.control_episodes,
        args.simulations,
        args.env_seed_start,
        args.agent_seed_start,
    )
    rows = run_benchmark(
        config,
        profiles,
        checkpoints,
        args.output_dir,
        profile_provenance=provenance,
        training_metrics=args.checkpoint_dir.parent / "metrics.json",
    )
    for row in rows:
        print(
            f"{row['level_id']} {row['agent_name']}: {row['wins']}/{row['attempted']} wins, errors={row['errors']}"
        )
    print(f"Saved {args.output_dir}")


if __name__ == "__main__":
    main()
