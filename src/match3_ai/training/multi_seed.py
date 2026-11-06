"""Approved Sprint 5.9 run profiles; each invocation owns a fresh CPU process."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from match3_ai.training.dqn import TrainingConfig, train_dqn

MASTER_SEEDS = (5901, 5902, 5903)


def experiment_config(master_seed: int) -> TrainingConfig:
    if master_seed not in MASTER_SEEDS:
        raise ValueError("master_seed must be one of 5901, 5902, 5903")
    return TrainingConfig(
        environment_steps=50000,
        seed=master_seed,
        train_seed_start=200000 + MASTER_SEEDS.index(master_seed) * 20000,
        evaluation_seed_start=300000,
        evaluation_agent_seed_start=600000,
        evaluation_episodes=100,
        evaluation_every=5000,
        seed_policy="sprint5_8",
        evaluate_step_zero=True,
        baseline_every_checkpoint=False,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--master-seed", type=int, choices=MASTER_SEEDS, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    config = experiment_config(args.master_seed)
    if args.review_only:
        print(json.dumps(asdict(config), indent=2))
    else:
        train_dqn(config, args.output_dir)


if __name__ == "__main__":
    main()
