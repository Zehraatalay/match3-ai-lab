"""Approved independent PPO run profiles; one invocation per fresh CPU process."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from match3_ai.training.ppo import PPOTrainingConfig, train_ppo

MASTER_SEEDS = (5915, 5916, 5917)


def experiment_config(master_seed):
    if type(master_seed) is not int or master_seed not in MASTER_SEEDS:
        raise ValueError("master_seed must be one of 5915, 5916, 5917")
    return replace(
        PPOTrainingConfig(),
        seed=master_seed,
        environment_steps=50000,
        evaluation_every=5000,
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
        train_ppo(config, args.output_dir)


if __name__ == "__main__":
    main()
