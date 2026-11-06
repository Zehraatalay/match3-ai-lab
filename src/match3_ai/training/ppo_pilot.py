"""Approved single-seed 20000-step PPO diagnostics profile."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from match3_ai.training.ppo import PPOTrainingConfig, train_ppo


def pilot_config():
    return replace(PPOTrainingConfig(), environment_steps=20000, evaluation_every=2500)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    config = pilot_config()
    if args.review_only:
        print(json.dumps(asdict(config), indent=2))
    else:
        train_ppo(config, args.output_dir)


if __name__ == "__main__":
    main()
