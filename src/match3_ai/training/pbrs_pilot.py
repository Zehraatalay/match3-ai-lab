"""Approved paired 5000-step sparse/PBRS profiles with no hyperparameter tuning."""

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

from match3_ai.training.dqn import train_dqn
from match3_ai.training.multi_seed import experiment_config


def pilot_config(mode):
    if mode not in ("sparse", "pbrs"):
        raise ValueError("mode must be sparse or pbrs")
    return replace(
        experiment_config(5903),
        environment_steps=5000,
        evaluation_every=1000,
        reward_mode=mode,
        shaping_beta=0.5,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("sparse", "pbrs"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    config = pilot_config(args.mode)
    if args.review_only:
        print(json.dumps(asdict(config), indent=2))
    else:
        train_dqn(config, args.output_dir)


if __name__ == "__main__":
    main()
