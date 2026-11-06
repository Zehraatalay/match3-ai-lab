"""The approved 5000-step Sprint 5.8 pilot, without final-test evaluation."""

import argparse
from dataclasses import asdict
from pathlib import Path

from match3_ai.training.dqn import TrainingConfig, train_dqn


def pilot_config() -> TrainingConfig:
    return TrainingConfig(
        environment_steps=5000,
        seed=5800,
        train_seed_start=200000,
        evaluation_seed_start=300000,
        evaluation_agent_seed_start=600000,
        evaluation_episodes=100,
        evaluation_every=1000,
        seed_policy="sprint5_8",
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    config = pilot_config()
    if args.review_only:
        import json

        print(json.dumps(asdict(config), indent=2))
        return
    train_dqn(config, args.output_dir)


if __name__ == "__main__":
    main()
