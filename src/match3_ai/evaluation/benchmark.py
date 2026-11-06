"""Run the reproducible level baseline benchmark and export its records."""

from __future__ import annotations

import argparse
from pathlib import Path

from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.evaluation.analytics import AnalyticsCollector, export_evaluation
from match3_ai.evaluation.experiments import LevelExperimentConfig, run_level_experiment
from match3_ai.evaluation.runner import EpisodeResult
from match3_ai.games.match3.engine import DEFAULT_MAX_CASCADES
from match3_ai.games.match3.environment import DEFAULT_LEVEL_INITIALIZATION_ATTEMPTS


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--env-seed", type=int, default=52000)
    parser.add_argument("--agent-seed", type=int, default=93000)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--max-cascades", type=int, default=DEFAULT_MAX_CASCADES)
    parser.add_argument(
        "--max-initialization-attempts",
        type=int,
        default=DEFAULT_LEVEL_INITIALIZATION_ATTEMPTS,
    )
    parser.add_argument("--max-seconds", type=float, default=300.0)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.output_dir is not None and args.output_dir.exists():
        parser.error("output directory already exists; choose a new directory")
    config = LevelExperimentConfig(
        num_episodes=args.episodes,
        base_seed=args.env_seed,
        base_agent_seed=args.agent_seed,
        max_steps=args.max_steps,
        max_cascades=args.max_cascades,
        max_initialization_attempts=args.max_initialization_attempts,
        max_duration_seconds=args.max_seconds,
    )
    attempted = 0

    def progress(episode: EpisodeResult) -> None:
        nonlocal attempted
        attempted += 1
        if episode.outcome == "ERROR":
            print(
                f"ERROR {episode.level_id} {episode.agent_name} seed={episode.seed} "
                f"context={episode.error_context}: {episode.error_message}",
                flush=True,
            )
        if attempted % config.num_episodes == 0:
            print(
                f"Attempted {attempted}: {episode.level_id} {episode.agent_name}",
                flush=True,
            )

    collector = AnalyticsCollector()
    result = run_level_experiment(
        {"RandomAgent": RandomAgent, "GreedyAgent": lambda seed: GreedyAgent()},
        config,
        collector=collector,
        progress=progress,
    )
    output = args.output_dir or (
        Path("experiments/results") / result.metadata["experiment_id"]
    )
    export_evaluation(collector, result.metadata, output)
    print(
        f"Attempted {len(result.episodes)}/{result.metadata['planned_episodes']}; "
        f"interruption={result.metadata['interruption_reason']}; output={output.resolve()}",
        flush=True,
    )


if __name__ == "__main__":
    main()
