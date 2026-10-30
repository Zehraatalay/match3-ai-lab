from __future__ import annotations

import csv
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from match3_ai.agents.base import Agent
from match3_ai.evaluation.runner import EpisodeResult, run_episode
from match3_ai.games.match3.environment import Match3Env


@dataclass(frozen=True, slots=True)
class ExperimentConfig:
    num_episodes: int = 1000
    base_seed: int = 42
    max_steps: int = 1000


@dataclass(frozen=True, slots=True)
class ExperimentSummary:
    num_episodes: int
    mean_reward: float
    std_reward: float
    min_reward: float
    max_reward: float
    completion_rate: float
    mean_cascades: float
    mean_reshuffles: float


@dataclass(frozen=True, slots=True)
class ExperimentResult:
    episodes: list[EpisodeResult]
    summary: ExperimentSummary


def run_experiment(
    env_factory: Callable[[], Match3Env],
    agent_factory: Callable[[int], Agent],
    config: ExperimentConfig,
) -> ExperimentResult:
    """evaluate an agent over multiple reproducible episodes."""

    if config.num_episodes <= 0:
        raise ValueError("num_episodes must be positive.")

    if config.max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    results: list[EpisodeResult] = []

    for episode_idx in range(config.num_episodes):
        episode_seed = config.base_seed + episode_idx

        env = env_factory()
        agent = agent_factory(episode_seed)

        try:
            result = run_episode(
                env=env,
                agent=agent,
                seed=episode_seed,
                max_steps=config.max_steps,
            )
            results.append(result)
        finally:
            env.close()

    rewards = np.array(
        [result.total_reward for result in results],
        dtype=np.float64,
    )

    summary = ExperimentSummary(
        num_episodes=len(results),
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        min_reward=float(np.min(rewards)),
        max_reward=float(np.max(rewards)),
        completion_rate=float(np.mean([result.completed for result in results])),
        mean_cascades=float(np.mean([result.total_cascades for result in results])),
        mean_reshuffles=float(np.mean([result.reshuffles for result in results])),
    )

    return ExperimentResult(
        episodes=results,
        summary=summary,
    )


def save_experiment_csv(
    experiment: ExperimentResult,
    output_path: str | Path,
) -> None:
    """save per-episode metrics to CSV."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(EpisodeResult.__dataclass_fields__),
        )

        writer.writeheader()

        for episode in experiment.episodes:
            writer.writerow(asdict(episode))
