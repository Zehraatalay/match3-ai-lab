from __future__ import annotations

import csv
import hashlib
import json
import logging
import math
import platform
from collections.abc import Callable
from dataclasses import asdict, dataclass
from importlib.metadata import version
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np

from match3_ai.agents.base import Agent
from match3_ai.evaluation.analytics import AnalyticsCollector
from match3_ai.evaluation.runner import EpisodeResult, run_episode
from match3_ai.games.match3.engine import DEFAULT_MAX_CASCADES
from match3_ai.games.match3.environment import (
    DEFAULT_LEVEL_INITIALIZATION_ATTEMPTS,
    Match3Env,
    Match3LevelEnv,
)

logger = logging.getLogger(__name__)


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


@dataclass(frozen=True, slots=True)
class LevelExperimentConfig:
    level_ids: tuple[str, ...] = (
        "first-match",
        "wooden-trouble",
        "double-trouble",
        "frozen-garden",
        "stone-breaker",
        "blocker-mayhem",
    )
    num_episodes: int = 100
    base_seed: int = 52000
    base_agent_seed: int = 93000
    environment_seeds: tuple[int, ...] | None = None
    agent_seeds: tuple[int, ...] | None = None
    max_steps: int = 50
    max_cascades: int = DEFAULT_MAX_CASCADES
    max_initialization_attempts: int = DEFAULT_LEVEL_INITIALIZATION_ATTEMPTS
    stop_after_cascade_errors: int = 2
    max_duration_seconds: float = 300.0

    def __post_init__(self) -> None:
        for name in (
            "num_episodes",
            "max_steps",
            "max_cascades",
            "max_initialization_attempts",
            "stop_after_cascade_errors",
        ):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if (
            not isinstance(self.level_ids, tuple)
            or not self.level_ids
            or any(not isinstance(value, str) or not value for value in self.level_ids)
            or len(set(self.level_ids)) != len(self.level_ids)
        ):
            raise ValueError("level_ids must be a nonempty tuple of unique IDs")
        if (
            isinstance(self.max_duration_seconds, bool)
            or not isinstance(self.max_duration_seconds, (int, float))
            or not math.isfinite(self.max_duration_seconds)
            or self.max_duration_seconds <= 0
        ):
            raise ValueError("max_duration_seconds must be finite and positive")
        for name, base in (
            ("environment_seeds", self.base_seed),
            ("agent_seeds", self.base_agent_seed),
        ):
            if type(base) is not int or base < 0:
                raise ValueError("base seeds must be non-negative integers")
            seeds = getattr(self, name)
            if seeds is None:
                seeds = tuple(range(base, base + self.num_episodes))
                object.__setattr__(self, name, seeds)
            if (
                not isinstance(seeds, tuple)
                or len(seeds) != self.num_episodes
                or any(type(value) is not int or value < 0 for value in seeds)
            ):
                raise ValueError(
                    f"{name} must contain num_episodes non-negative integers"
                )


@dataclass(frozen=True, slots=True)
class LevelExperimentResult:
    episodes: list[EpisodeResult]
    metadata: dict[str, Any]


def _source_fingerprint() -> str:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    for path in sorted(root.rglob("*.py")):
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def run_level_experiment(
    agents: dict[str, Callable[[int], Agent]],
    config: LevelExperimentConfig,
    *,
    collector: AnalyticsCollector | None = None,
    progress: Callable[[EpisodeResult], None] | None = None,
) -> LevelExperimentResult:
    """Evaluate paired seeds, stopping between episodes at explicit safety bounds."""
    if not agents or any(not isinstance(name, str) or not name for name in agents):
        raise ValueError("agents must have nonempty stable names")
    if any(not callable(factory) for factory in agents.values()):
        raise TypeError("agent factories must be callable")
    level_metadata = {}
    for level_id in config.level_ids:
        env = Match3LevelEnv(
            level_id,
            max_cascades=config.max_cascades,
            max_initialization_attempts=config.max_initialization_attempts,
        )
        try:
            _, info = env.reset(seed=config.environment_seeds[0])
            level_metadata[level_id] = {
                key: info[key]
                for key in (
                    "level_config_id",
                    "configured_level_seed",
                    "board_shape",
                    "move_limit",
                )
            }
        finally:
            env.close()
    identity = {
        "schema_version": 1,
        "config": asdict(config),
        "agent_names": list(agents),
        "levels": level_metadata,
        "source_fingerprint": _source_fingerprint(),
    }
    experiment_id = hashlib.sha256(
        json.dumps(identity, sort_keys=True).encode()
    ).hexdigest()[:24]
    records = []
    started = perf_counter()
    cascade_errors = 0
    interrupted = None
    for level_id in config.level_ids:
        for agent_name, factory in agents.items():
            for index, (env_seed, agent_seed) in enumerate(
                zip(config.environment_seeds, config.agent_seeds)
            ):
                if perf_counter() - started >= config.max_duration_seconds:
                    interrupted = "experiment_wall_time_limit"
                    break
                episode_id = hashlib.sha256(
                    f"{experiment_id}:{level_id}:{agent_name}:{index}".encode()
                ).hexdigest()[:24]
                env = None
                try:
                    env = Match3LevelEnv(
                        level_id,
                        max_cascades=config.max_cascades,
                        max_initialization_attempts=config.max_initialization_attempts,
                    )
                    agent = factory(agent_seed)
                except Exception as error:
                    logger.exception("Episode setup failed: %s", episode_id)
                    record = EpisodeResult(
                        seed=env_seed,
                        total_reward=0.0,
                        steps=0,
                        valid_actions=0,
                        invalid_actions=0,
                        total_cleared=0,
                        total_cascades=0,
                        reshuffles=0,
                        completed=False,
                        experiment_id=experiment_id,
                        episode_id=episode_id,
                        agent_name=agent_name,
                        level_id=level_id,
                        level_config_id=level_metadata[level_id]["level_config_id"],
                        agent_seed=agent_seed,
                        outcome="ERROR",
                        termination_reason="setup_exception",
                        cascade_limit=config.max_cascades,
                        error_type=type(error).__name__,
                        error_message=str(error),
                    )
                    if collector is not None:
                        collector.record_episode(record)
                    if env is not None:
                        env.close()
                else:
                    try:
                        record = run_episode(
                            env,
                            agent,
                            seed=env_seed,
                            max_steps=config.max_steps,
                            experiment_id=experiment_id,
                            episode_id=episode_id,
                            agent_name=agent_name,
                            agent_seed=agent_seed,
                            collector=collector,
                        )
                    finally:
                        env.close()
                records.append(record)
                if progress is not None:
                    progress(record)
                if record.error_type == "CascadeLimitExceeded":
                    cascade_errors += 1
                    if cascade_errors >= config.stop_after_cascade_errors:
                        interrupted = "repeated_cascade_limit_errors"
                        break
            if interrupted:
                break
        if interrupted:
            break
    metadata = {
        **identity,
        "experiment_id": experiment_id,
        "python_version": platform.python_version(),
        "numpy_version": version("numpy"),
        "gymnasium_version": version("gymnasium"),
        "reward_mode": "SPARSE",
        "cascade_budget_provisional": True,
        "planned_episodes": len(agents) * len(config.level_ids) * config.num_episodes,
        "attempted_episodes": len(records),
        "interruption_reason": interrupted,
        "elapsed_seconds": perf_counter() - started,
        "inference_timing": "perf_counter around select_action only; hardware dependent",
        "win_rate_denominator": "all attempted episodes, including ERROR and TIME_LIMIT",
        "state_metrics": "last completed transition, including for ERROR; unavailable values are null",
        "cascade_metrics": "completed resolutions plus processed cascades from limit exceptions",
        "unavailable_metrics": [
            "partial-resolution clears, blocker hits, and activations on ERROR"
        ],
        "summary": summarize_level_episodes(records),
    }
    return LevelExperimentResult(records, metadata)


def summarize_level_episodes(episodes: list[EpisodeResult]) -> list[dict[str, Any]]:
    """Descriptive means over available values and Wilson 95% win intervals."""
    groups = sorted({(item.level_id, item.agent_name) for item in episodes})
    summaries = []
    for level_id, agent_name in groups:
        group = [
            item
            for item in episodes
            if (item.level_id, item.agent_name) == (level_id, agent_name)
        ]
        count = len(group)
        wins = sum(item.outcome == "WON" for item in group)
        proportion = wins / count
        z = 1.959963984540054
        denominator = 1 + z * z / count
        center = (proportion + z * z / (2 * count)) / denominator
        radius = (
            z
            * math.sqrt(
                proportion * (1 - proportion) / count + z * z / (4 * count * count)
            )
            / denominator
        )
        summary = {
            "level_id": level_id,
            "agent_name": agent_name,
            "attempted": count,
            "wins": wins,
            "losses": sum(item.outcome == "LOST" for item in group),
            "errors": sum(item.outcome == "ERROR" for item in group),
            "time_limits": sum(item.outcome == "TIME_LIMIT" for item in group),
            "win_rate": proportion,
            "win_rate_denominator": count,
            "win_rate_ci95": [max(0.0, center - radius), min(1.0, center + radius)],
            "maximum_cascade_length": max(item.max_cascades_in_move for item in group),
        }
        for name in (
            "moves_used",
            "objective_completion_ratio",
            "blockers_remaining",
            "blocker_hp_remaining",
            "special_activations",
            "total_cascades",
            "total_reward",
            "inference_time_seconds",
        ):
            values = [
                getattr(item, name) for item in group if getattr(item, name) is not None
            ]
            summary[f"mean_{name}"] = sum(values) / len(values) if values else None
            summary[f"n_{name}"] = len(values)
        summaries.append(summary)
    return summaries
