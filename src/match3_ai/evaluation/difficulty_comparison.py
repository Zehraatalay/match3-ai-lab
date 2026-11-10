from dataclasses import dataclass

import numpy as np

from match3_ai.evaluation.difficulty_evaluator import DifficultyReport


@dataclass(frozen=True, slots=True)
class PairedComparison:
    agent_a: str
    agent_b: str
    paired_episodes: int
    win_rate_a: float
    win_rate_b: float
    difference: float
    ci_lower: float
    ci_upper: float


def compare_agents(
    report: DifficultyReport,
    *,
    agent_a: str = "Random",
    agent_b: str = "Greedy",
    bootstrap_samples: int = 2000,
    seed: int = 42,
) -> PairedComparison:
    if agent_a == agent_b:
        raise ValueError("agent names must differ")
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")

    outcomes = {}

    for episode in report.episodes:
        if episode.agent_name not in (agent_a, agent_b):
            continue

        key = (episode.agent_name, episode.seed)

        if key in outcomes:
            raise ValueError(f"duplicate episode: {key}")

        if episode.outcome not in ("WON", "LOST"):
            raise ValueError(f"cannot compare non-terminal outcome: {episode.outcome}")

        outcomes[key] = int(episode.outcome == "WON")

    seeds_a = {episode_seed for name, episode_seed in outcomes if name == agent_a}

    seeds_b = {episode_seed for name, episode_seed in outcomes if name == agent_b}

    if not seeds_a or seeds_a != seeds_b:
        raise ValueError("agents must have identical episode seed sets")

    seeds = sorted(seeds_a)

    wins_a = np.array(
        [outcomes[(agent_a, episode_seed)] for episode_seed in seeds],
        dtype=np.float64,
    )

    wins_b = np.array(
        [outcomes[(agent_b, episode_seed)] for episode_seed in seeds],
        dtype=np.float64,
    )

    paired_differences = wins_a - wins_b

    rng = np.random.default_rng(seed)

    bootstrap_indices = rng.integers(
        0,
        len(seeds),
        size=(bootstrap_samples, len(seeds)),
    )

    bootstrap_differences = paired_differences[bootstrap_indices].mean(axis=1)

    lower, upper = np.quantile(
        bootstrap_differences,
        [0.025, 0.975],
    )

    return PairedComparison(
        agent_a=agent_a,
        agent_b=agent_b,
        paired_episodes=len(seeds),
        win_rate_a=float(wins_a.mean()),
        win_rate_b=float(wins_b.mean()),
        difference=float(paired_differences.mean()),
        ci_lower=float(lower),
        ci_upper=float(upper),
    )
