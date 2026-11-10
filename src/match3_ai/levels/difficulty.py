from dataclasses import dataclass
from enum import Enum

from match3_ai.evaluation.difficulty_evaluator import (
    AgentDifficultyMetrics,
    DifficultyReport,
)


class Difficulty(str, Enum):
    EASY = "EASY"
    MEDIUM = "MEDIUM"
    HARD = "HARD"
    VERY_HARD = "VERY_HARD"


class DifficultyMatch(str, Enum):
    MATCH = "MATCH"
    UNCERTAIN = "UNCERTAIN"
    REJECT = "REJECT"


DIFFICULTY_RANGES = {
    Difficulty.EASY: (0.75, 1.0),
    Difficulty.MEDIUM: (0.40, 0.75),
    Difficulty.HARD: (0.10, 0.40),
    Difficulty.VERY_HARD: (0.0, 0.10),
}


@dataclass(frozen=True, slots=True)
class DifficultyClassification:
    level_id: str
    target: Difficulty
    status: DifficultyMatch

    win_rate: float
    ci_lower: float
    ci_upper: float

    episodes: int


def classify_difficulty(
    report: DifficultyReport,
    target: Difficulty,
    *,
    agent_name: str = "MCTS_greedy_outcome",
) -> DifficultyClassification:
    if not isinstance(report, DifficultyReport):
        raise TypeError("report must be a DifficultyReport")

    if not isinstance(target, Difficulty):
        raise TypeError("target must be a Difficulty")

    metrics: AgentDifficultyMetrics | None = next(
        (metric for metric in report.agent_metrics if metric.agent_name == agent_name),
        None,
    )

    if metrics is None:
        raise ValueError(f"Agent {agent_name!r} not found in difficulty report")

    if metrics.errors or metrics.time_limits:
        raise ValueError("Cannot classify difficulty from incomplete evaluation")

    lower, upper = DIFFICULTY_RANGES[target]

    if lower <= metrics.ci_lower and metrics.ci_upper <= upper:
        status = DifficultyMatch.MATCH

    elif lower <= metrics.win_rate <= upper:
        status = DifficultyMatch.UNCERTAIN

    else:
        status = DifficultyMatch.REJECT

    return DifficultyClassification(
        level_id=report.level_id,
        target=target,
        status=status,
        win_rate=metrics.win_rate,
        ci_lower=metrics.ci_lower,
        ci_upper=metrics.ci_upper,
        episodes=metrics.episodes,
    )
