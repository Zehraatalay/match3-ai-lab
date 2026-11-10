import pytest

from match3_ai.evaluation.difficulty_evaluator import (
    AgentDifficultyMetrics,
    DifficultyReport,
)
from match3_ai.levels.difficulty import (
    Difficulty,
    DifficultyMatch,
    classify_difficulty,
)


def make_report(
    *,
    win_rate: float,
    ci_lower: float,
    ci_upper: float,
    errors: int = 0,
    time_limits: int = 0,
) -> DifficultyReport:
    metrics = AgentDifficultyMetrics(
        agent_name="MCTS_greedy_outcome",
        episodes=100,
        wins=round(win_rate * 100),
        losses=100 - round(win_rate * 100),
        errors=errors,
        time_limits=time_limits,
        win_rate=win_rate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        mean_moves_used=None,
        mean_winning_moves=None,
        mean_objective_completion=None,
    )

    return DifficultyReport(
        level_id="test-level",
        agent_metrics=(metrics,),
    )


@pytest.mark.parametrize(
    ("target", "win_rate", "ci_lower", "ci_upper"),
    [
        (Difficulty.EASY, 0.90, 0.82, 0.95),
        (Difficulty.MEDIUM, 0.55, 0.45, 0.65),
        (Difficulty.HARD, 0.25, 0.17, 0.35),
        (Difficulty.VERY_HARD, 0.03, 0.01, 0.08),
    ],
)
def test_classification_match(
    target,
    win_rate,
    ci_lower,
    ci_upper,
):
    report = make_report(
        win_rate=win_rate,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
    )

    result = classify_difficulty(report, target)

    assert result.status == DifficultyMatch.MATCH


def test_classification_uncertain():
    report = make_report(
        win_rate=0.30,
        ci_lower=0.15,
        ci_upper=0.52,
    )

    result = classify_difficulty(report, Difficulty.HARD)

    assert result.status == DifficultyMatch.UNCERTAIN


def test_classification_reject():
    report = make_report(
        win_rate=0.80,
        ci_lower=0.70,
        ci_upper=0.87,
    )

    result = classify_difficulty(report, Difficulty.HARD)

    assert result.status == DifficultyMatch.REJECT


def test_missing_agent():
    report = DifficultyReport(
        level_id="test-level",
        agent_metrics=(),
    )

    with pytest.raises(ValueError, match="not found"):
        classify_difficulty(report, Difficulty.HARD)


def test_incomplete_evaluation():
    report = make_report(
        win_rate=0.30,
        ci_lower=0.15,
        ci_upper=0.52,
        errors=1,
    )

    with pytest.raises(ValueError, match="incomplete"):
        classify_difficulty(report, Difficulty.HARD)
