import pytest

from match3_ai.evaluation.difficulty_evaluator import (
    DifficultyEvaluator,
    wilson_interval,
)
from match3_ai.levels.generation import GenerationSpec, LevelGenerator


def test_wilson_interval_contains_observed_rate():
    lower, upper = wilson_interval(15, 30)

    assert lower < 0.5 < upper


def test_wilson_interval_bounds():
    for wins in (0, 5, 15, 30):
        lower, upper = wilson_interval(wins, 30)

        assert 0 <= lower <= upper <= 1


def test_difficulty_evaluation_is_deterministic():
    config = LevelGenerator(GenerationSpec(seed=42)).generate(candidate_id=0)

    evaluator = DifficultyEvaluator(
        episodes_per_agent=3,
        seed=100,
    )

    first = evaluator.evaluate(config)
    second = evaluator.evaluate(config)

    assert first == second


def test_difficulty_report_contains_both_agents():
    config = LevelGenerator(GenerationSpec(seed=42)).generate(candidate_id=1)

    evaluator = DifficultyEvaluator(
        episodes_per_agent=3,
        seed=100,
    )

    report = evaluator.evaluate(config)

    assert len(report.agent_metrics) == 2

    names = {metric.agent_name for metric in report.agent_metrics}

    assert names == {"Random", "Greedy"}

    for metric in report.agent_metrics:
        assert metric.episodes == 3
        assert metric.wins + metric.losses + metric.errors + metric.time_limits == 3


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episodes_per_agent": 0},
        {"episodes_per_agent": -1},
        {"episodes_per_agent": True},
        {"seed": -1},
        {"seed": True},
    ],
)
def test_invalid_evaluator_parameters(kwargs):
    with pytest.raises(ValueError):
        DifficultyEvaluator(**kwargs)
