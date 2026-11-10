import pytest

from match3_ai.evaluation.difficulty_comparison import compare_agents
from match3_ai.evaluation.difficulty_evaluator import DifficultyEvaluator
from match3_ai.levels.generation import GenerationSpec, LevelGenerator


@pytest.fixture(scope="module")
def report():
    config = LevelGenerator(GenerationSpec(seed=42)).generate(candidate_id=0)

    return DifficultyEvaluator(
        episodes_per_agent=10,
        seed=1000,
    ).evaluate(config)


def test_episode_results_are_preserved(report):
    assert len(report.episodes) == 20

    random_seeds = {
        episode.seed for episode in report.episodes if episode.agent_name == "Random"
    }

    greedy_seeds = {
        episode.seed for episode in report.episodes if episode.agent_name == "Greedy"
    }

    assert random_seeds == greedy_seeds
    assert len(random_seeds) == 10


def test_paired_comparison_is_deterministic(report):
    first = compare_agents(report, seed=42)
    second = compare_agents(report, seed=42)

    assert first == second


def test_paired_difference_matches_win_rates(report):
    comparison = compare_agents(report)

    assert comparison.difference == pytest.approx(
        comparison.win_rate_a - comparison.win_rate_b
    )


def test_paired_interval_contains_valid_values(report):
    comparison = compare_agents(report)

    assert -1 <= comparison.ci_lower <= 1
    assert -1 <= comparison.ci_upper <= 1
    assert comparison.ci_lower <= comparison.ci_upper


@pytest.mark.parametrize(
    "kwargs",
    [
        {"agent_a": "Random", "agent_b": "Random"},
        {"bootstrap_samples": 0},
        {"bootstrap_samples": -1},
        {"seed": -1},
    ],
)
def test_invalid_comparison_parameters(report, kwargs):
    with pytest.raises(ValueError):
        compare_agents(report, **kwargs)
