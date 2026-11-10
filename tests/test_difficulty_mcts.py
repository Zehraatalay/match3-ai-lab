import pytest

from match3_ai.evaluation.difficulty_evaluator import (
    DifficultyEvaluator,
)
from match3_ai.levels.generation import (
    GenerationSpec,
    LevelGenerator,
)


@pytest.fixture
def generated_level():
    generator = LevelGenerator(GenerationSpec(seed=42))
    return generator.generate(candidate_id=0)


def test_mcts_evaluator_smoke(generated_level):
    evaluator = DifficultyEvaluator(
        episodes_per_agent=3,
        seed=1000,
        agents=("MCTS_greedy_outcome",),
        mcts_simulations=10,
    )

    report = evaluator.evaluate(generated_level)

    assert report.level_id == generated_level.level_id
    assert len(report.agent_metrics) == 1
    assert len(report.episodes) == 3

    metrics = report.agent_metrics[0]

    assert metrics.agent_name == "MCTS_greedy_outcome"
    assert metrics.episodes == 3
    assert metrics.errors == 0
    assert metrics.time_limits == 0

    assert all(episode.outcome in {"WON", "LOST"} for episode in report.episodes)


def test_mcts_evaluator_reproducibility(generated_level):
    evaluator = DifficultyEvaluator(
        episodes_per_agent=2,
        seed=1000,
        agents=("MCTS_greedy_outcome",),
        mcts_simulations=10,
    )

    first = evaluator.evaluate(generated_level)
    second = evaluator.evaluate(generated_level)

    assert first.agent_metrics == second.agent_metrics
    assert first.episodes == second.episodes


def test_mcts_and_baselines_share_episode_seeds(generated_level):
    evaluator = DifficultyEvaluator(
        episodes_per_agent=2,
        seed=1000,
        agents=(
            "Random",
            "Greedy",
            "MCTS_greedy_outcome",
        ),
        mcts_simulations=10,
    )

    report = evaluator.evaluate(generated_level)

    seeds_by_agent = {
        agent_name: {
            episode.seed
            for episode in report.episodes
            if episode.agent_name == agent_name
        }
        for agent_name in evaluator.agents
    }

    assert seeds_by_agent["Random"] == {1000, 1001}
    assert seeds_by_agent["Greedy"] == {1000, 1001}
    assert seeds_by_agent["MCTS_greedy_outcome"] == {1000, 1001}


@pytest.mark.parametrize(
    "agents",
    [
        (),
        ("Random", "Random"),
        ("UnknownAgent",),
    ],
)
def test_evaluator_rejects_invalid_agents(agents):
    with pytest.raises(ValueError):
        DifficultyEvaluator(agents=agents)


@pytest.mark.parametrize(
    "simulations",
    [
        0,
        -1,
        True,
        1.5,
    ],
)
def test_evaluator_rejects_invalid_mcts_budget(simulations):
    with pytest.raises(ValueError):
        DifficultyEvaluator(mcts_simulations=simulations)
