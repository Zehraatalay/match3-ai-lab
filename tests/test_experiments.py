import csv

import pytest

from match3_ai.agents.random_agent import RandomAgent
from match3_ai.evaluation.experiments import (
    ExperimentConfig,
    run_experiment,
    save_experiment_csv,
)
from match3_ai.games.match3.environment import Match3Env


def make_env():
    return Match3Env(move_limit=5)


def make_agent(seed):
    return RandomAgent(seed=seed)


def test_experiment_runs_multiple_episodes():
    config = ExperimentConfig(
        num_episodes=10,
        base_seed=42,
    )

    result = run_experiment(make_env, make_agent, config)

    assert len(result.episodes) == 10
    assert result.summary.num_episodes == 10
    assert all(episode.completed for episode in result.episodes)


def test_experiment_uses_expected_seeds():
    config = ExperimentConfig(
        num_episodes=3,
        base_seed=100,
    )

    result = run_experiment(make_env, make_agent, config)

    assert [episode.seed for episode in result.episodes] == [
        100,
        101,
        102,
    ]


def test_experiment_is_reproducible():
    config = ExperimentConfig(
        num_episodes=10,
        base_seed=42,
    )

    result_a = run_experiment(make_env, make_agent, config)
    result_b = run_experiment(make_env, make_agent, config)

    assert result_a == result_b


def test_experiment_summary():
    config = ExperimentConfig(
        num_episodes=10,
        base_seed=42,
    )

    result = run_experiment(make_env, make_agent, config)

    rewards = [episode.total_reward for episode in result.episodes]

    assert result.summary.min_reward == min(rewards)
    assert result.summary.max_reward == max(rewards)
    assert result.summary.min_reward <= result.summary.mean_reward
    assert result.summary.mean_reward <= result.summary.max_reward
    assert result.summary.completion_rate == 1.0


@pytest.mark.parametrize("num_episodes", [0, -1])
def test_experiment_rejects_invalid_episode_count(num_episodes):
    config = ExperimentConfig(num_episodes=num_episodes)

    with pytest.raises(ValueError, match="num_episodes"):
        run_experiment(make_env, make_agent, config)


def test_experiment_saves_csv(tmp_path):
    config = ExperimentConfig(
        num_episodes=3,
        base_seed=42,
    )

    result = run_experiment(make_env, make_agent, config)

    output_path = tmp_path / "results.csv"

    save_experiment_csv(result, output_path)

    assert output_path.exists()

    with output_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))

    assert len(rows) == 3
    assert "total_reward" in rows[0]
    assert "seed" in rows[0]
