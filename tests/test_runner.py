import pytest

from match3_ai.agents.random_agent import RandomAgent
from match3_ai.evaluation.runner import run_episode
from match3_ai.games.match3.environment import Match3Env


class InvalidActionAgent:
    """Always selects an out-of-range action."""

    def select_action(self, observation):
        return 999999


class RepeatedInvalidMoveAgent:
    """Selects an action that is currently invalid."""

    def select_action(self, observation):
        action_mask = observation["action_mask"]

        for action_id, is_valid in enumerate(action_mask):
            if not is_valid:
                return action_id

        raise RuntimeError("No invalid action found.")


def test_episode_completes():
    env = Match3Env(move_limit=10)
    agent = RandomAgent(seed=42)

    result = run_episode(env, agent, seed=123)

    assert result.completed is True
    assert result.steps == 10
    assert result.valid_actions == 10
    assert result.invalid_actions == 0
    assert result.total_reward == result.total_cleared


def test_episode_is_reproducible():
    env_a = Match3Env(move_limit=10)
    env_b = Match3Env(move_limit=10)

    agent_a = RandomAgent(seed=42)
    agent_b = RandomAgent(seed=42)

    result_a = run_episode(env_a, agent_a, seed=123)
    result_b = run_episode(env_b, agent_b, seed=123)

    assert result_a == result_b


def test_runner_rejects_out_of_range_action():
    env = Match3Env(move_limit=10)

    with pytest.raises(ValueError, match="out-of-range"):
        run_episode(env, InvalidActionAgent(), seed=123)


def test_runner_stops_after_max_steps():
    env = Match3Env(move_limit=10)

    result = run_episode(
        env,
        RepeatedInvalidMoveAgent(),
        seed=123,
        max_steps=5,
    )

    assert result.steps == 5
    assert result.valid_actions == 0
    assert result.invalid_actions == 5
    assert result.completed is False


def test_runner_rejects_invalid_max_steps():
    env = Match3Env()
    agent = RandomAgent(seed=42)

    with pytest.raises(ValueError, match="max_steps"):
        run_episode(env, agent, max_steps=0)
