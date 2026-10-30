import numpy as np
import pytest

from match3_ai.agents.random_agent import RandomAgent
from match3_ai.games.match3.environment import Match3Env


def test_random_agent_selects_only_valid_actions():
    env = Match3Env()
    observation, _ = env.reset(seed=42)

    agent = RandomAgent(seed=123)

    for _ in range(100):
        action = agent.select_action(observation)

        assert observation["action_mask"][action] == 1


def test_random_agent_is_reproducible():
    env = Match3Env()
    observation, _ = env.reset(seed=42)

    agent_a = RandomAgent(seed=123)
    agent_b = RandomAgent(seed=123)

    actions_a = [agent_a.select_action(observation) for _ in range(20)]

    actions_b = [agent_b.select_action(observation) for _ in range(20)]

    assert actions_a == actions_b


def test_random_agent_raises_when_no_valid_actions():
    agent = RandomAgent(seed=42)

    observation = {
        "board": np.ones((3, 3), dtype=np.uint8),
        "action_mask": np.zeros(12, dtype=np.int8),
        "moves_remaining": 10,
    }

    with pytest.raises(RuntimeError, match="no valid actions"):
        agent.select_action(observation)


def test_random_agent_does_not_modify_observation():
    env = Match3Env()
    observation, _ = env.reset(seed=42)

    original_board = observation["board"].copy()
    original_mask = observation["action_mask"].copy()

    agent = RandomAgent(seed=123)
    agent.select_action(observation)

    np.testing.assert_array_equal(
        observation["board"],
        original_board,
    )

    np.testing.assert_array_equal(
        observation["action_mask"],
        original_mask,
    )
