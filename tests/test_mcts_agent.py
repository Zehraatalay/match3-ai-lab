import copy

import numpy as np
import pytest

from match3_ai.agents.mcts_agent import MonteCarloAgent
from match3_ai.agents.mcts_simulator import MCTSSimulator
from match3_ai.games.match3.environment import Match3LevelEnv


def test_monte_carlo_agent_selects_legal_action():
    env = Match3LevelEnv(level_id="first-match")
    observation, _ = env.reset(seed=42)

    agent = MonteCarloAgent(
        simulator=MCTSSimulator(env),
        simulations_per_action=2,
        seed=42,
    )

    action = agent.select_action(observation)

    assert observation["action_mask"][action] == 1

    env.close()


def test_monte_carlo_agent_preserves_original_environment():
    env = Match3LevelEnv(level_id="first-match")
    observation, _ = env.reset(seed=42)

    original_tiles = env.session.state.tiles.copy()
    original_rng = copy.deepcopy(env.session.engine.rng.bit_generator.state)

    agent = MonteCarloAgent(
        simulator=MCTSSimulator(env),
        simulations_per_action=2,
        seed=42,
    )

    agent.select_action(observation)

    assert np.array_equal(env.session.state.tiles, original_tiles)
    assert env.session.engine.rng.bit_generator.state == original_rng

    env.close()


def test_monte_carlo_agent_rejects_invalid_budget():
    env = Match3LevelEnv(level_id="first-match")
    env.reset(seed=42)

    with pytest.raises(ValueError):
        MonteCarloAgent(
            simulator=MCTSSimulator(env),
            simulations_per_action=0,
        )

    env.close()
