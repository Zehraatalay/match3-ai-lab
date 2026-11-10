import copy

import numpy as np
import pytest

from match3_ai.agents.mcts_simulator import MCTSSimulator
from match3_ai.agents.stochastic_mcts_agent import (
    MCTSNode,
    StochasticMCTSAgent,
)
from match3_ai.games.match3.environment import Match3LevelEnv


def test_uct_prefers_unvisited_nodes():
    node = MCTSNode()

    assert node.uct(10, 1.414) == float("inf")


def test_mcts_selects_legal_action():
    env = Match3LevelEnv(level_id="first-match")
    observation, _ = env.reset(seed=42)

    agent = StochasticMCTSAgent(
        simulator=MCTSSimulator(env),
        simulations=20,
        seed=42,
    )

    action = agent.select_action(observation)

    assert observation["action_mask"][action] == 1
    env.close()


def test_mcts_does_not_modify_original_environment():
    env = Match3LevelEnv(level_id="first-match")
    observation, _ = env.reset(seed=42)

    original_tiles = env.session.state.tiles.copy()
    original_rng = copy.deepcopy(env.session.engine.rng.bit_generator.state)
    original_moves = env.session.moves_remaining

    agent = StochasticMCTSAgent(
        simulator=MCTSSimulator(env),
        simulations=10,
        seed=42,
    )

    agent.select_action(observation)

    assert np.array_equal(env.session.state.tiles, original_tiles)
    assert env.session.engine.rng.bit_generator.state == original_rng
    assert env.session.moves_remaining == original_moves

    env.close()


@pytest.mark.parametrize("simulations", [0, -1])
def test_mcts_rejects_invalid_simulation_budget(simulations):
    env = Match3LevelEnv(level_id="first-match")
    env.reset(seed=42)

    with pytest.raises(ValueError):
        StochasticMCTSAgent(
            simulator=MCTSSimulator(env),
            simulations=simulations,
        )

    env.close()


def test_mcts_reproducibility():
    env = Match3LevelEnv(level_id="first-match")
    observation, _ = env.reset(seed=42)

    simulator = MCTSSimulator(env)

    agent_a = StochasticMCTSAgent(
        simulator=simulator,
        simulations=50,
        seed=123,
    )

    agent_b = StochasticMCTSAgent(
        simulator=simulator,
        simulations=50,
        seed=123,
    )

    action_a = agent_a.select_action(observation)
    action_b = agent_b.select_action(observation)

    assert action_a == action_b

    env.close()


def test_mcts_simulator_samples_stochastic_outcomes():
    env = Match3LevelEnv(level_id="first-match")
    observation, _ = env.reset(seed=42)

    simulator = MCTSSimulator(env)

    action = int(np.flatnonzero(observation["action_mask"])[0])

    outcomes = set()

    for seed in range(20):
        clone = simulator.clone_with_seed(seed)

        clone.step(action)

        outcomes.add(clone.session.state.tiles.tobytes())

        clone.close()

    assert len(outcomes) > 1

    env.close()


def test_uct_balances_exploration_and_exploitation():
    frequently_visited = MCTSNode(
        visits=50,
        value_sum=30.0,
    )

    rarely_visited = MCTSNode(
        visits=5,
        value_sum=2.0,
    )

    parent_visits = 100

    greedy_frequent = frequently_visited.uct(parent_visits, exploration_constant=0.0)
    greedy_rare = rarely_visited.uct(parent_visits, exploration_constant=0.0)

    assert greedy_frequent > greedy_rare

    exploratory_frequent = frequently_visited.uct(
        parent_visits, exploration_constant=2.0
    )
    exploratory_rare = rarely_visited.uct(parent_visits, exploration_constant=2.0)

    assert exploratory_rare > exploratory_frequent


@pytest.mark.parametrize("rollout_policy", ["random", "greedy"])
def test_mcts_rollout_policies_select_legal_actions(rollout_policy):
    env = Match3LevelEnv(level_id="first-match")
    observation, _ = env.reset(seed=42)

    agent = StochasticMCTSAgent(
        simulator=MCTSSimulator(env),
        simulations=20,
        rollout_policy=rollout_policy,
        seed=42,
    )

    action = agent.select_action(observation)

    assert observation["action_mask"][action] == 1

    env.close()


def test_mcts_rejects_invalid_rollout_policy():
    env = Match3LevelEnv(level_id="first-match")
    env.reset(seed=42)

    with pytest.raises(ValueError, match="rollout_policy"):
        StochasticMCTSAgent(
            simulator=MCTSSimulator(env),
            rollout_policy="invalid",
        )

    env.close()


import pytest


@pytest.mark.parametrize("value_mode", ["terminal", "outcome"])
def test_mcts_value_modes(value_mode):
    env = Match3LevelEnv(level_id="double-trouble")

    try:
        observation, _ = env.reset(seed=42)

        agent = StochasticMCTSAgent(
            simulator=MCTSSimulator(env),
            simulations=10,
            rollout_policy="random",
            value_mode=value_mode,
            seed=42,
        )

        action = agent.select_action(observation)

        assert observation["action_mask"][action]

        stats = agent.last_search_stats

        assert stats is not None
        assert stats.root_visits == 10

    finally:
        env.close()


def test_mcts_rejects_invalid_value_mode():
    env = Match3LevelEnv(level_id="double-trouble")

    try:
        env.reset(seed=42)

        with pytest.raises(ValueError, match="value_mode"):
            StochasticMCTSAgent(
                simulator=MCTSSimulator(env),
                value_mode="invalid",
            )

    finally:
        env.close()
