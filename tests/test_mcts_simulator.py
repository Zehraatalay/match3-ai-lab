import numpy as np

from match3_ai.agents.mcts_simulator import MCTSSimulator
from match3_ai.games.match3.environment import Match3LevelEnv


def test_simulation_does_not_modify_original_environment():
    env = Match3LevelEnv(level_id="double-trouble")
    observation, _ = env.reset(seed=42)

    simulator = MCTSSimulator(env)
    clone = simulator.clone()

    original_tiles = env.session.state.tiles.copy()

    legal_actions = simulator.legal_actions(clone)

    assert legal_actions
    assert set(legal_actions) == set(np.flatnonzero(observation["action_mask"]))

    reward, done = simulator.step(clone, legal_actions[0])

    assert isinstance(reward, float)
    assert isinstance(done, bool)
    assert np.array_equal(env.session.state.tiles, original_tiles)

    env.close()
    clone.close()


import copy


def test_mcts_search_does_not_modify_real_environment():
    env = Match3LevelEnv(level_id="wooden-trouble")

    try:
        observation, _ = env.reset(seed=42)

        original_observation = copy.deepcopy(observation)
        original_moves = env.session.moves_remaining
        original_objectives = copy.deepcopy(env._objective_info())

        simulator = MCTSSimulator(env)

        from match3_ai.agents.stochastic_mcts_agent import (
            StochasticMCTSAgent,
        )

        agent = StochasticMCTSAgent(
            simulator=simulator,
            simulations=30,
            rollout_policy="greedy",
            value_mode="outcome",
            seed=42,
        )

        agent.select_action(observation)

        current_observation = env._observation()

        import numpy as np

        np.testing.assert_array_equal(
            current_observation["action_mask"],
            original_observation["action_mask"],
        )

        assert env.session.moves_remaining == original_moves
        assert env._objective_info() == original_objectives

    finally:
        env.close()
