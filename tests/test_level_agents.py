from copy import deepcopy

import numpy as np
import pytest

from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.games.match3.specials import SpecialType

LEVEL_IDS = (
    "first-match",
    "wooden-trouble",
    "double-trouble",
    "frozen-garden",
    "stone-breaker",
    "blocker-mayhem",
)


@pytest.mark.parametrize("level_id", LEVEL_IDS)
@pytest.mark.parametrize("agent_type", [RandomAgent, GreedyAgent])
def test_agents_complete_levels_with_only_legal_actions(level_id, agent_type):
    env = Match3LevelEnv(level_id)
    agent = RandomAgent(seed=700) if agent_type is RandomAgent else GreedyAgent()
    observation, _ = env.reset(seed=500)
    for _ in range(env.config.move_limit):
        action = agent.select_action(observation)
        assert observation["action_mask"][action] == 1
        row, col, direction = env.decode_action(action)
        rows, cols = observation["board_shape"]
        second = (row, col + 1) if direction == 0 else (row + 1, col)
        assert 0 <= row < rows and 0 <= col < cols
        assert 0 <= second[0] < rows and 0 <= second[1] < cols
        assert observation["cell_mask"][row, col]
        assert observation["cell_mask"][second]
        observation, _, terminated, truncated, info = env.step(action)
        assert info["valid_action"]
        assert env.observation_space.contains(observation)
        assert not truncated
        if terminated:
            break
        assert observation["action_mask"].any()
    else:
        pytest.fail("episode exceeded its level move budget")
    env.close()


def test_random_level_actions_reproduce_local_generator_choices():
    env = Match3LevelEnv("blocker-mayhem")
    observation, _ = env.reset(seed=101)
    agent = RandomAgent(seed=303)
    other = RandomAgent(seed=303)
    reference_rng = np.random.default_rng(303)
    valid = np.flatnonzero(observation["action_mask"])
    expected = [int(reference_rng.choice(valid)) for _ in range(50)]
    assert [agent.select_action(observation) for _ in range(50)] == expected
    assert [other.select_action(observation) for _ in range(50)] == expected


@pytest.mark.parametrize("agent", [RandomAgent(seed=1), GreedyAgent()])
def test_level_agents_explicitly_reject_empty_mask(agent):
    env = Match3LevelEnv()
    observation, _ = env.reset(seed=11)
    observation["action_mask"][:] = 0
    with pytest.raises(RuntimeError, match="[Nn]o valid actions"):
        agent.select_action(observation)


def test_greedy_scores_canonical_swap_and_breaks_ties_by_lowest_id():
    observation = {
        "tiles": np.zeros((9, 8), dtype=np.uint8),
        "cell_mask": np.zeros((9, 8), dtype=np.int8),
        "board_shape": np.array([3, 3], dtype=np.int16),
        "action_mask": np.zeros(144, dtype=np.int8),
    }
    observation["tiles"][:3, :3] = [[1, 2, 1], [3, 1, 2], [2, 3, 1]]
    observation["cell_mask"][:3, :3] = 1
    observation["action_mask"][[0, 3]] = 1
    agent = GreedyAgent()
    assert agent.select_action(observation) == 3

    observation["tiles"][:3, :3] = [[1, 2, 3], [2, 3, 1], [3, 1, 2]]
    assert agent.select_action(observation) == 0
    assert agent.select_action(observation) == 0


def test_greedy_ignores_padding_and_masked_tiles_without_mutating_any_state():
    env = Match3LevelEnv("wooden-trouble")
    observation, _ = env.reset(seed=88)
    before = deepcopy(observation)
    board_before = env.session.board.copy()
    rng_before = deepcopy(env.session.engine.rng.bit_generator.state)
    agent = GreedyAgent()
    selected = agent.select_action(observation)
    assert selected == agent.select_action(observation)
    for key in observation:
        np.testing.assert_array_equal(observation[key], before[key])
    for field in ("tiles", "specials", "blockers", "blocker_hp", "cell_mask"):
        np.testing.assert_array_equal(
            getattr(env.session.board, field), getattr(board_before, field)
        )
    assert env.session.engine.rng.bit_generator.state == rng_before

    observation["tiles"][observation["cell_mask"] == 0] = 1
    assert agent.select_action(observation) == selected


@pytest.mark.parametrize("agent", [RandomAgent(seed=1), GreedyAgent()])
def test_agents_accept_ice_special_combination_and_exclude_fixed_blockers(agent):
    env = Match3LevelEnv()
    env.reset(seed=21)
    board = env.session.board.copy()
    board.specials[2, 2] = SpecialType.ROCKET_HORIZONTAL
    board.specials[2, 3] = SpecialType.BOMB
    board.specials[3, 2] = SpecialType.PROPELLER
    board.blockers[2, 2] = BlockerType.ICE
    board.blocker_hp[2, 2] = 2
    for position, blocker_type in [
        ((0, 0), BlockerType.CRATE),
        ((0, 1), BlockerType.STONE),
    ]:
        board.tiles[position] = 0
        board.specials[position] = 0
        board.blockers[position] = blocker_type
        board.blocker_hp[position] = 2
    env.session.state = BoardState(
        board.tiles, board.cell_mask, board.specials, board.blockers, board.blocker_hp
    )
    observation = env._observation()
    combo = env.encode_action(2, 2, 0)
    assert observation["action_mask"][combo] == 1
    assert not observation["action_mask"][env.encode_action(0, 0, 0)]
    assert not observation["action_mask"][env.encode_action(0, 1, 1)]
    observation["action_mask"][:] = 0
    observation["action_mask"][combo] = 1
    assert agent.select_action(observation) == combo
    _, _, _, _, info = env.step(combo)
    assert info["valid_action"]
    assert info["special_activations"]["count"] > 0
    assert info["special_activations"]["by_type"]["PROPELLER"] > 0
