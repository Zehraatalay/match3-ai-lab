from __future__ import annotations

from dataclasses import replace

import gymnasium as gym
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env

from match3_ai.games.match3.environment import Match3Env, Match3LevelEnv
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.levels import LevelStatus


def mask_from_moves(env: Match3LevelEnv) -> np.ndarray:
    assert env.session is not None
    mask = np.zeros(env.ACTION_COUNT, dtype=np.int8)
    for (row_a, col_a), (row_b, col_b) in find_valid_moves(env.session.board):
        if row_a == row_b:
            mask[env.encode_action(row_a, min(col_a, col_b), 0)] = 1
        else:
            mask[env.encode_action(min(row_a, row_b), col_a, 1)] = 1
    return mask


def first_action(observation: dict) -> int:
    return int(np.flatnonzero(observation["action_mask"])[0])


def test_packaged_levels_have_one_fixed_observation_space():
    env = Match3LevelEnv()
    assert len(env.level_ids) == 9
    assert env.action_space == gym.spaces.Discrete(144)

    for level_id in env.level_ids:
        observation, info = env.reset(options={"level_id": level_id})
        assert env.observation_space.contains(observation)
        assert observation["tiles"].shape == (9, 8)
        assert observation["action_mask"].shape == (144,)
        assert tuple(observation["board_shape"]) == info["board_shape"]
        rows, cols = info["board_shape"]
        assert not observation["cell_mask"][rows:, :].any()
        assert not observation["cell_mask"][:, cols:].any()
        assert observation["action_mask"].sum() != 0


def test_padding_is_neutral_and_irregular_holes_remain_masked():
    env = Match3LevelEnv("blocker-mayhem")
    observation, _ = env.reset(seed=22)
    board = env.session.board
    rows, cols = board.shape
    assert np.array_equal(observation["cell_mask"][:rows, :cols], board.cell_mask)
    assert not observation["cell_mask"][rows:, :].any()
    assert not observation["cell_mask"][:, cols:].any()
    assert np.all(observation["tiles"][rows:, :] == 0)
    assert np.all(observation["blocker_hp"][:, cols:] == 0)


@pytest.mark.parametrize("direction", [0, 1])
def test_canonical_action_encode_decode_round_trip(direction):
    seen = set()
    for row in range(9):
        for col in range(8):
            action_id = Match3LevelEnv.encode_action(row, col, direction)
            assert Match3LevelEnv.decode_action(action_id) == (row, col, direction)
            seen.add(action_id)
    assert len(seen) == 72


def test_action_mask_is_exactly_derived_from_existing_move_detection():
    env = Match3LevelEnv("blocker-mayhem")
    observation, _ = env.reset(seed=1)
    assert np.array_equal(observation["action_mask"], mask_from_moves(env))
    for action in np.flatnonzero(observation["action_mask"]):
        row, col, direction = env.decode_action(int(action))
        second = (row, col + 1) if direction == 0 else (row + 1, col)
        assert ((row, col), second) in find_valid_moves(env.session.board)


def test_objective_arrays_expose_type_color_raw_progress_and_target():
    env = Match3LevelEnv("first-match")
    observation, _ = env.reset(seed=4)
    active = observation["objective_mask"].astype(bool)
    assert active.sum() == len(env.config.objectives)
    assert np.all(observation["objective_progress"][active] == 0)
    assert np.all(observation["objective_target"][active] > 0)
    assert np.all(observation["objective_type"][active] > 0)

    collect = next(
        index
        for index, objective in enumerate(env.config.objectives)
        if objective.objective_type.value == "COLLECT_TILES"
    )
    assert observation["objective_color"][collect] == 0


def test_masked_out_action_is_noop_and_out_of_range_raises():
    env = Match3LevelEnv()
    observation, _ = env.reset(seed=9)
    invalid = int(np.flatnonzero(observation["action_mask"] == 0)[0])
    before = env.session.board.copy()
    moves = env.session.moves_remaining
    progress = env.session.objective_progress
    next_observation, reward, terminated, truncated, info = env.step(invalid)
    assert not info["valid_action"]
    assert reward == 0.0
    assert not terminated and not truncated
    assert env.session.moves_remaining == moves
    assert env.session.objective_progress == progress
    assert np.array_equal(env.session.board.tiles, before.tiles)
    assert np.array_equal(next_observation["action_mask"], observation["action_mask"])
    with pytest.raises(ValueError, match="outside the action space"):
        env.step(144)


def test_valid_action_consumes_one_level_move_and_reports_analytics():
    env = Match3LevelEnv()
    observation, _ = env.reset(seed=15)
    _, reward, terminated, truncated, info = env.step(first_action(observation))
    assert reward == 0.0
    assert not terminated and not truncated
    assert info["valid_action"]
    assert info["moves_remaining"] == env.config.move_limit - 1
    assert "cleared_by_color" in info
    assert "blocker_damage" in info
    assert "special_activations" in info
    assert "reshuffled" in info


def test_win_and_loss_are_terminated_not_truncated():
    winning = Match3LevelEnv("first-match")
    observation, _ = winning.reset(seed=3)
    collect_index = next(
        i
        for i, item in enumerate(winning.session.objective_progress)
        if item.objective_type.value == "COLLECT_TILES"
    )
    progress = list(winning.session.objective_progress)
    current = progress[collect_index]
    progress[collect_index] = replace(
        current,
        progress=current.target - 1,
        raw_progress=current.target - 1,
    )
    winning.session.objective_progress = tuple(progress)
    _, reward, terminated, truncated, _ = winning.step(first_action(observation))
    assert winning.session.status == LevelStatus.WON
    assert reward == 1.0 and terminated and not truncated

    losing = Match3LevelEnv("first-match")
    observation, _ = losing.reset(seed=3)
    losing.session.moves_remaining = 1
    _, reward, terminated, truncated, _ = losing.step(first_action(observation))
    assert losing.session.status == LevelStatus.LOST
    assert reward == -1.0 and terminated and not truncated


def test_zero_post_step_action_mask_is_a_terminal_failure(monkeypatch):
    env = Match3LevelEnv()
    observation, _ = env.reset(seed=15)
    real_mask = env._action_mask
    calls = 0

    def mask_after_step():
        nonlocal calls
        calls += 1
        return real_mask() if calls == 1 else np.zeros(env.ACTION_COUNT, dtype=np.int8)

    monkeypatch.setattr(env, "_action_mask", mask_after_step)
    _, reward, terminated, truncated, info = env.step(first_action(observation))
    assert terminated and not truncated
    assert reward == -1.0
    assert info["termination_reason"] == "no_legal_actions"
    assert info["level_status"] == LevelStatus.IN_PROGRESS.value


def test_explicit_seed_reproduces_board_and_transitions():
    first = Match3LevelEnv("blocker-mayhem")
    second = Match3LevelEnv("blocker-mayhem")
    obs_a, info_a = first.reset(seed=123)
    obs_b, _ = second.reset(seed=123)
    assert info_a["effective_episode_seed"] == 123
    assert info_a["board_seed"] != info_a["engine_seed"]
    for key in obs_a:
        assert np.array_equal(obs_a[key], obs_b[key])

    action = first_action(obs_a)
    out_a = first.step(action)
    out_b = second.step(action)
    for key in out_a[0]:
        assert np.array_equal(out_a[0][key], out_b[0][key])
    assert out_a[1:] == out_b[1:]


def test_different_episode_seeds_can_change_valid_initial_boards():
    env = Match3LevelEnv("first-match")
    first, _ = env.reset(seed=111)
    second, _ = env.reset(seed=112)
    assert not np.array_equal(first["tiles"], second["tiles"])
    assert np.any(first["action_mask"])
    assert np.any(second["action_mask"])


def test_level_switching_creates_fresh_session_and_default_seed_is_reported():
    env = Match3LevelEnv()
    initial, info = env.reset()
    assert info["configured_level_seed"] == env.config.seed
    assert info["effective_episode_seed"] == env.config.seed
    first_session = env.session
    other, other_info = env.reset(options={"level_id": "wooden-trouble"})
    assert env.session is not first_session
    assert other_info["level_id"] == "wooden-trouble"
    assert other["board_shape"].tolist() == [7, 7]
    assert env.session.moves_remaining == env.config.move_limit
    assert env.session.objective_progress[0].raw_progress == 0
    assert initial["board_shape"].tolist() == [6, 6]


def test_gymnasium_checker_and_legacy_environment_are_compatible():
    env = Match3LevelEnv()
    check_env(env, skip_render_check=True)
    legacy = Match3Env()
    observation, info = legacy.reset(seed=42)
    assert legacy.observation_space.contains(observation)
    assert "valid_actions" in info
