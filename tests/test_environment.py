import numpy as np
import pytest

from match3_ai.games.match3.environment import Match3Env
from match3_ai.games.match3.moves import is_valid_swap


def first_valid_action(observation):
    return int(np.flatnonzero(observation["action_mask"])[0])


@pytest.mark.parametrize(
    ("rows", "cols", "expected"),
    [
        (3, 3, 12),
        (5, 5, 40),
        (8, 8, 112),
        (6, 8, 82),
    ],
)
def test_action_space_size(rows, cols, expected):
    env = Match3Env(rows=rows, cols=cols)

    assert env.action_space.n == expected


def test_all_actions_are_unique():
    env = Match3Env(rows=8, cols=8)

    assert len(env.actions) == len(set(env.actions))


def test_actions_are_adjacent():
    env = Match3Env(rows=8, cols=8)

    for first, second in env.actions:
        distance = abs(first[0] - second[0]) + abs(first[1] - second[1])

        assert distance == 1


def test_reverse_swap_maps_to_same_action():
    env = Match3Env(rows=8, cols=8)

    first = (3, 3)
    second = (3, 4)

    forward = env._encode_action(first, second)
    reverse = env._encode_action(second, first)

    assert forward == reverse


def test_vertical_reverse_swap_maps_to_same_action():
    env = Match3Env(rows=8, cols=8)

    first = (3, 3)
    second = (4, 3)

    forward = env._encode_action(first, second)
    reverse = env._encode_action(second, first)

    assert forward == reverse


def test_encode_decode_round_trip():
    env = Match3Env(rows=8, cols=8)

    for action_id in range(env.action_space.n):
        first, second = env._decode_action(action_id)

        assert env._encode_action(first, second) == action_id
        assert env._encode_action(second, first) == action_id


def test_reset_returns_valid_observation():
    env = Match3Env()

    observation, info = env.reset(seed=42)

    assert env.observation_space.contains(observation)
    assert info["valid_actions"] > 0


def test_same_seed_produces_same_initial_board():
    env_a = Match3Env()
    env_b = Match3Env()

    obs_a, _ = env_a.reset(seed=42)
    obs_b, _ = env_b.reset(seed=42)

    assert np.array_equal(obs_a["board"], obs_b["board"])
    assert np.array_equal(
        obs_a["action_mask"],
        obs_b["action_mask"],
    )


def test_action_mask_matches_swap_validity():
    env = Match3Env(rows=6, cols=6)

    observation, _ = env.reset(seed=42)

    mask = observation["action_mask"]

    for action_id in range(env.action_space.n):
        first, second = env._decode_action(action_id)

        expected = is_valid_swap(
            env.state,
            first,
            second,
        )

        assert bool(mask[action_id]) == expected


def test_valid_action_decreases_moves_remaining():
    env = Match3Env(move_limit=10)

    observation, _ = env.reset(seed=42)

    action = first_valid_action(observation)

    next_obs, reward, terminated, truncated, info = env.step(action)

    assert info["valid_action"] is True
    assert next_obs["moves_remaining"] == 9
    assert reward >= 3
    assert not terminated
    assert not truncated


def test_invalid_action_does_not_consume_move():
    env = Match3Env(move_limit=10)

    observation, _ = env.reset(seed=42)

    invalid_actions = np.flatnonzero(observation["action_mask"] == 0)

    action = int(invalid_actions[0])

    next_obs, reward, terminated, truncated, info = env.step(action)

    assert info["valid_action"] is False
    assert next_obs["moves_remaining"] == 10
    assert reward == 0
    assert not terminated
    assert not truncated


def test_episode_truncates_at_move_limit():
    env = Match3Env(move_limit=1)

    observation, _ = env.reset(seed=42)

    action = first_valid_action(observation)

    _, _, terminated, truncated, _ = env.step(action)

    assert not terminated
    assert truncated


def test_step_before_reset_raises_error():
    env = Match3Env()

    with pytest.raises(RuntimeError):
        env.step(0)


def test_step_after_episode_raises_error():
    env = Match3Env(move_limit=1)

    observation, _ = env.reset(seed=42)

    action = first_valid_action(observation)

    env.step(action)

    with pytest.raises(RuntimeError):
        env.step(action)


@pytest.mark.parametrize(
    "action",
    [-1, 112, 9999],
)
def test_action_outside_space_raises_error(action):
    env = Match3Env(rows=8, cols=8)

    env.reset(seed=42)

    with pytest.raises(ValueError):
        env.step(action)


@pytest.mark.parametrize(
    ("rows", "cols", "num_colors", "move_limit"),
    [
        (2, 8, 5, 20),
        (8, 2, 5, 20),
        (8, 8, 2, 20),
        (8, 8, 256, 20),
        (8, 8, 5, 0),
    ],
)
def test_invalid_environment_parameters(
    rows,
    cols,
    num_colors,
    move_limit,
):
    with pytest.raises(ValueError):
        Match3Env(
            rows=rows,
            cols=cols,
            num_colors=num_colors,
            move_limit=move_limit,
        )
