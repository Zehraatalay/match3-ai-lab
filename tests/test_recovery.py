import numpy as np
import pytest

from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.recovery import (
    is_dead_board,
    recover_dead_board,
    reshuffle_board,
)
from match3_ai.games.match3.rules import all_find_matches


def make_dead_state() -> BoardState:
    tiles = np.array(
        [
            [1, 2, 3],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    mask = np.ones_like(tiles, dtype=np.bool_)

    return BoardState(tiles, mask)


def test_dead_board_detection():
    state = make_dead_state()

    assert is_dead_board(state)


def test_reshuffle_produces_playable_board():
    state = make_dead_state()

    rng = np.random.default_rng(42)

    result = reshuffle_board(
        state=state,
        rng=rng,
        max_attempts=1000,
    )

    assert not is_dead_board(result)
    assert not all_find_matches(result.tiles)


def test_reshuffle_preserves_tile_counts():
    state = make_dead_state()

    rng = np.random.default_rng(42)

    result = reshuffle_board(
        state=state,
        rng=rng,
    )

    original_counts = np.bincount(
        state.tiles.ravel(),
        minlength=4,
    )

    result_counts = np.bincount(
        result.tiles.ravel(),
        minlength=4,
    )

    assert np.array_equal(original_counts, result_counts)


def test_reshuffle_does_not_mutate_input():
    state = make_dead_state()

    original_tiles = state.tiles.copy()

    rng = np.random.default_rng(42)

    reshuffle_board(state, rng)

    assert np.array_equal(state.tiles, original_tiles)


def test_same_seed_produces_same_reshuffle():
    state = make_dead_state()

    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)

    result_a = reshuffle_board(state, rng_a)
    result_b = reshuffle_board(state, rng_b)

    assert np.array_equal(
        result_a.tiles,
        result_b.tiles,
    )


def test_recover_dead_board():
    state = make_dead_state()

    rng = np.random.default_rng(42)

    result = recover_dead_board(state, rng)

    assert not is_dead_board(result)
    assert not all_find_matches(result.tiles)


def test_recover_playable_board_returns_copy():
    tiles = np.array(
        [
            [1, 2, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    mask = np.ones_like(tiles, dtype=np.bool_)

    state = BoardState(tiles, mask)

    rng = np.random.default_rng(42)

    result = recover_dead_board(state, rng)

    assert np.array_equal(result.tiles, state.tiles)
    assert result is not state
    assert result.tiles is not state.tiles


def test_reshuffle_preserves_cell_mask():
    tiles = np.array(
        [
            [1, 2, 3, 1],
            [3, 0, 1, 2],
            [2, 3, 1, 2],
            [1, 2, 3, 1],
        ],
        dtype=np.uint8,
    )

    mask = tiles != 0

    state = BoardState(tiles, mask)

    rng = np.random.default_rng(42)

    result = reshuffle_board(state, rng)

    assert np.array_equal(
        result.cell_mask,
        state.cell_mask,
    )

    assert np.all(result.tiles[~result.cell_mask] == 0)


def test_reshuffle_impossible_board_raises_error():
    # Only two tiles of each color: no 3-match is possible.
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    mask = np.ones_like(tiles, dtype=np.bool_)

    state = BoardState(tiles, mask)

    rng = np.random.default_rng(42)

    with pytest.raises(RuntimeError):
        reshuffle_board(
            state,
            rng,
            max_attempts=50,
        )


@pytest.mark.parametrize(
    "max_attempts",
    [0, -1, 2.5, True],
)
def test_invalid_max_attempts(max_attempts):
    state = make_dead_state()

    rng = np.random.default_rng(42)

    with pytest.raises(ValueError):
        reshuffle_board(
            state,
            rng,
            max_attempts=max_attempts,
        )
