import numpy as np

from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.moves import (
    find_valid_moves,
    is_valid_swap,
)


def test_valid_swap():
    tiles = np.array(
        [
            [1, 2, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    mask = np.ones((3, 3), dtype=np.bool_)

    state = BoardState(tiles, mask)

    assert is_valid_swap(state, (0, 1), (1, 1))


def test_invalid_non_adjacent_swap():
    tiles = np.array(
        [
            [1, 2, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles,
        np.ones((3, 3), dtype=np.bool_),
    )

    assert not is_valid_swap(state, (0, 0), (2, 2))


def test_find_valid_moves():
    tiles = np.array(
        [
            [1, 2, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles,
        np.ones((3, 3), dtype=np.bool_),
    )

    moves = find_valid_moves(state)
    assert set(moves) == {
        ((0, 1), (1, 1)),
        ((1, 1), (1, 2)),
    }


def test_diagonal_swap_is_invalid():
    tiles = np.array(
        [
            [1, 2, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles,
        np.ones((3, 3), dtype=np.bool_),
    )

    assert not is_valid_swap(state, (0, 0), (1, 1))


def test_out_of_bounds_swap_is_invalid():
    tiles = np.array(
        [
            [1, 2, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles,
        np.ones((3, 3), dtype=np.bool_),
    )

    assert not is_valid_swap(state, (-1, 0), (0, 0))
    assert not is_valid_swap(state, (0, 0), (0, 3))


def test_swapping_identical_colors_is_invalid():
    tiles = np.array(
        [
            [1, 1, 2],
            [2, 3, 1],
            [3, 2, 3],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles,
        np.ones((3, 3), dtype=np.bool_),
    )

    assert not is_valid_swap(state, (0, 0), (0, 1))


def test_swapping_empty_cell_is_invalid():
    tiles = np.array(
        [
            [1, 0, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles,
        np.ones((3, 3), dtype=np.bool_),
    )

    assert not is_valid_swap(state, (0, 1), (1, 1))


def test_swapping_non_playable_cell_is_invalid():
    tiles = np.array(
        [
            [1, 0, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    mask = np.ones((3, 3), dtype=np.bool_)
    mask[0, 1] = False

    state = BoardState(tiles, mask)

    assert not is_valid_swap(state, (0, 1), (1, 1))


def test_valid_swap_does_not_modify_state():
    tiles = np.array(
        [
            [1, 2, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles,
        np.ones((3, 3), dtype=np.bool_),
    )

    original_tiles = state.tiles.copy()
    original_mask = state.cell_mask.copy()

    assert is_valid_swap(state, (0, 1), (1, 1))

    assert np.array_equal(state.tiles, original_tiles)
    assert np.array_equal(state.cell_mask, original_mask)


def test_invalid_swap_does_not_modify_state():
    tiles = np.array(
        [
            [1, 2, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles,
        np.ones((3, 3), dtype=np.bool_),
    )

    original_tiles = state.tiles.copy()
    original_mask = state.cell_mask.copy()

    assert not is_valid_swap(state, (0, 0), (2, 2))

    assert np.array_equal(state.tiles, original_tiles)
    assert np.array_equal(state.cell_mask, original_mask)


def test_existing_match_does_not_make_unrelated_swap_valid():
    tiles = np.array(
        [
            [1, 1, 1, 2],
            [2, 3, 4, 1],
            [3, 4, 2, 3],
            [4, 2, 3, 4],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles,
        np.ones((4, 4), dtype=np.bool_),
    )

    assert not is_valid_swap(state, (2, 0), (2, 1))
