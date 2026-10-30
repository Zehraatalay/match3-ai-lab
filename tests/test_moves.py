import numpy as np
import pytest

from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.moves import (
    find_valid_moves,
    is_valid_swap,
)
from match3_ai.games.match3.pattern_detection import detect_patterns
from match3_ai.games.match3.specials import SpecialType


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


def test_swap_creating_square_is_valid():
    tiles = np.array(
        [
            [1, 1, 2],
            [1, 2, 1],
            [3, 3, 2],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    assert detect_patterns(state.tiles) == []

    assert is_valid_swap(
        state,
        (1, 1),
        (1, 2),
    )


def test_swap_without_pattern_is_invalid():
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 3, 1],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    assert detect_patterns(state.tiles) == []

    assert not is_valid_swap(
        state,
        (0, 0),
        (0, 1),
    )


@pytest.mark.parametrize(
    "special_type",
    [
        SpecialType.ROCKET_HORIZONTAL,
        SpecialType.ROCKET_VERTICAL,
        SpecialType.BOMB,
        SpecialType.PROPELLER,
        SpecialType.LIGHT_BALL,
    ],
)
def test_special_normal_swap_is_valid(special_type):
    tiles = np.array(
        [
            [1, 2, 3],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles)

    if special_type == SpecialType.LIGHT_BALL:
        tiles[0, 0] = 255

    specials[0, 0] = special_type

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    assert is_valid_swap(
        state,
        (0, 0),
        (0, 1),
    )


def test_special_special_swap_is_valid():
    tiles = np.array(
        [
            [1, 1, 3],
            [3, 2, 1],
            [2, 3, 2],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles)

    specials[0, 0] = SpecialType.ROCKET_HORIZONTAL
    specials[0, 1] = SpecialType.BOMB

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    # same-color swap is valid because both tiles are special.
    assert is_valid_swap(
        state,
        (0, 0),
        (0, 1),
    )


def test_normal_swap_without_match_remains_invalid():
    tiles = np.array(
        [
            [1, 2, 3],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    assert not is_valid_swap(
        state,
        (0, 0),
        (0, 1),
    )


def test_find_valid_moves_includes_special_swaps():
    tiles = np.array(
        [
            [1, 2, 3],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles)
    specials[1, 1] = SpecialType.BOMB

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    valid_moves = find_valid_moves(state)

    assert ((0, 1), (1, 1)) in valid_moves
    assert ((1, 0), (1, 1)) in valid_moves
    assert ((1, 1), (1, 2)) in valid_moves
    assert ((1, 1), (2, 1)) in valid_moves
