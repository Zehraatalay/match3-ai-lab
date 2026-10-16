import numpy as np
import pytest

from match3_ai.games.match3.board import (
    BoardState,
    generate_random_board,
)


def test_board_shape():
    board = generate_random_board(
        rows=8,
        cols=8,
        num_colors=5,
        seed=42,
    )

    assert board.shape == (8, 8)


def test_board_color_range():
    board = generate_random_board(8, 8, 5, seed=42)

    assert np.all(board >= 1)
    assert np.all(board <= 5)


def test_same_seed_produces_same_board():
    board_a = generate_random_board(8, 8, 5, seed=42)
    board_b = generate_random_board(8, 8, 5, seed=42)

    assert np.array_equal(board_a, board_b)


def test_invalid_dimensions():
    with pytest.raises(ValueError):
        generate_random_board(0, 8, 5)


def test_invalid_color_count():
    with pytest.raises(ValueError):
        generate_random_board(8, 8, 2)


def test_board_state_valid():
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 1, 3],
            [3, 2, 1],
        ],
        dtype=np.uint8,
    )

    mask = np.ones((3, 3), dtype=np.bool_)

    state = BoardState(
        tiles=tiles,
        cell_mask=mask,
    )

    assert state.shape == (3, 3)
    assert state.rows == 3
    assert state.cols == 3


def test_board_state_copy_is_independent():
    tiles = np.ones((3, 3), dtype=np.uint8)
    mask = np.ones((3, 3), dtype=np.bool_)

    state = BoardState(
        tiles=tiles,
        cell_mask=mask,
    )

    copied = state.copy()

    copied.tiles[0, 0] = 5

    assert state.tiles[0, 0] == 1
    assert copied.tiles[0, 0] == 5


def test_non_playable_cells_must_be_empty():
    tiles = np.ones((3, 3), dtype=np.uint8)
    mask = np.ones((3, 3), dtype=np.bool_)

    mask[1, 1] = False

    with pytest.raises(ValueError):
        BoardState(
            tiles=tiles,
            cell_mask=mask,
        )


def test_board_state_invalid_dtype():
    tiles = np.ones((3, 3), dtype=np.int32)
    mask = np.ones((3, 3), dtype=np.bool_)

    with pytest.raises(TypeError):
        BoardState(
            tiles=tiles,
            cell_mask=mask,
        )


def test_board_state_invalid_shape():
    tiles = np.ones((3, 3), dtype=np.uint8)
    mask = np.ones((4, 4), dtype=np.bool_)

    with pytest.raises(ValueError):
        BoardState(
            tiles=tiles,
            cell_mask=mask,
        )


def test_irregular_board():
    tiles = np.array(
        [
            [0, 1, 2, 0],
            [1, 2, 3, 4],
            [0, 3, 1, 0],
        ],
        dtype=np.uint8,
    )

    mask = np.array(
        [
            [False, True, True, False],
            [True, True, True, True],
            [False, True, True, False],
        ],
        dtype=np.bool_,
    )

    state = BoardState(
        tiles=tiles,
        cell_mask=mask,
    )

    assert state.shape == (3, 4)
    assert not state.cell_mask[0, 0]
    assert state.tiles[0, 0] == 0


def test_board_state_does_not_share_input_arrays():
    tiles = np.ones((3, 3), dtype=np.uint8)
    mask = np.ones((3, 3), dtype=np.bool_)

    state = BoardState(
        tiles=tiles,
        cell_mask=mask,
    )

    tiles[0, 0] = 5
    mask[0, 0] = False

    assert state.tiles[0, 0] == 1
    assert state.cell_mask[0, 0]