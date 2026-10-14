import numpy as np
import pytest

from match3_ai.games.match3.board import generate_random_board


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

    assert np.all(board >= 0)
    assert np.all(board < 5)


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