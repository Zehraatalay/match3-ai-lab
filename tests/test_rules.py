import numpy as np

from match3_ai.games.match3.rules import (
    all_find_matches,
    col_find_matches,
    row_find_matches,
)


def test_horizontal_match():
    board = np.array(
        [
            [1, 1, 1, 2],
            [2, 3, 4, 1],
            [3, 4, 2, 3],
        ]
    )

    expected = {(0, 0), (0, 1), (0, 2)}

    assert row_find_matches(board) == expected


def test_vertical_match():
    board = np.array(
        [
            [1, 2, 3, 1],
            [1, 4, 2, 3],
            [1, 3, 1, 4],
            [2, 4, 1, 2],
        ]
    )

    expected = {(0, 0), (1, 0), (2, 0)}

    assert col_find_matches(board) == expected


def test_horizontal_and_vertical_matches():
    board = np.array(
        [
            [1, 2, 3, 4],
            [2, 2, 2, 4],
            [3, 1, 1, 4],
            [1, 2, 3, 1],
        ]
    )

    expected = {
        (0, 3),
        (1, 3),
        (2, 3),
        (1, 0),
        (1, 1),
        (1, 2),
    }

    assert all_find_matches(board) == expected


def test_no_matches():
    board = np.array(
        [
            [1, 2, 3],
            [2, 3, 1],
            [3, 1, 2],
        ]
    )

    assert all_find_matches(board) == set()


def test_four_tile_match():
    board = np.array(
        [
            [1, 1, 1, 1],
            [2, 3, 4, 2],
            [3, 4, 2, 3],
        ]
    )

    expected = {(0, 0), (0, 1), (0, 2), (0, 3)}

    assert all_find_matches(board) == expected


def test_five_tile_match():
    board = np.array(
        [
            [1, 2, 3],
            [1, 3, 2],
            [1, 4, 3],
            [1, 2, 4],
            [1, 3, 2],
        ]
    )

    expected = {
        (0, 0),
        (1, 0),
        (2, 0),
        (3, 0),
        (4, 0),
    }

    assert all_find_matches(board) == expected


def test_cross_matches():
    board = np.array(
        [
            [1, 2, 3, 4, 1],
            [2, 1, 3, 2, 4],
            [1, 3, 3, 3, 2],
            [4, 2, 3, 1, 4],
            [2, 4, 1, 2, 3],
        ]
    )

    expected = {
        (2, 1),
        (2, 2),
        (2, 3),
        (0, 2),
        (1, 2),
        (3, 2),
    }

    assert all_find_matches(board) == expected


def test_matches_at_boundaries():
    board = np.array(
        [
            [1, 2, 3, 4, 4, 4],
            [2, 3, 1, 2, 3, 1],
            [3, 1, 2, 3, 1, 2],
            [4, 2, 3, 1, 2, 3],
        ]
    )

    expected = {(0, 3), (0, 4), (0, 5)}

    assert all_find_matches(board) == expected


def test_empty_cells_do_not_match_horizontally():
    board = np.array(
        [
            [0, 0, 0, 1],
            [1, 2, 3, 2],
            [2, 3, 1, 3],
        ]
    )

    assert row_find_matches(board) == set()


def test_empty_cells_do_not_match_vertically():
    board = np.array(
        [
            [0, 1, 2],
            [0, 2, 3],
            [0, 3, 1],
        ]
    )

    assert col_find_matches(board) == set()


def test_empty_cells_break_matches():
    board = np.array(
        [
            [1, 1, 0, 1, 1],
            [2, 3, 1, 2, 3],
            [3, 2, 2, 3, 2],
        ]
    )

    assert all_find_matches(board) == set()


def test_matches_with_empty_cells():
    board = np.array(
        [
            [0, 0, 0, 1],
            [2, 2, 2, 3],
            [1, 3, 1, 2],
        ]
    )

    expected = {(1, 0), (1, 1), (1, 2)}

    assert all_find_matches(board) == expected
