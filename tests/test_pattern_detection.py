import numpy as np

from match3_ai.games.match3.pattern_detection import (
    detect_line_patterns,
    detect_square_patterns,
)


def test_horizontal_four():
    board = np.array(
        [
            [1, 1, 1, 1],
            [2, 3, 2, 3],
            [3, 2, 3, 2],
        ],
        dtype=np.uint8,
    )

    patterns = detect_line_patterns(board)

    assert len(patterns) == 1
    assert patterns[0].kind == "horizontal"
    assert len(patterns[0].cells) == 4


def test_vertical_five():
    board = np.array(
        [
            [1, 2, 3],
            [1, 3, 2],
            [1, 2, 3],
            [1, 3, 2],
            [1, 2, 3],
        ],
        dtype=np.uint8,
    )

    patterns = detect_line_patterns(board)

    assert any(
        pattern.kind == "vertical" and len(pattern.cells) == 5 for pattern in patterns
    )


def test_square_pattern():
    board = np.array(
        [
            [1, 1, 2],
            [1, 1, 3],
            [2, 3, 2],
        ],
        dtype=np.uint8,
    )

    patterns = detect_square_patterns(board)

    assert len(patterns) == 1
    assert patterns[0].kind == "square"
    assert len(patterns[0].cells) == 4


def test_no_patterns():
    board = np.array(
        [
            [1, 2, 3],
            [2, 3, 1],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )

    assert detect_line_patterns(board) == []
    assert detect_square_patterns(board) == []


def test_colorless_tiles_do_not_form_patterns():
    board = np.array(
        [
            [255, 255, 255],
            [1, 2, 3],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    assert detect_line_patterns(board) == []


def test_five_in_a_row_is_one_maximal_pattern():
    board = np.array(
        [
            [1, 1, 1, 1, 1],
            [2, 3, 2, 3, 2],
            [3, 2, 3, 2, 3],
        ],
        dtype=np.uint8,
    )

    patterns = detect_line_patterns(board)

    assert len(patterns) == 1
    assert len(patterns[0].cells) == 5


from match3_ai.games.match3.pattern_detection import (
    detect_patterns,
    group_patterns,
)


def test_connected_extra_tiles_join_match_group():
    board = np.array(
        [
            [2, 3, 1, 2],
            [1, 1, 1, 3],
            [2, 3, 1, 2],
            [3, 2, 1, 1],
        ],
        dtype=np.uint8,
    )

    patterns = detect_patterns(board)
    groups = group_patterns(board, patterns)

    assert len(groups) == 1

    expected_cells = {
        (0, 2),
        (1, 0),
        (1, 1),
        (1, 2),
        (2, 2),
        (3, 2),
        (3, 3),
    }

    assert groups[0].cells == expected_cells


def test_diagonal_neighbors_do_not_connect():
    board = np.array(
        [
            [1, 1, 1, 2],
            [2, 2, 2, 1],
            [3, 3, 1, 2],
            [2, 3, 2, 3],
        ],
        dtype=np.uint8,
    )

    patterns = detect_patterns(board)
    groups = group_patterns(board, patterns)

    color_one_groups = [group for group in groups if group.color == 1]

    assert len(color_one_groups) == 1

    assert (2, 2) not in color_one_groups[0].cells


def test_connected_patterns_form_one_group():
    board = np.array(
        [
            [1, 1, 1, 2, 3],
            [2, 3, 1, 2, 3],
            [2, 3, 1, 1, 1],
        ],
        dtype=np.uint8,
    )

    patterns = detect_patterns(board)
    groups = group_patterns(board, patterns)

    assert len(groups) == 1

    assert groups[0].cells == {
        (0, 0),
        (0, 1),
        (0, 2),
        (1, 2),
        (2, 2),
        (2, 3),
        (2, 4),
    }


def test_unmatched_connected_component_is_not_cleared():
    board = np.array(
        [
            [1, 1, 2, 3],
            [2, 3, 1, 2],
            [3, 2, 3, 1],
        ],
        dtype=np.uint8,
    )

    patterns = detect_patterns(board)
    groups = group_patterns(board, patterns)

    assert groups == []
