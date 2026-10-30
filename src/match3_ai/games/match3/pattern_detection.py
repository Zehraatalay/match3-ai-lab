from dataclasses import dataclass
from typing import Literal

import numpy as np

from match3_ai.games.match3.constants import COLORLESS, EMPTY

Position = tuple[int, int]

PatternKind = Literal["horizontal", "vertical", "square"]


@dataclass(frozen=True, slots=True)
class MatchPattern:
    cells: frozenset[Position]
    kind: PatternKind


@dataclass(frozen=True, slots=True)
class MatchGroup:
    cells: frozenset[Position]
    patterns: tuple[MatchPattern, ...]
    color: int


def detect_line_patterns(board: np.ndarray) -> list[MatchPattern]:
    """
    detect maximal horizontal and vertical runs of 3+ equal colors.
    """
    rows, cols = board.shape
    patterns = []

    # horizontal runs
    for row in range(rows):
        start = 0

        for col in range(1, cols + 1):
            if col == cols or board[row, col] != board[row, start]:
                length = col - start
                color = board[row, start]

                if length >= 3 and color not in (EMPTY, COLORLESS):
                    cells = frozenset((row, c) for c in range(start, col))

                    patterns.append(
                        MatchPattern(
                            cells=cells,
                            kind="horizontal",
                        )
                    )

                start = col

    # vertical runs
    for col in range(cols):
        start = 0

        for row in range(1, rows + 1):
            if row == rows or board[row, col] != board[start, col]:
                length = row - start
                color = board[start, col]

                if length >= 3 and color not in (EMPTY, COLORLESS):
                    cells = frozenset((r, col) for r in range(start, row))

                    patterns.append(
                        MatchPattern(
                            cells=cells,
                            kind="vertical",
                        )
                    )

                start = row

    return patterns


def detect_square_patterns(board: np.ndarray) -> list[MatchPattern]:
    """
    detect 2x2 squares containing four equal normal-colored tiles.
    """
    rows, cols = board.shape
    patterns = []

    for row in range(rows - 1):
        for col in range(cols - 1):
            color = board[row, col]

            if color in (EMPTY, COLORLESS):
                continue

            if (
                board[row, col + 1] == color
                and board[row + 1, col] == color
                and board[row + 1, col + 1] == color
            ):
                cells = frozenset(
                    {
                        (row, col),
                        (row, col + 1),
                        (row + 1, col),
                        (row + 1, col + 1),
                    }
                )

                patterns.append(
                    MatchPattern(
                        cells=cells,
                        kind="square",
                    )
                )

    return patterns


def detect_patterns(board: np.ndarray) -> list[MatchPattern]:
    """
    detect all line and square patterns.
    """
    return detect_line_patterns(board) + detect_square_patterns(board)


def flood_fill_same_color(
    board: np.ndarray,
    start: Position,
) -> frozenset[Position]:
    """
    find all orthogonally connected cells with the same color.
    uses iterative DFS.
    """
    rows, cols = board.shape

    start_row, start_col = start
    color = int(board[start_row, start_col])

    if color in (EMPTY, COLORLESS):
        return frozenset()

    visited: set[Position] = set()
    stack = [start]

    while stack:
        row, col = stack.pop()

        if (row, col) in visited:
            continue

        if not (0 <= row < rows and 0 <= col < cols):
            continue

        if int(board[row, col]) != color:
            continue

        visited.add((row, col))

        neighbors = [
            (row - 1, col),
            (row + 1, col),
            (row, col - 1),
            (row, col + 1),
        ]

        for neighbor in neighbors:
            if neighbor not in visited:
                stack.append(neighbor)

    return frozenset(visited)


def group_patterns(
    board: np.ndarray,
    patterns: list[MatchPattern],
) -> list[MatchGroup]:
    """
    group valid match patterns using same-color connectivity.
    a group starts from at least one valid pattern.
    all orthogonally connected tiles of the same color
    are included, even if they are not part of a line
    or square pattern themselves.
    """
    groups: list[MatchGroup] = []

    processed_cells: set[Position] = set()

    for pattern in patterns:
        start = min(pattern.cells)

        if start in processed_cells:
            continue

        connected_cells = flood_fill_same_color(
            board,
            start,
        )

        if not connected_cells:
            continue

        group_patterns_list = [
            candidate for candidate in patterns if candidate.cells & connected_cells
        ]

        row, col = start

        groups.append(
            MatchGroup(
                cells=connected_cells,
                patterns=tuple(group_patterns_list),
                color=int(board[row, col]),
            )
        )

        processed_cells.update(connected_cells)

    return groups
