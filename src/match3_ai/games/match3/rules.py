import numpy as np

EMPTY = 0


def row_find_matches(board: np.ndarray) -> set[tuple[int, int]]:
    rows, cols = board.shape
    result: set[tuple[int, int]] = set()

    for row in range(rows):
        start = 0

        for current in range(1, cols + 1):
            if current == cols or board[row, current] != board[row, start]:
                match_length = current - start

                if match_length >= 3 and board[row, start] != EMPTY:
                    coordinates = {(row, col) for col in range(start, current)}

                    result.update(coordinates)

                start = current

    return result


def col_find_matches(board: np.ndarray) -> set[tuple[int, int]]:
    rows, cols = board.shape
    result: set[tuple[int, int]] = set()

    for col in range(cols):
        start = 0

        for current in range(1, rows + 1):
            if current == rows or board[current, col] != board[start, col]:
                match_length = current - start

                if match_length >= 3 and board[start, col] != EMPTY:
                    coordinates = {(row, col) for row in range(start, current)}

                    result.update(coordinates)

                start = current

    return result


def all_find_matches(board: np.ndarray) -> set[tuple[int, int]]:
    horizontal = row_find_matches(board)
    vertical = col_find_matches(board)

    return horizontal | vertical
