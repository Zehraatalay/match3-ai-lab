import numpy as np


def row_find_matches(board: np.ndarray) -> set[tuple[int, int]]:
    rows, cols = board.shape
    result: set[tuple[int, int]] = set()

    for row in range(rows):
        start = 0

        for current in range(1, cols + 1):

            if current == cols or board[row, current] != board[row, start]:

                match_length = current - start

                if match_length >= 3:
                    coordinates = {
                        (row, col)
                        for col in range(start, current)
                    }

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

                if match_length >= 3:
                    coordinates = {
                        (row, col)
                        for row in range(start, current)
                    }

                    result.update(coordinates)

                start = current

    return result


import numpy as np


def row_find_matches(board: np.ndarray) -> set[tuple[int, int]]:
    rows, cols = board.shape
    result: set[tuple[int, int]] = set()

    for row in range(rows):
        start = 0

        for current in range(1, cols + 1):

            if current == cols or board[row, current] != board[row, start]:

                match_length = current - start

                if match_length >= 3:
                    coordinates = {
                        (row, col)
                        for col in range(start, current)
                    }

                    result.update(coordinates)

                start = current

    return result


def all_find_matches(board: np.ndarray) -> set[tuple[int, int]]:

    horizontal = row_find_matches(board)
    vertical = col_find_matches(board)

    return horizontal | vertical

board = np.array([
    [1, 2, 3, 4, 1],
    [2, 2, 2, 4, 3],
    [3, 1, 3, 4, 2],
    [1, 3, 2, 4, 1],
    [2, 1, 3, 2, 3]
])

print("Horizontal:", row_find_matches(board))
print("Vertical:", col_find_matches(board))
print("All:", all_find_matches(board))

