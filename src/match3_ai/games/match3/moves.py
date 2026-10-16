from match3_ai.games.match3.board import EMPTY, BoardState
from match3_ai.games.match3.rules import all_find_matches

Position = tuple[int, int]
Move = tuple[Position, Position]


def is_valid_swap(
    state: BoardState,
    pos_a: Position,
    pos_b: Position,
) -> bool:
    row_a, col_a = pos_a
    row_b, col_b = pos_b

    # check bounds
    if not (
        0 <= row_a < state.rows
        and 0 <= col_a < state.cols
        and 0 <= row_b < state.rows
        and 0 <= col_b < state.cols
    ):
        return False

    # check adjacency
    distance = abs(row_a - row_b) + abs(col_a - col_b)

    if distance != 1:
        return False

    # check playable cells
    if not state.cell_mask[row_a, col_a]:
        return False

    if not state.cell_mask[row_b, col_b]:
        return False

    # check empty cells
    if state.tiles[row_a, col_a] == EMPTY:
        return False

    if state.tiles[row_b, col_b] == EMPTY:
        return False

    # swapping identical colors cannot create a new match
    if state.tiles[row_a, col_a] == state.tiles[row_b, col_b]:
        return False

    # simulate swap
    # burada takas sonrasinda olusan matchin takas edilen cellsden en az 1ini icerip icermedigini kontrol ediyoruz
    # cünkü mevcut board zaten match iceriyorsa alakasiz bir takas da valid kabul edilebilir
    board = state.tiles.copy()
    board[pos_a], board[pos_b] = board[pos_b], board[pos_a]
    matches = all_find_matches(board)
    return pos_a in matches or pos_b in matches


def find_valid_moves(state: BoardState) -> list[Move]:
    valid_moves: list[Move] = []

    for row in range(state.rows):
        for col in range(state.cols):
            pos_a = (row, col)

            # only check right and down to avoid duplicates
            neighbors = [
                (row, col + 1),
                (row + 1, col),
            ]

            for pos_b in neighbors:
                if is_valid_swap(state, pos_a, pos_b):
                    valid_moves.append((pos_a, pos_b))

    return valid_moves
