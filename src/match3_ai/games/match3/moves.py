from match3_ai.games.match3.blockers import (
    BLOCKER_PROPERTIES,
    BlockerType,
)
from match3_ai.games.match3.board import EMPTY, BoardState
from match3_ai.games.match3.pattern_detection import detect_patterns
from match3_ai.games.match3.specials import SpecialType

Position = tuple[int, int]
Move = tuple[Position, Position]


# special control color controlden once bc renkler ayni olsa da swap valid olmali
def is_valid_swap(
    state: BoardState,
    pos_a: Position,
    pos_b: Position,
) -> bool:
    row_a, col_a = pos_a
    row_b, col_b = pos_b

    # 1. both positions must be inside the board.
    if not (
        0 <= row_a < state.rows
        and 0 <= col_a < state.cols
        and 0 <= row_b < state.rows
        and 0 <= col_b < state.cols
    ):
        return False

    # 2. only orthogonally adjacent cells can be swapped.
    if abs(row_a - row_b) + abs(col_a - col_b) != 1:
        return False

    # 3. both cells must be playable.
    if not (state.cell_mask[row_a, col_a] and state.cell_mask[row_b, col_b]):
        return False

    # sabit blockerlarin swap alanina girmesini engelle
    for position in (pos_a, pos_b):
        blocker_type = BlockerType(int(state.blockers[position]))
        if (
            blocker_type != BlockerType.NONE
            and not BLOCKER_PROPERTIES[blocker_type].allows_swap
        ):
            return False

    # 4. empty cells cannot participate in a swap.
    if state.tiles[row_a, col_a] == EMPTY or state.tiles[row_b, col_b] == EMPTY:
        return False

    special_a = SpecialType(int(state.specials[row_a, col_a]))
    special_b = SpecialType(int(state.specials[row_b, col_b]))

    # 5. any swap involving at least one special is valid.
    # this includes special + normal and special + special.
    if special_a != SpecialType.NONE or special_b != SpecialType.NONE:
        return True

    # 6. two identical normal tiles cannot create a new match.
    if state.tiles[row_a, col_a] == state.tiles[row_b, col_b]:
        return False

    # 7. for normal tiles, the swap must create a match.
    board = state.tiles.copy()

    board[pos_a], board[pos_b] = (
        board[pos_b],
        board[pos_a],
    )

    patterns = detect_patterns(board)

    return any(pos_a in pattern.cells or pos_b in pattern.cells for pattern in patterns)


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
