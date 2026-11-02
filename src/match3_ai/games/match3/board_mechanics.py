import numpy as np

from match3_ai.games.match3.blockers import BLOCKER_PROPERTIES
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.constants import COLORLESS, EMPTY
from match3_ai.games.match3.special_selection import SpecialCreation
from match3_ai.games.match3.specials import SpecialType

Position = tuple[int, int]


def clear_board_matches(
    state: BoardState,
    matches: set[Position],
) -> BoardState:
    """
    clear matched tiles and their associated special types.
    does not modify the original board.
    """
    tiles = state.tiles.copy()
    specials = state.specials.copy()

    for row, col in matches:
        tiles[row, col] = EMPTY
        specials[row, col] = SpecialType.NONE

    return BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=state.cell_mask,
        blockers=state.blockers,
        blocker_hp=state.blocker_hp,
    )


def apply_board_gravity(state: BoardState) -> BoardState:
    """
    apply gravity while keeping tile colors and special types together.
    does not modify the original board.
    """
    tiles = state.tiles.copy()
    specials = state.specials.copy()

    rows, cols = state.shape
    fixed_blocker_types = [
        blocker_type
        for blocker_type, properties in BLOCKER_PROPERTIES.items()
        if not properties.allows_gravity
    ]
    fixed_cells = ~state.cell_mask | np.isin(
        state.blockers,
        fixed_blocker_types,
    )

    for col in range(cols):
        segment_start: int | None = None

        for row in range(rows + 1):
            is_segment_cell = row < rows and not fixed_cells[row, col]

            if is_segment_cell and segment_start is None:
                segment_start = row
                continue

            if is_segment_cell or segment_start is None:
                continue

            write_row = row - 1

            for read_row in range(row - 1, segment_start - 1, -1):
                if tiles[read_row, col] == EMPTY:
                    continue

                if write_row != read_row:
                    tiles[write_row, col] = tiles[read_row, col]
                    specials[write_row, col] = specials[read_row, col]
                    tiles[read_row, col] = EMPTY
                    specials[read_row, col] = SpecialType.NONE

                write_row -= 1

            for empty_row in range(segment_start, write_row + 1):
                tiles[empty_row, col] = EMPTY
                specials[empty_row, col] = SpecialType.NONE

            segment_start = None

    return BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=state.cell_mask,
        blockers=state.blockers,
        blocker_hp=state.blocker_hp,
    )


def refill_board_state(
    state: BoardState,
    num_colors: int,
    rng: np.random.Generator,
) -> BoardState:
    """
    refill empty playable cells with normal colored tiles.
    existing special tiles are preserved.
    """
    tiles = state.tiles.copy()
    specials = state.specials.copy()

    refill_blocker_types = [
        blocker_type
        for blocker_type, properties in BLOCKER_PROPERTIES.items()
        if properties.blocks_refill
    ]
    empty_cells = (
        (tiles == EMPTY)
        & state.cell_mask
        & ~np.isin(state.blockers, refill_blocker_types)
    )
    count = int(np.count_nonzero(empty_cells))

    if count == 0:
        return state.copy()

    new_tiles = rng.integers(
        low=1,
        high=num_colors + 1,
        size=count,
        dtype=np.uint8,
    )

    tiles[empty_cells] = new_tiles
    specials[empty_cells] = SpecialType.NONE

    return BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=state.cell_mask,
        blockers=state.blockers,
        blocker_hp=state.blocker_hp,
    )


def place_special_creations(
    state: BoardState,
    creations: list[SpecialCreation],
    original_tiles: np.ndarray,  # cünkü once eslesen all tas temizlenecek sonra ilgili cellsin renkleri EMPTY olacak
) -> BoardState:
    """
    place newly created special tiles on a cleared board.
    normal specials preserve their original color.
    light Balls use COLORLESS (255).
    """
    tiles = state.tiles.copy()
    specials = state.specials.copy()

    for creation in creations:
        row, col = creation.position

        if creation.special_type == SpecialType.LIGHT_BALL:
            tiles[row, col] = COLORLESS
        else:
            # original tiles uzerinden aliyoruz rocket vs rengini
            # light ball zaten colorless
            tiles[row, col] = original_tiles[row, col]

        specials[row, col] = creation.special_type

    return BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=state.cell_mask,
        blockers=state.blockers,
        blocker_hp=state.blocker_hp,
    )
