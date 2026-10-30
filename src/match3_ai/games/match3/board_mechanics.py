import numpy as np

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
    )


def apply_board_gravity(state: BoardState) -> BoardState:
    """
    apply gravity while keeping tile colors and special types together.
    does not modify the original board.
    """
    tiles = state.tiles.copy()
    specials = state.specials.copy()

    rows, cols = state.shape

    for col in range(cols):
        playable_rows = [row for row in range(rows) if state.cell_mask[row, col]]

        tile_data = [
            (tiles[row, col], specials[row, col])
            for row in playable_rows
            if tiles[row, col] != EMPTY
        ]

        for row in playable_rows:
            tiles[row, col] = EMPTY
            specials[row, col] = SpecialType.NONE

        for row, (color, special) in zip(
            reversed(playable_rows),
            reversed(tile_data),
        ):
            tiles[row, col] = color
            specials[row, col] = special

    return BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=state.cell_mask,
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

    empty_cells = (tiles == EMPTY) & state.cell_mask
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
    )
