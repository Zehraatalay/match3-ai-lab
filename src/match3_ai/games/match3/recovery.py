import numpy as np

from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.rules import all_find_matches


def is_dead_board(state: BoardState) -> bool:
    """Return True if the board has no valid moves."""
    return len(find_valid_moves(state)) == 0


def reshuffle_board(
    state: BoardState,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> BoardState:
    """
    reshuffle existing tiles without changing their counts.
    the resulting board must:
    - preserve the original tile counts.
    - preserve the cell mask.
    - contain no existing matches.
    - have at least one valid move.
     raises RuntimeError if no valid arrangement is found.
    """

    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")

    # extract tiles from playable cells.
    original_values = state.tiles[state.cell_mask]

    # store the positions of playable cells.
    playable_positions = np.argwhere(state.cell_mask)

    for _ in range(max_attempts):
        # shuffle existing tiles without generating new ones.
        shuffled_values = rng.permutation(original_values)

        # start with an empty board.
        new_tiles = np.zeros_like(state.tiles)

        # place shuffled tiles into playable positions.
        new_tiles[
            playable_positions[:, 0],
            playable_positions[:, 1],
        ] = shuffled_values

        candidate = BoardState(
            tiles=new_tiles,
            cell_mask=state.cell_mask,
        )

        # reject boards containing existing matches.
        if all_find_matches(candidate.tiles):
            continue

        # reject boards without valid moves.
        if is_dead_board(candidate):
            continue

        return candidate

    raise RuntimeError(
        "unable to reshuffle the board into a playable state "
        f"after {max_attempts} attempts."
    )


def recover_dead_board(
    state: BoardState,
    rng: np.random.Generator,
    max_attempts: int = 1000,
) -> BoardState:
    """
    recover a dead board using reshuffling only.
    never generates new tiles.
    """

    if not is_dead_board(state):
        return state.copy()

    return reshuffle_board(
        state=state,
        rng=rng,
        max_attempts=max_attempts,
    )
