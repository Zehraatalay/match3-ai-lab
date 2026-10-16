import numpy as np

from match3_ai.games.match3.board import (
    BoardState,
    generate_random_board,
)
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.rules import all_find_matches


def generate_playable_board(
    rows: int,
    cols: int,
    num_colors: int,
    seed: int | None = None,
    max_attempts: int = 1000,
) -> BoardState:
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    rng = np.random.default_rng(seed)

    for _ in range(max_attempts):
        board_seed = int(rng.integers(0, 2**32))

        tiles = generate_random_board(
            rows=rows,
            cols=cols,
            num_colors=num_colors,
            seed=board_seed,
        )

        if all_find_matches(tiles):  # board baslangicta match icermiyor
            continue

        cell_mask = np.ones((rows, cols), dtype=np.bool_)

        state = BoardState(
            tiles=tiles,
            cell_mask=cell_mask,
        )

        if find_valid_moves(state):  # board playable oluyor
            return state

    raise RuntimeError(
        f"could not generate a playable board after {max_attempts} attempts"
    )
