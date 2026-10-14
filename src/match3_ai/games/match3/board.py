import numpy as np


def generate_random_board(
    rows: int,
    cols: int,
    num_colors: int,
    seed: int | None = None,
) -> np.ndarray:
    if rows <= 0 or cols <= 0:
        raise ValueError("board dims must be positive")

    if num_colors < 3:
        raise ValueError("at least 3 colors are required")

    rng = np.random.default_rng(seed)

    board = rng.integers(
        low=0,
        high=num_colors,
        size=(rows, cols),
        dtype=np.int32,
    )

    return board