from dataclasses import dataclass

import numpy as np

EMPTY = 0


@dataclass(slots=True, eq=False)
class BoardState:
    tiles: np.ndarray
    cell_mask: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(self.tiles, np.ndarray):
            raise TypeError("tiles must be a NumPy array")

        if not isinstance(self.cell_mask, np.ndarray):
            raise TypeError("cell_mask must be a NumPy array")

        if self.tiles.ndim != 2:
            raise ValueError("tiles must be a 2D array")

        if self.cell_mask.shape != self.tiles.shape:
            raise ValueError("cell_mask and tiles must have the same shape")

        if self.tiles.dtype != np.uint8:
            raise TypeError("tiles must have dtype uint8")

        if self.cell_mask.dtype != np.bool_:
            raise TypeError("cell_mask must have dtype bool")

        if np.any(self.tiles[~self.cell_mask] != EMPTY):
            raise ValueError("Non-playable cells must be EMPTY")

        self.tiles = self.tiles.copy()
        self.cell_mask = self.cell_mask.copy()

    @property
    def shape(self) -> tuple[int, int]:
        return self.tiles.shape

    @property
    def rows(self) -> int:
        return self.tiles.shape[0]

    @property
    def cols(self) -> int:
        return self.tiles.shape[1]

    def copy(self) -> "BoardState":
        return BoardState(
            tiles=self.tiles,
            cell_mask=self.cell_mask,
        )


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

    if num_colors > 255:
        raise ValueError("num_colors cannot exceed 255")

    rng = np.random.default_rng(seed)

    board = rng.integers(
        low=1,
        high=num_colors + 1,
        size=(rows, cols),
        dtype=np.uint8,
    )

    return board
