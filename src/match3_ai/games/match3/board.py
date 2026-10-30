from dataclasses import dataclass

import numpy as np

from match3_ai.games.match3.constants import COLORLESS, EMPTY
from match3_ai.games.match3.specials import SpecialType


@dataclass(slots=True, eq=False)
class BoardState:
    tiles: np.ndarray
    cell_mask: np.ndarray
    specials: np.ndarray | None = None

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
            raise ValueError("non-playable cells must be EMPTY")

        # backward compatibility:
        # existing BoardState calls do not provide specials.
        if self.specials is None:
            self.specials = np.zeros_like(
                self.tiles,
                dtype=np.uint8,
            )

        if not isinstance(self.specials, np.ndarray):
            raise TypeError("specials must be a NumPy array")

        if self.specials.shape != self.tiles.shape:
            raise ValueError("specials and tiles must have the same shape")

        if self.specials.dtype != np.uint8:
            raise TypeError("specials must have dtype uint8")

        valid_special_values = np.array(
            [special.value for special in SpecialType],
            dtype=np.uint8,
        )

        if not np.all(np.isin(self.specials, valid_special_values)):
            raise ValueError("unknown special tile type")

        # validation - bir special tile'ın empty cellde olmasını engelliyoruz
        if np.any(self.specials[~self.cell_mask] != SpecialType.NONE):
            raise ValueError("non-playable cells cannot contain specials")

        if np.any((self.specials != SpecialType.NONE) & (self.tiles == EMPTY)):
            raise ValueError("special tiles cannot occupy empty cells")

        light_ball_mask = self.specials == SpecialType.LIGHT_BALL

        # Light Balls must be colorless.
        if np.any(self.tiles[light_ball_mask] != COLORLESS):
            raise ValueError("Light Ball tiles must be colorless")

        # Other special tiles must have a normal color.
        other_specials = (self.specials != SpecialType.NONE) & (
            self.specials != SpecialType.LIGHT_BALL
        )

        if np.any(self.tiles[other_specials] == COLORLESS):
            raise ValueError("Only Light Balls can be colorless")

        # Colorless cells must contain a Light Ball.
        if np.any(
            (self.tiles == COLORLESS) & (self.specials != SpecialType.LIGHT_BALL)
        ):
            raise ValueError("Colorless tiles must contain a Light Ball")

        # defensive copying
        self.tiles = self.tiles.copy()
        self.cell_mask = self.cell_mask.copy()
        self.specials = self.specials.copy()

    @property
    def shape(self) -> tuple[int, int]:
        return self.tiles.shape

    @property
    def rows(self) -> int:
        return self.tiles.shape[0]

    @property
    def cols(self) -> int:
        return self.tiles.shape[1]

    # deep copy - constructor all arraysi kopyaladigi icin bagimsiz bir state olusmus oluyor
    def copy(self) -> "BoardState":
        return BoardState(
            tiles=self.tiles,
            cell_mask=self.cell_mask,
            specials=self.specials,
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

    if num_colors > 254:
        raise ValueError("num_colors cannot exceed 255")

    rng = np.random.default_rng(seed)

    board = rng.integers(
        low=1,
        high=num_colors + 1,
        size=(rows, cols),
        dtype=np.uint8,
    )

    return board
