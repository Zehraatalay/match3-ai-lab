from dataclasses import dataclass

import numpy as np

from match3_ai.games.match3.blockers import (
    BLOCKER_PROPERTIES,
    BlockerType,
)
from match3_ai.games.match3.constants import COLORLESS, EMPTY
from match3_ai.games.match3.specials import SpecialType


@dataclass(slots=True, eq=False)
class BoardState:
    tiles: np.ndarray
    cell_mask: np.ndarray
    specials: np.ndarray | None = None

    blockers: np.ndarray | None = None
    blocker_hp: np.ndarray | None = None

    def __post_init__(self) -> None:
        # --------------------------------------------------
        # Validate base arrays
        # --------------------------------------------------

        if not isinstance(self.tiles, np.ndarray):
            raise TypeError("tiles must be a NumPy array")

        if not isinstance(self.cell_mask, np.ndarray):
            raise TypeError("cell_mask must be a NumPy array")

        if self.tiles.ndim != 2:
            raise ValueError("tiles must be a 2D array")

        if self.cell_mask.shape != self.tiles.shape:
            raise ValueError(
                "cell_mask and tiles must have the same shape"
            )

        if self.tiles.dtype != np.uint8:
            raise TypeError("tiles must have dtype uint8")

        if self.cell_mask.dtype != np.bool_:
            raise TypeError("cell_mask must have dtype bool")

        if np.any(self.tiles[~self.cell_mask] != EMPTY):
            raise ValueError(
                "non-playable cells must be EMPTY"
            )

        # --------------------------------------------------
        # Initialize optional arrays
        # --------------------------------------------------

        if self.specials is None:
            self.specials = np.zeros_like(
                self.tiles,
                dtype=np.uint8,
            )

        if self.blockers is None:
            self.blockers = np.zeros_like(
                self.tiles,
                dtype=np.uint8,
            )

        if self.blocker_hp is None:
            self.blocker_hp = np.zeros_like(
                self.tiles,
                dtype=np.uint8,
            )

        # --------------------------------------------------
        # Validate array types and shapes
        # --------------------------------------------------

        for name, array in (
            ("specials", self.specials),
            ("blockers", self.blockers),
            ("blocker_hp", self.blocker_hp),
        ):
            if not isinstance(array, np.ndarray):
                raise TypeError(
                    f"{name} must be a NumPy array"
                )

            if array.shape != self.tiles.shape:
                raise ValueError(
                    f"{name} and tiles must have the same shape"
                )

            if array.dtype != np.uint8:
                raise TypeError(
                    f"{name} must have dtype uint8"
                )

        # --------------------------------------------------
        # Validate special types
        # --------------------------------------------------

        valid_special_values = np.array(
            [special.value for special in SpecialType],
            dtype=np.uint8,
        )

        if not np.all(
            np.isin(self.specials, valid_special_values)
        ):
            raise ValueError("unknown special tile type")

        if np.any(
            self.specials[~self.cell_mask]
            != SpecialType.NONE
        ):
            raise ValueError(
                "non-playable cells cannot contain specials"
            )

        if np.any(
            (self.specials != SpecialType.NONE)
            & (self.tiles == EMPTY)
        ):
            raise ValueError(
                "special tiles cannot occupy empty cells"
            )

        light_ball_mask = (
            self.specials == SpecialType.LIGHT_BALL
        )

        if np.any(
            self.tiles[light_ball_mask] != COLORLESS
        ):
            raise ValueError(
                "Light Ball tiles must be colorless"
            )

        other_specials = (
            (self.specials != SpecialType.NONE)
            & (self.specials != SpecialType.LIGHT_BALL)
        )

        if np.any(
            self.tiles[other_specials] == COLORLESS
        ):
            raise ValueError(
                "Only Light Balls can be colorless"
            )

        if np.any(
            (self.tiles == COLORLESS)
            & (self.specials != SpecialType.LIGHT_BALL)
        ):
            raise ValueError(
                "Colorless tiles must contain a Light Ball"
            )

        # --------------------------------------------------
        # Validate blocker types
        # --------------------------------------------------

        valid_blocker_values = np.array(
            [blocker.value for blocker in BlockerType],
            dtype=np.uint8,
        )

        if not np.all(
            np.isin(self.blockers, valid_blocker_values)
        ):
            raise ValueError("unknown blocker type")

        if np.any(
            self.blockers[~self.cell_mask]
            != BlockerType.NONE
        ):
            raise ValueError(
                "non-playable cells cannot contain blockers"
            )

        if np.any(
            self.blocker_hp[~self.cell_mask] != 0
        ):
            raise ValueError(
                "non-playable cells cannot have blocker HP"
            )

        # --------------------------------------------------
        # Validate blocker HP
        # --------------------------------------------------

        no_blocker_mask = (
            self.blockers == BlockerType.NONE
        )

        if np.any(
            self.blocker_hp[no_blocker_mask] != 0
        ):
            raise ValueError(
                "cells without blockers must have 0 HP"
            )

        has_blocker_mask = (
            self.blockers != BlockerType.NONE
        )

        if np.any(
            self.blocker_hp[has_blocker_mask] == 0
        ):
            raise ValueError(
                "blockers must have positive HP"
            )

        # --------------------------------------------------
        # Validate blocker and tile compatibility
        # --------------------------------------------------

        for blocker_type, properties in BLOCKER_PROPERTIES.items():
            blocker_mask = (
                self.blockers == blocker_type
            )

            if not np.any(blocker_mask):
                continue

            if not properties.contains_tile:
                if np.any(
                    self.tiles[blocker_mask] != EMPTY
                ):
                    raise ValueError(
                        f"{blocker_type.name} cannot contain tiles"
                    )

                if np.any(
                    self.specials[blocker_mask]
                    != SpecialType.NONE
                ):
                    raise ValueError(
                        f"{blocker_type.name} cannot contain specials"
                    )

        # --------------------------------------------------
        # Defensive copying
        # --------------------------------------------------

        self.tiles = self.tiles.copy()
        self.cell_mask = self.cell_mask.copy()
        self.specials = self.specials.copy()
        self.blockers = self.blockers.copy()
        self.blocker_hp = self.blocker_hp.copy()

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
        """Return an independent copy of the board state."""

        return BoardState(
            tiles=self.tiles,
            cell_mask=self.cell_mask,
            specials=self.specials,
            blockers=self.blockers,
            blocker_hp=self.blocker_hp,
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
