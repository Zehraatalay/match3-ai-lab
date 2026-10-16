from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.mechanics import (
    apply_gravity,
    clear_matches,
    refill_board,
)
from match3_ai.games.match3.moves import is_valid_swap
from match3_ai.games.match3.recovery import (
    is_dead_board,
    recover_dead_board,
)
from match3_ai.games.match3.rules import all_find_matches

Position = tuple[int, int]
Action = tuple[Position, Position]


@dataclass(frozen=True, slots=True)
class StepResult:
    state: BoardState
    valid: bool
    cleared_tiles: int
    cascade_count: int
    reshuffled: bool = False


class Match3Engine:
    def __init__(
        self,
        num_colors: int,
        seed: int | None = None,
        max_cascades: int = 100,
    ) -> None:
        if type(num_colors) is not int or not 3 <= num_colors <= 255:
            raise ValueError("num_colors must be an integer between 3 and 255")

        if type(max_cascades) is not int or max_cascades <= 0:
            raise ValueError("max_cascades must be a positive integer")

        self.num_colors = num_colors
        self.max_cascades = max_cascades
        self.rng = np.random.default_rng(seed)

    def step(
        self,
        state: BoardState,
        action: Action,
    ) -> StepResult:
        first, second = action

        if not is_valid_swap(state, first, second):
            return StepResult(
                state=state.copy(),
                valid=False,
                cleared_tiles=0,
                cascade_count=0,
            )

        tiles = state.tiles.copy()
        mask = state.cell_mask

        tiles[first], tiles[second] = (
            tiles[second],
            tiles[first],
        )

        total_cleared = 0
        cascade_count = 0

        matches = all_find_matches(tiles)

        while matches:
            if cascade_count >= self.max_cascades:
                raise RuntimeError("Maximum cascade limit exceeded")

            total_cleared += len(matches)
            cascade_count += 1

            tiles = clear_matches(tiles, matches)

            tiles = apply_gravity(
                tiles=tiles,
                cell_mask=mask,
            )

            tiles = refill_board(
                tiles=tiles,
                cell_mask=mask,
                num_colors=self.num_colors,
                rng=self.rng,
            )

            matches = all_find_matches(tiles)

        new_state = BoardState(
            tiles=tiles,
            cell_mask=mask,
        )

        reshuffled = False

        if is_dead_board(new_state):
            new_state = recover_dead_board(
                state=new_state,
                rng=self.rng,
            )

            reshuffled = True

        return StepResult(
            state=new_state,
            valid=True,
            cleared_tiles=total_cleared,
            cascade_count=cascade_count,
            reshuffled=reshuffled,
        )
