from __future__ import annotations

from dataclasses import dataclass

from match3_ai.games.match3.board import BoardState

Position = tuple[int, int]


@dataclass(frozen=True, slots=True)
class TileMovement:
    source: Position
    destination: Position


def calculate_gravity_movements(
    before: BoardState,
    after: BoardState,
) -> tuple[TileMovement, ...]:
    rows, cols = before.shape
    movements: list[TileMovement] = []

    for col in range(cols):
        playable_rows = [row for row in range(rows) if before.cell_mask[row, col]]

        occupied_before = [row for row in playable_rows if before.tiles[row, col] != 0]

        occupied_after = [row for row in playable_rows if after.tiles[row, col] != 0]

        for source_row, destination_row in zip(
            occupied_before,
            occupied_after,
        ):
            if source_row != destination_row:
                movements.append(
                    TileMovement(
                        source=(source_row, col),
                        destination=(destination_row, col),
                    )
                )

    return tuple(movements)
