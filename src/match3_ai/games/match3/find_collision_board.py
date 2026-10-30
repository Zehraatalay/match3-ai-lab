import numpy as np

from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.moves import is_valid_swap
from match3_ai.games.match3.pattern_detection import (
    detect_patterns,
    group_patterns,
)
from match3_ai.games.match3.specials import SpecialType

rng = np.random.default_rng(42)

for _ in range(100_000):
    tiles = rng.integers(1, 4, size=(5, 5), dtype=np.uint8)

    if detect_patterns(tiles):
        continue

    for row in range(5):
        for col in range(5):
            source = (row, col)

            for dr, dc in ((1, 0), (0, 1)):
                nr, nc = row + dr, col + dc

                if nr >= 5 or nc >= 5:
                    continue

                destination = (nr, nc)

                swapped = tiles.copy()
                swapped[source], swapped[destination] = (
                    swapped[destination],
                    swapped[source],
                )

                patterns = detect_patterns(swapped)

                if not patterns:
                    continue

                groups = group_patterns(swapped, patterns)

                if len(groups) < 2:
                    continue

                creation_groups = [group for group in groups if len(group.cells) >= 4]

                if not creation_groups:
                    continue

                for rocket_group in groups:
                    if rocket_group in creation_groups:
                        continue

                    for rocket_position in rocket_group.cells:
                        specials = np.zeros_like(tiles, dtype=np.uint8)
                        specials[rocket_position] = SpecialType.ROCKET_VERTICAL

                        state = BoardState(
                            tiles=tiles,
                            specials=specials,
                            cell_mask=np.ones_like(tiles, dtype=bool),
                        )

                        if not is_valid_swap(state, source, destination):
                            continue

                        print("FOUND BOARD:")
                        print(tiles)
                        print("ACTION:", (source, destination))
                        print("ROCKET POSITION:", rocket_position)
                        print("GROUPS:", [group.cells for group in groups])
                        raise SystemExit

print("No board found")
