import numpy as np

from match3_ai.games.match3.board import EMPTY


# match detection sonucunda match corrdinates belirleniyor
# clear_matches() sonrasında ise eslesme olan coordinates EMPTY oluyor
# tiles.copy() bc ai agents same board uzerinde baslangic state'ini kullanabilmeli
def clear_matches(
    tiles: np.ndarray,
    matches: set[tuple[int, int]],
) -> np.ndarray:
    result = tiles.copy()

    for row, col in matches:
        result[row, col] = EMPTY

    return result


def apply_gravity(
    tiles: np.ndarray,
    cell_mask: np.ndarray,
) -> np.ndarray:
    result = tiles.copy()

    rows, cols = result.shape
    # EMPTY olmayan degerleri buluyoruz her col icin
    for col in range(cols):
        playable_rows = [row for row in range(rows) if cell_mask[row, col]]
        # EMPTY olmayan degerleri asagidan yukari yerlestiriyoruz
        values = [
            result[row, col] for row in playable_rows if result[row, col] != EMPTY
        ]
        # EMPTY olan degerleri en uste ekliyoruz
        for row in playable_rows:
            result[row, col] = EMPTY

        for row, value in zip(reversed(playable_rows), reversed(values)):
            result[row, col] = value

    return result


def refill_board(
    tiles: np.ndarray,
    cell_mask: np.ndarray,
    num_colors: int,
    rng: np.random.Generator,
) -> np.ndarray:
    result = tiles.copy()
    # bool mask olusturuyoruz
    empty_cells = (result == EMPTY) & cell_mask

    count = int(np.count_nonzero(empty_cells))

    if count == 0:
        return result

    new_tiles = rng.integers(
        low=1,
        high=num_colors + 1,
        size=count,
        dtype=np.uint8,
    )
    # just EMPTY olan cells dolduruluyor
    result[empty_cells] = new_tiles

    return result
