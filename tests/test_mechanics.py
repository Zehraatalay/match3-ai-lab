import numpy as np

from match3_ai.games.match3.mechanics import (
    apply_gravity,
    clear_matches,
    refill_board,
)


def test_clear_matches():
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 2, 2],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )

    matches = {(1, 0), (1, 1), (1, 2)}

    result = clear_matches(tiles, matches)

    expected = np.array(
        [
            [1, 2, 3],
            [0, 0, 0],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(result, expected)
    assert tiles[1, 0] == 2


def test_apply_gravity():
    tiles = np.array(
        [
            [1, 0, 3],
            [0, 2, 0],
            [3, 0, 1],
            [2, 1, 0],
        ],
        dtype=np.uint8,
    )

    mask = np.ones_like(tiles, dtype=np.bool_)

    result = apply_gravity(tiles, mask)

    expected = np.array(
        [
            [0, 0, 0],
            [1, 0, 0],
            [3, 2, 3],
            [2, 1, 1],
        ],
        dtype=np.uint8,
    )

    assert np.array_equal(result, expected)


def test_refill_board():
    tiles = np.array(
        [
            [1, 0, 3],
            [0, 2, 0],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )

    mask = np.ones_like(tiles, dtype=np.bool_)

    rng = np.random.default_rng(42)

    result = refill_board(
        tiles=tiles,
        cell_mask=mask,
        num_colors=5,
        rng=rng,
    )

    assert np.all(result > 0)
    assert np.all(result <= 5)

    assert result[0, 0] == 1
    assert result[1, 1] == 2
    assert result[2, 2] == 2
