import numpy as np

from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.board_mechanics import (
    apply_board_gravity,
    clear_board_matches,
    refill_board_state,
)
from match3_ai.games.match3.specials import SpecialType


def make_state(tiles, specials=None):
    tiles = np.array(tiles, dtype=np.uint8)

    if specials is None:
        specials = np.zeros_like(tiles, dtype=np.uint8)
    else:
        specials = np.array(specials, dtype=np.uint8)

    mask = np.ones_like(tiles, dtype=np.bool_)

    return BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=mask,
    )


def test_clear_board_matches_removes_tiles():
    state = make_state(
        [
            [1, 1, 1],
            [2, 3, 2],
            [3, 2, 3],
        ]
    )

    result = clear_board_matches(
        state,
        {(0, 0), (0, 1), (0, 2)},
    )

    assert np.all(result.tiles[0] == 0)
    assert np.all(result.specials[0] == SpecialType.NONE)


def test_clear_board_matches_removes_special():
    state = make_state(
        [
            [1, 2, 3],
            [2, 3, 1],
            [3, 1, 2],
        ],
        [
            [0, 0, 0],
            [0, SpecialType.BOMB, 0],
            [0, 0, 0],
        ],
    )

    result = clear_board_matches(state, {(1, 1)})

    assert result.tiles[1, 1] == 0
    assert result.specials[1, 1] == SpecialType.NONE


def test_gravity_moves_special_with_tile():
    state = make_state(
        [
            [1, 2, 3],
            [0, 0, 0],
            [3, 1, 2],
        ],
        [
            [0, SpecialType.BOMB, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
    )

    result = apply_board_gravity(state)

    assert result.tiles[1, 1] == 2
    assert result.specials[1, 1] == SpecialType.BOMB

    assert result.tiles[0, 1] == 0
    assert result.specials[0, 1] == SpecialType.NONE


def test_gravity_preserves_tile_order():
    state = make_state(
        [
            [1],
            [0],
            [2],
            [0],
            [3],
        ]
    )

    result = apply_board_gravity(state)

    assert result.tiles[:, 0].tolist() == [0, 0, 1, 2, 3]


def test_refill_creates_normal_tiles():
    state = make_state(
        [
            [1, 0, 3],
            [0, 2, 0],
            [3, 1, 2],
        ]
    )

    rng = np.random.default_rng(42)

    result = refill_board_state(
        state,
        num_colors=5,
        rng=rng,
    )

    assert np.all(result.tiles > 0)
    assert np.all(result.specials == SpecialType.NONE)


def test_refill_preserves_existing_special():
    state = make_state(
        [
            [1, 0, 3],
            [2, 3, 1],
            [3, 1, 2],
        ],
        [
            [0, 0, 0],
            [0, SpecialType.ROCKET_HORIZONTAL, 0],
            [0, 0, 0],
        ],
    )

    rng = np.random.default_rng(42)

    result = refill_board_state(
        state,
        num_colors=5,
        rng=rng,
    )

    assert result.specials[1, 1] == SpecialType.ROCKET_HORIZONTAL
    assert result.specials[0, 1] == SpecialType.NONE


def test_board_mechanics_do_not_mutate_input():
    state = make_state(
        [
            [1, 0, 3],
            [2, 3, 1],
            [3, 1, 2],
        ],
        [
            [0, 0, 0],
            [0, SpecialType.BOMB, 0],
            [0, 0, 0],
        ],
    )

    original_tiles = state.tiles.copy()
    original_specials = state.specials.copy()

    apply_board_gravity(state)
    refill_board_state(
        state,
        num_colors=5,
        rng=np.random.default_rng(42),
    )
    clear_board_matches(state, {(1, 1)})

    assert np.array_equal(state.tiles, original_tiles)
    assert np.array_equal(state.specials, original_specials)
