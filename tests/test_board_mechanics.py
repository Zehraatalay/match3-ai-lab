import numpy as np
import pytest

from match3_ai.games.match3.blockers import BlockerType
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


@pytest.mark.parametrize("blocker_type", [BlockerType.CRATE, BlockerType.STONE])
def test_gravity_segments_around_multiple_fixed_blockers(blocker_type):
    tiles = np.array([[1], [0], [0], [2], [0], [0], [3]], dtype=np.uint8)
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[2, 0] = blocker_type
    blockers[5, 0] = blocker_type
    blocker_hp[[2, 5], 0] = 1
    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = apply_board_gravity(state)

    assert result.tiles[:, 0].tolist() == [0, 1, 0, 0, 2, 0, 3]
    assert np.array_equal(result.blockers, blockers)
    assert np.array_equal(result.blocker_hp, blocker_hp)


def test_gravity_segments_around_non_playable_cells():
    tiles = np.array([[1], [0], [0], [2], [0], [3]], dtype=np.uint8)
    cell_mask = np.ones_like(tiles, dtype=np.bool_)
    cell_mask[2, 0] = False
    tiles[2, 0] = 0
    state = BoardState(tiles=tiles, cell_mask=cell_mask)

    result = apply_board_gravity(state)

    assert result.tiles[:, 0].tolist() == [0, 1, 0, 0, 2, 3]
    assert result.tiles[2, 0] == 0


def test_gravity_moves_special_pairs_through_ice_overlay():
    tiles = np.array([[3], [0], [2], [0], [1]], dtype=np.uint8)
    specials = np.zeros_like(tiles)
    specials[0, 0] = SpecialType.BOMB
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[2, 0] = BlockerType.ICE
    blocker_hp[2, 0] = 2
    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = apply_board_gravity(state)

    assert result.tiles[:, 0].tolist() == [0, 0, 3, 2, 1]
    assert result.specials[2, 0] == SpecialType.BOMB
    assert result.blockers[2, 0] == BlockerType.ICE
    assert result.blocker_hp[2, 0] == 2


def test_refill_skips_fixed_blockers_and_fills_ice_and_open_cells():
    tiles = np.ones((3, 3), dtype=np.uint8)
    cell_mask = np.ones((3, 3), dtype=np.bool_)
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    tiles[0, 0] = 0
    tiles[0, 1] = 0
    tiles[0, 2] = 0
    tiles[1, 0] = 0
    cell_mask[2, 2] = False
    tiles[2, 2] = 0
    blockers[0, 0] = BlockerType.CRATE
    blocker_hp[0, 0] = 2
    blockers[0, 1] = BlockerType.STONE
    blocker_hp[0, 1] = 2
    blockers[0, 2] = BlockerType.ICE
    blocker_hp[0, 2] = 1
    state = BoardState(
        tiles=tiles,
        cell_mask=cell_mask,
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = refill_board_state(state, 3, np.random.default_rng(42))

    assert result.tiles[0, 0] == 0
    assert result.tiles[0, 1] == 0
    assert 1 <= result.tiles[0, 2] <= 3
    assert 1 <= result.tiles[1, 0] <= 3
    assert result.tiles[2, 2] == 0
    assert np.array_equal(result.blockers, blockers)
    assert np.array_equal(result.blocker_hp, blocker_hp)


def test_blocker_free_gravity_and_refill_remain_unchanged():
    state = make_state(
        [
            [1, 2, 3],
            [0, 0, 0],
            [3, 1, 2],
        ]
    )

    gravity_state = apply_board_gravity(state)
    assert gravity_state.tiles[:, 0].tolist() == [0, 1, 3]

    refilled_state = refill_board_state(
        gravity_state,
        num_colors=3,
        rng=np.random.default_rng(42),
    )
    assert np.all(refilled_state.tiles > 0)
