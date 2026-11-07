import numpy as np
import pytest

import match3_ai.games.match3.engine as engine_module
from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.board_mechanics import (
    apply_board_gravity,
    clear_board_matches,
    place_special_creations,
    refill_board_state,
)
from match3_ai.games.match3.engine import Match3Engine
from match3_ai.games.match3.moves import is_valid_swap
from match3_ai.games.match3.recovery import reshuffle_board
from match3_ai.games.match3.special_selection import SpecialCreation
from match3_ai.games.match3.specials import SpecialType


def make_board() -> BoardState:
    return BoardState(
        tiles=np.array(
            [
                [1, 2, 3],
                [2, 3, 1],
                [3, 1, 2],
            ],
            dtype=np.uint8,
        ),
        cell_mask=np.ones(
            (3, 3),
            dtype=np.bool_,
        ),
    )


def test_board_initializes_without_blockers():
    state = make_board()

    assert np.all(state.blockers == BlockerType.NONE)
    assert np.all(state.blocker_hp == 0)


def test_crate_can_be_created():
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 0, 1],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )

    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros((3, 3), dtype=np.uint8)

    blockers[1, 1] = BlockerType.CRATE
    blocker_hp[1, 1] = 2

    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones((3, 3), dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    assert state.blockers[1, 1] == BlockerType.CRATE
    assert state.blocker_hp[1, 1] == 2
    assert state.tiles[1, 1] == 0


def test_ice_can_share_cell_with_tile():
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 3, 1],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )

    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros((3, 3), dtype=np.uint8)

    blockers[1, 1] = BlockerType.ICE
    blocker_hp[1, 1] = 1

    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones((3, 3), dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    assert state.tiles[1, 1] == 3
    assert state.blockers[1, 1] == BlockerType.ICE


def test_ice_can_overlay_empty_cell_during_resolution():
    tiles = np.zeros((3, 3), dtype=np.uint8)
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[1, 1] = BlockerType.ICE
    blocker_hp[1, 1] = 1

    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones((3, 3), dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    assert state.tiles[1, 1] == 0
    assert state.blockers[1, 1] == BlockerType.ICE
    assert state.blocker_hp[1, 1] == 1


def test_crate_cannot_contain_tile():
    tiles = np.ones((3, 3), dtype=np.uint8)

    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros((3, 3), dtype=np.uint8)

    blockers[1, 1] = BlockerType.CRATE
    blocker_hp[1, 1] = 1

    with pytest.raises(
        ValueError,
        match="CRATE cannot contain tiles",
    ):
        BoardState(
            tiles=tiles,
            cell_mask=np.ones((3, 3), dtype=np.bool_),
            blockers=blockers,
            blocker_hp=blocker_hp,
        )


def test_blocker_requires_positive_hp():
    tiles = np.zeros((3, 3), dtype=np.uint8)

    blockers = np.zeros((3, 3), dtype=np.uint8)
    blockers[1, 1] = BlockerType.CRATE

    with pytest.raises(
        ValueError,
        match="blockers must have positive HP",
    ):
        BoardState(
            tiles=tiles,
            cell_mask=np.ones((3, 3), dtype=np.bool_),
            blockers=blockers,
        )


def test_board_copy_preserves_blockers():
    tiles = np.ones((3, 3), dtype=np.uint8)
    tiles[1, 1] = 0

    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros((3, 3), dtype=np.uint8)

    blockers[1, 1] = BlockerType.STONE
    blocker_hp[1, 1] = 2

    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones((3, 3), dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    copied_state = state.copy()

    copied_state.blocker_hp[1, 1] = 1

    assert state.blocker_hp[1, 1] == 2
    assert copied_state.blocker_hp[1, 1] == 1


@pytest.mark.parametrize("blocker_type", [BlockerType.CRATE, BlockerType.STONE])
def test_fixed_blockers_cannot_participate_in_swaps(blocker_type):
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 0, 1],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[1, 1] = blocker_type
    blocker_hp[1, 1] = 1
    state = BoardState(
        tiles,
        np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    assert not is_valid_swap(state, (1, 1), (1, 0))
    assert not is_valid_swap(state, (1, 0), (1, 1))


def test_ice_allows_swap_with_underlying_tile():
    tiles = np.array(
        [
            [1, 2, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[0, 1] = BlockerType.ICE
    blocker_hp[0, 1] = 1
    iced_state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    assert is_valid_swap(iced_state, (0, 1), (1, 1))


def _state_with_ice() -> BoardState:
    tiles = np.array(
        [
            [1, 2, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[2, 0] = BlockerType.ICE
    blocker_hp[2, 0] = 1

    return BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )


def test_valid_engine_swap_preserves_blockers_and_ice_position():
    state = _state_with_ice()
    result = Match3Engine(num_colors=3, seed=42).step(
        state,
        ((0, 1), (1, 1)),
    )

    assert result.valid
    assert np.array_equal(result.state.blockers, state.blockers)
    assert np.array_equal(result.state.blocker_hp, state.blocker_hp)
    assert result.state.blockers[2, 0] == BlockerType.ICE


def test_ice_stays_at_its_coordinate_when_its_tile_is_swapped(monkeypatch):
    state = _state_with_ice()
    blockers = state.blockers.copy()
    blocker_hp = state.blocker_hp.copy()
    blockers[0, 1] = BlockerType.ICE
    blocker_hp[0, 1] = 1
    blockers[2, 0] = BlockerType.NONE
    blocker_hp[2, 0] = 0
    specials = np.zeros_like(state.tiles)
    specials[1, 1] = SpecialType.BOMB
    state = BoardState(
        tiles=state.tiles,
        cell_mask=state.cell_mask,
        specials=specials,
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    monkeypatch.setattr(
        engine_module,
        "resolve_special_swap",
        lambda **kwargs: engine_module.ActivationResult(
            affected_cells=frozenset(),
            activated_positions=frozenset(),
        ),
    )
    monkeypatch.setattr(engine_module, "detect_patterns", lambda tiles: [])
    monkeypatch.setattr(engine_module, "is_dead_board", lambda state: False)

    result = Match3Engine(num_colors=3, seed=42).step(
        state,
        ((0, 1), (1, 1)),
    )

    assert result.valid
    assert result.state.blockers[0, 1] == BlockerType.ICE
    assert result.state.blocker_hp[0, 1] == 1
    assert result.state.specials[0, 1] == SpecialType.BOMB
    assert np.array_equal(result.state.blockers, state.blockers)
    assert np.array_equal(result.state.blocker_hp, state.blocker_hp)


@pytest.mark.parametrize("blocker_type", [BlockerType.CRATE, BlockerType.STONE])
def test_invalid_engine_swap_preserves_blocker_arrays(blocker_type):
    tiles = np.array(
        [
            [1, 2, 3],
            [0, 3, 1],
            [2, 1, 2],
        ],
        dtype=np.uint8,
    )
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[1, 0] = blocker_type
    blocker_hp[1, 0] = 2
    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = Match3Engine(num_colors=3, seed=42).step(
        state,
        ((1, 0), (1, 1)),
    )

    assert not result.valid
    assert np.array_equal(result.state.blockers, state.blockers)
    assert np.array_equal(result.state.blocker_hp, state.blocker_hp)


@pytest.mark.parametrize(
    "transform",
    ["clear", "gravity", "refill", "special_creation"],
)
def test_board_transformations_preserve_blocker_arrays(transform):
    tiles = np.array(
        [
            [0, 2, 3],
            [1, 3, 1],
            [2, 1, 2],
        ],
        dtype=np.uint8,
    )
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[2, 2] = BlockerType.ICE
    blocker_hp[2, 2] = 2
    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    if transform == "clear":
        result = clear_board_matches(state, {(0, 1)})
    elif transform == "gravity":
        result = apply_board_gravity(state)
    elif transform == "refill":
        result = refill_board_state(state, 3, np.random.default_rng(42))
    else:
        result = place_special_creations(
            state,
            [SpecialCreation(SpecialType.BOMB, (0, 1))],
            state.tiles,
        )

    assert np.array_equal(result.blockers, state.blockers)
    assert np.array_equal(result.blocker_hp, state.blocker_hp)


def test_blocker_free_swaps_keep_existing_behavior():
    state = BoardState(
        tiles=np.array(
            [
                [1, 2, 1],
                [3, 1, 2],
                [2, 3, 1],
            ],
            dtype=np.uint8,
        ),
        cell_mask=np.ones((3, 3), dtype=np.bool_),
    )

    assert is_valid_swap(state, (0, 1), (1, 1))


def test_reshuffle_preserves_blocker_arrays():
    tiles = np.array(
        [
            [0, 2, 3],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[0, 0] = BlockerType.CRATE
    blocker_hp[0, 0] = 2
    blockers[1, 1] = BlockerType.ICE
    blocker_hp[1, 1] = 1
    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = reshuffle_board(state, np.random.default_rng(42))

    assert np.array_equal(result.blockers, state.blockers)
    assert np.array_equal(result.blocker_hp, state.blocker_hp)
