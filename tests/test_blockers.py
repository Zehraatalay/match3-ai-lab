import numpy as np
import pytest

from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.board import BoardState


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