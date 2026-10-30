import numpy as np
import pytest

from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.constants import COLORLESS
from match3_ai.games.match3.rules import all_find_matches
from match3_ai.games.match3.specials import SpecialType


def make_board(
    specials: np.ndarray | None = None,
) -> BoardState:
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 3, 1],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )

    mask = np.ones((3, 3), dtype=np.bool_)

    return BoardState(
        tiles=tiles,
        cell_mask=mask,
        specials=specials,
    )


def test_default_specials_are_empty():
    board = make_board()

    assert board.specials.shape == board.tiles.shape
    assert board.specials.dtype == np.uint8
    assert np.all(board.specials == SpecialType.NONE)


def test_special_tile_can_be_created():
    specials = np.zeros((3, 3), dtype=np.uint8)
    specials[1, 1] = SpecialType.BOMB

    board = make_board(specials)

    assert board.specials[1, 1] == SpecialType.BOMB


def test_board_copy_preserves_specials():
    specials = np.zeros((3, 3), dtype=np.uint8)
    specials[1, 1] = SpecialType.ROCKET_HORIZONTAL

    board = make_board(specials)
    copied = board.copy()

    assert np.array_equal(board.specials, copied.specials)

    copied.specials[1, 1] = SpecialType.NONE

    assert board.specials[1, 1] == SpecialType.ROCKET_HORIZONTAL


def test_board_does_not_share_specials_memory():
    specials = np.zeros((3, 3), dtype=np.uint8)
    specials[0, 0] = SpecialType.PROPELLER

    board = make_board(specials)

    specials[0, 0] = SpecialType.NONE

    assert board.specials[0, 0] == SpecialType.PROPELLER


def test_invalid_special_shape_raises():
    specials = np.zeros((2, 2), dtype=np.uint8)

    with pytest.raises(ValueError, match="same shape"):
        make_board(specials)


def test_invalid_special_dtype_raises():
    specials = np.zeros((3, 3), dtype=np.int32)

    with pytest.raises(TypeError, match="uint8"):
        make_board(specials)


def test_unknown_special_type_raises():
    specials = np.zeros((3, 3), dtype=np.uint8)
    specials[1, 1] = 99

    with pytest.raises(ValueError, match="unknown special"):
        make_board(specials)


def test_special_on_empty_cell_raises():
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 0, 1],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )

    mask = np.ones((3, 3), dtype=np.bool_)

    specials = np.zeros((3, 3), dtype=np.uint8)
    specials[1, 1] = SpecialType.BOMB

    with pytest.raises(ValueError, match="empty cells"):
        BoardState(
            tiles=tiles,
            cell_mask=mask,
            specials=specials,
        )


def test_special_on_nonplayable_cell_raises():
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 0, 1],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )

    mask = np.ones((3, 3), dtype=np.bool_)
    mask[1, 1] = False

    specials = np.zeros((3, 3), dtype=np.uint8)
    specials[1, 1] = SpecialType.BOMB

    with pytest.raises(ValueError, match="non-playable"):
        BoardState(
            tiles=tiles,
            cell_mask=mask,
            specials=specials,
        )


def test_light_ball_is_colorless():

    tiles = np.array(
        [
            [1, 2, 3],
            [2, COLORLESS, 1],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros((3, 3), dtype=np.uint8)

    specials[1, 1] = SpecialType.LIGHT_BALL

    mask = np.ones((3, 3), dtype=np.bool_)

    board = BoardState(
        tiles=tiles,
        cell_mask=mask,
        specials=specials,
    )

    assert board.tiles[1, 1] == COLORLESS

    assert board.specials[1, 1] == SpecialType.LIGHT_BALL


def test_light_ball_with_normal_color_raises():

    specials = np.zeros((3, 3), dtype=np.uint8)

    specials[1, 1] = SpecialType.LIGHT_BALL

    with pytest.raises(ValueError, match="Light Ball"):
        make_board(specials)


def test_colorless_without_light_ball_raises():

    tiles = np.array(
        [
            [1, 2, 3],
            [2, COLORLESS, 1],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )

    mask = np.ones((3, 3), dtype=np.bool_)

    with pytest.raises(ValueError, match="Colorless"):
        BoardState(
            tiles=tiles,
            cell_mask=mask,
        )


def test_colorless_tiles_do_not_form_normal_matches():

    tiles = np.array(
        [
            [COLORLESS, COLORLESS, COLORLESS],
            [1, 2, 3],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    matches = all_find_matches(tiles)

    assert matches == set()
