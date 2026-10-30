import numpy as np
import pytest

from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.pattern_detection import detect_patterns
from match3_ai.games.match3.recovery import (
    is_dead_board,
    recover_dead_board,
    reshuffle_board,
)
from match3_ai.games.match3.specials import SpecialType


def make_dead_state() -> BoardState:
    tiles = np.array(
        [
            [1, 2, 3],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    mask = np.ones_like(tiles, dtype=np.bool_)

    return BoardState(
        tiles=tiles,
        cell_mask=mask,
    )


def test_dead_board_detection():
    state = make_dead_state()

    assert is_dead_board(state)


def test_reshuffle_produces_playable_board():
    state = make_dead_state()

    result = reshuffle_board(
        state=state,
        rng=np.random.default_rng(42),
        max_attempts=1000,
    )

    assert not is_dead_board(result)
    assert detect_patterns(result.tiles) == []


def test_reshuffle_preserves_tile_counts():
    state = make_dead_state()

    result = reshuffle_board(
        state,
        rng=np.random.default_rng(42),
    )

    original_counts = np.bincount(
        state.tiles.ravel(),
        minlength=256,
    )

    result_counts = np.bincount(
        result.tiles.ravel(),
        minlength=256,
    )

    assert np.array_equal(original_counts, result_counts)


def test_reshuffle_does_not_mutate_input():
    state = make_dead_state()

    original_tiles = state.tiles.copy()
    original_specials = state.specials.copy()
    original_mask = state.cell_mask.copy()

    reshuffle_board(
        state,
        rng=np.random.default_rng(42),
    )

    assert np.array_equal(state.tiles, original_tiles)
    assert np.array_equal(state.specials, original_specials)
    assert np.array_equal(state.cell_mask, original_mask)


def test_same_seed_produces_same_reshuffle():
    state = make_dead_state()

    result_a = reshuffle_board(
        state,
        np.random.default_rng(42),
    )

    result_b = reshuffle_board(
        state,
        np.random.default_rng(42),
    )

    assert np.array_equal(result_a.tiles, result_b.tiles)
    assert np.array_equal(result_a.specials, result_b.specials)


def test_recover_dead_board():
    state = make_dead_state()

    result = recover_dead_board(
        state,
        rng=np.random.default_rng(42),
    )

    assert not is_dead_board(result)
    assert detect_patterns(result.tiles) == []


def test_recover_playable_board_returns_copy():
    tiles = np.array(
        [
            [1, 2, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    assert not is_dead_board(state)

    result = recover_dead_board(
        state,
        rng=np.random.default_rng(42),
    )

    assert np.array_equal(result.tiles, state.tiles)
    assert np.array_equal(result.specials, state.specials)

    assert result is not state
    assert result.tiles is not state.tiles
    assert result.specials is not state.specials


def test_reshuffle_preserves_cell_mask():
    tiles = np.array(
        [
            [1, 2, 3, 1],
            [3, 0, 1, 2],
            [2, 3, 1, 2],
            [1, 2, 3, 1],
        ],
        dtype=np.uint8,
    )

    mask = tiles != 0

    state = BoardState(
        tiles=tiles,
        cell_mask=mask,
    )

    result = reshuffle_board(
        state,
        rng=np.random.default_rng(42),
    )

    assert np.array_equal(result.cell_mask, state.cell_mask)
    assert np.all(result.tiles[~result.cell_mask] == 0)
    assert np.all(result.specials[~result.cell_mask] == SpecialType.NONE)


def test_reshuffle_impossible_board_raises_error():
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )

    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    with pytest.raises(RuntimeError):
        reshuffle_board(
            state,
            rng=np.random.default_rng(42),
            max_attempts=50,
        )


@pytest.mark.parametrize(
    "max_attempts",
    [0, -1, 2.5, True],
)
def test_invalid_max_attempts(max_attempts):
    state = make_dead_state()

    with pytest.raises(ValueError):
        reshuffle_board(
            state,
            rng=np.random.default_rng(42),
            max_attempts=max_attempts,
        )


def test_reshuffle_preserves_special_tiles():
    tiles = np.array(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 1],
            [3, 4, 1, 2],
            [4, 1, 2, 3],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles)

    specials[0, 0] = SpecialType.ROCKET_HORIZONTAL
    specials[1, 1] = SpecialType.BOMB
    specials[2, 2] = SpecialType.PROPELLER

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    result = reshuffle_board(
        state,
        rng=np.random.default_rng(42),
    )

    for special_type in (
        SpecialType.ROCKET_HORIZONTAL,
        SpecialType.BOMB,
        SpecialType.PROPELLER,
    ):
        assert np.count_nonzero(result.specials == special_type) == 1

    assert detect_patterns(result.tiles) == []


def test_reshuffle_preserves_light_ball_color():
    tiles = np.array(
        [
            [1, 2, 3, 4],
            [2, 255, 4, 1],
            [3, 4, 1, 2],
            [4, 1, 2, 3],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles)
    specials[1, 1] = SpecialType.LIGHT_BALL

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    result = reshuffle_board(
        state,
        rng=np.random.default_rng(42),
    )

    positions = np.argwhere(result.specials == SpecialType.LIGHT_BALL)

    assert len(positions) == 1

    row, col = positions[0]

    assert result.tiles[row, col] == 255
    assert detect_patterns(result.tiles) == []


def test_reshuffle_keeps_special_attached_to_its_color():
    tiles = np.array(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 1],
            [3, 4, 1, 2],
            [4, 1, 2, 3],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles)
    specials[0, 1] = SpecialType.BOMB

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    result = reshuffle_board(
        state,
        rng=np.random.default_rng(42),
    )

    positions = np.argwhere(result.specials == SpecialType.BOMB)

    assert len(positions) == 1

    row, col = positions[0]

    assert result.tiles[row, col] == 2
