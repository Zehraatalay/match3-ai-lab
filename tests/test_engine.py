import numpy as np
import pytest

from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.engine import Match3Engine


def make_state() -> BoardState:
    tiles = np.array(
        [
            [1, 2, 1],
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


def test_valid_swap_resolves_matches():
    state = make_state()

    engine = Match3Engine(
        num_colors=3,
        seed=42,
    )

    result = engine.step(
        state,
        ((0, 1), (1, 1)),
    )

    assert result.valid is True
    assert result.cleared_tiles >= 3
    assert result.cascade_count >= 1

    assert np.all(result.state.tiles > 0)


def test_invalid_swap_does_not_change_state():
    state = make_state()

    engine = Match3Engine(
        num_colors=3,
        seed=42,
    )

    result = engine.step(
        state,
        ((0, 0), (2, 2)),
    )

    assert result.valid is False
    assert result.cleared_tiles == 0
    assert result.cascade_count == 0

    assert np.array_equal(
        result.state.tiles,
        state.tiles,
    )


def test_engine_does_not_mutate_input_state():
    state = make_state()

    original_tiles = state.tiles.copy()

    engine = Match3Engine(
        num_colors=3,
        seed=42,
    )

    engine.step(
        state,
        ((0, 1), (1, 1)),
    )

    assert np.array_equal(
        state.tiles,
        original_tiles,
    )


def test_same_seed_produces_same_result():
    state = make_state()

    engine_a = Match3Engine(
        num_colors=3,
        seed=42,
    )

    engine_b = Match3Engine(
        num_colors=3,
        seed=42,
    )

    action = ((0, 1), (1, 1))

    result_a = engine_a.step(state, action)
    result_b = engine_b.step(state, action)

    assert np.array_equal(
        result_a.state.tiles,
        result_b.state.tiles,
    )

    assert result_a.cleared_tiles == result_b.cleared_tiles
    assert result_a.cascade_count == result_b.cascade_count


@pytest.mark.parametrize(
    "num_colors",
    [0, 1, 2, 256, 3.5, True],
)
def test_invalid_num_colors(num_colors):
    with pytest.raises(ValueError):
        Match3Engine(num_colors=num_colors)


@pytest.mark.parametrize(
    "max_cascades",
    [0, -1, 2.5, True],
)
def test_invalid_max_cascades(max_cascades):
    with pytest.raises(ValueError):
        Match3Engine(
            num_colors=5,
            max_cascades=max_cascades,
        )
