import numpy as np
import pytest

from match3_ai.games.match3.generator import generate_playable_board
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.rules import all_find_matches


def test_generated_board_has_no_initial_matches():
    state = generate_playable_board(
        rows=8,
        cols=8,
        num_colors=5,
        seed=42,
    )

    assert all_find_matches(state.tiles) == set()


def test_generated_board_has_valid_moves():
    state = generate_playable_board(
        rows=8,
        cols=8,
        num_colors=5,
        seed=42,
    )

    assert len(find_valid_moves(state)) > 0


@pytest.mark.parametrize(
    "rows, cols, num_colors, seed",
    [
        (3, 3, 3, 42),
        (5, 5, 4, 123),
        (8, 8, 5, 42),
        (10, 10, 6, 999),
    ],
)
def test_generated_board_is_playable(
    rows,
    cols,
    num_colors,
    seed,
):
    state = generate_playable_board(
        rows=rows,
        cols=cols,
        num_colors=num_colors,
        seed=seed,
    )

    assert state.shape == (rows, cols)
    assert not all_find_matches(state.tiles)
    assert find_valid_moves(state)


@pytest.mark.parametrize(
    "max_attempts",
    [0, -1, 1.5, True, "10"],
)
def test_invalid_max_attempts(max_attempts):
    with pytest.raises(ValueError):
        generate_playable_board(
            rows=8,
            cols=8,
            num_colors=5,
            max_attempts=max_attempts,
        )


def test_generator_returns_correct_dtypes():
    state = generate_playable_board(
        rows=8,
        cols=8,
        num_colors=5,
        seed=42,
    )

    assert state.tiles.dtype == np.uint8
    assert state.cell_mask.dtype == np.bool_


def test_generator_creates_fully_playable_mask():
    state = generate_playable_board(
        rows=8,
        cols=8,
        num_colors=5,
        seed=42,
    )

    assert np.all(state.cell_mask)


def test_different_seeds_can_produce_different_boards():
    state_a = generate_playable_board(8, 8, 5, seed=42)
    state_b = generate_playable_board(8, 8, 5, seed=123)

    assert not np.array_equal(state_a.tiles, state_b.tiles)
