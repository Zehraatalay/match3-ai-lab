import numpy as np
import pytest

import match3_ai.games.match3.engine as engine_module
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.constants import COLORLESS, EMPTY
from match3_ai.games.match3.engine import Match3Engine
from match3_ai.games.match3.moves import is_valid_swap
from match3_ai.games.match3.pattern_detection import (
    detect_patterns,
    group_patterns,
)
from match3_ai.games.match3.special_selection import (
    select_special_creations,
)
from match3_ai.games.match3.specials import SpecialType


def make_state(rows: list[list[int]]) -> BoardState:
    tiles = np.array(rows, dtype=np.uint8)

    return BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
    )


@pytest.fixture
def capture_first_resolution(monkeypatch):
    """
    capture the board after the first match resolution,
    but before gravity and refill.
    """
    captured = {}

    original_gravity = engine_module.apply_board_gravity

    def capture_gravity(state):
        if "state" not in captured:
            captured["state"] = state.copy()

        return original_gravity(state)

    monkeypatch.setattr(
        engine_module,
        "apply_board_gravity",
        capture_gravity,
    )

    return captured


@pytest.mark.parametrize(
    (
        "name",
        "board",
        "action",
        "expected_special",
        "expected_position",
        "expected_group_size",
        "expected_color",
    ),
    [
        (
            "horizontal_rocket",
            [
                [1, 4, 3, 2, 4],
                [3, 2, 3, 4, 2],
                [1, 4, 1, 1, 2],
                [1, 1, 3, 3, 1],
                [2, 4, 1, 3, 3],
            ],
            ((2, 1), (3, 1)),
            SpecialType.ROCKET_HORIZONTAL,
            (2, 0),
            5,
            1,
        ),
        (
            "vertical_rocket",
            [
                [1, 3, 1, 3, 1],
                [2, 3, 3, 2, 1],
                [3, 2, 1, 2, 3],
                [3, 3, 4, 4, 1],
                [4, 4, 2, 1, 2],
            ],
            ((2, 0), (2, 1)),
            SpecialType.ROCKET_VERTICAL,
            (2, 1),
            6,
            3,
        ),
        (
            "bomb",
            [
                [3, 4, 2, 3, 3],
                [2, 4, 2, 3, 1],
                [2, 2, 3, 1, 3],
                [3, 3, 1, 3, 4],
                [2, 2, 3, 2, 3],
            ],
            ((2, 3), (3, 3)),
            SpecialType.BOMB,
            (0, 3),
            6,
            3,
        ),
        (
            "propeller",
            [
                [1, 3, 1, 3, 1],
                [2, 3, 3, 2, 1],
                [3, 2, 1, 2, 3],
                [3, 3, 4, 4, 1],
                [4, 4, 2, 1, 2],
            ],
            ((0, 2), (0, 3)),
            SpecialType.PROPELLER,
            (0, 1),
            4,
            3,
        ),
        (
            "light_ball",
            [
                [1, 1, 3, 4, 2],
                [1, 2, 3, 2, 1],
                [4, 2, 4, 3, 4],
                [3, 4, 3, 1, 4],
                [3, 2, 3, 2, 1],
            ],
            ((2, 2), (2, 3)),
            SpecialType.LIGHT_BALL,
            (0, 2),
            5,
            COLORLESS,
        ),
    ],
)
def test_special_creation_integration(
    name,
    board,
    action,
    expected_special,
    expected_position,
    expected_group_size,
    expected_color,
    capture_first_resolution,
):
    state = make_state(board)

    # 1. initial board must contain no unresolved patterns.
    assert detect_patterns(state.tiles) == [], name

    # 2. the swap must be valid.
    source, destination = action

    assert is_valid_swap(state, source, destination), name

    # 3. inspect the patterns produced by the swap.
    swapped_tiles = state.tiles.copy()

    swapped_tiles[source], swapped_tiles[destination] = (
        swapped_tiles[destination],
        swapped_tiles[source],
    )

    patterns = detect_patterns(swapped_tiles)

    assert patterns, name

    groups = group_patterns(swapped_tiles, patterns)

    # these scenarios intentionally contain one matched group.
    assert len(groups) == 1, name

    group = groups[0]

    assert len(group.cells) == expected_group_size, name

    # 4. verify the special-selection decision.
    creations = select_special_creations(
        groups,
        destination=destination,
    )

    assert len(creations) == 1, name

    assert creations[0].special_type == expected_special, name
    assert creations[0].position == expected_position, name

    # 5. execute the complete engine step.
    engine = Match3Engine(
        num_colors=4,
        seed=42,
    )

    result = engine.step(state, action)

    assert result.valid, name

    # 6. inspect the state immediately after special creation.
    first_resolution = capture_first_resolution["state"]

    row, col = expected_position

    assert first_resolution.specials[row, col] == expected_special, name

    assert first_resolution.tiles[row, col] == expected_color, name

    # exactly one special should be created.
    assert np.count_nonzero(first_resolution.specials != SpecialType.NONE) == 1, name

    # all other cells in the matched group must be empty.
    for cell in group.cells:
        if cell == expected_position:
            continue

        assert first_resolution.tiles[cell] == EMPTY, name
        assert first_resolution.specials[cell] == SpecialType.NONE, name

    # all cells outside the matched group must remain unchanged
    # at this point in the resolution.
    for row_index in range(state.rows):
        for col_index in range(state.cols):
            cell = (row_index, col_index)

            if cell not in group.cells:
                assert first_resolution.tiles[cell] == swapped_tiles[cell], name

    # 7. the first resolution clears group_size - 1 tiles.
    # additional cascades may increase the final count.
    assert result.cleared_tiles >= expected_group_size - 1, name
    assert result.cascade_count >= 1, name

    # 8. the final board must be fully resolved.
    assert detect_patterns(result.state.tiles) == [], name

    # the engine must preserve the input state.
    assert np.array_equal(state.tiles, np.array(board, dtype=np.uint8)), name
    assert np.all(state.specials == SpecialType.NONE), name


def test_single_swap_resolves_multiple_independent_groups(monkeypatch):
    state = make_state(
        [
            [1, 4, 4, 2],
            [2, 3, 1, 3],
            [4, 1, 3, 1],
            [4, 2, 4, 4],
        ]
    )

    action = ((1, 2), (2, 2))

    assert detect_patterns(state.tiles) == []

    source, destination = action

    assert is_valid_swap(state, source, destination)

    # inspect the patterns created by the swap.
    swapped_tiles = state.tiles.copy()

    swapped_tiles[source], swapped_tiles[destination] = (
        swapped_tiles[destination],
        swapped_tiles[source],
    )

    patterns = detect_patterns(swapped_tiles)
    groups = group_patterns(swapped_tiles, patterns)

    assert len(groups) == 2

    assert sorted(len(group.cells) for group in groups) == [3, 3]
    assert {group.color for group in groups} == {1, 3}

    # capture the first resolution before gravity.
    captured = {}
    original_gravity = engine_module.apply_board_gravity

    def capture_gravity(current_state):
        if "first_resolution" not in captured:
            captured["first_resolution"] = current_state.copy()

        return original_gravity(current_state)

    monkeypatch.setattr(
        engine_module,
        "apply_board_gravity",
        capture_gravity,
    )

    engine = Match3Engine(num_colors=4, seed=42)

    result = engine.step(state, action)

    assert result.valid

    first_resolution = captured["first_resolution"]

    # all six matched cells must be cleared.
    matched_cells = set()

    for group in groups:
        matched_cells.update(group.cells)

    assert len(matched_cells) == 6

    for row, col in matched_cells:
        assert first_resolution.tiles[row, col] == EMPTY
        assert first_resolution.specials[row, col] == SpecialType.NONE

    # neither group qualifies for special creation.
    assert np.all(first_resolution.specials == SpecialType.NONE)

    # later cascades may clear additional tiles.
    assert result.cleared_tiles >= 6
    assert result.cascade_count >= 1

    # the final board must contain no unresolved patterns.
    assert detect_patterns(result.state.tiles) == []


def test_cascade_five_match_creates_new_light_ball(monkeypatch):
    state = make_state(
        [
            [3, 4, 2, 3, 4],
            [4, 2, 3, 4, 2],
            [2, 3, 1, 2, 3],
            [1, 1, 2, 1, 1],
        ]
    )

    action = ((2, 2), (3, 2))

    assert detect_patterns(state.tiles) == []
    assert is_valid_swap(state, *action)

    captured_resolutions = []

    original_gravity = engine_module.apply_board_gravity
    original_refill = engine_module.refill_board_state

    def capture_gravity(current_state):
        # capture every resolution before gravity.
        captured_resolutions.append(current_state.copy())

        return original_gravity(current_state)

    refill_count = 0

    def controlled_refill(current_state, num_colors, rng):
        nonlocal refill_count

        refill_count += 1

        if refill_count == 1:
            # after the first resolution, four cells
            # in the top row are empty.
            tiles = current_state.tiles.copy()
            specials = current_state.specials.copy()

            empty_cells = (tiles == EMPTY) & current_state.cell_mask

            assert np.count_nonzero(empty_cells) == 4

            # fill all four empty cells with color 2.
            # the existing tile at (0, 2) is also color 2.
            tiles[empty_cells] = 2

            return BoardState(
                tiles=tiles,
                specials=specials,
                cell_mask=current_state.cell_mask,
            )

        # subsequent refills use the real implementation.
        return original_refill(
            current_state,
            num_colors=num_colors,
            rng=rng,
        )

    monkeypatch.setattr(
        engine_module,
        "apply_board_gravity",
        capture_gravity,
    )

    monkeypatch.setattr(
        engine_module,
        "refill_board_state",
        controlled_refill,
    )

    engine = Match3Engine(
        num_colors=4,
        seed=42,
    )

    result = engine.step(state, action)

    assert result.valid

    # at least two resolution rounds must occur.
    assert result.cascade_count >= 2
    assert len(captured_resolutions) >= 2

    first_resolution = captured_resolutions[0]
    second_resolution = captured_resolutions[1]

    # first resolution: a Light Ball is created.
    assert first_resolution.specials[3, 2] == SpecialType.LIGHT_BALL

    assert first_resolution.tiles[3, 2] == COLORLESS

    # second resolution: the cascade five-match creates
    # a new Light Ball at the lexicographically smallest cell.
    assert second_resolution.specials[0, 0] == SpecialType.LIGHT_BALL

    assert second_resolution.tiles[0, 0] == COLORLESS

    # the remaining four matched cells are cleared.
    assert np.all(second_resolution.tiles[0, 1:] == EMPTY)

    assert np.all(second_resolution.specials[0, 1:] == SpecialType.NONE)

    # the original Light Ball remains on the board.
    assert second_resolution.specials[3, 2] == SpecialType.LIGHT_BALL

    assert second_resolution.tiles[3, 2] == COLORLESS

    # the cascade creates a second Light Ball.
    assert np.count_nonzero(second_resolution.specials == SpecialType.LIGHT_BALL) == 2

    # only the two expected Light Balls should exist.
    assert np.count_nonzero(second_resolution.specials != SpecialType.NONE) == 2

    # first resolution clears 4 tiles.
    # second resolution clears 4 more tiles.
    # additional cascades may clear more.
    assert result.cleared_tiles >= 8

    # all remaining patterns must be resolved.
    assert detect_patterns(result.state.tiles) == []

    # all remaining patterns must be resolved.
    assert detect_patterns(result.state.tiles) == []
