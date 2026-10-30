import numpy as np
import pytest

from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.special_activation import (
    bomb_cells,
    get_geometric_combination_cells,
    light_ball_cells,
    light_ball_combination_transformations,
    light_ball_light_ball_cells,
    propeller_cells,
    propeller_combination_cells,
    propeller_launch_cells,
    rocket_bomb_cells,
    rocket_cells,
    rocket_rocket_cells,
    select_propeller_target,
)
from match3_ai.games.match3.special_resolver import (
    resolve_geometric_combination,
    resolve_light_ball_special_combination,
    resolve_matched_specials,
    resolve_propeller_combination,
    resolve_special_activation,
    resolve_special_swap,
)
from match3_ai.games.match3.specials import SpecialType


def make_state() -> BoardState:
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 1],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )

    return BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )


def test_horizontal_rocket_cells():
    state = make_state()

    affected = rocket_cells(
        state,
        position=(2, 2),
        horizontal=True,
    )

    assert affected == {
        (2, 0),
        (2, 1),
        (2, 2),
        (2, 3),
        (2, 4),
    }


def test_vertical_rocket_cells():
    state = make_state()

    affected = rocket_cells(
        state,
        position=(2, 2),
        horizontal=False,
    )

    assert affected == {
        (0, 2),
        (1, 2),
        (2, 2),
        (3, 2),
        (4, 2),
    }


def test_bomb_clears_3x3_area():
    state = make_state()

    affected = bomb_cells(
        state,
        position=(2, 2),
    )

    assert affected == {
        (1, 1),
        (1, 2),
        (1, 3),
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
    }


def test_bomb_clears_5x5_area():
    state = make_state()

    affected = bomb_cells(
        state,
        position=(2, 2),
        radius=2,
    )

    assert len(affected) == 25


def test_bomb_at_corner():
    state = make_state()

    affected = bomb_cells(
        state,
        position=(0, 0),
    )

    assert affected == {
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
    }


def test_propeller_launch_clears_only_orthogonal_neighbors():
    state = make_state()

    affected = propeller_launch_cells(
        state,
        position=(2, 2),
    )

    assert affected == {
        (2, 2),
        (1, 2),
        (3, 2),
        (2, 1),
        (2, 3),
    }

    assert (1, 1) not in affected
    assert (1, 3) not in affected
    assert (3, 1) not in affected
    assert (3, 3) not in affected


def test_propeller_selects_target_outside_launch_area():
    state = make_state()

    rng = np.random.default_rng(42)

    launch_area = propeller_launch_cells(
        state,
        position=(2, 2),
    )

    target = select_propeller_target(
        state,
        rng=rng,
        excluded_positions=launch_area,
    )

    assert target is not None
    assert target not in launch_area
    assert state.cell_mask[target]


def test_propeller_effect_includes_launch_and_target():
    state = make_state()

    rng = np.random.default_rng(42)

    affected = propeller_cells(
        state,
        position=(2, 2),
        rng=rng,
    )

    launch_area = propeller_launch_cells(
        state,
        position=(2, 2),
    )

    assert launch_area.issubset(affected)
    assert len(affected) == 6


def test_propeller_target_is_reproducible():
    state = make_state()

    affected_a = propeller_cells(
        state,
        position=(2, 2),
        rng=np.random.default_rng(42),
    )

    affected_b = propeller_cells(
        state,
        position=(2, 2),
        rng=np.random.default_rng(42),
    )

    assert affected_a == affected_b


def test_light_ball_clears_target_color():
    state = make_state()

    affected = light_ball_cells(
        state,
        target_color=1,
    )

    expected = {
        (row, col)
        for row in range(state.rows)
        for col in range(state.cols)
        if state.tiles[row, col] == 1
    }

    assert affected == expected
    assert len(affected) == 8


def test_special_effects_do_not_mutate_board():
    state = make_state()

    original_tiles = state.tiles.copy()
    original_specials = state.specials.copy()

    rocket_cells(state, (2, 2), horizontal=True)
    bomb_cells(state, (2, 2))
    propeller_cells(
        state,
        (2, 2),
        rng=np.random.default_rng(42),
    )
    light_ball_cells(state, target_color=1)

    assert np.array_equal(state.tiles, original_tiles)
    assert np.array_equal(state.specials, original_specials)


def test_resolver_activates_horizontal_rocket():
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 1],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles)
    specials[1, 1] = SpecialType.ROCKET_HORIZONTAL

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    result = resolve_special_activation(
        state,
        position=(1, 1),
        rng=np.random.default_rng(42),
    )

    assert result.affected_cells == frozenset(
        {
            (1, 0),
            (1, 1),
            (1, 2),
            (1, 3),
            (1, 4),
        }
    )

    assert result.activated_positions == frozenset(
        {
            (1, 1),
        }
    )


def test_rocket_triggers_bomb_chain_reaction():
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 1],
            [1, 2, 3, 1, 2],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles)

    specials[1, 1] = SpecialType.ROCKET_HORIZONTAL
    specials[1, 3] = SpecialType.BOMB

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    result = resolve_special_activation(
        state,
        position=(1, 1),
        rng=np.random.default_rng(42),
    )

    expected_rocket_cells = {
        (1, 0),
        (1, 1),
        (1, 2),
        (1, 3),
        (1, 4),
    }

    expected_bomb_cells = {
        (0, 2),
        (0, 3),
        (0, 4),
        (1, 2),
        (1, 3),
        (1, 4),
        (2, 2),
        (2, 3),
        (2, 4),
    }

    assert result.affected_cells == frozenset(
        expected_rocket_cells | expected_bomb_cells
    )

    assert result.activated_positions == frozenset(
        {
            (1, 1),
            (1, 3),
        }
    )


def test_chain_reaction_does_not_activate_same_special_twice():
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 3, 1],
            [3, 1, 2],
        ],
        dtype=np.uint8,
    )

    specials = np.zeros_like(tiles)

    specials[1, 0] = SpecialType.ROCKET_HORIZONTAL
    specials[1, 2] = SpecialType.ROCKET_VERTICAL
    specials[0, 2] = SpecialType.BOMB

    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=bool),
    )

    result = resolve_special_activation(
        state,
        position=(1, 0),
        rng=np.random.default_rng(42),
    )

    assert result.activated_positions == frozenset(
        {
            (1, 0),
            (1, 2),
            (0, 2),
        }
    )


def test_resolver_activates_light_ball_with_target_color():
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 255, 1],
            [3, 1, 2],
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

    result = resolve_special_activation(
        state,
        position=(1, 1),
        rng=np.random.default_rng(42),
        target_color=1,
    )

    assert result.affected_cells == frozenset(
        {
            (0, 0),
            (1, 1),
            (1, 2),
            (2, 1),
        }
    )


def test_resolver_rejects_normal_tile():
    state = make_state()

    with pytest.raises(
        ValueError,
        match="does not contain a special",
    ):
        resolve_special_activation(
            state,
            position=(1, 1),
            rng=np.random.default_rng(42),
        )


def test_resolver_requires_target_color_for_initial_light_ball():
    tiles = np.array(
        [
            [1, 2, 3],
            [2, 255, 1],
            [3, 1, 2],
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

    with pytest.raises(
        ValueError,
        match="requires a target color",
    ):
        resolve_special_activation(
            state,
            position=(1, 1),
            rng=np.random.default_rng(42),
        )


def test_rocket_rocket_combination():
    state = make_state()

    affected = rocket_rocket_cells(
        state,
        position=(2, 2),
    )

    expected = {(2, col) for col in range(5)} | {(row, 2) for row in range(5)}

    assert affected == expected
    assert len(affected) == 9


def test_rocket_bomb_combination():
    state = make_state()

    affected = rocket_bomb_cells(
        state,
        position=(2, 2),
    )

    expected = {
        (row, col)
        for row in range(5)
        for col in range(5)
        if abs(row - 2) <= 1 or abs(col - 2) <= 1
    }

    assert affected == expected
    assert len(affected) == 21


def test_bomb_bomb_combination():
    state = make_state()

    affected = get_geometric_combination_cells(
        state,
        position=(2, 2),
        special_a=SpecialType.BOMB,
        special_b=SpecialType.BOMB,
    )

    assert len(affected) == 25


def test_rocket_bomb_combination_is_order_independent():
    state = make_state()

    result_a = get_geometric_combination_cells(
        state,
        position=(2, 2),
        special_a=SpecialType.ROCKET_HORIZONTAL,
        special_b=SpecialType.BOMB,
    )

    result_b = get_geometric_combination_cells(
        state,
        position=(2, 2),
        special_a=SpecialType.BOMB,
        special_b=SpecialType.ROCKET_HORIZONTAL,
    )

    assert result_a == result_b


def test_unsupported_geometric_combination():
    state = make_state()

    with pytest.raises(
        ValueError,
        match="unsupported geometric combination",
    ):
        get_geometric_combination_cells(
            state,
            position=(2, 2),
            special_a=SpecialType.LIGHT_BALL,
            special_b=SpecialType.BOMB,
        )


def test_propeller_propeller_combination():
    state = make_state()

    affected = propeller_combination_cells(
        state=state,
        position=(2, 2),
        special_a=SpecialType.PROPELLER,
        special_b=SpecialType.PROPELLER,
        rng=np.random.default_rng(42),
    )

    launch_area = propeller_launch_cells(
        state,
        position=(2, 2),
    )

    assert launch_area.issubset(affected)

    # five launch cells + three distinct targets.
    assert len(affected) == 8


def test_propeller_rocket_combination():
    state = make_state()

    affected = propeller_combination_cells(
        state=state,
        position=(2, 2),
        special_a=SpecialType.PROPELLER,
        special_b=SpecialType.ROCKET_HORIZONTAL,
        rng=np.random.default_rng(42),
    )

    launch_area = propeller_launch_cells(
        state,
        position=(2, 2),
    )

    assert launch_area.issubset(affected)

    # two cross effects should affect more than the launch area.
    assert len(affected) > len(launch_area)


def test_propeller_bomb_combination():
    state = make_state()

    affected = propeller_combination_cells(
        state=state,
        position=(2, 2),
        special_a=SpecialType.PROPELLER,
        special_b=SpecialType.BOMB,
        rng=np.random.default_rng(42),
    )

    launch_area = propeller_launch_cells(
        state,
        position=(2, 2),
    )

    assert launch_area.issubset(affected)
    assert len(affected) > len(launch_area)


def test_propeller_rocket_order_independent():
    state = make_state()

    affected_a = propeller_combination_cells(
        state=state,
        position=(2, 2),
        special_a=SpecialType.PROPELLER,
        special_b=SpecialType.ROCKET_HORIZONTAL,
        rng=np.random.default_rng(42),
    )

    affected_b = propeller_combination_cells(
        state=state,
        position=(2, 2),
        special_a=SpecialType.ROCKET_HORIZONTAL,
        special_b=SpecialType.PROPELLER,
        rng=np.random.default_rng(42),
    )

    assert affected_a == affected_b


def test_propeller_combination_is_reproducible():
    state = make_state()

    affected_a = propeller_combination_cells(
        state=state,
        position=(2, 2),
        special_a=SpecialType.PROPELLER,
        special_b=SpecialType.PROPELLER,
        rng=np.random.default_rng(123),
    )

    affected_b = propeller_combination_cells(
        state=state,
        position=(2, 2),
        special_a=SpecialType.PROPELLER,
        special_b=SpecialType.PROPELLER,
        rng=np.random.default_rng(123),
    )

    assert affected_a == affected_b


def test_unsupported_propeller_combination():
    state = make_state()

    with pytest.raises(
        ValueError,
        match="unsupported Propeller combination",
    ):
        propeller_combination_cells(
            state=state,
            position=(2, 2),
            special_a=SpecialType.PROPELLER,
            special_b=SpecialType.LIGHT_BALL,
            rng=np.random.default_rng(42),
        )


def test_light_ball_rocket_transformations():
    state = make_state()

    transformations = light_ball_combination_transformations(
        state=state,
        special_type=SpecialType.ROCKET_HORIZONTAL,
        target_color=1,
        rng=np.random.default_rng(42),
    )

    expected_positions = {
        (row, col)
        for row in range(state.rows)
        for col in range(state.cols)
        if state.tiles[row, col] == 1
    }

    assert {
        transformation.position for transformation in transformations
    } == expected_positions

    assert all(
        transformation.special_type
        in (
            SpecialType.ROCKET_HORIZONTAL,
            SpecialType.ROCKET_VERTICAL,
        )
        for transformation in transformations
    )


def test_light_ball_rocket_directions_are_reproducible():
    state = make_state()

    transformations_a = light_ball_combination_transformations(
        state=state,
        special_type=SpecialType.ROCKET_HORIZONTAL,
        target_color=1,
        rng=np.random.default_rng(123),
    )

    transformations_b = light_ball_combination_transformations(
        state=state,
        special_type=SpecialType.ROCKET_HORIZONTAL,
        target_color=1,
        rng=np.random.default_rng(123),
    )

    assert transformations_a == transformations_b


def test_light_ball_bomb_transformations():
    state = make_state()

    transformations = light_ball_combination_transformations(
        state=state,
        special_type=SpecialType.BOMB,
        target_color=2,
        rng=np.random.default_rng(42),
    )

    assert len(transformations) == int(np.count_nonzero(state.tiles == 2))

    assert all(
        transformation.special_type == SpecialType.BOMB
        for transformation in transformations
    )


def test_light_ball_propeller_transformations():
    state = make_state()

    transformations = light_ball_combination_transformations(
        state=state,
        special_type=SpecialType.PROPELLER,
        target_color=3,
        rng=np.random.default_rng(42),
    )

    assert len(transformations) == int(np.count_nonzero(state.tiles == 3))

    assert all(
        transformation.special_type == SpecialType.PROPELLER
        for transformation in transformations
    )


def test_light_ball_transformations_exclude_consumed_positions():
    state = make_state()

    consumed = {
        (0, 0),
        (1, 2),
    }

    transformations = light_ball_combination_transformations(
        state=state,
        special_type=SpecialType.BOMB,
        target_color=1,
        rng=np.random.default_rng(42),
        consumed_positions=consumed,
    )

    positions = {transformation.position for transformation in transformations}

    assert positions.isdisjoint(consumed)


def test_light_ball_light_ball_clears_entire_board():
    state = make_state()

    affected = light_ball_light_ball_cells(state)

    assert len(affected) == 25


@pytest.mark.parametrize(
    "special_type",
    [
        SpecialType.NONE,
        SpecialType.LIGHT_BALL,
    ],
)
def test_light_ball_transformations_reject_invalid_type(special_type):
    state = make_state()

    with pytest.raises(
        ValueError,
        match="unsupported Light Ball combination",
    ):
        light_ball_combination_transformations(
            state=state,
            special_type=special_type,
            target_color=1,
            rng=np.random.default_rng(42),
        )


def test_light_ball_transformations_do_not_mutate_board():
    state = make_state()

    original_tiles = state.tiles.copy()
    original_specials = state.specials.copy()

    light_ball_combination_transformations(
        state=state,
        special_type=SpecialType.ROCKET_HORIZONTAL,
        target_color=1,
        rng=np.random.default_rng(42),
    )

    assert np.array_equal(state.tiles, original_tiles)
    assert np.array_equal(state.specials, original_specials)


def test_resolver_rocket_rocket_combination():
    state = make_state()

    state.specials[2, 1] = SpecialType.ROCKET_HORIZONTAL
    state.specials[2, 2] = SpecialType.ROCKET_VERTICAL

    result = resolve_geometric_combination(
        state=state,
        source=(2, 1),
        destination=(2, 2),
        rng=np.random.default_rng(42),
    )

    assert len(result.affected_cells) == 9

    assert result.activated_positions == frozenset(
        {
            (2, 1),
            (2, 2),
        }
    )


def test_resolver_rocket_bomb_combination():
    state = make_state()

    state.specials[2, 1] = SpecialType.ROCKET_HORIZONTAL
    state.specials[2, 2] = SpecialType.BOMB

    result = resolve_geometric_combination(
        state=state,
        source=(2, 1),
        destination=(2, 2),
        rng=np.random.default_rng(42),
    )

    assert len(result.affected_cells) == 21


def test_resolver_bomb_bomb_combination():
    state = make_state()

    state.specials[2, 1] = SpecialType.BOMB
    state.specials[2, 2] = SpecialType.BOMB

    result = resolve_geometric_combination(
        state=state,
        source=(2, 1),
        destination=(2, 2),
        rng=np.random.default_rng(42),
    )

    assert len(result.affected_cells) == 25


def test_geometric_combination_triggers_chain_reaction():
    state = make_state()

    state.specials[2, 1] = SpecialType.ROCKET_HORIZONTAL
    state.specials[2, 2] = SpecialType.ROCKET_VERTICAL

    # a Bomb outside the initial cross but reachable
    # through the chain reaction.
    state.specials[0, 2] = SpecialType.BOMB

    result = resolve_geometric_combination(
        state=state,
        source=(2, 1),
        destination=(2, 2),
        rng=np.random.default_rng(42),
    )

    assert (0, 2) in result.activated_positions

    # the Bomb at (0, 2) should also affect (0, 1).
    assert (0, 1) in result.affected_cells


def test_combination_partners_do_not_activate_individually():
    state = make_state()

    state.specials[2, 1] = SpecialType.ROCKET_HORIZONTAL
    state.specials[2, 2] = SpecialType.BOMB

    result = resolve_geometric_combination(
        state=state,
        source=(2, 1),
        destination=(2, 2),
        rng=np.random.default_rng(42),
    )

    assert result.activated_positions == frozenset(
        {
            (2, 1),
            (2, 2),
        }
    )


def test_resolver_propeller_propeller_combination():
    state = make_state()

    state.specials[2, 1] = SpecialType.PROPELLER
    state.specials[2, 2] = SpecialType.PROPELLER

    result = resolve_propeller_combination(
        state=state,
        source=(2, 1),
        destination=(2, 2),
        rng=np.random.default_rng(42),
    )

    assert (2, 1) in result.affected_cells
    assert (2, 2) in result.affected_cells

    assert result.activated_positions == frozenset(
        {
            (2, 1),
            (2, 2),
        }
    )


def test_resolver_propeller_rocket_combination():
    state = make_state()

    state.specials[2, 1] = SpecialType.PROPELLER
    state.specials[2, 2] = SpecialType.ROCKET_HORIZONTAL

    result = resolve_propeller_combination(
        state=state,
        source=(2, 1),
        destination=(2, 2),
        rng=np.random.default_rng(42),
    )

    assert (2, 1) in result.affected_cells
    assert (2, 2) in result.affected_cells

    assert result.activated_positions == frozenset(
        {
            (2, 1),
            (2, 2),
        }
    )


def test_resolver_propeller_bomb_combination():
    state = make_state()

    state.specials[2, 1] = SpecialType.PROPELLER
    state.specials[2, 2] = SpecialType.BOMB

    result = resolve_propeller_combination(
        state=state,
        source=(2, 1),
        destination=(2, 2),
        rng=np.random.default_rng(42),
    )

    assert (2, 1) in result.affected_cells
    assert (2, 2) in result.affected_cells

    assert result.activated_positions == frozenset(
        {
            (2, 1),
            (2, 2),
        }
    )


def test_resolver_propeller_combination_is_reproducible():
    state = make_state()

    state.specials[2, 1] = SpecialType.PROPELLER
    state.specials[2, 2] = SpecialType.ROCKET_HORIZONTAL

    result_a = resolve_propeller_combination(
        state=state,
        source=(2, 1),
        destination=(2, 2),
        rng=np.random.default_rng(42),
    )

    result_b = resolve_propeller_combination(
        state=state,
        source=(2, 1),
        destination=(2, 2),
        rng=np.random.default_rng(42),
    )

    assert result_a == result_b


def test_resolver_light_ball_rocket_combination():
    state = make_state()

    state.specials[2, 1] = SpecialType.LIGHT_BALL
    state.tiles[2, 1] = 255

    state.specials[2, 2] = SpecialType.ROCKET_HORIZONTAL

    original_specials = state.specials.copy()

    result = resolve_light_ball_special_combination(
        state=state,
        source=(2, 1),
        destination=(2, 2),
        rng=np.random.default_rng(42),
    )

    assert (2, 1) in result.affected_cells
    assert (2, 2) in result.affected_cells

    assert len(result.transformations) > 0

    assert np.array_equal(
        state.specials,
        original_specials,
    )


def test_resolver_light_ball_bomb_combination():
    state = make_state()

    state.specials[2, 1] = SpecialType.LIGHT_BALL
    state.tiles[2, 1] = 255

    state.specials[2, 2] = SpecialType.BOMB

    result = resolve_light_ball_special_combination(
        state=state,
        source=(2, 1),
        destination=(2, 2),
        rng=np.random.default_rng(42),
    )

    assert len(result.transformations) > 0

    assert all(
        transformation.special_type == SpecialType.BOMB
        for transformation in result.transformations
    )


def test_resolver_light_ball_propeller_combination():
    state = make_state()

    state.specials[2, 1] = SpecialType.LIGHT_BALL
    state.tiles[2, 1] = 255

    state.specials[2, 2] = SpecialType.PROPELLER

    result = resolve_light_ball_special_combination(
        state=state,
        source=(2, 1),
        destination=(2, 2),
        rng=np.random.default_rng(42),
    )

    assert len(result.transformations) > 0

    assert all(
        transformation.special_type == SpecialType.PROPELLER
        for transformation in result.transformations
    )


import pytest


@pytest.mark.parametrize(
    ("special_a", "special_b"),
    [
        (SpecialType.ROCKET_HORIZONTAL, SpecialType.BOMB),
        (SpecialType.ROCKET_HORIZONTAL, SpecialType.ROCKET_VERTICAL),
        (SpecialType.PROPELLER, SpecialType.BOMB),
        (SpecialType.PROPELLER, SpecialType.ROCKET_HORIZONTAL),
        (SpecialType.LIGHT_BALL, SpecialType.BOMB),
        (SpecialType.LIGHT_BALL, SpecialType.PROPELLER),
        (SpecialType.LIGHT_BALL, SpecialType.LIGHT_BALL),
    ],
)
def test_resolve_special_swap_combinations(
    special_a,
    special_b,
):
    state = make_state()

    source = (2, 1)
    destination = (2, 2)

    state.specials[source] = special_a
    state.specials[destination] = special_b

    if special_a == SpecialType.LIGHT_BALL:
        state.tiles[source] = 255

    if special_b == SpecialType.LIGHT_BALL:
        state.tiles[destination] = 255

    result = resolve_special_swap(
        state=state,
        source=source,
        destination=destination,
        rng=np.random.default_rng(42),
    )

    assert source in result.affected_cells
    assert destination in result.affected_cells


def test_special_swap_light_ball_with_normal_tile():
    state = make_state()

    source = (2, 1)
    destination = (2, 2)

    state.specials[source] = SpecialType.LIGHT_BALL
    state.tiles[source] = 255

    state.specials[destination] = SpecialType.NONE

    target_color = int(state.tiles[destination])

    expected_cells = light_ball_cells(
        state,
        target_color=target_color,
    )

    result = resolve_special_swap(
        state=state,
        source=source,
        destination=destination,
        rng=np.random.default_rng(42),
    )

    assert expected_cells.issubset(result.affected_cells)
    assert source in result.affected_cells
    assert source in result.activated_positions


def test_special_swap_light_ball_light_ball_no_chain_reaction():
    state = make_state()

    source = (2, 1)
    destination = (2, 2)

    state.specials[source] = SpecialType.LIGHT_BALL
    state.specials[destination] = SpecialType.LIGHT_BALL

    state.tiles[source] = 255
    state.tiles[destination] = 255

    # another special exists elsewhere on the board.
    state.specials[0, 0] = SpecialType.BOMB

    expected_cells = light_ball_light_ball_cells(state)

    result = resolve_special_swap(
        state=state,
        source=source,
        destination=destination,
        rng=np.random.default_rng(42),
    )

    assert result.affected_cells == frozenset(expected_cells)

    assert result.activated_positions == frozenset(
        {
            source,
            destination,
        }
    )

    assert result.transformations == ()


def test_special_swap_light_ball_rocket_activates_transformations():
    state = make_state()

    source = (2, 1)
    destination = (2, 2)

    state.specials[source] = SpecialType.LIGHT_BALL
    state.tiles[source] = 255

    state.specials[destination] = SpecialType.ROCKET_HORIZONTAL

    original_tiles = state.tiles.copy()
    original_specials = state.specials.copy()

    result = resolve_special_swap(
        state=state,
        source=source,
        destination=destination,
        rng=np.random.default_rng(42),
    )

    transformed_positions = {
        transformation.position for transformation in result.transformations
    }

    assert transformed_positions
    assert transformed_positions.issubset(result.activated_positions)

    assert transformed_positions.issubset(result.affected_cells)

    assert np.array_equal(state.tiles, original_tiles)
    assert np.array_equal(state.specials, original_specials)


def test_resolve_matched_specials_activates_rocket():
    state = make_state()

    rocket_position = (2, 2)

    state.specials[rocket_position] = SpecialType.ROCKET_HORIZONTAL

    matched_cells = {
        (2, 1),
        (2, 2),
        (2, 3),
    }

    result = resolve_matched_specials(
        state=state,
        matched_cells=matched_cells,
        rng=np.random.default_rng(42),
    )

    expected_row = {(2, col) for col in range(state.cols)}

    assert expected_row.issubset(result.affected_cells)

    assert rocket_position in result.activated_positions
