import numpy as np
import pytest

from match3_ai.games.match3.animation_events import AnimationEventType
from match3_ai.games.match3.blockers import (
    BlockerDamageEvent,
    BlockerType,
    DamageSource,
)
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.board_mechanics import (
    apply_blocker_damage,
    apply_board_gravity,
    refill_board_state,
)
from match3_ai.games.match3.engine import (
    Match3Engine,
    build_match_damage_events,
)
from match3_ai.games.match3.special_resolver import (
    resolve_geometric_combination,
    resolve_light_ball_light_ball_combination,
    resolve_light_ball_special_combination,
    resolve_propeller_combination,
    resolve_special_activation,
)
from match3_ai.games.match3.specials import SpecialType


def make_state(
    blockers: np.ndarray | None = None,
    blocker_hp: np.ndarray | None = None,
    specials: np.ndarray | None = None,
) -> BoardState:
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
    specials = np.zeros_like(tiles) if specials is None else specials
    if blockers is None:
        blockers = np.zeros_like(tiles)
    if blocker_hp is None:
        blocker_hp = np.zeros_like(tiles)
    tiles[
        np.isin(blockers, [BlockerType.CRATE, BlockerType.STONE])
    ] = 0
    return BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )


def test_damage_events_hit_a_blocker_once_per_event_and_destroy_at_zero():
    blockers = np.zeros((5, 5), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[2, 2] = BlockerType.STONE
    blocker_hp[2, 2] = 2
    state = make_state(blockers, blocker_hp)
    event = BlockerDamageEvent(
        sources=frozenset({DamageSource.ROCKET}),
        cells=frozenset({(2, 2), (2, 1), (2, 3)}),
    )

    damaged, records = apply_blocker_damage(state, (event, event, event))

    assert len(records) == 2
    assert [(record.previous_hp, record.remaining_hp) for record in records] == [
        (2, 1),
        (1, 0),
    ]
    assert records[-1].destroyed
    assert damaged.blockers[2, 2] == BlockerType.NONE
    assert damaged.blocker_hp[2, 2] == 0


def test_destroyed_blocker_is_open_to_gravity_and_refill():
    tiles = np.array([[1], [0], [2]], dtype=np.uint8)
    blockers = np.array([[0], [BlockerType.CRATE], [0]], dtype=np.uint8)
    blocker_hp = np.array([[0], [1], [0]], dtype=np.uint8)
    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )
    event = BlockerDamageEvent(
        sources=frozenset({DamageSource.BOMB}),
        cells=frozenset({(1, 0)}),
    )

    damaged, records = apply_blocker_damage(state, (event,))
    fallen = apply_board_gravity(damaged)
    refilled = refill_board_state(fallen, 3, np.random.default_rng(5))

    assert records[0].destroyed
    assert refilled.blockers[1, 0] == BlockerType.NONE
    assert np.all(refilled.tiles[:, 0] > 0)


def test_ordinary_match_damage_source_does_not_harm_stone():
    blockers = np.zeros((5, 5), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[2, 2] = BlockerType.STONE
    blocker_hp[2, 2] = 2
    state = make_state(blockers, blocker_hp)
    event = BlockerDamageEvent(
        sources=frozenset(
            {DamageSource.ADJACENT_MATCH, DamageSource.DIRECT_MATCH}
        ),
        cells=frozenset({(2, 2)}),
    )

    damaged, records = apply_blocker_damage(state, (event,))

    assert records == ()
    assert damaged is state
    assert damaged.blocker_hp[2, 2] == 2


def test_independent_match_groups_damage_the_same_crate_separately():
    blockers = np.zeros((5, 5), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[2, 2] = BlockerType.CRATE
    blocker_hp[2, 2] = 3
    state = make_state(blockers, blocker_hp)
    match_groups = (
        frozenset({(2, 1), (2, 0), (1, 0)}),
        frozenset({(1, 2), (0, 2), (0, 3)}),
    )
    events = build_match_damage_events(match_groups, 5, 5)

    damaged, records = apply_blocker_damage(state, events)

    assert len(records) == 2
    assert all(record.position == (2, 2) for record in records)
    assert damaged.blocker_hp[2, 2] == 1


def test_match_group_with_multiple_adjacent_cells_only_hits_crate_once():
    tiles = np.ones((3, 3), dtype=np.uint8)
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[1, 1] = BlockerType.CRATE
    blocker_hp[1, 1] = 3
    tiles[1, 1] = 0
    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )
    group = frozenset({(0, 0), (0, 1), (1, 0)})

    events = build_match_damage_events((group,), 3, 3)
    damaged, records = apply_blocker_damage(state, events)

    adjacent_events = [
        event
        for event in events
        if DamageSource.ADJACENT_MATCH in event.sources
    ]
    assert len(adjacent_events) == 1
    assert (1, 1) in adjacent_events[0].cells
    assert len(records) == 1
    assert damaged.blocker_hp[1, 1] == 2


@pytest.mark.parametrize(
    ("special_type", "target_color"),
    [
        (SpecialType.ROCKET_HORIZONTAL, None),
        (SpecialType.ROCKET_VERTICAL, None),
        (SpecialType.BOMB, None),
        (SpecialType.PROPELLER, None),
        (SpecialType.LIGHT_BALL, 2),
    ],
)
def test_each_special_type_emits_an_independent_blocker_hit(
    special_type: SpecialType,
    target_color: int | None,
):
    blockers = np.zeros((5, 5), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    target_position = (
        (1, 2)
        if special_type == SpecialType.ROCKET_VERTICAL
        else (2, 1)
    )
    blockers[target_position] = BlockerType.ICE
    blocker_hp[target_position] = 3
    specials = np.zeros_like(blockers)
    specials[2, 2] = special_type
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 2, 2, 3, 1],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )
    if special_type == SpecialType.LIGHT_BALL:
        tiles[2, 2] = 255
        tiles[target_position] = 2
    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = resolve_special_activation(
        state,
        (2, 2),
        np.random.default_rng(17),
        target_color=target_color,
    )
    damaged, records = apply_blocker_damage(state, result.damage_events)

    assert any(target_position in event.cells for event in result.damage_events)
    assert len(records) >= 1
    expected_source = {
        SpecialType.ROCKET_HORIZONTAL: DamageSource.ROCKET,
        SpecialType.ROCKET_VERTICAL: DamageSource.ROCKET,
        SpecialType.BOMB: DamageSource.BOMB,
        SpecialType.PROPELLER: DamageSource.PROPELLER,
        SpecialType.LIGHT_BALL: DamageSource.LIGHT_BALL,
    }[special_type]
    assert records[0].sources == frozenset({expected_source})
    assert damaged.blocker_hp[target_position] == 2


@pytest.mark.parametrize(
    "blocker_type",
    [BlockerType.CRATE, BlockerType.ICE, BlockerType.STONE],
)
def test_light_ball_light_ball_hits_blockers_boardwide(blocker_type):
    blockers = np.zeros((5, 5), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[0, 0] = blocker_type
    blocker_hp[0, 0] = 2
    specials = np.zeros_like(blockers)
    specials[1, 1] = SpecialType.LIGHT_BALL
    specials[1, 2] = SpecialType.LIGHT_BALL
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 255, 255, 2, 3],
            [3, 1, 2, 3, 1],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )
    if blocker_type != BlockerType.ICE:
        tiles[0, 0] = 0
    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        specials=specials,
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = resolve_light_ball_light_ball_combination(
        state,
        (1, 1),
        (1, 2),
    )
    damaged, records = apply_blocker_damage(state, result.damage_events)

    assert len(result.damage_events) == 1
    assert (0, 0) in result.damage_events[0].cells
    assert len(records) == 1
    assert damaged.blocker_hp[0, 0] == 1


def test_normal_light_ball_does_not_target_tileless_crate():
    blockers = np.zeros((5, 5), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[0, 0] = BlockerType.CRATE
    blocker_hp[0, 0] = 2
    specials = np.zeros_like(blockers)
    specials[2, 2] = SpecialType.LIGHT_BALL
    tiles = np.full((5, 5), 2, dtype=np.uint8)
    tiles[2, 2] = 255
    tiles[0, 0] = 0
    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = resolve_special_activation(
        state,
        (2, 2),
        np.random.default_rng(4),
        target_color=2,
    )
    damaged, records = apply_blocker_damage(state, result.damage_events)

    assert all((0, 0) not in event.cells for event in result.damage_events)
    assert records == ()
    assert damaged.blocker_hp[0, 0] == 2


def test_chain_triggered_rocket_adds_a_second_hit_in_activation_order():
    specials = np.zeros((5, 5), dtype=np.uint8)
    specials[2, 2] = SpecialType.BOMB
    specials[2, 3] = SpecialType.ROCKET_VERTICAL
    blockers = np.zeros_like(specials)
    blocker_hp = np.zeros_like(specials)
    blockers[1, 3] = BlockerType.STONE
    blocker_hp[1, 3] = 3
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 0, 3],
            [3, 1, 2, 3, 1],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )
    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = resolve_special_activation(
        state,
        (2, 2),
        np.random.default_rng(3),
    )
    damaged, records = apply_blocker_damage(state, result.damage_events)
    target_records = [record for record in records if record.position == (1, 3)]

    assert result.activation_order == ((2, 2), (2, 3))
    assert [record.sources for record in target_records] == [
        frozenset({DamageSource.BOMB}),
        frozenset({DamageSource.ROCKET}),
    ]
    assert [record.hit_order for record in target_records] == [0, 1]
    assert damaged.blocker_hp[1, 3] == 1


def test_rocket_rocket_keeps_two_independent_line_passes():
    specials = np.zeros((5, 5), dtype=np.uint8)
    specials[2, 1] = SpecialType.ROCKET_HORIZONTAL
    specials[2, 2] = SpecialType.ROCKET_VERTICAL
    blockers = np.zeros_like(specials)
    blocker_hp = np.zeros_like(specials)
    blockers[2, 2] = BlockerType.ICE
    blocker_hp[2, 2] = 3
    state = make_state(blockers, blocker_hp, specials)

    result = resolve_geometric_combination(
        state,
        (2, 1),
        (2, 2),
        np.random.default_rng(1),
    )

    center_hits = sum((2, 2) in event.cells for event in result.damage_events)
    assert center_hits == 2
    damaged, records = apply_blocker_damage(state, result.damage_events)
    assert len(records) == 2
    assert [record.hit_order for record in records] == [0, 1]
    assert damaged.blocker_hp[2, 2] == 1


@pytest.mark.parametrize(
    ("special_a", "special_b", "expected_passes"),
    [
        (
            SpecialType.ROCKET_HORIZONTAL,
            SpecialType.BOMB,
            6,
        ),
        (SpecialType.BOMB, SpecialType.BOMB, 1),
    ],
)
def test_geometric_combinations_keep_their_damage_passes(
    special_a: SpecialType,
    special_b: SpecialType,
    expected_passes: int,
):
    specials = np.zeros((5, 5), dtype=np.uint8)
    specials[2, 1] = special_a
    specials[2, 2] = special_b
    state = make_state(specials=specials)

    result = resolve_geometric_combination(
        state,
        (2, 1),
        (2, 2),
        np.random.default_rng(1),
    )

    assert len(result.damage_events) == expected_passes


@pytest.mark.parametrize(
    ("special_a", "special_b", "expected_hits"),
    [
        (SpecialType.ROCKET_HORIZONTAL, SpecialType.ROCKET_VERTICAL, 1),
        (SpecialType.ROCKET_HORIZONTAL, SpecialType.BOMB, 2),
        (SpecialType.BOMB, SpecialType.BOMB, 1),
    ],
)
@pytest.mark.parametrize(
    "blocker_type",
    [BlockerType.CRATE, BlockerType.ICE, BlockerType.STONE],
)
def test_geometric_combinations_damage_each_blocker_type_per_pass(
    special_a,
    special_b,
    expected_hits,
    blocker_type,
):
    specials = np.zeros((5, 5), dtype=np.uint8)
    specials[2, 1] = special_a
    specials[2, 2] = special_b
    blockers = np.zeros_like(specials)
    blocker_hp = np.zeros_like(specials)
    blockers[1, 2] = blocker_type
    blocker_hp[1, 2] = 10
    state = make_state(blockers, blocker_hp, specials)

    result = resolve_geometric_combination(
        state,
        (2, 1),
        (2, 2),
        np.random.default_rng(43),
    )
    damaged, records = apply_blocker_damage(state, result.damage_events)
    target_records = [record for record in records if record.position == (1, 2)]

    assert len(target_records) == expected_hits
    assert damaged.blocker_hp[1, 2] == 10 - expected_hits


def test_rocket_bomb_overlapping_line_passes_both_damage_the_blocker():
    specials = np.zeros((5, 5), dtype=np.uint8)
    specials[2, 1] = SpecialType.ROCKET_HORIZONTAL
    specials[2, 2] = SpecialType.BOMB
    blockers = np.zeros_like(specials)
    blocker_hp = np.zeros_like(specials)
    blockers[2, 2] = BlockerType.ICE
    blocker_hp[2, 2] = 3
    state = make_state(blockers, blocker_hp, specials)

    result = resolve_geometric_combination(
        state,
        (2, 1),
        (2, 2),
        np.random.default_rng(1),
    )
    damaged, records = apply_blocker_damage(state, result.damage_events)

    assert sum((2, 2) in event.cells for event in result.damage_events) == 2
    assert len(records) == 2
    assert damaged.blocker_hp[2, 2] == 1


def test_propeller_rocket_keeps_each_target_cross_pass_separate():
    specials = np.zeros((5, 5), dtype=np.uint8)
    specials[2, 1] = SpecialType.PROPELLER
    specials[2, 2] = SpecialType.ROCKET_HORIZONTAL
    state = make_state(specials=specials)

    result = resolve_propeller_combination(
        state,
        (2, 1),
        (2, 2),
        np.random.default_rng(23),
    )

    assert len(result.damage_events) == 5


@pytest.mark.parametrize(
    ("special_a", "special_b"),
    [
        (SpecialType.PROPELLER, SpecialType.PROPELLER),
        (SpecialType.PROPELLER, SpecialType.ROCKET_HORIZONTAL),
        (SpecialType.PROPELLER, SpecialType.BOMB),
    ],
)
@pytest.mark.parametrize(
    "blocker_type",
    [BlockerType.CRATE, BlockerType.ICE, BlockerType.STONE],
)
def test_propeller_combinations_apply_deterministic_hits_to_blockers(
    special_a,
    special_b,
    blocker_type,
):
    specials = np.zeros((5, 5), dtype=np.uint8)
    specials[2, 1] = special_a
    specials[2, 2] = special_b
    blockers = np.zeros_like(specials)
    blocker_hp = np.zeros_like(specials)
    blockers[1, 2] = blocker_type
    blocker_hp[1, 2] = 10
    state = make_state(blockers, blocker_hp, specials)
    result_a = resolve_propeller_combination(
        state,
        (2, 1),
        (2, 2),
        np.random.default_rng(29),
    )
    result_b = resolve_propeller_combination(
        state,
        (2, 1),
        (2, 2),
        np.random.default_rng(29),
    )
    damaged, records = apply_blocker_damage(state, result_a.damage_events)
    expected_hits = sum((1, 2) in event.cells for event in result_a.damage_events)

    assert result_a.damage_events == result_b.damage_events
    assert expected_hits >= 1
    assert len(records) == expected_hits
    assert damaged.blocker_hp[1, 2] == 10 - expected_hits


@pytest.mark.parametrize(
    "partner_type",
    [
        SpecialType.ROCKET_HORIZONTAL,
        SpecialType.BOMB,
        SpecialType.PROPELLER,
    ],
)
def test_light_ball_transformations_damage_ice_per_triggered_activation(
    partner_type,
):
    tiles = np.full((5, 5), 2, dtype=np.uint8)
    tiles[2, 1] = 255
    tiles[2, 2] = 1
    tiles[2, 3] = 1
    specials = np.zeros_like(tiles)
    specials[2, 1] = SpecialType.LIGHT_BALL
    specials[2, 2] = partner_type
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[2, 3] = BlockerType.ICE
    blocker_hp[2, 3] = 4
    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result_a = resolve_light_ball_special_combination(
        state,
        (2, 1),
        (2, 2),
        np.random.default_rng(37),
    )
    result_b = resolve_light_ball_special_combination(
        state,
        (2, 1),
        (2, 2),
        np.random.default_rng(37),
    )
    damaged, records = apply_blocker_damage(state, result_a.damage_events)

    assert len(result_a.transformations) == 1
    assert result_a.damage_events == result_b.damage_events
    assert sum((2, 3) in event.cells for event in result_a.damage_events) == 1
    assert len(records) == 1
    assert records[0].blocker_type == BlockerType.ICE
    assert damaged.blocker_hp[2, 3] == 3


def test_light_ball_bomb_transformations_keep_overlapping_activations_separate():
    tiles = np.full((5, 5), 2, dtype=np.uint8)
    tiles[2, 1] = 255
    tiles[2, 2] = 1
    tiles[1, 2] = 1
    tiles[3, 2] = 1
    specials = np.zeros_like(tiles)
    specials[2, 1] = SpecialType.LIGHT_BALL
    specials[2, 2] = SpecialType.BOMB
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[2, 2] = BlockerType.ICE
    blocker_hp[2, 2] = 4
    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = resolve_light_ball_special_combination(
        state,
        (2, 1),
        (2, 2),
        np.random.default_rng(41),
    )
    damaged, records = apply_blocker_damage(state, result.damage_events)
    center_hits = [record for record in records if record.position == (2, 2)]

    assert len(result.transformations) == 2
    assert len(center_hits) == 3
    assert [record.hit_order for record in center_hits] == [0, 1, 2]
    assert damaged.blocker_hp[2, 2] == 1


def test_special_swap_applies_blocker_damage_before_gravity():
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
    specials = np.zeros_like(tiles)
    specials[2, 1] = SpecialType.ROCKET_HORIZONTAL
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[2, 0] = BlockerType.CRATE
    blocker_hp[2, 0] = 20
    tiles[2, 0] = 0
    state = BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = Match3Engine(num_colors=3, seed=21).step(
        state,
        ((2, 1), (2, 2)),
    )

    assert result.valid
    assert result.state.blockers[2, 0] == BlockerType.CRATE
    assert result.state.blocker_hp[2, 0] == 20 - len(result.blocker_damage)
    assert result.blocker_damage[0].previous_hp == 20
    assert result.blocker_damage[0].sources == frozenset({DamageSource.ROCKET})


def test_combination_animation_records_preserve_damage_hit_order():
    specials = np.zeros((5, 5), dtype=np.uint8)
    specials[2, 1] = SpecialType.ROCKET_HORIZONTAL
    specials[2, 2] = SpecialType.ROCKET_VERTICAL
    blockers = np.zeros_like(specials)
    blocker_hp = np.zeros_like(specials)
    blockers[2, 2] = BlockerType.ICE
    blocker_hp[2, 2] = 10
    state = make_state(blockers, blocker_hp, specials)

    result = Match3Engine(num_colors=3, seed=47).step(
        state,
        ((2, 1), (2, 2)),
        record_events=True,
    )
    damage_event = next(
        event
        for event in result.animation_events
        if event.event_type == AnimationEventType.BLOCKER_DAMAGE
    )
    target_records = [
        record
        for record in damage_event.blocker_damage
        if record.position == (2, 2)
    ]
    event_types = [event.event_type for event in result.animation_events]

    assert [record.hit_order for record in target_records] == [0, 1]
    assert [record.previous_hp for record in target_records] == [10, 9]
    assert [record.remaining_hp for record in target_records] == [9, 8]
    assert event_types.index(AnimationEventType.SPECIAL_ACTIVATION) < event_types.index(
        AnimationEventType.BLOCKER_DAMAGE
    )
    assert event_types.index(AnimationEventType.BLOCKER_DAMAGE) < event_types.index(
        AnimationEventType.CLEAR
    )


def test_engine_applies_match_damage_and_records_hp_transition():
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
    blockers[1, 0] = BlockerType.CRATE
    blocker_hp[1, 0] = 20
    state = BoardState(
        tiles=np.where(blockers == BlockerType.CRATE, 0, tiles).astype(np.uint8),
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = Match3Engine(num_colors=3, seed=42).step(
        state,
        ((0, 1), (1, 1)),
        record_events=True,
    )

    assert result.valid
    assert any(record.position == (1, 0) for record in result.blocker_damage)
    assert [record.hit_order for record in result.blocker_damage] == list(
        range(len(result.blocker_damage))
    )
    assert result.blocker_damage[0].sources == frozenset(
        {DamageSource.ADJACENT_MATCH}
    )
    assert result.state.blocker_hp[1, 0] == 20 - len(result.blocker_damage)
    assert result.blocker_damage[0].previous_hp == 20
    damage_events = [
        event
        for event in result.animation_events
        if event.event_type == AnimationEventType.BLOCKER_DAMAGE
    ]
    assert damage_events
    assert damage_events[0].blocker_damage[0].previous_hp == 20
    assert damage_events[0].blocker_damage[0].hit_order == 0


def test_adjacent_match_does_not_damage_ice():
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
    blockers[1, 0] = BlockerType.ICE
    blocker_hp[1, 0] = 20
    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = Match3Engine(num_colors=3, seed=42).step(
        state,
        ((0, 1), (1, 1)),
    )

    assert result.valid
    assert all(record.position != (1, 0) for record in result.blocker_damage)
    assert result.state.blocker_hp[1, 0] == 20


def test_blocker_free_engine_step_has_no_damage_records():
    tiles = np.array(
        [
            [1, 2, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )
    state = BoardState(tiles, np.ones_like(tiles, dtype=np.bool_))

    result = Match3Engine(num_colors=3, seed=42).step(
        state,
        ((0, 1), (1, 1)),
    )

    assert result.valid
    assert result.blocker_damage == ()


def test_cascade_match_groups_damage_ice_in_later_phases():
    tiles = np.array(
        [
            [4, 4, 1, 2, 4],
            [3, 1, 3, 2, 3],
            [2, 4, 3, 3, 2],
            [4, 4, 1, 4, 1],
            [4, 3, 4, 1, 4],
        ],
        dtype=np.uint8,
    )
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[1, 1] = BlockerType.ICE
    blocker_hp[1, 1] = 20
    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = Match3Engine(num_colors=4, seed=31).step(
        state,
        ((0, 1), (1, 1)),
    )
    ice_hits = [record for record in result.blocker_damage if record.position == (1, 1)]

    assert result.cascade_count > 1
    assert len(ice_hits) > 1
    assert [record.hit_order for record in result.blocker_damage] == list(
        range(len(result.blocker_damage))
    )
    assert result.state.blocker_hp[1, 1] == 20 - len(ice_hits)


def test_direct_match_damages_ice_under_the_matched_tile():
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
    blockers[0, 0] = BlockerType.ICE
    blocker_hp[0, 0] = 20
    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    result = Match3Engine(num_colors=3, seed=42).step(
        state,
        ((0, 1), (1, 1)),
    )

    ice_hits = [record for record in result.blocker_damage if record.position == (0, 0)]
    assert ice_hits
    assert ice_hits[0].sources == frozenset({DamageSource.DIRECT_MATCH})
    assert result.state.blocker_hp[0, 0] == 20 - len(ice_hits)
