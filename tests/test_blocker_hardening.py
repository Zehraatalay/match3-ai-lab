import numpy as np
import pytest

from match3_ai.games.match3.animation_events import AnimationEventType
from match3_ai.games.match3.blockers import (
    BlockerDamageEvent,
    BlockerType,
    DamageSource,
)
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.board_mechanics import apply_blocker_damage
from match3_ai.games.match3.engine import Match3Engine, build_match_damage_events
from match3_ai.games.match3.pattern_detection import detect_patterns
from match3_ai.games.match3.specials import SpecialType


def make_board(
    rows: int,
    cols: int,
    blockers: np.ndarray | None = None,
    blocker_hp: np.ndarray | None = None,
    specials: np.ndarray | None = None,
) -> BoardState:
    tiles = np.fromfunction(
        lambda row, col: (row + 2 * col) % 3 + 1,
        (rows, cols),
        dtype=int,
    ).astype(np.uint8)
    blockers = (
        np.zeros((rows, cols), dtype=np.uint8)
        if blockers is None
        else blockers
    )
    blocker_hp = (
        np.zeros((rows, cols), dtype=np.uint8)
        if blocker_hp is None
        else blocker_hp
    )
    specials = (
        np.zeros((rows, cols), dtype=np.uint8)
        if specials is None
        else specials
    )
    tiles[np.isin(blockers, [BlockerType.CRATE, BlockerType.STONE])] = 0
    return BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones((rows, cols), dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )


@pytest.mark.parametrize(
    ("initial_hp", "expected_transitions"),
    [
        (1, [(1, 0)]),
        (2, [(2, 1), (1, 0)]),
    ],
)
def test_rocket_bomb_destroys_crate_before_same_move_gravity_and_refill(
    initial_hp,
    expected_transitions,
):
    specials = np.zeros((5, 5), dtype=np.uint8)
    specials[2, 1] = SpecialType.ROCKET_HORIZONTAL
    specials[2, 2] = SpecialType.BOMB
    blockers = np.zeros_like(specials)
    blocker_hp = np.zeros_like(specials)
    blockers[1, 1] = BlockerType.CRATE
    blocker_hp[1, 1] = initial_hp
    state = make_board(5, 5, blockers, blocker_hp, specials)

    result = Match3Engine(num_colors=3, seed=17).step(
        state,
        ((2, 1), (2, 2)),
        record_events=True,
    )

    assert result.valid
    target_records = [
        record for record in result.blocker_damage if record.position == (1, 1)
    ]
    assert [
        (record.previous_hp, record.remaining_hp) for record in target_records
    ] == expected_transitions
    assert [record.hit_order for record in target_records] == list(
        range(len(expected_transitions))
    )
    assert target_records[-1].destroyed
    assert result.state.blockers[1, 1] == BlockerType.NONE
    assert result.state.blocker_hp[1, 1] == 0
    assert result.state.tiles[1, 1] != 0
    assert result.state.specials[1, 1] == SpecialType.NONE

    event_types = [event.event_type for event in result.animation_events]
    damage_index = event_types.index(AnimationEventType.BLOCKER_DAMAGE)
    clear_index = event_types.index(AnimationEventType.CLEAR, damage_index)
    gravity_index = event_types.index(AnimationEventType.GRAVITY, clear_index)
    refill_index = event_types.index(AnimationEventType.REFILL, gravity_index)
    damage_event = result.animation_events[damage_index]
    clear_event = result.animation_events[clear_index]
    gravity_event = result.animation_events[gravity_index]
    refill_event = result.animation_events[refill_index]

    assert damage_event.blocker_damage == result.blocker_damage[
        : len(damage_event.blocker_damage)
    ]
    assert damage_event.state.blockers[1, 1] == BlockerType.NONE
    assert damage_event.state.blocker_hp[1, 1] == 0
    assert clear_event.state.blockers[1, 1] == BlockerType.NONE
    assert clear_event.state.tiles[1, 1] == 0
    assert gravity_event.state.blockers[1, 1] == BlockerType.NONE
    assert refill_event.state.tiles[1, 1] != 0
    assert np.all(result.state.tiles > 0)
    assert detect_patterns(result.state.tiles) == []

    for event in (damage_event, clear_event, gravity_event, refill_event):
        assert not np.shares_memory(event.state.tiles, result.state.tiles)
        assert not np.shares_memory(event.state.blockers, result.state.blockers)
        assert not np.shares_memory(event.state.blocker_hp, result.state.blocker_hp)


@pytest.mark.parametrize(
    ("group_cells", "blocker_position"),
    [
        (frozenset({(0, 1), (0, 2), (0, 3)}), (1, 1)),
        (frozenset({(0, 0), (0, 1), (0, 2)}), (1, 0)),
    ],
    ids=["edge", "corner"],
)
def test_edge_and_corner_match_groups_damage_adjacent_crate_once(
    group_cells,
    blocker_position,
):
    events = build_match_damage_events((group_cells,), 5, 5)
    blockers = np.zeros((5, 5), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[blocker_position] = BlockerType.CRATE
    blocker_hp[blocker_position] = 3
    state = make_board(5, 5, blockers, blocker_hp)

    damaged, records = apply_blocker_damage(state, events)

    assert len(records) == 1
    assert records[0].position == blocker_position
    assert records[0].sources == frozenset({DamageSource.ADJACENT_MATCH})
    assert damaged.blocker_hp[blocker_position] == 2


def test_ice_direct_match_at_board_edge_takes_one_hit():
    group_cells = frozenset({(0, 0), (0, 1), (0, 2)})
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[0, 0] = BlockerType.ICE
    blocker_hp[0, 0] = 2
    state = make_board(3, 3, blockers, blocker_hp)

    damaged, records = apply_blocker_damage(
        state,
        build_match_damage_events((group_cells,), 3, 3),
    )

    assert [(record.position, record.previous_hp, record.remaining_hp) for record in records] == [
        ((0, 0), 2, 1)
    ]
    assert damaged.blockers[0, 0] == BlockerType.ICE


def test_stone_at_corner_is_immune_to_match_damage_without_oob_access():
    group_cells = frozenset({(0, 0), (0, 1), (1, 0)})
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[1, 1] = BlockerType.STONE
    blocker_hp[1, 1] = 2
    state = make_board(3, 3, blockers, blocker_hp)

    damaged, records = apply_blocker_damage(
        state,
        build_match_damage_events((group_cells,), 3, 3),
    )

    assert records == ()
    assert damaged is state
    assert damaged.blocker_hp[1, 1] == 2


def test_one_special_pass_damages_multiple_blockers_once_in_position_order():
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[0, 0] = BlockerType.CRATE
    blocker_hp[0, 0] = 2
    blockers[0, 1] = BlockerType.ICE
    blocker_hp[0, 1] = 1
    blockers[0, 2] = BlockerType.STONE
    blocker_hp[0, 2] = 3
    state = make_board(3, 3, blockers, blocker_hp)
    event = BlockerDamageEvent(
        sources=frozenset({DamageSource.ROCKET}),
        cells=frozenset((row, col) for row in range(3) for col in range(3)),
    )

    damaged, records = apply_blocker_damage(state, (event, event))

    assert [record.position for record in records] == [
        (0, 0),
        (0, 1),
        (0, 2),
        (0, 0),
        (0, 2),
    ]
    assert [(record.previous_hp, record.remaining_hp) for record in records] == [
        (2, 1),
        (1, 0),
        (3, 2),
        (1, 0),
        (2, 1),
    ]
    assert [record.hit_order for record in records] == list(range(5))
    assert [record.destroyed for record in records] == [False, True, False, True, False]
    assert damaged.blockers[0, 0] == BlockerType.NONE
    assert damaged.blockers[0, 1] == BlockerType.NONE
    assert damaged.blocker_hp[0, 0] == damaged.blocker_hp[0, 1] == 0
    assert damaged.blockers[0, 2] == BlockerType.STONE
    assert damaged.blocker_hp[0, 2] == 1
