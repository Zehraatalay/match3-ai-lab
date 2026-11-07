import numpy as np
import pytest

from match3_ai.games.match3.animation_events import AnimationEventType
from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.engine import Match3Engine
from match3_ai.games.match3.generator import generate_playable_board
from match3_ai.games.match3.moves import find_valid_moves, is_valid_swap
from match3_ai.games.match3.pattern_detection import detect_patterns
from match3_ai.games.match3.recovery import (
    is_dead_board,
    recover_dead_board,
)
from match3_ai.games.match3.specials import SpecialType


def assert_stable_state(state: BoardState) -> None:
    shape = state.tiles.shape
    assert state.tiles.dtype == np.uint8
    assert state.specials.dtype == np.uint8
    assert state.cell_mask.dtype == np.bool_
    assert state.blockers.dtype == np.uint8
    assert state.blocker_hp.dtype == np.uint8
    assert all(
        array.shape == shape
        for array in (
            state.specials,
            state.cell_mask,
            state.blockers,
            state.blocker_hp,
        )
    )
    assert np.all(state.tiles[~state.cell_mask] == 0)
    assert np.all(state.specials[~state.cell_mask] == SpecialType.NONE)
    assert np.all(state.blockers[~state.cell_mask] == BlockerType.NONE)
    assert np.all(state.blocker_hp[~state.cell_mask] == 0)
    assert np.all(state.specials[state.tiles == 0] == SpecialType.NONE)
    assert np.all(state.blocker_hp[state.blockers == BlockerType.NONE] == 0)
    assert np.all(state.blocker_hp[state.blockers != BlockerType.NONE] > 0)
    assert np.all(
        state.tiles[np.isin(state.blockers, [BlockerType.CRATE, BlockerType.STONE])]
        == 0
    )
    assert np.all(
        state.specials[np.isin(state.blockers, [BlockerType.CRATE, BlockerType.STONE])]
        == SpecialType.NONE
    )
    assert np.all(
        state.tiles[state.cell_mask & (state.blockers == BlockerType.ICE)] != 0
    )
    assert detect_patterns(state.tiles) == []


def assert_only_destroyed_blockers_changed(
    previous: BoardState,
    current: BoardState,
    damage_records,
) -> None:
    expected_blockers = previous.blockers.copy()
    expected_hp = previous.blocker_hp.copy()

    for record in damage_records:
        position = record.position
        assert expected_blockers[position] == record.blocker_type
        assert int(expected_hp[position]) == record.previous_hp
        assert record.remaining_hp == record.previous_hp - 1
        assert record.destroyed == (record.remaining_hp == 0)
        if record.destroyed:
            expected_blockers[position] = BlockerType.NONE
        expected_hp[position] = record.remaining_hp

    assert np.array_equal(current.blockers, expected_blockers)
    assert np.array_equal(current.blocker_hp, expected_hp)

    destroyed_positions = {
        record.position for record in damage_records if record.destroyed
    }
    changed_positions = np.argwhere(previous.blockers != current.blockers)

    for row, col in changed_positions:
        position = (int(row), int(col))
        assert position in destroyed_positions
        assert previous.blockers[position] != BlockerType.NONE
        assert current.blockers[position] == BlockerType.NONE


def test_fixed_blocker_takes_damage_across_moves_then_opens_for_refill():
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
    blockers[0, 0] = BlockerType.CRATE
    blocker_hp[0, 0] = 2
    tiles[0, 0] = 0
    state = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )
    engine = Match3Engine(num_colors=4, seed=81)
    actions = (((1, 3), (2, 3)), ((1, 1), (2, 1)))

    first_state = state
    assert is_valid_swap(first_state, *actions[0])
    first_result = engine.step(first_state, actions[0])

    assert first_result.valid
    assert [
        (record.previous_hp, record.remaining_hp)
        for record in first_result.blocker_damage
    ] == [(2, 1)]
    assert first_result.state.blockers[0, 0] == BlockerType.CRATE
    assert_stable_state(first_result.state)
    assert_only_destroyed_blockers_changed(
        first_state,
        first_result.state,
        first_result.blocker_damage,
    )

    second_state = first_result.state
    second_snapshot = second_state.copy()
    assert is_valid_swap(second_state, *actions[1])
    second_result = engine.step(second_state, actions[1])

    assert second_result.valid
    assert second_result.blocker_damage[0].destroyed
    assert second_result.blocker_damage[0].position == (0, 0)
    assert second_result.state.blockers[0, 0] == BlockerType.NONE
    assert second_result.state.blocker_hp[0, 0] == 0
    assert second_result.state.tiles[0, 0] != 0
    assert np.array_equal(second_state.tiles, second_snapshot.tiles)
    assert np.array_equal(second_state.blockers, second_snapshot.blockers)
    assert_stable_state(second_result.state)
    assert_only_destroyed_blockers_changed(
        second_state,
        second_result.state,
        second_result.blocker_damage,
    )


def test_created_special_is_activated_on_a_later_legal_move():
    tiles = np.array(
        [
            [1, 1, 2, 1, 3],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 1],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )
    state = BoardState(tiles, np.ones_like(tiles, dtype=np.bool_))
    engine = Match3Engine(num_colors=3, seed=42)
    creation_action = ((0, 2), (1, 2))

    assert is_valid_swap(state, *creation_action)
    creation_result = engine.step(state, creation_action)
    assert creation_result.valid
    assert np.count_nonzero(creation_result.state.specials) == 1
    assert_stable_state(creation_result.state)

    special_position = tuple(
        int(value) for value in np.argwhere(creation_result.state.specials != 0)[0]
    )
    activation_action = next(
        action
        for action in find_valid_moves(creation_result.state)
        if special_position in action
    )
    activation_result = engine.step(creation_result.state, activation_action)

    assert activation_result.valid
    assert activation_result.activation_result is not None
    assert activation_result.activation_result.activation_order
    assert_stable_state(activation_result.state)


def test_dead_board_recovery_preserves_blockers_mask_and_special_pairs():
    tiles = np.array(
        [
            [1, 2, 3, 4],
            [2, 3, 4, 1],
            [3, 4, 1, 2],
            [4, 1, 2, 3],
        ],
        dtype=np.uint8,
    )
    cell_mask = np.ones_like(tiles, dtype=np.bool_)
    specials = np.zeros_like(tiles)
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    specials[0, 0] = SpecialType.ROCKET_HORIZONTAL
    blockers[0, 1] = BlockerType.CRATE
    blocker_hp[0, 1] = 2
    tiles[0, 1] = 0
    blockers[1, 0] = BlockerType.STONE
    blocker_hp[1, 0] = 2
    tiles[1, 0] = 0
    cell_mask[1, 1] = False
    tiles[1, 1] = 0
    blockers[2, 2] = BlockerType.ICE
    blocker_hp[2, 2] = 2
    state = BoardState(tiles, cell_mask, specials, blockers, blocker_hp)
    assert is_dead_board(state)

    result = recover_dead_board(
        state,
        np.random.default_rng(42),
        max_attempts=1000,
    )
    result_again = recover_dead_board(
        state,
        np.random.default_rng(42),
        max_attempts=1000,
    )

    assert not is_dead_board(result)
    assert detect_patterns(result.tiles) == []
    assert np.array_equal(result.cell_mask, cell_mask)
    assert np.array_equal(result.blockers, blockers)
    assert np.array_equal(result.blocker_hp, blocker_hp)
    assert np.array_equal(result.tiles, result_again.tiles)
    assert np.array_equal(result.specials, result_again.specials)
    assert np.array_equal(result.blockers, result_again.blockers)
    assert np.array_equal(result.blocker_hp, result_again.blocker_hp)
    assert np.count_nonzero(result.specials == SpecialType.ROCKET_HORIZONTAL) == 1
    special_row, special_col = np.argwhere(
        result.specials == SpecialType.ROCKET_HORIZONTAL
    )[0]
    assert result.tiles[special_row, special_col] == 1
    assert_stable_state(result)


def test_recovery_of_fixed_blocker_only_board_is_bounded():
    tiles = np.zeros((2, 2), dtype=np.uint8)
    blockers = np.array(
        [
            [BlockerType.CRATE, BlockerType.STONE],
            [BlockerType.STONE, BlockerType.CRATE],
        ],
        dtype=np.uint8,
    )
    blocker_hp = np.full((2, 2), 2, dtype=np.uint8)
    state = BoardState(
        tiles,
        np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )

    with pytest.raises(RuntimeError, match="after 5 attempts"):
        recover_dead_board(
            state,
            np.random.default_rng(3),
            max_attempts=5,
        )


def _make_stress_state() -> BoardState:
    base = generate_playable_board(6, 6, 4, seed=909)
    tiles = base.tiles.copy()
    cell_mask = base.cell_mask.copy()
    specials = base.specials.copy()
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)

    for position in ((1, 2), (4, 3)):
        cell_mask[position] = False
        tiles[position] = 0

    placements = (
        ((0, 0), BlockerType.CRATE, 2),
        ((2, 4), BlockerType.CRATE, 2),
        ((5, 1), BlockerType.CRATE, 2),
        ((1, 4), BlockerType.STONE, 2),
        ((3, 0), BlockerType.STONE, 2),
        ((5, 5), BlockerType.STONE, 2),
        ((0, 3), BlockerType.ICE, 3),
        ((2, 2), BlockerType.ICE, 3),
        ((4, 4), BlockerType.ICE, 3),
        ((5, 0), BlockerType.ICE, 3),
    )
    for position, blocker_type, hp in placements:
        blockers[position] = blocker_type
        blocker_hp[position] = hp
        if blocker_type in (BlockerType.CRATE, BlockerType.STONE):
            tiles[position] = 0

    specials[0, 5] = SpecialType.BOMB
    specials[4, 0] = SpecialType.ROCKET_HORIZONTAL
    return BoardState(tiles, cell_mask, specials, blockers, blocker_hp)


def _run_stress_sequence() -> tuple[list[tuple], list[BoardState]]:
    state = _make_stress_state()
    engine = Match3Engine(num_colors=4, seed=772)
    outcomes = []
    states = []

    for step_index in range(20):
        actions = find_valid_moves(state)
        assert actions
        action = actions[step_index % len(actions)]
        previous = state
        previous_snapshot = previous.copy()
        result = engine.step(previous, action, record_events=True)

        assert result.valid
        assert np.array_equal(previous.tiles, previous_snapshot.tiles)
        assert np.array_equal(previous.specials, previous_snapshot.specials)
        assert np.array_equal(previous.blockers, previous_snapshot.blockers)
        assert np.array_equal(previous.blocker_hp, previous_snapshot.blocker_hp)
        assert_stable_state(result.state)
        assert_only_destroyed_blockers_changed(
            previous,
            result.state,
            result.blocker_damage,
        )
        assert [record.hit_order for record in result.blocker_damage] == list(
            range(len(result.blocker_damage))
        )
        assert find_valid_moves(result.state)

        outcomes.append(
            (
                action,
                result.cascade_count,
                result.reshuffled,
                result.blocker_damage,
                tuple(event.event_type for event in result.animation_events),
            )
        )
        states.append(result.state.copy())
        state = result.state

    return outcomes, states


def test_seeded_long_gameplay_stress_is_stable_and_deterministic():
    outcomes_a, states_a = _run_stress_sequence()
    outcomes_b, states_b = _run_stress_sequence()

    assert outcomes_a == outcomes_b
    assert len(states_a) == len(states_b) == 20
    assert any(record.destroyed for outcome in outcomes_a for record in outcome[3])
    assert sum(record.destroyed for outcome in outcomes_a for record in outcome[3]) >= 2
    assert any(outcome[1] > 0 for outcome in outcomes_a)
    assert (
        sum(
            AnimationEventType.SPECIAL_ACTIVATION in outcome[4]
            for outcome in outcomes_a
        )
        > 1
    )
    assert any(
        AnimationEventType.SPECIAL_CREATION in outcome[4] for outcome in outcomes_a
    )

    for state_a, state_b in zip(states_a, states_b, strict=True):
        assert np.array_equal(state_a.tiles, state_b.tiles)
        assert np.array_equal(state_a.specials, state_b.specials)
        assert np.array_equal(state_a.blockers, state_b.blockers)
        assert np.array_equal(state_a.blocker_hp, state_b.blocker_hp)
        assert np.array_equal(state_a.cell_mask, state_b.cell_mask)
