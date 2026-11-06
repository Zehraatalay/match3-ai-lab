from pathlib import Path

import numpy as np
import pytest

from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.engine import StepResult
from match3_ai.games.match3.moves import is_valid_swap
from match3_ai.games.match3.specials import SpecialType
from match3_ai.levels import (
    BlockerPlacement,
    LevelConfig,
    LevelObjective,
    LevelSession,
    LevelStatus,
    ObjectiveType,
    load_level,
)

LEVEL_DATA = Path(__file__).parents[1] / "src/match3_ai/levels/data"


def make_config(
    objectives,
    *,
    rows=3,
    cols=3,
    num_colors=3,
    move_limit=5,
    seed=42,
    blockers=(),
    initial_specials=(),
):
    return LevelConfig(
        rows=rows,
        cols=cols,
        num_colors=num_colors,
        move_limit=move_limit,
        seed=seed,
        blockers=tuple(blockers),
        initial_specials=tuple(initial_specials),
        objectives=tuple(objectives),
    )


def normal_state() -> BoardState:
    tiles = np.array(
        [
            [1, 2, 1],
            [3, 1, 2],
            [2, 3, 1],
        ],
        dtype=np.uint8,
    )
    return BoardState(tiles, np.ones_like(tiles, dtype=np.bool_))


def set_session_state(session: LevelSession, state: BoardState) -> None:
    session.state = state


def fake_valid_result(state: BoardState, *, cleared=0, by_color=()) -> StepResult:
    return StepResult(
        state=state,
        valid=True,
        cleared_tiles=cleared,
        cascade_count=0,
        cleared_by_color=tuple(by_color),
    )


def blocker_state(session: LevelSession) -> BoardState:
    tiles = session.state.tiles.copy()
    specials = session.state.specials.copy()
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    for placement in session.config.blockers:
        blockers[placement.position] = placement.blocker_type
        blocker_hp[placement.position] = placement.hp
        if placement.blocker_type in (BlockerType.CRATE, BlockerType.STONE):
            tiles[placement.position] = 0
            specials[placement.position] = SpecialType.NONE
    return BoardState(
        tiles,
        session.state.cell_mask,
        specials,
        blockers,
        blocker_hp,
    )


def remove_blockers(state: BoardState, positions) -> BoardState:
    blockers = state.blockers.copy()
    blocker_hp = state.blocker_hp.copy()
    for position in positions:
        blockers[position] = BlockerType.NONE
        blocker_hp[position] = 0
    return BoardState(
        state.tiles,
        state.cell_mask,
        state.specials,
        blockers,
        blocker_hp,
    )


def test_session_initializes_state_moves_objectives_and_status():
    placements = (
        BlockerPlacement((0, 0), BlockerType.CRATE, "DOUBLE"),
        BlockerPlacement((2, 2), BlockerType.ICE, "THIN"),
    )
    config = make_config(
        [
            LevelObjective(ObjectiveType.CLEAR_ALL_BLOCKERS),
            LevelObjective(ObjectiveType.COLLECT_TILES, target=12),
        ],
        blockers=placements,
    )

    session = LevelSession(config)

    assert session.config is config
    assert session.state.shape == (3, 3)
    assert session.moves_remaining == config.move_limit
    assert session.status == LevelStatus.IN_PROGRESS
    assert len(session.objective_progress) == 2
    blocker_progress, collect_progress = session.objective_progress
    assert blocker_progress.target == 2
    assert blocker_progress.initial_blocker_count == 2
    assert blocker_progress.remaining_blocker_count == 2
    assert blocker_progress.cleared_blocker_count == 0
    assert blocker_progress.progress == 0
    assert not blocker_progress.completed
    assert collect_progress.target == 12
    assert collect_progress.progress == 0
    assert collect_progress.raw_progress == 0
    assert not collect_progress.completed


def test_deterministic_reset_restores_board_engine_budget_and_progress():
    config = make_config(
        [LevelObjective(ObjectiveType.COLLECT_TILES, target=100)],
        rows=6,
        cols=6,
    )
    session = LevelSession(config)
    session.state = normal_state()
    session.step(((0, 1), (1, 1)))

    first_reset = session.reset()
    first_arrays = tuple(
        array.copy()
        for array in (
            first_reset.tiles,
            first_reset.specials,
            first_reset.cell_mask,
            first_reset.blockers,
            first_reset.blocker_hp,
        )
    )
    first_random = session.engine.rng.random(4)

    second_reset = session.reset()
    second_arrays = (
        second_reset.tiles,
        second_reset.specials,
        second_reset.cell_mask,
        second_reset.blockers,
        second_reset.blocker_hp,
    )

    assert all(np.array_equal(a, b) for a, b in zip(first_arrays, second_arrays))
    assert np.array_equal(first_random, session.engine.rng.random(4))
    assert session.moves_remaining == config.move_limit
    assert session.objective_progress[0].raw_progress == 0
    assert session.status == LevelStatus.IN_PROGRESS


def test_invalid_move_consumes_no_move_and_does_not_update_objectives():
    session = LevelSession(
        make_config([LevelObjective(ObjectiveType.COLLECT_TILES, target=100)])
    )
    state = normal_state()
    set_session_state(session, state)

    result = session.step(((0, 0), (2, 2)))

    assert not result.valid
    assert session.moves_remaining == session.config.move_limit
    assert session.state is state
    assert session.objective_progress[0].raw_progress == 0
    assert session.status == LevelStatus.IN_PROGRESS


def test_valid_cascade_consumes_exactly_one_move():
    session = LevelSession(
        make_config([LevelObjective(ObjectiveType.COLLECT_TILES, target=100)])
    )
    set_session_state(session, normal_state())
    before = session.moves_remaining

    result = session.step(((0, 1), (1, 1)))

    assert result.valid
    assert result.cascade_count >= 1
    assert session.moves_remaining == before - 1
    assert session.objective_progress[0].raw_progress == result.cleared_tiles
    assert sum(result.cleared_by_color) == result.cleared_tiles


def test_special_combination_consumes_one_move():
    session = LevelSession(
        make_config([LevelObjective(ObjectiveType.COLLECT_TILES, target=100)])
    )
    state = normal_state()
    specials = state.specials.copy()
    specials[0, 1] = SpecialType.ROCKET_HORIZONTAL
    specials[1, 1] = SpecialType.BOMB
    set_session_state(
        session,
        BoardState(state.tiles, state.cell_mask, specials),
    )
    assert is_valid_swap(session.state, (0, 1), (1, 1))

    result = session.step(((0, 1), (1, 1)), record_events=True)

    assert result.valid
    assert any(
        event.special_combination is not None for event in result.animation_events
    )
    assert session.moves_remaining == session.config.move_limit - 1


def test_clear_all_blockers_uses_final_board_and_tracks_partial_progress(monkeypatch):
    placements = (BlockerPlacement((1, 0), BlockerType.CRATE, "DOUBLE"),)
    session = LevelSession(
        make_config(
            [LevelObjective(ObjectiveType.CLEAR_ALL_BLOCKERS)],
            blockers=placements,
        )
    )
    state = blocker_state(session)
    blocker_hp = state.blocker_hp.copy()
    blocker_hp[1, 0] = 1
    partial_state = BoardState(
        state.tiles,
        state.cell_mask,
        state.specials,
        state.blockers,
        blocker_hp,
    )
    set_session_state(session, state)
    monkeypatch.setattr(
        session.engine,
        "step",
        lambda _state, _action, record_events=False: fake_valid_result(partial_state),
    )

    result = session.step(((0, 0), (0, 1)))

    progress = session.objective_progress[0]
    assert result.valid
    assert result.blocker_damage == ()
    assert progress.initial_blocker_count == 1
    assert progress.remaining_blocker_count == 1
    assert progress.cleared_blocker_count == 0
    assert progress.progress == 0
    assert not progress.completed
    assert session.status == LevelStatus.IN_PROGRESS


def test_clear_all_blockers_counts_mixed_types_and_completes_only_at_zero():
    placements = (
        BlockerPlacement((0, 0), BlockerType.CRATE, "TRIPLE"),
        BlockerPlacement((1, 1), BlockerType.ICE, "THICK"),
        BlockerPlacement((2, 2), BlockerType.STONE, "DOUBLE"),
    )
    session = LevelSession(
        make_config(
            [LevelObjective(ObjectiveType.CLEAR_ALL_BLOCKERS)],
            blockers=placements,
        )
    )
    state = blocker_state(session)
    set_session_state(session, state)
    partial = remove_blockers(state, {(0, 0), (1, 1)})
    session.engine.step = lambda *_args, **_kwargs: fake_valid_result(partial)

    session.step(((0, 0), (0, 1)))

    progress = session.objective_progress[0]
    assert progress.initial_blocker_count == 3
    assert progress.remaining_blocker_count == 1
    assert progress.cleared_blocker_count == 2
    assert progress.progress == 2
    assert not progress.completed
    assert session.status == LevelStatus.IN_PROGRESS

    final = remove_blockers(partial, {(2, 2)})
    session.engine.step = lambda *_args, **_kwargs: fake_valid_result(final)
    session.step(((0, 0), (0, 1)))

    progress = session.objective_progress[0]
    assert progress.remaining_blocker_count == 0
    assert progress.cleared_blocker_count == 3
    assert progress.completed
    assert session.status == LevelStatus.WON


def test_any_color_collection_uses_actual_clear_count_and_caps_display_progress():
    session = LevelSession(
        make_config([LevelObjective(ObjectiveType.COLLECT_TILES, target=1)])
    )
    set_session_state(session, normal_state())

    result = session.step(((0, 1), (1, 1)))

    progress = session.objective_progress[0]
    assert result.cleared_tiles >= 3
    assert progress.raw_progress == result.cleared_tiles
    assert progress.progress == 1
    assert progress.completed
    assert session.status == LevelStatus.WON


def test_color_specific_collection_counts_special_effects_and_cascades():
    session = LevelSession(
        make_config(
            [LevelObjective(ObjectiveType.COLLECT_TILES, target=100, color=2)],
            rows=5,
            cols=5,
        )
    )
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
    specials[1, 1] = SpecialType.ROCKET_HORIZONTAL
    state = BoardState(tiles, np.ones_like(tiles, dtype=np.bool_), specials)
    set_session_state(session, state)
    assert is_valid_swap(state, (1, 1), (1, 2))

    result = session.step(((1, 1), (1, 2)))

    assert result.valid
    assert len(result.cleared_by_color) == session.config.num_colors
    assert session.objective_progress[0].raw_progress == result.cleared_by_color[1]
    assert result.cleared_by_color[1] > 0
    assert sum(result.cleared_by_color) <= result.cleared_tiles


def test_color_counts_exclude_a_tile_preserved_for_special_creation():
    tiles = np.array(
        [
            [1, 4, 3, 2, 4],
            [3, 2, 3, 4, 2],
            [1, 4, 1, 1, 2],
            [1, 1, 3, 3, 1],
            [2, 4, 1, 3, 3],
        ],
        dtype=np.uint8,
    )
    config = make_config(
        [LevelObjective(ObjectiveType.COLLECT_TILES, target=100)],
        rows=5,
        cols=5,
        num_colors=4,
    )
    session = LevelSession(config)
    set_session_state(session, BoardState(tiles, np.ones_like(tiles, dtype=np.bool_)))

    result = session.step(((2, 1), (3, 1)))

    assert result.valid
    assert np.count_nonzero(result.state.specials) >= 1
    assert sum(result.cleared_by_color) == result.cleared_tiles


def test_multiple_objectives_must_all_complete_to_win(monkeypatch):
    placements = (BlockerPlacement((1, 0), BlockerType.CRATE, "SINGLE"),)
    session = LevelSession(
        make_config(
            [
                LevelObjective(ObjectiveType.CLEAR_ALL_BLOCKERS),
                LevelObjective(ObjectiveType.COLLECT_TILES, target=10),
            ],
            blockers=placements,
        )
    )
    state = blocker_state(session)
    session.state = state
    final = remove_blockers(state, {(1, 0)})
    monkeypatch.setattr(
        session.engine,
        "step",
        lambda _state, _action, record_events=False: fake_valid_result(
            final,
            cleared=4,
            by_color=(1, 2, 1),
        ),
    )

    session.step(((0, 0), (0, 1)))

    assert session.objective_progress[0].completed
    assert session.objective_progress[1].progress == 4
    assert not session.objective_progress[1].completed
    assert session.status == LevelStatus.IN_PROGRESS

    monkeypatch.setattr(
        session.engine,
        "step",
        lambda _state, _action, record_events=False: fake_valid_result(
            final,
            cleared=6,
            by_color=(2, 2, 2),
        ),
    )
    session.step(((0, 0), (0, 1)))

    assert session.objective_progress[1].raw_progress == 10
    assert session.objective_progress[1].progress == 10
    assert session.status == LevelStatus.WON


def test_final_move_win_takes_priority_over_move_limit_loss():
    session = LevelSession(
        make_config(
            [LevelObjective(ObjectiveType.COLLECT_TILES, target=1)],
            move_limit=1,
        )
    )
    set_session_state(session, normal_state())

    result = session.step(((0, 1), (1, 1)))

    assert result.valid
    assert session.moves_remaining == 0
    assert session.status == LevelStatus.WON
    with pytest.raises(RuntimeError, match="terminal level session"):
        session.step(((0, 1), (1, 1)))


def test_incomplete_objective_at_zero_moves_loses_and_terminal_session_rejects_moves():
    session = LevelSession(
        make_config(
            [LevelObjective(ObjectiveType.COLLECT_TILES, target=100)],
            move_limit=1,
        )
    )
    set_session_state(session, normal_state())

    result = session.step(((0, 1), (1, 1)))

    assert result.valid
    assert session.moves_remaining == 0
    assert session.status == LevelStatus.LOST
    with pytest.raises(RuntimeError, match="terminal level session"):
        session.step(((0, 1), (1, 1)))
    assert session.moves_remaining == 0


@pytest.mark.parametrize(
    "filename",
    [
        "first-match.json",
        "wooden-trouble.json",
        "double-trouble.json",
        "frozen-garden.json",
        "stone-breaker.json",
        "blocker-mayhem.json",
    ],
)
def test_all_six_levels_initialize_sessions_and_reset_deterministically(filename):
    config = load_level(LEVEL_DATA / filename)
    session = LevelSession(config)
    initial = session.state
    initial_arrays = tuple(
        array.copy()
        for array in (
            initial.tiles,
            initial.specials,
            initial.cell_mask,
            initial.blockers,
            initial.blocker_hp,
        )
    )

    assert session.moves_remaining == config.move_limit
    assert session.status == LevelStatus.IN_PROGRESS
    assert len(session.objective_progress) == len(config.objectives)
    assert all(progress.progress == 0 for progress in session.objective_progress)
    assert np.count_nonzero(session.state.blockers != BlockerType.NONE) == len(
        config.blockers
    )
    for progress, objective in zip(
        session.objective_progress,
        config.objectives,
    ):
        expected_target = (
            len(config.blockers)
            if objective.objective_type == ObjectiveType.CLEAR_ALL_BLOCKERS
            else objective.target
        )
        assert progress.target == expected_target
        if objective.objective_type == ObjectiveType.CLEAR_ALL_BLOCKERS:
            assert progress.initial_blocker_count == len(config.blockers)

    reset_state = session.reset()

    assert reset_state is not initial
    assert session.moves_remaining == config.move_limit
    assert session.status == LevelStatus.IN_PROGRESS
    assert all(progress.progress == 0 for progress in session.objective_progress)
    for previous, current in zip(
        initial_arrays,
        (
            reset_state.tiles,
            reset_state.specials,
            reset_state.cell_mask,
            reset_state.blockers,
            reset_state.blocker_hp,
        ),
    ):
        assert np.array_equal(previous, current)
