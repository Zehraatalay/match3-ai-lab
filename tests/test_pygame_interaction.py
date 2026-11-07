import numpy as np
import pygame
import pytest

from match3_ai.games.match3.animation_events import AnimationEvent, AnimationEventType
from match3_ai.games.match3.blockers import (
    BlockerDamageRecord,
    BlockerType,
    DamageSource,
)
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.engine import Match3Engine
from match3_ai.games.match3.moves import is_valid_swap
from match3_ai.games.match3.specials import SpecialType
from match3_ai.ui.pygame_view import (
    CELL_SIZE,
    HEADER_HEIGHT,
    PADDING,
    SWAP_REJECTION_DURATION,
    Match3Viewer,
    generate_blocker_demo_board,
)


@pytest.fixture(autouse=True)
def pygame_headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    yield
    pygame.quit()


def make_state(
    tiles: np.ndarray | None = None,
    specials: np.ndarray | None = None,
    blockers: np.ndarray | None = None,
    blocker_hp: np.ndarray | None = None,
    cell_mask: np.ndarray | None = None,
) -> BoardState:
    if tiles is None:
        tiles = np.array(
            [
                [1, 2, 1],
                [3, 1, 2],
                [2, 3, 1],
            ],
            dtype=np.uint8,
        )
    tiles = tiles.copy()
    shape = tiles.shape
    specials = np.zeros(shape, dtype=np.uint8) if specials is None else specials.copy()
    blockers = np.zeros(shape, dtype=np.uint8) if blockers is None else blockers.copy()
    blocker_hp = (
        np.zeros(shape, dtype=np.uint8) if blocker_hp is None else blocker_hp.copy()
    )
    cell_mask = (
        np.ones(shape, dtype=np.bool_) if cell_mask is None else cell_mask.copy()
    )
    tiles[np.isin(blockers, [BlockerType.CRATE, BlockerType.STONE])] = 0
    tiles[~cell_mask] = 0
    specials[~cell_mask] = SpecialType.NONE
    return BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=cell_mask,
        blockers=blockers,
        blocker_hp=blocker_hp,
    )


def make_viewer(state: BoardState) -> Match3Viewer:
    viewer = Match3Viewer.__new__(Match3Viewer)
    viewer.rows = state.rows
    viewer.cols = state.cols
    viewer.num_colors = 3
    viewer.seed = 7
    viewer.demo_blockers = False
    viewer.state = state
    viewer.engine = Match3Engine(num_colors=3, seed=7)
    viewer.selected = None
    viewer.show_hints = False
    viewer.show_blocker_hp = False
    viewer.message = "Select two adjacent tiles."
    viewer.rejected_swap = None
    viewer.rejection_start_time = 0
    viewer.animation_events = ()
    viewer.animation_index = 0
    viewer.animation_start_time = 0
    viewer.is_animating = False
    viewer.highlighted_cells = frozenset()
    viewer.current_animation_type = None
    viewer.final_state = None
    viewer.previous_animation_state = None
    viewer.gravity_movements = ()
    viewer.screen = pygame.Surface(
        (
            state.cols * CELL_SIZE + 2 * PADDING,
            state.rows * CELL_SIZE + HEADER_HEIGHT + 2 * PADDING,
        )
    )
    viewer.small_font = pygame.font.SysFont("arial", 16)
    viewer.special_font = pygame.font.SysFont("arial", 30, bold=True)
    return viewer


def click_cell(viewer: Match3Viewer, position: tuple[int, int]) -> None:
    viewer.handle_click(viewer.cell_center(position))


def snapshot_arrays(state: BoardState) -> tuple[np.ndarray, ...]:
    return tuple(
        array.copy()
        for array in (
            state.tiles,
            state.specials,
            state.cell_mask,
            state.blockers,
            state.blocker_hp,
        )
    )


def test_successful_swap_animates_both_tiles_and_advances_to_engine_snapshot(
    monkeypatch,
):
    before = make_state()
    before_arrays = snapshot_arrays(before)
    viewer = make_viewer(before)
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 1000)

    click_cell(viewer, (0, 1))
    click_cell(viewer, (1, 1))

    swap_event = viewer.animation_events[0]
    assert swap_event.event_type == AnimationEventType.SWAP
    assert viewer.previous_animation_state is before
    assert viewer.current_animation_type == AnimationEventType.SWAP
    assert viewer.selected is None

    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 1125)
    assert viewer.get_swap_progress() == 0.5
    moving_tiles = []
    viewer.draw_tile = lambda tile, special, center: moving_tiles.append(
        (tile, special, center)
    )
    viewer.draw_swap_animation()

    assert len(moving_tiles) == 2
    assert moving_tiles[0][2] == moving_tiles[1][2]
    assert moving_tiles[0][2] not in {
        (
            PADDING + col * CELL_SIZE + CELL_SIZE // 2,
            HEADER_HEIGHT + row * CELL_SIZE + CELL_SIZE // 2,
        )
        for row, col in swap_event.affected_cells
    }

    moving_tiles.clear()
    viewer.screen.fill((24, 29, 43))
    viewer.draw_board()
    original_centers = {
        (
            PADDING + col * CELL_SIZE + CELL_SIZE // 2,
            HEADER_HEIGHT + row * CELL_SIZE + CELL_SIZE // 2,
        )
        for row, col in swap_event.affected_cells
    }
    assert all(center not in original_centers for _, _, center in moving_tiles)

    next_event = viewer.animation_events[1]
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 1250)
    viewer.update_animation()

    assert viewer.current_animation_type == next_event.event_type
    assert viewer.state is next_event.state
    for array, saved in zip(
        (
            before.tiles,
            before.specials,
            before.cell_mask,
            before.blockers,
            before.blocker_hp,
        ),
        before_arrays,
    ):
        assert np.array_equal(array, saved)

    final_state = viewer.final_state
    while viewer.is_animating:
        event = viewer.animation_events[viewer.animation_index]
        now = viewer.animation_start_time + viewer.get_animation_duration(event)
        monkeypatch.setattr(pygame.time, "get_ticks", lambda now=now: now)
        viewer.update_animation()

    assert viewer.state is final_state
    next_cell = next(
        (row, col)
        for row in range(viewer.rows)
        for col in range(viewer.cols)
        if viewer.is_selectable_cell((row, col))
    )
    click_cell(viewer, next_cell)
    assert viewer.selected == next_cell


def test_swap_animation_keeps_special_symbols_with_tiles(monkeypatch):
    specials = np.zeros((3, 3), dtype=np.uint8)
    specials[0, 1] = SpecialType.ROCKET_HORIZONTAL
    specials[1, 1] = SpecialType.BOMB
    before = make_state(specials=specials)
    viewer = make_viewer(before)
    click_cell(viewer, (0, 1))
    click_cell(viewer, (1, 1))
    moving_tiles = []
    viewer.draw_tile = lambda tile, special, center: moving_tiles.append(
        (tile, special, center)
    )

    viewer.draw_swap_animation()

    assert [item[1] for item in moving_tiles] == [
        SpecialType.ROCKET_HORIZONTAL,
        SpecialType.BOMB,
    ]
    assert any(
        event.special_combination is not None for event in viewer.animation_events
    )

    final_state = viewer.final_state
    while viewer.is_animating:
        event = viewer.animation_events[viewer.animation_index]
        now = viewer.animation_start_time + viewer.get_animation_duration(event)
        monkeypatch.setattr(pygame.time, "get_ticks", lambda now=now: now)
        viewer.update_animation()

    assert viewer.state is final_state
    assert not viewer.is_animating


def test_default_reset_remains_blocker_free():
    viewer = Match3Viewer(seed=42)
    viewer.reset()

    assert not viewer.demo_blockers
    assert not np.any(viewer.state.blockers)
    assert not np.any(viewer.state.blocker_hp)
    assert not viewer.is_animating
    assert viewer.animation_events == ()


def test_large_blocker_hit_sequences_keep_individual_hits_and_bound_duration():
    viewer = make_viewer(make_state())
    records = tuple(
        BlockerDamageRecord(
            sources=frozenset({DamageSource.ROCKET}),
            position=(0, index % 3),
            blocker_type=BlockerType.CRATE,
            hit_order=index,
            previous_hp=2,
            remaining_hp=1,
            destroyed=False,
        )
        for index in range(64)
    )
    event = AnimationEvent(
        event_type=AnimationEventType.BLOCKER_DAMAGE,
        state=viewer.state,
        blocker_damage=records,
    )

    duration = viewer.get_animation_duration(event)

    assert duration == len(records) * viewer.get_blocker_hit_duration(event)
    assert duration <= 3200
    assert viewer.get_blocker_hit_duration(event) >= 50


def advance_animation_to(
    viewer: Match3Viewer,
    event_type: AnimationEventType,
    monkeypatch,
) -> None:
    for _ in range(len(viewer.animation_events)):
        if viewer.current_animation_type == event_type:
            return
        event = viewer.animation_events[viewer.animation_index]
        now = viewer.animation_start_time + viewer.get_animation_duration(event)
        monkeypatch.setattr(pygame.time, "get_ticks", lambda now=now: now)
        viewer.update_animation()
    raise AssertionError(f"animation event {event_type.name} was not reached")


@pytest.mark.parametrize(
    "event_type",
    [
        AnimationEventType.SWAP,
        AnimationEventType.BLOCKER_DAMAGE,
        AnimationEventType.GRAVITY,
        AnimationEventType.REFILL,
    ],
)
def test_reset_during_playback_clears_temporary_state_and_restores_demo(
    monkeypatch,
    event_type,
):
    viewer = Match3Viewer(demo_blockers=True)
    initial_state = generate_blocker_demo_board()
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 1000)
    click_cell(viewer, (1, 4))
    click_cell(viewer, (2, 4))

    assert event_type in [event.event_type for event in viewer.animation_events]
    advance_animation_to(viewer, event_type, monkeypatch)
    assert viewer.is_animating

    viewer.selected = (0, 1)
    viewer.reset()

    assert viewer.demo_blockers
    assert viewer.seed == 314159
    assert not viewer.is_animating
    assert viewer.animation_events == ()
    assert viewer.animation_index == 0
    assert viewer.current_animation_type is None
    assert viewer.final_state is None
    assert viewer.previous_animation_state is None
    assert viewer.gravity_movements == ()
    assert viewer.rejected_swap is None
    assert viewer.selected is None
    for name in ("tiles", "specials", "blockers", "blocker_hp"):
        assert np.array_equal(getattr(viewer.state, name), getattr(initial_state, name))


def test_deterministic_demo_moves_complete_without_stale_ui_state(monkeypatch):
    viewer = Match3Viewer(demo_blockers=True)
    actions = (
        ((1, 4), (2, 4)),
        ((2, 0), (3, 0)),
        ((4, 3), (4, 4)),
    )

    for move_index, action in enumerate(actions):
        assert is_valid_swap(viewer.state, *action)
        click_cell(viewer, action[0])
        click_cell(viewer, action[1])
        assert viewer.is_animating
        final_state = viewer.final_state
        assert final_state is not None

        events = viewer.animation_events
        if move_index == 0:
            assert any(
                record.destroyed and record.blocker_type == BlockerType.ICE
                for event in events
                for record in event.blocker_damage
            )
        elif move_index == 1:
            assert (
                sum(event.event_type == AnimationEventType.MATCH for event in events)
                >= 2
            )
        else:
            assert any(
                event.event_type == AnimationEventType.SPECIAL_ACTIVATION
                for event in events
            )
            assert any(
                record.destroyed for event in events for record in event.blocker_damage
            )

        while viewer.is_animating:
            viewer.draw()
            event = viewer.animation_events[viewer.animation_index]
            now = viewer.animation_start_time + viewer.get_animation_duration(event)
            monkeypatch.setattr(
                pygame.time,
                "get_ticks",
                lambda now=now: now,
            )
            viewer.update_animation()

        assert viewer.state is final_state
        assert viewer.animation_events == ()
        assert viewer.selected is None
        for name in ("tiles", "specials", "blockers", "blocker_hp"):
            assert np.array_equal(
                getattr(viewer.state, name), getattr(final_state, name)
            )

    assert not viewer.is_animating
    assert not np.any(viewer.state.blockers == BlockerType.ICE)


def test_ice_stays_at_its_cell_while_covered_tile_moves(monkeypatch):
    specials = np.zeros((3, 3), dtype=np.uint8)
    specials[0, 1] = SpecialType.ROCKET_VERTICAL
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[0, 1] = BlockerType.ICE
    blocker_hp[0, 1] = 2
    before = make_state(specials=specials, blockers=blockers, blocker_hp=blocker_hp)
    viewer = make_viewer(before)
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 1000)
    click_cell(viewer, (0, 1))
    click_cell(viewer, (0, 2))
    ice_draws = []
    viewer.draw_blocker = lambda blocker_type, hp, center, damage_count=0: (
        ice_draws.append((blocker_type, center))
    )

    viewer.draw_board()

    assert viewer.state.blockers[0, 1] == BlockerType.ICE
    assert ice_draws == [
        (
            BlockerType.ICE,
            (PADDING + CELL_SIZE // 2 + CELL_SIZE, HEADER_HEIGHT + CELL_SIZE // 2),
        )
    ]


def test_invalid_swap_shakes_tiles_without_mutating_board(monkeypatch):
    before = make_state()
    before_arrays = snapshot_arrays(before)
    viewer = make_viewer(before)
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 1000)
    click_cell(viewer, (0, 0))
    click_cell(viewer, (0, 1))

    assert viewer.message == "Invalid move."
    assert not viewer.is_animating
    assert viewer.rejected_swap == ((0, 0), (0, 1))
    assert viewer.state is before

    draw_positions = []
    viewer.draw_tile = lambda tile, special, center: draw_positions.append(center)
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 1130)
    viewer.draw_rejected_swap()
    midpoint_positions = tuple(draw_positions)

    assert len(midpoint_positions) == 2
    assert midpoint_positions[0][0] > PADDING + CELL_SIZE // 2
    assert midpoint_positions[1][0] < PADDING + CELL_SIZE + CELL_SIZE // 2

    draw_positions.clear()
    monkeypatch.setattr(
        pygame.time,
        "get_ticks",
        lambda: 1000 + SWAP_REJECTION_DURATION,
    )
    viewer.draw_rejected_swap()
    assert draw_positions == [
        (PADDING + CELL_SIZE // 2, HEADER_HEIGHT + CELL_SIZE // 2),
        (PADDING + CELL_SIZE + CELL_SIZE // 2, HEADER_HEIGHT + CELL_SIZE // 2),
    ]

    for array, saved in zip(
        (
            before.tiles,
            before.specials,
            before.cell_mask,
            before.blockers,
            before.blocker_hp,
        ),
        before_arrays,
    ):
        assert np.array_equal(array, saved)


def test_rejection_blocks_input_until_return_animation_finishes(monkeypatch):
    viewer = make_viewer(make_state())
    viewer.start_swap_rejection((0, 0), (0, 1))
    viewer.selected = None
    click_cell(viewer, (2, 2))

    assert viewer.selected is None

    monkeypatch.setattr(
        pygame.time,
        "get_ticks",
        lambda: viewer.rejection_start_time + SWAP_REJECTION_DURATION,
    )
    viewer.update_animation()
    click_cell(viewer, (2, 2))

    assert viewer.rejected_swap is None
    assert viewer.selected == (2, 2)


def test_non_adjacent_selection_moves_to_second_playable_tile():
    viewer = make_viewer(make_state())

    click_cell(viewer, (0, 0))
    click_cell(viewer, (2, 2))

    assert viewer.selected == (2, 2)
    assert viewer.rejected_swap is None
    assert viewer.message == "Selected (2, 2)"


@pytest.mark.parametrize("blocker_type", [BlockerType.CRATE, BlockerType.STONE])
def test_fixed_blockers_cannot_be_selected_or_swapped(blocker_type):
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[1, 0] = blocker_type
    blocker_hp[1, 0] = 2
    state = make_state(blockers=blockers, blocker_hp=blocker_hp)
    viewer = make_viewer(state)

    click_cell(viewer, (1, 0))
    assert viewer.selected is None

    click_cell(viewer, (1, 1))
    click_cell(viewer, (1, 0))

    assert viewer.selected is None
    assert viewer.rejected_swap == ((1, 1), (1, 0))
    assert viewer.message == "Invalid move."
    assert viewer.state.blockers[1, 0] == blocker_type


def test_ice_covered_tiles_are_selectable_but_non_playable_cells_are_not():
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[0, 0] = BlockerType.ICE
    blocker_hp[0, 0] = 2
    cell_mask = np.ones((3, 3), dtype=np.bool_)
    cell_mask[2, 2] = False
    state = make_state(blockers=blockers, blocker_hp=blocker_hp, cell_mask=cell_mask)
    viewer = make_viewer(state)

    click_cell(viewer, (0, 0))
    assert viewer.selected == (0, 0)
    click_cell(viewer, (2, 2))
    assert viewer.selected == (0, 0)


def test_attempted_swap_into_non_playable_cell_shows_rejection():
    cell_mask = np.ones((3, 3), dtype=np.bool_)
    cell_mask[0, 1] = False
    viewer = make_viewer(make_state(cell_mask=cell_mask))

    click_cell(viewer, (0, 0))
    click_cell(viewer, (0, 1))

    assert viewer.selected is None
    assert viewer.message == "Invalid move."
    assert viewer.rejected_swap == ((0, 0), (0, 1))


def test_same_cell_deselects_and_reset_clears_selection_and_animation():
    viewer = make_viewer(make_state())
    click_cell(viewer, (0, 0))
    click_cell(viewer, (0, 0))
    assert viewer.selected is None

    viewer.selected = (0, 0)
    viewer.start_swap_rejection((0, 0), (0, 1))
    viewer.is_animating = True
    viewer.animation_events = ()

    viewer.reset()

    assert viewer.selected is None
    assert not viewer.is_animating
    assert viewer.rejected_swap is None
    assert viewer.animation_events == ()


def test_new_selection_is_blocked_during_engine_animation():
    viewer = make_viewer(make_state())
    viewer.is_animating = True
    viewer.selected = None
    state_before = viewer.state

    click_cell(viewer, (0, 0))

    assert viewer.selected is None
    assert viewer.state is state_before
