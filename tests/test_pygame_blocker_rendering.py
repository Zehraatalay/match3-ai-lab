from types import SimpleNamespace

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
from match3_ai.games.match3.board_mechanics import (
    apply_board_gravity,
    refill_board_state,
)
from match3_ai.games.match3.specials import SpecialType
from match3_ai.ui.gravity_animation import calculate_gravity_movements
from match3_ai.ui.pygame_view import (
    ANIMATION_DURATIONS,
    BLOCKER_HIT_DURATION,
    CELL_SIZE,
    HEADER_HEIGHT,
    PADDING,
    Match3Viewer,
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
    blockers: np.ndarray | None = None,
    blocker_hp: np.ndarray | None = None,
    specials: np.ndarray | None = None,
) -> BoardState:
    if tiles is None:
        tiles = np.tile(np.array([1, 2, 3], dtype=np.uint8), (3, 1))
    rows, cols = tiles.shape
    blockers = np.zeros((rows, cols), dtype=np.uint8) if blockers is None else blockers
    blocker_hp = (
        np.zeros((rows, cols), dtype=np.uint8) if blocker_hp is None else blocker_hp
    )
    specials = np.zeros((rows, cols), dtype=np.uint8) if specials is None else specials
    tiles = tiles.copy()
    tiles[np.isin(blockers, [BlockerType.CRATE, BlockerType.STONE])] = 0
    return BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=np.ones((rows, cols), dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )


def make_viewer(state: BoardState) -> Match3Viewer:
    viewer = Match3Viewer.__new__(Match3Viewer)
    viewer.rows = state.rows
    viewer.cols = state.cols
    viewer.state = state
    viewer.screen = pygame.Surface(
        (
            state.cols * CELL_SIZE + 2 * PADDING,
            state.rows * CELL_SIZE + HEADER_HEIGHT + 2 * PADDING,
        )
    )
    viewer.small_font = pygame.font.SysFont("arial", 16)
    viewer.special_font = pygame.font.SysFont("arial", 30, bold=True)
    viewer.is_animating = False
    viewer.current_animation_type = None
    viewer.animation_events = ()
    viewer.animation_index = 0
    viewer.animation_start_time = pygame.time.get_ticks()
    viewer.previous_animation_state = None
    viewer.gravity_movements = ()
    viewer.highlighted_cells = frozenset()
    viewer.selected = None
    viewer.show_blocker_hp = False
    return viewer


def render_board(viewer: Match3Viewer, state: BoardState) -> pygame.Surface:
    viewer.state = state
    viewer.screen.fill((24, 29, 43))
    viewer.draw_board()
    return viewer.screen.copy()


def cell_pixels(surface: pygame.Surface, position: tuple[int, int]) -> np.ndarray:
    row, col = position
    rect = pygame.Rect(
        PADDING + col * CELL_SIZE,
        HEADER_HEIGHT + row * CELL_SIZE,
        CELL_SIZE,
        CELL_SIZE,
    )
    return pygame.surfarray.array3d(surface.subsurface(rect))


def test_crate_and_stone_render_as_distinct_solid_obstacles():
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[1, 1] = BlockerType.CRATE
    blocker_hp[1, 1] = 2
    crate_state = make_state(blockers=blockers, blocker_hp=blocker_hp)
    viewer = make_viewer(crate_state)

    crate_pixels = cell_pixels(render_board(viewer, crate_state), (1, 1))
    assert (
        np.count_nonzero(
            (crate_pixels[:, :, 0] > crate_pixels[:, :, 1] * 1.3)
            & (crate_pixels[:, :, 0] > crate_pixels[:, :, 2] * 1.5)
        )
        > 100
    )

    blockers[1, 1] = BlockerType.STONE
    stone_state = make_state(blockers=blockers, blocker_hp=blocker_hp)
    stone_pixels = cell_pixels(render_board(viewer, stone_state), (1, 1))

    assert not np.array_equal(crate_pixels, stone_pixels)
    assert (
        np.count_nonzero(
            (stone_pixels[:, :, 0] > 70)
            & (np.abs(stone_pixels[:, :, 0].astype(int) - stone_pixels[:, :, 2]) < 45)
        )
        > 100
    )


def test_ice_overlay_keeps_underlying_tile_color_recognizable():
    base = make_state()
    viewer = make_viewer(base)
    base_pixels = cell_pixels(render_board(viewer, base), (1, 1))
    blockers = np.zeros_like(base.tiles)
    blocker_hp = np.zeros_like(base.tiles)
    blockers[1, 1] = BlockerType.ICE
    blocker_hp[1, 1] = 2
    iced = make_state(blockers=blockers, blocker_hp=blocker_hp)
    iced_pixels = cell_pixels(render_board(viewer, iced), (1, 1))

    assert not np.array_equal(base_pixels, iced_pixels)
    inner = iced_pixels[20:48, 20:48].astype(int)
    assert int(np.median(inner[:, :, 2])) > int(np.median(inner[:, :, 0]))


def test_ice_overlay_preserves_underlying_special_visibility():
    specials = np.zeros((3, 3), dtype=np.uint8)
    specials[1, 1] = SpecialType.ROCKET_HORIZONTAL
    base = make_state(specials=specials)
    viewer = make_viewer(base)
    base_pixels = cell_pixels(render_board(viewer, base), (1, 1))
    blockers = np.zeros_like(base.tiles)
    blocker_hp = np.zeros_like(base.tiles)
    blockers[1, 1] = BlockerType.ICE
    blocker_hp[1, 1] = 1
    iced = make_state(blockers=blockers, blocker_hp=blocker_hp, specials=specials)
    iced_pixels = cell_pixels(render_board(viewer, iced), (1, 1))

    assert not np.array_equal(base_pixels, iced_pixels)
    assert np.all(iced_pixels[36, 36] > 180)


def test_hp_badge_changes_with_current_remaining_hp():
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[1, 1] = BlockerType.STONE
    blocker_hp[1, 1] = 1
    viewer = make_viewer(make_state(blockers=blockers, blocker_hp=blocker_hp))
    viewer.show_blocker_hp = True
    one_hp = cell_pixels(render_board(viewer, viewer.state), (1, 1))

    blocker_hp[1, 1] = 12
    twelve_hp = make_state(blockers=blockers, blocker_hp=blocker_hp)
    twelve_hp_pixels = cell_pixels(render_board(viewer, twelve_hp), (1, 1))

    assert not np.array_equal(one_hp, twelve_hp_pixels)
    viewer.show_blocker_hp = True
    debug_pixels = cell_pixels(render_board(viewer, twelve_hp), (1, 1))
    viewer.show_blocker_hp = False
    hidden_pixels = cell_pixels(render_board(viewer, twelve_hp), (1, 1))
    assert not np.array_equal(debug_pixels, hidden_pixels)


def test_ice_over_empty_intermediate_snapshot_renders_safely():
    tiles = np.tile(np.array([1, 2, 3], dtype=np.uint8), (3, 1))
    tiles[1, 1] = 0
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[1, 1] = BlockerType.ICE
    blocker_hp[1, 1] = 3
    state = make_state(tiles, blockers, blocker_hp)
    viewer = make_viewer(state)

    pixels = cell_pixels(render_board(viewer, state), (1, 1))

    assert np.count_nonzero((pixels[:, :, 1] > 170) & (pixels[:, :, 2] > 200)) > 100


def test_blocker_arrays_are_rendered_from_animation_snapshot():
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[0, 2] = BlockerType.CRATE
    blocker_hp[0, 2] = 1
    snapshot = make_state(blockers=blockers, blocker_hp=blocker_hp)
    event = AnimationEvent(
        event_type=AnimationEventType.BLOCKER_DAMAGE,
        state=snapshot,
    )
    viewer = make_viewer(snapshot)
    viewer.is_animating = True
    viewer.current_animation_type = event.event_type
    viewer.animation_events = (event,)

    pixels = cell_pixels(render_board(viewer, event.state), (0, 2))

    assert (
        np.count_nonzero(
            (pixels[:, :, 0] > pixels[:, :, 1] * 1.3)
            & (pixels[:, :, 0] > pixels[:, :, 2] * 1.5)
        )
        > 100
    )


def test_blockers_stay_anchored_during_gravity_and_refill_playback():
    tiles = np.array(
        [
            [1, 2, 3, 1],
            [0, 3, 1, 2],
            [2, 0, 2, 3],
            [3, 1, 3, 1],
        ],
        dtype=np.uint8,
    )
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[1, 0] = BlockerType.ICE
    blocker_hp[1, 0] = 2
    blockers[2, 1] = BlockerType.CRATE
    blocker_hp[2, 1] = 1
    before = BoardState(
        tiles=tiles,
        cell_mask=np.ones_like(tiles, dtype=np.bool_),
        blockers=blockers,
        blocker_hp=blocker_hp,
    )
    after_gravity = apply_board_gravity(before)
    after_refill = refill_board_state(
        after_gravity,
        num_colors=3,
        rng=np.random.default_rng(7),
    )
    viewer = make_viewer(after_gravity)
    viewer.is_animating = True
    viewer.current_animation_type = AnimationEventType.GRAVITY
    viewer.previous_animation_state = before
    viewer.gravity_movements = calculate_gravity_movements(before, after_gravity)
    viewer.animation_start_time = (
        pygame.time.get_ticks() - ANIMATION_DURATIONS[AnimationEventType.GRAVITY]
    )

    gravity_pixels = render_board(viewer, after_gravity)
    gravity_ice_cell = cell_pixels(gravity_pixels, (1, 0))
    gravity_crate_cell = cell_pixels(gravity_pixels, (2, 1))

    viewer.current_animation_type = AnimationEventType.REFILL
    viewer.previous_animation_state = after_gravity
    refill_pixels = render_board(viewer, after_refill)

    assert np.array_equal(
        gravity_crate_cell,
        cell_pixels(refill_pixels, (2, 1)),
    )
    refill_ice_cell = cell_pixels(refill_pixels, (1, 0))
    assert (
        np.count_nonzero(
            (gravity_ice_cell[:, :, 1] > 170) & (gravity_ice_cell[:, :, 2] > 200)
        )
        > 100
    )
    assert (
        np.count_nonzero(
            (refill_ice_cell[:, :, 1] > 170) & (refill_ice_cell[:, :, 2] > 200)
        )
        > 100
    )
    assert after_gravity.blockers[1, 0] == BlockerType.ICE
    assert after_refill.blockers[2, 1] == BlockerType.CRATE


def make_damage_record(
    position: tuple[int, int],
    blocker_type: BlockerType,
    hit_order: int,
    previous_hp: int,
    remaining_hp: int,
) -> BlockerDamageRecord:
    return BlockerDamageRecord(
        sources=frozenset({DamageSource.ROCKET}),
        position=position,
        blocker_type=blocker_type,
        hit_order=hit_order,
        previous_hp=previous_hp,
        remaining_hp=remaining_hp,
        destroyed=remaining_hp == 0,
    )


def make_damage_phase(
    before: BoardState,
    records: tuple[BlockerDamageRecord, ...],
) -> tuple[Match3Viewer, AnimationEvent]:
    blockers = before.blockers.copy()
    blocker_hp = before.blocker_hp.copy()
    for record in records:
        if record.destroyed:
            blockers[record.position] = BlockerType.NONE
            blocker_hp[record.position] = 0
        else:
            blocker_hp[record.position] = record.remaining_hp
    after = make_state(
        tiles=before.tiles,
        blockers=blockers,
        blocker_hp=blocker_hp,
        specials=before.specials,
    )
    event = AnimationEvent(
        event_type=AnimationEventType.BLOCKER_DAMAGE,
        state=after,
        blocker_damage=records,
    )
    viewer = make_viewer(after)
    viewer.is_animating = True
    viewer.current_animation_type = AnimationEventType.BLOCKER_DAMAGE
    viewer.animation_events = (event,)
    viewer.previous_animation_state = before
    viewer.animation_start_time = 1000
    return viewer, event


@pytest.mark.parametrize(
    "blocker_type",
    [BlockerType.CRATE, BlockerType.STONE, BlockerType.ICE],
)
def test_each_blocker_type_gets_a_localized_damage_flash(
    monkeypatch,
    blocker_type,
):
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[1, 1] = blocker_type
    blocker_hp[1, 1] = 3
    before = make_state(blockers=blockers, blocker_hp=blocker_hp)
    record = make_damage_record((1, 1), blocker_type, 0, 3, 2)
    viewer, _ = make_damage_phase(before, (record,))
    monkeypatch.setattr(
        pygame.time,
        "get_ticks",
        lambda: viewer.animation_start_time + 100,
    )

    viewer_pixels = cell_pixels(render_board(viewer, viewer.state), (1, 1))
    normal_pixels = cell_pixels(
        render_board(make_viewer(viewer.state), viewer.state),
        (1, 1),
    )

    assert not np.array_equal(viewer_pixels, normal_pixels)


def test_single_hit_animates_impact_and_hp_transition(monkeypatch):
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[1, 1] = BlockerType.CRATE
    blocker_hp[1, 1] = 2
    before = make_state(blockers=blockers, blocker_hp=blocker_hp)
    record = make_damage_record((1, 1), BlockerType.CRATE, 0, 2, 1)
    viewer, _ = make_damage_phase(before, (record,))
    clock = [viewer.animation_start_time]
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: clock[0])

    initial_visual = viewer.get_blocker_damage_visuals()[(1, 1)]
    clock[0] += 100
    damaged_visual = viewer.get_blocker_damage_visuals()[(1, 1)]
    viewer.screen.fill((24, 29, 43))
    viewer.draw_board()

    assert viewer.get_animation_duration(viewer.animation_events[0]) == (
        BLOCKER_HIT_DURATION
    )
    assert initial_visual.hp == 2
    assert initial_visual.hit_progress == 0
    assert damaged_visual.hp == 1
    assert damaged_visual.damage_count == 1
    assert damaged_visual.visible
    assert np.any(
        cell_pixels(viewer.screen, (1, 1))
        != cell_pixels(render_board(make_viewer(before), before), (1, 1))
    )


def test_multiple_hits_play_in_order_and_destroy_only_on_final_hit(monkeypatch):
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[1, 1] = BlockerType.CRATE
    blocker_hp[1, 1] = 3
    before = make_state(blockers=blockers, blocker_hp=blocker_hp)
    records = (
        make_damage_record((1, 1), BlockerType.CRATE, 0, 3, 2),
        make_damage_record((1, 1), BlockerType.CRATE, 1, 2, 1),
        make_damage_record((1, 1), BlockerType.CRATE, 2, 1, 0),
    )
    viewer, event = make_damage_phase(before, records)
    clock = [viewer.animation_start_time]
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: clock[0])

    observed_hp = []
    for elapsed in (0, 100, 240, 340, 480, 580):
        clock[0] = viewer.animation_start_time + elapsed
        observed_hp.append(viewer.get_blocker_damage_visuals()[(1, 1)].hp)

    assert observed_hp == [3, 2, 2, 1, 1, 0]
    assert viewer.get_animation_duration(event) == 3 * BLOCKER_HIT_DURATION
    assert event.state.blockers[1, 1] == BlockerType.NONE

    clock[0] = viewer.animation_start_time + 640
    destruction_visual = viewer.get_blocker_damage_visuals()[(1, 1)]
    assert not destruction_visual.visible
    assert destruction_visual.destruction_progress is not None
    assert 0 < destruction_visual.destruction_progress < 1


def test_destroyed_blocker_stays_removed_in_next_animation_snapshot(monkeypatch):
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[1, 1] = BlockerType.CRATE
    blocker_hp[1, 1] = 1
    before = make_state(blockers=blockers, blocker_hp=blocker_hp)
    record = make_damage_record((1, 1), BlockerType.CRATE, 0, 1, 0)
    viewer, damage_event = make_damage_phase(before, (record,))
    clear_event = AnimationEvent(
        event_type=AnimationEventType.CLEAR,
        state=damage_event.state.copy(),
    )
    viewer.animation_events = (damage_event, clear_event)
    viewer.final_state = clear_event.state
    clock = [viewer.animation_start_time + BLOCKER_HIT_DURATION]
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: clock[0])

    viewer.update_animation()

    assert viewer.animation_index == 1
    assert viewer.current_animation_type == AnimationEventType.CLEAR
    assert viewer.state is clear_event.state
    assert viewer.previous_animation_state is damage_event.state
    assert viewer.state.blockers[1, 1] == BlockerType.NONE
    assert viewer.get_blocker_damage_visuals() == {}


def test_ice_destruction_keeps_underlying_tile_and_snapshots_unchanged(
    monkeypatch,
):
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[1, 1] = BlockerType.ICE
    blocker_hp[1, 1] = 1
    before = make_state(blockers=blockers, blocker_hp=blocker_hp)
    tile = before.tiles[1, 1]
    record = make_damage_record((1, 1), BlockerType.ICE, 0, 1, 0)
    viewer, event = make_damage_phase(before, (record,))
    arrays_before_draw = (
        before.tiles.copy(),
        before.blockers.copy(),
        before.blocker_hp.copy(),
        event.state.tiles.copy(),
        event.state.blockers.copy(),
        event.state.blocker_hp.copy(),
    )
    clock = [viewer.animation_start_time + 200]
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: clock[0])

    destroyed_overlay = render_board(viewer, event.state)
    normal_viewer = make_viewer(event.state)
    normal_viewer.show_blocker_hp = False
    normal_pixels = render_board(normal_viewer, event.state)
    destroyed_cell = cell_pixels(destroyed_overlay, (1, 1))
    normal_cell = cell_pixels(normal_pixels, (1, 1))

    assert event.state.tiles[1, 1] == tile
    assert not viewer.get_blocker_damage_visuals()[(1, 1)].visible
    assert np.array_equal(destroyed_cell[36, 36], normal_cell[36, 36])
    for actual, expected in zip(
        (
            before.tiles,
            before.blockers,
            before.blocker_hp,
            event.state.tiles,
            event.state.blockers,
            event.state.blocker_hp,
        ),
        arrays_before_draw,
    ):
        assert np.array_equal(actual, expected)


def test_one_damage_event_animates_multiple_blockers_in_record_order(monkeypatch):
    blockers = np.zeros((3, 3), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[0, 0] = BlockerType.CRATE
    blocker_hp[0, 0] = 2
    blockers[0, 2] = BlockerType.STONE
    blocker_hp[0, 2] = 2
    before = make_state(blockers=blockers, blocker_hp=blocker_hp)
    records = (
        make_damage_record((0, 0), BlockerType.CRATE, 5, 2, 1),
        make_damage_record((0, 2), BlockerType.STONE, 6, 2, 1),
    )
    viewer, _ = make_damage_phase(before, records)
    clock = [viewer.animation_start_time + BLOCKER_HIT_DURATION]
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: clock[0])

    visuals = viewer.get_blocker_damage_visuals()

    assert visuals[(0, 0)].hp == 1
    assert visuals[(0, 0)].hit_progress is None
    assert visuals[(0, 2)].hp == 2
    assert visuals[(0, 2)].hit_progress == 0

    clock[0] += BLOCKER_HIT_DURATION
    visuals = viewer.get_blocker_damage_visuals()
    assert visuals[(0, 2)].hp == 1


@pytest.mark.parametrize("toggle_count", [1, 2])
def test_d_key_toggles_hp_debug_overlay(monkeypatch, toggle_count):
    viewer = Match3Viewer()
    assert not viewer.show_blocker_hp
    viewer.draw = lambda: None
    viewer.update_animation = lambda: None
    viewer.clock = SimpleNamespace(tick=lambda fps: None)
    events = [
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d) for _ in range(toggle_count)
    ]
    events.append(pygame.event.Event(pygame.QUIT))
    monkeypatch.setattr(pygame.event, "get", lambda: events)

    viewer.run()

    assert viewer.show_blocker_hp is (toggle_count == 1)
