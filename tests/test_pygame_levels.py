import numpy as np
import pygame
import pytest

from match3_ai.games.match3.animation_events import (
    AnimationEvent,
    AnimationEventType,
    SpecialCombinationInfo,
)
from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.board_mechanics import apply_board_gravity
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.specials import SpecialType
from match3_ai.levels import LevelStatus
from match3_ai.ui import pygame_view
from match3_ai.ui.gravity_animation import TileMovement, calculate_gravity_movements
from match3_ai.ui.pygame_view import (
    PACKAGED_LEVELS,
    PADDING,
    Match3Viewer,
    load_packaged_level,
)


@pytest.fixture(autouse=True)
def pygame_headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    yield
    pygame.quit()


def click_cell(viewer: Match3Viewer, position: tuple[int, int]) -> None:
    viewer.handle_click(viewer.cell_center(position))


def test_packaged_level_order_and_selection_click():
    configs = [load_packaged_level(index) for index in range(1, 7)]
    assert len(PACKAGED_LEVELS) == 6
    assert [config.name for config in configs] == [
        "First Match",
        "Wooden Trouble",
        "Double Trouble",
        "Frozen Garden",
        "Stone Breaker",
        "Blocker Mayhem",
    ]

    viewer = Match3Viewer(level_selection=True)
    rect = viewer.level_selection_rect(3)
    viewer.handle_click(rect.center)

    assert viewer.level_number == 3
    assert viewer.level_session is not None
    assert viewer.level_session.config.name == "Double Trouble"
    assert viewer.state is viewer.level_session.board
    assert viewer.screen_mode == "gameplay"


def test_keyboard_selection_and_direct_level_launch():
    menu = Match3Viewer(level_selection=True)
    menu.handle_level_selection_key(pygame.K_DOWN)
    menu.handle_level_selection_key(pygame.K_RETURN)
    assert menu.level_number == 2

    direct = Match3Viewer(level_number=5)
    assert direct.level_number == 5
    assert direct.level_config == load_packaged_level(5)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [([], {"demo_blockers": False, "level_number": None, "level_selection": False}),
     (["--demo-blockers"], {"demo_blockers": True, "level_number": None, "level_selection": False}),
     (["--levels"], {"demo_blockers": False, "level_number": None, "level_selection": True}),
     (["--level", "4"], {"demo_blockers": False, "level_number": 4, "level_selection": False})],
)
def test_cli_modes_remain_separate(monkeypatch, arguments, expected):
    received = {}

    class ViewerStub:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def run(self):
            return None

    monkeypatch.setattr(pygame_view, "Match3Viewer", ViewerStub)
    pygame_view.main(arguments)

    for key, value in expected.items():
        assert received[key] == value


def test_level_move_is_routed_through_session_and_hud_waits_for_animation(
    monkeypatch,
):
    viewer = Match3Viewer(level_number=1)
    session = viewer.level_session
    assert session is not None
    action = find_valid_moves(session.board)[0]
    calls = []
    session_step = session.step

    def tracked_step(action, *, record_events=False):
        calls.append((action, record_events))
        return session_step(action, record_events=record_events)

    monkeypatch.setattr(session, "step", tracked_step)
    before = session.moves_remaining
    click_cell(viewer, action[0])
    click_cell(viewer, action[1])

    assert calls == [(action, True)]
    assert session.moves_remaining == before - 1
    assert viewer.engine is session.engine
    assert viewer.state is not session.board if viewer.is_animating else viewer.state is session.board

    if viewer.is_animating:
        assert viewer.displayed_moves_remaining == before
        ticks = [100_000]

        def advance_ticks():
            ticks[0] += 100_000
            return ticks[0]

        monkeypatch.setattr(pygame.time, "get_ticks", advance_ticks)
        while viewer.is_animating:
            viewer.update_animation()
    assert viewer.displayed_moves_remaining == before - 1
    assert viewer.state is session.board


def test_objective_hud_formats_blocker_any_color_and_color_specific_progress():
    viewer = Match3Viewer(level_number=6)
    blocker_progress = viewer.level_session.objective_progress[0]
    assert viewer.objective_label(blocker_progress) == "Blockers: 0 / 6 cleared"

    first_match = Match3Viewer(level_number=1)
    progress = first_match.level_session.objective_progress[0]
    assert first_match.objective_label(progress) == "Tiles: 0 / 30"

    config = load_packaged_level(1)
    from dataclasses import replace

    viewer.level_config = replace(
        config,
        objectives=(
            replace(config.objectives[0], color=2),
        ),
    )
    assert viewer.objective_label(progress, color=2) == "Blue tiles: 0 / 30"


@pytest.mark.parametrize("level_number", range(1, 7))
def test_level_hud_text_fits_inside_header(level_number):
    viewer = Match3Viewer(level_number=level_number)
    viewer.draw_header()
    hud = viewer.hud_rect
    content_x = hud.x + 12
    level_count = viewer.tiny_font.render("LEVEL 06 / 06", True, (0, 0, 0))
    name = viewer.fit_text(
        viewer.level_config.name,
        viewer.font,
        (0, 0, 0),
        hud.width - 24,
    )
    moves_card = pygame.Rect(content_x, hud.y + 68, 150, 46)
    objective_card = viewer.get_objective_card_rect()
    text_rects = [
        pygame.Rect(content_x, hud.y + 4, *viewer.font.size("MATCH-3 AI LAB")),
        pygame.Rect(
            hud.right - 12 - level_count.get_width(),
            hud.y + 12,
            *level_count.get_size(),
        ),
        pygame.Rect(content_x, hud.y + 31, *name.get_size()),
        pygame.Rect(
            moves_card.x + 10,
            moves_card.y + 4,
            *viewer.tiny_font.size("MOVES LEFT"),
        ),
        pygame.Rect(
            moves_card.x + 10,
            moves_card.y + 20,
            *viewer.small_font.size(str(viewer.displayed_moves_remaining)),
        ),
        pygame.Rect(
            objective_card.x + 10,
            objective_card.y + 3,
            *viewer.tiny_font.size("OBJECTIVES"),
        ),
    ]
    for index, (progress, objective) in enumerate(
        zip(viewer.displayed_objective_progress, viewer.level_config.objectives)
    ):
        label = viewer.fit_text(
            viewer.objective_label(progress, objective.color),
            viewer.tiny_font,
            (0, 0, 0),
            objective_card.width - 20,
        )
        text_rects.append(
            pygame.Rect(
                objective_card.x + 10,
                objective_card.y + 18 + index * 14,
                *label.get_size(),
            )
        )

    assert all(hud.contains(rect) for rect in text_rects)
    assert all(
        not text_rects[first].colliderect(text_rects[second])
        for first in range(len(text_rects))
        for second in range(first + 1, len(text_rects))
    )


@pytest.mark.parametrize("mode", ["default", "demo", "level"])
def test_layout_regions_are_separate_and_fit_the_window(mode):
    viewer = {
        "default": lambda: Match3Viewer(),
        "demo": lambda: Match3Viewer(demo_blockers=True),
        "level": lambda: Match3Viewer(level_number=1),
    }[mode]()

    assert viewer.hud_rect.bottom < viewer.board_region_rect.top
    assert viewer.board_region_rect.bottom < viewer.action_bar_rect.top
    assert viewer.board_region_rect.contains(viewer.board_rect)
    assert viewer.action_bar_rect == viewer.controls_rect
    assert viewer.hud_rect.left == viewer.board_region_rect.left == viewer.action_bar_rect.left
    assert viewer.hud_rect.right == viewer.board_region_rect.right == viewer.action_bar_rect.right
    assert viewer.action_bar_rect.bottom + PADDING - 6 == viewer.height
    assert viewer.action_bar_rect.right <= viewer.width


@pytest.mark.parametrize("level_number", range(1, 7))
def test_controls_text_fits_controls_region(level_number):
    viewer = Match3Viewer(level_number=level_number)
    viewer.draw_controls()
    buttons = viewer.action_bar_buttons()
    assert [action for action, _rect in buttons] == [
        "restart",
        "hint",
        "hp",
        "levels",
        "quit",
    ]
    for action, rect in buttons:
        text = viewer.fit_text(
            viewer.action_button_label(action),
            viewer.tiny_font,
            (0, 0, 0),
            rect.width - 10,
        )
        assert text.get_width() <= rect.width - 10
        assert text.get_height() <= rect.height
        assert viewer.action_bar_rect.contains(rect)
    assert all(
        not buttons[first][1].colliderect(buttons[second][1])
        for first in range(len(buttons))
        for second in range(first + 1, len(buttons))
    )


def test_action_bar_button_styles_reflect_hover_and_active_states():
    viewer = Match3Viewer(level_number=1)
    idle = viewer.action_button_colors("hp")
    hover = viewer.action_button_colors("hp", hovered=True)
    viewer.show_blocker_hp = True
    active = viewer.action_button_colors("hp")

    assert idle != hover
    assert active != idle
    assert viewer.action_button_label("hp") == "HP: ON"


def test_action_bar_buttons_handle_actions_without_selecting_board_cells():
    viewer = Match3Viewer(level_number=1)
    initial_tiles = viewer.state.tiles.copy()
    buttons = dict(viewer.action_bar_buttons())

    viewer.handle_click(buttons["hint"].center)
    assert viewer.show_hints
    assert viewer.selected is None

    viewer.handle_click(buttons["hp"].center)
    assert viewer.show_blocker_hp
    assert viewer.action_button_label("hp") == "HP: ON"

    viewer.selected = (0, 0)
    viewer.is_animating = True
    viewer.handle_click(buttons["restart"].center)
    assert not viewer.is_animating
    assert viewer.selected is None
    assert np.array_equal(viewer.state.tiles, initial_tiles)

    viewer.handle_click(dict(viewer.action_bar_buttons())["levels"].center)
    assert viewer.screen_mode == "selection"

    demo = Match3Viewer(demo_blockers=True)
    demo.handle_click(dict(demo.action_bar_buttons())["levels"].center)
    assert demo.screen_mode == "selection"

    viewer = Match3Viewer()
    board_position = viewer.cell_center((0, 0))
    viewer.handle_click(board_position)
    assert viewer.selected == (0, 0)
    viewer.handle_click(dict(viewer.action_bar_buttons())["quit"].center)
    assert viewer.quit_requested
    assert viewer.selected == (0, 0)


def test_keyboard_shortcuts_remain_available_with_action_bar(monkeypatch):
    viewer = Match3Viewer(level_number=1)
    events = [
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_h),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_d),
        pygame.event.Event(pygame.KEYDOWN, key=pygame.K_l),
        pygame.event.Event(pygame.QUIT),
    ]
    monkeypatch.setattr(pygame.event, "get", lambda: events)
    monkeypatch.setattr(viewer, "draw", lambda: None)

    class ClockStub:
        def tick(self, _fps):
            return None

    viewer.clock = ClockStub()

    viewer.run()

    assert viewer.show_hints
    assert viewer.show_blocker_hp
    assert viewer.screen_mode == "selection"


@pytest.mark.parametrize("level_number", range(1, 7))
def test_fixed_window_centers_all_six_board_sizes_and_maps_cells(level_number):
    viewer = Match3Viewer(level_number=level_number)
    assert (viewer.width, viewer.height) == (624, 890)
    assert viewer.screen.get_size() == (viewer.width, viewer.height)
    assert viewer.board_rect.centerx == viewer.board_region_rect.centerx
    assert viewer.board_rect.centery == viewer.board_region_rect.centery
    assert viewer.hud_rect.bottom < viewer.board_region_rect.top
    assert viewer.board_region_rect.bottom < viewer.action_bar_rect.top
    assert viewer.cell_at_pixel(*viewer.cell_center((0, 0))) == (0, 0)
    assert viewer.cell_at_pixel(*viewer.cell_center((viewer.rows - 1, viewer.cols - 1))) == (
        viewer.rows - 1,
        viewer.cols - 1,
    )


def test_gravity_movements_stay_within_mask_and_fixed_blocker_segments():
    mask = np.ones((9, 3), dtype=np.bool_)
    mask[7, 0] = False
    tiles = np.zeros((9, 3), dtype=np.uint8)
    tiles[0, 0] = 1
    tiles[3, 0] = 2
    tiles[5, 0] = 3
    tiles[8, 0] = 1
    specials = np.zeros_like(tiles)
    specials[3, 0] = SpecialType.BOMB
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[2, 0] = BlockerType.STONE
    blocker_hp[2, 0] = 2
    blockers[6, 0] = BlockerType.CRATE
    blocker_hp[6, 0] = 1
    before = BoardState(
        tiles,
        mask,
        specials=specials,
        blockers=blockers,
        blocker_hp=blocker_hp,
    )
    after = apply_board_gravity(before)

    assert calculate_gravity_movements(before, after) == (
        TileMovement((0, 0), (1, 0)),
        TileMovement((3, 0), (4, 0)),
    )
    assert after.specials[4, 0] == SpecialType.BOMB
    assert after.tiles[8, 0] == before.tiles[8, 0]


def test_mouse_coordinates_and_tile_center_share_board_translation():
    viewer = Match3Viewer(level_number=1)
    position = (3, 5)
    center = viewer.cell_center(position)

    assert viewer.cell_at_pixel(*center) == position
    assert viewer.cell_at_pixel(*viewer.hud_rect.center) is None
    assert viewer.cell_at_pixel(*viewer.action_bar_rect.center) is None


def test_blocker_effect_center_uses_board_cell_translation(monkeypatch):
    viewer = Match3Viewer(level_number=6)
    positions = []
    monkeypatch.setattr(
        viewer,
        "draw_blocker",
        lambda _type, _hp, center, damage_count=0: positions.append(center),
    )

    viewer.draw_blockers(viewer.state)
    position = next(
        (row, col)
        for row in range(viewer.rows)
        for col in range(viewer.cols)
        if viewer.state.blockers[row, col] != 0
    )
    assert positions
    assert viewer.cell_center(position) in positions


def test_gravity_animation_uses_board_cell_translation(monkeypatch):
    viewer = Match3Viewer(rows=3, cols=3, num_colors=3)
    viewer.previous_animation_state = viewer.state
    viewer.gravity_movements = (TileMovement((0, 0), (1, 0)),)
    viewer.animation_start_time = 0
    viewer.state = viewer.state.copy()
    viewer.is_animating = True
    centers = []
    elapsed = 72
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: elapsed)
    monkeypatch.setattr(
        viewer,
        "draw_tile",
        lambda _tile, _special, center: centers.append(center),
    )

    viewer.draw_gravity_animation()

    start = viewer.cell_center((0, 0))
    end = viewer.cell_center((1, 0))
    raw_progress = elapsed / 350
    progress = 1 - (1 - raw_progress) ** 3
    expected_center = (start[0], int(start[1] + (end[1] - start[1]) * progress))
    assert expected_center in centers


@pytest.mark.parametrize("level_number", range(1, 7))
def test_gravity_animation_alignment_for_each_level_size(monkeypatch, level_number):
    viewer = Match3Viewer(level_number=level_number)
    movement = next(
        TileMovement((row, col), (row + 1, col))
        for col in range(viewer.cols)
        for row in range(viewer.rows - 1)
        if viewer.state.cell_mask[row, col]
        and viewer.state.cell_mask[row + 1, col]
        and viewer.state.tiles[row, col] != 0
        and viewer.state.blockers[row, col] != BlockerType.CRATE
        and viewer.state.blockers[row, col] != BlockerType.STONE
        and viewer.state.blockers[row + 1, col] != BlockerType.CRATE
        and viewer.state.blockers[row + 1, col] != BlockerType.STONE
    )
    viewer.previous_animation_state = viewer.state
    viewer.gravity_movements = (movement,)
    viewer.animation_start_time = 0
    viewer.is_animating = True
    viewer.state = viewer.state.copy()
    centers = []
    elapsed = 170
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: elapsed)
    monkeypatch.setattr(
        viewer,
        "draw_tile",
        lambda _tile, _special, center: centers.append(center),
    )

    viewer.draw_gravity_animation()

    start = viewer.cell_center(movement.source)
    end = viewer.cell_center(movement.destination)
    progress = 1 - (1 - elapsed / 350) ** 3
    expected = (
        start[0],
        int(start[1] + (end[1] - start[1]) * progress),
    )
    assert expected in centers


def test_refill_animation_lands_tiles_at_board_cell_centers(monkeypatch):
    viewer = Match3Viewer(seed=17)
    previous_tiles = np.zeros_like(viewer.state.tiles)
    viewer.previous_animation_state = BoardState(
        previous_tiles,
        viewer.state.cell_mask,
    )
    viewer.animation_start_time = 0
    centers = []
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 500)
    monkeypatch.setattr(
        viewer,
        "draw_tile",
        lambda _tile, _special, center: centers.append(center),
    )

    viewer.draw_refill_animation()

    assert viewer.cell_center((0, 0)) in centers
    assert viewer.cell_center((viewer.rows - 1, viewer.cols - 1)) in centers


@pytest.mark.parametrize("level_number", range(1, 7))
def test_special_effects_receive_the_board_region_origin(monkeypatch, level_number):
    viewer = Match3Viewer(level_number=level_number)
    event = AnimationEvent(
        event_type=AnimationEventType.SPECIAL_ACTIVATION,
        state=viewer.state,
        special_combination=SpecialCombinationInfo(
            source=(1, 1),
            destination=(1, 2),
            source_type=SpecialType.PROPELLER,
            destination_type=SpecialType.ROCKET_HORIZONTAL,
            effect_cells=frozenset({(1, 1), (1, 2)}),
        ),
    )
    viewer.animation_events = (event,)
    viewer.animation_index = 0
    viewer.animation_start_time = 0
    viewer.is_animating = True
    viewer.current_animation_type = AnimationEventType.SPECIAL_ACTIVATION
    received = []
    monkeypatch.setattr(pygame.time, "get_ticks", lambda: 100)
    monkeypatch.setattr(
        pygame_view,
        "draw_activation_effect",
        lambda **kwargs: received.append(kwargs),
    )
    monkeypatch.setattr(
        pygame_view,
        "draw_combination_effect",
        lambda **kwargs: received.append(kwargs),
    )

    viewer.draw_special_activation()

    assert received
    assert all(
        item["padding"] == viewer.get_board_origin()[0]
        and item["header_height"] == viewer.get_board_origin()[1]
        for item in received
    )


def test_default_and_demo_keep_board_hit_testing_accurate():
    for viewer in (Match3Viewer(), Match3Viewer(demo_blockers=True)):
        cell = (4, 3)
        assert viewer.cell_at_pixel(*viewer.cell_center(cell)) == cell
        assert viewer.cell_at_pixel(viewer.board_rect.left, viewer.board_rect.top) is None


@pytest.mark.parametrize("level_number", range(1, 7))
def test_terminal_buttons_remain_inside_board_region(level_number):
    viewer = Match3Viewer(level_number=level_number)
    viewer.level_session.status = LevelStatus.WON
    viewer.finish_level_presentation()

    assert viewer.terminal_overlay_visible
    for _action, rect in viewer.terminal_buttons():
        assert viewer.board_rect.contains(rect)


def test_restart_restores_level_board_budget_and_clears_transient_state():
    viewer = Match3Viewer(level_number=2)
    session = viewer.level_session
    initial = session.board.copy()
    viewer.selected = (0, 0)
    viewer.terminal_overlay_visible = True
    viewer.is_animating = True
    viewer.animation_events = ()
    viewer.reset()

    assert viewer.level_session is session
    assert viewer.level_session.moves_remaining == session.config.move_limit
    assert viewer.level_session.status == LevelStatus.IN_PROGRESS
    assert viewer.state is session.board
    assert viewer.selected is None
    assert not viewer.is_animating
    assert not viewer.terminal_overlay_visible
    for name in ("tiles", "specials", "blockers", "blocker_hp", "cell_mask"):
        assert np.array_equal(getattr(viewer.state, name), getattr(initial, name))


def test_terminal_overlays_wait_for_animation_and_navigation_resets_level_state():
    viewer = Match3Viewer(level_number=1)
    viewer.level_session.status = LevelStatus.WON
    viewer.is_animating = True
    viewer.finish_level_presentation()
    assert not viewer.terminal_overlay_visible

    viewer.is_animating = False
    viewer.finish_level_presentation()
    assert viewer.terminal_overlay_visible
    assert any(action == "next" for action, _rect in viewer.terminal_buttons())

    viewer.handle_terminal_action("next")
    assert viewer.level_number == 2
    assert viewer.level_session.status == LevelStatus.IN_PROGRESS
    assert not viewer.terminal_overlay_visible

    viewer.open_level_selection()
    assert viewer.screen_mode == "selection"
    assert viewer.level_session is None
    assert not viewer.animation_events
    viewer.handle_level_selection_key(pygame.K_6)
    assert viewer.level_number == 6
    assert not any(action == "next" for action, _rect in viewer.terminal_buttons())


def test_loss_overlay_and_level_selection_action():
    viewer = Match3Viewer(level_number=3)
    viewer.level_session.status = LevelStatus.LOST
    viewer.finish_level_presentation()
    assert viewer.terminal_overlay_visible
    assert [action for action, _rect in viewer.terminal_buttons()] == ["retry", "select"]
    viewer.handle_terminal_action("select")
    assert viewer.screen_mode == "selection"
    assert viewer.selected_level_number == 3


def test_terminal_and_animation_states_block_board_moves():
    viewer = Match3Viewer(level_number=1)
    session = viewer.level_session
    moves_before = session.moves_remaining
    viewer.level_session.status = LevelStatus.LOST
    click_cell(viewer, (0, 0))
    assert session.moves_remaining == moves_before

    viewer.level_session.status = LevelStatus.IN_PROGRESS
    viewer.is_animating = True
    click_cell(viewer, (0, 0))
    assert session.moves_remaining == moves_before
