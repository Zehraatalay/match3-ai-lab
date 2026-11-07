from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from importlib import resources

import numpy as np
import pygame

from match3_ai.games.match3.animation_events import (
    AnimationEvent,
    AnimationEventType,
)
from match3_ai.games.match3.blockers import BLOCKER_PROPERTIES, BlockerType
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.engine import Match3Engine
from match3_ai.games.match3.generator import generate_playable_board
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.specials import SpecialType
from match3_ai.levels import (
    LevelConfig,
    LevelSession,
    LevelStatus,
    ObjectiveType,
    load_level,
)
from match3_ai.ui.gravity_animation import (
    TileMovement,
    calculate_gravity_movements,
)
from match3_ai.ui.special_effects import (
    draw_activation_effect,
    draw_combination_effect,
    draw_special_effects,
    get_active_special,
)

# Display configuration
CELL_SIZE = 72
PADDING = 24
HUD_TOP = 12
HUD_HEIGHT = 120
SECTION_GAP = 12
CONTROLS_HEIGHT = 56
MAX_LEVEL_ROWS = 9
MAX_LEVEL_COLS = 8
HEADER_HEIGHT = HUD_TOP + HUD_HEIGHT + SECTION_GAP + 6
FPS = 60

BACKGROUND = (24, 29, 43)
BOARD_BACKGROUND = (39, 47, 66)
CELL_BACKGROUND = (55, 65, 88)
TEXT_COLOR = (235, 240, 250)
SELECT_COLOR = (255, 255, 255)
HINT_COLOR = (255, 215, 80)

TILE_COLORS = [
    (235, 90, 90),  # Red
    (80, 175, 255),  # Blue
    (100, 210, 140),  # Green
    (255, 205, 80),  # Yellow
    (175, 120, 245),  # Purple
    (255, 145, 85),  # Orange
]

ANIMATION_DURATIONS = {
    AnimationEventType.SWAP: 250,
    AnimationEventType.MATCH: 400,
    AnimationEventType.SPECIAL_CREATION: 600,
    AnimationEventType.SPECIAL_ACTIVATION: 850,
    AnimationEventType.BLOCKER_DAMAGE: 350,
    AnimationEventType.CLEAR: 300,
    AnimationEventType.GRAVITY: 350,
    AnimationEventType.REFILL: 500,
    AnimationEventType.RESHUFFLE: 500,
}
# Special animation phase durations (milliseconds)
COMBINATION_DURATION = 850
CHAIN_ACTIVATION_DURATION = 450
BLOCKER_DEMO_SEED = 314159
BLOCKER_HIT_DURATION = 240
BLOCKER_MIN_HIT_DURATION = 50
BLOCKER_MAX_DAMAGE_DURATION = 3200
BLOCKER_HP_CHANGE_PROGRESS = 0.32
BLOCKER_DESTRUCTION_PROGRESS = 0.55
SWAP_REJECTION_DURATION = 260
PACKAGED_LEVELS = (
    "first-match.json",
    "wooden-trouble.json",
    "double-trouble.json",
    "frozen-garden.json",
    "stone-breaker.json",
    "blocker-mayhem.json",
)
COLOR_NAMES = ("Red", "Blue", "Green", "Yellow", "Purple", "Orange")


def load_packaged_level(level_number: int) -> LevelConfig:
    if type(level_number) is not int or not 1 <= level_number <= len(PACKAGED_LEVELS):
        raise ValueError(f"level number must be between 1 and {len(PACKAGED_LEVELS)}")

    resource = resources.files("match3_ai.levels").joinpath(
        "data", PACKAGED_LEVELS[level_number - 1]
    )
    with resources.as_file(resource) as level_path:
        return load_level(level_path)


@dataclass(frozen=True, slots=True)
class BlockerDamageVisual:
    hp: int
    damage_count: int
    hit_progress: float | None = None
    destruction_progress: float | None = None
    visible: bool = True


def generate_blocker_demo_board(
    seed: int = BLOCKER_DEMO_SEED,
) -> BoardState:
    state = generate_playable_board(
        rows=8,
        cols=8,
        num_colors=5,
        seed=seed,
    )
    tiles = state.tiles.copy()
    specials = state.specials.copy()
    blockers = np.zeros_like(state.blockers)
    blocker_hp = np.zeros_like(state.blocker_hp)

    fixed_blockers = (
        ((0, 0), BlockerType.CRATE, 1),
        ((0, 7), BlockerType.CRATE, 2),
        ((7, 0), BlockerType.CRATE, 3),
        ((3, 3), BlockerType.STONE, 1),
        ((7, 7), BlockerType.STONE, 2),
    )
    for position, blocker_type, hp in fixed_blockers:
        blockers[position] = blocker_type
        blocker_hp[position] = hp
        tiles[position] = 0
        specials[position] = SpecialType.NONE

    ice_positions = (
        ((1, 2), 1),
        ((4, 4), 2),
    )
    for position, hp in ice_positions:
        blockers[position] = BlockerType.ICE
        blocker_hp[position] = hp

    specials[4, 4] = SpecialType.BOMB

    return BoardState(
        tiles=tiles,
        specials=specials,
        cell_mask=state.cell_mask,
        blockers=blockers,
        blocker_hp=blocker_hp,
    )


class Match3Viewer:
    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        num_colors: int = 5,
        seed: int | None = 42,
        demo_blockers: bool = False,
        level_number: int | None = None,
        level_selection: bool = False,
    ) -> None:
        if demo_blockers and (level_number is not None or level_selection):
            raise ValueError("blocker demo mode cannot be combined with level mode")
        if level_number is not None and level_selection:
            raise ValueError(
                "direct level launch cannot be combined with level selection"
            )
        if demo_blockers and (rows, cols, num_colors) != (8, 8, 5):
            raise ValueError("blocker demo mode requires an 8x8 board with 5 colors")

        self.rows = rows
        self.cols = cols
        self.num_colors = num_colors
        self._configure_layout()
        self.demo_blockers = demo_blockers
        self.level_number: int | None = None
        self.level_config: LevelConfig | None = None
        self.level_session: LevelSession | None = None
        self.screen_mode = "selection" if level_selection else "gameplay"
        self.selected_level_number = 1
        self.level_catalog = (
            tuple(
                (number, load_packaged_level(number))
                for number in range(1, len(PACKAGED_LEVELS) + 1)
            )
            if level_selection
            else ()
        )
        self.terminal_overlay_visible = False
        self.displayed_moves_remaining: int | None = None
        self.displayed_objective_progress = ()
        self.seed = BLOCKER_DEMO_SEED if demo_blockers else seed

        self.state = self._generate_initial_state()
        self.engine = Match3Engine(
            num_colors=num_colors,
            seed=self.seed,
        )

        self.selected: tuple[int, int] | None = None
        self.show_hints = False
        self.show_blocker_hp = False
        self.quit_requested = False
        self.message = "Select two adjacent tiles."
        self.rejected_swap: tuple[tuple[int, int], tuple[int, int]] | None = None
        self.rejection_start_time = 0

        # Animation playback state
        self.animation_events: tuple[AnimationEvent, ...] = ()
        self.animation_index = 0
        self.animation_start_time = 0
        self.is_animating = False
        self.highlighted_cells: frozenset[tuple[int, int]] = frozenset()
        self.current_animation_type: AnimationEventType | None = None
        self.final_state: BoardState | None = None

        # Gravity animation state
        self.previous_animation_state: BoardState | None = None
        self.gravity_movements: tuple[TileMovement, ...] = ()

        pygame.init()

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Match-3 AI Lab")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 22)
        self.small_font = pygame.font.SysFont("arial", 16)
        self.tiny_font = pygame.font.SysFont("arial", 12)
        self.special_font = pygame.font.SysFont("arial", 30, bold=True)

        if level_number is not None:
            self.start_level(level_number)

    def reset_animation(self) -> None:
        self.animation_events = ()
        self.animation_index = 0
        self.animation_start_time = 0
        self.is_animating = False
        self.highlighted_cells = frozenset()
        self.current_animation_type = None
        self.final_state = None

        self.previous_animation_state = None
        self.gravity_movements = ()
        self.rejected_swap = None
        self.rejection_start_time = 0

    def _configure_layout(self) -> None:
        board_width = self.cols * CELL_SIZE
        board_height = self.rows * CELL_SIZE
        layout_width = MAX_LEVEL_COLS * CELL_SIZE
        layout_height = MAX_LEVEL_ROWS * CELL_SIZE
        self.width = layout_width + 2 * PADDING
        self.hud_rect = pygame.Rect(
            PADDING - 6,
            HUD_TOP,
            layout_width + 12,
            HUD_HEIGHT,
        )
        board_region_top = self.hud_rect.bottom + SECTION_GAP
        self.board_region_rect = pygame.Rect(
            PADDING - 6,
            board_region_top,
            layout_width + 12,
            layout_height + 12,
        )
        board_frame_width = board_width + 12
        board_frame_height = board_height + 12
        self.board_rect = pygame.Rect(
            self.board_region_rect.x
            + (self.board_region_rect.width - board_frame_width) // 2,
            self.board_region_rect.y
            + (self.board_region_rect.height - board_frame_height) // 2,
            board_frame_width,
            board_frame_height,
        )
        self.action_bar_rect = pygame.Rect(
            self.board_region_rect.x,
            self.board_region_rect.bottom + SECTION_GAP,
            self.board_region_rect.width,
            CONTROLS_HEIGHT,
        )
        self.controls_rect = self.action_bar_rect.copy()
        self.height = self.action_bar_rect.bottom + PADDING - 6
        self.board_origin = (self.board_rect.x + 6, self.board_rect.y + 6)

    def cell_origin(self, position: tuple[int, int]) -> tuple[int, int]:
        row, col = position
        board_x, board_y = self.get_board_origin()
        return board_x + col * CELL_SIZE, board_y + row * CELL_SIZE

    def get_board_origin(self) -> tuple[int, int]:
        return getattr(self, "board_origin", (PADDING, HEADER_HEIGHT))

    def cell_center(self, position: tuple[int, int]) -> tuple[int, int]:
        x, y = self.cell_origin(position)
        return x + CELL_SIZE // 2, y + CELL_SIZE // 2

    def get_board_rect(self) -> pygame.Rect:
        return getattr(
            self,
            "board_rect",
            pygame.Rect(
                PADDING - 6,
                HEADER_HEIGHT - 6,
                self.cols * CELL_SIZE + 12,
                self.rows * CELL_SIZE + 12,
            ),
        )

    def get_objective_card_rect(self) -> pygame.Rect:
        hud_rect = self.hud_rect
        moves_card_x = hud_rect.x + 12
        card_y = hud_rect.y + 68
        return pygame.Rect(
            moves_card_x + 150 + 12,
            card_y,
            hud_rect.right - 12 - (moves_card_x + 150 + 12),
            46,
        )

    def fit_text(
        self,
        text: str,
        font: pygame.font.Font,
        color: tuple[int, int, int],
        max_width: int,
    ) -> pygame.Surface:
        if font.size(text)[0] <= max_width:
            return font.render(text, True, color)
        value = text
        while value and font.size(f"{value}...")[0] > max_width:
            value = value[:-1]
        return font.render(f"{value}...", True, color)

    def reset(self) -> None:
        if getattr(self, "level_session", None) is not None:
            self.state = self.level_session.reset()
            self.engine = self.level_session.engine
            self.moves_remaining = self.level_session.moves_remaining
            self.displayed_moves_remaining = self.level_session.moves_remaining
            self.displayed_objective_progress = self.level_session.objective_progress
            self.selected = None
            self.show_hints = False
            self.message = "Level restarted."
            self.reset_animation()
            self.terminal_overlay_visible = False
            return

        self.seed = BLOCKER_DEMO_SEED if self.demo_blockers else None
        self.state = self._generate_initial_state()

        self.engine = Match3Engine(
            num_colors=self.num_colors,
            seed=self.seed,
        )

        self.selected = None
        self.show_hints = False
        self.message = "New board generated."

        self.reset_animation()

    def start_level(self, level_number: int) -> None:
        config = load_packaged_level(level_number)
        session = LevelSession(config)
        self.level_number = level_number
        self.selected_level_number = level_number
        self.level_config = config
        self.level_session = session
        self.demo_blockers = False
        self.rows = config.rows
        self.cols = config.cols
        self.num_colors = config.num_colors
        self._configure_layout()
        self.seed = config.seed
        self.state = session.board
        self.engine = session.engine
        self.moves_remaining = session.moves_remaining
        self.displayed_moves_remaining = session.moves_remaining
        self.displayed_objective_progress = session.objective_progress
        self.selected = None
        self.show_hints = False
        self.message = ""
        self.reset_animation()
        self.terminal_overlay_visible = False
        self.screen_mode = "gameplay"

    def open_level_selection(self) -> None:
        if self.is_animating:
            return
        self.screen_mode = "selection"
        if not self.level_catalog:
            self.level_catalog = tuple(
                (number, load_packaged_level(number))
                for number in range(1, len(PACKAGED_LEVELS) + 1)
            )
        self.selected_level_number = self.level_number or 1
        self.level_session = None
        self.level_config = None
        self.level_number = None
        self.selected = None
        self.reset_animation()
        self.terminal_overlay_visible = False

    def advance_to_next_level(self) -> bool:
        if self.level_number is None or self.level_number >= len(PACKAGED_LEVELS):
            return False
        self.start_level(self.level_number + 1)
        return True

    def _generate_initial_state(self) -> BoardState:
        if self.demo_blockers:
            return generate_blocker_demo_board(seed=self.seed)

        return generate_playable_board(
            rows=self.rows,
            cols=self.cols,
            num_colors=self.num_colors,
            seed=self.seed,
        )

    def cell_at_pixel(
        self,
        x: int,
        y: int,
    ) -> tuple[int, int] | None:
        board_rect = self.get_board_rect()
        cell_rect = pygame.Rect(
            board_rect.x + 6,
            board_rect.y + 6,
            self.cols * CELL_SIZE,
            self.rows * CELL_SIZE,
        )
        if not cell_rect.collidepoint(x, y):
            return None

        board_x = x - cell_rect.x
        board_y = y - cell_rect.y
        col = board_x // CELL_SIZE
        row = board_y // CELL_SIZE

        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)

        return None

    def handle_click(self, position: tuple[int, int]) -> None:
        if getattr(self, "screen_mode", "gameplay") == "selection":
            self.handle_level_selection_click(position)
            return
        for action, rect in self.action_bar_buttons():
            if rect.collidepoint(position):
                self.handle_action_bar_action(action)
                return
        if getattr(self, "terminal_overlay_visible", False):
            self.handle_terminal_click(position)
            return
        if (
            getattr(self, "level_session", None) is not None
            and self.level_session.status != LevelStatus.IN_PROGRESS
        ):
            return
        if self.is_animating or self.rejected_swap is not None:
            return

        cell = self.cell_at_pixel(*position)

        if cell is None:
            return

        if self.selected is None:
            if not self.is_selectable_cell(cell):
                return
            self.selected = cell
            self.message = f"Selected {cell}"
            return

        if self.selected == cell:
            self.selected = None
            self.message = "Selection cleared."
            return

        first = self.selected
        second = cell

        if abs(first[0] - second[0]) + abs(first[1] - second[1]) != 1:
            if self.is_selectable_cell(cell):
                self.selected = cell
                self.message = f"Selected {cell}"
            return

        self.selected = None

        if not self.is_selectable_cell(second):
            self.message = "Invalid move."
            self.start_swap_rejection(first, second)
            return

        input_state = self.state
        if getattr(self, "level_session", None) is not None:
            result = self.level_session.step(
                (first, second),
                record_events=True,
            )
        else:
            result = self.engine.step(
                self.state,
                (first, second),
                record_events=True,
            )

        if not result.valid:
            self.message = "Invalid move."
            self.start_swap_rejection(first, second)
            return

        self.message = (
            f"Cleared: {result.cleared_tiles} | Resolutions: {result.cascade_count}"
        )

        if result.reshuffled:
            self.message += " | Reshuffled!"

        self.final_state = result.state
        self.animation_events = result.animation_events
        self.animation_index = 0
        self.is_animating = bool(self.animation_events)

        self.previous_animation_state = None
        self.gravity_movements = ()

        if self.is_animating:
            self.animation_start_time = pygame.time.get_ticks()
            self.previous_animation_state = input_state

            first_event = self.animation_events[0]

            self.state = first_event.state
            self.highlighted_cells = first_event.affected_cells
            self.current_animation_type = first_event.event_type
        else:
            self.state = result.state
            self.reset_animation()
            self.finish_level_presentation()

    def finish_level_presentation(self) -> None:
        if getattr(self, "level_session", None) is None:
            return
        if self.is_animating:
            return
        self.displayed_moves_remaining = self.level_session.moves_remaining
        self.displayed_objective_progress = self.level_session.objective_progress
        self.terminal_overlay_visible = (
            self.level_session.status != LevelStatus.IN_PROGRESS
        )

    def handle_terminal_click(self, position: tuple[int, int]) -> None:
        for action, rect in self.terminal_buttons():
            if rect.collidepoint(position):
                self.handle_terminal_action(action)
                return

    def handle_terminal_action(self, action: str) -> None:
        if action in {"restart", "retry"}:
            self.reset()
        elif action == "next":
            self.advance_to_next_level()
        elif action == "select":
            self.open_level_selection()

    def terminal_buttons(self) -> tuple[tuple[str, pygame.Rect], ...]:
        if getattr(self, "level_session", None) is None:
            return ()
        actions = [
            "restart" if self.level_session.status == LevelStatus.WON else "retry"
        ]
        if self.level_session.status == LevelStatus.WON:
            if self.level_number is not None and self.level_number < len(
                PACKAGED_LEVELS
            ):
                actions.append("next")
            actions.append("select")
        else:
            actions.append("select")
        button_width = 116
        gap = 10
        total_width = len(actions) * button_width + (len(actions) - 1) * gap
        start_x = (self.width - total_width) // 2
        top = self.get_board_rect().centery + 45
        return tuple(
            (
                action,
                pygame.Rect(
                    start_x + index * (button_width + gap), top, button_width, 38
                ),
            )
            for index, action in enumerate(actions)
        )

    def is_selectable_cell(self, position: tuple[int, int]) -> bool:
        row, col = position
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return False
        if not self.state.cell_mask[position] or self.state.tiles[position] == 0:
            return False

        blocker_type = BlockerType(int(self.state.blockers[position]))
        return (
            blocker_type == BlockerType.NONE
            or BLOCKER_PROPERTIES[blocker_type].contains_tile
        )

    def start_swap_rejection(
        self,
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> None:
        self.rejected_swap = (first, second)
        self.rejection_start_time = pygame.time.get_ticks()

    def get_animation_duration(
        self,
        event: AnimationEvent,
    ) -> int:
        """return synchronized animation durations."""

        base_duration = ANIMATION_DURATIONS[event.event_type]

        if event.event_type == AnimationEventType.BLOCKER_DAMAGE:
            if event.blocker_damage:
                return len(event.blocker_damage) * self.get_blocker_hit_duration(event)
            return base_duration

        if event.event_type != AnimationEventType.SPECIAL_ACTIVATION:
            return base_duration

        activation_count = len(event.special_activations)

        if event.special_combination is not None:
            return COMBINATION_DURATION + activation_count * CHAIN_ACTIVATION_DURATION

        if activation_count > 0:
            return activation_count * CHAIN_ACTIVATION_DURATION

        return base_duration

    def get_blocker_hit_duration(self, event: AnimationEvent) -> int:
        hit_count = len(event.blocker_damage)
        if hit_count == 0:
            return BLOCKER_HIT_DURATION

        return max(
            BLOCKER_MIN_HIT_DURATION,
            min(
                BLOCKER_HIT_DURATION,
                BLOCKER_MAX_DAMAGE_DURATION // hit_count,
            ),
        )

    def get_blocker_damage_visuals(
        self,
    ) -> dict[tuple[int, int], BlockerDamageVisual]:
        if (
            not self.is_animating
            or self.current_animation_type != AnimationEventType.BLOCKER_DAMAGE
        ):
            return {}

        event = self.animation_events[self.animation_index]
        records = event.blocker_damage
        if not records:
            return {}

        elapsed = max(0, pygame.time.get_ticks() - self.animation_start_time)
        hit_duration = self.get_blocker_hit_duration(event)
        hit_index = min(elapsed // hit_duration, len(records) - 1)
        hit_progress = min(
            1.0,
            (elapsed - hit_index * hit_duration) / hit_duration,
        )
        visuals: dict[tuple[int, int], BlockerDamageVisual] = {}

        for record in records[:hit_index]:
            previous = visuals.get(record.position)
            damage_count = 1 if previous is None else previous.damage_count + 1
            visuals[record.position] = BlockerDamageVisual(
                hp=record.remaining_hp,
                damage_count=damage_count,
                destruction_progress=(1.0 if record.destroyed else None),
                visible=not record.destroyed,
            )

        current_record = records[hit_index]
        previous = visuals.get(current_record.position)
        damage_count = 0 if previous is None else previous.damage_count
        hit_landed = hit_progress >= BLOCKER_HP_CHANGE_PROGRESS
        if hit_landed:
            damage_count += 1

        destroying = (
            current_record.destroyed and hit_progress >= BLOCKER_DESTRUCTION_PROGRESS
        )
        visuals[current_record.position] = BlockerDamageVisual(
            hp=(
                current_record.remaining_hp
                if hit_landed
                else current_record.previous_hp
            ),
            damage_count=damage_count,
            hit_progress=hit_progress,
            destruction_progress=(
                (hit_progress - BLOCKER_DESTRUCTION_PROGRESS)
                / (1.0 - BLOCKER_DESTRUCTION_PROGRESS)
                if destroying
                else None
            ),
            visible=not destroying,
        )
        return visuals

    def update_animation(self) -> None:
        if getattr(self, "rejected_swap", None) is not None:
            if (
                pygame.time.get_ticks() - self.rejection_start_time
                >= SWAP_REJECTION_DURATION
            ):
                self.rejected_swap = None
                self.rejection_start_time = 0
            return

        if not self.is_animating:
            return

        now = pygame.time.get_ticks()

        # Get the currently playing animation event.
        current_event = self.animation_events[self.animation_index]

        duration = self.get_animation_duration(current_event)

        if now - self.animation_start_time < duration:
            return

        self.animation_index += 1
        self.animation_start_time = now

        if self.animation_index >= len(self.animation_events):
            final_state = self.final_state

            self.reset_animation()

            if final_state is not None:
                self.state = final_state
            self.finish_level_presentation()

            return

        event = self.animation_events[self.animation_index]

        previous_state = self.state

        self.state = event.state
        self.highlighted_cells = event.affected_cells
        self.current_animation_type = event.event_type

        self.previous_animation_state = previous_state
        self.gravity_movements = ()

        if event.event_type == AnimationEventType.GRAVITY:
            self.gravity_movements = calculate_gravity_movements(
                previous_state,
                event.state,
            )

    def get_swap_progress(self) -> float:
        if (
            not self.is_animating
            or self.current_animation_type != AnimationEventType.SWAP
        ):
            return 0.0

        event = self.animation_events[self.animation_index]
        elapsed = max(0, pygame.time.get_ticks() - self.animation_start_time)
        return min(1.0, elapsed / self.get_animation_duration(event))

    def draw_swap_animation(self) -> None:
        if self.previous_animation_state is None:
            return

        event = self.animation_events[self.animation_index]
        positions = sorted(event.affected_cells)
        if len(positions) != 2:
            return

        raw_progress = self.get_swap_progress()
        progress = raw_progress * raw_progress * (3.0 - 2.0 * raw_progress)
        first, second = positions

        for source, destination in ((first, second), (second, first)):
            tile = int(self.previous_animation_state.tiles[source])
            special = SpecialType(int(self.previous_animation_state.specials[source]))
            start_x, start_y = self.cell_center(source)
            end_x, end_y = self.cell_center(destination)
            center = (
                round(start_x + (end_x - start_x) * progress),
                round(start_y + (end_y - start_y) * progress),
            )
            self.draw_tile(tile, special, center)

    def draw_rejected_swap(self) -> None:
        if self.rejected_swap is None:
            return

        elapsed = max(0, pygame.time.get_ticks() - self.rejection_start_time)
        progress = min(1.0, elapsed / SWAP_REJECTION_DURATION)
        offset = 0.18 * math.sin(math.pi * progress)
        first, second = self.rejected_swap

        for source, destination in ((first, second), (second, first)):
            row, col = source
            if not (
                0 <= row < self.rows
                and 0 <= col < self.cols
                and self.state.cell_mask[source]
                and self.state.tiles[source] != 0
            ):
                continue

            tile = int(self.state.tiles[source])
            special = SpecialType(int(self.state.specials[source]))
            start_x, start_y = self.cell_center(source)
            end_x, end_y = self.cell_center(destination)
            center = (
                round(start_x + (end_x - start_x) * offset),
                round(start_y + (end_y - start_y) * offset),
            )
            self.draw_tile(tile, special, center)

    def draw_header(self) -> None:
        hud_rect = getattr(
            self,
            "hud_rect",
            pygame.Rect(
                PADDING - 6, HUD_TOP, self.width - 2 * (PADDING - 6), HUD_HEIGHT
            ),
        )
        pygame.draw.rect(
            self.screen,
            BOARD_BACKGROUND,
            hud_rect,
            border_radius=12,
        )
        content_x = hud_rect.x + 12

        if getattr(self, "level_session", None) is not None:
            title = self.font.render("MATCH-3 AI LAB", True, TEXT_COLOR)
            self.screen.blit(title, (content_x, hud_rect.y + 4))
            level_count = self.tiny_font.render(
                f"LEVEL {self.level_number:02} / {len(PACKAGED_LEVELS):02}",
                True,
                HINT_COLOR,
            )
            self.screen.blit(
                level_count,
                (hud_rect.right - 12 - level_count.get_width(), hud_rect.y + 12),
            )
            level_name = self.fit_text(
                self.level_config.name if self.level_config else "",
                self.font,
                TEXT_COLOR,
                hud_rect.width - 24,
            )
            self.screen.blit(level_name, (content_x, hud_rect.y + 31))

            card_y = hud_rect.y + 68
            moves_card = pygame.Rect(content_x, card_y, 150, 46)
            objective_card = self.get_objective_card_rect()
            for card in (moves_card, objective_card):
                pygame.draw.rect(
                    self.screen,
                    CELL_BACKGROUND,
                    card,
                    border_radius=8,
                )

            moves_label = self.tiny_font.render("MOVES LEFT", True, HINT_COLOR)
            self.screen.blit(moves_label, (moves_card.x + 10, moves_card.y + 4))
            moves_value = self.small_font.render(
                str(self.displayed_moves_remaining),
                True,
                TEXT_COLOR,
            )
            self.screen.blit(moves_value, (moves_card.x + 10, moves_card.y + 20))

            objective_heading = self.tiny_font.render(
                "OBJECTIVES",
                True,
                HINT_COLOR,
            )
            self.screen.blit(
                objective_heading,
                (objective_card.x + 10, objective_card.y + 3),
            )
            objectives = self.level_config.objectives if self.level_config else ()
            for index, (progress, objective) in enumerate(
                zip(self.displayed_objective_progress, objectives)
            ):
                label = self.objective_label(progress, objective.color)
                rendered = self.fit_text(
                    label,
                    self.tiny_font,
                    TEXT_COLOR,
                    objective_card.width - 20,
                )
                self.screen.blit(
                    rendered,
                    (objective_card.x + 10, objective_card.y + 18 + index * 14),
                )
            return

        title = self.font.render("MATCH-3 AI LAB", True, TEXT_COLOR)
        self.screen.blit(title, (content_x, hud_rect.y + 11))
        message_text = self.message
        if self.is_animating and self.current_animation_type is not None:
            message_text = (
                f"Animation: {self.current_animation_type.name}"
                f" ({self.animation_index + 1}/{len(self.animation_events)})"
            )
        message = self.fit_text(
            message_text,
            self.small_font,
            HINT_COLOR,
            hud_rect.width - 24,
        )
        self.screen.blit(message, (content_x, hud_rect.y + 54))

    def action_bar_buttons(self) -> tuple[tuple[str, pygame.Rect], ...]:
        action_bar_rect = getattr(
            self,
            "action_bar_rect",
            getattr(
                self,
                "controls_rect",
                pygame.Rect(
                    0,
                    0,
                    getattr(self, "width", self.screen.get_width()),
                    CONTROLS_HEIGHT,
                ),
            ),
        )
        side_padding = 12
        gap = 8
        button_width = (action_bar_rect.width - 2 * side_padding - 4 * gap) // 5
        used_width = 5 * button_width + 4 * gap
        start_x = action_bar_rect.x + (action_bar_rect.width - used_width) // 2
        actions = ("restart", "hint", "hp", "levels", "quit")
        return tuple(
            (
                action,
                pygame.Rect(
                    start_x + index * (button_width + gap),
                    action_bar_rect.y + 8,
                    button_width,
                    action_bar_rect.height - 16,
                ),
            )
            for index, action in enumerate(actions)
        )

    def handle_action_bar_action(self, action: str) -> None:
        if action == "restart":
            self.reset()
        elif action == "hint":
            self.show_hints = not self.show_hints
        elif action == "hp":
            self.show_blocker_hp = not self.show_blocker_hp
        elif action == "levels":
            self.open_level_selection()
        elif action == "quit":
            self.quit_requested = True

    def action_button_label(self, action: str) -> str:
        if action == "restart":
            return "Restart"
        if action == "hint":
            return "Hint"
        if action == "hp":
            return "HP: ON" if self.show_blocker_hp else "HP: OFF"
        if action == "levels":
            return "Levels"
        if action == "quit":
            return "Quit"
        raise ValueError(f"unknown action bar button: {action}")

    def action_button_colors(
        self,
        action: str,
        hovered: bool = False,
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        active = action == "hp" and self.show_blocker_hp
        active = active or (action == "hint" and self.show_hints)
        if hovered:
            return (74, 96, 128), (168, 196, 226)
        if active:
            return (55, 88, 119), (130, 181, 220)
        return (43, 53, 72), (91, 111, 140)

    def draw_controls(self) -> None:
        action_bar_rect = getattr(
            self,
            "action_bar_rect",
            getattr(
                self,
                "controls_rect",
                pygame.Rect(
                    0,
                    0,
                    getattr(self, "width", self.screen.get_width()),
                    CONTROLS_HEIGHT,
                ),
            ),
        )
        pygame.draw.rect(
            self.screen,
            BOARD_BACKGROUND,
            action_bar_rect,
            border_radius=10,
        )
        mouse_position = pygame.mouse.get_pos()
        for action, button_rect in self.action_bar_buttons():
            fill, border = self.action_button_colors(
                action,
                hovered=button_rect.collidepoint(mouse_position),
            )
            pygame.draw.rect(
                self.screen,
                fill,
                button_rect,
                border_radius=8,
            )
            pygame.draw.rect(
                self.screen,
                border,
                button_rect,
                width=1,
                border_radius=8,
            )
            label = self.fit_text(
                self.action_button_label(action),
                self.tiny_font,
                TEXT_COLOR,
                button_rect.width - 10,
            )
            self.screen.blit(label, label.get_rect(center=button_rect.center))

    def objective_label(self, progress, color: int | None = None) -> str:
        if progress.objective_type == ObjectiveType.CLEAR_ALL_BLOCKERS:
            return (
                f"Blockers: {progress.cleared_blocker_count} / "
                f"{progress.initial_blocker_count} cleared"
            )
        if progress.objective_type == ObjectiveType.COLLECT_TILES:
            if color is not None:
                color_name = (
                    COLOR_NAMES[color - 1]
                    if color <= len(COLOR_NAMES)
                    else f"Color {color}"
                )
                return f"{color_name} tiles: {progress.progress} / {progress.target}"
            return f"Tiles: {progress.progress} / {progress.target}"
        return f"{progress.progress} / {progress.target}"

    def draw_level_selection(self) -> None:
        self.screen.fill(BACKGROUND)
        title = self.font.render("SELECT A LEVEL", True, TEXT_COLOR)
        self.screen.blit(title, (PADDING, PADDING))
        instruction = self.small_font.render(
            "Click a level or use Up/Down and Enter. Press Esc to quit.",
            True,
            TEXT_COLOR,
        )
        self.screen.blit(instruction, (PADDING, PADDING + 38))

        for number, config in self.level_catalog:
            rect = self.level_selection_rect(number)
            selected = number == self.selected_level_number
            color = (77, 98, 132) if selected else BOARD_BACKGROUND
            pygame.draw.rect(self.screen, color, rect, border_radius=10)
            pygame.draw.rect(
                self.screen,
                HINT_COLOR if selected else CELL_BACKGROUND,
                rect,
                width=2,
                border_radius=10,
            )
            label = self.font.render(f"{number}. {config.name}", True, TEXT_COLOR)
            self.screen.blit(label, (rect.x + 18, rect.y + 12))
            details = self.small_font.render(
                f"{config.rows}x{config.cols}   {config.move_limit} moves",
                True,
                TEXT_COLOR,
            )
            self.screen.blit(
                details, (rect.right - details.get_width() - 18, rect.y + 17)
            )
        pygame.display.flip()

    @staticmethod
    def level_selection_rect(level_number: int) -> pygame.Rect:
        return pygame.Rect(
            PADDING,
            PADDING + 76 + (level_number - 1) * 76,
            8 * CELL_SIZE,
            62,
        )

    def handle_level_selection_click(self, position: tuple[int, int]) -> None:
        for number, _config in self.level_catalog:
            if self.level_selection_rect(number).collidepoint(position):
                self.selected_level_number = number
                self.start_level(number)
                return

    def handle_level_selection_key(self, key: int) -> None:
        if key == pygame.K_UP:
            self.selected_level_number = max(1, self.selected_level_number - 1)
        elif key == pygame.K_DOWN:
            self.selected_level_number = min(
                len(PACKAGED_LEVELS),
                self.selected_level_number + 1,
            )
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self.start_level(self.selected_level_number)
        elif pygame.K_1 <= key <= pygame.K_6:
            self.start_level(key - pygame.K_0)

    def draw_terminal_overlay(self) -> None:
        if not self.terminal_overlay_visible or self.level_session is None:
            return
        overlay = pygame.Surface((self.width, self.height), pygame.SRCALPHA)
        board_rect = self.get_board_rect()
        panel = pygame.Rect(
            board_rect.x + 34,
            board_rect.centery - 95,
            board_rect.width - 68,
            190,
        )
        pygame.draw.rect(overlay, (12, 17, 28, 232), panel, border_radius=18)
        pygame.draw.rect(overlay, (139, 164, 202, 255), panel, 2, border_radius=18)
        self.screen.blit(overlay, (0, 0))

        won = self.level_session.status == LevelStatus.WON
        title = "LEVEL COMPLETE" if won else "OUT OF MOVES"
        if won and self.level_number == len(PACKAGED_LEVELS):
            title = "ALL LEVELS COMPLETE"
        title_surface = self.font.render(title, True, HINT_COLOR if won else TEXT_COLOR)
        self.screen.blit(
            title_surface,
            title_surface.get_rect(center=(board_rect.centerx, panel.y + 42)),
        )
        name = self.font.render(self.level_config.name, True, TEXT_COLOR)
        self.screen.blit(
            name,
            name.get_rect(center=(board_rect.centerx, panel.y + 82)),
        )

        for action, rect in self.terminal_buttons():
            pygame.draw.rect(self.screen, (69, 91, 124), rect, border_radius=8)
            pygame.draw.rect(self.screen, (164, 190, 222), rect, 1, border_radius=8)
            label = {
                "restart": "Restart",
                "retry": "Retry",
                "next": "Next Level",
                "select": "Level Select",
            }[action]
            rendered = self.small_font.render(label, True, TEXT_COLOR)
            self.screen.blit(rendered, rendered.get_rect(center=rect.center))

    def draw_special(
        self,
        special: SpecialType,
        center: tuple[int, int],
    ) -> None:
        x, y = center

        white = (255, 255, 255)

        if special == SpecialType.ROCKET_HORIZONTAL:
            pygame.draw.line(
                self.screen,
                white,
                (x - 18, y),
                (x + 18, y),
                5,
            )

            pygame.draw.polygon(
                self.screen,
                white,
                [(x - 22, y), (x - 10, y - 9), (x - 10, y + 9)],
            )

            pygame.draw.polygon(
                self.screen,
                white,
                [(x + 22, y), (x - 9 + 19, y - 9), (x + 10, y + 9)],
            )

        elif special == SpecialType.ROCKET_VERTICAL:
            pygame.draw.line(
                self.screen,
                white,
                (x, y - 18),
                (x, y + 18),
                5,
            )

            pygame.draw.polygon(
                self.screen,
                white,
                [(x, y - 22), (x - 9, y - 10), (x + 9, y - 10)],
            )

            pygame.draw.polygon(
                self.screen,
                white,
                [(x, y + 22), (x - 9, y + 10), (x + 9, y + 10)],
            )

        elif special == SpecialType.BOMB:
            pygame.draw.circle(
                self.screen,
                (35, 35, 45),
                center,
                17,
            )

            pygame.draw.circle(
                self.screen,
                white,
                center,
                17,
                3,
            )

            label = self.special_font.render(
                "B",
                True,
                white,
            )

            self.screen.blit(
                label,
                label.get_rect(center=center),
            )

        elif special == SpecialType.PROPELLER:
            for dx, dy in [
                (-13, 0),
                (13, 0),
                (0, -13),
                (0, 13),
            ]:
                pygame.draw.circle(
                    self.screen,
                    white,
                    (x + dx, y + dy),
                    9,
                    2,
                )

            pygame.draw.circle(
                self.screen,
                (35, 35, 45),
                center,
                7,
            )

        elif special == SpecialType.LIGHT_BALL:
            pygame.draw.circle(
                self.screen,
                (30, 30, 40),
                center,
                23,
            )

            for index, color in enumerate(TILE_COLORS):
                angle = index * (2 * math.pi / len(TILE_COLORS))

                dot_x = x + int(15 * math.cos(angle))
                dot_y = y + int(15 * math.sin(angle))

                pygame.draw.circle(
                    self.screen,
                    color,
                    (dot_x, dot_y),
                    5,
                )

            pygame.draw.circle(
                self.screen,
                white,
                center,
                23,
                2,
            )

    def draw_tile(
        self,
        tile: int,
        special: SpecialType,
        center: tuple[int, int],
    ) -> None:
        if special == SpecialType.LIGHT_BALL:
            self.draw_special(special, center)
            return

        if tile <= 0:
            return

        color = TILE_COLORS[(tile - 1) % len(TILE_COLORS)]

        tile_rect = pygame.Rect(0, 0, CELL_SIZE - 18, CELL_SIZE - 18)
        tile_rect.center = center

        pygame.draw.rect(
            self.screen,
            color,
            tile_rect,
            border_radius=12,
        )

        if special != SpecialType.NONE:
            self.draw_special(special, center)

    def draw_blocker(
        self,
        blocker_type: BlockerType,
        hp: int,
        center: tuple[int, int],
        damage_count: int = 0,
    ) -> None:
        x, y = center
        size = CELL_SIZE - 10
        rect = pygame.Rect(0, 0, size, size)
        rect.center = center
        crack_level = min(3, max(damage_count, 3 - hp))

        if blocker_type == BlockerType.CRATE:
            pygame.draw.rect(
                self.screen,
                (63, 39, 29),
                rect,
                border_radius=9,
            )
            wood_rect = rect.inflate(-6, -6)
            pygame.draw.rect(
                self.screen,
                (151, 91, 48),
                wood_rect,
                border_radius=6,
            )
            pygame.draw.rect(
                self.screen,
                (194, 132, 76),
                wood_rect.inflate(-8, -8),
                2,
                border_radius=4,
            )
            pygame.draw.line(
                self.screen,
                (105, 60, 37),
                (wood_rect.left + 7, wood_rect.top + 5),
                (wood_rect.left + 7, wood_rect.bottom - 5),
                2,
            )
            pygame.draw.line(
                self.screen,
                (105, 60, 37),
                (wood_rect.right - 7, wood_rect.top + 5),
                (wood_rect.right - 7, wood_rect.bottom - 5),
                2,
            )
            nail_color = (224, 190, 137)
            for nail_position in (
                (wood_rect.left + 5, wood_rect.top + 5),
                (wood_rect.right - 5, wood_rect.top + 5),
                (wood_rect.left + 5, wood_rect.bottom - 5),
                (wood_rect.right - 5, wood_rect.bottom - 5),
            ):
                pygame.draw.circle(self.screen, nail_color, nail_position, 2)

            crate_cracks = (
                [(x - 17, y - 7), (x - 9, y - 2), (x - 12, y + 6)],
                [(x + 5, y - 15), (x + 1, y - 7), (x + 9, y - 2)],
                [(x + 10, y + 3), (x + 4, y + 9), (x + 12, y + 15)],
            )
            for crack in crate_cracks[:crack_level]:
                pygame.draw.lines(self.screen, (67, 43, 32), False, crack, 2)

            if self.show_blocker_hp:
                self.draw_blocker_hp_badge(hp, (rect.right - 3, rect.top + 3))
            return

        if blocker_type == BlockerType.STONE:
            points = [
                (rect.left + 8, rect.top),
                (rect.right - 8, rect.top),
                (rect.right, rect.top + 9),
                (rect.right - 3, rect.bottom - 8),
                (rect.right - 12, rect.bottom),
                (rect.left + 7, rect.bottom - 2),
                (rect.left, rect.bottom - 12),
                (rect.left + 2, rect.top + 8),
            ]
            pygame.draw.polygon(self.screen, (50, 64, 82), points)
            inner_points = [
                (rect.left + 9, rect.top + 5),
                (rect.right - 10, rect.top + 5),
                (rect.right - 5, rect.top + 12),
                (rect.right - 9, rect.bottom - 11),
                (rect.right - 15, rect.bottom - 7),
                (rect.left + 10, rect.bottom - 8),
                (rect.left + 6, rect.bottom - 15),
            ]
            pygame.draw.polygon(self.screen, (119, 143, 165), inner_points)
            pygame.draw.polygon(
                self.screen,
                (157, 177, 194),
                [
                    (rect.left + 12, rect.top + 8),
                    (x + 1, rect.top + 9),
                    (x - 8, y - 2),
                    (rect.left + 9, y + 1),
                ],
            )
            pygame.draw.polygon(
                self.screen,
                (91, 113, 137),
                [
                    (x + 2, y - 2),
                    (rect.right - 10, rect.top + 15),
                    (rect.right - 8, rect.bottom - 13),
                    (x + 8, y + 11),
                ],
            )
            pygame.draw.lines(
                self.screen,
                (54, 67, 84),
                False,
                [
                    (x - 18, y - 12),
                    (x - 8, y - 7),
                    (x - 4, y + 1),
                    (x + 5, y + 6),
                    (x + 2, y + 16),
                ],
                3,
            )
            pygame.draw.line(
                self.screen,
                (199, 216, 231),
                (x - 18, y - 12),
                (x - 8, y - 7),
                1,
            )
            if crack_level >= 2:
                pygame.draw.lines(
                    self.screen,
                    (54, 67, 84),
                    False,
                    [(x + 5, y + 6), (x + 12, y + 1), (x + 17, y + 5)],
                    2,
                )
            if crack_level >= 3:
                pygame.draw.lines(
                    self.screen,
                    (54, 67, 84),
                    False,
                    [(x - 8, y - 7), (x - 12, y - 16), (x - 6, y - 20)],
                    2,
                )
            if self.show_blocker_hp:
                self.draw_blocker_hp_badge(hp, (rect.right - 3, rect.top + 3))
            return

        if blocker_type == BlockerType.ICE:
            overlay = pygame.Surface((size, size), pygame.SRCALPHA)
            overlay_rect = pygame.Rect(2, 2, size - 4, size - 4)
            pygame.draw.rect(
                overlay,
                (94, 205, 235, 68),
                overlay_rect,
                border_radius=9,
            )
            pygame.draw.rect(
                overlay,
                (175, 239, 255, 225),
                overlay_rect,
                width=3,
                border_radius=9,
            )
            pygame.draw.lines(
                overlay,
                (220, 249, 255, 190),
                False,
                [(8, 15), (18, 21), (24, 30), (34, 34), (43, 47)],
                2,
            )
            pygame.draw.line(
                overlay,
                (220, 249, 255, 170),
                (18, 21),
                (12, 31),
                2,
            )
            pygame.draw.line(
                overlay,
                (220, 249, 255, 170),
                (34, 34),
                (43, 27),
                2,
            )
            extra_ice_cracks = (
                [(18, 21), (11, 16), (9, 9)],
                [(34, 34), (42, 37), (49, 35)],
                [(24, 30), (20, 38), (14, 42)],
            )
            for crack in extra_ice_cracks[:crack_level]:
                pygame.draw.lines(
                    overlay,
                    (238, 252, 255, 210),
                    False,
                    crack,
                    2,
                )
            self.screen.blit(overlay, rect.topleft)
            if self.show_blocker_hp:
                self.draw_blocker_hp_badge(hp, (rect.right - 3, rect.top + 3))

    def draw_blocker_impact(
        self,
        blocker_type: BlockerType,
        center: tuple[int, int],
        progress: float,
    ) -> None:
        pulse = max(0.0, 1.0 - progress / 0.48)
        if pulse <= 0:
            return

        colors = {
            BlockerType.CRATE: (255, 196, 105),
            BlockerType.STONE: (211, 232, 250),
            BlockerType.ICE: (160, 245, 255),
        }
        size = CELL_SIZE - 10
        overlay = pygame.Surface((size, size), pygame.SRCALPHA)
        alpha = int(190 * pulse)
        pygame.draw.rect(
            overlay,
            (*colors[blocker_type], alpha),
            pygame.Rect(3, 3, size - 6, size - 6),
            width=4,
            border_radius=9,
        )
        local_center = (size // 2, size // 2)
        for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            pygame.draw.line(
                overlay,
                (*colors[blocker_type], alpha),
                (local_center[0] + dx * 10, local_center[1] + dy * 10),
                (local_center[0] + dx * 17, local_center[1] + dy * 17),
                2,
            )
        self.screen.blit(
            overlay,
            (center[0] - size // 2, center[1] - size // 2),
        )

    def draw_blocker_destruction(
        self,
        blocker_type: BlockerType,
        center: tuple[int, int],
        progress: float,
    ) -> None:
        colors = {
            BlockerType.CRATE: (188, 119, 65),
            BlockerType.STONE: (157, 183, 204),
            BlockerType.ICE: (157, 235, 255),
        }
        size = CELL_SIZE - 10
        overlay = pygame.Surface((size, size), pygame.SRCALPHA)
        local_center = (size // 2, size // 2)
        alpha = int(225 * (1.0 - progress))
        for dx, dy in ((-1, -1), (1, -1), (-1, 1), (1, 1), (0, -1)):
            distance = 7 + int(17 * progress)
            px = local_center[0] + dx * distance
            py = local_center[1] + dy * distance
            radius = max(1, int(5 * (1.0 - progress)))
            shard = [
                (px - radius, py - radius),
                (px + radius + 1, py),
                (px - radius, py + radius + 1),
            ]
            pygame.draw.polygon(overlay, (*colors[blocker_type], alpha), shard)
        self.screen.blit(
            overlay,
            (center[0] - size // 2, center[1] - size // 2),
        )

    def draw_blocker_hp_badge(
        self,
        hp: int,
        top_right: tuple[int, int],
    ) -> None:
        label = self.small_font.render(str(hp), True, (255, 255, 255))
        badge = pygame.Rect(0, 0, max(20, label.get_width() + 8), 19)
        badge.topright = top_right
        pygame.draw.rect(
            self.screen,
            (28, 36, 49),
            badge,
            border_radius=7,
        )
        pygame.draw.rect(
            self.screen,
            (220, 232, 242),
            badge,
            width=1,
            border_radius=7,
        )
        self.screen.blit(label, label.get_rect(center=badge.center))

    def draw_blockers(self, state: BoardState) -> None:
        damage_visuals = self.get_blocker_damage_visuals()
        blocker_state = state
        if (
            self.is_animating
            and self.current_animation_type == AnimationEventType.BLOCKER_DAMAGE
            and self.previous_animation_state is not None
        ):
            blocker_state = self.previous_animation_state

        for row in range(self.rows):
            for col in range(self.cols):
                if not blocker_state.cell_mask[row, col]:
                    continue

                position = (row, col)
                blocker_type = BlockerType(int(blocker_state.blockers[position]))
                if blocker_type == BlockerType.NONE:
                    continue

                damage_visual = damage_visuals.get(position)
                center = (*self.cell_center((row, col)),)
                if damage_visual is not None and not damage_visual.visible:
                    self.draw_blocker_destruction(
                        blocker_type,
                        center,
                        damage_visual.destruction_progress or 0.0,
                    )
                    continue

                self.draw_blocker(
                    blocker_type,
                    (
                        damage_visual.hp
                        if damage_visual is not None
                        else int(blocker_state.blocker_hp[position])
                    ),
                    center,
                    damage_count=(
                        damage_visual.damage_count if damage_visual is not None else 0
                    ),
                )
                if damage_visual is not None and damage_visual.hit_progress is not None:
                    self.draw_blocker_impact(
                        blocker_type,
                        center,
                        damage_visual.hit_progress,
                    )

    def draw_gravity_animation(self) -> None:
        if self.previous_animation_state is None:
            return

        duration = ANIMATION_DURATIONS[AnimationEventType.GRAVITY]
        elapsed = pygame.time.get_ticks() - self.animation_start_time

        progress = min(1.0, max(0.0, elapsed / duration))

        # Ease-out cubic: fast start, smooth landing.
        progress = 1 - (1 - progress) ** 3

        moving_destinations = {
            movement.destination for movement in self.gravity_movements
        }

        # Draw tiles that do not move during gravity.
        for row in range(self.rows):
            for col in range(self.cols):
                if not self.state.cell_mask[row, col]:
                    continue

                if (row, col) in moving_destinations:
                    continue

                tile = int(self.state.tiles[row, col])
                special = SpecialType(int(self.state.specials[row, col]))

                center = (*self.cell_center((row, col)),)

                self.draw_tile(tile, special, center)

        # Draw moving tiles using their original appearance.
        for movement in self.gravity_movements:
            source_row, _ = movement.source
            destination_row, destination_col = movement.destination

            tile = int(self.previous_animation_state.tiles[movement.source])
            special = SpecialType(
                int(self.previous_animation_state.specials[movement.source])
            )

            interpolated_row = source_row + (destination_row - source_row) * progress

            center = (
                self.cell_center((0, destination_col))[0],
                int(
                    self.get_board_origin()[1]
                    + interpolated_row * CELL_SIZE
                    + CELL_SIZE // 2
                ),
            )

            self.draw_tile(tile, special, center)

    def draw_refill_animation(self) -> None:
        if self.previous_animation_state is None:
            return

        duration = ANIMATION_DURATIONS[AnimationEventType.REFILL]
        elapsed = pygame.time.get_ticks() - self.animation_start_time

        progress = min(1.0, max(0.0, elapsed / duration))

        # Ease-out cubic: fast start, smooth landing.
        progress = 1 - (1 - progress) ** 3

        for col in range(self.cols):
            new_tiles = [
                row
                for row in range(self.rows)
                if (
                    self.state.cell_mask[row, col]
                    and int(self.previous_animation_state.tiles[row, col]) == 0
                    and int(self.state.tiles[row, col]) != 0
                )
            ]

            new_tile_count = len(new_tiles)

            for row in range(self.rows):
                if not self.state.cell_mask[row, col]:
                    continue

                tile = int(self.state.tiles[row, col])
                special = SpecialType(int(self.state.specials[row, col]))

                if tile <= 0 and special != SpecialType.LIGHT_BALL:
                    continue

                if row in new_tiles:
                    # Spawn new tiles above the board.
                    spawn_index = new_tiles.index(row)
                    start_row = spawn_index - new_tile_count

                    interpolated_row = start_row + (row - start_row) * progress

                    center_y = int(
                        self.get_board_origin()[1]
                        + interpolated_row * CELL_SIZE
                        + CELL_SIZE // 2
                    )
                else:
                    center_y = self.cell_center((row, col))[1]

                center_x = self.cell_center((row, col))[0]

                self.draw_tile(
                    tile,
                    special,
                    (center_x, center_y),
                )

    def get_progressive_clear_cells(
        self,
        activation,
        progress: float,
    ) -> set[tuple[int, int]]:
        """Return cells visually cleared at the current activation progress."""

        progress = max(0.0, min(1.0, progress))

        source_row, source_col = activation.position
        special_type = activation.special_type

        affected_cells = activation.effect_cells

        if not affected_cells:
            return set()

        # Horizontal Rocket: clear outward from the source column.
        if special_type == SpecialType.ROCKET_HORIZONTAL:
            max_distance = max(abs(col - source_col) for _, col in affected_cells)

            return {
                (row, col)
                for row, col in affected_cells
                if abs(col - source_col) <= max_distance * progress
            }

        # Vertical Rocket: clear outward from the source row.
        if special_type == SpecialType.ROCKET_VERTICAL:
            max_distance = max(abs(row - source_row) for row, _ in affected_cells)

            return {
                (row, col)
                for row, col in affected_cells
                if abs(row - source_row) <= max_distance * progress
            }

        # Bomb: clear outward based on Euclidean distance.
        if special_type == SpecialType.BOMB:
            max_distance = max(
                math.hypot(row - source_row, col - source_col)
                for row, col in affected_cells
            )

            return {
                (row, col)
                for row, col in affected_cells
                if math.hypot(
                    row - source_row,
                    col - source_col,
                )
                <= max_distance * progress
            }

        # Other specials: progressively reveal affected cells.
        ordered_cells = sorted(affected_cells)

        clear_count = int(len(ordered_cells) * progress)

        return set(ordered_cells[:clear_count])

    def get_visible_animation_state(self) -> BoardState:
        """Build a visual board while preserving pending chain activations."""

        if (
            not self.is_animating
            or self.current_animation_type != AnimationEventType.SPECIAL_ACTIVATION
        ):
            return self.state

        event = self.animation_events[self.animation_index]
        activations = event.special_activations

        if not activations:
            return self.state

        elapsed = pygame.time.get_ticks() - self.animation_start_time

        # Wait until the combination animation finishes.
        if event.special_combination is not None:
            combination_duration = COMBINATION_DURATION

            if elapsed < combination_duration:
                return self.state

            elapsed -= combination_duration

        activation_duration = CHAIN_ACTIVATION_DURATION
        activation_count = len(activations)

        activation_index = min(
            int(elapsed / activation_duration),
            activation_count - 1,
        )

        local_elapsed = elapsed - activation_index * activation_duration

        local_progress = min(
            1.0,
            max(0.0, local_elapsed / activation_duration),
        )

        visual_state = self.state.copy()

        # Collect cells cleared by completed activations.
        cells_to_clear: set[tuple[int, int]] = set()

        for previous_activation in activations[:activation_index]:
            cells_to_clear.update(previous_activation.effect_cells)
            cells_to_clear.add(previous_activation.position)

        # Progressively clear cells affected by the active special.
        current_activation = activations[activation_index]

        cells_to_clear.update(
            self.get_progressive_clear_cells(
                current_activation,
                local_progress,
            )
        )
        if local_progress > 0:
            cells_to_clear.add(current_activation.position)

        # Preserve specials whose activation has not started yet.
        pending_positions = {
            activation.position for activation in activations[activation_index + 1 :]
        }

        cells_to_clear.difference_update(pending_positions)

        # Apply visual clearing without modifying the actual game state.
        for row, col in cells_to_clear:
            if (
                0 <= row < self.rows
                and 0 <= col < self.cols
                and visual_state.cell_mask[row, col]
            ):
                visual_state.tiles[row, col] = 0
                visual_state.specials[row, col] = SpecialType.NONE

        return visual_state

    def get_transformed_cells(self) -> set[tuple[int, int]]:
        """Find cells whose special type changed during creation."""

        if (
            not self.is_animating
            or self.current_animation_type != AnimationEventType.SPECIAL_CREATION
            or self.previous_animation_state is None
        ):
            return set()

        transformed_cells: set[tuple[int, int]] = set()

        previous_state = self.previous_animation_state

        for row in range(self.rows):
            for col in range(self.cols):
                if not self.state.cell_mask[row, col]:
                    continue

                previous_special = SpecialType(int(previous_state.specials[row, col]))

                current_special = SpecialType(int(self.state.specials[row, col]))

                if (
                    previous_special != current_special
                    and current_special != SpecialType.NONE
                ):
                    transformed_cells.add((row, col))

        return transformed_cells

    def draw_special_creation_animation(self) -> None:
        """Animate newly created specials with a flash effect."""

        if (
            not self.is_animating
            or self.current_animation_type != AnimationEventType.SPECIAL_CREATION
            or self.previous_animation_state is None
        ):
            return

        transformed_cells = self.get_transformed_cells()

        if not transformed_cells:
            return

        duration = ANIMATION_DURATIONS[AnimationEventType.SPECIAL_CREATION]

        elapsed = pygame.time.get_ticks() - self.animation_start_time

        progress = min(
            1.0,
            max(0.0, elapsed / duration),
        )

        overlay = pygame.Surface(
            self.screen.get_size(),
            pygame.SRCALPHA,
        )

        for row, col in transformed_cells:
            center = (*self.cell_center((row, col)),)

            # Phase 1: Keep the original tile visible.
            if progress < 0.25:
                continue

            # Phase 2: Transformation flash.
            if progress < 0.60:
                flash_progress = (progress - 0.25) / 0.35

                pulse = math.sin(flash_progress * math.pi)

                radius = max(
                    1,
                    int(12 + 28 * pulse),
                )

                alpha = int(220 * pulse)

                pygame.draw.circle(
                    overlay,
                    (255, 245, 170, alpha),
                    center,
                    radius,
                )

                pygame.draw.circle(
                    overlay,
                    (255, 255, 255, min(255, alpha + 20)),
                    center,
                    max(1, int(radius * 0.6)),
                )

            # Phase 3: Special appearance pulse.
            else:
                appear_progress = (progress - 0.60) / 0.40

                radius = max(
                    1,
                    int(30 * (1.0 - appear_progress)),
                )

                alpha = int(160 * (1.0 - appear_progress))

                pygame.draw.circle(
                    overlay,
                    (255, 255, 200, alpha),
                    center,
                    radius,
                    width=4,
                )

        self.screen.blit(overlay, (0, 0))

    def draw_special_activation(self) -> None:
        """render combination effects followed by chain reactions."""

        if (
            not self.is_animating
            or self.current_animation_type != AnimationEventType.SPECIAL_ACTIVATION
        ):
            return

        event = self.animation_events[self.animation_index]

        duration = self.get_animation_duration(event)
        elapsed = pygame.time.get_ticks() - self.animation_start_time

        progress = min(
            1.0,
            max(0.0, elapsed / duration),
        )

        combination = event.special_combination

        # --------------------------------------------------
        # Combination phase
        # --------------------------------------------------

        if combination is not None:
            combination_duration = COMBINATION_DURATION

            if elapsed < combination_duration:
                combination_progress = min(
                    1.0,
                    max(0.0, elapsed / combination_duration),
                )

                draw_activation_effect(
                    surface=self.screen,
                    affected_cells=combination.effect_cells,
                    progress=combination_progress,
                    cell_size=CELL_SIZE,
                    padding=self.get_board_origin()[0],
                    header_height=self.get_board_origin()[1],
                )

                draw_combination_effect(
                    surface=self.screen,
                    combination=combination,
                    progress=combination_progress,
                    cell_size=CELL_SIZE,
                    padding=self.get_board_origin()[0],
                    header_height=self.get_board_origin()[1],
                    rows=self.rows,
                    cols=self.cols,
                )

                return

            # Remaining time is reserved for chain reactions.
            elapsed -= combination_duration
            duration -= combination_duration

            if duration <= 0:
                return

            progress = min(
                1.0,
                max(0.0, elapsed / duration),
            )

        # --------------------------------------------------
        # Chain reaction phase
        # --------------------------------------------------

        active_special, local_progress = get_active_special(
            event.special_activations,
            progress,
        )

        if active_special is None:
            if combination is None:
                draw_activation_effect(
                    surface=self.screen,
                    affected_cells=event.affected_cells,
                    progress=progress,
                    cell_size=CELL_SIZE,
                    padding=self.get_board_origin()[0],
                    header_height=self.get_board_origin()[1],
                )

            return

        draw_activation_effect(
            surface=self.screen,
            affected_cells=active_special.effect_cells,
            progress=local_progress,
            cell_size=CELL_SIZE,
            padding=self.get_board_origin()[0],
            header_height=self.get_board_origin()[1],
        )

        draw_special_effects(
            surface=self.screen,
            activations=(active_special,),
            progress=local_progress,
            cell_size=CELL_SIZE,
            padding=self.get_board_origin()[0],
            header_height=self.get_board_origin()[1],
            rows=self.rows,
            cols=self.cols,
        )

    def draw_board(self) -> None:
        visual_state = self.get_visible_animation_state()

        board_rect = self.get_board_rect()

        pygame.draw.rect(
            self.screen,
            BOARD_BACKGROUND,
            board_rect,
            border_radius=12,
        )

        # Draw cell backgrounds.
        for row in range(self.rows):
            for col in range(self.cols):
                if not self.state.cell_mask[row, col]:
                    continue

                x, y = self.cell_origin((row, col))

                cell_rect = pygame.Rect(
                    x + 3,
                    y + 3,
                    CELL_SIZE - 6,
                    CELL_SIZE - 6,
                )

                pygame.draw.rect(
                    self.screen,
                    CELL_BACKGROUND,
                    cell_rect,
                    border_radius=10,
                )

        # Draw gravity animation instead of static tiles.
        if (
            self.is_animating
            and self.current_animation_type == AnimationEventType.GRAVITY
            and self.previous_animation_state is not None
        ):
            self.draw_gravity_animation()

        elif (
            self.is_animating
            and self.current_animation_type == AnimationEventType.REFILL
            and self.previous_animation_state is not None
        ):
            self.draw_refill_animation()

        else:
            moving_cells: set[tuple[int, int]] = set()
            if (
                self.is_animating
                and self.current_animation_type == AnimationEventType.SWAP
            ):
                moving_cells.update(
                    self.animation_events[self.animation_index].affected_cells
                )
            elif getattr(self, "rejected_swap", None) is not None:
                moving_cells.update(self.rejected_swap)

            for row in range(self.rows):
                for col in range(self.cols):
                    if not visual_state.cell_mask[row, col]:
                        continue

                    if (row, col) in moving_cells:
                        continue

                    tile = int(visual_state.tiles[row, col])
                    special = SpecialType(int(visual_state.specials[row, col]))

                    center = self.cell_center((row, col))

                    if (
                        self.is_animating
                        and self.current_animation_type
                        == AnimationEventType.SPECIAL_CREATION
                        and self.previous_animation_state is not None
                    ):
                        transformed_cells = self.get_transformed_cells()

                        if (row, col) in transformed_cells:
                            duration = ANIMATION_DURATIONS[
                                AnimationEventType.SPECIAL_CREATION
                            ]

                            elapsed = (
                                pygame.time.get_ticks() - self.animation_start_time
                            )

                            progress = min(
                                1.0,
                                max(0.0, elapsed / duration),
                            )

                            if progress < 0.60:
                                tile = int(
                                    self.previous_animation_state.tiles[row, col]
                                )

                                special = SpecialType(
                                    int(
                                        self.previous_animation_state.specials[row, col]
                                    )
                                )

                    self.draw_tile(tile, special, center)

            if (
                self.is_animating
                and self.current_animation_type == AnimationEventType.SWAP
            ):
                self.draw_swap_animation()
            elif getattr(self, "rejected_swap", None) is not None:
                self.draw_rejected_swap()

        self.draw_blockers(visual_state)

        # Draw highlights and selection borders.
        for row in range(self.rows):
            for col in range(self.cols):
                if not self.state.cell_mask[row, col]:
                    continue

                x, y = self.cell_origin((row, col))

                cell_rect = pygame.Rect(
                    x + 3,
                    y + 3,
                    CELL_SIZE - 6,
                    CELL_SIZE - 6,
                )

                if (
                    (row, col) in self.highlighted_cells
                    and self.current_animation_type
                    != AnimationEventType.SPECIAL_ACTIVATION
                ):
                    pygame.draw.rect(
                        self.screen,
                        HINT_COLOR,
                        cell_rect,
                        width=4,
                        border_radius=10,
                    )
                if self.selected == (row, col):
                    pygame.draw.rect(
                        self.screen,
                        SELECT_COLOR,
                        cell_rect,
                        width=4,
                        border_radius=10,
                    )

    def draw_hints(self) -> None:
        if (
            not self.show_hints
            or self.is_animating
            or getattr(self, "rejected_swap", None) is not None
        ):
            return

        moves = find_valid_moves(self.state)

        for first, second in moves:
            for row, col in (first, second):
                x, y = self.cell_center((row, col))

                pygame.draw.circle(
                    self.screen,
                    HINT_COLOR,
                    (x, y),
                    7,
                )

    def draw(self) -> None:
        if getattr(self, "screen_mode", "gameplay") == "selection":
            self.draw_level_selection()
            return
        self.screen.fill(BACKGROUND)

        self.draw_header()
        previous_clip = self.screen.get_clip()
        self.screen.set_clip(self.get_board_rect())
        self.draw_board()
        self.draw_hints()

        self.draw_special_creation_animation()
        self.draw_special_activation()
        self.draw_terminal_overlay()
        self.screen.set_clip(previous_clip)
        self.draw_controls()

        pygame.display.flip()

    def run(self) -> None:
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if self.screen_mode == "selection":
                        if event.key == pygame.K_ESCAPE:
                            running = False
                        else:
                            self.handle_level_selection_key(event.key)
                    elif event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_r:
                        self.reset()

                    elif event.key == pygame.K_h:
                        self.show_hints = not self.show_hints

                    elif event.key == pygame.K_d:
                        self.show_blocker_hp = not self.show_blocker_hp
                    elif event.key == pygame.K_l and self.level_session is not None:
                        self.open_level_selection()
                    elif (
                        event.key == pygame.K_n
                        and self.terminal_overlay_visible
                        and self.level_session is not None
                        and self.level_session.status == LevelStatus.WON
                    ):
                        self.advance_to_next_level()

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            if self.quit_requested:
                running = False

            if self.screen_mode == "gameplay":
                self.update_animation()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Play Match-3 AI Lab")
    parser.add_argument(
        "--demo-blockers",
        action="store_true",
        help="start with a reproducible board containing blockers",
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--levels",
        action="store_true",
        help="open the packaged level selection screen",
    )
    mode.add_argument(
        "--level",
        type=int,
        choices=range(1, len(PACKAGED_LEVELS) + 1),
        metavar="NUMBER",
        help="start a packaged level directly (1-6)",
    )
    args = parser.parse_args(argv)

    if args.demo_blockers and (args.levels or args.level is not None):
        parser.error("--demo-blockers cannot be combined with level mode")

    viewer = Match3Viewer(
        rows=8,
        cols=8,
        num_colors=5,
        seed=42,
        demo_blockers=args.demo_blockers,
        level_number=args.level,
        level_selection=args.levels,
    )

    viewer.run()


if __name__ == "__main__":
    main()
