from __future__ import annotations

import math

import pygame

from match3_ai.games.match3.animation_events import (
    AnimationEvent,
    AnimationEventType,
)
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.engine import Match3Engine
from match3_ai.games.match3.generator import generate_playable_board
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.specials import SpecialType
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
HEADER_HEIGHT = 110
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
    AnimationEventType.CLEAR: 300,
    AnimationEventType.GRAVITY: 350,
    AnimationEventType.REFILL: 500,
    AnimationEventType.RESHUFFLE: 500,
}
# Special animation phase durations (milliseconds)
COMBINATION_DURATION = 850
CHAIN_ACTIVATION_DURATION = 450

class Match3Viewer:
    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        num_colors: int = 5,
        seed: int | None = 42,
    ) -> None:
        self.rows = rows
        self.cols = cols
        self.num_colors = num_colors
        self.seed = seed

        self.state = generate_playable_board(
            rows=rows,
            cols=cols,
            num_colors=num_colors,
            seed=seed,
        )

        self.engine = Match3Engine(
            num_colors=num_colors,
            seed=seed,
        )

        self.selected: tuple[int, int] | None = None
        self.show_hints = False
        self.message = "Select two adjacent tiles."

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

        self.width = cols * CELL_SIZE + 2 * PADDING
        self.height = rows * CELL_SIZE + HEADER_HEIGHT + 2 * PADDING

        pygame.init()

        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Match-3 AI Lab")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 22)
        self.small_font = pygame.font.SysFont("arial", 16)
        self.special_font = pygame.font.SysFont("arial", 30, bold=True)

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

    def reset(self) -> None:
        self.seed = None

        self.state = generate_playable_board(
            rows=self.rows,
            cols=self.cols,
            num_colors=self.num_colors,
            seed=self.seed,
        )

        self.engine = Match3Engine(
            num_colors=self.num_colors,
            seed=self.seed,
        )

        self.selected = None
        self.show_hints = False
        self.message = "New board generated."

        self.reset_animation()

    def cell_at_pixel(
        self,
        x: int,
        y: int,
    ) -> tuple[int, int] | None:
        board_x = x - PADDING
        board_y = y - HEADER_HEIGHT

        if board_x < 0 or board_y < 0:
            return None

        col = board_x // CELL_SIZE
        row = board_y // CELL_SIZE

        if 0 <= row < self.rows and 0 <= col < self.cols:
            return (row, col)

        return None

    def handle_click(self, position: tuple[int, int]) -> None:
        if self.is_animating:
            return

        cell = self.cell_at_pixel(*position)

        if cell is None:
            return

        row, col = cell

        if not self.state.cell_mask[row, col]:
            return

        if self.selected is None:
            self.selected = cell
            self.message = f"Selected {cell}"
            return

        if self.selected == cell:
            self.selected = None
            self.message = "Selection cleared."
            return

        first = self.selected
        second = cell

        self.selected = None

        result = self.engine.step(
            self.state,
            (first, second),
            record_events=True,
        )

        if not result.valid:
            self.message = "Invalid move."
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

            first_event = self.animation_events[0]

            self.state = first_event.state
            self.highlighted_cells = first_event.affected_cells
            self.current_animation_type = first_event.event_type
        else:
            self.state = result.state
            self.reset_animation()

    def get_animation_duration(
        self,
        event: AnimationEvent,
    ) -> int:
        """return synchronized animation durations."""

        base_duration = ANIMATION_DURATIONS[event.event_type]

        if event.event_type != AnimationEventType.SPECIAL_ACTIVATION:
            return base_duration

        activation_count = len(event.special_activations)

        if event.special_combination is not None:
            return (
                COMBINATION_DURATION
                + activation_count * CHAIN_ACTIVATION_DURATION
            )

        if activation_count > 0:
            return activation_count * CHAIN_ACTIVATION_DURATION

        return base_duration
    def update_animation(self) -> None:
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

    def draw_header(self) -> None:
        title = self.font.render(
            "MATCH-3 AI LAB",
            True,
            TEXT_COLOR,
        )

        self.screen.blit(title, (PADDING, 16))

        instruction = self.small_font.render(
            "R: Reset   H: Hints   ESC: Quit",
            True,
            TEXT_COLOR,
        )

        self.screen.blit(instruction, (PADDING, 48))

        display_message = self.message

        if self.is_animating and self.current_animation_type is not None:
            display_message = (
                f"Animation: {self.current_animation_type.name}"
                f" ({self.animation_index + 1}/{len(self.animation_events)})"
            )

        message = self.small_font.render(
            display_message,
            True,
            HINT_COLOR,
        )

        self.screen.blit(message, (PADDING, 78))

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

                center = (
                    PADDING + col * CELL_SIZE + CELL_SIZE // 2,
                    HEADER_HEIGHT + row * CELL_SIZE + CELL_SIZE // 2,
                )

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
                PADDING + destination_col * CELL_SIZE + CELL_SIZE // 2,
                int(HEADER_HEIGHT + interpolated_row * CELL_SIZE + CELL_SIZE // 2),
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
                        HEADER_HEIGHT + interpolated_row * CELL_SIZE + CELL_SIZE // 2
                    )
                else:
                    center_y = HEADER_HEIGHT + row * CELL_SIZE + CELL_SIZE // 2

                center_x = PADDING + col * CELL_SIZE + CELL_SIZE // 2

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
            max_distance = max(
                abs(col - source_col)
                for _, col in affected_cells
            )

            return {
                (row, col)
                for row, col in affected_cells
                if abs(col - source_col) <= max_distance * progress
            }

        # Vertical Rocket: clear outward from the source row.
        if special_type == SpecialType.ROCKET_VERTICAL:
            max_distance = max(
                abs(row - source_row)
                for row, _ in affected_cells
            )

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
                ) <= max_distance * progress
            }

        # Other specials: progressively reveal affected cells.
        ordered_cells = sorted(affected_cells)

        clear_count = int(len(ordered_cells) * progress)

        return set(ordered_cells[:clear_count])
    def get_visible_animation_state(self) -> BoardState:
        """Build a visual board while preserving pending chain activations."""

        if (
            not self.is_animating
            or self.current_animation_type
            != AnimationEventType.SPECIAL_ACTIVATION
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
            activation.position
            for activation in activations[activation_index + 1 :]
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
            or self.current_animation_type
            != AnimationEventType.SPECIAL_CREATION
            or self.previous_animation_state is None
        ):
            return set()

        transformed_cells: set[tuple[int, int]] = set()

        previous_state = self.previous_animation_state

        for row in range(self.rows):
            for col in range(self.cols):
                if not self.state.cell_mask[row, col]:
                    continue

                previous_special = SpecialType(
                    int(previous_state.specials[row, col])
                )

                current_special = SpecialType(
                    int(self.state.specials[row, col])
                )

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
            or self.current_animation_type
            != AnimationEventType.SPECIAL_CREATION
            or self.previous_animation_state is None
        ):
            return

        transformed_cells = self.get_transformed_cells()

        if not transformed_cells:
            return

        duration = ANIMATION_DURATIONS[
            AnimationEventType.SPECIAL_CREATION
        ]

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
            center = (
                PADDING + col * CELL_SIZE + CELL_SIZE // 2,
                HEADER_HEIGHT + row * CELL_SIZE + CELL_SIZE // 2,
            )

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
            or self.current_animation_type
            != AnimationEventType.SPECIAL_ACTIVATION
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
                    padding=PADDING,
                    header_height=HEADER_HEIGHT,
                )

                draw_combination_effect(
                    surface=self.screen,
                    combination=combination,
                    progress=combination_progress,
                    cell_size=CELL_SIZE,
                    padding=PADDING,
                    header_height=HEADER_HEIGHT,
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
                    padding=PADDING,
                    header_height=HEADER_HEIGHT,
                )

            return

        draw_activation_effect(
            surface=self.screen,
            affected_cells=active_special.effect_cells,
            progress=local_progress,
            cell_size=CELL_SIZE,
            padding=PADDING,
            header_height=HEADER_HEIGHT,
        )

        draw_special_effects(
            surface=self.screen,
            activations=(active_special,),
            progress=local_progress,
            cell_size=CELL_SIZE,
            padding=PADDING,
            header_height=HEADER_HEIGHT,
            rows=self.rows,
            cols=self.cols,
        )
    def draw_board(self) -> None:
        visual_state = self.get_visible_animation_state()

        board_rect = pygame.Rect(
            PADDING - 6,
            HEADER_HEIGHT - 6,
            self.cols * CELL_SIZE + 12,
            self.rows * CELL_SIZE + 12,
        )

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

                x = PADDING + col * CELL_SIZE
                y = HEADER_HEIGHT + row * CELL_SIZE

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
            for row in range(self.rows):
                for col in range(self.cols):
                    if not visual_state.cell_mask[row, col]:
                        continue

                    tile = int(visual_state.tiles[row, col])
                    special = SpecialType(int(visual_state.specials[row, col]))

                    center = (
                        PADDING + col * CELL_SIZE + CELL_SIZE // 2,
                        HEADER_HEIGHT + row * CELL_SIZE + CELL_SIZE // 2,
                    )

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

                            elapsed = pygame.time.get_ticks() - self.animation_start_time

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
        # Draw highlights and selection borders.
        for row in range(self.rows):
            for col in range(self.cols):
                if not self.state.cell_mask[row, col]:
                    continue

                x = PADDING + col * CELL_SIZE
                y = HEADER_HEIGHT + row * CELL_SIZE

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
        if not self.show_hints or self.is_animating:
            return

        moves = find_valid_moves(self.state)

        for first, second in moves:
            for row, col in (first, second):
                x = PADDING + col * CELL_SIZE + CELL_SIZE // 2
                y = HEADER_HEIGHT + row * CELL_SIZE + CELL_SIZE // 2

                pygame.draw.circle(
                    self.screen,
                    HINT_COLOR,
                    (x, y),
                    7,
                )

    def draw(self) -> None:
        self.screen.fill(BACKGROUND)

        self.draw_header()
        self.draw_board()
        self.draw_hints()

        self.draw_special_creation_animation()
        self.draw_special_activation()

        pygame.display.flip()
    def run(self) -> None:
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_r:
                        self.reset()

                    elif event.key == pygame.K_h:
                        self.show_hints = not self.show_hints

                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)

            self.update_animation()
            self.draw()
            self.clock.tick(FPS)

        pygame.quit()


def main() -> None:
    viewer = Match3Viewer(
        rows=8,
        cols=8,
        num_colors=5,
        seed=42,
    )

    viewer.run()


if __name__ == "__main__":
    main()
