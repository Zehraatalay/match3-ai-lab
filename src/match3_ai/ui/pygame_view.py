from __future__ import annotations

import pygame

from match3_ai.games.match3.engine import Match3Engine
from match3_ai.games.match3.generator import generate_playable_board
from match3_ai.games.match3.moves import find_valid_moves

# display configuration
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
        # board mevcut state
        self.state = generate_playable_board(
            rows=rows,
            cols=cols,
            num_colors=num_colors,
            seed=seed,
        )
        # movements isleyen game engine
        self.engine = Match3Engine(
            num_colors=num_colors,
            seed=seed,
        )
        # burada seed ayni olsa da generator ve engine farkli RNG objects kullaniyor

        self.selected: tuple[int, int] | None = None
        self.show_hints = False
        self.message = "Select two adjacent tiles."

        self.width = cols * CELL_SIZE + 2 * PADDING
        self.height = rows * CELL_SIZE + HEADER_HEIGHT + 2 * PADDING

        pygame.init()

        self.screen = pygame.display.set_mode((self.width, self.height))

        pygame.display.set_caption("Match-3 AI Lab")

        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("arial", 22)
        self.small_font = pygame.font.SysFont("arial", 16)

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
        )

        if result.valid:
            self.state = result.state

            self.message = (
                f"Cleared: {result.cleared_tiles} | Resolutions: {result.cascade_count}"
            )

        else:
            self.message = "Invalid move."

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

        message = self.small_font.render(
            self.message,
            True,
            HINT_COLOR,
        )

        self.screen.blit(message, (PADDING, 78))

    def draw_board(self) -> None:
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

                tile = int(self.state.tiles[row, col])

                if tile > 0:
                    color = TILE_COLORS[(tile - 1) % len(TILE_COLORS)]

                    tile_rect = cell_rect.inflate(-12, -12)

                    pygame.draw.rect(
                        self.screen,
                        color,
                        tile_rect,
                        border_radius=12,
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
        if not self.show_hints:
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
