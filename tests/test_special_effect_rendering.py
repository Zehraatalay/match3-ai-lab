import pygame
import pytest

from match3_ai.games.match3.animation_events import SpecialCombinationInfo
from match3_ai.games.match3.specials import SpecialType
from match3_ai.ui.pygame_view import CELL_SIZE, HEADER_HEIGHT, PADDING
from match3_ai.ui.special_effects import cell_center, draw_combination_effect


@pytest.fixture(autouse=True)
def pygame_headless(monkeypatch):
    monkeypatch.setenv("SDL_VIDEODRIVER", "dummy")
    monkeypatch.setenv("SDL_AUDIODRIVER", "dummy")
    pygame.init()
    yield
    pygame.quit()


def test_propeller_combination_traces_are_drawn_at_recorded_effect_cells():
    effect_cells = frozenset({(0, 0), (4, 4)})
    combination = SpecialCombinationInfo(
        source=(2, 1),
        destination=(2, 2),
        source_type=SpecialType.PROPELLER,
        destination_type=SpecialType.ROCKET_HORIZONTAL,
        effect_cells=effect_cells,
    )
    surface = pygame.Surface((8 * CELL_SIZE, 8 * CELL_SIZE), pygame.SRCALPHA)

    draw_combination_effect(
        surface=surface,
        combination=combination,
        progress=0.8,
        cell_size=CELL_SIZE,
        padding=PADDING,
        header_height=HEADER_HEIGHT,
        rows=5,
        cols=5,
    )

    for position in effect_cells:
        center = cell_center(position, CELL_SIZE, PADDING, HEADER_HEIGHT)
        assert surface.get_at(center).a > 0

    unrelated_center = cell_center((0, 4), CELL_SIZE, PADDING, HEADER_HEIGHT)
    assert surface.get_at(unrelated_center).a == 0
