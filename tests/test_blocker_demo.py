import numpy as np
import pygame
import pytest

from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.generator import generate_playable_board
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.pattern_detection import detect_patterns
from match3_ai.games.match3.specials import SpecialType
from match3_ai.ui import pygame_view
from match3_ai.ui.pygame_view import (
    BLOCKER_DEMO_SEED,
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


def test_demo_board_contains_requested_blockers_and_is_playable():
    viewer = Match3Viewer(demo_blockers=True)
    state = viewer.state

    assert (viewer.rows, viewer.cols, viewer.num_colors) == (8, 8, 5)
    assert state.shape == (8, 8)
    assert np.count_nonzero(state.blockers == BlockerType.CRATE) == 3
    assert np.count_nonzero(state.blockers == BlockerType.STONE) == 2
    assert np.count_nonzero(state.blockers == BlockerType.ICE) == 2

    assert state.blocker_hp[0, 0] == 1
    assert state.blocker_hp[0, 7] == 2
    assert state.blocker_hp[7, 0] == 3
    assert state.blocker_hp[3, 3] == 1
    assert state.blocker_hp[7, 7] == 2
    assert state.blocker_hp[1, 2] == 1
    assert state.blocker_hp[4, 4] == 2

    fixed_mask = np.isin(state.blockers, [BlockerType.CRATE, BlockerType.STONE])
    assert np.all(state.tiles[fixed_mask] == 0)
    assert np.all(state.specials[fixed_mask] == SpecialType.NONE)
    assert np.all(state.tiles[state.blockers == BlockerType.ICE] > 0)
    assert state.specials[4, 4] == SpecialType.BOMB
    assert np.all(state.blocker_hp[state.blockers == BlockerType.NONE] == 0)
    assert np.all(state.blocker_hp[state.blockers != BlockerType.NONE] > 0)
    assert np.all(state.cell_mask)
    assert detect_patterns(state.tiles) == []
    assert find_valid_moves(state)
    assert viewer.seed == BLOCKER_DEMO_SEED
    assert viewer.engine.rng.bit_generator.state == np.random.default_rng(
        BLOCKER_DEMO_SEED
    ).bit_generator.state


def test_demo_board_generation_is_reproducible_for_same_seed():
    first = generate_blocker_demo_board(seed=83)
    second = generate_blocker_demo_board(seed=83)

    assert np.array_equal(first.tiles, second.tiles)
    assert np.array_equal(first.specials, second.specials)
    assert np.array_equal(first.blockers, second.blockers)
    assert np.array_equal(first.blocker_hp, second.blocker_hp)


def test_demo_reset_restores_initial_board_and_rng_seed():
    viewer = Match3Viewer(demo_blockers=True)
    initial = generate_blocker_demo_board()
    viewer.state.tiles[2, 2] = (int(viewer.state.tiles[2, 2]) % 5) + 1
    viewer.state.blocker_hp[0, 0] = 8

    viewer.reset()

    assert np.array_equal(viewer.state.tiles, initial.tiles)
    assert np.array_equal(viewer.state.specials, initial.specials)
    assert np.array_equal(viewer.state.blockers, initial.blockers)
    assert np.array_equal(viewer.state.blocker_hp, initial.blocker_hp)
    assert viewer.seed == BLOCKER_DEMO_SEED
    assert viewer.engine.rng.bit_generator.state == np.random.default_rng(
        BLOCKER_DEMO_SEED
    ).bit_generator.state


def test_default_viewer_still_uses_existing_blocker_free_seeded_board():
    viewer = Match3Viewer()
    expected = generate_playable_board(8, 8, 5, seed=42)

    assert not viewer.demo_blockers
    assert viewer.seed == 42
    assert np.array_equal(viewer.state.tiles, expected.tiles)
    assert np.all(viewer.state.blockers == BlockerType.NONE)
    assert np.all(viewer.state.blocker_hp == 0)


@pytest.mark.parametrize(
    ("arguments", "expected_demo"),
    [([], False), (["--demo-blockers"], True)],
)
def test_cli_flag_selects_demo_mode_without_changing_default(
    monkeypatch,
    arguments,
    expected_demo,
):
    received = {}

    class ViewerStub:
        def __init__(self, **kwargs):
            received.update(kwargs)

        def run(self):
            return None

    monkeypatch.setattr(pygame_view, "Match3Viewer", ViewerStub)

    pygame_view.main(arguments)

    assert received["demo_blockers"] is expected_demo


def test_demo_mode_rejects_dimensions_that_do_not_match_its_fixed_layout():
    with pytest.raises(ValueError, match="8x8 board with 5 colors"):
        Match3Viewer(rows=7, demo_blockers=True)
