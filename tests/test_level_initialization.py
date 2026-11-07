from pathlib import Path

import numpy as np
import pytest

from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.constants import COLORLESS
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.pattern_detection import detect_patterns
from match3_ai.games.match3.specials import SpecialType
from match3_ai.levels import (
    BlockerPlacement,
    InitialSpecial,
    LevelConfig,
    LevelGenerationError,
    LevelObjective,
    ObjectiveType,
    create_level_board,
    load_level,
)

LEVEL_DATA = Path(__file__).parents[1] / "src/match3_ai/levels/data"


def make_config(
    *,
    seed: int = 31,
    rows: int = 6,
    cols: int = 6,
    num_colors: int = 5,
    cell_mask=None,
    blockers=(),
    initial_specials=(),
):
    objectives = (
        (LevelObjective(ObjectiveType.CLEAR_ALL_BLOCKERS),)
        if blockers
        else (LevelObjective(ObjectiveType.COLLECT_TILES, target=1),)
    )
    return LevelConfig(
        rows=rows,
        cols=cols,
        num_colors=num_colors,
        move_limit=20,
        seed=seed,
        cell_mask=cell_mask,
        blockers=tuple(blockers),
        initial_specials=tuple(initial_specials),
        objectives=objectives,
    )


def assert_valid_initial_board(state: BoardState, config: LevelConfig) -> None:
    assert state.shape == (config.rows, config.cols)
    assert np.all(state.tiles[~state.cell_mask] == 0)
    assert np.all(state.blockers[~state.cell_mask] == BlockerType.NONE)
    assert np.all(state.blocker_hp[~state.cell_mask] == 0)
    tile_capable = state.cell_mask & ~np.isin(
        state.blockers,
        [BlockerType.CRATE, BlockerType.STONE],
    )
    assert np.all(
        np.isin(
            state.tiles[tile_capable],
            list(range(1, config.num_colors + 1)) + [COLORLESS],
        )
    )
    assert np.all(state.tiles[state.blockers == BlockerType.CRATE] == 0)
    assert np.all(state.tiles[state.blockers == BlockerType.STONE] == 0)
    assert np.all(state.blocker_hp[state.blockers != BlockerType.NONE] > 0)
    assert detect_patterns(state.tiles) == []
    assert find_valid_moves(state)
    for placement in config.blockers:
        assert state.blockers[placement.position] == placement.blocker_type
        assert state.blocker_hp[placement.position] == placement.hp
    for placement in config.initial_specials:
        assert state.specials[placement.position] == placement.special_type


def test_same_seed_produces_identical_boards_and_different_seed_changes_board():
    config = make_config(seed=801, rows=8, cols=8)
    first = create_level_board(config)
    second = create_level_board(config)
    different_seed = create_level_board(make_config(seed=802, rows=8, cols=8))

    for name in ("tiles", "specials", "cell_mask", "blockers", "blocker_hp"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert not np.array_equal(first.tiles, different_seed.tiles)


def test_generation_does_not_modify_numpy_global_random_state():
    np.random.seed(1234)
    state_before = np.random.get_state()

    create_level_board(make_config())

    state_after = np.random.get_state()
    assert state_before[0] == state_after[0]
    assert np.array_equal(state_before[1], state_after[1])
    assert state_before[2:] == state_after[2:]


def test_masked_cells_are_empty_and_playable_cells_have_valid_colors():
    mask = tuple(
        tuple(not (row == 0 and col == 0) for col in range(6)) for row in range(6)
    )
    config = make_config(cell_mask=mask)

    state = create_level_board(config)

    assert_valid_initial_board(state, config)
    assert not state.cell_mask[0, 0]
    assert state.tiles[0, 0] == 0


def test_all_blocker_types_and_hp_variants_are_placed_exactly():
    placements = (
        BlockerPlacement((0, 0), BlockerType.CRATE, "SINGLE"),
        BlockerPlacement((0, 5), BlockerType.CRATE, "DOUBLE"),
        BlockerPlacement((5, 0), BlockerType.CRATE, "TRIPLE"),
        BlockerPlacement((1, 1), BlockerType.ICE, "THIN"),
        BlockerPlacement((1, 4), BlockerType.ICE, "THICK"),
        BlockerPlacement((4, 1), BlockerType.STONE, "SINGLE"),
        BlockerPlacement((4, 4), BlockerType.STONE, "DOUBLE"),
    )
    config = make_config(rows=6, cols=6, blockers=placements)

    state = create_level_board(config)

    assert_valid_initial_board(state, config)
    assert [int(state.blocker_hp[item.position]) for item in placements] == [
        1,
        2,
        3,
        1,
        2,
        1,
        2,
    ]
    assert np.all(state.tiles[state.blockers == BlockerType.ICE] > 0)
    assert np.all(state.tiles[state.blockers == BlockerType.CRATE] == 0)
    assert np.all(state.tiles[state.blockers == BlockerType.STONE] == 0)


def test_all_special_types_are_preserved_including_light_ball_and_ice_overlay():
    placements = (BlockerPlacement((2, 2), BlockerType.ICE, "THICK"),)
    special_placements = (
        InitialSpecial((1, 1), SpecialType.ROCKET_HORIZONTAL),
        InitialSpecial((1, 2), SpecialType.ROCKET_VERTICAL),
        InitialSpecial((2, 2), SpecialType.BOMB),
        InitialSpecial((3, 3), SpecialType.PROPELLER),
        InitialSpecial((4, 4), SpecialType.LIGHT_BALL),
    )
    config = make_config(
        blockers=placements,
        initial_specials=special_placements,
    )

    state = create_level_board(config)

    assert_valid_initial_board(state, config)
    assert state.specials[2, 2] == SpecialType.BOMB
    assert state.blockers[2, 2] == BlockerType.ICE
    assert state.tiles[4, 4] == COLORLESS
    assert state.specials[4, 4] == SpecialType.LIGHT_BALL
    for placement in special_placements[:-1]:
        assert 1 <= state.tiles[placement.position] <= config.num_colors


def test_repeated_calls_return_fresh_independent_board_states():
    config = make_config()
    first = create_level_board(config)
    second = create_level_board(config)

    assert first is not second
    assert not np.shares_memory(first.tiles, second.tiles)
    assert not np.shares_memory(first.specials, second.specials)
    assert not np.shares_memory(first.blockers, second.blockers)
    assert not np.shares_memory(first.blocker_hp, second.blocker_hp)
    first.tiles[0, 0] = 0
    assert second.tiles[0, 0] != 0


def test_constrained_mask_fails_explicitly_after_bounded_attempts():
    mask = tuple(
        tuple((row, col) in {(0, 0), (3, 3)} for col in range(4)) for row in range(4)
    )
    config = make_config(rows=4, cols=4, cell_mask=mask)

    with pytest.raises(LevelGenerationError, match="after 4 attempts"):
        create_level_board(config, max_attempts=4)


def test_generation_rejects_invalid_input_and_attempt_limits():
    with pytest.raises(TypeError, match="LevelConfig"):
        create_level_board({})
    with pytest.raises(ValueError, match="positive integer"):
        create_level_board(make_config(), max_attempts=0)


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
def test_all_six_levels_generate_valid_deterministic_initial_boards(filename):
    config = load_level(LEVEL_DATA / filename)

    first = create_level_board(config)
    second = create_level_board(config)

    assert_valid_initial_board(first, config)
    for name in ("tiles", "specials", "cell_mask", "blockers", "blocker_hp"):
        assert np.array_equal(getattr(first, name), getattr(second, name))
    assert np.count_nonzero(first.blockers == BlockerType.CRATE) == sum(
        item.blocker_type == BlockerType.CRATE for item in config.blockers
    )
    assert np.count_nonzero(first.blockers == BlockerType.ICE) == sum(
        item.blocker_type == BlockerType.ICE for item in config.blockers
    )
    assert np.count_nonzero(first.blockers == BlockerType.STONE) == sum(
        item.blocker_type == BlockerType.STONE for item in config.blockers
    )
