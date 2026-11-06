from pathlib import Path

import numpy as np
import pytest

from match3_ai.games.match3.blockers import BlockerType, DamageSource
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.board_mechanics import (
    apply_board_gravity,
    refill_board_state,
)
from match3_ai.games.match3.engine import Match3Engine
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.pattern_detection import detect_patterns
from match3_ai.games.match3.special_activation import rocket_cells
from match3_ai.levels import create_level_board, load_level

LEVEL_DATA = Path(__file__).parents[1] / "src/match3_ai/levels/data"
LEVELS = (
    "first-match.json",
    "wooden-trouble.json",
    "double-trouble.json",
    "frozen-garden.json",
    "stone-breaker.json",
    "blocker-mayhem.json",
)
EXPECTED_LEVELS = (
    (6, 6, 5, 5, 36, 0),
    (7, 7, 3, 10, 37, 4),
    (8, 7, 4, 15, 40, 5),
    (8, 8, 4, 15, 40, 8),
    (8, 8, 4, 15, 44, 2),
    (9, 8, 4, 15, 56, 6),
)
EXPECTED_MASKS = (
    None,
    ("##...##", "#.....#", ".......", ".......", ".......", "#.....#", "##...##"),
    ("##...##", "#.....#", ".......", ".......", ".......", "#.....#", "##...##", "##...##"),
    ("###..###", "##....##", "#......#", "........", "........", "#......#", "##....##", "###..###"),
    ("##....##", "##....##", "........", "...##...", "...##...", "........", "##....##", "##....##"),
    ("##....##", "#......#", "........", ".......#", "........", "#.......", "........", "##.....#", "###...##"),
)


@pytest.mark.parametrize(
    ("filename", "expected", "expected_mask"),
    tuple(zip(LEVELS, EXPECTED_LEVELS, EXPECTED_MASKS)),
)
def test_level_design_dimensions_masks_and_move_limits(
    filename,
    expected,
    expected_mask,
):
    config = load_level(LEVEL_DATA / filename)
    rows, cols, colors, moves, playable_cells, blocker_count = expected

    assert (config.rows, config.cols, config.num_colors, config.move_limit) == (
        rows,
        cols,
        colors,
        moves,
    )
    state = create_level_board(config)
    assert int(np.count_nonzero(state.cell_mask)) == playable_cells
    assert int(np.count_nonzero(state.blockers)) == blocker_count
    assert detect_patterns(state.tiles) == []
    assert find_valid_moves(state)
    assert all(
        state.cell_mask[position]
        for move in find_valid_moves(state)
        for position in move
    )

    if expected_mask is None:
        assert config.cell_mask is None
    else:
        actual_mask = tuple(
            "".join("." if cell else "#" for cell in row)
            for row in state.cell_mask
        )
        assert actual_mask == expected_mask


def test_level_blocker_variants_and_objectives_match_the_design():
    configs = [load_level(LEVEL_DATA / filename) for filename in LEVELS]
    assert [
        (item.blocker_type, item.variant)
        for item in configs[1].blockers
    ] == [(BlockerType.CRATE, "SINGLE")] * 4
    assert sorted(item.variant for item in configs[2].blockers) == [
        "DOUBLE",
        "DOUBLE",
        "DOUBLE",
        "SINGLE",
        "SINGLE",
    ]
    assert sorted(item.variant for item in configs[3].blockers) == [
        "THICK",
        "THICK",
        "THICK",
        "THICK",
        "THIN",
        "THIN",
        "THIN",
        "THIN",
    ]
    assert [item.blocker_type for item in configs[4].blockers] == [
        BlockerType.STONE
    ] * 2
    assert {item.variant for item in configs[4].blockers} == {"SINGLE", "DOUBLE"}
    assert {item.blocker_type for item in configs[5].blockers} == {
        BlockerType.CRATE,
        BlockerType.ICE,
        BlockerType.STONE,
    }
    assert all(
        sum(item.blocker_type == blocker_type for item in configs[5].blockers) == 2
        for blocker_type in (BlockerType.CRATE, BlockerType.ICE, BlockerType.STONE)
    )
    assert configs[0].objectives[0].target == 30
    assert configs[5].objectives[1].target == 30


def test_gravity_and_refill_preserve_mask_and_blocker_boundaries():
    mask = np.ones((7, 3), dtype=np.bool_)
    mask[4, 0] = False
    tiles = np.zeros((7, 3), dtype=np.uint8)
    tiles[[0, 3, 5], 0] = [1, 2, 3]
    blockers = np.zeros_like(tiles)
    blocker_hp = np.zeros_like(tiles)
    blockers[2, 0] = BlockerType.STONE
    blocker_hp[2, 0] = 2
    blockers[6, 0] = BlockerType.CRATE
    blocker_hp[6, 0] = 1
    state = BoardState(tiles, mask, blockers=blockers, blocker_hp=blocker_hp)

    fallen = apply_board_gravity(state)
    refilled = refill_board_state(fallen, 3, np.random.default_rng(91))

    assert fallen.tiles[:, 0].tolist() == [0, 1, 0, 2, 0, 3, 0]
    assert refilled.tiles[2, 0] == 0
    assert refilled.tiles[4, 0] == 0
    assert refilled.tiles[6, 0] == 0
    assert np.array_equal(refilled.cell_mask, mask)
    assert np.array_equal(refilled.blockers, blockers)
    assert np.array_equal(refilled.blocker_hp, blocker_hp)


def test_special_effect_stops_at_masked_cells_without_crossing_into_them():
    mask = np.ones((5, 5), dtype=np.bool_)
    mask[2, 2] = False
    tiles = np.ones((5, 5), dtype=np.uint8)
    tiles[~mask] = 0
    state = BoardState(tiles, mask)

    affected = rocket_cells(state, (2, 1), horizontal=True)

    assert affected == {(2, col) for col in (0, 1, 3, 4)}
    assert (2, 2) not in affected


def test_match_damage_near_masked_gap_is_resolved_without_crossing_gap():
    tiles = np.array(
        [
            [1, 2, 1, 0],
            [0, 1, 2, 3],
            [2, 3, 1, 2],
            [3, 2, 3, 1],
        ],
        dtype=np.uint8,
    )
    mask = np.ones((4, 4), dtype=np.bool_)
    mask[0, 3] = False
    blockers = np.zeros((4, 4), dtype=np.uint8)
    blocker_hp = np.zeros_like(blockers)
    blockers[1, 0] = BlockerType.CRATE
    blocker_hp[1, 0] = 2
    tiles[1, 0] = 0
    state = BoardState(tiles, mask, blockers=blockers, blocker_hp=blocker_hp)

    result = Match3Engine(num_colors=3, seed=83).step(
        state,
        ((0, 1), (1, 1)),
        record_events=True,
    )

    assert result.valid
    assert result.state.blocker_hp[1, 0] == 1
    assert any(
        record.position == (1, 0)
        and record.sources == frozenset({DamageSource.ADJACENT_MATCH})
        for record in result.blocker_damage
    )
