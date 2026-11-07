"""Initial-phase feature checks without gameplay, gravity, refill, or cascades."""

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.specials import SpecialType

SCRIPT = (
    Path(__file__).resolve().parents[1] / "analysis/14_spatial_feature_engineering.py"
)
SPEC = importlib.util.spec_from_file_location("spatial_feature_engineering", SCRIPT)
spatial = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = spatial
SPEC.loader.exec_module(spatial)


@pytest.fixture(autouse=True)
def forbid_gameplay(monkeypatch):
    from match3_ai.games.match3 import board_mechanics, engine
    from match3_ai.levels.session import LevelSession

    def forbidden(*args, **kwargs):
        pytest.fail("Gameplay, clearing, gravity, or refill was executed")

    monkeypatch.setattr(engine.Match3Engine, "step", forbidden)
    monkeypatch.setattr(LevelSession, "step", forbidden)
    for module in (board_mechanics, engine):
        for name in (
            "clear_board_matches",
            "apply_board_gravity",
            "refill_board_state",
        ):
            monkeypatch.setattr(module, name, forbidden)


def make_board(blocker=BlockerType.NONE, position=(1, 0), hp=2):
    tiles = np.array(
        [
            [1, 2, 1, 3],
            [2, 1, 3, 2],
            [3, 2, 3, 1],
            [2, 3, 1, 2],
        ],
        dtype=np.uint8,
    )
    blockers = np.zeros_like(tiles)
    health = np.zeros_like(tiles)
    if blocker != BlockerType.NONE:
        blockers[position] = blocker
        health[position] = hp
        if blocker in (BlockerType.CRATE, BlockerType.STONE):
            tiles[position] = 0
    return BoardState(
        tiles, np.ones_like(tiles, dtype=bool), blockers=blockers, blocker_hp=health
    )


MOVE = ((0, 1), (1, 1))


@pytest.mark.parametrize(
    "blocker,position,expected",
    [
        (BlockerType.CRATE, (1, 0), 1),
        (BlockerType.CRATE, (1, 3), 0),
        (BlockerType.ICE, (0, 0), 1),
        (BlockerType.ICE, (1, 0), 0),
        (BlockerType.STONE, (1, 0), 0),
    ],
)
def test_ordinary_damage_sources(blocker, position, expected):
    board = make_board(blocker, position)
    assert MOVE in find_valid_moves(board)
    result = spatial.evaluate_move(board, MOVE, np.random.default_rng(42))
    assert spatial.hp_loss(result.ordinary_records) == expected
    assert spatial.hp_loss(result.immediate_records) == expected


def test_multiple_events_and_independent_ordinary_damage():
    board = make_board(BlockerType.ICE, (0, 0), hp=2)
    board.specials[0, 2] = SpecialType.ROCKET_HORIZONTAL
    result = spatial.evaluate_move(board, MOVE, np.random.default_rng(42))
    assert spatial.hp_loss(result.ordinary_records) == 1
    assert spatial.hp_loss(result.immediate_records) == 2
    assert [(r.previous_hp, r.remaining_hp) for r in result.immediate_records] == [
        (2, 1),
        (1, 0),
    ]
    assert sum(r.destroyed for r in result.immediate_records) == 1


@pytest.mark.parametrize(
    "blocker", [BlockerType.CRATE, BlockerType.ICE, BlockerType.STONE]
)
def test_special_swap_uses_only_special_events(blocker, monkeypatch):
    board = make_board(blocker, (0, 3), hp=1)
    board.specials[0, 0] = SpecialType.ROCKET_HORIZONTAL
    move = ((0, 0), (0, 1))

    def forbidden(*args):
        pytest.fail("Ordinary events must not run for a special swap")

    monkeypatch.setattr(spatial, "detect_patterns", forbidden)
    result = spatial.evaluate_move(board, move, np.random.default_rng(42))
    assert result.ordinary_records == ()
    assert spatial.hp_loss(result.immediate_records) == 1
    assert sum(r.destroyed for r in result.immediate_records) == 1


def test_board_and_live_rng_unchanged_and_results_repeatable():
    board = make_board(BlockerType.ICE, (2, 2), hp=2)
    board.specials[0, 0] = SpecialType.PROPELLER
    before = board.copy()
    rng = np.random.default_rng(42)
    rng_before = json.dumps(rng.bit_generator.state, sort_keys=True)
    move = ((0, 0), (0, 1))
    first = spatial.evaluate_move(board, move, rng)
    assert first == spatial.evaluate_move(board, move, rng)
    assert spatial.extract_spatial_features(
        board, rng
    ) == spatial.extract_spatial_features(board, rng)
    assert json.dumps(rng.bit_generator.state, sort_keys=True) == rng_before
    for name in ("tiles", "specials", "cell_mask", "blockers", "blocker_hp"):
        np.testing.assert_array_equal(getattr(board, name), getattr(before, name))


def test_zero_legal_moves():
    board = BoardState(np.array([[1]], dtype=np.uint8), np.ones((1, 1), dtype=bool))
    features, count = spatial.extract_spatial_features(board, np.random.default_rng(42))
    assert count == 0
    assert set(features) == set(spatial.FEATURES)
    assert all(value == 0 for value in features.values())


def test_aggregation_includes_zero_damage_moves(monkeypatch):
    board = make_board(BlockerType.ICE, (0, 0), hp=2)
    board.specials[0, 2] = SpecialType.ROCKET_HORIZONTAL
    damage = spatial.evaluate_move(board, MOVE, np.random.default_rng(42))
    monkeypatch.setattr(spatial, "find_valid_moves", lambda _: [MOVE, MOVE])
    values = iter([damage, spatial.MoveDamage((), ())])
    monkeypatch.setattr(spatial, "evaluate_move", lambda *args: next(values))
    features, count = spatial.extract_spatial_features(board, np.random.default_rng(42))
    assert count == 2
    assert features == {
        "ordinary_damage_move_count": 1,
        "ordinary_damage_move_ratio": 0.5,
        "mean_ordinary_hp_loss": 0.5,
        "max_ordinary_hp_loss": 1,
        "immediate_damage_move_count": 1,
        "immediate_damage_move_ratio": 0.5,
        "mean_immediate_hp_loss": 1.0,
        "max_immediate_hp_loss": 2,
        "max_immediate_blockers_destroyed": 1,
    }


@pytest.mark.parametrize("field", ["fingerprint", "board_seed", "engine_seed"])
def test_reconstruction_mismatches_fail(field, monkeypatch):
    monkeypatch.setattr(spatial, "observation_fingerprint", lambda _: "verified")
    recorded = {"fingerprint": "verified", "board_seed": 1, "engine_seed": 2}
    recorded[field] = "wrong"
    with pytest.raises(RuntimeError, match="reconstruction mismatch"):
        spatial.verify_initial_identity(
            {}, {"board_seed": 1, "engine_seed": 2}, recorded, ("variant", 1)
        )


def test_configuration_hash_mismatch_fails_before_reset(tmp_path, monkeypatch):
    path = tmp_path / "config.json"
    path.write_text("{}")
    monkeypatch.setattr(
        spatial,
        "load_calibrated_boards",
        lambda _: (
            {"variant": {"path": "config.json", "sha256": "wrong"}},
            {},
        ),
    )
    with pytest.raises(ValueError, match="Configuration SHA256 mismatch"):
        spatial.reconstruct_features(tmp_path)


@pytest.mark.parametrize("problem", [None, "paired_identity", "duplicate", "missing"])
def test_calibrated_loading_filters_originals_and_validates_pairs(tmp_path, problem):
    names = ("wooden__calibrated", "double__calibrated", "frozen__calibrated")
    manifest = {
        "configurations": {name: {} for name in (*names, "wooden__original")},
        "environment_seeds": list(range(761000, 761200)),
    }
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    rows = []
    for name in names:
        for seed in manifest["environment_seeds"]:
            for agent in ("Random", "Greedy"):
                rows.append(
                    {
                        "variant_id": name,
                        "seed": seed,
                        "record": {"agent_name": agent},
                        "initial_identity": {
                            "fingerprint": str(seed),
                            "board_seed": seed,
                            "engine_seed": seed + 1,
                        },
                    }
                )
    if problem == "paired_identity":
        rows[1]["initial_identity"]["board_seed"] = -1
    elif problem == "duplicate":
        rows.append(rows[0])
    elif problem == "missing":
        rows.pop()
    rows.append({"variant_id": "wooden__original"})
    (tmp_path / "episodes.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    if problem:
        with pytest.raises(ValueError):
            spatial.load_calibrated_boards(tmp_path)
    else:
        specifications, boards = spatial.load_calibrated_boards(tmp_path)
        assert set(specifications) == set(names)
        assert len(boards) == 600
        assert all(not key[0].endswith("__original") for key in boards)
