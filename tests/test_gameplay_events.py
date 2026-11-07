"""Small fixture transitions test telemetry without running diagnostic episodes."""

import json

import numpy as np
import pytest

from match3_ai.evaluation.cross_level_benchmark import validate_seeds
from match3_ai.evaluation.gameplay_events import (
    GameplayCollector,
    InstrumentedLevelEnv,
    structured,
)
from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.engine import Match3Engine
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.games.match3.specials import SpecialType


def test_exact_stone_damage_targets_and_recording_rng_parity():
    tiles = np.array(
        [
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
            [3, 1, 2, 3, 0],
            [1, 2, 3, 1, 2],
            [2, 3, 1, 2, 3],
        ],
        dtype=np.uint8,
    )
    specials = np.zeros_like(tiles)
    specials[2, 1] = SpecialType.ROCKET_HORIZONTAL
    blockers = np.zeros_like(tiles)
    blockers[2, 4] = BlockerType.STONE
    hp = np.zeros_like(tiles)
    hp[2, 4] = 2
    state = BoardState(
        tiles,
        cell_mask=np.ones_like(tiles, dtype=bool),
        specials=specials,
        blockers=blockers,
        blocker_hp=hp,
    )
    plain, traced = Match3Engine(3, seed=42), Match3Engine(3, seed=42)
    a = plain.step(state, ((2, 1), (2, 2)))
    b = traced.step(state, ((2, 1), (2, 2)), record_events=True)
    assert structured(a.state) == structured(b.state)
    assert structured(plain.rng.bit_generator.state) == structured(
        traced.rng.bit_generator.state
    )
    assert a.cascade_count == b.cascade_count
    assert structured(a.blocker_damage) == structured(b.blocker_damage)
    damage = [d for event in b.animation_events for d in event.blocker_damage]
    stone_hits = [d for d in damage if d.position == (2, 4)]
    assert stone_hits
    assert structured(stone_hits[0])["sources"] == ["ROCKET"]
    assert stone_hits[0].previous_hp == 2
    assert stone_hits[0].remaining_hp == 1
    activations = [s for e in b.animation_events for s in e.special_activations]
    assert any((2, 4) in s.effect_cells for s in activations)
    assert (
        sum(e.event_type.name == "MATCH" for e in b.animation_events) == b.cascade_count
    )
    assert not a.animation_events


def test_instrumented_transitions_replay_identically(tmp_path):
    path = tmp_path / "events.jsonl"
    plain = Match3LevelEnv("stone-breaker")
    with path.open("x") as handle:
        collector = GameplayCollector(handle, "test", "fixture")
        traced = InstrumentedLevelEnv("stone-breaker", collector)
        left, left_info = plain.reset(seed=123)
        right, right_info = traced.reset(seed=123)
        assert structured(left) == structured(right)
        assert left_info == right_info
        actions = []
        for _ in range(3):
            action = int(np.flatnonzero(left["action_mask"])[0])
            actions.append(action)
            a, b = plain.step(action), traced.step(action)
            assert structured(a[0]) == structured(b[0])
            assert a[1:4] == b[1:4]
            assert a[4] == {k: v for k, v in b[4].items() if k != "resolution_events"}
            assert structured(
                plain.session.engine.rng.bit_generator.state
            ) == structured(traced.session.engine.rng.bit_generator.state)
            left = a[0]
        replay, _ = plain.reset(seed=123)
        for action in actions:
            replay, *_ = plain.step(action)
        assert structured(replay) == structured(right if not actions else b[0])
        events = [json.loads(line) for line in path.read_text().splitlines()]
        assert events[0]["kind"] == "episode_start"
        assert events[0]["observation"]["blocker_hp"]
        assert [e["sequence"] for e in events] == list(range(len(events)))
        assert [e["move_index"] for e in events if e["kind"] == "move_start"] == [
            1,
            2,
            3,
        ]
        assert [e["moves_remaining"] for e in events if e["kind"] == "move_end"] == [
            14,
            13,
            12,
        ]
    plain.close()
    traced.close()


def test_diagnostic_freshness_checks_event_records(tmp_path):
    seeds = tuple(range(730000, 730050))
    random_agent_seeds = tuple(range(740000, 740050))

    validate_seeds(
        seeds,
        tmp_path,
        random_agent_seeds=random_agent_seeds,
        expected_count=50,
    )

    (tmp_path / "events.jsonl").write_text(
        json.dumps(
            {"kind": "episode_start", "info": {"effective_episode_seed": 730000}}
        )
        + "\n"
    )

    with pytest.raises(ValueError, match="already used"):
        validate_seeds(
            seeds,
            tmp_path,
            random_agent_seeds=random_agent_seeds,
            expected_count=50,
        )


def test_move_and_episode_summaries_are_streamed(tmp_path):
    from match3_ai.agents.greedy_agent import GreedyAgent
    from match3_ai.evaluation.runner import run_episode

    path = tmp_path / "events.jsonl"
    with path.open("x") as handle:
        collector = GameplayCollector(handle, "test", "one-transition")
        env = InstrumentedLevelEnv("stone-breaker", collector)
        try:
            record = run_episode(
                env,
                GreedyAgent(),
                seed=123,
                max_steps=1,
                experiment_id="test",
                episode_id="one-transition",
                collector=collector,
            )
        finally:
            env.close()
        collector.validate()
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        assert rows[-1]["kind"] == "episode_end"
        assert rows[-1]["record"]["outcome"] == record.outcome
        summaries = [row["record"] for row in rows if row["kind"] == "move_summary"]
        assert len(summaries) == 1
        matches = [
            row
            for row in rows
            if row["kind"] == "resolution" and row["event"]["event_type"] == "MATCH"
        ]
        assert len(matches) == summaries[0]["cascade_count"]
        assert [row["cascade_round"] for row in matches] == list(
            range(1, len(matches) + 1)
        )


def test_creation_event_preserves_special_type_and_position():
    tiles = np.array(
        [
            [1, 4, 3, 2, 4],
            [3, 2, 3, 4, 2],
            [1, 4, 1, 1, 2],
            [1, 1, 3, 3, 1],
            [2, 4, 1, 3, 3],
        ],
        dtype=np.uint8,
    )
    state = BoardState(tiles, cell_mask=np.ones_like(tiles, dtype=bool))
    result = Match3Engine(4, seed=42).step(state, ((2, 1), (3, 1)), record_events=True)
    creations = [
        structured(e)
        for e in result.animation_events
        if e.event_type.name == "SPECIAL_CREATION"
    ]
    assert creations
    assert [2, 0] in creations[0]["affected_cells"]
    assert (
        creations[0]["state"]["specials"][2][0] == SpecialType.ROCKET_HORIZONTAL.value
    )
