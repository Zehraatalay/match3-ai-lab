"""Version integrity and historical RNG namespace regression checks."""

import hashlib
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels import load_level

DATA = Path(__file__).parents[1] / "src/match3_ai/levels/data"
ORIGINAL_HASHES = {
    "blocker-mayhem": "3df8484f8019e53995f0b910e55c5454bbe9ab7c1ef0562e628937c9b5267756",
    "double-trouble": "af80e1e58875f1d4a84dfc11b354f027e166f6e66b3b646630bd5d9283d68b0c",
    "first-match": "f2c70b26a288bca389bebc33ad3a4c34792e2306637533b8f0fc5da45eb3b5a9",
    "frozen-garden": "aef2e9717571069f096d4d106bd5378ace4ffc85e937fbadf6f2a5891be6ff49",
    "stone-breaker": "db216a4f6ef11fa9cc15f6766aa6d5805f164d7d32f6aa20b976a85f245cca2e",
    "wooden-trouble": "34ca75f633592dc24cc73bb15cf24184e900e82848b98655c17c2f5e1757aafb",
}
CONFIRMED_HASHES = {
    "wooden-trouble-v2": "2cf5a4a71fbf2b0f173fd5cd97bad2e9650496a937356828e8838a483111db48",
    "double-trouble-v2": "99f8686fdd3507221e6096c3eabff1b8838afa9ce4d8b386a00167684bfa5a50",
    "frozen-garden-v2": "322aba2c05b16dddd9e8bfe6710187b7632929921733cb915b47477ca346f8ed",
}


def test_original_files_and_registry_positions_preserved():
    env = Match3LevelEnv()
    try:
        assert env.level_ids[:6] == tuple(ORIGINAL_HASHES)
        assert set(env.level_ids[6:]) == set(CONFIRMED_HASHES)
        for name, digest in ORIGINAL_HASHES.items():
            assert (
                hashlib.sha256((DATA / f"{name}.json").read_bytes()).hexdigest()
                == digest
            )
    finally:
        env.close()


@pytest.mark.parametrize("name", ORIGINAL_HASHES)
def test_original_seed_namespaces_preserved(name):
    index = tuple(ORIGINAL_HASHES).index(name)
    children = np.random.SeedSequence([123, index]).spawn(2)
    expected = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    env = Match3LevelEnv(name)
    try:
        observation, info = env.reset(seed=123)
        assert [info["board_seed"], info["engine_seed"]] == expected
        assert env.observation_space.contains(observation)
    finally:
        env.close()


@pytest.mark.parametrize("name", CONFIRMED_HASHES)
def test_v2_matches_confirmed_config_with_only_id_changed(name):
    path = DATA / f"{name}.json"
    config = load_level(path)
    assert config.level_id == name
    raw = json.loads(path.read_text())
    raw["level_id"] = name.removesuffix("-v2")
    assert (
        hashlib.sha256(json.dumps(raw, indent=2).encode()).hexdigest()
        == CONFIRMED_HASHES[name]
    )
    parent = raw["level_id"]
    baseline = Match3LevelEnv(parent, level_config=replace(config, level_id=parent))
    versioned = Match3LevelEnv(name)
    try:
        left, left_info = baseline.reset(seed=123)
        right, right_info = versioned.reset(seed=123)
        for key in left:
            np.testing.assert_array_equal(left[key], right[key])
        assert left_info["board_seed"] == right_info["board_seed"]
        assert left_info["engine_seed"] == right_info["engine_seed"]
        assert left_info["level_config_id"] != right_info["level_config_id"]
        assert versioned.observation_space.contains(right)
        assert versioned.action_space == baseline.action_space
    finally:
        baseline.close()
        versioned.close()
