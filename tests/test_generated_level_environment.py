import pytest

from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.generation import GenerationSpec, LevelGenerator


def test_generated_level_can_initialize_environment():
    config = LevelGenerator(GenerationSpec(seed=42)).generate(candidate_id=0)

    env = Match3LevelEnv(
        config.level_id,
        level_config=config,
    )

    try:
        observation, info = env.reset(seed=100)

        assert info["level_id"] == config.level_id
        assert observation["moves_remaining"] == config.move_limit
        assert observation["action_mask"].any()
    finally:
        env.close()


def test_unknown_level_without_config_is_rejected():
    with pytest.raises(ValueError, match="unknown level_id"):
        Match3LevelEnv("nonexistent-level")


def test_generated_level_id_mismatch_is_rejected():
    config = LevelGenerator(GenerationSpec(seed=42)).generate(candidate_id=0)

    with pytest.raises(ValueError, match="retain"):
        Match3LevelEnv(
            "different-id",
            level_config=config,
        )


def test_generated_level_does_not_change_packaged_level_order():
    baseline = Match3LevelEnv("first-match")

    config = LevelGenerator(GenerationSpec(seed=42)).generate(candidate_id=0)

    generated = Match3LevelEnv(
        config.level_id,
        level_config=config,
    )

    try:
        assert generated.level_ids[:-1] == baseline.level_ids
        assert generated.level_ids[-1] == config.level_id
    finally:
        baseline.close()
        generated.close()
