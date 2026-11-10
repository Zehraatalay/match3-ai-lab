import pytest

from match3_ai.agents.observation_encoder import ObservationEncoder
from match3_ai.evaluation.behavior_data import (
    complete_state_fingerprint,
    observation_fingerprint,
)
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.generation import (
    GenerationSpec,
    LevelGenerator,
)


@pytest.mark.parametrize("move_limit", [15, 16, 20])
def test_fingerprint_supports_generated_move_limits(move_limit):
    spec = GenerationSpec(
        min_moves=move_limit,
        max_moves=move_limit,
        seed=42,
    )

    level = LevelGenerator(spec).generate(candidate_id=0)

    env = Match3LevelEnv(
        level.level_id,
        level_config=level,
    )

    try:
        observation, _ = env.reset(seed=1000)

        first = complete_state_fingerprint(env)
        second = complete_state_fingerprint(env)

        assert first == second
        assert len(first) == 64

        observation_hash = observation_fingerprint(observation)

        assert len(observation_hash) == 64

    finally:
        env.close()


@pytest.mark.parametrize("move_limit", [16, 20])
def test_encoder_v1_still_rejects_large_move_limits(move_limit):
    spec = GenerationSpec(
        min_moves=move_limit,
        max_moves=move_limit,
        seed=42,
    )

    level = LevelGenerator(spec).generate(candidate_id=0)

    env = Match3LevelEnv(
        level.level_id,
        level_config=level,
    )

    try:
        observation, _ = env.reset(seed=1000)

        with pytest.raises(
            ValueError,
            match="moves_remaining is outside encoder bounds",
        ):
            ObservationEncoder().encode(observation)

    finally:
        env.close()
