import pytest

from match3_ai.agents.objective_rollout import ObjectiveRolloutPolicy
from match3_ai.games.match3.environment import Match3LevelEnv


def test_objective_rollout_selects_legal_action():
    env = Match3LevelEnv(level_id="double-trouble")
    observation, _ = env.reset(seed=42)

    policy = ObjectiveRolloutPolicy(
        max_candidates=5,
        seed=42,
    )

    action = policy.select_action(env)

    assert observation["action_mask"][action] == 1

    env.close()


def test_objective_rollout_preserves_original_state():
    env = Match3LevelEnv(level_id="double-trouble")
    env.reset(seed=42)

    original_board = env.session.state.tiles.copy()
    original_moves = env.session.moves_remaining

    policy = ObjectiveRolloutPolicy(
        max_candidates=5,
        seed=42,
    )

    policy.select_action(env)

    assert (env.session.state.tiles == original_board).all()
    assert env.session.moves_remaining == original_moves

    env.close()


@pytest.mark.parametrize("max_candidates", [0, -1])
def test_objective_rollout_rejects_invalid_candidates(max_candidates):
    with pytest.raises(ValueError):
        ObjectiveRolloutPolicy(max_candidates=max_candidates)
