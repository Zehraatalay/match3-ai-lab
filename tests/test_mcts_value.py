import pytest

from match3_ai.agents.mcts_value import MCTSValueFunction
from match3_ai.games.match3.environment import Match3LevelEnv


@pytest.mark.parametrize(
    "win_weight,loss_weight",
    [
        (-0.1, 0.2),
        (1.0, 0.2),
        (0.2, -0.1),
        (0.2, 1.0),
    ],
)
def test_invalid_weights(win_weight, loss_weight):
    with pytest.raises(ValueError):
        MCTSValueFunction(win_weight, loss_weight)


def test_nonterminal_value_is_zero():
    env = Match3LevelEnv(level_id="double-trouble")

    try:
        env.reset(seed=42)

        value_function = MCTSValueFunction()

        assert value_function.evaluate(env, 0.0) == 0.0

    finally:
        env.close()


def test_objective_completion_is_bounded():
    env = Match3LevelEnv(level_id="double-trouble")

    try:
        env.reset(seed=42)

        completion = MCTSValueFunction.objective_completion(env)

        assert 0.0 <= completion <= 1.0

    finally:
        env.close()
