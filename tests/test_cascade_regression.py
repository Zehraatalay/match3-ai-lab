import numpy as np
import pytest

from match3_ai.games.match3.animation_events import AnimationEventType
from match3_ai.games.match3.engine import CascadeLimitExceeded, Match3Engine
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.pattern_detection import detect_patterns
from match3_ai.levels import LevelSession


@pytest.mark.parametrize("entry_point", ["environment", "session"])
def test_wooden_trouble_seed_52003_exceeds_explicit_lower_budget(entry_point):
    env = Match3LevelEnv("wooden-trouble", max_cascades=100)
    observation, info = env.reset(seed=52003)
    assert env.decode_action(68) == (4, 2, 0)
    assert observation["action_mask"][68] == 1
    assert info["board_seed"] == 355501557
    assert info["engine_seed"] == 1545427802
    session = (
        env.session
        if entry_point == "environment"
        else LevelSession(
            env.config,
            board_seed=info["board_seed"],
            engine_seed=info["engine_seed"],
            max_cascades=100,
        )
    )
    original = session.board.copy()
    assert session.engine.max_cascades == 100
    with pytest.raises(CascadeLimitExceeded, match="limit=100") as caught:
        if entry_point == "environment":
            env.step(68)
        else:
            session.step(((4, 2), (4, 3)))
    assert caught.value.cascades_processed == 100
    assert caught.value.context["action"] == ((4, 2), (4, 3))
    assert caught.value.context["remaining_patterns"] == 4
    if entry_point == "environment":
        assert caught.value.context["level_id"] == "wooden-trouble"
        assert caught.value.context["effective_episode_seed"] == 52003
        assert caught.value.context["action_id"] == 68
    assert session.moves_remaining == env.config.move_limit
    for field in ("tiles", "specials", "blockers", "blocker_hp", "cell_mask"):
        np.testing.assert_array_equal(
            getattr(session.board, field), getattr(original, field)
        )
    session.reset()
    assert session.engine.max_cascades == 100


def test_wooden_trouble_seed_52003_is_a_finite_124_phase_cascade():
    """The approved default resolves the original failure without changing its chain."""
    env = Match3LevelEnv("wooden-trouble")
    _, info = env.reset(seed=52003)
    original = env.session.board.copy()
    engine = Match3Engine(
        num_colors=env.config.num_colors,
        seed=info["engine_seed"],
    )
    result = engine.step(original, ((4, 2), (4, 3)), record_events=True)
    assert result.valid
    assert result.cascade_count == 124
    assert result.cleared_tiles == 1868
    assert not result.reshuffled
    assert not detect_patterns(result.state.tiles)
    assert find_valid_moves(result.state)
    assert not result.state.blockers.any()
    assert not result.state.blocker_hp.any()
    assert len(result.blocker_damage) == 4
    assert sum(len(item.activation_steps) for item in result.activation_history) == 101

    refills = [
        event.state
        for event in result.animation_events
        if event.event_type == AnimationEventType.REFILL
    ]
    assert len(refills) == 124
    assert (
        len({(state.tiles.tobytes(), state.specials.tobytes()) for state in refills})
        == 124
    )
    assert all(detect_patterns(state.tiles) for state in refills[:-1])
    assert not detect_patterns(refills[-1].tiles)
    for state in refills:
        np.testing.assert_array_equal(state.cell_mask, original.cell_mask)
        assert np.all(state.tiles[state.cell_mask] > 0)
        assert np.all(state.tiles[~state.cell_mask] == 0)
        assert np.all(state.specials[~state.cell_mask] == 0)
    assert engine.max_cascades == env.session.engine.max_cascades == 200
    for field in ("tiles", "specials", "blockers", "blocker_hp", "cell_mask"):
        np.testing.assert_array_equal(
            getattr(env.session.board, field), getattr(original, field)
        )

    first = env.step(68)
    env.reset(seed=52003)
    second = env.step(68)
    for key in first[0]:
        np.testing.assert_array_equal(first[0][key], second[0][key])
    assert first[1:] == second[1:]
    assert first[4]["cascades"] == 124
    assert first[4]["level_status"] == "WON"


@pytest.mark.parametrize("budget", [0, -1, True, 2.5, "200", None])
def test_invalid_cascade_budgets_are_rejected(budget):
    with pytest.raises(ValueError, match="positive integer"):
        Match3Engine(num_colors=3, max_cascades=budget)
    with pytest.raises(ValueError, match="positive integer"):
        Match3LevelEnv(max_cascades=budget)
    config = Match3LevelEnv().config
    with pytest.raises(ValueError, match="positive integer"):
        LevelSession(config, max_cascades=budget)
