from dataclasses import replace

import numpy as np
import pytest

from match3_ai.evaluation.experiments import LevelExperimentConfig
from match3_ai.evaluation.initialization_benchmark import (
    benchmark_initialization,
    compare_first_match_colors,
)
from match3_ai.games.match3.environment import Match3Env, Match3LevelEnv
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.pattern_detection import detect_patterns
from match3_ai.levels import LevelGenerationError, LevelSession, create_level_board


@pytest.fixture
def historical_three_color_level(monkeypatch):
    original = Match3LevelEnv._load_packaged_levels

    def load_configs():
        configs = original()
        configs["first-match"] = replace(configs["first-match"], num_colors=3)
        return configs

    monkeypatch.setattr(
        Match3LevelEnv, "_load_packaged_levels", staticmethod(load_configs)
    )


def test_seed_52028_initializes_under_level_environment_default_and_replays():
    env = Match3LevelEnv("first-match")
    first, first_info = env.reset(seed=52028)
    assert first_info["max_initialization_attempts"] == 1000
    assert first_info["effective_episode_seed"] == 52028
    assert first_info["board_seed"] == 783324025
    assert first_info["engine_seed"] == 3597872301
    assert env.session.max_initialization_attempts == 1000
    assert env.config.num_colors == 5
    assert not detect_patterns(env.session.board.tiles)
    assert find_valid_moves(env.session.board)
    assert env.observation_space.contains(first)
    action = int(np.flatnonzero(first["action_mask"])[0])
    transition = env.step(action)
    replay, replay_info = env.reset(seed=52028)
    assert first_info == replay_info
    for key in first:
        np.testing.assert_array_equal(first[key], replay[key])
    repeated = env.step(action)
    for key in transition[0]:
        np.testing.assert_array_equal(transition[0][key], repeated[0][key])
    assert transition[1:] == repeated[1:]


def test_generator_and_session_defaults_remain_1000_with_explicit_override():
    config = replace(Match3LevelEnv("first-match").config, num_colors=3)
    with pytest.raises(LevelGenerationError, match="first-match.*after 1000 attempts"):
        create_level_board(config, seed=783324025)
    with pytest.raises(LevelGenerationError, match="after 1000 attempts"):
        LevelSession(config, board_seed=783324025)
    session = LevelSession(
        config, board_seed=783324025, max_initialization_attempts=5000
    )
    initial = session.board.copy()
    initial_rng = session.engine.rng.random(3)
    session.reset()
    assert session.max_initialization_attempts == 5000
    np.testing.assert_array_equal(initial_rng, session.engine.rng.random(3))
    for name in ("tiles", "specials", "cell_mask", "blockers", "blocker_hp"):
        np.testing.assert_array_equal(
            getattr(initial, name), getattr(session.board, name)
        )


@pytest.mark.parametrize("budget", [1, 1000])
def test_explicit_low_environment_budget_fails_clearly_without_seed_retry(
    budget, historical_three_color_level
):
    env = Match3LevelEnv("first-match", max_initialization_attempts=budget)
    with pytest.raises(
        LevelGenerationError, match=f"first-match.*after {budget} attempts"
    ):
        env.reset(seed=52028)
    assert env.session is None


@pytest.mark.parametrize("budget", [0, -1, True, None, 1.5, "5000"])
def test_invalid_initialization_budgets_rejected_at_every_adapter(budget):
    config = Match3LevelEnv().config
    for constructor, kwargs in (
        (Match3LevelEnv, {}),
        (LevelSession, {"config": config}),
        (LevelExperimentConfig, {}),
    ):
        with pytest.raises(ValueError, match="positive integer"):
            constructor(**kwargs, max_initialization_attempts=budget)


def test_larger_budget_does_not_change_successful_boards_or_gameplay():
    low = Match3LevelEnv(max_initialization_attempts=1000)
    high = Match3LevelEnv(max_initialization_attempts=5000)
    first, _ = low.reset(seed=52003)
    second, _ = high.reset(seed=52003)
    for key in first:
        np.testing.assert_array_equal(first[key], second[key])
    action = int(np.flatnonzero(first["action_mask"])[0])
    result_low, result_high = low.step(action), high.step(action)
    for key in result_low[0]:
        np.testing.assert_array_equal(result_low[0][key], result_high[0][key])
    assert result_low[1:] == result_high[1:]
    legacy = Match3Env()
    observation, _ = legacy.reset(seed=52003)
    assert legacy.observation_space.contains(observation)
    assert legacy.action_space.n == 112


def test_all_levels_and_different_seeds_remain_valid_and_keep_configuration_seeds():
    env = Match3LevelEnv()
    for level in env.level_ids:
        observations = []
        for seed in (52028, 52029):
            observation, info = env.reset(seed=seed, options={"level_id": level})
            assert info["configured_level_seed"] == env.config.seed
            assert info["max_initialization_attempts"] == 1000
            assert observation["action_mask"].any()
            assert not detect_patterns(env.session.board.tiles)
            assert env.observation_space.contains(observation)
            observations.append(observation)
        assert not np.array_equal(observations[0]["tiles"], observations[1]["tiles"])


def test_initialization_benchmark_counts_actual_attempts_and_keeps_reproducibility(
    historical_three_color_level,
):
    kwargs = {
        "level_ids": ("first-match",),
        "seeds": (52028,),
        "max_initialization_attempts": 5000,
    }
    result = benchmark_initialization(**kwargs)
    repeated = benchmark_initialization(**kwargs)
    assert result["records"][0]["attempts"] == 1180
    assert result["summary"][0]["success_rate"] == 1
    assert (
        result["records"][0]["board_fingerprint"]
        == repeated["records"][0]["board_fingerprint"]
    )
    failed = benchmark_initialization(
        ("first-match",), (52028,), max_initialization_attempts=1000
    )
    assert failed["records"][0]["attempts"] == 1000
    assert failed["summary"][0]["failures"] == 1


def test_paired_color_comparison_changes_only_colors_and_validates_accepted_boards():
    result = compare_first_match_colors((52028,))
    three, five = result["summary"]
    assert three["successes"] == 0 and three["mean_attempts_all"] == 1000
    assert five["successes"] == five["valid_boards"] == 1
    before = dict(result["configurations"]["3"])
    after = dict(result["configurations"]["5"])
    assert before.pop("num_colors") == 3
    assert after.pop("num_colors") == 5
    assert before == after
    first, second = result["records"]
    assert first["board_seed"] == second["board_seed"]
    assert first["engine_seed"] == second["engine_seed"]
    assert second["initial_patterns"] == 0 and second["legal_moves"] > 0
