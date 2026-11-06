import csv
import json
from dataclasses import replace

import numpy as np
import pytest

from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.evaluation.analytics import AnalyticsCollector, export_evaluation
from match3_ai.evaluation.experiments import LevelExperimentConfig, run_level_experiment
from match3_ai.evaluation.runner import run_episode
from match3_ai.games.match3.environment import Match3LevelEnv


class MaskedActionAgent:
    def select_action(self, observation):
        return int(np.flatnonzero(observation["action_mask"] == 0)[0])


def test_known_cascade_completes_as_win_with_default_budget():
    collector = AnalyticsCollector()
    result = run_episode(
        Match3LevelEnv("wooden-trouble"),
        GreedyAgent(),
        seed=52003,
        agent_seed=93003,
        collector=collector,
    )
    assert result.outcome == "WON"
    assert result.completed
    assert result.total_cascades == result.max_cascades_in_move == 124
    assert result.cascade_limit == 200
    assert result.moves_used == 1
    assert result.blockers_remaining == result.blocker_hp_remaining == 0
    assert result.objective_completion_ratio == 1
    assert result.special_activations == 101
    assert result.inference_time_seconds >= 0
    assert collector.moves[0].action_id == 68
    collector.validate()


def test_limit_error_preserves_diagnostics_and_attempt_trace():
    collector = AnalyticsCollector()
    result = run_episode(
        Match3LevelEnv("wooden-trouble", max_cascades=100),
        GreedyAgent(),
        seed=52003,
        agent_seed=17,
        collector=collector,
    )
    assert result.outcome == "ERROR"
    assert not result.completed
    assert result.error_type == "CascadeLimitExceeded"
    assert result.error_context["max_cascades"] == 100
    assert result.error_context["cascades_processed"] == 100
    assert result.error_context["episode_seed"] == 52003
    assert result.error_context["action_id"] == 68
    assert result.error_context["level_id"] == "wooden-trouble"
    assert result.total_cascades == result.max_cascades_in_move == 100
    assert result.moves_used == 0
    assert result.total_reward == 0
    assert len(collector.moves) == 1
    move = collector.moves[0]
    assert move.action_id == 68
    assert move.error_type == "CascadeLimitExceeded"
    assert not move.resolution_completed
    assert move.reward is move.valid_action is move.blocker_damage is None
    assert move.cascade_count == 100
    collector.validate()


def test_safety_bound_is_not_reported_as_a_loss():
    collector = AnalyticsCollector()
    result = run_episode(
        Match3LevelEnv(),
        MaskedActionAgent(),
        seed=12,
        max_steps=3,
        collector=collector,
    )
    assert result.outcome == "TIME_LIMIT"
    assert result.termination_reason == "max_steps"
    assert not result.completed
    assert result.moves_used == 0
    assert result.steps == result.invalid_actions == 3
    assert [move.move_index for move in collector.moves] == [1, 2, 3]
    collector.validate()


@pytest.mark.parametrize(
    ("status", "terminated", "truncated", "reason", "expected"),
    [
        ("WON", True, False, None, "WON"),
        ("LOST", True, False, None, "LOST"),
        ("IN_PROGRESS", True, False, "no_legal_actions", "ERROR"),
        ("IN_PROGRESS", False, True, None, "TIME_LIMIT"),
    ],
)
def test_outcome_uses_level_status_and_termination_contract(
    monkeypatch, status, terminated, truncated, reason, expected
):
    env = Match3LevelEnv()
    original_step = env.step

    def step(action):
        observation, reward, _, _, info = original_step(action)
        info["level_status"] = status
        info["termination_reason"] = reason
        return observation, reward, terminated, truncated, info

    monkeypatch.setattr(env, "step", step)
    result = run_episode(env, RandomAgent(10), seed=11)
    assert result.outcome == expected
    if reason:
        assert result.termination_reason == reason


def test_reset_and_agent_exceptions_become_explicit_errors(monkeypatch):
    env = Match3LevelEnv()

    def broken_reset(**kwargs):
        raise ValueError("reset failed")

    monkeypatch.setattr(env, "reset", broken_reset)
    collector = AnalyticsCollector()
    result = run_episode(env, GreedyAgent(), seed=4, collector=collector)
    assert result.outcome == "ERROR"
    assert result.error_message == "reset failed"
    assert result.steps == 0
    assert result.objective_completion_ratio is None
    assert not collector.moves
    collector.validate()


def test_agent_failure_has_a_missing_action_and_no_invented_step_results():
    class BrokenAgent:
        def select_action(self, observation):
            raise LookupError("agent failed")

    collector = AnalyticsCollector()
    result = run_episode(Match3LevelEnv(), BrokenAgent(), seed=13, collector=collector)
    assert result.outcome == "ERROR"
    assert result.error_type == "LookupError"
    assert collector.moves[0].action_id is None
    assert collector.moves[0].reward is None
    assert collector.moves[0].cascade_count is None
    collector.validate()


def test_factory_failures_are_recorded_without_move_records():
    def broken_factory(seed):
        raise ValueError("factory failed")

    collector = AnalyticsCollector()
    result = run_level_experiment(
        {"Broken": broken_factory},
        LevelExperimentConfig(level_ids=("first-match",), num_episodes=1),
        collector=collector,
    )
    assert result.episodes[0].outcome == "ERROR"
    assert result.episodes[0].termination_reason == "setup_exception"
    assert result.episodes[0].error_message == "factory failed"
    assert not collector.moves
    collector.validate()


def test_analytics_does_not_change_reproducible_episode_or_action_trace():
    env = Match3LevelEnv()
    plain = run_episode(env, RandomAgent(81), seed=80, agent_seed=81)
    collector = AnalyticsCollector()
    collected = run_episode(
        env, RandomAgent(81), seed=80, agent_seed=81, collector=collector
    )
    assert plain == collected
    other = AnalyticsCollector()
    replay = run_episode(env, RandomAgent(81), seed=80, agent_seed=81, collector=other)
    assert replay == collected
    assert other.moves == collector.moves
    collector.validate()


def test_experiment_pairs_environment_seeds_and_separates_agent_seeds():
    config = LevelExperimentConfig(
        level_ids=("first-match", "wooden-trouble"),
        num_episodes=2,
        environment_seeds=(70, 71),
        agent_seeds=(90, 91),
    )
    constructed = []

    def make_agent(seed):
        constructed.append(seed)
        return RandomAgent(seed)

    collector = AnalyticsCollector()
    result = run_level_experiment(
        {"A": make_agent, "B": make_agent}, config, collector=collector
    )
    assert constructed == [90, 91] * 4
    assert len(result.episodes) == 8
    assert len({item.episode_id for item in result.episodes}) == 8
    for level in config.level_ids:
        first = [
            item
            for item in result.episodes
            if item.level_id == level and item.agent_name == "A"
        ]
        second = [
            item
            for item in result.episodes
            if item.level_id == level and item.agent_name == "B"
        ]
        assert [item.seed for item in first] == [70, 71]
        assert [item.board_seed for item in first] == [
            item.board_seed for item in second
        ]
        assert all(item.agent_seed != item.seed for item in first + second)
    replay = run_level_experiment({"A": make_agent, "B": make_agent}, config)
    assert replay.episodes == result.episodes
    assert replay.metadata["experiment_id"] == result.metadata["experiment_id"]
    assert all(item.level_config_id for item in result.episodes)
    collector.validate()


def test_csv_metadata_and_foreign_keys_round_trip_without_overwrite(tmp_path):
    collector = AnalyticsCollector()
    result = run_level_experiment(
        {"Random": RandomAgent},
        LevelExperimentConfig(level_ids=("first-match",), num_episodes=2),
        collector=collector,
    )
    output = export_evaluation(collector, result.metadata, tmp_path / "experiment")
    with (output / "episodes.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    with (output / "moves.csv").open(newline="") as handle:
        moves = list(csv.DictReader(handle))
    assert len(episodes) == 2
    assert len(moves) == sum(item.steps for item in result.episodes)
    assert all(
        move["episode_id"] in {item["episode_id"] for item in episodes}
        for move in moves
    )
    assert json.loads(episodes[0]["objective_progress"])
    metadata = json.loads((output / "experiment_metadata.json").read_text())
    assert metadata["experiment_id"] == result.metadata["experiment_id"]
    assert metadata["config"]["environment_seeds"] == [52000, 52001]
    assert metadata["summary"][0]["win_rate_denominator"] == 2
    original = (output / "episodes.csv").read_bytes()
    with pytest.raises(FileExistsError):
        export_evaluation(collector, result.metadata, output)
    assert (output / "episodes.csv").read_bytes() == original


def test_collector_rejects_duplicates_orphans_and_detaches_nested_records():
    collector = AnalyticsCollector()
    result = run_episode(Match3LevelEnv(), RandomAgent(1), seed=2, collector=collector)
    with pytest.raises(ValueError, match="duplicate episode"):
        collector.record_episode(result)
    with pytest.raises(ValueError, match="duplicate move"):
        collector.record_move(collector.moves[0])
    orphan = AnalyticsCollector()
    orphan.record_move(replace(collector.moves[0], episode_id="unknown"))
    with pytest.raises(ValueError, match="unknown episode"):
        orphan.validate()
    result.objective_progress[0]["progress"] = 999999
    assert collector.episodes[0].objective_progress[0]["progress"] != 999999


def test_repeated_cascade_errors_stop_experiment_without_retries():
    collector = AnalyticsCollector()
    result = run_level_experiment(
        {"Greedy": lambda seed: GreedyAgent()},
        LevelExperimentConfig(
            level_ids=("wooden-trouble",),
            num_episodes=3,
            environment_seeds=(52003, 52003, 52003),
            max_cascades=100,
        ),
        collector=collector,
    )
    assert len(result.episodes) == 2
    assert all(item.outcome == "ERROR" for item in result.episodes)
    assert result.metadata["interruption_reason"] == "repeated_cascade_limit_errors"
    assert result.metadata["planned_episodes"] == 3
    assert result.metadata["summary"][0]["errors"] == 2
    collector.validate()


def test_wall_time_bound_stops_before_launching_another_episode(monkeypatch):
    from match3_ai.evaluation import experiments

    times = iter([0.0, 2.0, 2.0])
    monkeypatch.setattr(experiments, "perf_counter", lambda: next(times))
    result = run_level_experiment(
        {"Random": RandomAgent},
        LevelExperimentConfig(
            level_ids=("first-match",), num_episodes=1, max_duration_seconds=1
        ),
    )
    assert not result.episodes
    assert result.metadata["interruption_reason"] == "experiment_wall_time_limit"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"num_episodes": 0},
        {"max_steps": True},
        {"max_cascades": 0},
        {"environment_seeds": (1,)},
        {"agent_seeds": (-1,)},
        {"max_duration_seconds": float("nan")},
        {"level_ids": ("x", "x")},
    ],
)
def test_experiment_config_rejects_invalid_bounds_and_seed_sets(kwargs):
    with pytest.raises(ValueError):
        LevelExperimentConfig(**kwargs)
