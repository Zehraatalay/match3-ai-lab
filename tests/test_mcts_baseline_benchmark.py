"""Paired benchmark integration, persistence, and metric regression tests."""

import csv
import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.agents.stochastic_mcts_agent import StochasticMCTSAgent
from match3_ai.evaluation import mcts_baseline_benchmark as benchmark
from match3_ai.evaluation.runner import EpisodeResult, run_episode


def record(**kwargs):
    defaults = {
        "seed": 10,
        "total_reward": -1,
        "steps": 1,
        "valid_actions": 1,
        "invalid_actions": 0,
        "total_cleared": 0,
        "total_cascades": 0,
        "reshuffles": 0,
        "completed": True,
        "episode_id": "episode",
        "level_id": benchmark.LEVELS[0],
        "agent_name": "Random",
        "outcome": "LOST",
        "moves_used": 6,
        "objective_completion_ratio": 0.5,
    }
    return EpisodeResult(**(defaults | kwargs))


def timing(episode_id, calls=1, decision=1.0, elapsed=2.0):
    return {
        "episode_id": episode_id,
        "action_selection_calls": calls,
        "decision_time_seconds": decision,
        "episode_time_seconds": elapsed,
    }


@pytest.mark.parametrize(
    "name,kind,policy,value",
    [
        ("Random", RandomAgent, None, None),
        ("Greedy", GreedyAgent, None, None),
        ("MCTS_random_terminal", StochasticMCTSAgent, "random", "terminal"),
        ("MCTS_greedy_outcome", StochasticMCTSAgent, "greedy", "outcome"),
    ],
)
def test_agent_configuration_mapping(name, kind, policy, value):
    env = benchmark.BenchmarkEnvironment(benchmark.LEVELS[0])
    try:
        env.reset(seed=100)
        agent = benchmark.make_agent(name, env, 200, 7)
        assert isinstance(agent, kind)
        if policy:
            assert agent.rollout_policy == policy and agent.value_mode == value
            assert agent.simulations == 7 and agent.simulator.env is env
    finally:
        env.close()


def test_seed_schedule_is_independent_and_repeatable():
    config = benchmark.BenchmarkConfig(episodes=2)
    seeds = [
        config.agent_seed(level, episode, agent)
        for level in range(3)
        for episode in range(2)
        for agent in range(4)
    ]
    assert len(set(seeds)) == 24
    assert not set(seeds) & set(range(config.env_seed_start, config.env_seed_start + 2))
    assert seeds == [
        config.agent_seed(level, episode, agent)
        for level in range(3)
        for episode in range(2)
        for agent in range(4)
    ]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episodes": 0},
        {"simulations": -1},
        {"episodes": True},
        {"env_seed_start": -1},
        {"agent_seed_start": -1},
        {"env_seed_start": 20_000_000},
        {"env_seed_start": 400000},
    ],
)
def test_invalid_bounds(kwargs):
    with pytest.raises(ValueError):
        benchmark.BenchmarkConfig(**kwargs)


def test_lazy_binding_tracks_live_environment_and_does_not_mutate_it():
    env = benchmark.BenchmarkEnvironment(benchmark.LEVELS[0])
    adapter = benchmark.EpisodeAgent("MCTS_random_terminal", env, 200, 2, {}, 100)
    assert adapter.agent is None and env.session is None
    try:
        observation, _ = env.reset(seed=100)
        before = benchmark.complete_state_fingerprint(env)
        action = adapter.select_action(observation)
        assert benchmark.complete_state_fingerprint(env) == before
        assert adapter.agent.simulator.env is env and adapter.pairing_verified
        observation, _, terminated, truncated, _ = env.step(action)
        if not terminated and not truncated:
            after_step = benchmark.complete_state_fingerprint(env)
            second = adapter.select_action(observation)
            assert observation["action_mask"][second]
            assert benchmark.complete_state_fingerprint(env) == after_step
        assert adapter.calls >= 1
    finally:
        env.close()


def test_pairing_mismatch_prevents_action_selection(monkeypatch):
    env = benchmark.BenchmarkEnvironment(benchmark.LEVELS[0])
    try:
        observation, _ = env.reset(seed=100)
        identities = {(env.level_id, 100): ("wrong", 1, 2, 3)}
        adapter = benchmark.EpisodeAgent("Random", env, 200, 2, identities, 100)
        monkeypatch.setattr(
            benchmark,
            "make_agent",
            lambda *a: pytest.fail("agent constructed before pairing"),
        )
        with pytest.raises(RuntimeError, match="pairing mismatch"):
            adapter.select_action(observation)
        assert adapter.calls == 0 and not adapter.pairing_verified
    finally:
        env.close()


def test_mutating_search_is_detected(monkeypatch):
    env = benchmark.BenchmarkEnvironment(benchmark.LEVELS[0])

    class BrokenSearch:
        def select_action(self, observation):
            env.session.engine.rng.random()
            return int(np.flatnonzero(observation["action_mask"])[0])

    monkeypatch.setattr(benchmark, "make_agent", lambda *args: BrokenSearch())
    try:
        observation, _ = env.reset(seed=100)
        adapter = benchmark.EpisodeAgent("MCTS_random_terminal", env, 200, 2, {}, 100)
        with pytest.raises(RuntimeError, match="mutated"):
            adapter.select_action(observation)
        assert adapter.integrity_error and adapter.calls == 1
    finally:
        env.close()


@pytest.mark.parametrize("name", list(benchmark.AGENTS))
def test_repeated_episode_outcomes_and_initial_identities_match(name):
    results, identities = [], []
    for _ in range(2):
        env = benchmark.BenchmarkEnvironment(benchmark.LEVELS[0])
        try:
            adapter = benchmark.EpisodeAgent(name, env, 200, 2, {}, 100)
            result = run_episode(
                env,
                adapter,
                seed=100,
                max_steps=env.config.move_limit,
                agent_name=name,
                agent_seed=200,
            )
            values = asdict(result)
            values.pop("inference_time_seconds")
            results.append(values)
            identities.append(env.initial_identity)
        finally:
            env.close()
    assert results[0] == results[1] and identities[0] == identities[1]
    assert result.outcome in ("WON", "LOST")


def test_metric_denominators_and_missing_values():
    records = [
        record(
            episode_id="one", outcome="WON", moves_used=2, objective_completion_ratio=1
        ),
        record(
            episode_id="two",
            outcome="ERROR",
            completed=False,
            objective_completion_ratio=None,
        ),
    ]
    rows = benchmark.summarize(
        records, [timing("one", 1, 4, 5), timing("two", 3, 6, 7)]
    )
    row = rows[0]
    assert row["win_rate"] == 0.5 and row["attempted"] == 2 and row["errors"] == 1
    assert row["mean_decision_time_seconds"] == 2.5
    assert row["action_selection_calls"] == 4
    assert row["mean_objective_completion"] == 1 and row["n_objective_completion"] == 1
    assert row["mean_winning_moves"] == 2 and row["n_winning_moves"] == 1
    assert row["mean_episode_time_seconds"] == 6
    assert row["win_rate_ci95_low"] == pytest.approx(0.0945312)
    assert row["win_rate_ci95_high"] == pytest.approx(0.9054688)


def test_zero_wins_and_zero_decisions_remain_missing():
    row = benchmark.summarize(
        [record(objective_completion_ratio=None)], [timing("episode", 0, 0)]
    )[0]
    assert row["mean_winning_moves"] is None and row["n_winning_moves"] == 0
    assert row["mean_decision_time_seconds"] is None
    assert row["mean_objective_completion"] is None
    assert row["win_rate"] == 0 and row["win_rate_ci95_high"] > 0


@pytest.fixture
def fake_runner(monkeypatch):
    class Agent:
        def select_action(self, observation):
            return int(np.flatnonzero(observation["action_mask"])[0])

    monkeypatch.setattr(benchmark, "make_agent", lambda *args: Agent())

    def run(
        env, agent, *, seed, experiment_id, episode_id, agent_name, agent_seed, **kwargs
    ):
        observation, info = env.reset(seed=seed)
        agent.select_action(observation)
        return record(
            seed=seed,
            episode_id=episode_id,
            experiment_id=experiment_id,
            agent_name=agent_name,
            agent_seed=agent_seed,
            level_id=env.level_id,
            board_seed=info["board_seed"],
            engine_seed=info["engine_seed"],
            level_config_id=info["level_config_id"],
        )

    monkeypatch.setattr(benchmark, "run_episode", run)
    return run


def test_paired_incremental_csv_and_complete_manifest(tmp_path, fake_runner):
    output = tmp_path / "new"
    rows = benchmark.run_benchmark(
        benchmark.BenchmarkConfig(episodes=1, simulations=2), output
    )
    assert len(rows) == 12
    with (output / "episodes.csv").open(newline="") as handle:
        episodes = list(csv.DictReader(handle))
    assert len(episodes) == 12
    assert len({row["agent_seed"] for row in episodes}) == 12
    for level in benchmark.LEVELS:
        selected = [row for row in episodes if row["level_id"] == level]
        assert (
            len(
                {
                    (
                        row["seed"],
                        row["initial_fingerprint"],
                        row["board_seed"],
                        row["engine_seed"],
                    )
                    for row in selected
                }
            )
            == 1
        )
        assert all(row["pairing_verified"] == "True" for row in selected)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete" and manifest["records_written"] == 12
    assert len(manifest["level_sha256"]) == 3
    assert "package_source_sha256" in manifest and "denominators" in manifest


def test_error_and_time_limit_records_are_preserved(tmp_path, fake_runner, monkeypatch):
    def failed(*args, **kwargs):
        result = fake_runner(*args, **kwargs)
        return replace(
            result,
            outcome="ERROR" if kwargs["agent_name"] == "Random" else "TIME_LIMIT",
            completed=False,
            error_type="InjectedError",
            error_message="preserved",
        )

    monkeypatch.setattr(benchmark, "run_episode", failed)
    output = tmp_path / "errors"
    rows = benchmark.run_benchmark(
        benchmark.BenchmarkConfig(episodes=1, simulations=2), output
    )
    assert sum(row["errors"] for row in rows) == 3
    assert sum(row["time_limits"] for row in rows) == 9
    assert (
        json.loads((output / "manifest.json").read_text())["status"]
        == "complete_with_errors"
    )
    assert "InjectedError" in (output / "episodes.csv").read_text()


def test_interruption_preserves_previous_episode_and_summary(
    tmp_path, fake_runner, monkeypatch
):
    output = tmp_path / "interrupted"
    calls = []

    def interrupted(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            with (output / "episodes.csv").open(newline="") as handle:
                assert len(list(csv.DictReader(handle))) == 1
            raise KeyboardInterrupt("test interruption")
        return fake_runner(*args, **kwargs)

    monkeypatch.setattr(benchmark, "run_episode", interrupted)
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_benchmark(benchmark.BenchmarkConfig(episodes=1), output)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "interrupted" and manifest["records_written"] == 1
    assert manifest["attempted_episodes"] == 2 and manifest["active_episode"]
    with (output / "summary.csv").open(newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 1


def test_existing_output_is_untouched(tmp_path, monkeypatch):
    marker = tmp_path / "history.txt"
    marker.write_text("preserved")
    monkeypatch.setattr(
        benchmark, "build_manifest", lambda _: pytest.fail("built inputs")
    )
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(benchmark.BenchmarkConfig(), tmp_path)
    assert marker.read_text() == "preserved"
