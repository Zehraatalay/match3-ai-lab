"""Controlled configuration, provenance, import, and paired contrast checks."""

import csv
import json
import shutil

import pytest

from match3_ai.evaluation import mcts_baseline_benchmark as benchmark
from match3_ai.evaluation.mcts_ablation import (
    analyze_results,
    digest,
    paired_comparisons,
    validate_reuse,
)
from match3_ai.evaluation.mcts_provenance import (
    IMPLEMENTATION_VERSION,
    LEGACY_IMPLEMENTATION,
)


@pytest.mark.parametrize("names", [(), ("bad",), ("Random", "Random")])
def test_invalid_selections(names):
    with pytest.raises(ValueError, match="agents"):
        benchmark.BenchmarkConfig(agents=names)


def test_legacy_seeds_do_not_depend_on_selected_agents():
    default = benchmark.BenchmarkConfig(
        episodes=100, env_seed_start=10001000, agent_seed_start=20010000
    )
    ablation = benchmark.BenchmarkConfig(
        100, 30, 10001000, 20010000, benchmark.MCTS_ABLATION
    )
    assert default.agents == tuple(benchmark.AGENTS)
    old, new = [], []
    for level in range(3):
        for episode in range(100):
            for index in range(4):
                expected = 20010000 + ((level * 100 + episode) * 4 + index)
                assert default.agent_seed(level, episode, index) == expected
                assert ablation.agent_seed(level, episode, index) == expected
                old.append(expected)
            new.extend(ablation.agent_seed(level, episode, index) for index in (4, 5))
    assert len(set(new)) == 600 and set(old).isdisjoint(new)
    assert benchmark.AGENT_REGISTRY["MCTS_random_outcome"] == ("random", "outcome")
    assert benchmark.AGENT_REGISTRY["MCTS_greedy_terminal"] == ("greedy", "terminal")


@pytest.fixture(scope="module")
def source_run(tmp_path_factory):
    path = tmp_path_factory.mktemp("ablation") / "source"
    benchmark.run_benchmark(benchmark.BenchmarkConfig(episodes=1, simulations=1), path)
    return path


def test_complete_reuse_imports_only_missing_configurations(source_run, tmp_path):
    config = benchmark.BenchmarkConfig(1, 1, agents=benchmark.MCTS_ABLATION)
    before = {
        path.name: digest(path) for path in source_run.iterdir() if path.is_file()
    }
    imported, provenance = validate_reuse(config, source_run)
    assert len(imported) == provenance["records_reusable"] == 6
    assert provenance["new_episodes_required"] == 6
    output = tmp_path / "combined"
    summary = benchmark.run_benchmark(config, output, reuse_from=source_run)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["records_reused"] == manifest["records_new"] == 6
    assert manifest["status"] == "complete" and len(summary) == 12
    with (output / "episodes.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert (
        len({(row["level_id"], row["seed"], row["agent_name"]) for row in rows}) == 12
    )
    assert len({row["episode_id"] for row in rows}) == 12
    assert all(
        row["origin_manifest_sha256"]
        for row in rows
        if row["record_origin"] == "imported"
    )
    analysis = analyze_results(output, tmp_path / "analysis", samples=100)
    assert len(analysis) == 60 and all(row["n_pairs"] == 1 for row in analysis)
    assert before == {
        path.name: digest(path) for path in source_run.iterdir() if path.is_file()
    }


@pytest.mark.parametrize(
    "problem",
    [
        "incomplete",
        "semantics",
        "critical_source",
        "level",
        "weights",
        "agent_seed",
        "environment_seed",
        "fingerprint",
        "duplicate",
        "missing",
        "budget",
        "error",
        "timing",
    ],
)
def test_incompatible_history_is_rejected_before_output(source_run, tmp_path, problem):
    source = tmp_path / "copy"
    shutil.copytree(source_run, source)
    manifest = json.loads((source / "manifest.json").read_text())
    with (source / "episodes.csv").open() as handle:
        reader = csv.DictReader(handle)
        fields = reader.fieldnames
        rows = list(reader)
    if problem == "incomplete":
        manifest["status"] = "interrupted"
    elif problem == "semantics":
        manifest["algorithm_semantics"] = "different"
    elif problem == "critical_source":
        manifest["semantic_source_sha256"] = {}
    elif problem == "level":
        manifest["level_sha256"][benchmark.LEVELS[0]] = "wrong"
    elif problem == "weights":
        manifest["agents"]["MCTS_random_terminal"]["win_efficiency_weight"] = 0.5
    elif problem == "agent_seed":
        rows[0]["agent_seed"] = "999"
    elif problem == "environment_seed":
        rows[0]["effective_episode_seed"] = "999"
    elif problem == "fingerprint":
        rows[0]["initial_fingerprint"] = "wrong"
    elif problem == "duplicate":
        rows[1] = rows[0]
    elif problem == "missing":
        rows.pop()
    elif problem == "budget":
        manifest["config"]["simulations"] = 30
    elif problem == "error":
        rows[0]["outcome"] = "ERROR"
    elif problem == "timing":
        rows[0]["decision_time_seconds"] = "nan"
    (source / "manifest.json").write_text(json.dumps(manifest))
    with (source / "episodes.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output = tmp_path / "must_not_exist"
    with pytest.raises(ValueError):
        benchmark.run_benchmark(
            benchmark.BenchmarkConfig(episodes=1, simulations=1),
            output,
            reuse_from=source,
        )
    assert not output.exists()


def factorial_rows():
    rows = []
    values = {
        "MCTS_random_terminal": [0, 0, 1, 1],
        "MCTS_random_outcome": [0, 1, 1, 1],
        "MCTS_greedy_terminal": [0, 1, 0, 1],
        "MCTS_greedy_outcome": [1, 1, 1, 1],
    }
    for name, wins in values.items():
        for seed, win in enumerate(wins):
            rows.append(
                {
                    "level_id": "test",
                    "seed": seed,
                    "agent_name": name,
                    "initial_fingerprint": str(seed),
                    "board_seed": seed,
                    "engine_seed": seed + 1,
                    "level_config_id": "level",
                    "simulations": 30,
                    "outcome": "WON" if win else "LOST",
                    "objective_completion_ratio": float(win),
                    "decision_time_seconds": 2.0,
                    "action_selection_calls": 2,
                    "episode_time_seconds": 3.0,
                    "implementation_version": IMPLEMENTATION_VERSION,
                }
            )
    return rows


def test_factorial_effects_exact_and_reproducible():
    rows = factorial_rows()
    comparisons = paired_comparisons(rows, samples=200)
    assert comparisons == paired_comparisons(rows, samples=200)
    wins = {row["contrast"]: row for row in comparisons if row["metric"] == "win"}
    assert wins["rollout_terminal_GT_minus_RT"]["difference"] == 0
    assert wins["rollout_outcome_GO_minus_RO"]["difference"] == 0.25
    assert wins["value_random_RO_minus_RT"]["difference"] == 0.25
    assert wins["value_greedy_GO_minus_GT"]["difference"] == 0.5
    assert wins["interaction"]["difference"] == 0.25
    assert wins["rollout_terminal_GT_minus_RT"]["left_only_wins"] == 1
    assert wins["rollout_terminal_GT_minus_RT"]["right_only_wins"] == 1
    assert all(
        row["n_pairs"] == 4 and row["ci95_low"] <= row["difference"] <= row["ci95_high"]
        for row in comparisons
    )


def test_mixed_implementation_timings_not_compared():
    rows = factorial_rows()
    for row in rows:
        if row["agent_name"] in ("MCTS_random_terminal", "MCTS_greedy_outcome"):
            row["implementation_version"] = LEGACY_IMPLEMENTATION
    comparisons = paired_comparisons(rows, samples=100)
    assert all(
        row["status"] == "mixed_timing_versions" and row["difference"] is None
        for row in comparisons
        if row["metric"] in ("decision_latency", "episode_time")
    )
    assert all(row["n_pairs"] == 4 for row in comparisons if row["metric"] == "win")


@pytest.mark.parametrize("problem", ["duplicate", "unmatched", "fingerprint", "budget"])
def test_invalid_paired_input_rejected(problem):
    rows = factorial_rows()
    if problem == "duplicate":
        rows.append(rows[0])
    elif problem == "unmatched":
        rows.pop()
    elif problem == "fingerprint":
        rows[0]["initial_fingerprint"] = "wrong"
    else:
        rows[0]["simulations"] = 20
    with pytest.raises(ValueError):
        paired_comparisons(rows, samples=100)
