import csv
import json

import pytest

from match3_ai.evaluation.difficulty_benchmark import (
    BenchmarkConfig,
    run_difficulty_benchmark,
)


def test_benchmark_writes_expected_files(tmp_path):
    config = BenchmarkConfig(
        level_count=2,
        episodes_per_agent=3,
        bootstrap_samples=100,
    )

    run_difficulty_benchmark(
        config=config,
        output_dir=tmp_path,
    )

    assert (tmp_path / "summary.csv").exists()
    assert (tmp_path / "episodes.csv").exists()
    assert (tmp_path / "metadata.json").exists()

    with (tmp_path / "summary.csv").open() as file:
        summary = list(csv.DictReader(file))

    with (tmp_path / "episodes.csv").open() as file:
        episodes = list(csv.DictReader(file))

    metadata = json.loads((tmp_path / "metadata.json").read_text())

    assert len(summary) == 2
    assert len(episodes) == 12
    assert metadata["total_episodes"] == 12


def test_benchmark_is_reproducible(tmp_path):
    config = BenchmarkConfig(
        level_count=1,
        episodes_per_agent=5,
        bootstrap_samples=100,
    )

    first = tmp_path / "first"
    second = tmp_path / "second"

    run_difficulty_benchmark(
        config=config,
        output_dir=first,
    )

    run_difficulty_benchmark(
        config=config,
        output_dir=second,
    )

    for filename in (
        "summary.csv",
        "episodes.csv",
        "metadata.json",
    ):
        assert (first / filename).read_bytes() == (second / filename).read_bytes()


def test_benchmark_records_validation_metadata(tmp_path):
    config = BenchmarkConfig(
        level_count=2,
        max_candidates=5,
        episodes_per_agent=3,
        bootstrap_samples=100,
    )

    run_difficulty_benchmark(
        config=config,
        output_dir=tmp_path,
    )

    metadata = json.loads((tmp_path / "metadata.json").read_text())

    assert metadata["accepted_levels"] == 2
    assert metadata["attempted_candidates"] >= 2
    assert metadata["rejected_candidates"] >= 0

    assert (
        metadata["attempted_candidates"]
        == metadata["accepted_levels"] + metadata["rejected_candidates"]
    )


def test_benchmark_rejects_insufficient_candidate_budget(tmp_path):
    config = BenchmarkConfig(
        level_count=5,
        max_candidates=3,
    )

    with pytest.raises(ValueError, match="max_candidates"):
        run_difficulty_benchmark(
            config=config,
            output_dir=tmp_path,
        )


def test_benchmark_writes_agent_metrics_and_comparisons(tmp_path):
    config = BenchmarkConfig(
        level_count=2,
        max_candidates=5,
        episodes_per_agent=3,
        bootstrap_samples=100,
    )

    run_difficulty_benchmark(
        config=config,
        output_dir=tmp_path,
    )

    with (tmp_path / "agent_metrics.csv").open() as file:
        metrics = list(csv.DictReader(file))

    with (tmp_path / "paired_comparisons.csv").open() as file:
        comparisons = list(csv.DictReader(file))

    assert len(metrics) == 4
    assert len(comparisons) == 2

    assert {row["agent_name"] for row in metrics} == {
        "Random",
        "Greedy",
    }


def test_benchmark_supports_mcts(tmp_path):
    config = BenchmarkConfig(
        level_count=1,
        max_candidates=3,
        episodes_per_agent=2,
        bootstrap_samples=100,
        agents=(
            "Random",
            "Greedy",
            "MCTS_greedy_outcome",
        ),
        mcts_simulations=10,
    )

    run_difficulty_benchmark(
        config=config,
        output_dir=tmp_path,
    )

    with (tmp_path / "agent_metrics.csv").open() as file:
        metrics = list(csv.DictReader(file))

    with (tmp_path / "paired_comparisons.csv").open() as file:
        comparisons = list(csv.DictReader(file))

    with (tmp_path / "episodes.csv").open() as file:
        episodes = list(csv.DictReader(file))

    metadata = json.loads((tmp_path / "metadata.json").read_text())

    assert len(metrics) == 3
    assert len(comparisons) == 3
    assert len(episodes) == 6

    assert {row["agent_name"] for row in metrics} == {
        "Random",
        "Greedy",
        "MCTS_greedy_outcome",
    }

    assert metadata["total_episodes"] == 6
    assert metadata["mcts_configuration"]["simulations"] == 10

    assert all(row["outcome"] in {"WON", "LOST"} for row in episodes)


def test_benchmark_rejects_missing_baselines(tmp_path):
    config = BenchmarkConfig(
        agents=("MCTS_greedy_outcome",),
    )

    with pytest.raises(ValueError, match="Random and Greedy"):
        run_difficulty_benchmark(
            config=config,
            output_dir=tmp_path,
        )


@pytest.mark.parametrize(
    "agents",
    [
        (),
        ("Random", "Random", "Greedy"),
        ("Random", "Greedy", "Unknown"),
    ],
)
def test_benchmark_rejects_invalid_agents(tmp_path, agents):
    config = BenchmarkConfig(agents=agents)

    with pytest.raises(ValueError):
        run_difficulty_benchmark(
            config=config,
            output_dir=tmp_path,
        )
