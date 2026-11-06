import csv
import json
from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from match3_ai.evaluation.analysis import (
    compare_agents,
    exact_mcnemar,
    load_dataset,
    numeric_summary,
    paired_bootstrap,
    run_analysis,
    validate_dataset,
    wilson_interval,
)
from match3_ai.evaluation.analytics import MoveRecord
from match3_ai.evaluation.runner import EpisodeResult


def write_csv(path, records, record_type):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(record_type.__dataclass_fields__)
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    key: json.dumps(value)
                    if isinstance(value, (dict, list, tuple))
                    else value
                    for key, value in asdict(record).items()
                }
            )


@pytest.fixture
def benchmark(tmp_path):
    root = tmp_path / "input"
    root.mkdir()
    episodes, moves = [], []
    outcomes = {
        "RandomAgent": ["WON", "WON", "LOST", "ERROR", "LOST"],
        "GreedyAgent": ["WON", "LOST", "WON", "ERROR", "LOST"],
    }
    for agent, agent_outcomes in outcomes.items():
        for index, outcome in enumerate(agent_outcomes):
            failed = outcome == "ERROR"
            progress = (
                None
                if failed
                else [
                    {
                        "type": "COLLECT_TILES",
                        "color": None,
                        "progress": int(outcome == "WON"),
                        "target": 1,
                        "completed": outcome == "WON",
                    }
                ]
            )
            episode = EpisodeResult(
                seed=10 + index,
                total_reward=0 if failed else 1 if outcome == "WON" else -1,
                steps=0 if failed else 1,
                valid_actions=0 if failed else 1,
                invalid_actions=0,
                total_cleared=0 if failed else 3,
                total_cascades=0 if failed else 2,
                reshuffles=0,
                completed=not failed,
                experiment_id="synthetic",
                episode_id=f"{agent}-{index}",
                agent_name=agent,
                level_id="fixture",
                level_config_id=None if failed else "fixturehash",
                agent_seed=20 + index,
                effective_episode_seed=None if failed else 10 + index,
                board_seed=None if failed else 100 + index,
                engine_seed=None if failed else 200 + index,
                outcome=outcome,
                termination_reason="exception"
                if failed
                else "objectives_complete"
                if outcome == "WON"
                else "move_limit",
                moves_used=None if failed else 1,
                objective_progress=progress,
                objective_completion_ratio=None if failed else int(outcome == "WON"),
                blockers_remaining=None if failed else int(outcome == "LOST"),
                blocker_hp_remaining=None if failed else int(outcome == "LOST"),
                special_activations=0 if failed else 1,
                max_cascades_in_move=0 if failed else 2,
                cascade_limit=None if failed else 200,
                inference_time_seconds=0 if failed else 0.001,
                error_type="LevelGenerationError" if failed else None,
                error_message="initialization failed" if failed else None,
                error_context={"episode_seed": 10 + index, "action_id": None}
                if failed
                else None,
            )
            episodes.append(episode)
            if not failed:
                moves.append(
                    MoveRecord(
                        experiment_id="synthetic",
                        episode_id=episode.episode_id,
                        move_index=1,
                        action_id=0,
                        valid_action=True,
                        reward=episode.total_reward,
                        moves_remaining=0,
                        cleared_tiles=3,
                        cleared_by_color=(3, 0, 0),
                        cascade_count=2,
                        blocker_damage={"hits": 0, "destroyed": 0, "by_type": {}},
                        special_activations={"count": 1, "by_type": {"BOMB": 1}},
                        objective_progress=progress,
                        legal_action_count=2,
                        resolution_completed=True,
                    )
                )
    write_csv(root / "episodes.csv", episodes, EpisodeResult)
    write_csv(root / "moves.csv", moves, MoveRecord)
    metadata = {
        "experiment_id": "synthetic",
        "attempted_episodes": 10,
        "planned_episodes": 10,
        "agent_names": list(outcomes),
        "levels": {"fixture": {"level_config_id": "fixturehash"}},
        "config": {
            "level_ids": ["fixture"],
            "num_episodes": 5,
            "environment_seeds": list(range(10, 15)),
            "agent_seeds": list(range(20, 25)),
        },
        "interruption_reason": None,
    }
    (root / "experiment_metadata.json").write_text(json.dumps(metadata))
    return root


def test_loading_types_missing_values_and_input_hashes(benchmark):
    data = load_dataset(benchmark)
    assert len(data.episodes) == 10
    assert len(data.moves) == 8
    assert type(data.episodes[0]["seed"]) is int
    assert type(data.episodes[0]["total_reward"]) is float
    assert data.episodes[0]["completed"] is True
    assert isinstance(data.episodes[0]["objective_progress"], list)
    assert data.episodes[3]["moves_used"] is None
    assert len(data.hashes) == 3
    quality = validate_dataset(data)
    assert quality["missing"]["initialization_error"]["episodes.moves_used"] == 2
    assert quality["missing"]["structural"]["moves.error_type"] == 8
    assert not quality["missing"]["unexpected"]


@pytest.mark.parametrize(
    "damage",
    [
        "duplicate_episode",
        "duplicate_move",
        "orphan",
        "gap",
        "outcome",
        "missing",
        "metadata",
        "agent_seed",
        "paired_seed",
        "reward",
    ],
)
def test_integrity_checks_reject_invalid_data(benchmark, damage):
    data = deepcopy(load_dataset(benchmark))
    if damage == "duplicate_episode":
        data.episodes[1]["episode_id"] = data.episodes[0]["episode_id"]
    elif damage == "duplicate_move":
        data.moves.append(deepcopy(data.moves[0]))
    elif damage == "orphan":
        data.moves[0]["episode_id"] = "unknown"
    elif damage == "gap":
        data.moves[0]["move_index"] = 2
    elif damage == "outcome":
        data.episodes[0]["outcome"] = "UNKNOWN"
    elif damage == "missing":
        data.episodes[0]["moves_used"] = None
    elif damage == "metadata":
        data.metadata["attempted_episodes"] = 11
    elif damage == "agent_seed":
        data.episodes[0]["agent_seed"] = 999
    elif damage == "paired_seed":
        data.episodes[5]["board_seed"] = 999
    elif damage == "reward":
        data.moves[0]["reward"] = 100
    with pytest.raises(ValueError):
        validate_dataset(data)


def test_pairs_align_by_seed_not_csv_order_and_exclude_only_invalid_pairs(benchmark):
    data = load_dataset(benchmark)
    data.episodes.reverse()
    result = compare_agents(data, samples=1000)[0]
    assert result["valid_pairs"] == 4
    assert result["excluded_pairs"] == {"error_or_time_limit": 1}
    assert [
        result[key] for key in ("both_win", "random_only", "greedy_only", "both_lose")
    ] == [1, 1, 1, 1]
    assert result["greedy_minus_random"] == 0
    assert result["p_value"] == 1
    next(
        row
        for row in data.episodes
        if row["seed"] == 10 and row["agent_name"] == "RandomAgent"
    )["outcome"] = "TIME_LIMIT"
    filtered = compare_agents(data, samples=1000)[0]
    assert filtered["valid_pairs"] == 3
    assert filtered["excluded_pairs"]["error_or_time_limit"] == 2
    assert len(data.episodes) == 10


@pytest.mark.parametrize(
    ("b", "c", "expected"), [(0, 0, 1), (0, 6, 0.03125), (1, 9, 0.021484375), (3, 3, 1)]
)
def test_exact_mcnemar_known_binomial_probabilities(b, c, expected):
    assert exact_mcnemar(b, c) == pytest.approx(expected)
    assert exact_mcnemar(c, b) == pytest.approx(expected)


def test_wilson_and_numeric_summaries_handle_missingness_explicitly():
    low, high = wilson_interval(0, 100)
    assert low == pytest.approx(0, abs=1e-12)
    assert high == pytest.approx(0.0369934982)
    assert wilson_interval(0, 0) is None
    summary = numeric_summary([1, None, 3])
    assert summary["n"] == 2 and summary["missing"] == 1
    assert summary["mean"] == summary["median"] == 2
    assert summary["std"] == pytest.approx(np.sqrt(2))
    assert numeric_summary([None])["mean"] is None


def test_paired_bootstrap_is_reproducible_and_keeps_pair_differences():
    differences = [-1] * 10 + [0] * 70 + [1] * 20
    interval = paired_bootstrap(differences, seed=55)
    assert interval == paired_bootstrap(differences, seed=55)
    assert interval[0] <= 0.1 <= interval[1]
    assert paired_bootstrap([0] * 100, seed=55) == [0, 0]
    assert paired_bootstrap([], seed=55) is None
    with pytest.raises(ValueError):
        paired_bootstrap(differences, seed=55, samples=0)


def test_report_figures_and_output_paths_preserve_benchmark(benchmark, tmp_path):
    pytest.importorskip("matplotlib")
    before = load_dataset(benchmark).hashes
    output = tmp_path / "analysis"
    result = run_analysis(
        benchmark, output, bootstrap_samples=1000, initialization_investigation=False
    )
    assert (output / "report.md").exists()
    text = (output / "report.md").read_text()
    assert "Exact two-sided McNemar" in text
    assert "not evidence of equivalence" in text
    assert len(result["figures"]) == 8
    for name in result["figures"]:
        assert (output / "figures" / f"{name}.png").stat().st_size > 1000
        assert (output / "figures" / f"{name}.pdf").stat().st_size > 1000
    assert json.loads((output / "analysis.json").read_text())["input_hashes"] == before
    assert load_dataset(benchmark).hashes == before
    with pytest.raises(FileExistsError):
        run_analysis(benchmark, output)
    with pytest.raises(ValueError, match="separate"):
        run_analysis(benchmark, benchmark)
