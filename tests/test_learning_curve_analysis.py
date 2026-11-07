import csv
import json

import pytest

from match3_ai.evaluation.learning_curve_analysis import analyze, load_run, summarize


@pytest.fixture
def run_dirs(tmp_path):
    paths = []
    for index, seed in enumerate((5990, 5991, 5992)):
        path = tmp_path / str(seed)
        (path / "evaluation").mkdir(parents=True)
        summary = {
            "environment_steps": 150000,
            "training_config": {
                "seed": seed,
                "environment_steps": 150000,
                "evaluation_seed_start": 310000,
                "evaluation_episodes": 4,
                "evaluation_every": 25000,
                "evaluate_step_zero": True,
            },
        }
        (path / "training_summary.json").write_text(json.dumps(summary))
        rows = []
        for step in range(0, 150001, 25000):
            wins = (3, 2, 4)[index] if step == 125000 else 2
            for board in range(4):
                won = board < wins if step != 150000 else board in (0, 2)
                rows.append(
                    {
                        "agent_name": "DQNAgent",
                        "episode_id": f"run:{step}:DQNAgent:{board}",
                        "seed": 310000 + board,
                        "outcome": "WON" if won else "LOST",
                        "invalid_actions": 0,
                        "board_seed": 100 + board,
                        "engine_seed": 200 + board,
                        "level_config_id": "fixed",
                    }
                )
        with (path / "evaluation/episodes.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        paths.append(path)
    return paths


def test_checkpoint_statistics_pairing_and_selection(run_dirs):
    runs = [load_run(p, "DQNAgent") for p in run_dirs]
    result = summarize(runs, (100000, 125000, 150000), samples=200)
    row = next(r for r in result["curve"] if r["step"] == 125000)
    assert row["mean"] == 0.75 and row["sample_std"] == 0.25
    assert result["best_individual_checkpoint"]["seed"] == 5992
    assert result["best_cohort_checkpoint"]["step"] == 125000
    assert result["best_per_seed"][1]["step"] == 0
    assert result["final_endpoint"]["mean"] == 0.5
    assert len(result["paired_comparisons"]) == 9
    assert all(
        r["holm_p"] >= r["mcnemar_exact_p"] for r in result["paired_comparisons"]
    )
    comparison = next(
        r
        for r in result["paired_comparisons"]
        if r["seed"] == 5990 and r["earlier"] == 100000 and r["later"] == 150000
    )
    assert comparison["counts"] == {
        "both_win": 1,
        "left_only_win": 1,
        "right_only_win": 1,
        "both_lose": 1,
    }
    assert result["cohort_differences"][0]["n"] == 4


@pytest.mark.parametrize("change", ["missing_identity", "changed_identity"])
def test_unverifiable_boards_skip_paired_inference(run_dirs, change):
    runs = [load_run(p, "DQNAgent") for p in run_dirs]
    row = runs[0]["groups"][125000][310000]
    row["board_seed"] = "" if change == "missing_identity" else "different"
    result = summarize(runs, (100000, 125000, 150000), samples=20)
    assert not result["paired_identifiers_verified"]
    assert result["paired_comparisons"] == []


def test_duplicate_board_is_rejected(run_dirs):
    path = run_dirs[0] / "evaluation/episodes.csv"
    lines = path.read_text().splitlines(True)
    path.write_text("".join([*lines, lines[1]]))
    with pytest.raises(ValueError, match="duplicate"):
        load_run(run_dirs[0], "DQNAgent")


def test_missing_checkpoint_and_duplicate_training_seed_are_rejected(run_dirs):
    runs = [load_run(p, "DQNAgent") for p in run_dirs]
    with pytest.raises(ValueError, match="distinct training seeds"):
        summarize([runs[0], runs[0]], (100000, 125000, 150000))
    runs[0]["groups"].pop(125000)
    with pytest.raises(ValueError, match="share checkpoints"):
        summarize(runs, (100000, 125000, 150000))


def test_outputs_are_exclusive_and_inputs_unchanged(run_dirs, tmp_path):
    before = {
        p: p.read_bytes() for run in run_dirs for p in run.rglob("*") if p.is_file()
    }
    output = tmp_path / "analysis"
    analyze(run_dirs, output)
    assert {p.name for p in output.iterdir()} == {
        "analysis.json",
        "report.md",
        "learning_curves.png",
        "learning_curves.pdf",
        "checkpoint_comparison.png",
        "checkpoint_comparison.pdf",
    }
    assert "not independent test estimates" in (output / "report.md").read_text()
    assert all(p.read_bytes() == content for p, content in before.items())
    with pytest.raises(FileExistsError):
        analyze(run_dirs, output)
