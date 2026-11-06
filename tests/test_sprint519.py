import csv
import json

import numpy as np
import pytest

pytest.importorskip("torch")

from match3_ai.training.dqn import TrainingConfig
from match3_ai.training.overnight_analysis import paired_statistics
from match3_ai.training.ppo import train_ppo
from match3_ai.training.seed_policy import (
    PARTITIONS,
    seed_policy_metadata,
    validate_evaluation_partition,
    validate_partition,
)
from match3_ai.training.sprint519 import StudyPPOConfig, configurations
from match3_ai.training.sprint519_analysis import group_records


def test_fresh_partition_is_explicit_and_legacy_contract_is_preserved():
    assert PARTITIONS["validation"] == (300000, 300999)
    validate_evaluation_partition("sprint5_19_validation", 310000, 200)
    with pytest.raises(ValueError):
        validate_partition("validation", 310000, 200)
    with pytest.raises(ValueError):
        validate_evaluation_partition("sprint5_19_validation", 310000, 201)
    with pytest.raises(ValueError):
        validate_evaluation_partition("sprint5_19_validation", 400000, 1)
    assert seed_policy_metadata()["evaluation_split"] == "validation"
    assert (
        seed_policy_metadata("sprint5_19_validation")["evaluation_split"]
        == "sprint5_19_validation"
    )


def test_frozen_profiles_keep_interventions_isolated_and_epsilon_unchanged():
    runs = configurations()
    assert (
        len(runs) == 9 and sum(r["config"]["environment_steps"] for r in runs) == 900000
    )
    for seed in (5970, 5971, 5972):
        pair = [
            r["config"]
            for r in runs
            if r["kind"] == "ppo" and r["config"]["seed"] == seed
        ]
        assert {k for k in pair[0] if pair[0][k] != pair[1][k]} == {
            "learning_rate_schedule"
        }
        assert pair[0]["checkpoint_steps"] == [0, 50000, 75000, 100000]
    for r in runs:
        c = dict(r["config"])
        c["hidden_dims"] = tuple(c["hidden_dims"])
        if r["kind"] == "dqn":
            config = TrainingConfig(**c)
            assert config.epsilon(50000) == pytest.approx(0.05)
            assert config.epsilon(100000) == pytest.approx(0.05)
            assert (
                config.replay_capacity == 50000
                and config.algorithm == "double"
                and config.architecture == "dueling"
            )
        else:
            c["checkpoint_steps"] = tuple(c["checkpoint_steps"])
            StudyPPOConfig(**c)


def test_explicit_ppo_cadence_omits_unrequested_milestone(tmp_path):
    cfg = StudyPPOConfig(
        level_id="first-match",
        environment_steps=12,
        seed=12,
        evaluation_episodes=1,
        evaluation_seed_start=300000,
        evaluation_every=3,
        rollout_steps=8,
        minibatch_size=4,
        hidden_dims=(8,),
        checkpoint_steps=(0, 8, 12),
    )
    result = train_ppo(cfg, tmp_path / "run", evaluation_steps=cfg.checkpoint_steps)
    assert [r["environment_step"] for r in result["evaluation_rounds"]] == [0, 8, 12]
    assert (tmp_path / "run/policy_step_8.pt").is_file()
    assert not (tmp_path / "run/policy_step_3.pt").exists()
    with pytest.raises(ValueError, match="evaluation steps"):
        train_ppo(cfg, tmp_path / "bad", evaluation_steps=(0, 6, 6, 12))


def test_group_coverage_cannot_silently_exclude_a_board():
    rows = [
        {
            "agent_name": "PPOAgent",
            "episode_id": f"job:100000:PPOAgent:{i}",
            "seed": str(i),
            "outcome": "WON",
            "invalid_actions": "0",
            "error_type": "",
        }
        for i in (0, 1)
    ]
    assert set(group_records(rows, [100000], [0, 1], "PPOAgent")[100000]) == {0, 1}
    with pytest.raises(ValueError, match="coverage"):
        group_records(rows[:1], [100000], [0, 1], "PPOAgent")
    with pytest.raises(ValueError, match="coverage"):
        group_records(rows + [rows[0]], [100000], [0, 1], "PPOAgent")


def test_shared_board_contrast_and_training_uncertainty_do_not_pool_seeds():
    a = np.array([[1, 0, 1, 0]] * 3)
    b = np.array([[0, 0, 1, 0]] * 3)
    stat = paired_statistics(a, b, samples=200, seed=519519)
    assert stat["difference"] == 0.25 and stat["n_boards"] == 4
    assert stat["n_training_pairs"] == 3 and stat["training_pair_sample_std"] == 0
    assert stat["training_pair_signflip_sensitivity_p"] == 0.25
    draws = np.random.default_rng(519519).integers(4, size=(200, 4))
    means = np.array([1, 0, 0, 0])[draws].mean(-1)
    np.testing.assert_allclose(
        stat["conditional_board_ci95"], np.quantile(means, [0.025, 0.975])
    )


def test_actual_smoke_evidence_contains_complete_export_and_status_when_available():
    from pathlib import Path

    root = Path("experiments/results/sprint5_19_rl_optimization")
    evidence = root / "smoke_evidence.json"
    if not evidence.exists():
        pytest.skip("production smoke artifacts are not available")
    path = Path(json.loads(evidence.read_text())["summary_path"])
    results = json.loads(path.read_text())
    assert results["ppo_e2e"]["optimizer_updates"] == 32
    assert results["dueling_e2e"]["optimizer_updates"] == 113
    for name in results:
        record = path.parent / "records" / name
        assert json.loads((record / "status.json").read_text())["status"] == "completed"
        p = path.parent / "runs" / name / "evaluation/experiment_metadata.json"
        metadata = json.loads(p.read_text())
        assert metadata["experiment_id"] == results[name]["experiment_id"]
        with p.with_name("episodes.csv").open() as f:
            assert all(
                r["experiment_id"] == metadata["experiment_id"]
                for r in csv.DictReader(f)
            )


def test_supervisor_halts_remaining_queue_after_child_failure(tmp_path, monkeypatch):
    from datetime import UTC, datetime, timedelta

    from match3_ai.training import sprint519

    now = datetime.now(UTC)
    protocol = {
        "training_cutoff_utc": (now + timedelta(hours=1)).isoformat(),
        "deadline_utc": (now + timedelta(hours=2)).isoformat(),
        "runs": [{"id": f"job{i}", "estimated_seconds": 1} for i in range(6)],
    }
    launched = []

    class FailedProcess:
        def __init__(self, command, **kwargs):
            self.pid = 100 + len(launched)
            self.returncode = 1
            launched.append(command)

        def poll(self):
            return 1

        def wait(self, timeout=None):
            return 1

    monkeypatch.setattr(sprint519, "verify", lambda _: protocol)
    monkeypatch.setattr(sprint519.subprocess, "Popen", FailedProcess)
    monkeypatch.setattr(sprint519.time, "sleep", lambda _: None)
    sprint519.supervise(tmp_path)
    workers = [command for command in launched if "worker" in command]
    assert len(workers) == 3
    result = json.loads((tmp_path / "execution_complete.json").read_text())
    assert result["safety_stop"]
    assert all(
        result["statuses"][f"job{i}"] == "not_started_safety_stop" for i in range(3, 6)
    )
    assert (
        json.loads((tmp_path / "supervisor_complete.json").read_text())[
            "active_children"
        ]
        == []
    )


def test_report_and_all_figures_export_from_synthetic_records(tmp_path):
    from match3_ai.training.sprint519_analysis import figures, label, report_text

    models = {}
    for spec in configurations():
        steps = (
            [0, 50000, 75000, 100000]
            if spec["kind"] == "ppo"
            else [0, 25000, 50000, 75000, 100000]
        )
        rows = [
            {
                "environment_step": str(step),
                **dict.fromkeys(
                    (
                        "approximate_kl",
                        "clip_fraction",
                        "entropy",
                        "value_loss",
                        "gradient_norm",
                        "td_loss",
                        "q_mean",
                        "epsilon",
                        "replay_size",
                    ),
                    ".1",
                ),
                "learning_rate": ".0003",
            }
            for step in (1000, 100000)
        ]
        models[spec["id"]] = {
            "label": label(spec),
            "curve": [
                {
                    "step": step,
                    "wins": 100,
                    "win_rate": 0.5,
                    "blocker_hp_remaining": 1.0,
                }
                for step in steps
            ],
            "plot_updates": rows,
            "late_stability": dict.fromkeys(
                ("signed_75k_to_100k", "negative_part_regression", "absolute_change"),
                0.0,
            ),
            "training_seconds": 2.0,
            "evaluation_seconds": 1.0,
            "process_seconds_at_50k": 1.0,
        }
    paths = figures(models, tmp_path / "figures")
    assert len(paths) == 12
    assert all((tmp_path / "figures" / name).stat().st_size > 100 for name in paths)
    stat = paired_statistics(np.ones((3, 4)), np.ones((3, 4)), samples=10)
    stat.update(status="estimated", holm_p_primary_two=1.0)
    aggregates = {
        name: {
            "n_completed_models": 3,
            "mean_final_win_rate": 0.5,
            "training_seed_sample_std": 0.0,
            "mean_process_seconds": 2.0,
            "mean_objective_completion": 0.9,
            "mean_remaining_hp": 1.0,
            "late_stability": {
                k: {"mean": 0.0}
                for k in (
                    "signed_75k_to_100k",
                    "negative_part_regression",
                    "absolute_change",
                )
            },
        }
        for name in ("PPO constant", "PPO LR decay", "Dueling DDQN")
    }
    result = {
        "manifest_sha256": "fixture",
        "finalized_utc": "2026-01-01",
        "within_deadline": True,
        "aggregates": aggregates,
        "comparisons": {"A": stat, "B": stat},
        "models": models,
        "highest_fixed_endpoint": None,
        "highest_scheduled_checkpoint": None,
        "inventory": [],
        "historical_artifacts_verified": 0,
        "preexisting_file_integrity": {"only_intended_changes": []},
        "validation": {},
    }
    text = report_text(
        tmp_path,
        {
            "actual_start_utc": "2026-01-01T00:00:00+00:00",
            "deadline_utc": "2026-01-01T08:00:00+00:00",
        },
        result,
    )
    assert "Holm" in text and "Istanbul" in text and "conditional" in text
