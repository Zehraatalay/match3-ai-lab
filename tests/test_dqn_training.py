import csv
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")

from match3_ai.training.dqn import TrainingConfig, train_dqn


def tiny_config():
    return TrainingConfig(
        level_id="first-match",
        environment_steps=12,
        seed=57,
        replay_capacity=32,
        batch_size=2,
        warmup_steps=2,
        train_every=1,
        target_sync_updates=2,
        evaluation_every=6,
        evaluation_episodes=1,
        epsilon_decay_steps=12,
        hidden_dims=(8,),
    )


def test_bounded_training_updates_evaluates_checkpoints_and_resumes(tmp_path):
    config = tiny_config()
    result = train_dqn(config, tmp_path / "first")
    assert result["environment_steps"] == result["replay_size"] == 12
    assert result["parameters_changed"] and result["finite_losses"]
    assert result["optimizer_updates"] == 11
    assert result["target_synchronizations"] == 5
    assert result["invalid_actions"] == 0
    assert result["checkpoint_inference_valid"]
    assert len(result["evaluation_rounds"]) == 2
    assert (tmp_path / "first/evaluation/moves.csv").exists()
    with (tmp_path / "first/training_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert all(np.isfinite(float(row["td_loss"])) for row in rows if row["td_loss"])
    repeated = train_dqn(config, tmp_path / "repeat")
    assert result["final_weights"] == repeated["final_weights"]
    assert result["mean_td_loss"] == repeated["mean_td_loss"]
    resumed = train_dqn(
        replace(config, environment_steps=16),
        tmp_path / "resumed",
        resume_from=result["checkpoint"],
    )
    assert resumed["environment_steps"] == resumed["replay_size"] == 16
    assert resumed["optimizer_updates"] == 15
    assert resumed["training_start_step"] == 12
    assert resumed["parameters_changed"]
    with pytest.raises(FileExistsError):
        train_dqn(config, tmp_path / "first")


def test_config_rejects_seed_overlap_and_bad_bounds():
    with pytest.raises(ValueError, match="disjoint"):
        TrainingConfig(train_seed_start=110000)
    with pytest.raises(ValueError):
        TrainingConfig(batch_size=10, replay_capacity=2)
    config = tiny_config()
    assert config.epsilon(0) == 1
    assert config.epsilon(12) == pytest.approx(0.05)
