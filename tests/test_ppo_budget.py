import hashlib
import json
from dataclasses import asdict

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from match3_ai.agents.ppo_agent import PPOAgent
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.training.ppo import PPOTrainingConfig
from match3_ai.training.ppo_budget import (
    MASTER_SEEDS,
    MILESTONES,
    experiment_config,
    freeze_protocol,
    load_protocol,
    optimized_progress,
)
from match3_ai.training.ppo_budget_analysis import (
    create_figures,
    joint_budget_statistics,
    write_report,
)


def test_approved_profiles_and_collected_optimized_accounting():
    base = asdict(PPOTrainingConfig())
    for seed in MASTER_SEEDS:
        config = asdict(experiment_config(seed))
        assert {key for key in base if base[key] != config[key]} == {
            "seed",
            "environment_steps",
            "evaluation_every",
            "evaluation_seed_start",
            "evaluation_episodes",
        }
        assert (
            config["environment_steps"] == 100000
            and config["evaluation_episodes"] == 200
            and config["evaluation_seed_start"] == 300600
        )
    assert [optimized_progress(s)["optimized"] for s in MILESTONES] == [
        0,
        24832,
        49920,
        74752,
        100000,
    ]
    assert [optimized_progress(s)["pending"] for s in MILESTONES] == [
        0,
        168,
        80,
        248,
        0,
    ]
    assert [optimized_progress(s)["optimizer_updates"] for s in MILESTONES] == [
        0,
        1552,
        3120,
        4672,
        6252,
    ]
    assert optimized_progress(100000)["rollouts"] == 391
    assert 390 * 256 + 160 == 100000 and 390 * 16 + 4 * 3 == 6252
    with pytest.raises(ValueError):
        experiment_config(5917)


def test_joint_bootstrap_matches_explicit_common_cube_indices_and_holm_family():
    cube = np.random.default_rng(4).integers(0, 2, size=(3, 5, 8))
    samples = 200
    result = joint_budget_statistics(
        cube, samples=samples, seed=7, metrics={"objective_completion": cube * 0.5}
    )
    indices = np.random.default_rng(7).integers(8, size=(samples, 8))
    explicit = cube[:, :, indices].mean(-1)
    difference = (explicit[:, 4] - explicit[:, 2]).mean(0)
    np.testing.assert_allclose(
        result["comparisons"][1]["ci95"], np.quantile(difference, [0.025, 0.975])
    )
    assert (
        result["joint_resampled_indices_sha256"]
        == hashlib.sha256(indices.astype(np.int64).tobytes()).hexdigest()
    )
    assert len(result["comparisons"]) == 8 and all(
        r["holm_p"] >= r["p_value"] for r in result["comparisons"]
    )
    np.testing.assert_allclose(
        result["metrics"]["objective_completion"]["conditional_mean_ci95"],
        np.array(result["metrics"]["win_rate"]["conditional_mean_ci95"]) * 0.5,
    )


def test_identical_models_do_not_artificially_multiply_board_sample_size():
    base = np.array(
        [[0, 1, 0, 1], [0, 1, 0, 1], [1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 1, 0]]
    )
    cube = np.stack([base] * 3)
    result = joint_budget_statistics(cube, samples=200, seed=3)
    cohort = result["comparisons"][0]
    individual = result["comparisons"][2]
    assert cohort["n_boards"] == individual["n_boards"] == 4
    np.testing.assert_allclose(cohort["ci95"], individual["ci95"])
    with pytest.raises(ValueError):
        joint_budget_statistics(cube[:2])
    with pytest.raises(ValueError):
        joint_budget_statistics(cube, metrics={"bad": np.full(cube.shape, np.nan)})


def test_protocol_provenance_seed_conflicts_and_immutability(tmp_path):
    history = tmp_path / "history"
    history.mkdir()
    prior = history / "prior.json"
    prior.write_text(json.dumps({"seed": 300600}))
    with pytest.raises(ValueError, match="historical overlap"):
        freeze_protocol(history, tmp_path / "conflict")
    assert not (tmp_path / "conflict").exists()
    prior.write_text(
        json.dumps(
            {
                "evaluation_seed_start": 300000,
                "evaluation_episodes": 100,
                "seeds": list(range(300100, 300600)),
            }
        )
    )
    result = freeze_protocol(history, tmp_path / "protocol")
    path = tmp_path / "protocol/protocol.json"
    assert load_protocol(path) == result
    assert result["validation_seeds"] == list(range(300600, 300800))
    assert (
        len(result["hypotheses"]) == 8
        and len({s["initial_weights"] for s in result["streams"].values()}) == 3
    )
    with pytest.raises(FileExistsError):
        freeze_protocol(history, tmp_path / "protocol")
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="changed"):
        load_protocol(path)


def test_checkpoint_budget_metadata_and_deterministic_inference(tmp_path):
    agent = PPOAgent(7, hidden_dims=(8,))
    progress = optimized_progress(50000)
    agent.save_checkpoint(
        tmp_path / "policy.pt",
        training_progress=progress,
        validation_metrics={"wins": 0},
    )
    loaded, payload = PPOAgent.load_checkpoint(tmp_path / "policy.pt")
    assert (
        payload["training_progress"] == progress
        and payload["validation_metrics"]["wins"] == 0
    )
    observation, _ = Match3LevelEnv("first-match").reset(seed=300000)
    state = loaded.policy_rng.get_state().clone()
    assert (
        loaded.select_action(observation)
        == agent.select_action(observation)
        == loaded.select_action(observation)
    )
    assert torch.equal(state, loaded.policy_rng.get_state())


def test_budget_report_and_all_required_figures_export_without_training(tmp_path):
    cube = np.random.default_rng(5).integers(0, 2, size=(3, 5, 8))
    statistics = joint_budget_statistics(
        cube,
        samples=200,
        seed=6,
        metrics={"objective_completion": cube * 0.5, "remaining_blocker_hp": 1 - cube},
    )
    runs = {}
    for index, seed in enumerate(MASTER_SEEDS):
        rounds = [
            {
                "environment_step": step,
                "win_rate": float(cube[index, j].mean()),
                "wins": int(cube[index, j].sum()),
                "win_rate_ci95": [0, 1],
                "mean_objective_completion_ratio": 0.5,
                "mean_blocker_hp_remaining": 1,
                "mean_moves_used": 15,
                "mean_total_reward": 0,
            }
            for j, step in enumerate(MILESTONES)
        ]
        updates = [
            {
                "environment_step": step,
                "policy_loss": 0.1,
                "value_loss": 0.2,
                "entropy": 1,
                "approximate_kl": 0.01,
                "clip_fraction": 0.1,
                "gradient_norm": 0.5,
            }
            for step in (256, 512)
        ]
        runs[seed] = {
            "summary": {
                "evaluation_rounds": rounds,
                "rollouts": 391,
                "optimizer_updates": 6252,
                "collection_optimization_seconds": 15,
                "evaluation_seconds": 5,
                "training_seconds": 20,
                "optimization_statistics": {
                    "value_loss": {"min": 0.1, "max": 0.2, "mean": 0.15}
                },
            },
            "checkpoint_checks": [optimized_progress(s) for s in MILESTONES],
            "rollouts": [
                {
                    "environment_step": 256,
                    "explained_variance": ".1",
                    "completed_episodes": "1",
                    "training_win_rate": ".5",
                    "mean_training_episode_reward": "0",
                }
            ],
            "updates": updates,
        }
    figures = create_figures(runs, statistics, tmp_path / "figures")
    assert len(figures) == 26 and all(
        (tmp_path / "figures" / name).is_file() for name in figures
    )
    result = {
        "protocol_sha256": "fixture",
        "statistics": statistics,
        "historical_files_preserved": 1,
        "figures": figures,
    }
    write_report(result, runs, tmp_path / "report.md")
    text = (tmp_path / "report.md").read_text()
    assert "Collected" in text and "Optimized" in text and "conditional" in text
