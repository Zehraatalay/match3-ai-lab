from dataclasses import asdict, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from match3_ai.agents.ppo_agent import PPOAgent
from match3_ai.training.ppo import PPOTrainingConfig, train_ppo
from match3_ai.training.ppo_multi_seed import MASTER_SEEDS, experiment_config


def test_profiles_preserve_ppo_hyperparameters_and_share_validation_seeds():
    original = asdict(PPOTrainingConfig())
    for seed in MASTER_SEEDS:
        config = asdict(experiment_config(seed))
        assert {key for key in config if config[key] != original[key]} == {
            "seed",
            "environment_steps",
            "evaluation_every",
        }
        assert (
            config["environment_steps"] == 50000 and config["evaluation_every"] == 5000
        )
        assert (
            config["evaluation_seed_start"] == 300000
            and config["evaluation_episodes"] == 100
        )
    with pytest.raises(ValueError):
        experiment_config(5913)


def test_independent_model_policy_shuffle_and_environment_streams():
    derived, initial_weights, environment_draws = [], [], []
    for seed in MASTER_SEEDS:
        env_stream, agent_stream = np.random.SeedSequence(seed).spawn(2)
        agent_seed = int(agent_stream.generate_state(1)[0])
        agent = PPOAgent(agent_seed, hidden_dims=(8,))
        derived.extend(
            [int(env_stream.generate_state(1)[0]), *agent.stream_seeds.values()]
        )
        initial_weights.append(agent.network.actor.weight.detach().numpy().tobytes())
        environment_draws.append(
            np.random.default_rng(env_stream).integers(200000, 300000, size=16)
        )
    assert len(set(derived)) == 12 and len(set(initial_weights)) == 3
    assert all(
        not np.array_equal(environment_draws[0], draws)
        for draws in environment_draws[1:]
    )
    assert all(
        ((draws >= 200000) & (draws <= 299999)).all() for draws in environment_draws
    )


def test_run_isolation_failure_preservation_and_checkpoint_metrics(
    tmp_path, monkeypatch
):
    import match3_ai.training.ppo as module

    config = replace(
        experiment_config(5915),
        level_id="first-match",
        environment_steps=2,
        rollout_steps=2,
        optimization_epochs=1,
        minibatch_size=2,
        hidden_dims=(8,),
        evaluation_episodes=1,
    )
    first = train_ppo(config, tmp_path / "completed")
    checkpoint = tmp_path / "completed/checkpoint_final.pt"
    before = checkpoint.read_bytes()
    with pytest.raises(FileExistsError):
        train_ppo(replace(config, seed=5916), tmp_path / "completed")
    assert checkpoint.read_bytes() == before
    loaded, payload = PPOAgent.load_checkpoint(checkpoint)
    assert payload["training_config"]["seed"] == 5915
    assert (
        payload["validation_metrics"]["win_rate"]
        == first["evaluation_rounds"][-1]["win_rate"]
    )
    assert payload["training_progress"]["environment_step"] == 2
    assert (
        payload["training_progress"]["training_seconds"]
        >= payload["training_progress"]["evaluation_seconds"]
    )
    assert loaded.seed == payload["seed"]

    def fail_evaluation(*args, **kwargs):
        raise RuntimeError("injected independent run failure")

    monkeypatch.setattr(module, "evaluate_ppo", fail_evaluation)
    with pytest.raises(RuntimeError, match="injected"):
        train_ppo(replace(config, seed=5916), tmp_path / "failed")
    assert (tmp_path / "failed/training_error.json").is_file()
    assert checkpoint.read_bytes() == before


def test_shared_board_interval_does_not_treat_models_as_independent_episodes():
    from match3_ai.evaluation.behavior_statistics import paired_difference
    from match3_ai.training.ppo_multi_seed_analysis import (
        learning_statistics,
        shared_board_difference,
    )

    policies = np.array([[1, 0, 1, 0], [1, 0, 1, 0], [1, 0, 1, 0]])
    reference = np.zeros(4)
    actual = shared_board_difference(policies, reference)
    expected = paired_difference(policies[0], reference, seed=51515)
    assert actual == expected and actual["n"] == 4
    stats = learning_statistics(
        [[{"environment_step": 0, "win_rate": value}] for value in (0.3, 0.5, 0.7)]
    )
    assert stats["mean"] == pytest.approx([0.5])
    assert stats["sample_std"] == pytest.approx([0.2])
    with pytest.raises(ValueError):
        shared_board_difference(policies, [0, 1])
    with pytest.raises(ValueError):
        learning_statistics(
            [
                [{"environment_step": 0, "win_rate": 0.3}],
                [{"environment_step": 1, "win_rate": 0.7}],
            ]
        )


def test_three_run_aggregation_reuses_audits_preserves_sources_and_exports_figures(
    tmp_path,
):
    import hashlib

    from match3_ai.agents.dqn_agent import DQNAgent
    from match3_ai.training.ppo_analysis import analyze_pilot
    from match3_ai.training.ppo_multi_seed_analysis import aggregate_runs

    reference = tmp_path / "dqn.pt"
    DQNAgent(5903, hidden_dims=(8,)).save_checkpoint(
        reference,
        training_config={"level_id": "first-match", "seed": 5903},
        training_step=0,
    )
    original_hash = hashlib.sha256(reference.read_bytes()).hexdigest()
    runs, analyses = [], []
    for seed in MASTER_SEEDS:
        run, analysis = tmp_path / f"run_{seed}", tmp_path / f"analysis_{seed}"
        config = replace(
            experiment_config(seed),
            level_id="first-match",
            environment_steps=6,
            rollout_steps=4,
            optimization_epochs=1,
            minibatch_size=4,
            hidden_dims=(8,),
            evaluation_episodes=1,
            evaluation_every=3,
        )
        train_ppo(config, run)
        analyze_pilot(run, analysis, [reference])
        assert f"master seed {seed}" in (analysis / "report.md").read_text()
        runs.append(run)
        analyses.append(analysis)
    result = aggregate_runs(runs, analyses, tmp_path / "aggregate")
    assert len(result["figures"]) == 24
    assert result["seed_streams_unique"] and not result["final_test_used"]
    assert len(result["per_seed"]) == 3 and len(result["paired_tests"]) == 12
    assert result["conditional_shared_board_differences"]["RandomAgent"]["n"] == 1
    assert all(
        row["holm_adjusted_p"] >= row["mcnemar_exact_p"]
        for row in result["paired_tests"]
    )
    assert original_hash == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert (tmp_path / "aggregate/report.md").is_file()
    with pytest.raises(FileExistsError):
        aggregate_runs(runs, analyses, tmp_path / "aggregate")
