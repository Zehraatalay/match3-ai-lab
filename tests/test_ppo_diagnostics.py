import csv
from dataclasses import asdict, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from match3_ai.agents.ppo_agent import PPOAgent, masked_distribution
from match3_ai.training.ppo import PPOTrainingConfig, train_ppo
from match3_ai.training.ppo_pilot import pilot_config
from match3_ai.training.ppo_update import (
    clipped_objective,
    explained_variance,
    generalized_advantages,
    rollout_diagnostics,
)


def test_ratio_uses_masked_probabilities_and_diagnostics_match_analytic_values():
    mask = torch.zeros(1, 144, dtype=torch.bool)
    mask[:, :2] = True
    old = torch.zeros(1, 144)
    new = old.clone()
    new[0, 0] = np.log(3)
    new[0, 143] = 1000
    action = torch.tensor([0])
    old_dist, new_dist = masked_distribution(old, mask), masked_distribution(new, mask)
    zeros = torch.zeros(1)
    loss, metrics = clipped_objective(
        new_dist.log_prob(action),
        old_dist.log_prob(action),
        torch.ones(1),
        zeros,
        zeros,
        new_dist.entropy(),
        value_coefficient=0,
        entropy_coefficient=0,
    )
    assert (
        metrics["ratio_mean"]
        == metrics["ratio_min"]
        == metrics["ratio_max"]
        == pytest.approx(1.5)
    )
    assert metrics["approximate_kl"] == pytest.approx(0.5 - np.log(1.5))
    assert metrics["clip_fraction"] == 1 and loss.item() == pytest.approx(-1.2)
    assert metrics["entropy"] == pytest.approx(
        -0.75 * np.log(0.75) - 0.25 * np.log(0.25)
    )
    _, identity = clipped_objective(
        old_dist.log_prob(action),
        old_dist.log_prob(action),
        torch.ones(1),
        zeros,
        zeros,
        old_dist.entropy(),
    )
    assert (
        identity["ratio_mean"] == 1
        and identity["approximate_kl"] == identity["clip_fraction"] == 0
    )


def test_explained_variance_and_raw_advantages_are_well_defined():
    targets = np.array([-1.0, 0, 1])
    assert explained_variance(targets, targets) == 1
    assert explained_variance(np.zeros(3), targets) == 0
    assert explained_variance(-targets, targets) == -3
    assert explained_variance([0, 1], [1, 1]) is None
    with pytest.raises(ValueError, match="finite"):
        explained_variance([np.nan], [1])
    diag = rollout_diagnostics(targets, targets, [-2, 0, 2])
    assert diag["advantage_mean"] == 0 and diag["advantage_std"] == pytest.approx(
        np.std([-2, 0, 2])
    )


def test_gae_boundaries_stop_carry_but_only_terminal_stops_bootstrap():
    advantages, returns = generalized_advantages(
        [1, -1, 10],
        [2, 3, 5],
        [4, 99, 6],
        np.array([False, True, False]),
        np.array([True, True, False]),
        gamma=0.5,
        gae_lambda=1,
    )
    np.testing.assert_allclose(advantages, [1, -4, 8])
    np.testing.assert_allclose(returns, [3, -1, 13])


def test_milestones_do_not_change_training_and_checkpoints_describe_pending_rollouts(
    tmp_path,
):
    config = PPOTrainingConfig(
        level_id="first-match",
        environment_steps=12,
        rollout_steps=8,
        optimization_epochs=1,
        minibatch_size=4,
        hidden_dims=(8,),
        evaluation_episodes=1,
    )
    control = train_ppo(config, tmp_path / "control")
    measured = train_ppo(replace(config, evaluation_every=5), tmp_path / "measured")
    assert control["final_weights"] == measured["final_weights"]
    assert control["optimization_statistics"] == measured["optimization_statistics"]
    assert [r["environment_step"] for r in measured["evaluation_rounds"]] == [
        0,
        5,
        10,
        12,
    ]
    for step, optimized in ((0, 0), (5, 0), (10, 8), (12, 12)):
        agent, payload = PPOAgent.load_checkpoint(
            tmp_path / f"measured/policy_step_{step}.pt"
        )
        assert payload["architecture"] == agent.architecture_metadata()
        assert payload["training_progress"]["last_optimized_step"] == optimized
        assert payload["training_progress"]["pending_rollout_steps"] == step - optimized
    with (tmp_path / "measured/rollouts.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2 and all(float(r["learning_rate"]) == 3e-4 for r in rows)
    assert all(np.isfinite(float(r["advantage_std"])) for r in rows)


def test_extended_profile_changes_only_budget_and_evaluation_frequency():
    before, after = asdict(PPOTrainingConfig()), asdict(pilot_config())
    assert {key for key in before if before[key] != after[key]} == {
        "environment_steps",
        "evaluation_every",
    }
    assert after["environment_steps"] == 20000 and after["evaluation_every"] == 2500
    with pytest.raises(ValueError):
        PPOTrainingConfig(evaluation_seed_start=400000)
    with pytest.raises(ValueError):
        PPOTrainingConfig(evaluation_every=0)


def test_analysis_exports_ten_figures_and_checks_checkpoint_inference(tmp_path):
    import hashlib

    from match3_ai.agents.dqn_agent import DQNAgent
    from match3_ai.training.ppo_analysis import analyze_pilot, paired_comparison

    config = PPOTrainingConfig(
        level_id="first-match",
        environment_steps=4,
        rollout_steps=4,
        optimization_epochs=1,
        minibatch_size=4,
        hidden_dims=(8,),
        evaluation_episodes=1,
        evaluation_every=2,
    )
    train_ppo(config, tmp_path / "run")
    reference = tmp_path / "dqn.pt"
    DQNAgent(7, hidden_dims=(8,)).save_checkpoint(
        reference, training_config={"level_id": "first-match"}, training_step=0
    )
    before = hashlib.sha256(reference.read_bytes()).hexdigest()
    result = analyze_pilot(tmp_path / "run", tmp_path / "analysis", [reference])
    assert len(result["figures"]) == 20
    assert len(result["checkpoint_checks"]) == 3
    assert all(r["states_checked"] == 1 for r in result["checkpoint_checks"])
    assert (
        result["comparisons_final_ppo_minus_reference"]["RandomAgent"]["outcomes"][
            "difference"
        ]["n"]
        == 1
    )
    assert not result["final_test_used"]
    assert before == hashlib.sha256(reference.read_bytes()).hexdigest()
    assert (tmp_path / "analysis/report.md").is_file()
    with pytest.raises(FileExistsError):
        analyze_pilot(tmp_path / "run", tmp_path / "analysis", [reference])
    with pytest.raises(ValueError, match="identical"):
        paired_comparison({1: {}}, {2: {}})
