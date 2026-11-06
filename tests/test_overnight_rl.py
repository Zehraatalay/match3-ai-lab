from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from match3_ai.agents.dqn_agent import DQNAgent, masked_bellman_targets
from match3_ai.agents.dueling_q_network import (
    DuelingQNetwork,
    masked_double_bellman_targets,
)
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.training.dqn import TrainingConfig, train_dqn
from match3_ai.training.overnight import Journal, OvernightPPOConfig, late_learning_rate
from match3_ai.training.overnight_analysis import paired_statistics
from match3_ai.training.ppo import train_ppo


def test_double_targets_select_online_evaluate_target_and_mask():
    online, target = torch.zeros(4, 144), torch.zeros(4, 144)
    masks = torch.zeros(4, 144, dtype=torch.bool)
    masks[:3, :2] = True
    online[:, 0], online[:, 1], online[:, 2] = 5, 2, 999
    target[:, 0], target[:, 1], target[:, 2] = 3, 8, 999
    terminal = torch.tensor([False, False, True, True])
    truncated = torch.tensor([False, True, False, True])
    rewards = torch.tensor([1.0, 1.0, -1.0, -1.0], requires_grad=True)
    actual = masked_double_bellman_targets(
        rewards, online, target, terminal, truncated, masks, 0.9
    )
    torch.testing.assert_close(actual, torch.tensor([3.7, 3.7, -1.0, -1.0]))
    vanilla = masked_bellman_targets(rewards, target, terminal, truncated, masks, 0.9)
    torch.testing.assert_close(vanilla[:2], torch.tensor([8.2, 8.2]))
    assert not actual.requires_grad
    masks[1] = False
    with pytest.raises(ValueError, match="empty"):
        masked_double_bellman_targets(
            rewards, online, target, terminal, truncated, masks
        )


def test_dueling_aggregation_dimensions_gradient_and_initialization():
    a, b = DuelingQNetwork((8,), 4), DuelingQNetwork((8,), 4)
    x = torch.ones(2, 1467)
    y = a(x)
    assert y.shape == (2, 144)
    torch.testing.assert_close(y, b(x))
    torch.testing.assert_close(y.mean(-1), a.value(a.trunk(x)).squeeze(-1))
    y[:, 0].sum().backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in a.parameters()
    )
    assert a.value.weight.grad.abs().sum() > 0
    assert a.advantage.weight.grad.abs().sum() > 0


@pytest.mark.parametrize("architecture", ["mlp", "dueling"])
def test_double_checkpoint_roundtrip_legal_actions_and_training(tmp_path, architecture):
    env = Match3LevelEnv("first-match")
    observation, _ = env.reset(seed=300000)
    agent = DQNAgent(5, (8,), architecture=architecture, algorithm="double")
    action = agent.select_action(observation)
    assert observation["action_mask"][action]
    path = tmp_path / "policy.pt"
    agent.save_checkpoint(path)
    loaded, payload = DQNAgent.load_checkpoint(path)
    assert loaded.algorithm == payload["algorithm"] == "double"
    assert loaded.architecture == architecture
    assert (
        action == loaded.select_action(observation) == agent.select_action(observation)
    )
    cfg = TrainingConfig(
        level_id="first-match",
        seed=5,
        environment_steps=12,
        train_seed_start=200000,
        evaluation_seed_start=300000,
        evaluation_episodes=1,
        evaluation_every=12,
        replay_capacity=32,
        warmup_steps=2,
        batch_size=2,
        train_every=1,
        hidden_dims=(8,),
        algorithm="double",
        architecture=architecture,
        evaluate_step_zero=True,
        baseline_every_checkpoint=False,
    )
    events = []
    result = train_dqn(
        cfg, tmp_path / "run", observer=lambda event, data: events.append((event, data))
    )
    repeat = train_dqn(cfg, tmp_path / "repeat")
    assert result["final_weights"] == repeat["final_weights"]
    assert result["finite_losses"] and result["checkpoint_inference_valid"]
    assert result["optimizer_updates"] == 11
    assert any(event == "milestone" for event, _ in events)
    env.close()


def test_lr_boundaries_and_identical_control_treatment_prefix(tmp_path):
    assert late_learning_rate(0) == 0.0003
    assert late_learning_rate(49920) == 0.0003
    assert late_learning_rate(50000) == 0.0003
    assert late_learning_rate(75000) == pytest.approx(0.000165)
    assert late_learning_rate(100000) == pytest.approx(0.00003)
    cfg = OvernightPPOConfig(
        level_id="first-match",
        seed=9,
        environment_steps=12,
        rollout_steps=8,
        minibatch_size=4,
        hidden_dims=(8,),
        evaluation_seed_start=300000,
        evaluation_episodes=1,
        evaluation_every=6,
    )
    events = []
    a = train_ppo(cfg, tmp_path / "control")
    b = train_ppo(
        replace(cfg, learning_rate_schedule="late_linear"),
        tmp_path / "treatment",
        learning_rate_schedule=late_learning_rate,
        observer=lambda e, d: events.append((e, d)),
    )
    assert a["initial_weights"] == b["initial_weights"]
    assert a["final_weights"] == b["final_weights"]
    assert any(e == "rollout" for e, _ in events)


def test_legacy_checkpoint_without_algorithm_field_still_loads(tmp_path):
    agent = DQNAgent(5, (8,))
    path = tmp_path / "new.pt"
    agent.save_checkpoint(path)
    data = torch.load(path, weights_only=True)
    data.pop("algorithm")
    legacy = tmp_path / "legacy.pt"
    torch.save(data, legacy)
    loaded, _ = DQNAgent.load_checkpoint(legacy)
    assert loaded.algorithm == "vanilla"
    for a, b in zip(agent.online.parameters(), loaded.online.parameters()):
        np.testing.assert_array_equal(a.detach().numpy(), b.detach().numpy())


def test_paired_analysis_keeps_board_and_training_units_distinct():
    a = np.array([[1, 1, 0, 0], [1, 1, 0, 0], [1, 1, 0, 0]])
    b = np.array([[0, 1, 0, 0]] * 3)
    result = paired_statistics(a, b, samples=200, seed=4)
    assert result["difference"] == 0.25
    assert result["n_boards"] == 4 and result["n_training_pairs"] == 3
    assert result["training_pair_sample_std"] == 0
    assert result["training_pair_signflip_sensitivity_p"] == 0.25
    assert all(r["counts"]["treatment_only"] == 1 for r in result["individual"])
    with pytest.raises(ValueError):
        paired_statistics(a, b[:, :2])


def test_journal_exports_actual_run_namespace_without_executing_episode(tmp_path):
    from match3_ai.evaluation.analytics import AnalyticsCollector
    from match3_ai.evaluation.runner import EpisodeResult

    collector = AnalyticsCollector()
    collector.record_episode(
        EpisodeResult(
            seed=1,
            total_reward=1.0,
            steps=0,
            valid_actions=0,
            invalid_actions=0,
            total_cleared=0,
            total_cascades=0,
            reshuffles=0,
            completed=True,
            experiment_id="actual-training-run",
            episode_id="actual-training-run:0:PPOAgent:0",
            agent_name="PPOAgent",
            outcome="WON",
        )
    )
    journal = Journal(tmp_path, {"id": "external-job-id"})
    journal("milestone", {"environment_step": 0, "collector": collector})
    journal.finish("completed")
    assert (
        tmp_path / "records/external-job-id/evaluation/step_0/episodes.csv"
    ).is_file()
