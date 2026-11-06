import csv
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from match3_ai.agents.ppo_agent import ActorCritic, PPOAgent, masked_distribution
from match3_ai.evaluation.analytics import AnalyticsCollector
from match3_ai.evaluation.runner import EpisodeResult
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.training.ppo import (
    PPOTrainingConfig,
    evaluate_ppo,
    fingerprint,
    train_ppo,
)
from match3_ai.training.ppo_update import (
    clipped_objective,
    generalized_advantages,
    normalize_advantages,
    optimize_ppo,
)


def observation():
    return Match3LevelEnv("first-match").reset(seed=300000)[0]


def tiny_config():
    return PPOTrainingConfig(
        level_id="first-match",
        environment_steps=12,
        seed=5913,
        rollout_steps=8,
        optimization_epochs=2,
        minibatch_size=4,
        hidden_dims=(8,),
        evaluation_episodes=2,
    )


def test_actor_critic_shapes_finiteness_and_isolated_initialization():
    state = torch.get_rng_state().clone()
    first, second = ActorCritic(seed=7), ActorCritic(seed=7)
    assert torch.equal(state, torch.get_rng_state())
    assert sum(p.numel() for p in first.parameters()) == 427409
    logits, values = first(torch.zeros(3, 1467))
    assert logits.shape == (3, 144) and values.shape == (3,)
    assert torch.isfinite(logits).all() and torch.isfinite(values).all()
    for a, b in zip(first.parameters(), second.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_masked_distribution_probabilities_log_probs_entropy_and_gradients():
    logits = torch.zeros(2, 144, requires_grad=True)
    masks = torch.zeros(2, 144, dtype=torch.bool)
    masks[0, [2, 7]] = True
    masks[1, 8] = True
    distribution = masked_distribution(logits, masks)
    assert torch.equal(
        distribution.probs[~masks], torch.zeros_like(distribution.probs[~masks])
    )
    torch.testing.assert_close(distribution.probs.sum(-1), torch.ones(2))
    torch.testing.assert_close(
        distribution.log_prob(torch.tensor([7, 8])),
        torch.tensor([-np.log(2), 0], dtype=torch.float32),
    )
    torch.testing.assert_close(
        distribution.entropy(), torch.tensor([np.log(2), 0], dtype=torch.float32)
    )
    (
        distribution.entropy().sum() - distribution.log_prob(torch.tensor([7, 8])).sum()
    ).backward()
    assert torch.isfinite(logits.grad).all() and not logits.grad[~masks].any()
    with pytest.raises(ValueError, match="valid action"):
        masked_distribution(logits, torch.zeros_like(masks))
    with pytest.raises(ValueError, match="finite"):
        masked_distribution(torch.full_like(logits, float("nan")), masks)


@pytest.mark.parametrize("level", Match3LevelEnv().level_ids)
def test_ppo_observations_sampling_and_deterministic_evaluation_on_all_levels(level):
    env = Match3LevelEnv(level)
    obs, _ = env.reset(seed=300000)
    agent = PPOAgent(7, hidden_dims=(8,))
    before = deepcopy(obs)
    for _ in range(8):
        action, log_prob, value = agent.act(obs)
        assert obs["action_mask"][action] and np.isfinite([log_prob, value]).all()
        features, mask = agent.encoder.encode(obs)
        actual, entropy, values = agent.evaluate_actions(
            torch.from_numpy(features[None]),
            torch.from_numpy(mask[None]),
            torch.tensor([action]),
        )
        assert actual.item() == pytest.approx(log_prob)
        assert values.item() == pytest.approx(value) and entropy.item() >= 0
    state = agent.policy_rng.get_state().clone()
    assert len({agent.select_action(obs) for _ in range(3)}) == 1
    assert torch.equal(state, agent.policy_rng.get_state())
    for key in obs:
        np.testing.assert_array_equal(obs[key], before[key])
    after, _, _, _, info = env.step(agent.select_action(obs))
    assert info["valid_action"] and env.observation_space.contains(after)
    env.close()


def test_sampling_ignores_larger_invalid_logits_and_handles_terminal_empty_masks():
    obs = observation()
    obs["action_mask"][:] = 0
    obs["action_mask"][[2, 7]] = 1
    agent = PPOAgent(7, hidden_dims=(8,))
    with torch.no_grad():
        for p in agent.network.parameters():
            p.zero_()
        agent.network.actor.bias[0] = 1000
        agent.network.actor.bias[7] = 0.5
    assert agent.select_action(obs) == 7
    assert {agent.act(obs)[0] for _ in range(100)} == {2, 7}
    obs["action_mask"][:] = 0
    assert np.isfinite(agent.value(obs))
    with pytest.raises(ValueError, match="valid action"):
        agent.act(obs)


@pytest.mark.parametrize(
    "terminal,expected", [(False, [2.024, 1.7]), (True, [0.08, -1.0])]
)
def test_gae_hand_calculation_truncation_bootstrap_and_terminal_precedence(
    terminal, expected
):
    advantages, returns = generalized_advantages(
        [0, 1],
        [1, 2],
        [2, 3],
        np.array([False, terminal]),
        np.array([False, True]),
        gamma=0.9,
        gae_lambda=0.8,
    )
    np.testing.assert_allclose(advantages, expected, rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(returns, np.array(expected) + [1, 2], rtol=1e-6)


def test_gae_never_propagates_reset_episode_advantage_and_bootstraps_rollout_tail():
    advantages, _ = generalized_advantages(
        [0, 100],
        [1, 0],
        [3, 0],
        np.array([False, True]),
        np.array([True, False]),
        gamma=0.9,
        gae_lambda=1,
    )
    assert advantages.tolist() == pytest.approx([1.7, 100])
    tail, _ = generalized_advantages(
        [0], [1], [3], np.array([False]), np.array([False]), gamma=0.9
    )
    assert tail.item() == pytest.approx(1.7)


def test_clipped_policy_objective_both_advantage_signs_value_entropy_and_diagnostics():
    ratio = torch.tensor([1.5, 0.5, 1.5, 0.5])
    advantages = torch.tensor([1.0, 1, -1, -1])
    zeros = torch.zeros(4)
    loss, metrics = clipped_objective(
        ratio.log(), zeros, advantages, torch.ones(4), zeros, torch.full((4,), 2.0)
    )
    assert metrics["policy_loss"] == pytest.approx(0.15)
    assert metrics["value_loss"] == pytest.approx(0.5)
    assert metrics["entropy"] == 2
    assert metrics["clip_fraction"] == 1
    assert metrics["approximate_kl"] >= 0
    assert loss.item() == pytest.approx(0.15 + 0.5 * 0.5 - 0.01 * 2)
    normalized = normalize_advantages(torch.tensor([1.0, 2, 3]))
    assert normalized.mean().item() == pytest.approx(0)
    assert normalized.std(unbiased=False).item() == pytest.approx(1)
    assert normalize_advantages(torch.tensor([2.0])).item() == 0
    assert not normalize_advantages(torch.ones(4)).any()


def test_ppo_update_masked_rollout_fixed_targets_and_checkpoint_rng_roundtrip(tmp_path):
    agent = PPOAgent(7, hidden_dims=(8,))
    obs = observation()
    features, mask = agent.encoder.encode(obs)
    actions, logs = zip(*(agent.act(obs)[:2] for _ in range(4)))
    batch = {
        "features": torch.from_numpy(np.stack([features] * 4)),
        "masks": torch.from_numpy(np.stack([mask] * 4)),
        "actions": torch.tensor(actions),
        "log_probs": torch.tensor(logs),
        "advantages": torch.tensor([1.0, -1, 0.5, -0.5]),
        "returns": torch.tensor([1.0, -1, 1, -1]),
    }
    original = {key: value.clone() for key, value in batch.items()}
    before = fingerprint(agent)
    metrics = optimize_ppo(agent, batch, epochs=2, minibatch_size=3)
    assert len(metrics) == 4 and fingerprint(agent) != before
    assert all(np.isfinite(list(row.values())).all() for row in metrics)
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in agent.network.parameters()
    )
    for key, value in batch.items():
        torch.testing.assert_close(value, original[key], rtol=0, atol=0)
    agent.save_checkpoint(
        tmp_path / "ppo.pt",
        training_config={"gamma": 0.99},
        training_progress={"environment_step": 4},
    )
    loaded, payload = PPOAgent.load_checkpoint(tmp_path / "ppo.pt")
    assert payload["training_progress"]["environment_step"] == 4
    assert payload["encoder_config"] == asdict(agent.encoder.config)
    assert fingerprint(agent) == fingerprint(loaded)
    assert agent.select_action(obs) == loaded.select_action(obs)
    assert [agent.act(obs) for _ in range(10)] == [loaded.act(obs) for _ in range(10)]
    np.testing.assert_array_equal(
        agent.shuffle_rng.permutation(10), loaded.shuffle_rng.permutation(10)
    )
    for a, b in zip(agent.optimizer.state.values(), loaded.optimizer.state.values()):
        for key in a:
            torch.testing.assert_close(a[key], b[key], rtol=0, atol=0)
    with pytest.raises(FileExistsError):
        agent.save_checkpoint(tmp_path / "ppo.pt")
    bad_actions = batch["actions"].clone()
    bad_actions[0] = int(np.flatnonzero(~mask)[0])
    with pytest.raises(ValueError, match="mask-valid"):
        agent.evaluate_actions(batch["features"], batch["masks"], bad_actions)


def test_tiny_training_reproducible_sparse_and_evaluation_rng_isolation(tmp_path):
    config = tiny_config()
    state = torch.get_rng_state().clone()
    first = train_ppo(config, tmp_path / "first")
    repeated = train_ppo(config, tmp_path / "repeat")
    assert torch.equal(state, torch.get_rng_state())
    assert first["environment_steps"] == 12 and first["rollouts"] == 2
    assert first["optimizer_updates"] == 6
    assert (
        first["initial_weights"] != first["final_weights"] == repeated["final_weights"]
    )
    assert first["optimization_statistics"] == repeated["optimization_statistics"]
    assert first["invalid_actions"] == first["evaluation_invalid_actions"] == 0
    assert first["checkpoint_inference_valid"]
    assert (tmp_path / "first/transitions.csv").read_text() == (
        tmp_path / "repeat/transitions.csv"
    ).read_text()
    with (tmp_path / "first/transitions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert all(
        200000 <= int(r["episode_seed"]) <= 299999 and float(r["reward"]) in (-1, 0, 1)
        for r in rows
    )
    agent, _ = PPOAgent.load_checkpoint(first["checkpoint"])
    collector = AnalyticsCollector()
    before = agent.policy_rng.get_state().clone()
    result = evaluate_ppo(config, agent, step=12, run_id="test", collector=collector)
    assert torch.equal(before, agent.policy_rng.get_state())
    assert len(result) == 2 and all(r.invalid_actions == 0 for r in result)
    with pytest.raises(FileExistsError):
        train_ppo(config, tmp_path / "first")


@pytest.mark.parametrize("terminated", [False, True])
def test_trainer_bootstraps_final_observation_before_reset(
    tmp_path, monkeypatch, terminated
):
    import match3_ai.training.ppo as module

    base = Match3LevelEnv()
    initial = observation()

    class BoundaryEnvironment:
        config = base.config

        def reset(self, seed):
            return deepcopy(initial), {}

        def step(self, action):
            final = deepcopy(initial)
            final["moves_remaining"] = 4
            if terminated:
                final["action_mask"][:] = 0
            return (
                final,
                float(terminated),
                terminated,
                not terminated,
                {
                    "valid_action": True,
                    "level_status": "WON" if terminated else "IN_PROGRESS",
                },
            )

        def close(self):
            pass

    def evaluation(config, agent, *, step, run_id, collector):
        record = EpisodeResult(
            seed=300000,
            total_reward=1,
            steps=0,
            valid_actions=0,
            invalid_actions=0,
            total_cleared=0,
            total_cascades=0,
            reshuffles=0,
            completed=True,
            experiment_id=run_id,
            episode_id=f"{run_id}:{step}",
            agent_name="PPOAgent",
            level_id="first-match",
            outcome="WON",
        )
        collector.record_episode(record)
        return [record]

    monkeypatch.setattr(module, "environment", lambda config: BoundaryEnvironment())
    monkeypatch.setattr(module, "evaluate_ppo", evaluation)
    monkeypatch.setattr(
        PPOAgent, "value", lambda self, obs: float(obs["moves_remaining"])
    )
    train_ppo(replace(tiny_config(), environment_steps=2), tmp_path / "run")
    with (tmp_path / "run/transitions.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [float(r["next_value"]) for r in rows] == [0 if terminated else 4] * 2


def test_config_rejects_final_test_seeds_and_invalid_hyperparameters():
    for kwargs in (
        {"evaluation_seed_start": 400000},
        {"rollout_steps": 0},
        {"gae_lambda": 1.1},
        {"clip_ratio": 1},
        {"entropy_coefficient": -1},
        {"hidden_dims": ()},
    ):
        with pytest.raises(ValueError):
            PPOTrainingConfig(**kwargs)
