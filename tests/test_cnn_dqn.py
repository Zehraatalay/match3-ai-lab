from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from match3_ai.agents.cnn_q_network import CNNQNetwork
from match3_ai.agents.dqn_agent import DQNAgent
from match3_ai.agents.observation_encoder import ObservationEncoder
from match3_ai.agents.replay_buffer import ReplayBuffer
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.training.architecture_pilot import pilot_config
from match3_ai.training.dqn import train_dqn


@pytest.mark.parametrize("level", Match3LevelEnv().level_ids)
def test_cnn_encoding_is_bitwise_equivalent_on_all_board_shapes(level):
    observation, _ = Match3LevelEnv(level).reset(seed=300000)
    before = deepcopy(observation)
    encoder = ObservationEncoder()
    flat, flat_mask = encoder.encode(observation)
    spatial, global_features, mask = encoder.encode_cnn(observation)
    assert spatial.shape == (20, 9, 8) and global_features.shape == (27,)
    assert spatial.dtype == global_features.dtype == np.float32
    np.testing.assert_array_equal(np.r_[spatial.ravel(), global_features], flat)
    np.testing.assert_array_equal(mask, flat_mask)
    rows, cols = map(int, observation["board_shape"])
    assert not spatial[:, rows:, :].any() and not spatial[:, :, cols:].any()
    holes = ~observation["cell_mask"].astype(bool)
    assert not spatial[:19, holes].any()
    assert spatial[19, :rows, :cols].all()
    for key in before:
        np.testing.assert_array_equal(observation[key], before[key])


def test_cnn_encoder_preserves_colorless_specials_hp_and_objective_values():
    observation, _ = Match3LevelEnv("blocker-mayhem").reset(seed=300000)
    observation["tiles"][2, 3] = 255
    observation["specials"][2, 3] = 5
    observation["blockers"][2, 3] = 2
    observation["blocker_hp"][2, 3] = 3
    spatial, global_features, mask = ObservationEncoder().encode_cnn(observation)
    assert (
        spatial[6, 2, 3]
        == spatial[12, 2, 3]
        == spatial[15, 2, 3]
        == spatial[17, 2, 3]
        == 1
    )
    baseline = global_features.copy()
    observation["objective_target"][1] += 1
    assert not np.array_equal(ObservationEncoder().encode_cnn(observation)[1], baseline)
    observation["action_mask"][:] = 0
    np.testing.assert_array_equal(
        ObservationEncoder().encode_cnn(observation)[0], spatial
    )
    assert mask.any()


def test_cnn_forward_shapes_parameters_and_rng_isolation():
    before = torch.get_rng_state().clone()
    first, second = CNNQNetwork(seed=42), CNNQNetwork(seed=42)
    assert torch.equal(before, torch.get_rng_state())
    assert sum(p.numel() for p in first.parameters()) == 358344
    assert all(
        torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())
    )
    flat = torch.zeros((3, 1467))
    assert first(flat).shape == (3, 144)
    torch.testing.assert_close(
        first(flat), first(flat[:, :1440].reshape(3, 20, 9, 8), flat[:, 1440:])
    )
    with pytest.raises(ValueError, match="flat"):
        first(torch.zeros(3, 1466))
    with pytest.raises(ValueError, match="global"):
        first(torch.zeros(3, 20, 9, 8), torch.zeros(3, 26))


def test_cnn_masking_exploration_and_deterministic_inference():
    observation, _ = Match3LevelEnv("blocker-mayhem").reset(seed=300000)
    observation["action_mask"][:] = 0
    observation["action_mask"][[5, 7]] = 1
    agent = DQNAgent(5903, hidden_dims=(8,), architecture="cnn")
    with torch.no_grad():
        for p in agent.online.parameters():
            p.zero_()
        agent.online.head[-1].bias[0] = 100000
        agent.online.head[-1].bias[7] = 2
    assert [agent.select_action(observation) for _ in range(3)] == [7, 7, 7]
    assert {agent.select_action(observation, epsilon=1) for _ in range(100)} == {5, 7}
    observation["action_mask"][:] = 0
    with pytest.raises(RuntimeError, match="no valid"):
        agent.select_action(observation)


def test_cnn_updates_finite_gradients_frozen_target_and_checkpoint_roundtrip(tmp_path):
    observation, _ = Match3LevelEnv("blocker-mayhem").reset(seed=300000)
    action = int(np.flatnonzero(observation["action_mask"])[0])
    replay = ReplayBuffer(4, seed=12)
    for reward in (0, 1, -1, 0):
        replay.add(observation, action, reward, observation, False, False)
    agent = DQNAgent(5903, hidden_dims=(8,), architecture="cnn")
    target = [p.clone() for p in agent.target.parameters()]
    online = [p.clone() for p in agent.online.parameters()]
    metrics = agent.update(replay.sample(4))
    assert all(np.isfinite(value) for value in metrics.values())
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all()
        for p in agent.online.parameters()
    )
    assert any(not torch.equal(a, b) for a, b in zip(online, agent.online.parameters()))
    assert all(torch.equal(a, b) for a, b in zip(target, agent.target.parameters()))
    agent.synchronize_target()
    assert all(
        torch.equal(a, b)
        for a, b in zip(agent.online.parameters(), agent.target.parameters())
    )
    agent.save_checkpoint(tmp_path / "cnn.pt")
    loaded, payload = DQNAgent.load_checkpoint(tmp_path / "cnn.pt")
    assert payload["architecture"] == loaded.architecture == "cnn"
    assert payload["network_config"] == CNNQNetwork.architecture_config
    assert loaded.select_action(observation) == agent.select_action(observation)
    assert all(
        torch.equal(a, b)
        for a, b in zip(agent.online.parameters(), loaded.online.parameters())
    )
    assert len(loaded.optimizer.state) == len(agent.optimizer.state)
    with pytest.raises(FileExistsError):
        agent.save_checkpoint(tmp_path / "cnn.pt")


def test_legacy_mlp_checkpoint_without_architecture_fields_loads(tmp_path):
    observation, _ = Match3LevelEnv().reset(seed=300000)
    agent = DQNAgent(5903, hidden_dims=(8,))
    agent.save_checkpoint(tmp_path / "modern.pt")
    payload = torch.load(tmp_path / "modern.pt", weights_only=True)
    payload.pop("architecture")
    payload.pop("network_config")
    with (tmp_path / "legacy.pt").open("xb") as handle:
        torch.save(payload, handle)
    loaded, _ = DQNAgent.load_checkpoint(tmp_path / "legacy.pt")
    assert loaded.architecture == "mlp"
    assert loaded.select_action(observation) == agent.select_action(observation)
    assert all(
        torch.equal(a, b)
        for a, b in zip(agent.online.parameters(), loaded.online.parameters())
    )


def test_controlled_profiles_and_tiny_cnn_training_reproducibility(tmp_path):
    mlp, cnn = asdict(pilot_config("mlp")), asdict(pilot_config("cnn"))
    assert [key for key in mlp if mlp[key] != cnn[key]] == ["architecture"]
    assert cnn["reward_mode"] == "sparse" and cnn["environment_steps"] == 5000
    config = replace(
        pilot_config("cnn"),
        level_id="first-match",
        environment_steps=12,
        hidden_dims=(8,),
        replay_capacity=32,
        batch_size=2,
        warmup_steps=2,
        train_every=1,
        target_sync_updates=2,
        evaluation_every=6,
        evaluation_episodes=1,
    )
    first = train_dqn(config, tmp_path / "first")
    repeated = train_dqn(config, tmp_path / "repeat")
    assert first["architecture"] == "cnn" and first["parameter_count"] > 0
    assert first["optimizer_updates"] == 11 and first["finite_losses"]
    assert first["final_weights"] == repeated["final_weights"]
    assert first["evaluation_invalid_actions"] == first["invalid_actions"] == 0
    resumed = train_dqn(
        replace(config, environment_steps=16),
        tmp_path / "resume",
        resume_from=first["checkpoint"],
    )
    assert resumed["training_start_step"] == 12 and resumed["replay_size"] == 16
    with pytest.raises(ValueError, match="compatible"):
        train_dqn(
            replace(config, architecture="mlp", environment_steps=16),
            tmp_path / "wrong",
            resume_from=first["checkpoint"],
        )


def test_architecture_analysis_audits_paired_runs_and_preserves_artifacts(tmp_path):
    import hashlib

    from match3_ai.training.architecture_analysis import analyze_pilot

    config = replace(
        pilot_config("mlp"),
        level_id="first-match",
        environment_steps=12,
        hidden_dims=(8,),
        replay_capacity=32,
        batch_size=2,
        warmup_steps=2,
        train_every=1,
        target_sync_updates=2,
        evaluation_every=6,
        evaluation_episodes=2,
    )
    for architecture in ("mlp", "cnn"):
        train_dqn(replace(config, architecture=architecture), tmp_path / architecture)
    paths = list((tmp_path / "mlp").rglob("*.pt")) + list(
        (tmp_path / "cnn").rglob("*.pt")
    )
    before = {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    result = analyze_pilot(tmp_path / "mlp", tmp_path / "cnn", tmp_path / "analysis")
    assert result["paired_final_outcomes"]["difference"]["n"] == 2
    assert len(result["figures"]) == 16
    assert all((tmp_path / "analysis/figures" / p).is_file() for p in result["figures"])
    assert result["inference_benchmark"]["policies"]["cnn"]["samples"] == 6
    assert result["inference_benchmark"]["policies"]["mlp"]["median"] > 0
    assert not result["final_test_used"]
    assert before == {p: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    with pytest.raises(FileExistsError):
        analyze_pilot(tmp_path / "mlp", tmp_path / "cnn", tmp_path / "analysis")
