from copy import deepcopy
from dataclasses import asdict, replace
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from match3_ai.levels.config import LevelConfig, LevelObjective, ObjectiveType
from match3_ai.training import neural_mcts as training
from match3_ai.training.neural_mcts import (
    NeuralMCTSTrainer,
    NeuralMCTSTrainingConfig,
    policy_value_loss,
)
from match3_ai.training.search_replay import SearchExample, SearchReplayBuffer


def tiny_level(target=1):
    return LevelConfig(
        rows=3,
        cols=3,
        num_colors=3,
        move_limit=1,
        level_id="first-match",
        objectives=(LevelObjective(ObjectiveType.COLLECT_TILES, target=target),),
    )


def tiny_config(**kwargs):
    return NeuralMCTSTrainingConfig(
        collection_rounds=2,
        episodes_per_round=1,
        simulations=1,
        replay_capacity=8,
        batch_size=1,
        updates_per_round=1,
        **kwargs,
    )


def example(index=0, outcome=1):
    mask = np.zeros(144, dtype=bool)
    mask[[0, 1]] = True
    policy = np.zeros(144)
    policy[[0, 1]] = [0.25, 0.75]
    return SearchExample(
        np.zeros((20, 9, 8)),
        np.zeros(30),
        mask,
        policy,
        outcome,
        {"position": index, "nested": {"value": 1}},
    )


def test_replay_capacity_copy_isolation_and_eviction():
    replay = SearchReplayBuffer(2)
    first = example(0)
    replay.extend([first, example(1), example(2, -1)])
    assert len(replay) == 2
    assert {e["position"] for e in replay.sample(2).provenance} == {1, 2}
    replay = SearchReplayBuffer(2)
    replay.extend([first])
    first.provenance["nested"]["value"] = 99
    batch = replay.sample(1)
    assert batch.provenance[0]["nested"]["value"] == 1
    batch.spatial[:] = 10
    batch.policy_target[:] = 0
    batch.provenance[0]["nested"]["value"] = 88
    again = replay.sample(1)
    assert not again.spatial.any() and again.policy_target.sum() == 1
    assert again.provenance[0]["nested"]["value"] == 1


def test_replay_sampling_serialization_and_state_copy():
    replay = SearchReplayBuffer(4, seed=7)
    replay.extend([example(i) for i in range(4)])
    state = replay.state_dict()
    restored = SearchReplayBuffer(4, seed=99)
    restored.load_state_dict(state)
    for _ in range(3):
        assert replay.sample(2).provenance == restored.sample(2).provenance
    state["examples"][0]["spatial"][:] = 22
    assert not restored.sample(4).spatial.any()
    with pytest.raises(ValueError):
        restored.sample(5)
    with pytest.raises(ValueError):
        SearchReplayBuffer(3).load_state_dict(replay.state_dict())


@pytest.mark.parametrize(
    "change",
    ["negative", "illegal", "unnormalized", "nan", "outcome", "empty_mask", "spatial"],
)
def test_invalid_examples_rejected(change):
    data = deepcopy(vars(example()))
    if change == "negative":
        data["policy_target"][0] = -0.1
    elif change == "illegal":
        data["policy_target"][143] = 0.1
    elif change == "unnormalized":
        data["policy_target"][:] = 0
    elif change == "nan":
        data["policy_target"][0] = np.nan
    elif change == "outcome":
        data["outcome"] = 0
    elif change == "empty_mask":
        data["action_mask"][:] = False
    else:
        data["spatial"][0, 0, 0] = np.inf
    with pytest.raises(ValueError):
        SearchExample(**data)


def test_loss_exact_masking_and_finite_gradients():
    logits = torch.zeros(1, 144, requires_grad=True)
    with torch.no_grad():
        logits[0, 143] = 1000
    values = torch.tensor([0.5], requires_grad=True)
    mask = torch.zeros(1, 144, dtype=torch.bool)
    mask[0, :2] = True
    targets = torch.zeros(1, 144)
    targets[0, :2] = torch.tensor([0.25, 0.75])
    total, policy, value = policy_value_loss(
        logits, values, mask, targets, torch.ones(1), value_loss_weight=2
    )
    assert float(policy.detach()) == pytest.approx(np.log(2))
    assert float(value.detach()) == 0.25
    assert float(total.detach()) == pytest.approx(np.log(2) + 0.5)
    total.backward()
    assert torch.isfinite(logits.grad).all() and torch.isfinite(values.grad).all()
    assert logits.grad[0, 143] == 0
    targets[0, 143] = 0.1
    with pytest.raises(ValueError, match="illegal"):
        policy_value_loss(logits, values, mask, targets, torch.ones(1))


class ControlledEnv:
    instances: ClassVar[list] = []
    mode = "WON"

    def __init__(self, *args, **kwargs):
        self.config = kwargs["level_config"]
        self.closed = False
        self.instances.append(self)

    def reset(self, *, seed):
        self.seed = seed
        self.observation = {
            key: np.zeros((9, 8), dtype=np.uint8)
            for key in ("tiles", "specials", "blockers", "blocker_hp", "cell_mask")
        }
        self.observation["cell_mask"][:3, :3] = 1
        self.observation["tiles"][:3, :3] = 1
        self.observation.update(
            board_shape=np.array([3, 3]),
            moves_remaining=1,
            objective_type=np.array([2, 0]),
            objective_color=np.array([0, 0]),
            objective_target=np.array([1, 0]),
            objective_progress=np.array([0, 0]),
            objective_mask=np.array([1, 0]),
            action_mask=np.zeros(144, dtype=bool),
        )
        self.observation["action_mask"][:2] = True
        return self.observation, {"board_seed": 11, "engine_seed": 22}

    def step(self, action):
        if self.mode == "ERROR":
            raise RuntimeError("controlled error")
        return (
            self.observation,
            (1 if self.mode == "WON" else -1) if self.mode != "TRUNCATED" else 0,
            self.mode != "TRUNCATED",
            self.mode == "TRUNCATED",
            {"valid_action": True, "level_status": self.mode},
        )

    def close(self):
        self.closed = True


class ControlledAgent:
    def __init__(self, simulator, network, context, **kwargs):
        self.encoder = training.PolicyValueEncoder()
        policy = np.zeros(144)
        policy[:2] = [0.25, 0.75]
        self.last_search_stats = SimpleNamespace(policy_target=policy)

    def select_action(self, observation):
        return 0


@pytest.fixture
def controlled(monkeypatch):
    ControlledEnv.instances = []
    monkeypatch.setattr(training, "Match3LevelEnv", ControlledEnv)
    monkeypatch.setattr(training, "MCTSSimulator", lambda env: env)
    monkeypatch.setattr(training, "NeuralMCTSAgent", ControlledAgent)
    return ControlledEnv


@pytest.mark.parametrize(
    "mode,target", [("WON", 1), ("LOST", -1), ("TRUNCATED", None), ("ERROR", None)]
)
def test_terminal_assignment_discard_and_policy_extraction(
    controlled, monkeypatch, mode, target
):
    monkeypatch.setattr(controlled, "mode", mode)
    trainer = NeuralMCTSTrainer(tiny_config(), tiny_level())
    result = trainer.collect_episode()
    assert result.outcome == mode
    assert len(trainer.replay) == 0
    assert controlled.instances[-1].closed
    if target is None:
        assert not result.examples
    else:
        assert len(result.examples) == 1 and result.examples[0].outcome == target
        np.testing.assert_allclose(result.examples[0].policy_target[:2], [0.25, 0.75])
        assert result.examples[0].provenance["search_config"]["simulations"] == 1
        assert result.examples[0].provenance["board_seed"] == 11


def test_collection_isolation_and_no_network_or_environment_mutation(
    controlled, monkeypatch
):
    monkeypatch.setattr(controlled, "mode", "WON")
    trainer = NeuralMCTSTrainer(tiny_config(), tiny_level())
    weights = {
        name: tensor.clone() for name, tensor in trainer.network.state_dict().items()
    }
    first, second = trainer.collect_episode(), trainer.collect_episode()
    assert controlled.instances[0] is not controlled.instances[1]
    assert first.provenance["attempt"] == 0 and second.provenance["attempt"] == 1
    snapshot = deepcopy(controlled.instances[0].observation)
    controlled.instances[1].observation["tiles"][:] = 0
    for key in snapshot:
        np.testing.assert_array_equal(
            controlled.instances[0].observation[key], snapshot[key]
        )
    for name, tensor in trainer.network.state_dict().items():
        torch.testing.assert_close(tensor, weights[name], rtol=0, atol=0)
    assert first.examples[0].spatial.any()


def assert_nested_equal(first, second):
    if isinstance(first, dict):
        assert first.keys() == second.keys()
        for key in first:
            assert_nested_equal(first[key], second[key])
    elif isinstance(first, (tuple, list)):
        assert len(first) == len(second)
        for a, b in zip(first, second):
            assert_nested_equal(a, b)
    elif isinstance(first, torch.Tensor):
        torch.testing.assert_close(first, second, rtol=0, atol=0)
    elif isinstance(first, np.ndarray):
        np.testing.assert_array_equal(first, second)
    else:
        assert first == second


def test_round_checkpoint_resume_equivalence_with_real_search(tmp_path):
    config, level = tiny_config(), tiny_level()
    torch_before = torch.get_rng_state().clone()
    full = NeuralMCTSTrainer(config, level)
    full.run()
    partial = NeuralMCTSTrainer(config, level)
    partial.run(until_round=1)
    path = tmp_path / "round.pt"
    partial.save_checkpoint(path)
    restored = NeuralMCTSTrainer.load_checkpoint(path, level)
    assert restored.rounds_completed == restored.updates == 1
    restored.run()
    assert_nested_equal(full.network.state_dict(), restored.network.state_dict())
    assert_nested_equal(full.optimizer.state_dict(), restored.optimizer.state_dict())
    assert_nested_equal(full.replay.state_dict(), restored.replay.state_dict())
    assert_nested_equal(full.metrics, restored.metrics)
    assert_nested_equal(
        {k: r.bit_generator.state for k, r in full.rngs.items()},
        {k: r.bit_generator.state for k, r in restored.rngs.items()},
    )
    assert torch.equal(torch_before, torch.get_rng_state())
    with pytest.raises(FileExistsError):
        restored.save_checkpoint(path)
    with pytest.raises(ValueError, match="level"):
        NeuralMCTSTrainer.load_checkpoint(path, tiny_level(9))


def test_real_natural_loss_assignment():
    trainer = NeuralMCTSTrainer(tiny_config(), tiny_level(65535))
    episode = trainer.collect_episode()
    assert episode.outcome == "LOST" and len(episode.examples) == 1
    assert episode.examples[0].outcome == -1


def test_metrics_checkpoint_paths_and_failure_exclusion(
    controlled, monkeypatch, tmp_path
):
    monkeypatch.setattr(controlled, "mode", "ERROR")
    config = replace(tiny_config(), checkpoint_dir=str(tmp_path / "checkpoints"))
    trainer = NeuralMCTSTrainer(config, tiny_level())
    metrics = trainer.run()
    assert len(trainer.replay) == trainer.updates == 0
    assert all(row["total_loss"] is None for row in metrics)
    assert metrics[0]["episodes"][0]["error"].startswith("RuntimeError")
    assert (tmp_path / "checkpoints/round_0002.pt").exists()
    with pytest.raises(FileExistsError):
        NeuralMCTSTrainer(config, tiny_level()).run()


def test_nonfinite_gradients_rejected_without_parameter_update():
    trainer = NeuralMCTSTrainer(tiny_config(), tiny_level())
    trainer.replay.extend([example()])
    parameter = next(trainer.network.parameters())
    before = parameter.detach().clone()
    hook = parameter.register_hook(lambda grad: torch.full_like(grad, float("nan")))
    try:
        with pytest.raises(ValueError, match="gradients"):
            trainer.update()
        torch.testing.assert_close(parameter, before, rtol=0, atol=0)
        assert trainer.updates == 0
    finally:
        hook.remove()


@pytest.mark.parametrize(
    "field,value",
    [
        ("collection_rounds", 0),
        ("episodes_per_round", True),
        ("simulations", 1.5),
        ("learning_rate", 0),
        ("learning_rate", float("nan")),
        ("value_loss_weight", -1),
        ("replay_capacity", 1),
        ("seed", True),
        ("environment_seed_start", 400000),
        ("dirichlet_epsilon", 2),
        ("checkpoint_dir", ""),
    ],
)
def test_invalid_config(field, value):
    with pytest.raises(ValueError):
        NeuralMCTSTrainingConfig(**{field: value})


def test_smoke_defaults_prepared_not_executed():
    config = NeuralMCTSTrainingConfig()
    assert (
        config.collection_rounds,
        config.episodes_per_round,
        config.simulations,
        config.replay_capacity,
        config.batch_size,
        config.updates_per_round,
    ) == (2, 4, 8, 256, 16, 10)
    assert "search_depth_cap" not in asdict(config)


def test_existing_neural_agents_remain_unchanged(controlled, monkeypatch):
    from match3_ai.agents.dqn_agent import DQNAgent
    from match3_ai.agents.ppo_agent import PPOAgent

    monkeypatch.setattr(controlled, "mode", "WON")
    dqn, ppo = DQNAgent(hidden_dims=(8,)), PPOAgent(hidden_dims=(8,))
    before_dqn = deepcopy(dqn.online.state_dict())
    before_ppo = deepcopy(ppo.network.state_dict())
    trainer = NeuralMCTSTrainer(tiny_config(), tiny_level())
    trainer.run()
    assert_nested_equal(before_dqn, dqn.online.state_dict())
    assert_nested_equal(before_ppo, ppo.network.state_dict())


def test_checkpoint_boundary_and_corrupt_metadata_rejection(tmp_path):
    trainer = NeuralMCTSTrainer(tiny_config(), tiny_level())
    trainer._at_boundary = False
    with pytest.raises(RuntimeError, match="boundary"):
        trainer.save_checkpoint(tmp_path / "invalid.pt")
    trainer._at_boundary = True
    trainer.save_checkpoint(tmp_path / "initial.pt")
    payload = torch.load(tmp_path / "initial.pt", weights_only=True)
    payload["encoder_version"] = 999
    torch.save(payload, tmp_path / "corrupt.pt")
    with pytest.raises(ValueError, match="unsupported"):
        NeuralMCTSTrainer.load_checkpoint(tmp_path / "corrupt.pt", tiny_level())


def test_invalid_metadata_and_nonfinite_loss_rejected():
    data = deepcopy(vars(example()))
    data["provenance"] = {"invalid": float("nan")}
    with pytest.raises(ValueError, match="finite JSON"):
        SearchExample(**data)
    mask = torch.ones(1, 144, dtype=torch.bool)
    policy = torch.full((1, 144), 1 / 144)
    with pytest.raises(ValueError, match="finite"):
        policy_value_loss(
            torch.zeros(1, 144),
            torch.tensor([float("nan")]),
            mask,
            policy,
            torch.ones(1),
        )


def test_explicit_no_legal_actions_terminal_is_a_natural_loss(controlled, monkeypatch):
    monkeypatch.setattr(controlled, "mode", "LOST")
    original_step = controlled.step

    def dead_board_step(self, action):
        observation, reward, terminated, truncated, info = original_step(self, action)
        info.update(level_status="IN_PROGRESS", termination_reason="no_legal_actions")
        return observation, reward, terminated, truncated, info

    monkeypatch.setattr(controlled, "step", dead_board_step)
    result = NeuralMCTSTrainer(tiny_config(), tiny_level()).collect_episode()
    assert result.outcome == "LOST" and result.examples[0].outcome == -1
    assert result.examples[0].provenance["termination_reason"] == "no_legal_actions"
