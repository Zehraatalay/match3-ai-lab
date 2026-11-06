from copy import deepcopy

import numpy as np
import pytest

from match3_ai.agents.observation_encoder import ObservationEncoder
from match3_ai.agents.replay_buffer import ReplayBuffer
from match3_ai.games.match3.environment import Match3LevelEnv


@pytest.fixture
def observation():
    return Match3LevelEnv().reset(seed=71)[0]


@pytest.mark.parametrize(
    "level",
    [
        "first-match",
        "wooden-trouble",
        "double-trouble",
        "frozen-garden",
        "stone-breaker",
        "blocker-mayhem",
    ],
)
def test_encoder_supports_all_levels_and_separate_masks(level):
    observation, _ = Match3LevelEnv(level).reset(seed=71)
    before = deepcopy(observation)
    features, mask = ObservationEncoder().encode(observation)
    assert features.shape == (1467,) and features.dtype == np.float32
    assert mask.shape == (144,) and mask.dtype == np.bool_
    assert np.isfinite(features).all()
    for key in before:
        np.testing.assert_array_equal(observation[key], before[key])
    observation["action_mask"][:] = 0
    after, _ = ObservationEncoder().encode(observation)
    np.testing.assert_array_equal(features, after)
    assert mask.any()


def test_categorical_planes_colorless_and_stationary_blocker_features(observation):
    observation["tiles"][0, 0] = 255
    observation["specials"][0, 0] = 5
    observation["blockers"][0, 1] = 2
    observation["blocker_hp"][0, 1] = 2
    features, _ = ObservationEncoder().encode(observation)
    planes = features[:1440].reshape(20, 9, 8)
    assert planes[6, 0, 0] == planes[12, 0, 0] == 1
    assert planes[:7, 0, 0].sum() == 1
    assert planes[15, 0, 1] == 1
    assert planes[17, 0, 1] == pytest.approx(2 / 3)
    changed = deepcopy(observation)
    changed["tiles"][0, 0] = 1
    other, _ = ObservationEncoder().encode(changed)
    assert other[72] == 1 and features[72] == 0
    assert other[6 * 72] == 0 and features[6 * 72] == 1


def test_padding_masked_holes_and_objective_targets_are_encoded(observation):
    encoder = ObservationEncoder()
    features, _ = encoder.encode(observation)
    planes = features[:1440].reshape(20, 9, 8)
    assert not planes[:, 6:, :].any()
    masked, _ = Match3LevelEnv("blocker-mayhem").reset(seed=71)
    positions = np.argwhere(masked["cell_mask"] == 0)
    row, col = positions[0]
    masked["tiles"][row, col] = 3
    masked_features, _ = encoder.encode(masked)
    masked_planes = masked_features[:1440].reshape(20, 9, 8)
    assert not masked_planes[:19, row, col].any()
    assert masked_planes[19, row, col] == 1
    changed = deepcopy(observation)
    changed["objective_target"][0] = 100
    assert not np.array_equal(encoder.encode(changed)[0], features)
    changed["tiles"][0, 0] = 7
    with pytest.raises(ValueError, match="unknown tile"):
        encoder.encode(changed)


def add_transition(buffer, observation, reward=1.0, *, terminal=False):
    action = int(np.flatnonzero(observation["action_mask"])[0])
    after = deepcopy(observation)
    if terminal:
        after["action_mask"][:] = 0
    buffer.add(observation, action, reward, after, terminal, False)


def test_replay_ring_capacity_dtypes_and_copy_isolation(observation):
    replay = ReplayBuffer(3, seed=42)
    for reward in range(5):
        add_transition(replay, observation, reward, terminal=True)
    assert len(replay) == 3
    original_tile = int(observation["tiles"][0, 0])
    observation["tiles"][0, 0] = 0
    batch = replay.sample(3)
    assert set(batch.rewards) == {2, 3, 4}
    assert np.all(batch.observations["tiles"][:, 0, 0] == original_tile)
    assert batch.observations["tiles"].dtype == np.uint8
    assert batch.observations["objective_target"].dtype == np.uint16
    assert batch.actions.dtype == np.int64 and batch.rewards.dtype == np.float32
    assert batch.terminated.all() and not batch.next_masks.any()
    batch.observations["tiles"][:] = 0
    assert replay.observations["tiles"].any()


def test_replay_sampling_and_state_restore_are_reproducible(observation):
    first, second = ReplayBuffer(8, 12), ReplayBuffer(8, 12)
    for index in range(8):
        add_transition(first, observation, index)
        add_transition(second, observation, index)
    np.testing.assert_array_equal(first.sample(4).rewards, second.sample(4).rewards)
    restored = ReplayBuffer(8, 999)
    restored.load_state_dict(first.state_dict())
    np.testing.assert_array_equal(first.sample(4).rewards, restored.sample(4).rewards)
    with pytest.raises(ValueError):
        first.sample(9)


def test_replay_rejects_nonterminal_empty_next_mask(observation):
    replay = ReplayBuffer(2)
    action = int(np.flatnonzero(observation["action_mask"])[0])
    after = deepcopy(observation)
    after["action_mask"][:] = 0
    with pytest.raises(ValueError, match="nonterminal"):
        replay.add(observation, action, 0, after, False, True)
    with pytest.raises(ValueError, match="finite"):
        replay.add(observation, action, float("nan"), observation, False, False)


def test_q_network_shape_seed_and_global_torch_rng_isolation():
    torch = pytest.importorskip("torch")
    from match3_ai.agents.dqn_agent import QNetwork

    before = torch.get_rng_state().clone()
    first, second = QNetwork(seed=3), QNetwork(seed=3)
    assert torch.equal(before, torch.get_rng_state())
    assert first(torch.zeros(4, 1467)).shape == (4, 144)
    assert sum(parameter.numel() for parameter in first.parameters()) == 427280
    assert all(
        torch.equal(a, b) for a, b in zip(first.parameters(), second.parameters())
    )


def test_masked_exploitation_exploration_and_empty_masks(observation):
    torch = pytest.importorskip("torch")
    from match3_ai.agents.dqn_agent import DQNAgent

    first, second = (
        DQNAgent(seed=3, hidden_dims=(8,)),
        DQNAgent(seed=3, hidden_dims=(8,)),
    )
    observation["action_mask"][:] = 0
    observation["action_mask"][[5, 7]] = 1
    with torch.no_grad():
        for parameter in first.online.parameters():
            parameter.zero_()
        first.online.layers[-1].bias[0] = 100000
        first.online.layers[-1].bias[5] = 2
        first.online.layers[-1].bias[7] = 3
    rng_state = deepcopy(first.rng.bit_generator.state)
    assert first.select_action(observation) == 7
    assert first.rng.bit_generator.state == rng_state
    sequence = [first.select_action(observation, epsilon=1) for _ in range(200)]
    assert sequence == [
        second.select_action(observation, epsilon=1) for _ in range(200)
    ]
    assert 60 < sequence.count(5) < 140
    assert set(sequence) == {5, 7}
    observation["action_mask"][:] = 0
    with pytest.raises(RuntimeError, match="no valid"):
        first.select_action(observation)


def test_bellman_masks_terminal_and_truncated_rows_without_nan():
    torch = pytest.importorskip("torch")
    from match3_ai.agents.dqn_agent import masked_bellman_targets

    q = torch.zeros(4, 144)
    q[:, 0], q[:, 1] = 100000, 2
    q[1] = torch.nan
    masks = torch.zeros(4, 144, dtype=torch.bool)
    masks[[0, 2], 1] = True
    targets = masked_bellman_targets(
        torch.tensor([1.0, 2.0, 3.0, 4.0]),
        q,
        torch.tensor([False, True, False, True]),
        torch.tensor([False, False, True, True]),
        masks,
        gamma=0.5,
    )
    torch.testing.assert_close(targets, torch.tensor([2.0, 2.0, 4.0, 4.0]))
    assert not targets.requires_grad and torch.isfinite(targets).all()
    with pytest.raises(ValueError, match="empty"):
        masked_bellman_targets(
            torch.zeros(1),
            q[:1],
            torch.tensor([False]),
            torch.tensor([True]),
            torch.zeros(1, 144, dtype=torch.bool),
        )


def test_updates_target_sync_and_checkpoint_round_trip(observation, tmp_path):
    torch = pytest.importorskip("torch")
    from match3_ai.agents.dqn_agent import DQNAgent

    agent = DQNAgent(4, hidden_dims=(8,))
    replay = ReplayBuffer(4)
    for _ in range(4):
        add_transition(replay, observation, terminal=True)
    initial = [parameter.detach().clone() for parameter in agent.online.parameters()]
    metrics = agent.update(replay.sample(4))
    assert all(np.isfinite(value) for value in metrics.values())
    assert any(
        not torch.equal(a, b) for a, b in zip(initial, agent.online.parameters())
    )
    assert all(torch.equal(a, b) for a, b in zip(initial, agent.target.parameters()))
    agent.synchronize_target()
    assert all(
        torch.equal(a, b)
        for a, b in zip(agent.online.parameters(), agent.target.parameters())
    )
    path = tmp_path / "agent.pt"
    agent.save_checkpoint(path, training_config={"gamma": 0.99}, training_step=10)
    loaded, metadata = DQNAgent.load_checkpoint(path)
    assert metadata["training_step"] == 10
    assert metadata["training_config"] == {"gamma": 0.99}
    assert loaded.select_action(observation) == agent.select_action(observation)
    assert loaded.select_action(observation, 1) == agent.select_action(observation, 1)
    assert len(loaded.optimizer.state) == len(agent.optimizer.state)
    with pytest.raises(FileExistsError):
        agent.save_checkpoint(path)
