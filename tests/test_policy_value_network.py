from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from match3_ai.agents.policy_value_encoder import (
    PolicyValueContext,
    PolicyValueEncoder,
)
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.config import LevelConfig, LevelObjective, ObjectiveType

torch = pytest.importorskip("torch")

from match3_ai.agents.policy_value_network import (
    PolicyValueNetwork,
    masked_policy_logits,
)


@pytest.fixture
def task():
    config = LevelConfig(
        rows=3,
        cols=3,
        num_colors=3,
        move_limit=20,
        objectives=(LevelObjective(ObjectiveType.COLLECT_TILES, target=10, color=2),),
    )
    observation = {
        key: np.zeros((9, 8), dtype=np.uint8)
        for key in ("tiles", "specials", "blockers", "blocker_hp", "cell_mask")
    }
    observation["tiles"][:3, :3] = [[1, 2, 1], [2, 1, 3], [3, 3, 2]]
    observation["cell_mask"][:3, :3] = 1
    observation.update(
        board_shape=np.array([3, 3]),
        moves_remaining=12,
        objective_type=np.array([2, 0]),
        objective_color=np.array([2, 0]),
        objective_target=np.array([10, 0]),
        objective_progress=np.array([4, 0]),
        objective_mask=np.array([1, 0]),
        action_mask=np.zeros(144, dtype=bool),
    )
    observation["action_mask"][[0, 1, 32]] = True
    return observation, config


def tensors(observation, config):
    encoded = PolicyValueEncoder().encode(
        observation, PolicyValueContext.from_level_config(config)
    )
    return tuple(torch.from_numpy(array[None]) for array in encoded)


def test_encoding_contract_and_no_mutation(task):
    observation, config = task
    original, original_config = deepcopy(observation), asdict(config)
    encoded = PolicyValueEncoder().encode(
        observation, PolicyValueContext.from_level_config(config)
    )
    assert PolicyValueEncoder.version == 1
    assert encoded.spatial.shape == (20, 9, 8)
    assert encoded.global_features.shape == (30,)
    assert encoded.action_mask.shape == (144,)
    assert encoded.spatial.dtype == encoded.global_features.dtype == np.float32
    assert encoded.action_mask.dtype == np.bool_
    np.testing.assert_allclose(
        encoded.global_features[:6], [3 / 9, 3 / 8, 12 / 32, 20 / 32, 12 / 20, 3 / 5]
    )
    assert encoded.global_features[6 + 2] == 1
    assert encoded.global_features[6 + 3 + 2] == 1
    assert encoded.global_features[6 + 10] == pytest.approx(0.4)
    assert not encoded.spatial[:, 3:, :].any()
    assert not encoded.spatial[:, :, 3:].any()
    for key, value in original.items():
        np.testing.assert_array_equal(observation[key], value)
    assert asdict(config) == original_config
    encoded.spatial[:] = 0
    encoded.action_mask[:] = False
    for key, value in original.items():
        np.testing.assert_array_equal(observation[key], value)


def test_spatial_categories_and_holes(task):
    observation, config = task
    observation["tiles"][1, 1] = 255
    observation["specials"][1, 1] = 5
    observation["blockers"][1, 1] = 2
    observation["blocker_hp"][1, 1] = 2
    observation["tiles"][2, 2] = 0
    observation["cell_mask"][2, 2] = 0
    observation["tiles"][0, 2] = 0
    observation["blockers"][0, 2] = 3
    observation["blocker_hp"][0, 2] = 1
    spatial, _, _ = PolicyValueEncoder().encode(
        observation, PolicyValueContext.from_level_config(config)
    )
    assert spatial[6, 1, 1] == spatial[12, 1, 1] == spatial[15, 1, 1] == 1
    assert spatial[17, 1, 1] == pytest.approx(2 / 3)
    assert spatial[0, 0, 2] == spatial[16, 0, 2] == 1
    assert not spatial[:19, 2, 2].any()
    assert spatial[19, 2, 2] == 1


def test_exact_action_ordering(task):
    maps = torch.empty(2, 2, 9, 8)
    for row in range(9):
        for col in range(8):
            for direction in range(2):
                action = Match3LevelEnv.encode_action(row, col, direction)
                maps[0, direction, row, col] = action
                maps[1, direction, row, col] = 1000 + action
    logits = PolicyValueNetwork.flatten_policy(maps)
    torch.testing.assert_close(logits[0], torch.arange(144, dtype=torch.float32))
    torch.testing.assert_close(logits[1], torch.arange(144, dtype=torch.float32) + 1000)
    assert not torch.equal(maps.flatten(1), logits)
    spatial, globals_, _ = tensors(*task)
    model = PolicyValueNetwork()
    handle = model.policy_head.register_forward_hook(lambda module, args, output: maps)
    try:
        actual, _ = model(spatial.repeat(2, 1, 1, 1), globals_.repeat(2, 1))
        torch.testing.assert_close(actual, logits)
    finally:
        handle.remove()


def test_shapes_range_gradients_and_safe_masking(task):
    observation, config = task
    spatial, globals_, mask = tensors(observation, config)
    model = PolicyValueNetwork(seed=42)
    raw, values = model(spatial.repeat(2, 1, 1, 1), globals_.repeat(2, 1))
    assert raw.shape == (2, 144) and values.shape == (2,)
    assert torch.isfinite(raw).all() and torch.isfinite(values).all()
    assert ((values >= -1) & (values <= 1)).all()
    logits, value = model(spatial, globals_, mask)
    probs = logits.softmax(dim=-1)
    assert (probs[~mask] == 0).all()
    torch.testing.assert_close(probs.sum(dim=-1), torch.ones(1))
    loss = -logits.log_softmax(dim=-1)[0, 0] + (value - 1).square().mean()
    loss.backward()
    assert all(
        p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()
    )
    assert any(p.grad.abs().sum() > 0 for p in model.policy_head.parameters())
    assert any(p.grad.abs().sum() > 0 for p in model.value_head.parameters())


def test_mask_excludes_larger_logits_and_preserves_inputs():
    logits = torch.full((1, 144), 1000.0, requires_grad=True)
    with torch.no_grad():
        logits[0, 3] = -1000
    mask = torch.zeros(1, 144, dtype=torch.bool)
    mask[0, 3] = True
    before = logits.detach().clone()
    probs = masked_policy_logits(logits + 100, mask).softmax(-1)
    assert probs[0, 3] == 1 and probs.sum() == 1
    torch.testing.assert_close(logits, before)


def test_terminal_empty_mask_is_explicit(task):
    observation, config = task
    observation["action_mask"][:] = False
    spatial, globals_, mask = tensors(observation, config)
    assert not mask.any()
    model = PolicyValueNetwork()
    assert torch.isfinite(model(spatial, globals_)[1]).all()
    with pytest.raises(ValueError, match="at least one legal"):
        model(spatial, globals_, mask)
    mixed = torch.ones(2, 144, dtype=torch.bool)
    mixed[1] = False
    with pytest.raises(ValueError, match="at least one legal"):
        masked_policy_logits(torch.zeros(2, 144), mixed)


def test_context_affects_both_network_heads(task):
    observation, config = task
    spatial, globals_, _ = tensors(observation, config)
    model = PolicyValueNetwork(seed=42)
    baseline = model(spatial, globals_)
    for key, value in (
        ("objective_target", 20),
        ("objective_progress", 7),
        ("objective_color", 1),
        ("objective_type", 1),
    ):
        changed = deepcopy(observation)
        changed[key][0] = value
        if key == "objective_type":
            changed["objective_color"][0] = 0
        _, altered, _ = tensors(changed, config)
        assert not torch.equal(altered, globals_)
        outputs = model(spatial, altered)
        assert not torch.equal(outputs[0], baseline[0])
        assert not torch.equal(outputs[1], baseline[1])


def test_static_move_limit_and_palette_are_encoded(task):
    observation, _ = task
    for moves, colors in ((20, 3), (32, 3), (20, 5)):
        context = PolicyValueContext(3, 3, colors, moves)
        encoded = PolicyValueEncoder().encode(observation, context)
        assert encoded.global_features[3] == pytest.approx(moves / 32)
        assert encoded.global_features[4] == pytest.approx(12 / moves)
        assert encoded.global_features[5] == pytest.approx(colors / 5)
    changed = deepcopy(observation)
    changed["moves_remaining"] = 32
    assert (
        PolicyValueEncoder()
        .encode(changed, PolicyValueContext(3, 3, 3, 32))
        .global_features[2]
        == 1
    )


def test_maximum_board_two_objectives_and_terminal_moves(task):
    observation, _ = task
    observation["board_shape"] = np.array([9, 8])
    observation["cell_mask"][:] = 1
    observation["tiles"][:] = 5
    observation["moves_remaining"] = 0
    observation["objective_mask"][:] = 1
    observation["objective_type"] = np.array([1, 2])
    observation["objective_color"] = np.array([0, 5])
    observation["objective_target"] = np.array([3, 65535])
    observation["objective_progress"] = np.array([3, 32768])
    encoded = PolicyValueEncoder().encode(observation, PolicyValueContext(9, 8, 5, 32))
    assert encoded.spatial[5].all()
    assert encoded.global_features[2] == encoded.global_features[4] == 0
    assert encoded.global_features[6 + 10] == 1
    assert encoded.global_features[18 + 9] == 1
    assert encoded.global_features[18 + 10] == pytest.approx(32768 / 65535)


def test_identity_rng_and_future_metadata_are_excluded(task):
    observation, config = task
    encoder = PolicyValueEncoder()
    first = encoder.encode(observation, PolicyValueContext.from_level_config(config))
    observation.update(level_id="other", rng_state={"seed": 99}, future_outcome=1)
    changed = replace(config, level_id="other", name="Other", seed=99)
    second = encoder.encode(observation, PolicyValueContext.from_level_config(changed))
    for before, after in zip(first, second):
        np.testing.assert_array_equal(before, after)


@pytest.mark.parametrize(
    "field,value",
    [
        ("rows", 10),
        ("cols", 9),
        ("num_colors", 6),
        ("num_colors", 2),
        ("move_limit", 33),
        ("move_limit", 0),
        ("move_limit", True),
        ("rows", 3.0),
    ],
)
def test_unsupported_static_context(field, value):
    args = {"rows": 3, "cols": 3, "num_colors": 3, "move_limit": 20}
    args[field] = value
    with pytest.raises(ValueError, match=field):
        PolicyValueContext(**args)


def test_unsupported_level_objectives(task):
    _, config = task
    for objectives in (
        (),
        config.objectives * 3,
        (replace(config.objectives[0], target=65536),),
    ):
        with pytest.raises(ValueError, match="objective"):
            PolicyValueContext.from_level_config(replace(config, objectives=objectives))


@pytest.mark.parametrize(
    "key,value",
    [
        ("moves_remaining", 21),
        ("board_shape", [4, 3]),
        ("objective_type", [3, 0]),
        ("objective_target", [0, 0]),
        ("objective_target", [65536, 0]),
        ("objective_color", [4, 0]),
        ("objective_mask", [1, 2]),
        ("action_mask", [0] * 143),
    ],
)
def test_invalid_observation_fields(task, key, value):
    observation, config = task
    observation[key] = np.asarray(value)
    with pytest.raises(ValueError):
        PolicyValueEncoder().encode(
            observation, PolicyValueContext.from_level_config(config)
        )


def test_invalid_palette_hp_geometry_and_missing_fields(task):
    observation, config = task
    context = PolicyValueContext.from_level_config(config)
    for key, value in (
        ("tiles", 4),
        ("specials", 6),
        ("blockers", 4),
        ("blocker_hp", 4),
    ):
        changed = deepcopy(observation)
        changed[key][0, 0] = value
        with pytest.raises(ValueError):
            PolicyValueEncoder().encode(changed, context)
    observation["action_mask"][143] = True
    with pytest.raises(ValueError, match="endpoints"):
        PolicyValueEncoder().encode(observation, context)
    del observation["tiles"]
    with pytest.raises(ValueError, match="missing"):
        PolicyValueEncoder().encode(observation, context)


def test_seeded_models_are_deterministic_without_rng_or_input_mutation(task):
    spatial, globals_, mask = tensors(*task)
    before = torch.get_rng_state().clone()
    first, second = PolicyValueNetwork(42), PolicyValueNetwork(42)
    assert torch.equal(before, torch.get_rng_state())
    spatial_before, globals_before, mask_before = (
        spatial.clone(),
        globals_.clone(),
        mask.clone(),
    )
    with torch.inference_mode():
        expected = first(spatial, globals_, mask)
        for actual in (first(spatial, globals_, mask), second(spatial, globals_, mask)):
            for a, b in zip(expected, actual):
                torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(spatial, spatial_before)
    torch.testing.assert_close(globals_, globals_before)
    assert torch.equal(mask, mask_before)


@pytest.mark.parametrize("seed", [-1, True, 1.5])
def test_invalid_model_seed(seed):
    with pytest.raises(ValueError, match="seed"):
        PolicyValueNetwork(seed)


def test_bad_network_inputs():
    model = PolicyValueNetwork()
    with pytest.raises(ValueError, match="spatial"):
        model(torch.zeros(1, 19, 9, 8), torch.zeros(1, 30))
    with pytest.raises(ValueError, match="global"):
        model(torch.zeros(1, 20, 9, 8), torch.zeros(1, 29))
    with pytest.raises(ValueError, match="finite"):
        model(torch.full((1, 20, 9, 8), float("nan")), torch.zeros(1, 30))
    with pytest.raises(ValueError, match="boolean"):
        masked_policy_logits(torch.zeros(1, 144), torch.ones(1, 144))
    with pytest.raises(ValueError, match="finite"):
        masked_policy_logits(
            torch.full((1, 144), float("inf")), torch.ones(1, 144, dtype=torch.bool)
        )
