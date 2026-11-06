import json
from copy import deepcopy

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from match3_ai.agents.dqn_agent import DQNAgent
from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.evaluation.behavior_analysis import (
    StateObserver,
    checkpoint_config_matches,
    run_analysis,
    verify_step_zero,
)
from match3_ai.evaluation.behavior_data import (
    ObservationStore,
    clone_environment,
    complete_state_fingerprint,
    counterfactual_outcomes,
    greedy_scores,
    observation_fingerprint,
    restore_by_prefix,
)
from match3_ai.evaluation.behavior_statistics import (
    action_preferences,
    clustered_agreement,
    expected_random_agreement,
    export_preferences,
    holm_adjust,
    paired_difference,
    paired_outcomes,
    reward_distribution,
)
from match3_ai.games.match3.environment import Match3LevelEnv


@pytest.fixture
def environment():
    env = Match3LevelEnv("blocker-mayhem")
    env.reset(seed=300000)
    return env


def test_checkpoint_compatibility_and_true_step_zero(tmp_path):
    assert checkpoint_config_matches(
        {"hidden_dims": (256, 128), "gamma": 0.99},
        {"hidden_dims": [256, 128], "gamma": 0.99},
    )
    assert not checkpoint_config_matches({"gamma": 0.99}, {"gamma": 0.9})
    agent = DQNAgent(5901, hidden_dims=(8,))
    agent.save_checkpoint(tmp_path / "zero.pt", training_step=0)
    loaded, payload = DQNAgent.load_checkpoint(tmp_path / "zero.pt")
    verify_step_zero(loaded, payload)
    with torch.no_grad():
        next(loaded.online.parameters()).add_(0.01)
    with pytest.raises(ValueError, match="reproducible"):
        verify_step_zero(loaded, payload)
    payload["training_step"] = 1
    with pytest.raises(ValueError, match="step-zero"):
        verify_step_zero(agent, payload)


def test_fingerprint_deduplication_and_safe_typed_export(environment, tmp_path):
    observation = environment._observation()
    original = observation_fingerprint(observation)
    store = ObservationStore()
    assert store.intern(observation) == store.intern(deepcopy(observation)) == 0
    changed = deepcopy(observation)
    changed["objective_target"][1] += 1
    assert store.intern(changed) == 1
    changed["action_mask"][:] = 0
    assert observation_fingerprint(changed) != store.fingerprints[1]
    observation["blocker_hp"][:] = 0
    assert store.fingerprints[0] == original
    store.export(tmp_path / "states.npz")
    with np.load(tmp_path / "states.npz", allow_pickle=False) as data:
        assert data["tiles"].shape == (2, 9, 8)
        assert data["blocker_hp"][0].sum() == 9
        assert data["action_mask"].dtype == np.bool_
        assert data["objective_target"].dtype == np.uint16
    with pytest.raises(FileExistsError):
        store.export(tmp_path / "states.npz")


def test_complete_clone_rng_independence_and_prefix_restoration(environment):
    first = int(np.flatnonzero(environment._observation()["action_mask"])[0])
    environment.step(first)
    before = complete_state_fingerprint(environment)
    clone = clone_environment(environment)
    assert clone.session.engine.rng is not environment.session.engine.rng
    for key in ("tiles", "specials", "blockers", "blocker_hp", "cell_mask"):
        assert not np.shares_memory(
            getattr(clone.session.board, key), getattr(environment.session.board, key)
        )
    restored = restore_by_prefix("blocker-mayhem", 300000, [first], before)
    action = int(np.flatnonzero(environment._observation()["action_mask"])[0])
    a, b = clone.step(action), restored.step(action)
    assert observation_fingerprint(a[0]) == observation_fingerprint(b[0])
    assert a[1:] == b[1:]
    assert complete_state_fingerprint(clone) == complete_state_fingerprint(restored)
    assert complete_state_fingerprint(environment) == before
    assert json.dumps(clone.session.engine.rng.bit_generator.state) != json.dumps(
        environment.session.engine.rng.bit_generator.state
    )
    with pytest.raises(ValueError, match="verification"):
        restore_by_prefix("blocker-mayhem", 300000, [], before)


def test_exact_counterfactuals_reproduce_and_preserve_original(environment):
    observation = environment._observation()
    actions = [int(a) for a in np.flatnonzero(observation["action_mask"])[:2]]
    before = complete_state_fingerprint(environment)
    left = counterfactual_outcomes(environment, actions + actions)
    right = counterfactual_outcomes(environment, actions)
    assert len(left) == len(set(actions)) and left == right
    assert complete_state_fingerprint(environment) == before
    actual = clone_environment(environment).step(actions[0])
    assert observation_fingerprint(actual[0]) == left[actions[0]]["after_fingerprint"]
    assert actual[4]["blocker_damage"]["hits"] == left[actions[0]]["blocker_hits"]


def test_shared_decisions_agreement_and_source_provenance(environment):
    models = {
        f"DQN{master}_{stage}": DQNAgent(master + (stage == "final"), hidden_dims=(8,))
        for master in (5901, 5902, 5903)
        for stage in ("zero", "final")
    }
    store, cache, occurrences, counterfactuals = ObservationStore(), {}, [], []
    before = complete_state_fingerprint(environment)
    for source in ("GreedyAgent", "DQN5901_final"):
        policy = GreedyAgent() if source == "GreedyAgent" else models[source]
        observer = StateObserver(
            environment,
            policy,
            source,
            300000,
            source,
            store,
            models,
            cache,
            occurrences,
            counterfactuals,
        )
        observer.select_action(environment._observation())
    assert len(store.observations) == 1 and len(occurrences) == 2
    assert {row["source_policy"] for row in occurrences} == {
        "GreedyAgent",
        "DQN5901_final",
    }
    for row in occurrences:
        for master in (5901, 5902, 5903):
            assert row[f"agree_greedy_{master}"] == (
                row["actions"][f"DQN{master}_final"] == row["actions"]["GreedyAgent"]
            )
            assert row[f"agree_zero_{master}"] == (
                row["actions"][f"DQN{master}_final"]
                == row["actions"][f"DQN{master}_zero"]
            )
        assert row["expected_random_agreement"] == 1 / row["legal_actions"]
    assert complete_state_fingerprint(environment) == before


def test_no_reserved_seed_leakage(environment, tmp_path):
    with pytest.raises(ValueError, match="validation"):
        restore_by_prefix(
            "blocker-mayhem", 400000, [], complete_state_fingerprint(environment)
        )
    with pytest.raises(ValueError):
        run_analysis(tmp_path, tmp_path / "out", episodes=1001)
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(FileExistsError):
        run_analysis(tmp_path, existing)


def test_greedy_scores_reproduce_exact_tie_breaking(environment):
    observation = environment._observation()
    scores = greedy_scores(observation)
    best = max(scores.values())
    assert GreedyAgent().select_action(observation) == min(
        action for action, score in scores.items() if score == best
    )
    assert expected_random_agreement(observation["action_mask"]) == 1 / len(scores)
    assert expected_random_agreement(np.array([0, 1, 0])) == 1
    with pytest.raises(ValueError):
        expected_random_agreement(np.zeros(144))


def test_action_preferences_use_analytic_masked_uniform_not_sampled_actions(tmp_path):
    masks = np.zeros((2, 144), dtype=bool)
    masks[0, [0, 1]] = True
    masks[1, [1, 2]] = True
    rows = [
        {
            "source_policy": "RandomAgent",
            "state_id": i,
            "greedy_best_ties": i + 1,
            "actions": {"GreedyAgent": i, "DQN5901_final": 1},
        }
        for i in range(2)
    ]
    result = action_preferences(rows, masks)
    np.testing.assert_allclose(
        result["expected_uniform_action_probabilities"][:3], [0.25, 0.5, 0.25]
    )
    assert result["expected_random_tie_optimal"] == 0.75
    assert (
        result["policies"]["DQN5901_final"][
            "total_variation_from_uniform_mask_marginal"
        ]
        == 0.5
    )
    assert result["policies"]["DQN5901_final"]["marginal_entropy_bits"] == 0
    (tmp_path / "state_occurrences.jsonl").write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n"
    )
    np.savez_compressed(tmp_path / "observations.npz", action_mask=masks)
    exported = export_preferences(tmp_path, tmp_path / "preferences.json")
    assert exported["Pooled"] == result
    with pytest.raises(FileExistsError):
        export_preferences(tmp_path, tmp_path / "preferences.json")


def test_paired_outcomes_and_exact_statistical_calculations():
    result = paired_outcomes(
        {0: "WON", 1: "WON", 2: "LOST", 3: "LOST"},
        {0: "WON", 1: "LOST", 2: "WON", 3: "LOST"},
        samples=100,
    )
    assert result["counts"] == {
        "both_win": 1,
        "left_only_win": 1,
        "right_only_win": 1,
        "both_lose": 1,
    }
    assert result["difference"]["absolute_difference"] == 0
    assert result["mcnemar_exact_p"] == 1
    stronger = paired_outcomes(
        {i: "WON" for i in range(10)}, {i: "LOST" for i in range(10)}, samples=100
    )
    assert stronger["mcnemar_exact_p"] == pytest.approx(2 / 1024)
    assert stronger["difference"]["ci95"] == [1, 1]
    np.testing.assert_allclose(holm_adjust([0.01, 0.04, 0.03]), [0.03, 0.06, 0.06])
    with pytest.raises(ValueError):
        paired_outcomes({0: "WON"}, {1: "WON"})


def test_paired_bootstrap_is_reproducible_and_relative_zero_is_undefined():
    a = paired_difference([1, 3, 5], [0, 1, 2], seed=17, samples=500)
    assert a == paired_difference([1, 3, 5], [0, 1, 2], seed=17, samples=500)
    assert a["absolute_difference"] == 2
    assert paired_difference([1, 2], [0, 0], samples=100)["relative_change"] is None
    assert paired_difference([0, 0], [-1, -1], samples=100)["relative_change"] is None


def test_cluster_bootstrap_retains_episode_dependence_and_source_weights():
    rows = [
        {
            "environment_seed": seed,
            "source_policy": "RandomAgent",
            "agree": float(seed == 1),
        }
        for seed in (1, 2)
        for _ in range(100)
    ]
    result = clustered_agreement(rows, "agree", samples=500)
    assert result["rate"] == 0.5 and result["ci95"] == [0, 1]
    assert result == clustered_agreement(rows, "agree", samples=500)
    duplicate = clustered_agreement(rows * 2, "agree", samples=500)
    assert duplicate["rate"] == result["rate"] and duplicate["ci95"] == result["ci95"]
    mixture = [
        {"source_policy": name, "environment_seed": 1, "agree": value}
        for name, count, value in (
            ("RandomAgent", 10, 1),
            ("GreedyAgent", 100, 0),
            ("DQN5901_final", 1, 0),
        )
        for _ in range(count)
    ]
    assert clustered_agreement(mixture, "agree", samples=100)["rate"] == pytest.approx(
        1 / 3
    )


def test_reward_sparsity_uses_real_terminal_boundaries():
    episodes = [
        {"episode_id": "a", "steps": 3, "total_reward": 1, "outcome": "WON"},
        {"episode_id": "b", "steps": 2, "total_reward": -1, "outcome": "LOST"},
    ]
    moves = [
        {
            "episode_id": name,
            "move_index": index + 1,
            "reward": reward,
            "objective_progress": [{"progress": (index + 1) * 10, "target": 30}],
            "blocker_damage": {"hits": 1},
        }
        for name, rewards in (("a", [0.0, 0.0, 1.0]), ("b", [0.0, -1.0]))
        for index, reward in enumerate(rewards)
    ]
    result = reward_distribution(episodes, moves)
    assert result["fractions"] == {"-1": 0.2, "0": 0.6, "1": 0.2}
    assert result["before_terminal"][0][
        "objective_completion_before_terminal"
    ] == pytest.approx(2 / 3)
    assert result["before_terminal"][0]["blocker_hits_before_terminal"] == 2
