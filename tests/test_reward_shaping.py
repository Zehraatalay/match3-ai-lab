import csv
import json
from copy import deepcopy
from dataclasses import asdict, replace

import numpy as np
import pytest

from match3_ai.training.reward_shaping import objective_potential, training_reward


def observation(progress=(0, 0), targets=(6, 30), kinds=(1, 2), mask=(1, 1)):
    return {
        "objective_progress": np.array(progress, dtype=float),
        "objective_target": np.array(targets, dtype=float),
        "objective_type": np.array(kinds),
        "objective_mask": np.array(mask),
    }


def potential_observation(value):
    return observation((value, 0), (1, 30), mask=(1, 0))


def test_equal_objective_weights_clamping_and_inactive_slots():
    assert objective_potential(observation((3, 15))) == 0.5
    assert objective_potential(observation((6, 0))) == 0.5
    assert objective_potential(observation((12, -5))) == 0.5
    assert (
        objective_potential(observation((3, 100), kinds=(1, 999), mask=(1, 0))) == 0.5
    )
    assert objective_potential(observation(mask=(0, 0))) == 0
    assert objective_potential(None, terminated=True) == 0


@pytest.mark.parametrize("target", [0, -1, np.nan, np.inf])
def test_invalid_targets_contribute_zero_without_renormalizing(target):
    assert objective_potential(observation((3, 15), (target, 30))) == 0.25
    assert objective_potential(observation((np.nan, 15))) == 0.25


def test_unknown_types_and_invalid_shapes_fail_explicitly():
    with pytest.raises(ValueError, match="type"):
        objective_potential(observation(kinds=(1, 3)))
    with pytest.raises(ValueError, match="shapes"):
        objective_potential(observation(mask=(1,)))
    with pytest.raises(ValueError, match="binary"):
        objective_potential(observation(mask=(2, 1)))


def test_potential_differences_use_same_gamma_and_are_deterministic():
    before, after = potential_observation(0.5), potential_observation(0.75)
    result = training_reward(before, after, 0.0, terminated=False, mode="pbrs")
    assert result.shaping_bonus == pytest.approx(0.5 * (0.99 * 0.75 - 0.5))
    assert result == training_reward(before, after, 0.0, terminated=False, mode="pbrs")
    other = training_reward(
        before, after, 0.0, terminated=False, mode="pbrs", gamma=0.5
    )
    assert other.shaping_bonus == pytest.approx(-0.0625)


@pytest.mark.parametrize("sparse", [-1.0, 0.0, 1.0])
def test_beta_zero_and_default_sparse_are_exactly_compatible(sparse):
    before, after = potential_observation(0.2), potential_observation(0.8)
    for options in ({}, {"mode": "pbrs", "beta": 0}):
        result = training_reward(before, after, sparse, terminated=False, **options)
        assert result.sparse_reward == result.shaped_reward == sparse
        assert result.shaping_bonus == 0


def test_termination_zeros_potential_but_truncation_bootstraps():
    before, after = potential_observation(0.5), potential_observation(0.75)
    terminated = training_reward(
        before, after, 1.0, terminated=True, truncated=True, mode="pbrs"
    )
    assert terminated.potential_next == 0 and terminated.shaped_reward == 0.75
    truncated = training_reward(
        before, after, 0.0, terminated=False, truncated=True, mode="pbrs"
    )
    assert truncated.potential_next == 0.75
    assert truncated.shaping_bonus == pytest.approx(0.12125)


def test_finite_horizon_telescoping_and_truncated_boundary_identity():
    gamma, beta = 0.99, 0.5
    for potentials, rewards in (
        ([0.2, 0.4, 0.8, 1.0], [0.0, 0.0, 1.0]),
        ([0.2, 0.3, 1.0], [0.0, -1.0]),
    ):
        shaped = []
        for index, reward in enumerate(rewards):
            result = training_reward(
                potential_observation(potentials[index]),
                potential_observation(potentials[index + 1]),
                reward,
                terminated=index == len(rewards) - 1,
                mode="pbrs",
                gamma=gamma,
                beta=beta,
            )
            shaped.append(result.shaped_reward)
        difference = sum(
            gamma**i * (a - b) for i, (a, b) in enumerate(zip(shaped, rewards))
        )
        assert difference == pytest.approx(-beta * potentials[0])
    potentials = [0.2, 0.4, 0.8]
    rewards = [0.0, 0.0]
    shaped = [
        training_reward(
            potential_observation(potentials[i]),
            potential_observation(potentials[i + 1]),
            0.0,
            terminated=False,
            truncated=i == 1,
            mode="pbrs",
        ).shaped_reward
        for i in range(2)
    ]
    partial = sum(gamma**i * r for i, r in enumerate(shaped))
    assert partial == pytest.approx(beta * (gamma**2 * 0.8 - 0.2))
    sparse_bootstrap = 0.7
    shaped_bootstrap = sparse_bootstrap - beta * 0.8
    assert (
        partial + gamma** 2 * shaped_bootstrap - gamma** 2 * sparse_bootstrap
        == pytest.approx(-beta * 0.2)
    )


def test_blocker_only_objective_does_not_reward_unrelated_tile_clears():
    before = observation((0, 0), mask=(1, 0))
    after = observation((0, 1000), mask=(1, 0))
    result = training_reward(before, after, 0.0, terminated=False, mode="pbrs")
    assert result.shaping_bonus == 0


def test_invariance_assumptions_do_not_hold_for_wrong_discount_or_terminal_boundary():
    potentials = [0.2, 0.4, 1.0]
    wrong_discount = [
        training_reward(
            potential_observation(potentials[i]),
            potential_observation(potentials[i + 1]),
            0.0,
            terminated=i == 1,
            mode="pbrs",
            gamma=0.9,
        ).shaping_bonus
        for i in range(2)
    ]
    assert wrong_discount[0] + 0.99 * wrong_discount[1] != pytest.approx(-0.5 * 0.2)
    incorrect_terminal = []
    for length in (2, 3):
        path = [0.2] + [0.4] * (length - 1) + [1.0]
        bonuses = [
            training_reward(
                potential_observation(path[i]),
                potential_observation(path[i + 1]),
                0.0,
                terminated=False,
                mode="pbrs",
            ).shaping_bonus
            for i in range(length)
        ]
        incorrect_terminal.append(
            sum(0.99**i * bonus for i, bonus in enumerate(bonuses))
        )
    assert incorrect_terminal[0] != pytest.approx(incorrect_terminal[1])


def test_reward_computation_does_not_mutate_observation_or_environment_rng():
    from match3_ai.evaluation.behavior_data import complete_state_fingerprint
    from match3_ai.games.match3.environment import Match3LevelEnv

    env = Match3LevelEnv("blocker-mayhem")
    obs, _ = env.reset(seed=300000)
    copy = deepcopy(obs)
    fingerprint = complete_state_fingerprint(env)
    for _ in range(4):
        training_reward(obs, obs, 0.0, terminated=False, mode="pbrs")
    assert complete_state_fingerprint(env) == fingerprint
    for key in obs:
        np.testing.assert_array_equal(obs[key], copy[key])


def test_controlled_profiles_differ_only_in_reward_mode():
    pytest.importorskip("torch")
    from match3_ai.training.pbrs_pilot import pilot_config

    sparse, shaped = asdict(pilot_config("sparse")), asdict(pilot_config("pbrs"))
    assert [key for key in sparse if sparse[key] != shaped[key]] == ["reward_mode"]
    assert sparse["seed"] == 5903 and sparse["environment_steps"] == 5000
    assert sparse["shaping_beta"] == 0.5 and sparse["gamma"] == 0.99
    assert sparse["evaluation_episodes"] == 100 and sparse["evaluation_every"] == 1000


def test_training_reward_replay_logs_and_sparse_evaluation_are_isolated(tmp_path):
    pytest.importorskip("torch")
    from match3_ai.agents.dqn_agent import DQNAgent
    from match3_ai.training.dqn import train_dqn
    from match3_ai.training.pbrs_pilot import pilot_config

    config = replace(
        pilot_config("sparse"),
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
    sparse = train_dqn(config, tmp_path / "sparse")
    zero = train_dqn(
        replace(config, reward_mode="pbrs", shaping_beta=0), tmp_path / "zero"
    )
    assert sparse["final_weights"] == zero["final_weights"]
    shaped = train_dqn(replace(config, reward_mode="pbrs"), tmp_path / "shaped")
    repeated = train_dqn(replace(config, reward_mode="pbrs"), tmp_path / "repeat")
    assert shaped["initial_weights"] == sparse["initial_weights"]
    assert shaped["final_weights"] == repeated["final_weights"]
    _, payload = DQNAgent.load_checkpoint(shaped["checkpoint"])
    with (tmp_path / "shaped/training_metrics.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    np.testing.assert_allclose(
        payload["training_state"]["replay"]["rewards"].numpy(),
        [float(r["shaped_reward"]) for r in rows],
        rtol=1e-7,
        atol=1e-7,
    )
    assert any(float(r["shaping_bonus"]) != 0 for r in rows)
    assert all(float(r["sparse_reward"]) in (-1, 0, 1) for r in rows)
    assert shaped["evaluation_reward_mode"] == "sparse"
    episodes = json.loads((tmp_path / "shaped/training_episodes.json").read_text())
    for episode in episodes:
        assert episode["reward"] == episode["sparse_return"]
        boundary = 0.5 * config.gamma ** episode["steps"] * episode["final_potential"]
        assert episode["discounted_shaped_return"] - episode[
            "discounted_sparse_return"
        ] == pytest.approx(boundary, abs=1e-12)
    with (tmp_path / "shaped/evaluation/moves.csv").open() as handle:
        assert all(float(row["reward"]) in (-1, 0, 1) for row in csv.DictReader(handle))
    with (tmp_path / "shaped/evaluation/episodes.csv").open() as handle:
        assert all(
            float(row["total_reward"]) in (-1, 1) for row in csv.DictReader(handle)
        )
    with pytest.raises(ValueError, match="compatible"):
        train_dqn(
            replace(config, environment_steps=16, reward_mode="pbrs"),
            tmp_path / "wrong_resume",
            resume_from=sparse["checkpoint"],
        )
    pytest.importorskip("matplotlib")
    from match3_ai.training.pbrs_analysis import analyze_pilot

    analysis = analyze_pilot(
        tmp_path / "sparse",
        tmp_path / "shaped",
        tmp_path / "analysis",
        historical_checkpoint=tmp_path / "sparse/policy_step_12.pt",
    )
    assert analysis["initial_weights_identical"] and analysis["rng_streams_identical"]
    assert analysis["historical_sparse_weights_match"]
    assert not analysis["final_test_used"]
    assert len(analysis["figures"]) == 16
    assert analysis["runs"]["pbrs"]["max_telescoping_error"] < 1e-12
    assert all(
        (tmp_path / "analysis/figures" / name).stat().st_size > 1000
        for name in analysis["figures"]
    )
    with pytest.raises(FileExistsError):
        analyze_pilot(tmp_path / "sparse", tmp_path / "shaped", tmp_path / "analysis")
