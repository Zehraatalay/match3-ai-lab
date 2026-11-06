import csv
import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from match3_ai.agents.dqn_agent import DQNAgent
from match3_ai.agents.ppo_agent import PPOAgent
from match3_ai.evaluation.fresh_analysis import (
    analyze,
    comparisons,
    critic_metrics,
    paired_estimate,
    representative_seeds,
)
from match3_ai.evaluation.fresh_evaluation import load_policy, policy_state, run_policy
from match3_ai.evaluation.fresh_protocol import (
    freeze_protocol,
    historical_inventory,
    load_protocol,
)
from match3_ai.games.match3.environment import Match3LevelEnv


@pytest.fixture
def frozen_fixture(tmp_path):
    history = tmp_path / "history"
    config = asdict(Match3LevelEnv("blocker-mayhem").config)
    level_id = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()
    for seed in (5901, 5902, 5903):
        DQNAgent(seed, hidden_dims=(8,)).save_checkpoint(
            history / f"sprint5_9_seed_{seed}/checkpoint_final.pt",
            training_config={"seed": seed, "level_id": "blocker-mayhem"},
            training_step=50000,
        )
    for seed in (5915, 5916, 5917):
        agent = PPOAgent(seed, hidden_dims=(8,))
        for filename, step in (("policy_step_0.pt", 0), ("checkpoint_final.pt", 50000)):
            agent.save_checkpoint(
                history / f"sprint5_15_ppo_seed_{seed}/{filename}",
                training_config={"seed": seed, "level_id": "blocker-mayhem"},
                training_progress={
                    "environment_step": step,
                    "level_config_id": level_id,
                },
            )
    protocol_dir = tmp_path / "protocol"
    protocol = freeze_protocol(history, protocol_dir, count=2)
    assert protocol["seeds"] == [300000, 300001]
    return protocol_dir / "protocol.json", protocol, tmp_path / "results"


def test_history_inventory_excludes_recorded_values_and_declared_ranges(tmp_path):
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "evaluation_seed_start": 300000,
                "evaluation_episodes": 100,
                "unrelated_conservative_value": 300150,
            }
        )
    )
    (tmp_path / "episodes.csv").write_text("seed,episode\n300120,1\n200001,2\n")
    (tmp_path / "state.jsonl").write_text('{"environment_seed":300121}\n')
    np.savez(tmp_path / "seeds.npz", environment_seeds=np.array([300122]))
    result = historical_inventory(tmp_path)
    assert set(result["excluded_validation_values"]) == set(range(300000, 300100)) | {
        300120,
        300121,
        300122,
        300150,
    }
    assert len(result["files_sha256"]) == 4


def test_protocol_is_frozen_and_all_policy_loads_are_deterministic(frozen_fixture):
    path, protocol, _ = frozen_fixture
    assert load_protocol(path) == protocol
    assert len(protocol["policies"]) == 11 and len(protocol["secondaries"]) == 9
    observation, _ = Match3LevelEnv("blocker-mayhem").reset(seed=300000)
    for specification in protocol["policies"]:
        first = load_policy(specification, protocol)
        second = load_policy(specification, protocol)
        state = policy_state(first)
        assert first.select_action(observation) == second.select_action(observation)
        if specification["kind"] in ("ppo", "dqn"):
            assert first.select_action(observation) == second.select_action(observation)
            assert policy_state(first) == state
    path.write_text(path.read_text() + " ")
    with pytest.raises(ValueError, match="digest"):
        load_protocol(path)


def test_paired_statistical_family_and_critic_diagnostics():
    policy = np.array([1, 1, 0, 0])
    reference = np.array([1, 0, 1, 0])
    estimate = paired_estimate(policy, reference, samples=200, seed=1, binary=True)
    assert estimate["counts"] == {
        "both_win": 1,
        "left_only": 1,
        "right_only": 1,
        "both_lose": 1,
    }
    assert estimate["p_value"] == 1 and estimate["difference"] == 0
    wins = {
        "Greedy": reference,
        "Random": reference,
        **{f"DQN{s}": reference for s in (5901, 5902, 5903)},
        **{
            f"PPO{s}_{phase}": policy if phase == "final" else reference
            for s in (5915, 5916, 5917)
            for phase in ("initial", "final")
        },
    }
    result = comparisons(wins, {"bootstrap_samples": 200, "bootstrap_seed": 1})
    assert result["primary"]["n_boards"] == 4 and len(result["secondary"]) == 9
    assert all(r["holm_p"] >= r["p_value"] for r in result["secondary"].values())
    perfect = critic_metrics([-1, 0, 1], [-1, 0, 1])
    assert (
        perfect["rmse"] == perfect["prediction_bias_V_minus_return"] == 0
        and perfect["explained_variance"] == 1
    )
    biased = critic_metrics([1, 2, 3], [-1, 0, 1])
    assert biased["rmse"] == 2 and biased["explained_variance"] == 1
    assert critic_metrics([0, 1], [1, 1])["explained_variance"] is None


def test_complete_shared_board_evaluation_and_analysis(frozen_fixture, tmp_path):
    path, protocol, root = frozen_fixture
    checkpoint_hashes = {
        p["checkpoint"]: p["sha256"] for p in protocol["policies"] if "checkpoint" in p
    }
    for specification in protocol["policies"]:
        result = run_policy(path, specification["name"], root / specification["name"])
        assert result["episodes"] == 2
        assert result["summary"]["errors"] == 0
    result = analyze(path, root, tmp_path / "analysis")
    assert result["episodes"] == 22 and len(result["figures"]) == 8
    assert (
        len(result["statistics"]["secondary"]) == 9
        and len(result["critic_behavior_agreement"]) == 6
    )
    with (tmp_path / "analysis/paired_outcomes.csv").open() as handle:
        rows = list(csv.DictReader(handle))
    assert [int(r["seed"]) for r in rows] == [300000, 300001]
    assert all(
        hashlib.sha256(Path(name).read_bytes()).hexdigest() == digest
        for name, digest in checkpoint_hashes.items()
    )
    assert not result["final_test_used"]
    with pytest.raises(FileExistsError):
        run_policy(path, "Random", root / "Random")
    with pytest.raises(FileExistsError):
        analyze(path, root, tmp_path / "analysis")
    state_path = root / "Greedy/initial_states.json"
    states = json.loads(state_path.read_text())
    states[0]["fingerprint"] = "incorrect"
    state_path.write_text(json.dumps(states))
    with pytest.raises(ValueError, match="identical initial"):
        analyze(path, root, tmp_path / "misaligned")


def test_representative_selection_uses_lowest_seeds_and_keeps_empty_categories():
    wins = {
        name: np.array([1, 0, 1, 0])
        for name in [
            "Random",
            "Greedy",
            *[f"DQN{s}" for s in (5901, 5902, 5903)],
            *[
                f"PPO{s}_{phase}"
                for s in (5915, 5916, 5917)
                for phase in ("initial", "final")
            ],
        ]
    }
    selected = representative_seeds(wins, [300000, 300001, 300002, 300003], 1)
    assert selected["all eleven win"]["selected_seeds"] == [300000]
    assert selected["all eleven lose"]["selected_seeds"] == [300001]
    assert selected["PPO majority win / Greedy loss"]["selected_seeds"] == []
