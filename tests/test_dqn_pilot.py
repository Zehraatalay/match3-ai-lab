import csv
from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

pytest.importorskip("torch")

from match3_ai.agents.dqn_agent import DQNAgent
from match3_ai.evaluation.analytics import AnalyticsCollector
from match3_ai.training.analysis import analyze_pilot, create_learning_curves
from match3_ai.training.dqn import evaluate_agents, train_dqn
from match3_ai.training.pilot import pilot_config
from match3_ai.training.seed_policy import PARTITIONS, seed_policy_metadata


def test_pilot_preserves_reviewed_hyperparameters_and_reserves_test_partition():
    config = pilot_config()
    assert config.environment_steps == 5000
    assert config.level_id == "blocker-mayhem"
    assert config.evaluation_every == 1000 and config.evaluation_episodes == 100
    assert config.replay_capacity == 50000 and config.batch_size == 64
    assert config.warmup_steps == config.target_sync_updates == 1000
    assert config.epsilon(0) == 1 and config.epsilon(5000) == pytest.approx(0.905)
    sets = [set(range(low, high + 1)) for low, high in PARTITIONS.values()]
    assert all(not a & b for index, a in enumerate(sets) for b in sets[index + 1 :])
    assert seed_policy_metadata()["final_test_used"] is False
    assert config.train_seed_start == 200000
    assert config.evaluation_seed_start == 300000
    assert config.evaluation_agent_seed_start == 600000


@pytest.mark.parametrize(
    "change",
    [
        {"train_seed_start": 199999},
        {"environment_steps": 100001},
        {"evaluation_seed_start": 400000},
        {"evaluation_seed_start": 299999},
        {"evaluation_episodes": 1001},
        {"evaluation_agent_seed_start": 300050},
        {"seed_policy": "unknown"},
    ],
)
def test_seed_policy_rejects_leakage_before_environment_creation(change):
    with pytest.raises(ValueError):
        replace(pilot_config(), **change)


def small_config():
    return replace(
        pilot_config(),
        level_id="first-match",
        environment_steps=12,
        evaluation_episodes=1,
        evaluation_every=6,
        hidden_dims=(8,),
        replay_capacity=32,
        batch_size=2,
        warmup_steps=2,
        train_every=1,
        target_sync_updates=2,
    )


def test_fixed_validation_is_deterministic_isolated_and_checkpoint_equivalent(tmp_path):
    torch = pytest.importorskip("torch")
    config = replace(small_config(), evaluation_episodes=2)
    agent = DQNAgent(58, hidden_dims=(8,))
    before_rng = deepcopy(agent.rng.bit_generator.state)
    before_weights = {
        key: value.clone() for key, value in agent.online.state_dict().items()
    }
    target = {key: value.clone() for key, value in agent.target.state_dict().items()}
    optimizer = deepcopy(agent.optimizer.state_dict())
    observed_modes = []
    hook = agent.online.register_forward_pre_hook(
        lambda model, inputs: observed_modes.append(model.training)
    )
    first_collector, second_collector = AnalyticsCollector(), AnalyticsCollector()
    first = evaluate_agents(
        config, agent, step=0, run_id="test", collector=first_collector
    )
    assert agent.online.training
    agent.save_checkpoint(
        tmp_path / "policy.pt", training_config={"level_id": config.level_id}
    )
    loaded, _ = DQNAgent.load_checkpoint(tmp_path / "policy.pt")
    loaded.online.eval()
    second = evaluate_agents(
        config, loaded, step=0, run_id="test", collector=second_collector
    )
    hook.remove()
    assert first == second
    assert first_collector.moves == second_collector.moves
    assert not loaded.online.training
    assert observed_modes and not any(observed_modes)
    assert agent.rng.bit_generator.state == before_rng
    assert optimizer == agent.optimizer.state_dict()
    assert all(
        torch.equal(before_weights[key], value)
        for key, value in agent.online.state_dict().items()
    )
    assert all(
        torch.equal(target[key], value)
        for key, value in agent.target.state_dict().items()
    )
    assert {record.seed for record in first} == {300000, 300001}
    assert all(
        record.outcome in ("WON", "LOST") and record.invalid_actions == 0
        for record in first
    )
    for index in range(2):
        paired = [record for record in first if record.seed == 300000 + index]
        assert len({record.board_seed for record in paired}) == 1
        assert len({record.engine_seed for record in paired}) == 1
    with pytest.raises(FileExistsError):
        agent.save_checkpoint(tmp_path / "policy.pt")


def test_evaluation_restores_mode_after_error(monkeypatch):
    import match3_ai.training.dqn as training

    def fail(*args, **kwargs):
        raise RuntimeError("injected evaluation error")

    agent = DQNAgent(58, hidden_dims=(8,))
    monkeypatch.setattr(training, "run_episode", fail)
    with pytest.raises(RuntimeError, match="injected"):
        evaluate_agents(small_config(), agent, step=0, run_id="test")
    assert agent.online.training


def test_metrics_reconcile_and_learning_curves_preserve_inputs(tmp_path):
    pytest.importorskip("matplotlib")
    output = tmp_path / "pilot"
    result = train_dqn(small_config(), output)
    with (output / "training_metrics.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 12
    assert int(rows[-1]["optimizer_updates"]) == 11
    assert int(rows[-1]["target_synchronizations"]) == 5
    assert int(rows[-1]["completed_episodes"]) == result["completed_training_episodes"]
    assert float(rows[-1]["training_win_rate"]) == result["training_win_rate"]
    assert all(row["valid_action"] == "True" for row in rows)
    assert result["evaluation_seconds"] > 0
    assert result["collection_optimization_seconds"] > 0
    assert result["evaluation_errors"] == result["evaluation_invalid_actions"] == 0
    assert result["seed_policy"] == seed_policy_metadata()
    analysis = analyze_pilot(output, tmp_path / "analysis")
    assert analysis["optimizer_updates_recorded"] == 11
    assert analysis["final_test_evaluated"] is False
    assert len(analysis["figures"]) == 14
    assert all(
        (tmp_path / "analysis/figures" / name).stat().st_size > 1000
        for name in analysis["figures"]
    )
    assert np.isfinite(analysis["projected_three_run_seconds"])
    assert (tmp_path / "analysis/report.md").exists()
    with pytest.raises(FileExistsError):
        analyze_pilot(output, tmp_path / "analysis")
    with pytest.raises(FileExistsError):
        create_learning_curves(rows, result, tmp_path / "analysis/figures")
    with pytest.raises(FileExistsError):
        train_dqn(small_config(), output)
    path = output / "evaluation/episodes.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, episodes = reader.fieldnames, list(reader)
    episodes[0]["seed"] = "400000"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episodes)
    with pytest.raises(ValueError, match="leakage"):
        analyze_pilot(output, tmp_path / "leakage")
