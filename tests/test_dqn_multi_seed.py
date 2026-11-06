import csv
import json
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from match3_ai.agents.dqn_agent import DQNAgent
from match3_ai.training.dqn import train_dqn
from match3_ai.training.multi_seed import MASTER_SEEDS, experiment_config
from match3_ai.training.multi_seed_analysis import analyze_runs


def tiny_config(seed=5901):
    return replace(
        experiment_config(seed),
        level_id="first-match",
        environment_steps=12,
        evaluation_every=6,
        evaluation_episodes=1,
        hidden_dims=(8,),
        replay_capacity=32,
        batch_size=2,
        warmup_steps=2,
        train_every=1,
        target_sync_updates=2,
    )


def test_long_profiles_have_independent_streams_and_reviewed_schedule():
    configs = [experiment_config(seed) for seed in MASTER_SEEDS]
    assert [c.train_seed_start for c in configs] == [200000, 220000, 240000]
    fingerprints = []
    for config in configs:
        streams = np.random.SeedSequence(config.seed).spawn(2)
        agent_seed, replay_seed = (
            int(stream.generate_state(1)[0]) for stream in streams
        )
        agent = DQNAgent(agent_seed, hidden_dims=(8,))
        fingerprints.append((agent_seed, replay_seed, agent.network_seed))
        assert config.epsilon(50000) == pytest.approx(0.05)
        assert config.epsilon(49999) == pytest.approx(0.050019)
        assert config.evaluation_seed_start == 300000
        assert config.evaluation_episodes == 100 and config.evaluation_every == 5000
        assert config.evaluate_step_zero and not config.baseline_every_checkpoint
        updates = (
            config.environment_steps - config.warmup_steps
        ) // config.train_every + 1
        assert updates == 12251 and updates // config.target_sync_updates == 12
    assert all(len({row[i] for row in fingerprints}) == 3 for i in range(3))


def test_step_zero_is_untrained_and_milestones_include_metrics(tmp_path):
    config = tiny_config()
    result = train_dqn(config, tmp_path / "run")
    assert [r["environment_step"] for r in result["evaluation_rounds"]] == [0, 6, 12]
    assert len(result["evaluation_rounds"][0]["summary"]) == 3
    assert all(len(r["summary"]) == 1 for r in result["evaluation_rounds"][1:])
    assert result["replay_size"] == 12 and result["optimizer_updates"] == 11
    assert result["invalid_actions"] == result["evaluation_invalid_actions"] == 0
    initial, payload = DQNAgent.load_checkpoint(tmp_path / "run/policy_step_0.pt")
    assert payload["training_step"] == 0 and not initial.optimizer.state
    fresh = DQNAgent(result["seed_metadata"]["agent_seed"], hidden_dims=(8,))
    assert all(
        torch.equal(a, b)
        for a, b in zip(initial.online.parameters(), fresh.online.parameters())
    )
    assert payload["training_state"] == {}
    assert payload["seed_metadata"]["master_seed"] == 5901
    assert len(payload["validation_metrics"]) == 3
    _, final = DQNAgent.load_checkpoint(tmp_path / "run/policy_step_12.pt")
    assert final["optimizer"]["state"] and len(final["validation_metrics"]) == 1
    assert (tmp_path / "run/validation/step_0/episodes.csv").exists()
    assert (tmp_path / "run/validation/step_6/moves.csv").exists()
    assert (
        json.loads((tmp_path / "run/training_episodes_step_0.json").read_text()) == []
    )


def test_interruption_preserves_completed_checkpoints_and_validation(
    monkeypatch, tmp_path
):
    original = DQNAgent.select_action
    calls = 0

    def interrupt(self, observation, epsilon=0):
        nonlocal calls
        if epsilon > 0:
            calls += 1
            if calls == 8:
                raise KeyboardInterrupt("injected interruption")
        return original(self, observation, epsilon)

    monkeypatch.setattr(DQNAgent, "select_action", interrupt)
    with pytest.raises(KeyboardInterrupt):
        train_dqn(tiny_config(), tmp_path / "run")
    checkpoint = tmp_path / "run/policy_step_6.pt"
    assert checkpoint.exists()
    _, payload = DQNAgent.load_checkpoint(checkpoint)
    assert payload["training_step"] == 6
    assert (tmp_path / "run/validation/step_6/experiment_metadata.json").exists()
    error = json.loads((tmp_path / "run/training_error.json").read_text())
    assert error["error_type"] == "KeyboardInterrupt" and error["environment_step"] == 7
    with pytest.raises(FileExistsError):
        train_dqn(tiny_config(), tmp_path / "run")


def test_atomic_checkpoint_does_not_publish_partial_file(monkeypatch, tmp_path):
    agent = DQNAgent(5901, hidden_dims=(8,))
    safe = tmp_path / "safe.pt"
    agent.save_checkpoint(safe)
    original = safe.read_bytes()

    def fail(payload, handle):
        handle.write(b"partial checkpoint")
        raise KeyboardInterrupt()

    monkeypatch.setattr(torch, "save", fail)
    with pytest.raises(KeyboardInterrupt):
        agent.save_checkpoint(tmp_path / "incomplete.pt")
    assert not (tmp_path / "incomplete.pt").exists()
    assert safe.read_bytes() == original
    assert list(tmp_path.iterdir()) == [safe]


def test_multi_seed_analysis_validates_runs_and_creates_ten_curves(tmp_path):
    pytest.importorskip("matplotlib")
    paths = []
    for seed in MASTER_SEEDS:
        path = tmp_path / str(seed)
        train_dqn(tiny_config(seed), path)
        paths.append(path)
    result = analyze_runs(paths, tmp_path / "analysis")
    assert result["final_test_evaluated"] is False
    assert len(result["runs"]) == 3
    assert len(result["mean_validation_curve"]) == 3
    assert len(result["figures"]) == 20
    for figure in result["figures"]:
        assert (tmp_path / "analysis/figures" / figure).stat().st_size > 1000
    assert all(report["optimizer_updates"] == 11 for report in result["runs"])
    assert all(report["invalid_actions"] == 0 for report in result["runs"])
    assert all(report["step_zero"]["step"] == 0 for report in result["runs"])
    assert result["aggregate"]["win_rate"]["sample_std"] >= 0
    with pytest.raises(FileExistsError):
        analyze_runs(paths, tmp_path / "analysis")
    with pytest.raises(ValueError, match="independent"):
        analyze_runs([paths[0]] * 3, tmp_path / "duplicate")
    with pytest.raises(ValueError, match="three"):
        analyze_runs(paths[:2], tmp_path / "missing")
    path = paths[0] / "evaluation/episodes.csv"
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields, episodes = reader.fieldnames, list(reader)
    episodes[0]["seed"] = "400000"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(episodes)
    with pytest.raises(ValueError, match="leakage"):
        analyze_runs(paths, tmp_path / "leakage")
