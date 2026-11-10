import csv
import json
import shutil
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from match3_ai.evaluation import neural_mcts_benchmark as benchmark
from match3_ai.evaluation.neural_mcts_benchmark import (
    EvaluationConfig,
    load_policy_value_checkpoint,
    run_benchmark,
)
from match3_ai.evaluation.runner import EpisodeResult
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.config import LevelConfig, LevelObjective, ObjectiveType
from match3_ai.training.neural_mcts import NeuralMCTSTrainer, NeuralMCTSTrainingConfig


@pytest.fixture
def saved(tmp_path):
    level = LevelConfig(
        rows=3,
        cols=3,
        num_colors=3,
        move_limit=1,
        level_id="first-match",
        objectives=(LevelObjective(ObjectiveType.COLLECT_TILES, target=1),),
    )
    trainer = NeuralMCTSTrainer(NeuralMCTSTrainingConfig(), level)
    path = tmp_path / "policy.pt"
    trainer.save_checkpoint(path)
    return level, path


def test_inference_load_is_frozen_and_preserves_checkpoint(saved):
    _, path = saved
    before = benchmark.file_sha256(path)
    model, metadata = load_policy_value_checkpoint(path)
    assert not model.training and not any(p.requires_grad for p in model.parameters())
    assert metadata["sha256"] == before == benchmark.file_sha256(path)
    payload = torch.load(path, weights_only=True)
    payload["encoder_version"] = 2
    bad = path.with_name("bad.pt")
    torch.save(payload, bad)
    with pytest.raises(ValueError, match="incompatible"):
        load_policy_value_checkpoint(bad)


def test_paired_outputs_reproducibility_and_overwrite_protection(saved, tmp_path):
    level, path = saved
    config = EvaluationConfig(
        episodes=2, easy_episodes=1, control_episodes=1, simulations=1
    )
    first = run_benchmark(
        config, {"medium": level}, {"Neural_round_0001": path}, tmp_path / "first"
    )
    second = run_benchmark(
        config, {"medium": level}, {"Neural_round_0001": path}, tmp_path / "second"
    )
    for a, b in zip(first, second):
        for key in (
            "wins",
            "losses",
            "errors",
            "mean_moves_used",
            "mean_score_tiles_cleared",
        ):
            assert a[key] == b[key]
        assert a["win_rate_ci95_low"] < 1 and a["win_rate_ci95_high"] == pytest.approx(
            1
        )
    episodes = list(csv.DictReader((tmp_path / "first/episodes.csv").open()))
    for seed in {e["seed"] for e in episodes}:
        group = [e for e in episodes if e["seed"] == seed]
        assert len(group) == 5
        assert len({e["initial_fingerprint"] for e in group}) == 1
        assert all(e["pairing_verified"] == "True" for e in group)
    assert (
        json.loads((tmp_path / "first/manifest.json").read_text())["status"]
        == "complete"
    )
    assert (tmp_path / "first/checkpoint_progression.csv").exists()
    with pytest.raises(FileExistsError):
        run_benchmark(
            config, {"medium": level}, {"Neural_round_0001": path}, tmp_path / "first"
        )


def test_evaluation_binding_disables_noise_and_rejects_pairing_mismatch(saved):
    level, path = saved
    env = Match3LevelEnv(level.level_id, level_config=level)
    model, _ = load_policy_value_checkpoint(path)
    try:
        observation, _ = env.reset(seed=31000000)
        bound = benchmark.EvaluationAgent(
            "Neural_round_1", env, model, 44, 1, {}, ("medium", 31000000)
        )
        assert observation["action_mask"][bound.select_action(observation)]
        assert bound.agent.evaluation and bound.agent.dirichlet_epsilon == 0
        bad = benchmark.EvaluationAgent(
            "Random",
            env,
            None,
            44,
            1,
            {("medium", 31000000): ("bad",)},
            ("medium", 31000000),
        )
        with pytest.raises(ValueError, match="pairing"):
            bad.select_action(observation)
    finally:
        env.close()


def test_failure_preservation_and_missing_latency(saved, tmp_path, monkeypatch):
    level, path = saved

    def failed(env, agent, seed, **kwargs):
        return EpisodeResult(
            seed=seed,
            total_reward=0,
            steps=0,
            valid_actions=0,
            invalid_actions=0,
            total_cleared=0,
            total_cascades=0,
            reshuffles=0,
            completed=False,
            agent_name=kwargs["agent_name"],
            episode_id=kwargs["episode_id"],
            level_id=level.level_id,
            outcome="ERROR",
            error_message="controlled reset failure",
        )

    monkeypatch.setattr(benchmark, "run_episode", failed)
    rows = run_benchmark(
        EvaluationConfig(episodes=1, simulations=1),
        {"medium": level},
        {"Neural_round_1": path},
        tmp_path / "errors",
    )
    assert all(
        r["errors"] == 1 and r["mean_decision_time_seconds"] is None for r in rows
    )
    assert "controlled reset failure" in (tmp_path / "errors/episodes.csv").read_text()


@pytest.mark.parametrize(
    "field,value",
    [
        ("episodes", 0),
        ("simulations", True),
        ("env_seed_start", 400000),
        ("agent_seed_start", 200000),
        ("env_seed_start", 41000000),
    ],
)
def test_seed_and_budget_validation(field, value):
    with pytest.raises(ValueError):
        EvaluationConfig(**{field: value})


def test_metric_denominators():
    records = [
        EpisodeResult(
            seed=i,
            total_reward=outcome,
            steps=2,
            valid_actions=2,
            invalid_actions=0,
            total_cleared=6,
            total_cascades=0,
            reshuffles=0,
            completed=True,
            agent_name="Random",
            level_id="medium",
            episode_id=str(i),
            outcome="WON" if outcome == 1 else "LOST",
            moves_used=2,
        )
        for i, outcome in enumerate([1, -1])
    ]
    extras = [
        {
            "episode_id": str(i),
            "episode_time_seconds": 4,
            "action_selection_calls": calls,
            "decision_time_seconds": seconds,
        }
        for i, calls, seconds in [(0, 1, 1), (1, 3, 9)]
    ]
    row = benchmark.comparison_rows(records, extras)[0]
    assert row["mean_decision_time_seconds"] == 2.5
    assert row["mean_score_tiles_cleared"] == 6 and row["mean_moves_used"] == 2


def test_checkpoint_round_counter_mismatch_is_rejected(saved):
    _, path = saved
    payload = torch.load(path, weights_only=True)
    wrong_round = path.with_name("round_0100.pt")
    torch.save(payload, wrong_round)
    with pytest.raises(ValueError, match="round counter"):
        load_policy_value_checkpoint(wrong_round)


def test_interruption_preserves_completed_records(saved, tmp_path, monkeypatch):
    level, path = saved
    original = benchmark.run_episode
    calls = 0

    def interrupted(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 3:
            raise KeyboardInterrupt
        return original(*args, **kwargs)

    monkeypatch.setattr(benchmark, "run_episode", interrupted)
    output = tmp_path / "interrupted"
    with pytest.raises(KeyboardInterrupt):
        run_benchmark(
            EvaluationConfig(episodes=1, simulations=1),
            {"medium": level},
            {"Neural_round_1": path},
            output,
        )
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "interrupted_or_failed"
    assert manifest["records_written"] == 2
    with (output / "episodes.csv").open() as handle:
        assert len(list(csv.DictReader(handle))) == 2


def test_public_profile_fixtures_work_without_local_results(saved, tmp_path):
    _, path = saved
    _, metadata = load_policy_value_checkpoint(path)
    fixtures = Path(__file__).resolve().parents[1] / "experiments/fixtures/az4_profiles"
    shutil.copytree(fixtures, tmp_path / "experiments/fixtures/az4_profiles")
    profiles, provenance = benchmark.load_profiles(metadata, root=tmp_path)
    assert profiles["medium"].num_colors == profiles["hard_compatible"].num_colors == 5
    assert all("experiments/fixtures" in path for path in provenance["files"])
    assert not (tmp_path / "results").exists()
