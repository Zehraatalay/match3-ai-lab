"""Pure validation and orchestration tests; no environment episodes are executed."""

import json
import sys
from dataclasses import replace
from types import SimpleNamespace

import pytest

from match3_ai.evaluation import cross_level_benchmark as benchmark
from match3_ai.evaluation.runner import EpisodeResult


@pytest.mark.parametrize("start", [200000, 300000, 310000, 400000, 500000, 699900])
def test_protected_seed_ranges(tmp_path, start):
    with pytest.raises(ValueError, match="protected"):
        benchmark.validate_seeds(tuple(range(start, start + 200)), tmp_path)


def test_freshness_and_duplicate_seeds(tmp_path):
    benchmark.validate_seeds(benchmark.SEEDS, tmp_path)
    with pytest.raises(ValueError, match="distinct"):
        benchmark.validate_seeds((710000,) * 200, tmp_path)
    (tmp_path / "episodes.csv").write_text("effective_episode_seed\n710199\n")
    with pytest.raises(ValueError, match="already used"):
        benchmark.validate_seeds(benchmark.SEEDS, tmp_path)


def test_existing_output_rejected_before_loading(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "prepare", lambda _: pytest.fail("loaded inputs"))
    with pytest.raises(FileExistsError):
        benchmark.run_benchmark(tmp_path, tmp_path)


def test_partial_results_survive_failure_without_replacement(tmp_path, monkeypatch):
    monkeypatch.setattr(
        benchmark,
        "prepare",
        lambda *a, **k: (
            {"levels": {"first-match": {"move_limit": 5}}, "policies": []},
            {},
        ),
    )

    class FakeEnvironment:
        initial_fingerprint = "identical"
        pairing_error = False

        def __init__(self, *args):
            pass

        def close(self):
            pass

    calls = []

    def fake_episode(env, agent, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise KeyboardInterrupt("test interruption")
        return EpisodeResult(
            seed=kwargs["seed"],
            total_reward=0,
            steps=5,
            valid_actions=5,
            invalid_actions=0,
            total_cleared=0,
            total_cascades=0,
            reshuffles=0,
            completed=True,
            outcome="LOST",
            level_id="first-match",
            agent_name=kwargs["agent_name"],
            moves_used=5,
        )

    monkeypatch.setattr(benchmark, "PairedEnvironment", FakeEnvironment)
    monkeypatch.setattr(benchmark, "run_episode", fake_episode)
    output = tmp_path / "new"
    with pytest.raises(KeyboardInterrupt):
        benchmark.run_benchmark(tmp_path, output, levels=("first-match",))
    rows = (output / "episodes.jsonl").read_text().splitlines()
    assert len(rows) == 1
    assert json.loads(rows[0])["initial_fingerprint"] == "identical"
    assert json.loads((output / "failure.json").read_text())["completed_records"] == 1
    assert "interrupted" in (output / "report.md").read_text()
    assert len(calls) == 2
    assert all(call["max_steps"] == 5 and call["seed"] == 710000 for call in calls)


def test_pairing_check_uses_level_and_seed_without_real_reset(monkeypatch):
    monkeypatch.setattr(benchmark.Match3LevelEnv, "__init__", lambda *a, **k: None)
    monkeypatch.setattr(benchmark.Match3LevelEnv, "reset", lambda *a, **k: ({}, {}))
    fingerprints = iter(["board-a", "board-a", "board-b", "other-level"])
    monkeypatch.setattr(
        benchmark, "observation_fingerprint", lambda _: next(fingerprints)
    )
    identities = {}
    for _ in range(2):
        benchmark.PairedEnvironment("first-match", identities, 710000).reset(
            seed=710000
        )
    env = benchmark.PairedEnvironment("first-match", identities, 710000)
    with pytest.raises(RuntimeError, match="Initial boards differ"):
        env.reset(seed=710000)
    assert env.pairing_error
    benchmark.PairedEnvironment("wooden-trouble", identities, 710000).reset(seed=710000)
    assert len(identities) == 2


def test_incremental_results_reserve_used_seeds(tmp_path):
    (tmp_path / "episodes.jsonl").write_text('{"seed": 710000}\n')
    with pytest.raises(ValueError, match="already used"):
        benchmark.validate_seeds(benchmark.SEEDS, tmp_path)


@pytest.mark.parametrize("levels", [benchmark.LEVELS, benchmark.CALIBRATED_LEVELS])
def test_prepare_preserves_selected_levels_and_hashes(tmp_path, monkeypatch, levels):
    def load_checkpoint(path):
        seed, step, _ = next(
            item for item in benchmark.SELECTION if item[2] == path.parent.name
        )
        config = {
            "seed": seed,
            "level_id": "blocker-mayhem",
            "train_seed_start": 500000,
            "environment_steps": 150000,
        }
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"frozen checkpoint")
        (path.parent / "training_summary.json").write_text(
            json.dumps({"training_config": config})
        )
        network = SimpleNamespace(eval=lambda: None, requires_grad_=lambda _: None)
        policy = SimpleNamespace(
            architecture="dueling", algorithm="double", online=network, target=network
        )
        return policy, {"training_config": config, "training_step": step}

    for _, _, directory in benchmark.SELECTION:
        parent = tmp_path / "sprint5_20_rl_optimization" / directory
        parent.mkdir(parents=True)
        (parent / "training_summary.json").write_text(
            json.dumps(
                {
                    "training_config": {
                        "seed": next(
                            s for s, _, d in benchmark.SELECTION if d == directory
                        ),
                        "level_id": "blocker-mayhem",
                        "train_seed_start": 500000,
                        "environment_steps": 150000,
                    }
                }
            )
        )
    monkeypatch.setattr(benchmark.DQNAgent, "load_checkpoint", load_checkpoint)
    manifest, policies = benchmark.prepare(
        tmp_path,
        levels=levels,
        environment_seeds=tuple(range(790000, 790200)),
        random_agent_seeds=tuple(range(791000, 791200)),
    )
    assert tuple(manifest["levels"]) == levels
    assert manifest["planned_episodes"] == 6000
    assert len(policies) == 3
    assert manifest["final_test_used"] is False
    assert "development evaluation" in manifest["evaluation_label"]
    assert len(manifest["benchmark_configuration_sha256"]) == 64
    assert all(len(item["sha256"]) == 64 for item in manifest["levels"].values())
    assert all(
        len(item["training_config_sha256"]) == 64 for item in manifest["policies"]
    )


@pytest.mark.parametrize("key", ["environment_seed", "agent_seed", "random_seed"])
@pytest.mark.parametrize("container", [None, "info", "record"])
def test_seed_freshness_covers_nested_records_and_both_roles(tmp_path, key, container):
    row = {key: 791199}
    if container:
        row = {container: row}
    (tmp_path / "episodes.jsonl").write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="already used"):
        benchmark.validate_seeds(
            tuple(range(790000, 790200)),
            tmp_path,
            random_agent_seeds=tuple(range(791000, 791200)),
        )


@pytest.mark.parametrize("calibrated", [False, True])
def test_cli_preserves_defaults_and_accepts_calibrated_selection(
    monkeypatch, calibrated
):
    args = ["benchmark", "--output-dir", "unused"]
    if calibrated:
        args += [
            "--levels",
            *benchmark.CALIBRATED_LEVELS,
            "--environment-seed-start",
            "790000",
            "--random-agent-seed-start",
            "791000",
        ]
    calls = []
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(benchmark, "run_benchmark", lambda *a, **k: calls.append(k))
    benchmark.main()
    assert tuple(calls[0]["levels"]) == (
        benchmark.CALIBRATED_LEVELS if calibrated else benchmark.LEVELS
    )
    assert calls[0]["environment_seed_start"] == (790000 if calibrated else 710000)
    assert calls[0]["random_agent_seed_start"] == (791000 if calibrated else 720000)


def test_report_compares_versions_and_baselines_with_failed_attempts(tmp_path):
    base = EpisodeResult(
        seed=790000,
        total_reward=0,
        steps=2,
        valid_actions=2,
        invalid_actions=0,
        total_cleared=0,
        total_cascades=0,
        reshuffles=0,
        completed=True,
        outcome="WON",
        moves_used=2,
        objective_completion_ratio=1.0,
    )
    records = []
    for level in ("wooden-trouble", "wooden-trouble-v2"):
        for agent in ("DDQN_5990_125000", "Random", "Greedy"):
            record = replace(base, level_id=level, agent_name=agent)
            records.append(record)
            records.append(
                replace(
                    record,
                    seed=790001,
                    outcome="WON"
                    if agent.startswith("DDQN") and level.endswith("v2")
                    else "ERROR",
                    completed=False,
                    moves_used=None,
                    objective_completion_ratio=None,
                )
            )
    benchmark.write_report(tmp_path, records, "complete", expected_count=2)
    report = (tmp_path / "report.md").read_text()
    summaries = json.loads((tmp_path / "summary.json").read_text())
    random = next(row for row in summaries if row["agent_name"] == "Random")
    assert random["win_rate"] == 0.5
    assert random["errors"] == 1
    assert random["win_rate_ci95"] == pytest.approx([0.0945312, 0.9054688])
    assert "Wilson 95% CI" in report
    assert "development results" in report
    assert (
        "| wooden-trouble -> wooden-trouble-v2 | DDQN_5990_125000 | +50.000 |" in report
    )
    assert "| wooden-trouble-v2 | DDQN_5990_125000 | +50.000 | +50.000 |" in report
