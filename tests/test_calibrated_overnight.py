"""Scheduling, provenance, restart, and bounded calibrated evaluation checks."""

import fcntl
import json
from copy import deepcopy
from pathlib import Path

import pytest

from match3_ai.training import calibrated_overnight as overnight


@pytest.fixture(autouse=True)
def isolate_seed_metadata(monkeypatch):
    monkeypatch.setattr(
        overnight.seed_policy,
        "STUDY_VALIDATION_PARTITIONS",
        dict(overnight.seed_policy.STUDY_VALIDATION_PARTITIONS),
    )


def test_seed_separation_and_reserved_final_partition():
    assert overnight.seed_plan_valid(
        (200000, 299999), overnight.VALIDATION, overnight.FINAL
    )
    assert not overnight.seed_plan_valid((1, 10), [4], [20])
    assert not overnight.seed_plan_valid((1, 10), [20], [20])
    assert not overnight.seed_plan_valid((1, 10), [20], [400000])
    assert not overnight.seed_plan_valid((1, 10), [20, 20], [30])


@pytest.mark.parametrize("kind", ["ddqn", "ppo"])
def test_deterministic_specs_and_budget_and_level_identity(kind):
    for variant in overnight.VARIANTS:
        spec = overnight.training_spec(variant, kind, "screen", 4096)
        assert spec == overnight.training_spec(variant, kind, "screen", 4096)
        assert spec["config"]["level_id"] == variant.split("__")[0]
        assert spec["config"]["max_episode_steps"] == (
            10 if variant.startswith("frozen") else 6
        )
        assert (
            overnight.training_spec(variant, kind, "optimize", 8192)["id"] != spec["id"]
        )
        assert spec["config"]["evaluation_seed_start"] == 810000


def test_checkpoint_compatibility_requires_full_optimizer_and_replay():
    config = overnight.training_spec(overnight.VARIANTS[0], "ddqn", "screen", 4096)[
        "config"
    ]
    payload = {
        "training_config": {**config, "environment_steps": 2048},
        "training_step": 2048,
        "optimizer": {"state": {1: {"step": 2}}},
        "training_state": {"replay": {"size": 100}, "level_config_id": "verified"},
    }
    assert overnight.resume_compatible(payload, config, "verified")
    assert not overnight.resume_compatible(payload, config, "different")
    for key in ("optimizer", "training_state"):
        changed = deepcopy(payload)
        changed[key] = {}
        assert not overnight.resume_compatible(changed, config, "verified")
    changed = deepcopy(payload)
    changed["training_config"]["learning_rate"] = 0.01
    assert not overnight.resume_compatible(changed, config, "verified")
    changed = deepcopy(payload)
    changed["training_step"] = 4096
    assert not overnight.resume_compatible(changed, config, "verified")


def test_budget_enforces_reserve_and_expired_jobs_never_start(tmp_path, monkeypatch):
    assert overnight.can_start(100, 500, 100, 200)
    assert not overnight.can_start(100, 500, 100, 300)
    monkeypatch.setattr(
        overnight.subprocess,
        "Popen",
        lambda *a, **k: pytest.fail("expired job launched"),
    )
    assert overnight.execute(tmp_path, {"id": "expired"}, cutoff=0) is None
    (tmp_path / "STOP").touch()
    assert overnight.execute(tmp_path, {"id": "stopped"}, cutoff=1e20) is None


def test_failed_job_retries_in_new_directory_then_resumes_cached_result(
    tmp_path, monkeypatch
):
    spec = {"id": "bounded_job", "kind": "ddqn", "config": {"hidden_dims": (256, 128)}}
    attempts = []

    class Process:
        pid = 123
        returncode = 1

        def __init__(self, args, **kwargs):
            attempt = Path(args[args.index("--attempt") + 1])
            attempts.append(attempt)
            if len(attempts) == 2:
                overnight.atomic_json(
                    attempt / "result.json", {"status": "complete", "spec": spec}
                )
                self.returncode = 0

        def poll(self):
            return self.returncode

    monkeypatch.setattr(overnight.subprocess, "Popen", Process)
    result = overnight.execute(tmp_path, spec, cutoff=1e20)
    assert result["status"] == "complete" and len(attempts) == 2
    assert (attempts[0] / "failure.json").exists()
    assert (attempts[0] / "worker.log").exists()
    assert attempts[0] != attempts[1]
    assert overnight.execute(tmp_path, spec, cutoff=1e20) == result
    assert len(attempts) == 2
    with pytest.raises(ValueError, match="different specification"):
        overnight.execute(tmp_path, {**spec, "kind": "ppo"}, cutoff=1e20)


def test_launch_prevents_duplicate_controller(tmp_path, monkeypatch):
    monkeypatch.setattr(
        overnight, "verify", lambda _: {"created_epoch": 0, "wall_seconds": 1e20}
    )
    (tmp_path / "smoke_passed.json").write_text("{}")
    with (tmp_path / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(BlockingIOError):
            overnight.launch(tmp_path)


def test_smoke_gate_blocks_full_launch(tmp_path, monkeypatch):
    monkeypatch.setattr(overnight, "verify", lambda _: {})
    with pytest.raises(RuntimeError, match="Smoke tests"):
        overnight.launch(tmp_path)


def test_plateau_requires_multiple_metrics_and_sufficient_training():
    curve = [{"win_rate": 0.2, "mean_objective_completion_ratio": 0.5}] * 3
    result = {"environment_steps": 8192, "curve": curve, "best_validation": curve[0]}
    assert overnight.plateau(result, 0.5)
    assert not overnight.plateau({**result, "environment_steps": 4096}, 0.5)
    assert not overnight.plateau(
        {
            **result,
            "curve": [
                curve[0],
                {"win_rate": 0.2, "mean_objective_completion_ratio": 0.7},
                curve[0],
            ],
        },
        0.5,
    )
    assert not overnight.plateau(result, 0.25)


def test_archived_environment_and_paired_evaluation_reproduce(tmp_path):
    archive = Path(__file__).resolve().parent / "fixtures/calibration"
    archived = json.loads((archive / "manifest.json").read_text())
    variant = overnight.VARIANTS[0]
    level = overnight.load_level(archive / archived["configurations"][variant]["path"])
    config = type(
        "Config",
        (),
        {
            "level_id": level.level_id,
            "max_cascades": 200,
            "max_initialization_attempts": 1000,
        },
    )
    env = overnight.adapted_environment(config, level)
    try:
        first, first_info = env.reset(seed=810000)
        second, second_info = env.reset(seed=810000)
        assert level.move_limit == 6
        assert overnight.observation_fingerprint(
            first
        ) == overnight.observation_fingerprint(second)
        assert first_info["board_seed"] == second_info["board_seed"]
        assert env.observation_space.contains(first)
    finally:
        env.close()
    manifest = {
        "validation_seeds": [810000, 810001],
        "validation_random_start": 830000,
        "experiment_id": "test",
    }
    spec = {"variant": variant, "kind": "baseline"}
    comparisons = []
    for name in ("first", "second"):
        attempt = tmp_path / name
        attempt.mkdir()
        result = overnight.evaluate(tmp_path, attempt, spec, manifest, level)
        assert result["paired_boards_verified"] and result["episodes"] == 4
        comparisons.append(
            [
                (
                    row["agent_name"],
                    row["win_rate"],
                    row["mean_total_reward"],
                    row["mean_moves_used"],
                )
                for row in result["summary"]
            ]
        )
    assert comparisons[0] == comparisons[1]
