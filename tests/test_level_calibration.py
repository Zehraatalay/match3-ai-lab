"""Lightweight calibration configuration, statistics, and persistence checks."""

import json
from dataclasses import asdict, replace

import numpy as np
import pytest

from match3_ai.evaluation import level_calibration as calibration
from match3_ai.evaluation.runner import EpisodeResult
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.loader import load_level


def resolved(tmp_path):
    result = {}
    for name, spec in calibration.resolved_variants().items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(spec["config"]))
        result[name] = {"config": load_level(path), "factor": spec["factor"]}
    return result


def episode(agent="Random", outcome="LOST", **kwargs):
    defaults = {
        "seed": 760000,
        "total_reward": 0,
        "steps": 10,
        "valid_actions": 10,
        "invalid_actions": 0,
        "total_cleared": 0,
        "total_cascades": 10,
        "reshuffles": 0,
        "completed": True,
        "level_id": "wooden-trouble",
        "agent_name": agent,
        "outcome": outcome,
        "moves_used": 10,
        "blockers_remaining": 2,
        "blocker_hp_remaining": 2,
        "objective_completion_ratio": 0.5,
    }
    return EpisodeResult(**(defaults | kwargs))


def test_all_twelve_variants_change_only_one_factor(tmp_path):
    variants = calibration.resolved_variants()
    assert len(variants) == 12
    configs = resolved(tmp_path)
    for name, spec in variants.items():
        original = variants[f"{spec['config']['level_id']}__original"]["config"]
        changes = {key for key in original if original[key] != spec["config"][key]}
        assert changes == ({spec["factor"]} if spec["factor"] else set())
        assert configs[name]["config"].level_id == spec["config"]["level_id"]
    assert (
        sum(b.hp for b in configs["double-trouble__triple-crates"]["config"].blockers)
        == 15
    )
    assert (
        sum(b.hp for b in configs["frozen-garden__thick-ice"]["config"].blockers) == 16
    )


def test_config_injection_is_local_and_retains_seed_namespace():
    original = load_level(calibration.LEVEL_ROOT / "double-trouble.json")
    variant = replace(original, move_limit=10)
    plain = Match3LevelEnv("double-trouble")
    injected = Match3LevelEnv("double-trouble", level_config=variant)
    other = Match3LevelEnv("double-trouble")
    try:
        a, ai = plain.reset(seed=123)
        b, bi = injected.reset(seed=123)
        assert plain.level_ids == injected.level_ids
        assert (ai["board_seed"], ai["engine_seed"]) == (
            bi["board_seed"],
            bi["engine_seed"],
        )
        for key in ("tiles", "specials", "blockers", "blocker_hp", "action_mask"):
            np.testing.assert_array_equal(a[key], b[key])
        assert ai["level_config_id"] != bi["level_config_id"]
        assert other.config == original
        assert b["moves_remaining"] == 10
    finally:
        plain.close()
        injected.close()
        other.close()
    with pytest.raises(ValueError, match="retain"):
        Match3LevelEnv("wooden-trouble", level_config=variant)
    with pytest.raises(ValueError, match="padded"):
        Match3LevelEnv(
            "double-trouble", level_config=replace(original, rows=10, cell_mask=None)
        )
    with pytest.raises(TypeError, match="LevelConfig"):
        Match3LevelEnv("double-trouble", level_config={})


def test_pairing_checks_engine_seed_before_action(tmp_path):
    from match3_ai.evaluation.gameplay_events import GameplayCollector

    config = load_level(calibration.LEVEL_ROOT / "double-trouble.json")
    identities = {}
    with (tmp_path / "events.jsonl").open("x") as handle:
        for name in ("Random", "Greedy"):
            env = calibration.ScreeningEnvironment(
                config, GameplayCollector(handle, "test", name), identities
            )
            env.reset(seed=123)
            env.close()
        identities[123] = dict(identities[123], engine_seed=-1)
        env = calibration.ScreeningEnvironment(
            config, GameplayCollector(handle, "test", "bad"), identities
        )
        with pytest.raises(RuntimeError, match="engine seed"):
            env.reset(seed=123)
        assert env.pairing_error
        env.close()


def test_report_metrics_and_targets(tmp_path):
    configs = resolved(tmp_path)
    variant = "wooden-trouble__original"
    records = []
    for agent, wins in (("Random", 20), ("Greedy", 30)):
        for i in range(50):
            result = episode(
                agent,
                "WON" if i < wins else "LOST",
                moves_used=5 if i < wins else 10,
                blocker_hp_remaining=0 if i < wins else 2,
            )
            records.append({"variant_id": variant, "record": asdict(result)})
    moves = {(variant, "Random"): [1, 2, 21, 100], (variant, "Greedy"): [1, 2, 3]}
    summary = calibration.summarize(records, moves, configs)
    random = next(row for row in summary["groups"] if row["agent_name"] == "Random")
    assert random["win_rate"] == 0.4
    assert random["move_budget_utilization_wins"] == 0.5
    assert random["move_budget_utilization_losses"] == 1
    assert random["mean_normalized_hp_removed"] == 0.7
    assert random["cascade_median"] == 11.5
    assert random["cascade_frequency_above20"] == 0.5
    assert random["win_rate_ci95"][0] < 0.4 < random["win_rate_ci95"][1]
    assert summary["candidates"][variant] == {
        "meets_provisional_targets": True,
        "cascade_review_required": True,
    }
    calibration.save_report(tmp_path, records, moves, configs, "complete")
    assert "no final variant selected" in (tmp_path / "report.md").read_text()
    assert not calibration.summarize(records[:1], moves, configs)["candidates"][
        variant
    ]["meets_provisional_targets"]


def test_failure_preserves_incremental_episode_without_retry(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "validate_seed_plan", lambda _: None)
    variants = calibration.resolved_variants()
    monkeypatch.setattr(
        calibration, "resolved_variants", lambda: dict(list(variants.items())[:1])
    )
    monkeypatch.setattr(calibration, "ENVIRONMENT_SEEDS", (760000,))
    calls = []

    class FakeEnv:
        initial = None
        pairing_error = False

        def __init__(self, *args):
            pass

        def close(self):
            pass

    def fake_run(env, agent, **kwargs):
        calls.append(kwargs)
        if len(calls) == 2:
            raise RuntimeError("interrupted fixture")
        record = episode(
            kwargs["agent_name"],
            "ERROR",
            steps=0,
            valid_actions=0,
            moves_used=None,
            blocker_hp_remaining=None,
            experiment_id=kwargs["experiment_id"],
            episode_id=kwargs["episode_id"],
            error_type="LevelGenerationError",
            error_message="fixture generation failure",
        )
        kwargs["collector"].record_episode(record)
        return record

    monkeypatch.setattr(calibration, "ScreeningEnvironment", FakeEnv)
    monkeypatch.setattr(calibration, "run_episode", fake_run)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="interrupted fixture"):
        calibration.run_screening(tmp_path, output)
    assert len((output / "episodes.jsonl").read_text().splitlines()) == 1
    assert json.loads((output / "failure.json").read_text())["completed_records"] == 1
    summary = json.loads((output / "summary.json").read_text())
    assert summary["groups"][0]["generation_failures"] == 1
    assert summary["groups"][0]["cascade_median"] is None
    assert summary["status"] == "interrupted"
    assert len(calls) == 2
    with pytest.raises(FileExistsError):
        calibration.run_screening(tmp_path, output)


def test_protected_seed_and_saved_record_checks(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "ENVIRONMENT_SEEDS", tuple(range(730000, 730050)))
    with pytest.raises(ValueError, match="protected"):
        calibration.validate_seed_plan(tmp_path)
    monkeypatch.setattr(calibration, "ENVIRONMENT_SEEDS", tuple(range(760000, 760050)))
    calibration.validate_seed_plan(tmp_path)
    (tmp_path / "events.jsonl").write_text('{"record":{"agent_seed":762001}}\n')
    with pytest.raises(ValueError, match="already used"):
        calibration.validate_seed_plan(tmp_path)


def test_stage1b_exact_combined_candidates_and_loader_roundtrip(tmp_path):
    expected = {
        "wooden-trouble": {(6, 4), (5, 4), (10, 8), (6, 8)},
        "double-trouble": {(8, 8), (6, 8), (15, 15), (8, 15)},
        "frozen-garden": {(10, 12), (8, 12), (15, 16), (10, 16)},
    }
    variants = calibration.resolved_variants("1b")
    assert len(variants) == 12
    actual = {level: set() for level in expected}
    for name, spec in variants.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(spec["config"]))
        config = load_level(path)
        assert config.num_colors == 5
        actual[config.level_id].add(
            (config.move_limit, sum(b.hp for b in config.blockers))
        )
        original = json.loads(
            (calibration.LEVEL_ROOT / f"{config.level_id}.json").read_text()
        )
        assert {key for key in original if original[key] != spec["config"][key]} == set(
            spec["factor"]
        )
    assert actual == expected
    with pytest.raises(ValueError, match="stage"):
        calibration.resolved_variants("confirmation")


def test_stage1b_seeds_exclude_stage1_and_used_records(tmp_path, monkeypatch):
    calibration.validate_seed_plan(tmp_path, "1b")
    (tmp_path / "episodes.jsonl").write_text('{"record":{"seed":780049}}\n')
    with pytest.raises(ValueError, match="already used"):
        calibration.validate_seed_plan(tmp_path, "1b")
    monkeypatch.setattr(
        calibration, "STAGE1B_ENVIRONMENT_SEEDS", calibration.ENVIRONMENT_SEEDS
    )
    with pytest.raises(ValueError, match="protected"):
        calibration.validate_seed_plan(tmp_path, "1b")


def test_stage1b_descriptive_comparison_and_stage1_validation(tmp_path):
    prior = {"status": "complete", "groups": []}
    for variant, spec in calibration.resolved_variants().items():
        for agent in ("Random", "Greedy"):
            prior["groups"].append(
                {
                    "variant_id": variant,
                    "agent_name": agent,
                    "attempted": 50,
                    "errors": 0,
                    "time_limits": 0,
                    "win_rate": 0.9,
                    "mean_moves_used": 3,
                }
            )
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior))
    assert calibration.load_stage1_summary(path) == prior
    variants = calibration.resolved_variants("1b")
    name, spec = next(iter(variants.items()))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(spec["config"]))
    configs = {name: {"config": load_level(config_path), "factor": spec["factor"]}}
    record = episode(outcome="WON", moves_used=2)
    records = [{"variant_id": name, "record": asdict(record)}]
    calibration.save_report(
        tmp_path, records, {}, configs, "interrupted", stage="1b", prior_summary=prior
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    comparisons = summary["stage1_descriptive_comparison"]
    assert len(comparisons) == 2
    assert {row["reference_variant_id"] for row in comparisons} == {
        "wooden-trouble__original",
        "wooden-trouble__colors5",
    }
    assert all(not row["board_matched"] for row in comparisons)
    assert comparisons[0]["win_rate_difference_pp"] == pytest.approx(10)
    assert comparisons[0]["mean_moves_difference"] == -1
    report = (tmp_path / "report.md").read_text()
    assert "Stage 1B" in report and "do not identify causal effects" in report
    prior["status"] = "interrupted"
    path.write_text(json.dumps(prior))
    with pytest.raises(ValueError, match="complete"):
        calibration.load_stage1_summary(path)


def confirmation_artifacts(tmp_path):
    directory = tmp_path / "stage1b"
    (directory / "configs").mkdir(parents=True)
    manifest = {
        "stage": "screening_1b",
        "configurations": {},
        "packaged_config_sha256": {},
    }
    groups = []
    for name, spec in calibration.resolved_variants("1b").items():
        path = directory / "configs" / f"{name}.json"
        calibration.write_json(path, spec["config"])
        manifest["configurations"][name] = {
            "path": f"configs/{name}.json",
            "sha256": calibration.sha256(path),
            "level_id": spec["config"]["level_id"],
            "factor": spec["factor"],
        }
        for agent in ("Random", "Greedy"):
            groups.append(
                {
                    "variant_id": name,
                    "agent_name": agent,
                    "attempted": 50,
                    "errors": 0,
                    "time_limits": 0,
                    "win_rate": 0.5,
                    "mean_moves_used": 4,
                }
            )
    manifest["packaged_config_sha256"] = {
        level: calibration.sha256(calibration.LEVEL_ROOT / f"{level}.json")
        for level in calibration.LEVELS
    }
    calibration.write_json(directory / "manifest.json", manifest)
    calibration.write_json(
        directory / "summary.json", {"status": "complete", "groups": groups}
    )
    return directory


def test_confirmation_loads_pinned_bytes_and_originals(tmp_path):
    directory = confirmation_artifacts(tmp_path)
    variants, provenance, _prior = calibration.load_confirmation_inputs(directory)
    assert len(variants) == 6
    assert provenance["selected_config_sha256"] == calibration.FROZEN_SELECTION
    assert not provenance["retuning_or_replacement_allowed"]
    for name in calibration.FROZEN_SELECTION:
        assert variants[name]["source_sha256"] == calibration.FROZEN_SELECTION[name]
    name = next(iter(calibration.FROZEN_SELECTION))
    path = directory / "configs" / f"{name}.json"
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="hash"):
        calibration.load_confirmation_inputs(directory)


def test_confirmation_rejects_changed_original_hash(tmp_path):
    directory = confirmation_artifacts(tmp_path)
    path = directory / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["packaged_config_sha256"]["wooden-trouble"] = "wrong"
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="Original packaged"):
        calibration.load_confirmation_inputs(directory)


def test_confirmation_seeds_and_historical_collisions(tmp_path, monkeypatch):
    calibration.validate_seed_plan(tmp_path, "2")
    assert calibration.screening_seeds("2") == (
        tuple(range(761000, 761200)),
        tuple(range(763000, 763200)),
    )
    (tmp_path / "events.jsonl").write_text(
        '{"info":{"effective_episode_seed":761199}}\n'
    )
    with pytest.raises(ValueError, match="already used"):
        calibration.validate_seed_plan(tmp_path, "2")
    monkeypatch.setattr(
        calibration, "CONFIRMATION_ENVIRONMENT_SEEDS", tuple(range(780000, 780200))
    )
    with pytest.raises(ValueError, match="protected"):
        calibration.validate_seed_plan(tmp_path, "2")


def test_confirmation_200_episode_rules_intervals_and_comparison(tmp_path):
    directory = confirmation_artifacts(tmp_path)
    variants, _, prior = calibration.load_confirmation_inputs(directory)
    name = next(iter(calibration.FROZEN_SELECTION))
    config = load_level(variants[name]["source_path"])
    configs = {name: {"config": config, "factor": variants[name]["factor"]}}
    records = []
    for agent, wins in (("Random", 80), ("Greedy", 120)):
        for index in range(200):
            record = episode(
                agent,
                "WON" if index < wins else "LOST",
                moves_used=3 if index < wins else 6,
            )
            records.append({"variant_id": name, "record": asdict(record)})
    calibration.save_report(
        tmp_path,
        records,
        {(name, "Random"): [1] * 200, (name, "Greedy"): [2] * 200},
        configs,
        "complete",
        stage="2",
        prior_summary=prior,
    )
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["candidates"][name]["meets_confirmation_targets"]
    assert len(summary["stage1b_descriptive_comparison"]) == 2
    assert all(row["first_move_win_ci95"][1] > 0 for row in summary["groups"])
    assert all(row["win_interval_vs_target"] == "within" for row in summary["groups"])
    assert "not retuned or replaced" in (tmp_path / "report.md").read_text()
    bad = calibration.summarize(
        records, {(name, "Random"): [100]}, configs, expected_episodes=200
    )
    calibration.confirmation_decisions(bad)
    assert not bad["candidates"][name]["meets_confirmation_targets"]
    partial = calibration.summarize(records[:50], {}, configs, expected_episodes=200)
    calibration.confirmation_decisions(partial)
    assert not partial["candidates"][name]["meets_confirmation_targets"]


def test_confirmation_runner_copies_frozen_artifacts_and_keeps_failures(
    tmp_path, monkeypatch
):
    directory = confirmation_artifacts(tmp_path)
    monkeypatch.setattr(calibration, "CONFIRMATION_ENVIRONMENT_SEEDS", (761000,))
    monkeypatch.setattr(calibration, "CONFIRMATION_RANDOM_SEEDS", (763000,))
    monkeypatch.setattr(calibration, "validate_seed_plan", lambda *args: None)
    calls = []

    class FakeEnv:
        initial = None
        pairing_error = False

        def __init__(self, *args):
            pass

        def close(self):
            pass

    def fake_run(env, agent, **kwargs):
        calls.append(kwargs)
        record = episode(
            kwargs["agent_name"],
            "ERROR",
            steps=0,
            valid_actions=0,
            moves_used=None,
            blocker_hp_remaining=None,
            experiment_id=kwargs["experiment_id"],
            episode_id=kwargs["episode_id"],
            agent_seed=kwargs["agent_seed"],
            error_type="LevelGenerationError",
        )
        kwargs["collector"].record_episode(record)
        return record

    monkeypatch.setattr(calibration, "ScreeningEnvironment", FakeEnv)
    monkeypatch.setattr(calibration, "run_episode", fake_run)
    output = tmp_path / "confirmation"
    calibration.run_screening(tmp_path, output, stage="2", stage1b_dir=directory)
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["stage"] == "confirmation"
    assert len(manifest["configurations"]) == 6
    assert len(calls) == 12
    assert all(call["seed"] == 761000 for call in calls)
    assert all(
        call["agent_seed"] == 763000 for call in calls if call["agent_name"] == "Random"
    )
    assert len((output / "episodes.jsonl").read_text().splitlines()) == 12
    for name, digest in calibration.FROZEN_SELECTION.items():
        assert calibration.sha256(output / "configs" / f"{name}.json") == digest
    assert "frozen_selection_provenance" in manifest
