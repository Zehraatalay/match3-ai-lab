"""Audited result reuse and paired descriptive MCTS factorial comparisons."""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from dataclasses import asdict, fields
from pathlib import Path

import numpy as np

from match3_ai.evaluation.behavior_statistics import paired_difference
from match3_ai.evaluation.mcts_provenance import (
    ALGORITHM_SEMANTICS,
    IMPLEMENTATION_VERSION,
    LEGACY_CLI_SHA256,
    LEGACY_IMPLEMENTATION,
    LEGACY_PACKAGE_SHA256,
)
from match3_ai.evaluation.runner import EpisodeResult

INT_FIELDS = {
    "seed",
    "steps",
    "valid_actions",
    "invalid_actions",
    "total_cleared",
    "total_cascades",
    "reshuffles",
    "agent_seed",
    "effective_episode_seed",
    "board_seed",
    "engine_seed",
    "moves_used",
    "blockers_remaining",
    "blocker_hp_remaining",
    "special_activations",
    "max_cascades_in_move",
    "cascade_limit",
}
FLOAT_FIELDS = {"total_reward", "objective_completion_ratio", "inference_time_seconds"}
JSON_FIELDS = {"objective_progress", "error_context"}


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode_episode(row: dict) -> EpisodeResult:
    values = {}
    for field in fields(EpisodeResult):
        if field.name not in row:
            raise ValueError(f"Missing historical episode field: {field.name}")
        value = row[field.name]
        if value == "":
            value = None
        elif field.name in INT_FIELDS:
            value = int(value)
        elif field.name in FLOAT_FIELDS:
            value = float(value)
            if not math.isfinite(value):
                raise ValueError("Nonfinite historical metric")
        elif field.name in JSON_FIELDS:
            value = json.loads(value)
        elif field.name == "completed":
            if value not in ("True", "False"):
                raise ValueError("Invalid completion flag")
            value = value == "True"
        values[field.name] = value
    return EpisodeResult(**values)


def validate_reuse(config, source: Path, target_manifest: dict | None = None):
    """Verify a complete source, reconstruct its identities, and select eligible rows.

    This performs resets, never gameplay or search. Incompatible input is rejected
    before creating an output directory or running any new evaluation episodes.
    """
    from match3_ai.evaluation.mcts_baseline_benchmark import (
        AGENT_REGISTRY,
        LEVELS,
        BenchmarkConfig,
        BenchmarkEnvironment,
        build_manifest,
    )

    source = source.resolve()
    raw_manifest = (source / "manifest.json").read_bytes()
    raw_episodes = (source / "episodes.csv").read_bytes()
    manifest_hash = hashlib.sha256(raw_manifest).hexdigest()
    episodes_hash = hashlib.sha256(raw_episodes).hexdigest()
    manifest = json.loads(raw_manifest)
    target = target_manifest or build_manifest(config)
    if manifest.get("status") != "complete":
        raise ValueError("Historical benchmark is incomplete or contains errors")
    legacy = "schema_version" not in manifest
    if legacy:
        if (
            manifest.get("package_source_sha256") != LEGACY_PACKAGE_SHA256
            or manifest.get("cli_sha256") != LEGACY_CLI_SHA256
        ):
            raise ValueError(
                "Historical sources have no audited compatibility certificate"
            )
        implementation = LEGACY_IMPLEMENTATION
    else:
        if (
            manifest.get("schema_version") != 2
            or manifest.get("algorithm_semantics") != ALGORITHM_SEMANTICS
            or manifest.get("semantic_source_sha256")
            != target["semantic_source_sha256"]
            or manifest.get("implementation_version") != IMPLEMENTATION_VERSION
        ):
            raise ValueError(
                "Historical algorithm semantics or critical-source hashes differ"
            )
        implementation = IMPLEMENTATION_VERSION
    for name in (
        "level_sha256",
        "versions",
        "max_cascades",
        "max_initialization_attempts",
        "denominators",
    ):
        if manifest.get(name) != target[name]:
            raise ValueError(f"Historical {name} differs")
    for name in ("episodes", "simulations", "env_seed_start", "agent_seed_start"):
        if manifest["config"].get(name) != getattr(config, name):
            raise ValueError(f"Historical {name} differs")
    source_names = tuple(manifest["agents"])
    source_config = BenchmarkConfig(
        config.episodes,
        config.simulations,
        config.env_seed_start,
        config.agent_seed_start,
        source_names,
    )
    seeds = list(range(config.env_seed_start, config.env_seed_start + config.episodes))
    if manifest.get("environment_seeds") != seeds:
        raise ValueError("Historical environment seeds differ")
    for name, settings in manifest["agents"].items():
        if name not in AGENT_REGISTRY:
            raise ValueError("Unknown historical configuration")
        options = AGENT_REGISTRY[name]
        expected = (
            {
                "rollout_policy": options[0],
                "value_mode": options[1],
                "simulations": config.simulations,
                "exploration_constant": math.sqrt(2),
                "win_efficiency_weight": 0.2,
                "loss_progress_weight": 0.2,
            }
            if options
            else {"policy": name}
        )
        if settings != expected:
            raise ValueError(f"Historical configuration options differ: {name}")
    expected_keys = {
        (level, seed, name)
        for level in LEVELS
        for seed in seeds
        for name in source_names
    }
    rows = list(csv.DictReader(io.StringIO(raw_episodes.decode("utf-8"))))
    if (
        len(rows) != len(expected_keys)
        or manifest.get("records_written") != len(rows)
        or manifest.get("planned_episodes") != len(rows)
        or manifest.get("attempted_episodes") != len(rows)
    ):
        raise ValueError("Historical episode coverage is incomplete")
    identities = {}
    for level in LEVELS:
        env = BenchmarkEnvironment(level)
        try:
            for seed in seeds:
                env.reset(seed=seed)
                identities[(level, seed)] = env.initial_identity
        finally:
            env.close()
    seen, ids, eligible = set(), set(), {}
    for row in rows:
        record = decode_episode(row)
        key = (record.level_id, record.seed, record.agent_name)
        if key not in expected_keys or key in seen or record.episode_id in ids:
            raise ValueError("Duplicate or unexpected historical episode identity")
        seen.add(key)
        ids.add(record.episode_id)
        if (
            record.experiment_id != manifest["experiment_id"]
            or not record.completed
            or record.outcome not in ("WON", "LOST")
            or record.invalid_actions != 0
            or record.error_type is not None
            or record.error_message is not None
        ):
            raise ValueError(
                "Historical episode has errors or invalid completion metadata"
            )
        index = list(AGENT_REGISTRY).index(record.agent_name)
        expected_agent_seed = source_config.agent_seed(
            LEVELS.index(record.level_id), record.seed - config.env_seed_start, index
        )
        if (
            record.agent_seed != expected_agent_seed
            or record.effective_episode_seed != record.seed
        ):
            raise ValueError("Historical environment or agent seed mismatch")
        identity = (
            row["initial_fingerprint"],
            record.board_seed,
            record.engine_seed,
            record.level_config_id,
        )
        if (
            identity != identities[(record.level_id, record.seed)]
            or row["pairing_verified"] != "True"
        ):
            raise ValueError("Historical initial-state fingerprint or seed mismatch")
        is_mcts = AGENT_REGISTRY[record.agent_name] is not None
        expected_budget = str(config.simulations) if is_mcts else ""
        if row["simulations"] != expected_budget:
            raise ValueError("Historical episode simulation budget mismatch")
        calls = int(row["action_selection_calls"])
        decision_time, episode_time = (
            float(row["decision_time_seconds"]),
            float(row["episode_time_seconds"]),
        )
        if (
            calls != record.steps
            or calls <= 0
            or not all(
                math.isfinite(value) and value >= 0
                for value in (decision_time, episode_time)
            )
            or record.objective_completion_ratio is None
            or not 0 <= record.objective_completion_ratio <= 1
            or record.moves_used is None
        ):
            raise ValueError("Historical episode metrics are incomplete or invalid")
        if (
            record.inference_time_seconds is None
            or record.inference_time_seconds < 0
            or record.total_reward != (1 if record.outcome == "WON" else -1)
            or record.moves_used != record.valid_actions
            or record.objective_progress is None
        ):
            raise ValueError("Historical completed-episode metrics are inconsistent")
        row_implementation = row.get("implementation_version") or implementation
        if row_implementation not in (LEGACY_IMPLEMENTATION, IMPLEMENTATION_VERSION):
            raise ValueError("Unknown historical timing implementation")
        if record.agent_name in config.agents:
            extra = {
                "episode_id": record.episode_id,
                "initial_fingerprint": row["initial_fingerprint"],
                "pairing_verified": True,
                "action_selection_calls": calls,
                "decision_time_seconds": decision_time,
                "episode_time_seconds": episode_time,
                "simulations": config.simulations if is_mcts else None,
                "implementation_version": row_implementation,
                "record_origin": "imported",
                "origin_episode_id": record.episode_id,
                "origin_manifest_sha256": manifest_hash,
                "origin_episodes_sha256": episodes_hash,
            }
            eligible[key] = record, extra
    if seen != expected_keys:
        raise ValueError("Missing historical identities")
    if (
        digest(source / "manifest.json") != manifest_hash
        or digest(source / "episodes.csv") != episodes_hash
    ):
        raise ValueError("Historical input changed during validation")
    return eligible, {
        "directory": str(source),
        "manifest_sha256": manifest_hash,
        "episodes_sha256": episodes_hash,
        "source_package_sha256": manifest["package_source_sha256"],
        "algorithm_semantics": ALGORITHM_SEMANTICS,
        "records_verified": len(rows),
        "records_reusable": len(eligible),
        "new_episodes_required": len(LEVELS) * config.episodes * len(config.agents)
        - len(eligible),
        "compatibility_basis": "audited critical-source boundary and exact reference-trace equivalence; unchanged valid-input search semantics",
        "timing_warning": "Historical recomputation and optimized observation-reuse timings are labeled separately; do not interpret mixed-version latency contrasts",
    }


CONTRASTS = {
    "rollout_terminal_GT_minus_RT": {
        "MCTS_greedy_terminal": 1,
        "MCTS_random_terminal": -1,
    },
    "rollout_outcome_GO_minus_RO": {
        "MCTS_greedy_outcome": 1,
        "MCTS_random_outcome": -1,
    },
    "value_random_RO_minus_RT": {"MCTS_random_outcome": 1, "MCTS_random_terminal": -1},
    "value_greedy_GO_minus_GT": {"MCTS_greedy_outcome": 1, "MCTS_greedy_terminal": -1},
    "interaction": {
        "MCTS_greedy_outcome": 1,
        "MCTS_greedy_terminal": -1,
        "MCTS_random_outcome": -1,
        "MCTS_random_terminal": 1,
    },
}


def paired_comparisons(rows: list[dict], *, samples=5000, seed=51000) -> list[dict]:
    """Compare complete matched boards; bootstrap whole-board contrasts jointly."""
    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if not rows:
        raise ValueError("No ablation records")
    indexed = {}
    for row in rows:
        key = (row["level_id"], int(row["seed"]), row["agent_name"])
        if key in indexed:
            raise ValueError("Duplicate ablation board/configuration identity")
        indexed[key] = row
    results = []
    for level in sorted({row["level_id"] for row in rows}):
        budgets = {
            row["simulations"]
            for row in rows
            if row["level_id"] == level and row["agent_name"].startswith("MCTS_")
        }
        if len(budgets) != 1:
            raise ValueError("Compare one common simulation budget at a time")
        for contrast, coefficients in CONTRASTS.items():
            seed_sets = [
                {key[1] for key in indexed if key[0] == level and key[2] == name}
                for name in coefficients
            ]
            if not seed_sets[0] or any(values != seed_sets[0] for values in seed_sets):
                raise ValueError(f"Unmatched board coverage for {level}, {contrast}")
            aligned = [
                {name: indexed[(level, board_seed, name)] for name in coefficients}
                for board_seed in sorted(seed_sets[0])
            ]
            for board in aligned:
                identities = {
                    (
                        row["initial_fingerprint"],
                        row["board_seed"],
                        row["engine_seed"],
                        row["level_config_id"],
                    )
                    for row in board.values()
                }
                if len(identities) != 1:
                    raise ValueError(
                        "Paired ablation fingerprints or engine seeds disagree"
                    )
            implementations = {
                row.get("implementation_version", LEGACY_IMPLEMENTATION)
                for board in aligned
                for row in board.values()
            }
            for metric in (
                "win",
                "objective_completion",
                "decision_latency",
                "episode_time",
            ):
                base = {
                    "level_id": level,
                    "contrast": contrast,
                    "metric": metric,
                    "planned_pairs": len(aligned),
                    "implementation_versions": ";".join(sorted(implementations)),
                }
                if (
                    metric in ("decision_latency", "episode_time")
                    and len(implementations) > 1
                ):
                    results.append(
                        {
                            **base,
                            "n_pairs": 0,
                            "excluded_pairs": len(aligned),
                            "status": "mixed_timing_versions",
                            "difference": None,
                            "ci95_low": None,
                            "ci95_high": None,
                            "left_only_wins": None,
                            "right_only_wins": None,
                            "positive_board_contrasts": None,
                            "negative_board_contrasts": None,
                            "zero_board_contrasts": None,
                        }
                    )
                    continue
                left, right, discordant_left, discordant_right = [], [], 0, 0
                for board in aligned:
                    if any(
                        row["outcome"] not in ("WON", "LOST") for row in board.values()
                    ):
                        continue
                    values = {}
                    for name, row in board.items():
                        values[name] = (
                            int(row["outcome"] == "WON")
                            if metric == "win"
                            else row.get("objective_completion_ratio")
                            if metric == "objective_completion"
                            else row["decision_time_seconds"]
                            / row["action_selection_calls"]
                            if metric == "decision_latency"
                            and row["action_selection_calls"]
                            else row.get("episode_time_seconds")
                            if metric == "episode_time"
                            else None
                        )
                    if any(
                        value is None or not math.isfinite(value)
                        for value in values.values()
                    ):
                        continue
                    left.append(
                        sum(
                            value * coefficients[name]
                            for name, value in values.items()
                            if coefficients[name] > 0
                        )
                    )
                    right.append(
                        sum(
                            -value * coefficients[name]
                            for name, value in values.items()
                            if coefficients[name] < 0
                        )
                    )
                    if metric == "win" and len(coefficients) == 2:
                        discordant_left += left[-1] == 1 and right[-1] == 0
                        discordant_right += right[-1] == 1 and left[-1] == 0
                effect = (
                    paired_difference(left, right, seed=seed, samples=samples)
                    if left
                    else None
                )
                differences = np.asarray(left) - np.asarray(right)
                results.append(
                    {
                        **base,
                        "n_pairs": len(left),
                        "excluded_pairs": len(aligned) - len(left),
                        "status": "exploratory" if effect else "no_complete_pairs",
                        "difference": effect["absolute_difference"] if effect else None,
                        "ci95_low": effect["ci95"][0] if effect else None,
                        "ci95_high": effect["ci95"][1] if effect else None,
                        "left_only_wins": int(discordant_left)
                        if metric == "win" and len(coefficients) == 2
                        else None,
                        "right_only_wins": int(discordant_right)
                        if metric == "win" and len(coefficients) == 2
                        else None,
                        "positive_board_contrasts": int((differences > 0).sum()),
                        "negative_board_contrasts": int((differences < 0).sum()),
                        "zero_board_contrasts": int((differences == 0).sum()),
                    }
                )
    return results


def analyze_results(
    source: Path, output: Path, *, samples=5000, seed=51000
) -> list[dict]:
    from match3_ai.evaluation.analytics import _csv_value
    from match3_ai.evaluation.mcts_baseline_benchmark import (
        BenchmarkConfig,
        atomic_json,
        build_manifest,
    )

    if output.exists():
        raise FileExistsError(output)
    manifest = json.loads((source / "manifest.json").read_text())
    config = BenchmarkConfig(**manifest["config"])
    validated, _ = validate_reuse(config, source, build_manifest(config))
    rows = [{**asdict(record), **extras} for record, extras in validated.values()]
    comparisons = paired_comparisons(rows, samples=samples, seed=seed)
    output.mkdir(parents=True, exist_ok=False)
    with (output / "paired_comparisons.csv").open("x", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(comparisons[0]))
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(value) for key, value in row.items()}
            for row in comparisons
        )
    atomic_json(
        output / "analysis_manifest.json",
        {
            "source": str(source.resolve()),
            "source_manifest_sha256": digest(source / "manifest.json"),
            "source_episodes_sha256": digest(source / "episodes.csv"),
            "bootstrap_seed": seed,
            "bootstrap_samples": samples,
            "interpretation": "Exploratory paired board-bootstrap intervals; no significance tests or causal/superiority claims. Conditional on one agent RNG stream per board. Timing contrasts are withheld across implementation versions.",
        },
    )
    return comparisons
