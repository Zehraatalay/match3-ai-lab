"""Baseline-only calibration screening and independent frozen confirmation."""

import argparse
import csv
import json
import math
import os
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np

from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.evaluation.behavior_data import observation_fingerprint
from match3_ai.evaluation.experiments import (
    _source_fingerprint,
    summarize_level_episodes,
)
from match3_ai.evaluation.fresh_protocol import sha256
from match3_ai.evaluation.gameplay_events import GameplayCollector, structured
from match3_ai.evaluation.runner import EpisodeResult, run_episode
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.loader import load_level
from match3_ai.training.seed_policy import (
    PARTITIONS,
    SPRINT520_TRAINING_PARTITION,
    STUDY_VALIDATION_PARTITIONS,
)

LEVELS = ("wooden-trouble", "double-trouble", "frozen-garden")
ENVIRONMENT_SEEDS = tuple(range(760000, 760050))
RANDOM_SEEDS = tuple(range(762000, 762050))
STAGE1B_ENVIRONMENT_SEEDS = tuple(range(780000, 780050))
STAGE1B_RANDOM_SEEDS = tuple(range(781000, 781050))
CONFIRMATION_ENVIRONMENT_SEEDS = tuple(range(761000, 761200))
CONFIRMATION_RANDOM_SEEDS = tuple(range(763000, 763200))
FROZEN_SELECTION = {
    "wooden-trouble__colors5-double-moves6": "2cf5a4a71fbf2b0f173fd5cd97bad2e9650496a937356828e8838a483111db48",
    "double-trouble__colors5-moves6": "99f8686fdd3507221e6096c3eabff1b8838afa9ce4d8b386a00167684bfa5a50",
    "frozen-garden__colors5-moves10": "322aba2c05b16dddd9e8bfe6710187b7632929921733cb915b47477ca346f8ed",
}
LEVEL_ROOT = Path(__file__).resolve().parents[1] / "levels/data"


def write_json(path, value):
    with path.open("x") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.flush()
        os.fsync(handle.fileno())


def append_json(handle, value):
    handle.write(json.dumps(structured(value), allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def screening_seeds(stage):
    if stage == "1":
        return ENVIRONMENT_SEEDS, RANDOM_SEEDS
    if stage == "1b":
        return STAGE1B_ENVIRONMENT_SEEDS, STAGE1B_RANDOM_SEEDS
    if stage == "2":
        return CONFIRMATION_ENVIRONMENT_SEEDS, CONFIRMATION_RANDOM_SEEDS
    raise ValueError("Unknown screening stage")


def combined_variants():
    """Resolve the approved combined candidates directly from packaged originals."""
    variants = {}
    parameters = {
        "wooden-trouble": ((6, None), (5, None), (None, "DOUBLE"), (6, "DOUBLE")),
        "double-trouble": ((8, None), (6, None), (None, "TRIPLE"), (8, "TRIPLE")),
        "frozen-garden": ((10, None), (8, None), (None, "THICK"), (10, "THICK")),
    }
    for level, candidates in parameters.items():
        for moves, hp_variant in candidates:
            config = json.loads((LEVEL_ROOT / f"{level}.json").read_text())
            config["num_colors"] = 5
            factors, name = ["num_colors"], "colors5"
            if hp_variant is not None:
                for blocker in config["blockers"]:
                    blocker["variant"] = hp_variant
                factors.append("blockers")
                name += "-" + hp_variant.lower()
            if moves is not None:
                config["move_limit"] = moves
                factors.append("move_limit")
                name += f"-moves{moves}"
            variants[f"{level}__{name}"] = {"config": config, "factor": factors}
    return variants


def resolved_variants(stage="1"):
    """Return detached loader-compatible JSON, retaining native level IDs."""
    if stage == "1b":
        return combined_variants()
    screening_seeds(stage)
    variants = {}
    for level in LEVELS:
        original = json.loads((LEVEL_ROOT / f"{level}.json").read_text())
        choices = [("original", None, None)]
        if level == "wooden-trouble":
            choices += [
                ("colors4", "num_colors", 4),
                ("colors5", "num_colors", 5),
                ("double-crates", "blockers", "DOUBLE"),
            ]
        elif level == "double-trouble":
            choices += [
                ("colors5", "num_colors", 5),
                ("moves10", "move_limit", 10),
                ("triple-crates", "blockers", "TRIPLE"),
            ]
        else:
            choices += [
                ("colors5", "num_colors", 5),
                ("moves10", "move_limit", 10),
                ("thick-ice", "blockers", "THICK"),
            ]
        for name, factor, value in choices:
            config = json.loads(json.dumps(original))
            if factor == "blockers":
                for blocker in config["blockers"]:
                    blocker["variant"] = value
            elif factor:
                config[factor] = value
            variants[f"{level}__{name}"] = {"config": config, "factor": factor}
    return variants


def validate_seed_plan(root, stage="1"):
    """Reject protected, diagnostic, confirmation, and previously used seed ranges."""
    environment_seeds, random_seeds = screening_seeds(stage)
    seeds = set(environment_seeds + random_seeds)
    bounds = [
        *PARTITIONS.values(),
        *STUDY_VALIDATION_PARTITIONS.values(),
        SPRINT520_TRAINING_PARTITION,
        (710000, 710199),
        (720000, 720199),
        (730000, 730049),
        (740000, 740049),
    ]
    if stage != "2":
        bounds.append((761000, 761199))
    if stage == "1b":
        bounds += [(760000, 760049), (762000, 762049), (763000, 763199)]
    if stage == "2":
        bounds += [
            (760000, 760049),
            (762000, 762049),
            (780000, 780049),
            (781000, 781049),
        ]
    if len(seeds) != (400 if stage == "2" else 100) or any(
        low <= seed <= high for seed in seeds for low, high in bounds
    ):
        raise ValueError("Screening seeds overlap protected or reserved ranges")
    for path in sorted(Path(root).rglob("*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for key in (
                    "seed",
                    "environment_seed",
                    "effective_episode_seed",
                    "agent_seed",
                ):
                    if row.get(key) and int(row[key]) in seeds:
                        raise ValueError(f"Screening seed already used: {path}")
    for path in sorted(Path(root).rglob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                for record in (row, row.get("info") or {}, row.get("record") or {}):
                    if any(
                        record.get(key) in seeds
                        for key in (
                            "seed",
                            "environment_seed",
                            "effective_episode_seed",
                            "agent_seed",
                        )
                    ):
                        raise ValueError(f"Screening seed already used: {path}")


class ScreeningEnvironment(Match3LevelEnv):
    """Capture reset identity without enabling expensive resolution snapshots."""

    def __init__(self, config, collector, identities):
        super().__init__(config.level_id, level_config=config)
        self.collector = collector
        self.identities = identities
        self.initial = None
        self.pairing_error = False

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.initial = {
            "fingerprint": observation_fingerprint(observation),
            "board_seed": info["board_seed"],
            "engine_seed": info["engine_seed"],
        }
        self.collector.emit(
            "episode_start",
            observation=observation,
            info=info,
            initial_identity=self.initial,
        )
        seed = info["effective_episode_seed"]
        if self.identities.setdefault(seed, self.initial) != self.initial:
            self.pairing_error = True
            raise RuntimeError("Initial board or engine seed differs between agents")
        return observation, info


def mean(values):
    return float(np.mean(values)) if values else None


def summarize(records, moves, configs, *, expected_episodes=50):
    """Keep unavailable metrics null and partial groups ineligible for targets."""
    summaries = []
    for variant, spec in configs.items():
        config = spec["config"]
        limit = config.move_limit
        initial_hp = sum(blocker.hp for blocker in config.blockers)
        for agent in ("Random", "Greedy"):
            group = [
                row
                for row in records
                if row["variant_id"] == variant and row["record"]["agent_name"] == agent
            ]
            if not group:
                continue
            episodes = [EpisodeResult(**row["record"]) for row in group]
            summary = summarize_level_episodes(episodes)[0]
            rounds = moves.get((variant, agent), [])
            wins = [item for item in episodes if item.outcome == "WON"]
            losses = [item for item in episodes if item.outcome == "LOST"]
            first = sum(
                item.outcome == "WON" and item.moves_used == 1 for item in episodes
            ) / len(episodes)
            summary.update(
                {
                    "variant_id": variant,
                    "factor": spec["factor"],
                    "move_budget_utilization_wins": mean(
                        [item.moves_used / limit for item in wins]
                    ),
                    "move_budget_utilization_losses": mean(
                        [item.moves_used / limit for item in losses]
                    ),
                    "first_move_win_rate": first,
                    "mean_normalized_hp_removed": mean(
                        [
                            (initial_hp - item.blocker_hp_remaining) / initial_hp
                            for item in episodes
                            if item.blocker_hp_remaining is not None
                        ]
                    ),
                    "generation_failures": sum(
                        item.error_type == "LevelGenerationError" for item in episodes
                    ),
                    "reshuffles": sum(item.reshuffles for item in episodes),
                    "cascade_completed_moves": len(rounds),
                    "cascade_median": float(np.median(rounds)) if rounds else None,
                    "cascade_p95": float(np.percentile(rounds, 95)) if rounds else None,
                    "cascade_p99": float(np.percentile(rounds, 99)) if rounds else None,
                    "cascade_max": max(rounds) if rounds else None,
                    "cascade_frequency_above20": mean([value > 20 for value in rounds]),
                    "cascade_frequent_above20": mean([value > 20 for value in rounds])
                    > 0.01
                    if rounds
                    else False,
                    "cascade_reaches100": any(value >= 100 for value in rounds)
                    or any(item.max_cascades_in_move >= 100 for item in episodes),
                }
            )
            low, high = (0.2, 0.6) if agent == "Random" else (0.4, 0.8)
            summary["meets_agent_targets"] = (
                len(episodes) == expected_episodes
                and not summary["errors"]
                and not summary["time_limits"]
                and low <= summary["win_rate"] <= high
                and first <= 0.1
            )
            summaries.append(summary)
    candidates = {}
    for variant in configs:
        rows = [row for row in summaries if row["variant_id"] == variant]
        candidates[variant] = {
            "meets_provisional_targets": len(rows) == 2
            and all(row["meets_agent_targets"] for row in rows),
            "cascade_review_required": any(
                row["cascade_frequent_above20"] or row["cascade_reaches100"]
                for row in rows
            ),
        }
    return {"groups": summaries, "candidates": candidates}


def save_report(
    output, records, moves, configs, status, *, stage="1", prior_summary=None
):
    summary = summarize(
        records, moves, configs, expected_episodes=200 if stage == "2" else 50
    )
    if stage == "2":
        confirmation_decisions(summary)
    summary["status"] = status
    if prior_summary is not None:
        comparison_key = (
            "stage1b_descriptive_comparison"
            if stage == "2"
            else "stage1_descriptive_comparison"
        )
        summary[comparison_key] = descriptive_comparison(
            summary, prior_summary, confirmation=stage == "2"
        )
    (output / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False))
    lines = [
        f"# Stage {'2 confirmation' if stage == '2' else ('1B screening' if stage == '1b' else '1 screening')} level calibration",
        "",
        f"Status: {status}",
        "",
        (
            "Frozen candidates are not retuned or replaced. Point-estimate targets and uncertainty are reported separately."
            if stage == "2"
            else "Provisional point-estimate targets only; no final variant selected. Incomplete/error groups are ineligible."
        ),
        "Cascade rounds include the initial match; statistics cover completed valid moves. Partial error resolutions are excluded.",
        "Frequent long cascades means >1% of moves exceed 20 rounds; 100+ rounds always requires review.",
        "",
        "| Variant / agent | n | Win % [Wilson 95%] | Budget used W/L | First-win % | Cascade median/p95/p99/max; >20 % | Blockers/HP left | HP removed / objective | Generation/errors/time limits/reshuffles |",
        "|---|---:|---|---|---:|---|---|---|---|",
    ]

    def fmt(value):
        return "NA" if value is None else f"{value:.3f}"

    for row in summary["groups"]:
        frequency = row["cascade_frequency_above20"]
        frequency_text = "NA" if frequency is None else f"{frequency:.1%}"
        ci = row["win_rate_ci95"]
        lines.append(
            f"| {row['variant_id']} / {row['agent_name']} | {row['attempted']} | "
            f"{row['win_rate']:.1%} [{ci[0]:.1%}, {ci[1]:.1%}] | "
            f"{fmt(row['move_budget_utilization_wins'])}/{fmt(row['move_budget_utilization_losses'])} | "
            f"{row['first_move_win_rate']:.1%} | "
            f"{fmt(row['cascade_median'])}/{fmt(row['cascade_p95'])}/{fmt(row['cascade_p99'])}/{fmt(row['cascade_max'])}; {frequency_text} | "
            f"{fmt(row['mean_blockers_remaining'])}/{fmt(row['mean_blocker_hp_remaining'])} | "
            f"{fmt(row['mean_normalized_hp_removed'])}/{fmt(row['mean_objective_completion_ratio'])} | "
            f"{row['generation_failures']}/{row['errors']}/{row['time_limits']}/{row['reshuffles']} |"
        )
    lines += ["", "Candidate checks (both agents must pass):", ""]
    for variant, flags in summary["candidates"].items():
        lines.append(
            f"- {variant}: targets={flags['meets_provisional_targets']}; cascade review={flags['cascade_review_required']}"
        )
    lines += [
        "",
        (
            "Color changes can change generated boards despite common seeds. "
            "HP and move-limit changes also change trajectories. These are screening results, not causal explanations or independent confirmation."
        ),
    ]
    if prior_summary is not None:
        lines += [
            "",
            (
                "Screening comparisons use different boards and are descriptive only; "
                "differences do not identify causal effects."
            ),
            "",
            "| Candidate / agent | Screening reference | Win-rate difference (pp) | Mean moves difference |",
            "|---|---|---:|---:|",
        ]
        for row in summary[comparison_key]:
            lines.append(
                f"| {row['variant_id']} / {row['agent_name']} | {row['reference_variant_id']} | "
                f"{row['win_rate_difference_pp']:+.1f} | {fmt(row['mean_moves_difference'])} |"
            )
    if stage == "2":
        lines += [
            "",
            (
                "Confirmation decisions use the preregistered point-estimate bands and "
                "cascade flags. Marginal Wilson 95% intervals describe uncertainty; "
                "they are not simultaneous intervals or additional pass requirements."
            ),
            "",
        ]
        for row in summary["groups"]:
            ci = row["first_move_win_ci95"]
            lines.append(
                f"- {row['variant_id']} / {row['agent_name']}: first-move Wilson 95% "
                f"[{ci[0]:.1%}, {ci[1]:.1%}]; win interval vs target: {row['win_interval_vs_target']}; "
                f"first-move upper bound ≤10%: {row['first_move_upper_bound_within_target']}"
            )
        for variant, decision in summary["candidates"].items():
            lines.append(
                f"- {variant}: confirmation targets={decision['meets_confirmation_targets']}; "
                f"frozen candidate={variant in FROZEN_SELECTION}"
            )
    (output / "report.md").write_text("\n".join(lines) + "\n")


def descriptive_comparison(summary, prior, *, confirmation=False):
    references = {
        (row["variant_id"], row["agent_name"]): row for row in prior["groups"]
    }
    comparisons = []
    for row in summary["groups"]:
        reference_ids = (
            (row["variant_id"],)
            if confirmation
            else (f"{row['level_id']}__original", f"{row['level_id']}__colors5")
        )
        for reference in reference_ids:
            if confirmation and (reference, row["agent_name"]) not in references:
                continue
            baseline = references[reference, row["agent_name"]]
            comparisons.append(
                {
                    "variant_id": row["variant_id"],
                    "agent_name": row["agent_name"],
                    "reference_variant_id": reference,
                    "win_rate_difference_pp": 100
                    * (row["win_rate"] - baseline["win_rate"]),
                    "mean_moves_difference": row["mean_moves_used"]
                    - baseline["mean_moves_used"]
                    if row["mean_moves_used"] is not None
                    and baseline["mean_moves_used"] is not None
                    else None,
                    "board_matched": False,
                }
            )
    return comparisons


def load_stage1_summary(path):
    summary = json.loads(Path(path).read_text())
    expected = {
        (variant, agent)
        for variant in resolved_variants()
        for agent in ("Random", "Greedy")
    }
    rows = summary["groups"]
    if (
        summary["status"] != "complete"
        or len(rows) != 24
        or {(row["variant_id"], row["agent_name"]) for row in rows} != expected
        or any(
            row["attempted"] != 50 or row["errors"] or row["time_limits"]
            for row in rows
        )
    ):
        raise ValueError(
            "Stage 1 comparison requires a complete, failure-free screening summary"
        )
    return summary


def confirmation_decisions(summary):
    for row in summary["groups"]:
        n, p = row["attempted"], row["first_move_win_rate"]
        z = 1.959963984540054
        denominator = 1 + z * z / n
        center = (p + z * z / (2 * n)) / denominator
        radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
        row["first_move_win_ci95"] = [
            max(0.0, center - radius),
            min(1.0, center + radius),
        ]
        low, high = (0.2, 0.6) if row["agent_name"] == "Random" else (0.4, 0.8)
        left, right = row["win_rate_ci95"]
        row["win_interval_vs_target"] = (
            "within"
            if low <= left and right <= high
            else "outside"
            if right < low or left > high
            else "overlaps boundary"
        )
        row["first_move_upper_bound_within_target"] = (
            row["first_move_win_ci95"][1] <= 0.1
        )
    for decision in summary["candidates"].values():
        decision["meets_confirmation_targets"] = (
            decision["meets_provisional_targets"]
            and not decision["cascade_review_required"]
        )


def load_confirmation_inputs(directory):
    """Load hash-pinned selection artifacts; never regenerate selected candidates."""
    directory = Path(directory)
    manifest_path, summary_path = (
        directory / "manifest.json",
        directory / "summary.json",
    )
    manifest = json.loads(manifest_path.read_text())
    prior = json.loads(summary_path.read_text())
    if (
        manifest["stage"] != "screening_1b"
        or prior["status"] != "complete"
        or len(prior["groups"]) != 24
        or any(
            row["attempted"] != 50 or row["errors"] or row["time_limits"]
            for row in prior["groups"]
        )
    ):
        raise ValueError(
            "Confirmation requires complete, failure-free Stage 1B artifacts"
        )
    expected = {
        (variant, agent)
        for variant in resolved_variants("1b")
        for agent in ("Random", "Greedy")
    }
    if {(row["variant_id"], row["agent_name"]) for row in prior["groups"]} != expected:
        raise ValueError(
            "Stage 1B summary identities differ from the screening protocol"
        )
    variants = {}
    for name, digest in FROZEN_SELECTION.items():
        spec = manifest["configurations"][name]
        path = directory / spec["path"]
        config = load_level(path)
        if (
            spec["sha256"] != digest
            or sha256(path) != digest
            or config.level_id != spec["level_id"]
        ):
            raise ValueError(f"Frozen candidate hash or identity mismatch: {name}")
        variants[name] = {
            "config": json.loads(path.read_text()),
            "factor": spec["factor"],
            "source_path": str(path.resolve()),
            "source_sha256": digest,
        }
    for level in LEVELS:
        path = LEVEL_ROOT / f"{level}.json"
        digest = manifest["packaged_config_sha256"][level]
        if sha256(path) != digest:
            raise ValueError(f"Original packaged configuration changed: {level}")
        variants[f"{level}__original"] = {
            "config": json.loads(path.read_text()),
            "factor": None,
            "source_path": str(path.resolve()),
            "source_sha256": digest,
        }
    provenance = {
        "selection_basis": "User-frozen selection based only on Random/Greedy Stage 1B screening; no DQN performance",
        "stage1b_manifest_path": str(manifest_path.resolve()),
        "stage1b_manifest_sha256": sha256(manifest_path),
        "stage1b_summary_path": str(summary_path.resolve()),
        "stage1b_summary_sha256": sha256(summary_path),
        "selected_config_sha256": dict(FROZEN_SELECTION),
        "board_matched_to_screening": False,
        "retuning_or_replacement_allowed": False,
    }
    return variants, provenance, prior


def run_screening(
    results_root, output, *, stage="1", stage1_summary=None, stage1b_dir=None
):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    environment_seeds, random_seeds = screening_seeds(stage)
    prior, provenance = None, None
    if stage == "2":
        stage1b_dir = (
            Path(stage1b_dir)
            if stage1b_dir is not None
            else Path(results_root) / "level_calibration_stage1b"
        )
        variants, provenance, prior = load_confirmation_inputs(stage1b_dir)
        validate_seed_plan(results_root, stage)
    elif stage == "1b":
        stage1_summary = (
            Path(stage1_summary)
            if stage1_summary is not None
            else Path(results_root) / "level_calibration_stage1/summary.json"
        )
        prior = load_stage1_summary(stage1_summary)
        validate_seed_plan(results_root, stage)
        variants = resolved_variants(stage)
    else:
        validate_seed_plan(results_root)
        variants = resolved_variants()
    output.mkdir(parents=True, exist_ok=False)
    (output / "configs").mkdir()
    configs, specifications = {}, {}
    for variant, spec in variants.items():
        path = output / "configs" / f"{variant}.json"
        if stage == "2":
            with path.open("xb") as handle:
                handle.write(Path(spec["source_path"]).read_bytes())
                handle.flush()
                os.fsync(handle.fileno())
            if sha256(path) != spec["source_sha256"]:
                raise ValueError("Frozen configuration changed while copying")
        else:
            write_json(path, spec["config"])
        configs[variant] = {"config": load_level(path), "factor": spec["factor"]}
        specifications[variant] = {
            "path": f"configs/{variant}.json",
            "sha256": sha256(path),
            "factor": spec["factor"],
            "level_id": spec["config"]["level_id"],
        }
    originals = {level: sha256(LEVEL_ROOT / f"{level}.json") for level in LEVELS}
    manifest = {
        "schema_version": 1,
        "stage": "confirmation"
        if stage == "2"
        else ("screening" if stage == "1" else "screening_1b"),
        "planned_episodes": len(configs) * 2 * len(environment_seeds),
        "environment_seeds": environment_seeds,
        "random_agent_seeds": random_seeds,
        "agents": ["Random", "Greedy"],
        "configurations": specifications,
        "packaged_config_sha256": originals,
        "source_fingerprint": _source_fingerprint(),
        "numpy_version": np.__version__,
        "max_cascades": 200,
        "max_initialization_attempts": 1000,
        "targets": {
            "Random": [0.2, 0.6],
            "Greedy": [0.4, 0.8],
            "first_move_win_max": 0.1,
        },
        "cascade_frequent_threshold": 0.01,
        "cascade_round_definition": "includes initial match",
        "seed_namespace": "original packaged level ordering, independent of variant name",
        "paired_scope": "within configuration; across variants boards may differ",
        "no_replacement": True,
        "final_test_used": False,
    }
    if provenance is not None:
        manifest["frozen_selection_provenance"] = provenance
        manifest["decision_rules"] = (
            "Both agents satisfy win bands and first-move wins <=10%; no >1% moves above20 rounds or any 100+ round chain; no retuning or replacements"
        )
    elif prior is not None:
        manifest["stage1_comparison"] = {
            "path": str(stage1_summary.resolve()),
            "sha256": sha256(stage1_summary),
            "board_matched": False,
        }
    write_json(output / "manifest.json", manifest)
    experiment_id = sha256(output / "manifest.json")[:24]
    records, moves = [], defaultdict(list)
    status, active = "interrupted", None
    try:
        with (
            (output / "episodes.jsonl").open("x") as journal,
            (output / "events.jsonl").open("x") as events,
        ):
            for variant, spec in configs.items():
                config, identities = spec["config"], {}
                for index, seed in enumerate(environment_seeds):
                    for name, agent in (
                        ("Random", RandomAgent(random_seeds[index])),
                        ("Greedy", GreedyAgent()),
                    ):
                        active = {"variant_id": variant, "agent": name, "seed": seed}
                        collector = GameplayCollector(
                            events, experiment_id, f"{variant}:{name}:{seed}"
                        )
                        env = ScreeningEnvironment(config, collector, identities)
                        try:
                            record = run_episode(
                                env,
                                agent,
                                seed=seed,
                                max_steps=config.move_limit,
                                experiment_id=experiment_id,
                                episode_id=collector.episode_id,
                                agent_name=name,
                                agent_seed=random_seeds[index]
                                if name == "Random"
                                else None,
                                collector=collector,
                            )
                        finally:
                            env.close()
                        row = {
                            "variant_id": variant,
                            "seed": seed,
                            "initial_identity": env.initial,
                            "record": asdict(record),
                        }
                        append_json(journal, row)
                        records.append(row)
                        moves[variant, name].extend(
                            move.cascade_count
                            for move in collector.moves
                            if move.valid_action
                            and move.resolution_completed
                            and move.cascade_count is not None
                        )
                        if record.outcome not in ("WON", "LOST"):
                            collector.emit(
                                "failure",
                                error_type=record.error_type,
                                error_message=record.error_message,
                                outcome=record.outcome,
                                variant_id=variant,
                            )
                        collector.validate()
                        if env.pairing_error:
                            raise RuntimeError("Board-pairing integrity failure")
            if any(
                sha256(LEVEL_ROOT / f"{level}.json") != digest
                for level, digest in originals.items()
            ):
                raise RuntimeError("Packaged configuration changed")
        if stage == "2" and any(
            sha256(spec["source_path"]) != spec["source_sha256"]
            for spec in variants.values()
        ):
            raise RuntimeError("Frozen source artifact changed")
        status = "complete"
    except BaseException as error:
        write_json(
            output / "failure.json",
            {
                "active_episode": active,
                "completed_records": len(records),
                "error_type": type(error).__name__,
                "error": str(error),
            },
        )
        raise
    finally:
        save_report(
            output, records, moves, configs, status, stage=stage, prior_summary=prior
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("experiments/results")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("1", "1b", "2"), default="1")
    parser.add_argument("--stage1-summary", type=Path)
    parser.add_argument("--stage1b-dir", type=Path)
    args = parser.parse_args()
    run_screening(
        args.results_root,
        args.output_dir,
        stage=args.stage,
        stage1_summary=args.stage1_summary,
        stage1b_dir=args.stage1b_dir,
    )


if __name__ == "__main__":
    main()
