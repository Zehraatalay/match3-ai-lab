"""Reproduce the Sprint 5.5 analysis without modifying benchmark inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np

from match3_ai.evaluation.analytics import MoveRecord
from match3_ai.evaluation.runner import EpisodeResult

OUTCOMES = {"WON", "LOST", "ERROR", "TIME_LIMIT"}
EPISODE_INTS = {
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
EPISODE_FLOATS = {
    "total_reward",
    "objective_completion_ratio",
    "inference_time_seconds",
}
EPISODE_JSON = {"objective_progress", "error_context"}
MOVE_INTS = {
    "move_index",
    "action_id",
    "moves_remaining",
    "cleared_tiles",
    "cascade_count",
    "legal_action_count",
}
MOVE_JSON = {
    "cleared_by_color",
    "blocker_damage",
    "special_activations",
    "objective_progress",
}
RESET_MISSING = {
    "level_config_id",
    "effective_episode_seed",
    "board_seed",
    "engine_seed",
    "moves_used",
    "objective_progress",
    "objective_completion_ratio",
    "blockers_remaining",
    "blocker_hp_remaining",
    "cascade_limit",
}
METRICS = (
    "moves_used",
    "objective_completion_ratio",
    "blockers_remaining",
    "blocker_hp_remaining",
    "special_activations",
    "total_cascades",
    "max_cascades_in_move",
    "total_cleared",
    "total_reward",
    "inference_time_seconds",
)


@dataclass(frozen=True)
class Dataset:
    episodes: list[dict[str, Any]]
    moves: list[dict[str, Any]]
    metadata: dict[str, Any]
    hashes: dict[str, str]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _read_csv(path, fields, integers, floats, booleans, nested):
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        _require(
            reader.fieldnames is not None
            and len(reader.fieldnames) == len(set(reader.fieldnames))
            and set(reader.fieldnames) == set(fields),
            f"unexpected columns in {path.name}",
        )
        rows = []
        for index, raw in enumerate(reader, start=2):
            _require(
                None not in raw and all(value is not None for value in raw.values()),
                f"malformed row {index} in {path.name}",
            )
            row = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = None
                elif key in integers:
                    row[key] = int(value)
                elif key in floats:
                    row[key] = float(value)
                    _require(math.isfinite(row[key]), f"nonfinite {key} in {path.name}")
                elif key in booleans:
                    _require(value in ("True", "False"), f"invalid boolean {key}")
                    row[key] = value == "True"
                elif key in nested:
                    row[key] = json.loads(value)
                else:
                    row[key] = value
            rows.append(row)
    return rows


def load_dataset(input_dir: str | Path) -> Dataset:
    root = Path(input_dir)
    episodes = _read_csv(
        root / "episodes.csv",
        EpisodeResult.__dataclass_fields__,
        EPISODE_INTS,
        EPISODE_FLOATS,
        {"completed"},
        EPISODE_JSON,
    )
    moves = _read_csv(
        root / "moves.csv",
        MoveRecord.__dataclass_fields__,
        MOVE_INTS,
        {"reward"},
        {"valid_action", "resolution_completed"},
        MOVE_JSON,
    )
    metadata = json.loads(
        (root / "experiment_metadata.json").read_text(encoding="utf-8")
    )
    hashes = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest()
        for name in (
            "episodes.csv",
            "moves.csv",
            "experiment_metadata.json",
            "summary.md",
        )
        if (root / name).exists()
    }
    return Dataset(episodes, moves, metadata, hashes)


def validate_dataset(data: Dataset) -> dict:
    episodes, moves, metadata = data.episodes, data.moves, data.metadata
    _require(bool(episodes), "episode dataset is empty")
    _require(
        len(episodes) == metadata["attempted_episodes"],
        "metadata episode count mismatch",
    )
    _require(
        len(episodes) <= metadata["planned_episodes"], "too many attempted episodes"
    )
    by_id = {row["episode_id"]: row for row in episodes}
    _require(len(by_id) == len(episodes), "duplicate episode ID")
    pair_keys = {(row["level_id"], row["agent_name"], row["seed"]) for row in episodes}
    _require(
        len(pair_keys) == len(episodes), "ambiguous duplicate agent/level/seed pairing"
    )
    move_keys = {(row["episode_id"], row["move_index"]) for row in moves}
    _require(len(move_keys) == len(moves), "duplicate move index")
    missing = {
        "structural": Counter(),
        "initialization_error": Counter(),
        "failed_resolution": Counter(),
        "unexpected": Counter(),
    }
    for row in episodes:
        _require(row["outcome"] in OUTCOMES, "unknown outcome category")
        _require(row["agent_name"] in metadata["agent_names"], "unknown agent")
        _require(row["level_id"] in metadata["levels"], "unknown level")
        _require(
            row["experiment_id"] == metadata["experiment_id"],
            "episode experiment mismatch",
        )
        if row["level_config_id"] is not None:
            _require(
                row["level_config_id"]
                == metadata["levels"][row["level_id"]]["level_config_id"],
                "level fingerprint mismatch",
            )
        for key, value in row.items():
            if value is not None:
                continue
            if (
                key in {"error_type", "error_message", "error_context"}
                and row["outcome"] != "ERROR"
            ):
                category = "structural"
            elif (
                key in RESET_MISSING and row["outcome"] == "ERROR" and row["steps"] == 0
            ):
                category = "initialization_error"
            else:
                category = "unexpected"
            missing[category][f"episodes.{key}"] += 1
        if row["outcome"] == "ERROR":
            _require(bool(row["termination_reason"]), "error has no termination reason")
        if row["objective_completion_ratio"] is not None:
            _require(
                0 <= row["objective_completion_ratio"] <= 1,
                "objective ratio outside [0,1]",
            )
        for key in EPISODE_INTS:
            _require(
                row[key] is None or row[key] >= 0, f"negative episode value: {key}"
            )
        config = metadata["config"]
        _require(
            row["seed"] in config["environment_seeds"], "unconfigured environment seed"
        )
        index = config["environment_seeds"].index(row["seed"])
        _require(
            row["agent_seed"] == config["agent_seeds"][index],
            "agent seed metadata mismatch",
        )

    traces = defaultdict(list)
    for row in moves:
        _require(row["episode_id"] in by_id, "orphan move")
        _require(
            row["experiment_id"] == metadata["experiment_id"],
            "move experiment mismatch",
        )
        _require(
            row["move_index"] is not None and row["move_index"] > 0,
            "invalid move index",
        )
        traces[row["episode_id"]].append(row)
        for key, value in row.items():
            if value is not None:
                continue
            if key in {"error_type", "error_message"} and row["resolution_completed"]:
                category = "structural"
            elif not row["resolution_completed"] and key in {
                "action_id",
                "valid_action",
                "reward",
                "cleared_tiles",
                "cleared_by_color",
                "cascade_count",
                "blocker_damage",
                "special_activations",
            }:
                category = "failed_resolution"
            else:
                category = "unexpected"
            missing[category][f"moves.{key}"] += 1
        if row["special_activations"] is not None:
            activation = row["special_activations"]
            _require(
                sum(activation["by_type"].values()) == activation["count"],
                "special-type counts do not sum",
            )
    for episode_id, episode in by_id.items():
        trace = sorted(traces[episode_id], key=lambda row: row["move_index"])
        _require(
            [row["move_index"] for row in trace]
            == list(range(1, episode["steps"] + 1)),
            "noncontiguous or incomplete move trace",
        )
        _require(
            math.isclose(
                sum(row["reward"] or 0 for row in trace),
                episode["total_reward"],
                abs_tol=1e-9,
            ),
            "reward total mismatch",
        )
        _require(
            sum(row["cascade_count"] or 0 for row in trace)
            == episode["total_cascades"],
            "cascade total mismatch",
        )
        if all(row["resolution_completed"] for row in trace):
            _require(
                sum(row["valid_action"] for row in trace) == episode["valid_actions"],
                "valid-action total mismatch",
            )
            _require(
                sum(row["cleared_tiles"] for row in trace) == episode["total_cleared"],
                "tile-clear total mismatch",
            )
            _require(
                sum(row["special_activations"]["count"] for row in trace)
                == episode["special_activations"],
                "special total mismatch",
            )
        _require(
            max((row["cascade_count"] or 0 for row in trace), default=0)
            == episode["max_cascades_in_move"],
            "maximum cascade mismatch",
        )

    _require(
        not missing["unexpected"],
        f"unexpected missing values: {dict(missing['unexpected'])}",
    )
    for level_id in metadata["levels"]:
        for seed in metadata["config"]["environment_seeds"]:
            pair = [
                row
                for row in episodes
                if row["level_id"] == level_id and row["seed"] == seed
            ]
            if len(pair) == len(metadata["agent_names"]):
                for key in ("board_seed", "engine_seed"):
                    known = {row[key] for row in pair if row[key] is not None}
                    _require(len(known) <= 1, "paired engine/board seeds mismatch")
        if metadata.get("interruption_reason") is None:
            for agent in metadata["agent_names"]:
                _require(
                    sum(
                        row["level_id"] == level_id and row["agent_name"] == agent
                        for row in episodes
                    )
                    == metadata["config"]["num_episodes"],
                    "missing level/agent attempts",
                )
    for summary in metadata.get("summary", []):
        group = [
            row
            for row in episodes
            if row["level_id"] == summary["level_id"]
            and row["agent_name"] == summary["agent_name"]
        ]
        _require(len(group) == summary["attempted"], "metadata group count mismatch")
        _require(
            sum(row["outcome"] == "WON" for row in group) == summary["wins"],
            "metadata win count mismatch",
        )
    return {
        "episode_shape": [len(episodes), len(EpisodeResult.__dataclass_fields__)],
        "move_shape": [len(moves), len(MoveRecord.__dataclass_fields__)],
        "episode_columns": {
            key: "integer"
            if key in EPISODE_INTS
            else "float"
            if key in EPISODE_FLOATS
            else "JSON"
            if key in EPISODE_JSON
            else "boolean"
            if key == "completed"
            else "string"
            for key in EpisodeResult.__dataclass_fields__
        },
        "move_columns": {
            key: "integer"
            if key in MOVE_INTS
            else "JSON"
            if key in MOVE_JSON
            else "boolean"
            if key in {"valid_action", "resolution_completed"}
            else "float"
            if key == "reward"
            else "string"
            for key in MoveRecord.__dataclass_fields__
        },
        "missing": {key: dict(value) for key, value in missing.items()},
        "outcomes": dict(Counter(row["outcome"] for row in episodes)),
        "agents": sorted({row["agent_name"] for row in episodes}),
        "levels": list(metadata["levels"]),
        "unique_episode_ids": len(by_id),
        "duplicate_episodes": 0,
        "duplicate_moves": 0,
        "orphan_moves": 0,
    }


def numeric_summary(values) -> dict:
    raw = list(values)
    numbers = np.asarray([value for value in raw if value is not None], dtype=float)
    if not numbers.size:
        return {
            "n": 0,
            "missing": len(raw),
            "mean": None,
            "median": None,
            "std": None,
            "p90": None,
            "p95": None,
            "p99": None,
            "max": None,
        }
    return {
        "n": len(numbers),
        "missing": len(raw) - len(numbers),
        "mean": float(numbers.mean()),
        "median": float(np.median(numbers)),
        "std": float(numbers.std(ddof=1)) if numbers.size > 1 else None,
        **{f"p{q}": float(np.percentile(numbers, q)) for q in (90, 95, 99)},
        "max": float(numbers.max()),
    }


def wilson_interval(wins: int, n: int) -> list[float] | None:
    if not n:
        return None
    z = 1.959963984540054
    p = wins / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    radius = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, center - radius), min(1.0, center + radius)]


def describe_episodes(data: Dataset) -> list[dict]:
    result = []
    for level in data.metadata["config"]["level_ids"]:
        for agent in data.metadata["agent_names"]:
            group = [
                row
                for row in data.episodes
                if row["level_id"] == level and row["agent_name"] == agent
            ]
            counts = Counter(row["outcome"] for row in group)
            populations = {}
            for name, rows in {
                "attempts": group,
                "completed": [
                    row for row in group if row["outcome"] in {"WON", "LOST"}
                ],
                "wins": [row for row in group if row["outcome"] == "WON"],
                "losses": [row for row in group if row["outcome"] == "LOST"],
            }.items():
                populations[name] = {
                    "episodes": len(rows),
                    "metrics": {
                        key: numeric_summary(row[key] for row in rows)
                        for key in METRICS
                    },
                }
            result.append(
                {
                    "level_id": level,
                    "agent_name": agent,
                    "attempted": len(group),
                    "counts": {
                        outcome: counts[outcome] for outcome in sorted(OUTCOMES)
                    },
                    "win_rate": counts["WON"] / len(group) if group else None,
                    "wilson_ci95": wilson_interval(counts["WON"], len(group)),
                    "populations": populations,
                }
            )
    return result


def exact_mcnemar(random_only: int, greedy_only: int) -> float:
    """Two-sided exact binomial test on discordant paired binary outcomes."""
    if any(type(value) is not int or value < 0 for value in (random_only, greedy_only)):
        raise ValueError("discordant counts must be non-negative integers")
    n = random_only + greedy_only
    return min(
        1.0,
        2
        * sum(math.comb(n, k) for k in range(min(random_only, greedy_only) + 1))
        / (2**n),
    )


def paired_bootstrap(
    differences, *, seed: int, samples: int = 20000
) -> list[float] | None:
    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    values = np.asarray(differences, dtype=float)
    if not values.size:
        return None
    _require(
        np.all(np.isin(values, [-1, 0, 1])), "paired differences must be -1, 0 or 1"
    )
    probabilities = [np.mean(values == value) for value in (-1, 0, 1)]
    draws = np.random.default_rng(seed).multinomial(
        len(values), probabilities, size=samples
    )
    means = (draws[:, 2] - draws[:, 0]) / len(values)
    return np.quantile(means, [0.025, 0.975]).tolist()


def compare_agents(
    data: Dataset, *, seed: int = 55005, samples: int = 20000
) -> list[dict]:
    _require(
        set(data.metadata["agent_names"]) == {"RandomAgent", "GreedyAgent"},
        "paired comparison requires RandomAgent and GreedyAgent",
    )
    results = []
    for level in data.metadata["config"]["level_ids"]:
        indexed = defaultdict(dict)
        for row in data.episodes:
            if row["level_id"] == level:
                _require(
                    row["agent_name"] not in indexed[row["seed"]],
                    "duplicate paired seed",
                )
                indexed[row["seed"]][row["agent_name"]] = row["outcome"]
        table = Counter()
        excluded = Counter()
        differences = []
        for _, pair in sorted(indexed.items()):
            if len(pair) != 2:
                excluded["unmatched_seed"] += 1
                continue
            if any(value not in {"WON", "LOST"} for value in pair.values()):
                excluded["error_or_time_limit"] += 1
                continue
            random_win = pair["RandomAgent"] == "WON"
            greedy_win = pair["GreedyAgent"] == "WON"
            table[(random_win, greedy_win)] += 1
            differences.append(int(greedy_win) - int(random_win))
        b, c = table[(True, False)], table[(False, True)]
        level_seed = int.from_bytes(
            hashlib.sha256(f"{seed}:{level}".encode()).digest()[:4], "big"
        )
        results.append(
            {
                "level_id": level,
                "valid_pairs": len(differences),
                "excluded_pairs": dict(excluded),
                "both_win": table[(True, True)],
                "random_only": b,
                "greedy_only": c,
                "both_lose": table[(False, False)],
                "discordant_pairs": b + c,
                "greedy_minus_random": float(np.mean(differences))
                if differences
                else None,
                "bootstrap_ci95": paired_bootstrap(
                    differences, seed=level_seed, samples=samples
                ),
                "bootstrap_seed": level_seed,
                "bootstrap_samples": samples,
                "p_value": exact_mcnemar(b, c) if differences else None,
                "matched_odds_ratio": c / b if b else None,
                "odds_ratio_status": "finite"
                if b
                else "infinite"
                if c
                else "undefined_no_discordance",
            }
        )
    ordered = sorted(
        (row for row in results if row["p_value"] is not None),
        key=lambda row: row["p_value"],
    )
    running = 0.0
    for rank, row in enumerate(ordered):
        running = max(running, min(1.0, (len(ordered) - rank) * row["p_value"]))
        row["holm_p_value"] = running
    return results


def analyze_moves(data: Dataset) -> dict:
    episodes = {row["episode_id"]: row for row in data.episodes}
    completed = [row for row in data.moves if row["resolution_completed"]]
    overall = numeric_summary(row["cascade_count"] for row in completed)
    groups = []
    for level in data.metadata["config"]["level_ids"]:
        for agent in data.metadata["agent_names"]:
            rows = [
                row
                for row in completed
                if episodes[row["episode_id"]]["level_id"] == level
                and episodes[row["episode_id"]]["agent_name"] == agent
            ]
            groups.append(
                {
                    "level_id": level,
                    "agent_name": agent,
                    "cascades": numeric_summary(row["cascade_count"] for row in rows),
                    "over_50": sum(row["cascade_count"] > 50 for row in rows),
                    "over_100": sum(row["cascade_count"] > 100 for row in rows),
                }
            )
    impacts = []
    for threshold in (50, 100, overall["p99"]):
        retained = [row for row in completed if row["cascade_count"] <= threshold]
        tail = [row for row in completed if row["cascade_count"] > threshold]
        tile_total = sum(row["cleared_tiles"] for row in completed)
        activation_total = sum(row["special_activations"]["count"] for row in completed)
        impacts.append(
            {
                "threshold": threshold,
                "tail_moves": len(tail),
                "total_moves": len(completed),
                "tile_share": sum(row["cleared_tiles"] for row in tail) / tile_total
                if tile_total
                else None,
                "special_share": sum(
                    row["special_activations"]["count"] for row in tail
                )
                / activation_total
                if activation_total
                else None,
                "tiles_all": numeric_summary(row["cleared_tiles"] for row in completed),
                "tiles_without_tail": numeric_summary(
                    row["cleared_tiles"] for row in retained
                ),
                "specials_all": numeric_summary(
                    row["special_activations"]["count"] for row in completed
                ),
                "specials_without_tail": numeric_summary(
                    row["special_activations"]["count"] for row in retained
                ),
            }
        )
    outliers = []
    for row in sorted(completed, key=lambda row: row["cascade_count"], reverse=True)[
        :10
    ]:
        episode = episodes[row["episode_id"]]
        outliers.append(
            {
                key: episode[key]
                for key in ("episode_id", "agent_name", "level_id", "seed", "outcome")
            }
            | {
                key: row[key]
                for key in (
                    "move_index",
                    "action_id",
                    "cascade_count",
                    "cleared_tiles",
                    "special_activations",
                )
            }
        )
    special_types = Counter()
    for row in completed:
        special_types.update(row["special_activations"]["by_type"])
    special_groups = []
    for agent in data.metadata["agent_names"]:
        for outcome in ("WON", "LOST"):
            group_episodes = [
                row
                for row in data.episodes
                if row["agent_name"] == agent and row["outcome"] == outcome
            ]
            group_moves = [
                row
                for row in completed
                if episodes[row["episode_id"]]["agent_name"] == agent
                and episodes[row["episode_id"]]["outcome"] == outcome
            ]
            types = Counter()
            for row in group_moves:
                types.update(row["special_activations"]["by_type"])
            special_groups.append(
                {
                    "agent_name": agent,
                    "outcome": outcome,
                    "episodes": len(group_episodes),
                    "moves": len(group_moves),
                    "per_episode": numeric_summary(
                        row["special_activations"] for row in group_episodes
                    ),
                    "types": dict(types),
                }
            )
    return {
        "overall_cascades": overall,
        "over_50": sum(row["cascade_count"] > 50 for row in completed),
        "over_100": sum(row["cascade_count"] > 100 for row in completed),
        "by_level_agent": groups,
        "by_outcome": {
            outcome: numeric_summary(
                row["cascade_count"]
                for row in completed
                if episodes[row["episode_id"]]["outcome"] == outcome
            )
            for outcome in ("WON", "LOST")
        },
        "tail_sensitivity": impacts,
        "outliers": outliers,
        "special_activation_types": dict(special_types),
        "specials_by_agent_outcome": special_groups,
    }


def generate_figures(
    data: Dataset, descriptive: list[dict], comparisons: list[dict], output: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    output.mkdir(exist_ok=False)
    levels = data.metadata["config"]["level_ids"]
    agents = data.metadata["agent_names"]
    colors = {"RandomAgent": "#0072B2", "GreedyAgent": "#D55E00"}
    labels = [level.replace("-", " ").title() for level in levels]
    paths = []

    def save(fig, name):
        fig.savefig(output / f"{name}.png", dpi=300)
        fig.savefig(
            output / f"{name}.pdf", metadata={"CreationDate": None, "ModDate": None}
        )
        plt.close(fig)
        paths.append(name)

    def grouped_boxes(ax, metric, rows, *, losses_only=False):
        for agent_index, agent in enumerate(agents):
            positions, values = [], []
            for level_index, level in enumerate(levels):
                positions.append(level_index + (agent_index - 0.5) * 0.32)
                values.append(
                    [
                        row[metric]
                        for row in rows
                        if row["level_id"] == level
                        and row["agent_name"] == agent
                        and row[metric] is not None
                        and (not losses_only or row["outcome"] == "LOST")
                    ]
                )
            box = ax.boxplot(
                values,
                positions=positions,
                widths=0.26,
                patch_artist=True,
                showmeans=True,
                meanprops={
                    "marker": "D",
                    "markerfacecolor": "white",
                    "markeredgecolor": colors[agent],
                    "markersize": 3,
                },
                flierprops={"marker": ".", "markersize": 3, "alpha": 0.5},
            )
            for patch in box["boxes"]:
                patch.set(facecolor=colors[agent], alpha=0.6)
        ax.set_xticks(range(len(levels)), labels, rotation=20, ha="right")
        ax.legend(
            handles=[
                Patch(
                    facecolor=colors[agent], label=agent.replace("Agent", ""), alpha=0.6
                )
                for agent in agents
            ]
        )

    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.alpha": 0.2,
            "pdf.fonttype": 42,
        }
    ):
        for name, intervals in (
            ("01_win_rates", False),
            ("02_win_rate_intervals", True),
        ):
            fig, ax = plt.subplots(figsize=(10, 4.8), layout="constrained")
            for index, agent in enumerate(agents):
                rows = [
                    next(
                        row
                        for row in descriptive
                        if row["level_id"] == level and row["agent_name"] == agent
                    )
                    for level in levels
                ]
                values = np.array([row["win_rate"] for row in rows])
                positions = np.arange(len(levels)) + (index - 0.5) * 0.32
                if intervals:
                    bounds = np.array([row["wilson_ci95"] for row in rows])
                    ax.errorbar(
                        positions,
                        values,
                        yerr=np.maximum(
                            0, np.array([values - bounds[:, 0], bounds[:, 1] - values])
                        ),
                        fmt="o",
                        capsize=4,
                        color=colors[agent],
                        label=agent.replace("Agent", ""),
                    )
                else:
                    ax.bar(
                        positions,
                        values,
                        0.3,
                        color=colors[agent],
                        label=agent.replace("Agent", ""),
                    )
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_major_formatter(PercentFormatter(1))
            ax.set_xticks(range(len(levels)), labels, rotation=20, ha="right")
            ax.set_ylabel("Wins / all attempted episodes")
            ax.set_title(
                "Win rate by level"
                + (" with Wilson 95% intervals" if intervals else "")
            )
            ax.legend(loc="lower left")
            save(fig, name)

        for metric, name, title, ylabel in (
            (
                "objective_completion_ratio",
                "03_objective_completion",
                "Objective completion distributions",
                "Mean objective progress / target",
            ),
            (
                "moves_used",
                "04_moves_used",
                "Moves consumed by level and agent",
                "Consumed level moves",
            ),
        ):
            fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
            grouped_boxes(ax, metric, data.episodes)
            ax.set_title(title)
            ax.set_ylabel(ylabel)
            if metric == "objective_completion_ratio":
                ax.set_ylim(-0.02, 1.05)
                ax.yaxis.set_major_formatter(PercentFormatter(1))
            else:
                ax.set_ylim(bottom=0)
            save(fig, name)

        by_id = {row["episode_id"]: row for row in data.episodes}
        completed_moves = [row for row in data.moves if row["resolution_completed"]]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), layout="constrained")
        cascades = [row["cascade_count"] for row in completed_moves]
        axes[0].hist(
            cascades, bins=np.arange(max(cascades) + 2) - 0.5, color="#0072B2", log=True
        )
        axes[0].set(
            xlabel="Cascades per move",
            ylabel="Moves (log scale)",
            title="Cascade distribution, including zero",
        )
        grouped_boxes(
            axes[1],
            "cascade_count",
            [
                dict(
                    row,
                    level_id=by_id[row["episode_id"]]["level_id"],
                    agent_name=by_id[row["episode_id"]]["agent_name"],
                )
                for row in completed_moves
            ],
        )
        axes[1].set_yscale("symlog", linthresh=5)
        axes[1].set_ylim(bottom=0)
        axes[1].set(
            ylabel="Cascades (symlog; linear below 5)",
            title="Cascade distribution by level",
        )
        save(fig, "05_cascade_distribution")

        fig, ax = plt.subplots(figsize=(9, 5), layout="constrained")
        tick_positions, tick_labels = [], []
        for outcome_index, outcome in enumerate(("WON", "LOST")):
            for agent_index, agent in enumerate(agents):
                rows = [
                    row["special_activations"]
                    for row in data.episodes
                    if row["outcome"] == outcome
                    and row["agent_name"] == agent
                    and row["special_activations"] is not None
                ]
                position = outcome_index * 3 + agent_index
                box = ax.boxplot(
                    [rows],
                    positions=[position],
                    widths=0.6,
                    patch_artist=True,
                    flierprops={"marker": ".", "markersize": 3, "alpha": 0.4},
                )
                box["boxes"][0].set(facecolor=colors[agent], alpha=0.6)
                tick_positions.append(position)
                tick_labels.append(
                    f"{outcome}\n{agent.replace('Agent', '')}\nn={len(rows)}"
                )
        ax.set_xticks(tick_positions, tick_labels)
        ax.set_yscale("symlog", linthresh=5)
        ax.set_ylim(bottom=0)
        ax.set(
            ylabel="Special activations per episode (symlog)",
            title="Special activations by outcome (levels pooled)",
        )
        save(fig, "06_special_activations_by_outcome")

        fig, ax = plt.subplots(figsize=(10, 5), layout="constrained")
        grouped_boxes(ax, "blocker_hp_remaining", data.episodes, losses_only=True)
        ax.set(
            ylabel="Blocker HP remaining",
            title="Remaining blocker HP on losses; blank groups have no losses",
        )
        ax.set_ylim(bottom=0)
        save(fig, "07_blocker_hp_on_losses")

        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for index, comparison in enumerate(comparisons):
            value = 100 * comparison["greedy_minus_random"]
            low, high = (100 * bound for bound in comparison["bootstrap_ci95"])
            ax.errorbar(
                value,
                index,
                xerr=[[max(0, value - low)], [max(0, high - value)]],
                fmt="o",
                color="#009E73",
                capsize=4,
            )
        ax.axvline(0, linestyle="--", color="gray")
        ax.set_yticks(
            range(len(levels)),
            [
                f"{label} (n={row['valid_pairs']})"
                for label, row in zip(labels, comparisons)
            ],
        )
        ax.set(
            xlabel="Greedy − Random win rate (percentage points)",
            title="Paired differences with percentile bootstrap 95% intervals",
        )
        save(fig, "08_paired_differences")
    return paths


def _table(headers, rows):
    return "\n".join(
        [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
            *("| " + " | ".join(str(value) for value in row) + " |" for row in rows),
        ]
    )


def _fmt(value, digits=2):
    return "missing" if value is None else f"{value:.{digits}f}"


def _interpret_comparison(row):
    level = row["level_id"]
    if not row["discordant_pairs"]:
        return f"{level}: all {row['valid_pairs']} observed pairs are concordant. The test has no discordant evidence to distinguish policies; its p=1 and empirical bootstrap interval [0,0] do not establish equality or equivalence outside this seed set."
    low, high = row["bootstrap_ci95"]
    return (
        f"{level}: Greedy−Random is {100 * row['greedy_minus_random']:+.2f} percentage points "
        f"over {row['valid_pairs']} valid pairs ({row['discordant_pairs']} discordant); "
        f"bootstrap 95% interval [{100 * low:.2f}, {100 * high:.2f}] points; "
        f"exact p={row['p_value']:.5f}, Holm p={row['holm_p_value']:.5f}. "
        + (
            "The two-sided test does not establish a policy difference. The interval and sparse-pair limitations leave practical benefits or harms uncertain; this is not proof of equivalence."
            if row["holm_p_value"] >= 0.05
            else "The adjusted test detects a marginal win-probability difference in this sample; its practical value still depends on the effect interval, runtime and external validation."
        )
    )


def write_report(
    data, quality, descriptive, comparisons, move_analysis, diagnostics, figures, output
):
    lines = [
        "# Sprint 5.5: Exploratory data analysis and paired statistical evaluation",
        "",
        "## Executive summary",
        "",
        f"The benchmark contains {len(data.episodes):,} attempted episodes and {len(data.moves):,} move records. Outcomes are {quality['outcomes']}. All attempts are retained in descriptive analysis; only ERROR/TIME_LIMIT or unmatched pairs are excluded from the paired win/loss tests. No benchmark was rerun and no game or agent rules were changed.",
        "",
        "Greedy's observed advantage is modest and varies by level. The paired tests below quantify uncertainty; nonsignificance does not demonstrate equivalence. No human difficulty or causal conclusions are justified by this observational agent benchmark.",
        "",
        "## Dataset and reproducible transformations",
        "",
        f"Experiment `{data.metadata['experiment_id']}`; two agents and six levels. Paired environment seeds {data.metadata['config']['environment_seeds'][0]}–{data.metadata['config']['environment_seeds'][-1]}, with separately configured agent seeds. Shapes: episodes {quality['episode_shape']}, moves {quality['move_shape']}.",
        "",
        "CSV integers, finite floats and booleans are explicitly parsed; nested CSV JSON is decoded. Empty cells become None, never zero. Outcome/agent/level categories, fingerprints, configured seeds, unique IDs, contiguous move indices, foreign keys, and per-episode sums are validated against metadata. No rows are dropped or imputed. Input SHA-256 hashes are in analysis.json and are verified again after output generation. Full column/type dictionaries are also in analysis.json.",
        "",
        "## Data quality and missingness",
        "",
        "Structural missingness: error diagnostics on successful records. Initialization-error missingness: no observation or reset info was returned, so derived seeds, board/objective metrics, budget and per-row configuration ID are missing. The level fingerprint remains available in the original metadata; it is not silently filled into the CSV. Failed-resolution missingness is classified separately if present. Unexpected missingness causes validation to fail.",
        "",
        _table(
            ["Category", "Missing cells"],
            [(key, sum(values.values())) for key, values in quality["missing"].items()],
        ),
        "",
        f"Unique episodes: {quality['unique_episode_ids']}; duplicate episode IDs: 0; duplicate move keys: 0; orphan moves: 0. Reward, cascade, tile-clear and activation totals reconcile with move records. Metadata category counts and paired board/engine seeds agree.",
        "",
        "## Descriptive statistics",
        "",
        "Win rates below use every attempted episode, including errors, and Wilson 95% intervals. Mean/median moves use nonmissing observations (n=99 for First Match per agent; n=100 elsewhere). Objective progress is the mean of capped per-objective progress/target ratios, not a probability of winning.",
        "",
        _table(
            [
                "Level",
                "Agent",
                "Attempted",
                "W/L/E/T",
                "Win rate [95% CI]",
                "Moves mean/median (n)",
                "Objective mean (n)",
            ],
            [
                [
                    row["level_id"],
                    row["agent_name"],
                    row["attempted"],
                    "/".join(
                        str(row["counts"][key])
                        for key in ("WON", "LOST", "ERROR", "TIME_LIMIT")
                    ),
                    f"{100 * row['win_rate']:.1f}% [{100 * row['wilson_ci95'][0]:.1f}, {100 * row['wilson_ci95'][1]:.1f}]",
                    f"{_fmt(row['populations']['attempts']['metrics']['moves_used']['mean'])}/{_fmt(row['populations']['attempts']['metrics']['moves_used']['median'])} ({row['populations']['attempts']['metrics']['moves_used']['n']})",
                    f"{100 * row['populations']['attempts']['metrics']['objective_completion_ratio']['mean']:.2f}% ({row['populations']['attempts']['metrics']['objective_completion_ratio']['n']})",
                ]
                for row in descriptive
            ],
        ),
        "",
        "The next table uses all available attempted-episode values. Blocker/HP means exclude the two unavailable initial boards. Activations, cascades, rewards and inference totals include the observed zero work for failed resets. Time is total select_action time per episode, in milliseconds; it excludes engine execution and is hardware dependent. Exact n, missing counts, mean, median, sample standard deviation and quantiles for each metric are in analysis.json.",
        "",
        _table(
            [
                "Level",
                "Agent",
                "Blockers / HP (n)",
                "Specials",
                "Cascades / max",
                "Reward",
                "Inference ms (n)",
            ],
            [
                [
                    row["level_id"],
                    row["agent_name"],
                    f"{_fmt(row['populations']['attempts']['metrics']['blockers_remaining']['mean'])}/{_fmt(row['populations']['attempts']['metrics']['blocker_hp_remaining']['mean'])} ({row['populations']['attempts']['metrics']['blocker_hp_remaining']['n']})",
                    _fmt(
                        row["populations"]["attempts"]["metrics"][
                            "special_activations"
                        ]["mean"]
                    ),
                    f"{_fmt(row['populations']['attempts']['metrics']['total_cascades']['mean'])}/{_fmt(row['populations']['attempts']['metrics']['max_cascades_in_move']['max'], 0)}",
                    _fmt(
                        row["populations"]["attempts"]["metrics"]["total_reward"][
                            "mean"
                        ]
                    ),
                    f"{1000 * row['populations']['attempts']['metrics']['inference_time_seconds']['mean']:.3f} ({row['populations']['attempts']['metrics']['inference_time_seconds']['n']})",
                ]
                for row in descriptive
            ],
        ),
        "",
        "### Conditional populations",
        "",
        "All-attempt, completed (WON or LOST), wins-only and losses-only summaries are retained separately. The table below shows conditional means and sample sizes; missing observations are excluded metric by metric without removing their episode from other summaries. Comparing wins and losses conditions on the outcome and can reflect move-budget/exposure differences. Empty populations are reported as missing, not zero.",
        "",
        _table(
            [
                "Level",
                "Agent",
                "Population n",
                "Moves mean/median",
                "Objective mean",
                "HP mean",
                "Specials mean",
            ],
            [
                [
                    row["level_id"],
                    row["agent_name"],
                    f"{population} {values['episodes']}",
                    f"{_fmt(values['metrics']['moves_used']['mean'])}/{_fmt(values['metrics']['moves_used']['median'])}",
                    _fmt(values["metrics"]["objective_completion_ratio"]["mean"]),
                    _fmt(values["metrics"]["blocker_hp_remaining"]["mean"]),
                    _fmt(values["metrics"]["special_activations"]["mean"]),
                ]
                for row in descriptive
                for population, values in row["populations"].items()
                if population != "attempts"
            ],
        ),
        "",
        "## Paired agent comparison and statistical tests",
        "",
        "Research question: does Greedy have a different win probability than Random on the same initial seed within each level? H0: equal marginal win probabilities (the two discordant directions are equally probable). H1: they differ. Exact two-sided McNemar tests condition on b+c discordant pairs and use 2·P[Binomial(b+c, 0.5) ≤ min(b,c)], capped at 1. No independent-proportion test is used.",
        "",
        "Pairs align on level and environment seed; ERROR, TIME_LIMIT and unmatched pairs are excluded only from these tests. Assumptions: independent sampled seed pairs within each level and paired binary outcomes; agents receive the same initial board/game RNG seeds but may subsequently consume their RNG streams differently. The same seed set reused across levels makes cross-level comparisons dependent and exploratory.",
        "",
        "Effect size is Greedy-minus-Random paired risk difference in percentage points, (c−b)/n. A matched odds ratio c/b is additionally available in analysis.json; zero denominators are explicitly labeled undefined/infinite. Confidence intervals resample entire paired outcomes using the empirical multinomial counts of differences −1, 0, +1; 20,000 fixed-seed draws by default, with per-level seeds derived from the configured bootstrap seed. Percentile intervals are approximate; an all-concordant sample yields a degenerate interval and is not evidence of equivalence.",
        "",
        _table(
            [
                "Level",
                "Valid / excluded pairs",
                "Both win",
                "R only (b)",
                "G only (c)",
                "Both lose",
                "G−R pp [bootstrap 95% CI]",
                "Exact p",
                "Holm p",
            ],
            [
                [
                    row["level_id"],
                    f"{row['valid_pairs']} / {sum(row['excluded_pairs'].values())}",
                    row["both_win"],
                    row["random_only"],
                    row["greedy_only"],
                    row["both_lose"],
                    f"{100 * row['greedy_minus_random']:.2f} [{100 * row['bootstrap_ci95'][0]:.2f}, {100 * row['bootstrap_ci95'][1]:.2f}]",
                    _fmt(row["p_value"], 5),
                    _fmt(row["holm_p_value"], 5),
                ]
                for row in comparisons
            ],
        ),
        "",
        "Six level comparisons form an exploratory family. Raw and Holm-adjusted p-values are both reported. Unadjusted bootstrap intervals are not simultaneous family intervals. Statistical significance would not establish practical benefit; nonsignificance can reflect limited power and does not establish equivalence. Ceiling win rates provide little information for differentiating policies.",
        "",
        *(text for row in comparisons for text in (_interpret_comparison(row), "")),
        "## Cascade distribution and sensitivity to extreme moves",
        "",
        f"All {move_analysis['overall_cascades']['n']:,} completed moves contribute, including zero-cascade special moves. Sample standard deviation uses ddof=1; percentiles use NumPy's linear convention. Moves >50: {move_analysis['over_50']}; >100: {move_analysis['over_100']}.",
        "",
        _table(
            ["Mean", "Median", "Sample SD", "p90", "p95", "p99", "Maximum"],
            [
                [
                    _fmt(move_analysis["overall_cascades"][key])
                    for key in ("mean", "median", "std", "p90", "p95", "p99", "max")
                ]
            ],
        ),
        "",
        _table(
            ["Level", "Agent", "Moves n", "Mean / median", "p95 / max", ">50", ">100"],
            [
                [
                    row["level_id"],
                    row["agent_name"],
                    row["cascades"]["n"],
                    f"{_fmt(row['cascades']['mean'])}/{_fmt(row['cascades']['median'])}",
                    f"{_fmt(row['cascades']['p95'])}/{_fmt(row['cascades']['max'], 0)}",
                    row["over_50"],
                    row["over_100"],
                ]
                for row in move_analysis["by_level_agent"]
            ],
        ),
        "",
        _table(
            ["Final outcome", "Move n", "Mean", "Median", "p95", "Maximum"],
            [
                [
                    outcome,
                    stats["n"],
                    _fmt(stats["mean"]),
                    _fmt(stats["median"]),
                    _fmt(stats["p95"]),
                    _fmt(stats["max"], 0),
                ]
                for outcome, stats in move_analysis["by_outcome"].items()
            ],
        ),
        "",
        "Move observations within an episode are dependent, and pooled outcome comparisons confound level, agent, move exposure and stopping rules. These distribution comparisons are descriptive, not causal tests. The following sensitivity analysis removes a tail only to show its influence; no records are deleted from primary summaries.",
        "",
        _table(
            [
                "Tail threshold >",
                "Tail moves / all",
                "Share of clears",
                "Share of activations",
                "Mean clears all / without tail",
                "Mean activations all / without tail",
            ],
            [
                [
                    _fmt(row["threshold"]),
                    f"{row['tail_moves']}/{row['total_moves']}",
                    f"{100 * row['tile_share']:.2f}%",
                    f"{100 * row['special_share']:.2f}%",
                    f"{_fmt(row['tiles_all']['mean'])}/{_fmt(row['tiles_without_tail']['mean'])}",
                    f"{_fmt(row['specials_all']['mean'])}/{_fmt(row['specials_without_tail']['mean'])}",
                ]
                for row in move_analysis["tail_sensitivity"]
            ],
        ),
        "",
        "Median clear/activation counts and p90/p95/p99 are retained alongside these means in analysis.json. This distinguishes a small tail contribution from broad distributional differences.",
        "",
        _table(
            ["Level", "Agent", "Seed", "Move", "Action", "Cascades", "Tiles cleared"],
            [
                [
                    row["level_id"],
                    row["agent_name"],
                    row["seed"],
                    row["move_index"],
                    row["action_id"],
                    row["cascade_count"],
                    row["cleared_tiles"],
                ]
                for row in move_analysis["outliers"]
            ],
        ),
        "",
        "## Special activations",
        "",
        "Episode totals count recorded activation steps. Move records also contain an actual by_type dictionary, so the available breakdown is aggregated below. These are activation-step counts, not special-creation counts, independent damage-pass counts or a complete inventory of consumed combination partners. No unavailable special geometry or causal contribution is invented.",
        "",
        _table(
            ["Recorded special type", "Activation steps"],
            sorted(move_analysis["special_activation_types"].items()),
        ),
        "",
        _table(
            [
                "Agent",
                "Outcome",
                "Episode n / move n",
                "Activations mean / median per episode",
                "Recorded type counts",
            ],
            [
                [
                    row["agent_name"],
                    row["outcome"],
                    f"{row['episodes']} / {row['moves']}",
                    f"{_fmt(row['per_episode']['mean'])}/{_fmt(row['per_episode']['median'])}",
                    ", ".join(
                        f"{key}={value}" for key, value in sorted(row["types"].items())
                    ),
                ]
                for row in move_analysis["specials_by_agent_outcome"]
            ],
        ),
        "",
        "Per-level/agent and wins/losses comparisons appear in the descriptive and conditional tables. Longer episodes and longer cascades create more opportunities to activate specials; pooled activation differences do not establish that activations cause winning. Robust move-level summaries and tail sensitivity above quantify the concentration of activity.",
        "",
        "## Level findings: observations and hypotheses",
        "",
        "Observed: Wooden Trouble and Frozen Garden are near the win-rate ceiling for both baselines; Stone Breaker has low win rates and leaves substantial blocker HP. Blocker Mayhem often shows high objective completion despite losses, so aggregate completion can conceal the last surviving blocker or incomplete objective. First Match initialization is missing for one paired seed, and its error rate must remain visible.",
        "",
        "Hypotheses for future controlled tests: three-color boards may support long chains, while STONE objectives may expose the limitation of immediate normal-match scoring. These levels differ jointly in mask, dimensions, colors, blocker layout, HP and objectives; no single feature is identified as the cause. Shorter moves-to-win also reflect early stopping, not necessarily superior efficiency across unmatched outcome sets. These are baseline-agent outcomes, not human difficulty estimates.",
        "",
        "## Initialization error investigation (read-only)",
        "",
    ]
    if diagnostics is None:
        lines += [
            "Not run for this invocation (--skip-initialization-investigation).",
            "",
        ]
    else:
        lines += [
            f"Historical three-color First Match seed 52028 reproduced twice: {diagnostics['repetitions']}. The live level may have since changed; this diagnosis explicitly preserves the original three-color configuration. Every candidate in the first 1,000 contained a valid existing match; legal-move checking was never reached. Horizontal/vertical/square categories overlap. No evidence was found of rejecting a match-free playable board unnecessarily.",
            "",
            f"Derived board seed {diagnostics['board_seed']}, engine seed {diagnostics['engine_seed']}. An isolated diagnostic call with a 10,000-attempt bound, using the unchanged generator and same seed, first accepted at attempt {diagnostics['continuation']['first_playable_attempt']}, yielding {diagnostics['continuation']['legal_moves']} legal moves and no initial matches. The configuration is therefore feasible; this particular deterministic candidate sequence exhausts the normal 1,000-attempt bound.",
            "",
            f"A separate fixed-seed Monte Carlo sample of {diagnostics['sample_size']:,} 6×6, three-color candidates produced {diagnostics['accepted']} acceptable boards, {diagnostics['matched']} boards with matches and {diagnostics['dead_match_free']} match-free dead boards. Estimated acceptance: {100 * diagnostics['estimated_acceptance_probability']:.3f}%; approximate independent-attempt failure probability at 1,000 draws: {100 * diagnostics['approximate_1000_attempt_failure_probability']:.3f}%. This probability is an estimate, not a guarantee for a particular seeded PRNG sequence.",
            "",
            f"The candidate acceptance Wilson 95% interval is [{100 * diagnostics['acceptance_ci95'][0]:.3f}%, {100 * diagnostics['acceptance_ci95'][1]:.3f}%]. Under the same independence approximation, this maps to a 1,000-attempt failure-probability interval of [{100 * diagnostics['failure_probability_ci95'][0]:.4f}%, {100 * diagnostics['failure_probability_ci95'][1]:.4f}%]. This captures Monte Carlo sampling uncertainty, not systematic PRNG or candidate-dependence effects.",
            "",
            "Repeated RL resets can encounter the same bounded-generation failure, independently of policy. Proposed separate change for approval: expose the initialization attempt budget and rejection diagnostics through LevelSession/Match3LevelEnv, then evaluate a larger explicit budget or a bounded constructive initializer. A larger budget costs reset time and cannot guarantee success; constructive generation changes the sampling distribution and needs a separate review. Do not silently retry with another episode seed. No production generator, seed, budget or level was changed here.",
            "",
        ]
    lines += [
        "## Figures",
        "",
        "PNG files are 300 dpi; PDF versions are vector exports. Full 0–100% axes are used for win rates. Boxplots show distributions with means where marked; empty loss groups stay empty. Symlog/log scales are labeled explicitly. Initialization errors have no board-dependent observations and are omitted from those plots only, with denominators documented above.",
        "",
    ]
    for name in figures:
        lines += [
            f"### {name.replace('_', ' ')}",
            "",
            f"![{name}](figures/{name}.png)",
            "",
            f"[Vector PDF](figures/{name}.pdf)",
            "",
        ]
    lines += [
        "## Limitations and recommendations",
        "",
        "Only 100 paired seeds per level, two simple policies and one fixed set of six levels are observed. Win estimates near 0/100% have wide uncertainty despite ceiling point estimates. Six exploratory comparisons require multiplicity awareness; percentile bootstrap intervals can be unreliable with sparse discordance. Moves are clustered within episodes, and timing depends on hardware. Error exclusion changes the hypothesis-test population (First Match n=99 pairs), while descriptive win denominators remain 100 attempts per agent.",
        "",
        "Future work should preserve held-out evaluation seeds and separate training seeds, repair bounded-reset reliability in an approved sprint, report per-objective completion rather than relying only on its average, and compare an objective/special-aware baseline against these references. Use larger paired evaluation sets and pre-specified practical effect thresholds before claiming equivalence or a useful improvement. Cascade budget 200 remains provisional: observed maximum 153 does not guarantee all future chains fit. No new agents, reward shaping, DQN or PPO were implemented.",
        "",
        "## Reproduction",
        "",
        "Run `python -m match3_ai.evaluation.analysis --input-dir <benchmark-dir> --output-dir <new-analysis-dir>` inside match3-ai. Install only the optional `[analysis]` extra when Matplotlib is unavailable. NumPy and standard-library CSV/JSON/math suffice for statistics; pandas/SciPy are not required. Existing output paths are rejected. Bootstrap seed, sample count, software versions, hashes, missingness details and all conditional numeric summaries are recorded in analysis.json. Original CSV/JSON/summary inputs remain untouched.",
        "",
    ]
    (output / "report.md").write_text("\n".join(lines), encoding="utf-8")


def run_analysis(
    input_dir: str | Path,
    output_dir: str | Path,
    *,
    bootstrap_seed: int = 55005,
    bootstrap_samples: int = 20000,
    initialization_investigation: bool = True,
) -> dict:
    source, output = Path(input_dir).resolve(), Path(output_dir).resolve()
    _require(
        output != source and output not in source.parents,
        "analysis output must be separate from benchmark input",
    )
    if output.exists():
        raise FileExistsError(f"analysis output already exists: {output}")
    data = load_dataset(source)
    quality = validate_dataset(data)
    descriptive = describe_episodes(data)
    comparisons = compare_agents(data, seed=bootstrap_seed, samples=bootstrap_samples)
    move_analysis = analyze_moves(data)
    diagnostics = None
    if initialization_investigation:
        from match3_ai.evaluation.initialization_diagnostics import (
            investigate_initialization,
        )

        diagnostics = investigate_initialization()
        diagnostics["acceptance_ci95"] = wilson_interval(
            diagnostics["accepted"], diagnostics["sample_size"]
        )
        low, high = diagnostics["acceptance_ci95"]
        diagnostics["failure_probability_ci95"] = [
            (1 - high) ** 1000,
            (1 - low) ** 1000,
        ]
    output.mkdir(parents=True, exist_ok=False)
    figures = generate_figures(data, descriptive, comparisons, output / "figures")
    results = {
        "experiment_id": data.metadata["experiment_id"],
        "input_hashes": data.hashes,
        "analysis_source_hashes": {
            name: hashlib.sha256(
                Path(__file__).with_name(name).read_bytes()
            ).hexdigest()
            for name in ("analysis.py", "initialization_diagnostics.py")
        },
        "bootstrap_seed": bootstrap_seed,
        "bootstrap_samples": bootstrap_samples,
        "versions": {name: version(name) for name in ("numpy", "matplotlib")},
        "quality": quality,
        "descriptive": descriptive,
        "paired_tests": comparisons,
        "moves": move_analysis,
        "initialization_diagnostics": diagnostics,
        "figures": figures,
    }
    (output / "analysis.json").write_text(
        json.dumps(results, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    write_report(
        data,
        quality,
        descriptive,
        comparisons,
        move_analysis,
        diagnostics,
        figures,
        output,
    )
    for name, digest in data.hashes.items():
        _require(
            hashlib.sha256((source / name).read_bytes()).hexdigest() == digest,
            "benchmark input changed during analysis",
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("experiments/results/6beb0e805eea3e9c25cc9de5"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/sprint5_5_6beb0e805eea3e9c25cc9de5"),
    )
    parser.add_argument("--bootstrap-seed", type=int, default=55005)
    parser.add_argument("--bootstrap-samples", type=int, default=20000)
    parser.add_argument("--skip-initialization-investigation", action="store_true")
    args = parser.parse_args()
    result = run_analysis(
        args.input_dir,
        args.output_dir,
        bootstrap_seed=args.bootstrap_seed,
        bootstrap_samples=args.bootstrap_samples,
        initialization_investigation=not args.skip_initialization_investigation,
    )
    print(
        f"Validated {result['quality']['episode_shape'][0]} episodes and {result['quality']['move_shape'][0]} moves; report: {args.output_dir.resolve() / 'report.md'}"
    )


if __name__ == "__main__":
    main()
