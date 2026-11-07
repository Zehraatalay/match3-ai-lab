"""Initial blocker damage opportunities on verified Stage 2 development boards.

Ordinary damage includes only ordinary match events. Immediate damage includes
existing special activations and their chains before the first gravity/refill.
Each move starts from an independent copy of the board and engine RNG. Means
and ratios include all legal moves, including zero-damage moves. Empty move
sets produce zeros. Stochastic effects condition on the recorded engine state;
they are not expectations over random special targets.
"""

import hashlib
import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from match3_ai.evaluation.behavior_data import observation_fingerprint
from match3_ai.games.match3.blockers import BlockerDamageRecord
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.board_mechanics import apply_blocker_damage
from match3_ai.games.match3.engine import build_match_damage_events
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.games.match3.moves import Move, find_valid_moves
from match3_ai.games.match3.pattern_detection import detect_patterns, group_patterns
from match3_ai.games.match3.special_resolver import (
    resolve_matched_specials,
    resolve_special_swap,
)
from match3_ai.games.match3.specials import SpecialType
from match3_ai.levels.loader import load_level

ROOT = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = ROOT / "experiments/results/level_calibration_stage2_confirmation"
OUTPUT_PATH = ROOT / "analysis/data/stage2_spatial_features.csv"
FEATURES = (
    "ordinary_damage_move_count",
    "ordinary_damage_move_ratio",
    "mean_ordinary_hp_loss",
    "max_ordinary_hp_loss",
    "immediate_damage_move_count",
    "immediate_damage_move_ratio",
    "mean_immediate_hp_loss",
    "max_immediate_hp_loss",
    "max_immediate_blockers_destroyed",
)


@dataclass(frozen=True)
class MoveDamage:
    ordinary_records: tuple[BlockerDamageRecord, ...]
    immediate_records: tuple[BlockerDamageRecord, ...]


def hp_loss(records: tuple[BlockerDamageRecord, ...]) -> int:
    """Sum accepted HP reductions, preserving repeated hits on a blocker."""
    return sum(record.previous_hp - record.remaining_hp for record in records)


def evaluate_move(
    initial: BoardState, move: Move, engine_rng: np.random.Generator
) -> MoveDamage:
    """Evaluate a legal move, stopping at damage before clearing or gravity."""
    state = initial.copy()
    rng = deepcopy(engine_rng)
    source, destination = move
    special_swap = (
        initial.specials[source] != SpecialType.NONE
        or initial.specials[destination] != SpecialType.NONE
    )
    for array in (state.tiles, state.specials):
        array[source], array[destination] = array[destination], array[source]

    if special_swap:
        activation = resolve_special_swap(state, source, destination, rng)
        ordinary_records = ()
        events = activation.damage_events
    else:
        groups = group_patterns(state.tiles, detect_patterns(state.tiles))
        matched = set().union(*(group.cells for group in groups))
        ordinary_events = build_match_damage_events(
            tuple(group.cells for group in groups), state.rows, state.cols
        )
        activation = resolve_matched_specials(state, matched, rng)
        _, ordinary_records = apply_blocker_damage(state.copy(), ordinary_events)
        events = ordinary_events + activation.damage_events

    _, immediate_records = apply_blocker_damage(state, events)
    return MoveDamage(ordinary_records, immediate_records)


def extract_spatial_features(
    initial: BoardState, engine_rng: np.random.Generator
) -> tuple[dict[str, int | float], int]:
    """Return nine features and the number of independently evaluated legal moves."""
    moves = find_valid_moves(initial)
    damages = [evaluate_move(initial, move, engine_rng) for move in moves]
    count = len(moves)
    features = {}
    for name in ("ordinary", "immediate"):
        losses = [hp_loss(getattr(damage, f"{name}_records")) for damage in damages]
        damaging = sum(loss > 0 for loss in losses)
        features[f"{name}_damage_move_count"] = damaging
        features[f"{name}_damage_move_ratio"] = damaging / count if count else 0.0
        features[f"mean_{name}_hp_loss"] = sum(losses) / count if count else 0.0
        features[f"max_{name}_hp_loss"] = max(losses, default=0)
    features["max_immediate_blockers_destroyed"] = max(
        (
            sum(record.destroyed for record in damage.immediate_records)
            for damage in damages
        ),
        default=0,
    )
    return {name: features[name] for name in FEATURES}, count


def load_calibrated_boards(experiment_dir: Path) -> tuple[dict, dict]:
    """Load calibrated configurations and paired identities without reconstruction."""
    manifest = json.loads((experiment_dir / "manifest.json").read_text())
    specifications = {
        name: value
        for name, value in manifest["configurations"].items()
        if not name.endswith("__original")
    }
    if len(specifications) != 3:
        raise ValueError("Expected exactly three calibrated variants")
    boards, agents = {}, {}
    with (experiment_dir / "episodes.jsonl").open() as handle:
        for line in handle:
            episode = json.loads(line)
            variant = episode["variant_id"]
            if variant.endswith("__original"):
                continue
            if variant not in specifications:
                raise ValueError(f"Unknown calibrated variant: {variant}")
            key = (variant, int(episode["seed"]))
            agent = episode["record"]["agent_name"]
            if agent in agents.setdefault(key, set()):
                raise ValueError(f"Duplicate agent episode: {key}, {agent}")
            agents[key].add(agent)
            previous = boards.setdefault(key, episode)
            if previous["initial_identity"] != episode["initial_identity"]:
                raise ValueError(f"Paired initial identity mismatch: {key}")
    expected_seeds = set(manifest["environment_seeds"])
    if len(expected_seeds) != 200 or len(boards) != 600:
        raise ValueError("Expected 600 calibrated boards and 200 evaluation seeds")
    for variant in specifications:
        if {seed for name, seed in boards if name == variant} != expected_seeds:
            raise ValueError(f"Unexpected calibrated seed coverage: {variant}")
    if any(names != {"Random", "Greedy"} for names in agents.values()):
        raise ValueError("Each calibrated board must have paired Random/Greedy records")
    return specifications, boards


def verify_initial_identity(
    observation: dict, info: dict, recorded: dict, key: tuple
) -> None:
    reconstructed = {
        "fingerprint": observation_fingerprint(observation),
        "board_seed": info["board_seed"],
        "engine_seed": info["engine_seed"],
    }
    if reconstructed != recorded:
        raise RuntimeError(f"Initial board reconstruction mismatch: {key}")


def reconstruct_features(
    experiment_dir: Path = EXPERIMENT_DIR,
) -> tuple[pd.DataFrame, int]:
    """Verify and reconstruct only the 600 calibrated initial boards."""
    specifications, boards = load_calibrated_boards(experiment_dir)
    rows, total_moves = [], 0
    for variant, specification in specifications.items():
        config_path = experiment_dir / specification["path"]
        if (
            hashlib.sha256(config_path.read_bytes()).hexdigest()
            != specification["sha256"]
        ):
            raise ValueError(f"Configuration SHA256 mismatch: {variant}")
        config = load_level(config_path)
        if config.level_id != specification["level_id"]:
            raise ValueError(f"Configuration level mismatch: {variant}")
        print(f"Verifying calibrated variant: {variant}", flush=True)
        for key in sorted(key for key in boards if key[0] == variant):
            _, seed = key
            episode = boards[key]
            env = Match3LevelEnv(config.level_id, level_config=config)
            try:
                observation, info = env.reset(seed=seed)
                verify_initial_identity(
                    observation, info, episode["initial_identity"], key
                )
                initial = env.session.board.copy()
                features, move_count = extract_spatial_features(
                    initial, env.session.engine.rng
                )
                if move_count != int(np.count_nonzero(observation["action_mask"])):
                    raise RuntimeError(f"Legal move count mismatch: {key}")
                rows.append(
                    {
                        "variant_id": variant,
                        "seed": seed,
                        "initial_fingerprint": episode["initial_identity"][
                            "fingerprint"
                        ],
                        **features,
                    }
                )
                total_moves += move_count
            finally:
                env.close()
    frame = pd.DataFrame(rows)
    if len(frame) != 600 or frame.duplicated(["variant_id", "seed"]).any():
        raise ValueError("Expected one feature row per calibrated board")
    if frame.isna().any().any():
        raise ValueError("Missing values in spatial features")
    return frame, total_moves


def main() -> None:
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite existing features: {OUTPUT_PATH}")
    frame, total_moves = reconstruct_features()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x") as handle:
        frame.to_csv(handle, index=False)
    print(f"\nVerified boards: {len(frame)}")
    print(f"Total legal moves evaluated: {total_moves}")
    print("\nMissing-value counts:\n" + frame.isna().sum().to_string())
    numeric = frame[list(FEATURES)]
    print(
        "\nFeature variability:\n"
        + numeric.agg(["min", "max", "mean", "std", "nunique"]).T.to_string()
    )
    print(
        "\nMean features by calibrated variant:\n"
        + frame.groupby("variant_id")[list(FEATURES)].mean().to_string()
    )
    print(f"\nSaved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
