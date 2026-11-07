import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from match3_ai.evaluation.behavior_data import observation_fingerprint
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.loader import load_level

# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "results" / "level_calibration_stage2_confirmation"
)

OUTPUT_DIR = PROJECT_ROOT / "analysis" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

BOARD_OUTPUT = OUTPUT_DIR / "stage2_board_features.csv"
ML_OUTPUT = OUTPUT_DIR / "stage2_ml_dataset.csv"


# ==================================================
# 1. HELPERS
# ==================================================


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def calculate_entropy(values):
    """
    Shannon entropy of tile colors.
    """

    if len(values) == 0:
        return 0.0

    _, counts = np.unique(values, return_counts=True)

    probabilities = counts / counts.sum()

    return float(-(probabilities * np.log2(probabilities)).sum())


def extract_board_features(observation, config):
    """
    Extract features available before the episode starts.
    """

    tiles = np.asarray(observation["tiles"])
    specials = np.asarray(observation["specials"])
    blockers = np.asarray(observation["blockers"])
    blocker_hp = np.asarray(observation["blocker_hp"])
    cell_mask = np.asarray(observation["cell_mask"])
    action_mask = np.asarray(observation["action_mask"])

    playable_mask = cell_mask.astype(bool)

    # Only occupied, playable, non-blocker cells.
    tile_mask = playable_mask & (tiles > 0) & (blockers == 0)

    tile_colors = tiles[tile_mask]

    playable_cells = int(playable_mask.sum())

    blocker_count = int(((blockers > 0) & playable_mask).sum())

    total_blocker_hp = int(blocker_hp[playable_mask].sum())

    if len(tile_colors) > 0:
        _, color_counts = np.unique(
            tile_colors,
            return_counts=True,
        )

        dominant_color_ratio = float(color_counts.max() / color_counts.sum())
    else:
        dominant_color_ratio = 0.0

    objective_mask = np.asarray(observation["objective_mask"]).astype(bool)

    objective_target = np.asarray(observation["objective_target"])

    features = {
        "move_limit": int(config.move_limit),
        "num_colors": int(config.num_colors),
        "board_rows": int(observation["board_shape"][0]),
        "board_cols": int(observation["board_shape"][1]),
        "playable_cells": playable_cells,
        "initial_blocker_count": blocker_count,
        "initial_blocker_hp": total_blocker_hp,
        "blocker_density": (
            blocker_count / playable_cells if playable_cells > 0 else 0.0
        ),
        "occupied_tile_count": int(tile_mask.sum()),
        "color_entropy": calculate_entropy(tile_colors),
        "dominant_color_ratio": dominant_color_ratio,
        "initial_special_count": int(((specials > 0) & playable_mask).sum()),
        "legal_moves_count": int(np.count_nonzero(action_mask)),
        "objective_target_total": int(objective_target[objective_mask].sum()),
    }

    return features


# ==================================================
# 2. LOAD EXPERIMENT DATA
# ==================================================


def main():

    manifest = json.loads((EXPERIMENT_DIR / "manifest.json").read_text())

    episode_records = []

    with (EXPERIMENT_DIR / "episodes.jsonl").open() as file:
        for line in file:
            episode_records.append(json.loads(line))

    print("\nLoaded episodes:", len(episode_records))

    # One reconstruction per unique variant + seed.
    unique_boards = {}

    for episode in episode_records:
        key = (
            episode["variant_id"],
            int(episode["seed"]),
        )

        if key not in unique_boards:
            unique_boards[key] = episode
        else:
            previous = unique_boards[key]

            if previous["initial_identity"] != episode["initial_identity"]:
                raise ValueError(f"Paired initial identity mismatch: {key}")

    print("Unique boards:", len(unique_boards))

    if len(episode_records) != 2400:
        raise ValueError("Unexpected episode count.")

    if len(unique_boards) != 1200:
        raise ValueError("Unexpected unique board count.")

    # ==================================================
    # 3. RECONSTRUCT BOARDS
    # ==================================================

    board_rows = []

    for variant_id, specification in manifest["configurations"].items():
        config_path = EXPERIMENT_DIR / specification["path"]

        if sha256_file(config_path) != specification["sha256"]:
            raise ValueError(f"Configuration hash mismatch: {variant_id}")

        config = load_level(config_path)

        variant_keys = sorted(
            [key for key in unique_boards if key[0] == variant_id],
            key=lambda key: key[1],
        )

        print(f"\nProcessing {variant_id}: {len(variant_keys)} boards")

        for _, seed in variant_keys:
            episode = unique_boards[(variant_id, seed)]

            env = Match3LevelEnv(
                config.level_id,
                level_config=config,
            )

            try:
                observation, info = env.reset(seed=seed)

                recorded = episode["initial_identity"]

                reconstructed_identity = {
                    "fingerprint": observation_fingerprint(observation),
                    "board_seed": info["board_seed"],
                    "engine_seed": info["engine_seed"],
                }

                if reconstructed_identity != recorded:
                    raise RuntimeError(
                        f"Reconstruction mismatch: {variant_id}, seed={seed}"
                    )

                features = extract_board_features(
                    observation,
                    config,
                )

                board_rows.append(
                    {
                        "variant_id": variant_id,
                        "seed": seed,
                        "initial_fingerprint": recorded["fingerprint"],
                        **features,
                    }
                )

            finally:
                env.close()

    board_df = pd.DataFrame(board_rows)

    if len(board_df) != 1200:
        raise ValueError("Board feature count mismatch.")

    board_df.to_csv(
        BOARD_OUTPUT,
        index=False,
    )

    # ==================================================
    # 4. JOIN WITH EPISODE OUTCOMES
    # ==================================================

    outcome_rows = []

    for episode in episode_records:
        record = episode["record"]

        outcome_rows.append(
            {
                "variant_id": episode["variant_id"],
                "seed": int(episode["seed"]),
                "agent_name": record["agent_name"],
                "outcome": record["outcome"],
                "win": int(record["outcome"] == "WON"),
            }
        )

    outcomes_df = pd.DataFrame(outcome_rows)

    if outcomes_df.duplicated(["variant_id", "seed", "agent_name"]).any():
        raise ValueError("Duplicate episode records.")

    ml_df = outcomes_df.merge(
        board_df,
        on=["variant_id", "seed"],
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    if not ml_df["_merge"].eq("both").all():
        raise ValueError("Unmatched episode records.")

    ml_df = ml_df.drop(columns="_merge")

    if len(ml_df) != 2400:
        raise ValueError("ML dataset row count mismatch.")

    if ml_df.isna().any().any():
        raise ValueError("Missing values in ML dataset.")

    ml_df.to_csv(
        ML_OUTPUT,
        index=False,
    )

    # ==================================================
    # 5. DATASET SUMMARY
    # ==================================================

    print("\n========== FEATURE ENGINEERING ==========")

    print("\nBoard dataset:", board_df.shape)
    print("ML dataset:", ml_df.shape)

    print("\nFeatures:")
    print(board_df.columns.tolist())

    print("\nTarget distribution:")
    print(ml_df["win"].value_counts())

    print("\nMean features by outcome:")

    numeric_features = [
        "move_limit",
        "num_colors",
        "playable_cells",
        "initial_blocker_count",
        "initial_blocker_hp",
        "blocker_density",
        "color_entropy",
        "dominant_color_ratio",
        "legal_moves_count",
    ]

    print(ml_df.groupby("outcome")[numeric_features].mean().round(3).to_string())

    print("\nSaved:")
    print(BOARD_OUTPUT)
    print(ML_OUTPUT)

    print("\nFeature engineering completed.")


if __name__ == "__main__":
    main()
