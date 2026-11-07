import hashlib
import json
from pathlib import Path

import pandas as pd

from match3_ai.evaluation.behavior_data import observation_fingerprint
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.loader import load_level

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "results" / "level_calibration_stage2_confirmation"
)

OUTPUT_DIR = PROJECT_ROOT / "analysis" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SAMPLES_PER_VARIANT = 5


def sha256_file(path):
    digest = hashlib.sha256()

    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            digest.update(chunk)

    return digest.hexdigest()


def main():
    manifest = json.loads((EXPERIMENT_DIR / "manifest.json").read_text())

    episodes = pd.read_json(
        EXPERIMENT_DIR / "episodes.jsonl",
        lines=True,
    )

    results = []

    for variant_id, specification in manifest["configurations"].items():
        config_path = EXPERIMENT_DIR / specification["path"]

        actual_hash = sha256_file(config_path)
        expected_hash = specification["sha256"]

        if actual_hash != expected_hash:
            raise ValueError(f"Configuration hash mismatch: {variant_id}")

        config = load_level(config_path)

        variant_episodes = episodes[episodes["variant_id"] == variant_id]

        # One episode per seed is enough for reconstruction.
        samples = (
            variant_episodes.drop_duplicates("seed")
            .sort_values("seed")
            .head(SAMPLES_PER_VARIANT)
        )

        for _, episode in samples.iterrows():
            seed = int(episode["seed"])

            env = Match3LevelEnv(
                config.level_id,
                level_config=config,
            )

            try:
                observation, info = env.reset(seed=seed)

                actual_fingerprint = observation_fingerprint(observation)

                recorded = episode["initial_identity"]

                fingerprint_match = actual_fingerprint == recorded["fingerprint"]

                board_seed_match = info["board_seed"] == recorded["board_seed"]

                engine_seed_match = info["engine_seed"] == recorded["engine_seed"]

                results.append(
                    {
                        "variant_id": variant_id,
                        "seed": seed,
                        "fingerprint_match": fingerprint_match,
                        "board_seed_match": board_seed_match,
                        "engine_seed_match": engine_seed_match,
                        "all_match": (
                            fingerprint_match and board_seed_match and engine_seed_match
                        ),
                    }
                )

            finally:
                env.close()

    results_df = pd.DataFrame(results)

    output_path = OUTPUT_DIR / "stage2_board_reconstruction.csv"

    results_df.to_csv(output_path, index=False)

    print("\n========== BOARD RECONSTRUCTION ==========")

    print(results_df.to_string(index=False))

    print("\nTotal samples:", len(results_df))

    print("Successful reconstructions:", results_df["all_match"].sum())

    print("Failed reconstructions:", (~results_df["all_match"]).sum())

    print("\nSaved:", output_path)

    if not results_df["all_match"].all():
        raise RuntimeError("Board reconstruction validation failed.")


if __name__ == "__main__":
    main()
