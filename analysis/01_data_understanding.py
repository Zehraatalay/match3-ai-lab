from pathlib import Path

import pandas as pd

# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = (
    PROJECT_ROOT / "experiments" / "results" / "level_calibration_stage2_confirmation"
)

episodes = pd.read_json(
    DATA_DIR / "episodes.jsonl",
    lines=True,
)

# --------------------------------------------------
# 2. Flatten nested JSON
# --------------------------------------------------

records_df = pd.json_normalize(episodes["record"])

identity_df = pd.json_normalize(episodes["initial_identity"]).add_prefix("initial_")

episode_df = pd.concat(
    [
        episodes[["variant_id"]].reset_index(drop=True),
        records_df.reset_index(drop=True),
        identity_df.reset_index(drop=True),
    ],
    axis=1,
)

# --------------------------------------------------
# 3. Dataset overview
# --------------------------------------------------

print("\n========== DATASET OVERVIEW ==========")

print("Shape:", episode_df.shape)

print("\nColumns:")
print(episode_df.columns.tolist())

print("\nData types:")
print(episode_df.dtypes.to_string())

# --------------------------------------------------
# 4. Data quality
# --------------------------------------------------

print("\n========== DATA QUALITY ==========")

print("\nMissing values:")
print(
    episode_df.isna()
    .sum()
    .loc[lambda x: x > 0]
    .sort_values(ascending=False)
    .to_string()
)

print("\nDuplicate episode IDs:")
print(episode_df["episode_id"].duplicated().sum())

print("\nErrors:")
print(episode_df["error_type"].value_counts(dropna=False))

# --------------------------------------------------
# 5. Experiment structure
# --------------------------------------------------

print("\n========== EXPERIMENT STRUCTURE ==========")

print("\nAgents:")
print(episode_df["agent_name"].value_counts())

print("\nVariants:")
print(episode_df["variant_id"].value_counts())

print("\nEpisodes per variant and agent:")
print(episode_df.groupby(["variant_id", "agent_name"]).size().to_string())

# --------------------------------------------------
# 6. Outcome inspection
# --------------------------------------------------

print("\n========== OUTCOMES ==========")

print("\nCompleted:")
print(episode_df["completed"].value_counts(dropna=False))

print("\nReward distribution:")
print(episode_df["total_reward"].value_counts(dropna=False))

print("\nObjective completion:")
print(episode_df["objective_completion_ratio"].describe())

print("\nMoves used:")
print(episode_df["moves_used"].describe())

# --------------------------------------------------
# 7. Export flattened dataset
# --------------------------------------------------

OUTPUT_DIR = PROJECT_ROOT / "analysis" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

output_path = OUTPUT_DIR / "stage2_episodes_flat.csv"

episode_df.to_csv(output_path, index=False)

print(f"\nDataset saved to: {output_path}")
