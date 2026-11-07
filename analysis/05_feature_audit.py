from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "analysis" / "data" / "stage2_episodes_flat.csv"

OUTPUT_DIR = PROJECT_ROOT / "analysis" / "tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DATA_PATH)


# --------------------------------------------------
# 1. Feature classification
# --------------------------------------------------

feature_groups = {
    "pre_game": [
        "variant_id",
        "agent_name",
        "level_id",
        "level_config_id",
    ],
    "identifiers": [
        "seed",
        "episode_id",
        "experiment_id",
        "board_seed",
        "engine_seed",
        "initial_fingerprint",
        "initial_board_seed",
        "initial_engine_seed",
    ],
    "post_game": [
        "moves_used",
        "total_cascades",
        "total_cleared",
        "special_activations",
        "reshuffles",
        "objective_completion_ratio",
        "blockers_remaining",
        "blocker_hp_remaining",
        "total_reward",
    ],
    "target": [
        "outcome",
    ],
}

rows = []

for group_name, columns in feature_groups.items():
    for column in columns:
        if column not in df.columns:
            continue

        rows.append(
            {
                "feature": column,
                "group": group_name,
                "dtype": str(df[column].dtype),
                "missing_pct": df[column].isna().mean() * 100,
                "unique_values": df[column].nunique(),
            }
        )

audit = pd.DataFrame(rows)

print("\n========== FEATURE AVAILABILITY AUDIT ==========")
print(audit.to_string(index=False))

audit.to_csv(
    OUTPUT_DIR / "stage2_feature_audit.csv",
    index=False,
)


# --------------------------------------------------
# 2. Pre-game feature diversity
# --------------------------------------------------

print("\n========== PRE-GAME FEATURES ==========")

for column in feature_groups["pre_game"]:
    if column in df.columns:
        print(f"\n{column}:")
        print(df[column].value_counts().head(15))


# --------------------------------------------------
# 3. Target distribution
# --------------------------------------------------

print("\n========== TARGET DISTRIBUTION ==========")

print(df["outcome"].value_counts())

print("\nTarget by level and agent:")

target_distribution = pd.crosstab(
    [df["variant_id"], df["agent_name"]],
    df["outcome"],
    normalize="index",
)

print(target_distribution.to_string())

print("\nFeature audit completed.")
