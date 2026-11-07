from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

# ==================================================
# 1. PATH CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "analysis" / "data" / "stage2_episodes_flat.csv"

FIGURE_DIR = PROJECT_ROOT / "analysis" / "figures"
TABLE_DIR = PROJECT_ROOT / "analysis" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================
# 2. LOAD AND PREPARE DATA
# ==================================================

df = pd.read_csv(DATA_PATH)

df["win"] = df["outcome"].eq("WON").astype(int)

df["cascades_per_move"] = df["total_cascades"] / df["moves_used"].replace(
    0, float("nan")
)

df["cleared_per_move"] = df["total_cleared"] / df["moves_used"].replace(0, float("nan"))

level_names = {
    "double-trouble__original": "Double Original",
    "double-trouble__colors5-moves6": "Double v2",
    "frozen-garden__original": "Frozen Original",
    "frozen-garden__colors5-moves10": "Frozen v2",
    "wooden-trouble__original": "Wooden Original",
    "wooden-trouble__colors5-double-moves6": "Wooden v2",
}

df["level_label"] = df["variant_id"].map(level_names)

if df["level_label"].isna().any():
    raise ValueError("Unknown level variant found in dataset.")


# ==================================================
# 3. GAMEPLAY METRICS BY OUTCOME
# ==================================================

metrics = [
    "total_cascades",
    "cascades_per_move",
    "total_cleared",
    "cleared_per_move",
    "special_activations",
    "reshuffles",
    "max_cascades_in_move",
]

summary = (
    df.groupby(["variant_id", "level_label", "agent_name", "outcome"])[metrics]
    .mean()
    .reset_index()
)

summary.to_csv(
    TABLE_DIR / "stage2_gameplay_mechanics.csv",
    index=False,
)

print("\n========== GAMEPLAY MECHANICS ==========")
print(summary.to_string(index=False))


# ==================================================
# 4. CASCADES PER MOVE: WON VS LOST
# ==================================================

calibrated = df[~df["variant_id"].str.endswith("__original")].copy()

cascade_summary = (
    calibrated.groupby(["level_label", "agent_name", "outcome"])["cascades_per_move"]
    .mean()
    .reset_index()
)

cascade_summary.to_csv(
    TABLE_DIR / "stage2_cascade_outcome_summary.csv",
    index=False,
)

pivot = cascade_summary.pivot_table(
    index=["level_label", "agent_name"],
    columns="outcome",
    values="cascades_per_move",
)

fig, ax = plt.subplots(figsize=(12, 6))

pivot.plot(
    kind="bar",
    ax=ax,
    rot=0,
)

ax.set_title("Cascade Efficiency by Outcome")
ax.set_xlabel("Level and Agent")
ax.set_ylabel("Mean Cascades per Move")

ax.set_xticklabels([f"{level}\n{agent}" for level, agent in pivot.index])

ax.legend(title="Outcome")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3,
        fontsize=9,
    )

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "stage2_cascades_per_move.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ==================================================
# 5. ORIGINAL VS V2: ALL EPISODES
# ==================================================

# Compare complete episode populations, not only wins.
comparison_metrics = [
    "win",
    "moves_used",
    "total_cascades",
    "cascades_per_move",
    "total_cleared",
    "cleared_per_move",
    "special_activations",
    "objective_completion_ratio",
]

level_comparison = (
    df.groupby(["level_label", "agent_name"])[comparison_metrics].mean().reset_index()
)

level_comparison.to_csv(
    TABLE_DIR / "stage2_original_vs_v2_mechanics.csv",
    index=False,
)

print("\n========== ORIGINAL VS V2 ==========")
print(level_comparison.to_string(index=False))


# ==================================================
# 6. ORIGINAL VS V2 CASCADE FIGURE
# ==================================================

cascade_comparison = level_comparison.pivot(
    index="level_label",
    columns="agent_name",
    values="cascades_per_move",
)

level_order = [
    "Wooden Original",
    "Wooden v2",
    "Double Original",
    "Double v2",
    "Frozen Original",
    "Frozen v2",
]

cascade_comparison = cascade_comparison.reindex(level_order)

fig, ax = plt.subplots(figsize=(11, 6))

cascade_comparison.plot(
    kind="bar",
    ax=ax,
    rot=20,
)

ax.set_title("Cascade Efficiency: Original vs Calibrated Levels")
ax.set_xlabel("Level Configuration")
ax.set_ylabel("Mean Cascades per Move")

for container in ax.containers:
    ax.bar_label(
        container,
        fmt="%.2f",
        padding=3,
        fontsize=9,
    )

fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "stage2_original_vs_v2_cascades.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close(fig)


# ==================================================
# 7. OUTPUT SUMMARY
# ==================================================

print("\n========== OUTPUT FILES ==========")

for path in [
    TABLE_DIR / "stage2_gameplay_mechanics.csv",
    TABLE_DIR / "stage2_cascade_outcome_summary.csv",
    TABLE_DIR / "stage2_original_vs_v2_mechanics.csv",
    FIGURE_DIR / "stage2_cascades_per_move.png",
    FIGURE_DIR / "stage2_original_vs_v2_cascades.png",
]:
    print(f"{path.relative_to(PROJECT_ROOT)} | exists={path.exists()}")

print("\nGameplay Mechanics Analysis completed.")
