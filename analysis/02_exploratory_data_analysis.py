from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "analysis" / "data" / "stage2_episodes_flat.csv"

OUTPUT_DIR = PROJECT_ROOT / "analysis" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# Load dataset
df = pd.read_csv(DATA_PATH)

# Create target variable
df["win"] = df["outcome"].eq("WON").astype(int)


# Calculate win rates
win_rates = (
    df.groupby(["variant_id", "agent_name"])
    .agg(
        episodes=("win", "size"),
        wins=("win", "sum"),
        win_rate=("win", "mean"),
    )
    .reset_index()
)

win_rates["win_rate_pct"] = win_rates["win_rate"] * 100
TABLE_DIR = PROJECT_ROOT / "analysis" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

win_rates.to_csv(
    TABLE_DIR / "stage2_win_rates.csv",
    index=False,
)
print("\n========== WIN RATE ANALYSIS ==========")
print(win_rates.to_string(index=False))


# Visualize
pivot = win_rates.pivot(
    index="variant_id",
    columns="agent_name",
    values="win_rate_pct",
)

ax = pivot.plot(
    kind="bar",
    figsize=(12, 6),
    rot=30,
)

ax.set_title("Win Rate by Level Configuration and Agent")
ax.set_xlabel("Level Configuration")
ax.set_ylabel("Win Rate (%)")
ax.set_ylim(0, 105)

plt.tight_layout()

output_path = OUTPUT_DIR / "win_rate_comparison.png"

plt.savefig(output_path, dpi=150)
plt.show()

print(f"\nFigure saved to: {output_path}")

# --------------------------------------------------
# 2. Loss analysis
# --------------------------------------------------

losses = df[df["win"] == 0].copy()

loss_summary = (
    losses.groupby(["variant_id", "agent_name"])
    .agg(
        losses=("win", "size"),
        mean_moves=("moves_used", "mean"),
        mean_objective_completion=("objective_completion_ratio", "mean"),
        mean_blocker_hp_remaining=("blocker_hp_remaining", "mean"),
        mean_blockers_remaining=("blockers_remaining", "mean"),
    )
    .reset_index()
)

print("\n========== LOSS ANALYSIS ==========")
print(loss_summary.to_string(index=False))


# --------------------------------------------------
# 3. Objective completion by outcome
# --------------------------------------------------

completion_summary = (
    df.groupby(["variant_id", "agent_name", "outcome"])
    .agg(
        episodes=("outcome", "size"),
        mean_completion=("objective_completion_ratio", "mean"),
        median_completion=("objective_completion_ratio", "median"),
    )
    .reset_index()
)

print("\n========== OBJECTIVE COMPLETION ==========")
print(completion_summary.to_string(index=False))


# --------------------------------------------------
# 4. Visualize objective completion in losses
# --------------------------------------------------

loss_completion = losses.pivot_table(
    index="variant_id",
    columns="agent_name",
    values="objective_completion_ratio",
    aggfunc="mean",
)

ax = loss_completion.plot(
    kind="bar",
    figsize=(12, 6),
    rot=30,
)

ax.set_title("Average Objective Completion in Lost Episodes")
ax.set_xlabel("Level Configuration")
ax.set_ylabel("Mean Objective Completion")
ax.set_ylim(0, 1.05)

plt.tight_layout()

output_path = OUTPUT_DIR / "loss_objective_completion.png"

plt.savefig(output_path, dpi=150)
plt.show()

print(f"\nFigure saved to: {output_path}")

# --------------------------------------------------
# LOSS ANALYSIS
# --------------------------------------------------

losses = df[df["win"] == 0].copy()

loss_summary = (
    losses.groupby(["variant_id", "agent_name"])
    .agg(
        losses=("win", "size"),
        mean_moves=("moves_used", "mean"),
        mean_objective_completion=("objective_completion_ratio", "mean"),
        mean_blocker_hp_remaining=("blocker_hp_remaining", "mean"),
        mean_blockers_remaining=("blockers_remaining", "mean"),
    )
    .reset_index()
)

print("\n========== LOSS ANALYSIS ==========")
print(loss_summary.to_string(index=False))


# --------------------------------------------------
# SAVE ANALYSIS TABLE
# --------------------------------------------------

TABLE_DIR = PROJECT_ROOT / "analysis" / "tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

loss_summary.to_csv(
    TABLE_DIR / "stage2_loss_summary.csv",
    index=False,
)


# --------------------------------------------------
# LOSS OBJECTIVE COMPLETION FIGURE
# --------------------------------------------------

loss_completion = losses.pivot_table(
    index="variant_id",
    columns="agent_name",
    values="objective_completion_ratio",
    aggfunc="mean",
)

ax = loss_completion.plot(
    kind="bar",
    figsize=(12, 6),
    rot=30,
)

ax.set_title("Objective Completion in Lost Episodes")
ax.set_xlabel("Level Configuration")
ax.set_ylabel("Mean Objective Completion")
ax.set_ylim(0, 1.05)

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "stage2_loss_objective_completion.png",
    dpi=200,
    bbox_inches="tight",
)

plt.close()
