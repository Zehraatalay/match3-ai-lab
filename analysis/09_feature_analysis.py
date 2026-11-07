from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "analysis" / "data" / "stage2_ml_dataset.csv"

FIGURE_DIR = PROJECT_ROOT / "analysis" / "figures"
TABLE_DIR = PROJECT_ROOT / "analysis" / "tables"

FIGURE_DIR.mkdir(parents=True, exist_ok=True)
TABLE_DIR.mkdir(parents=True, exist_ok=True)


FEATURES = [
    "move_limit",
    "num_colors",
    "board_rows",
    "board_cols",
    "playable_cells",
    "initial_blocker_count",
    "initial_blocker_hp",
    "blocker_density",
    "occupied_tile_count",
    "color_entropy",
    "dominant_color_ratio",
    "initial_special_count",
    "legal_moves_count",
    "objective_target_total",
]


def main():

    df = pd.read_csv(DATA_PATH)

    # Focus on calibrated configurations.
    calibrated = df[~df["variant_id"].str.endswith("__original")].copy()

    print("\n========== DATASET ==========")

    print("All episodes:", len(df))
    print("Calibrated episodes:", len(calibrated))

    # ==================================================
    # 1. FEATURE VARIABILITY
    # ==================================================

    variability = pd.DataFrame(
        {
            "feature": FEATURES,
            "unique_values": [calibrated[col].nunique() for col in FEATURES],
            "std": [calibrated[col].std() for col in FEATURES],
            "missing_pct": [calibrated[col].isna().mean() * 100 for col in FEATURES],
        }
    )

    variability.to_csv(
        TABLE_DIR / "stage2_feature_variability.csv",
        index=False,
    )

    print("\n========== FEATURE VARIABILITY ==========")

    print(variability.to_string(index=False))

    # ==================================================
    # 2. CORRELATION MATRIX
    # ==================================================

    correlation = calibrated[FEATURES].corr()

    correlation.to_csv(TABLE_DIR / "stage2_feature_correlations.csv")

    fig, ax = plt.subplots(figsize=(12, 10))

    image = ax.imshow(
        correlation,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
    )

    ax.set_xticks(range(len(FEATURES)))
    ax.set_yticks(range(len(FEATURES)))

    ax.set_xticklabels(
        FEATURES,
        rotation=90,
        fontsize=8,
    )

    ax.set_yticklabels(
        FEATURES,
        fontsize=8,
    )

    fig.colorbar(image, ax=ax)

    ax.set_title("Feature Correlation Matrix — Calibrated Levels")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "stage2_feature_correlations.png",
        dpi=160,
    )

    plt.close(fig)

    # ==================================================
    # 3. LEGAL MOVES DISTRIBUTION
    # ==================================================

    fig, ax = plt.subplots(figsize=(9, 5))

    for outcome, label in [
        (0, "Lost"),
        (1, "Won"),
    ]:
        values = calibrated.loc[
            calibrated["win"] == outcome,
            "legal_moves_count",
        ]

        ax.hist(
            values,
            bins=15,
            alpha=0.55,
            density=True,
            label=label,
        )

    ax.set_xlabel("Initial Legal Moves")
    ax.set_ylabel("Density")
    ax.set_title("Legal Moves Distribution by Outcome")

    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "stage2_legal_moves_distribution.png",
        dpi=160,
    )

    plt.close(fig)

    # ==================================================
    # 4. WIN RATE BY LEGAL MOVES
    # ==================================================

    # Avoid unstable estimates for tiny groups.
    summary = (
        calibrated.groupby("legal_moves_count")
        .agg(
            episodes=("win", "size"),
            wins=("win", "sum"),
            win_rate=("win", "mean"),
        )
        .reset_index()
    )

    summary.to_csv(
        TABLE_DIR / "stage2_win_rate_by_legal_moves.csv",
        index=False,
    )

    reliable = summary[summary["episodes"] >= 20].copy()

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        reliable["legal_moves_count"],
        reliable["win_rate"],
        marker="o",
    )

    ax.set_xlabel("Initial Legal Moves")
    ax.set_ylabel("Win Rate")
    ax.set_ylim(0, 1)

    ax.set_title("Win Rate vs Initial Legal Moves")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "stage2_win_rate_by_legal_moves.png",
        dpi=160,
    )

    plt.close(fig)

    # ==================================================
    # 5. WITHIN-LEVEL ANALYSIS
    # ==================================================

    within_level = (
        calibrated.groupby(["variant_id", "agent_name"])
        .apply(
            lambda group: pd.Series(
                {
                    "episodes": len(group),
                    "win_rate": group["win"].mean(),
                    "legal_moves_win_corr": (
                        group["legal_moves_count"].corr(group["win"])
                    ),
                    "entropy_win_corr": (group["color_entropy"].corr(group["win"])),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )

    within_level.to_csv(
        TABLE_DIR / "stage2_within_level_correlations.csv",
        index=False,
    )

    print("\n========== WITHIN-LEVEL ANALYSIS ==========")

    print(within_level.to_string(index=False))

    print("\nFeature analysis completed.")


if __name__ == "__main__":
    main()
