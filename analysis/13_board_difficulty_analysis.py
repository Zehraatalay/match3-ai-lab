from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "analysis/data/stage2_ml_dataset.csv"

TABLE_DIR = ROOT / "analysis/tables"
FIGURE_DIR = ROOT / "analysis/figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


BOARD_FEATURES = [
    "color_entropy",
    "dominant_color_ratio",
    "legal_moves_count",
]


def classify_board(row):
    random_win = int(row["Random"])
    greedy_win = int(row["Greedy"])

    if random_win == 1 and greedy_win == 1:
        return "Both Win"

    if random_win == 0 and greedy_win == 0:
        return "Both Lose"

    if random_win == 1 and greedy_win == 0:
        return "Random Only"

    return "Greedy Only"


def main():
    df = pd.read_csv(DATA_PATH)

    calibrated = df[~df["variant_id"].str.endswith("__original")].copy()

    # ==================================================
    # 1. CREATE BOARD-LEVEL DATASET
    # ==================================================

    board_outcomes = calibrated.pivot(
        index=["variant_id", "seed"],
        columns="agent_name",
        values="win",
    ).reset_index()

    if board_outcomes[["Random", "Greedy"]].isna().any().any():
        raise ValueError("Missing paired agent outcome.")

    board_features = calibrated[
        ["variant_id", "seed", *BOARD_FEATURES]
    ].drop_duplicates(subset=["variant_id", "seed"])

    board_df = board_outcomes.merge(
        board_features,
        on=["variant_id", "seed"],
        validate="one_to_one",
    )

    board_df["difficulty_group"] = board_df.apply(
        classify_board,
        axis=1,
    )

    board_df["wins_out_of_two"] = board_df["Random"] + board_df["Greedy"]

    if len(board_df) != 600:
        raise ValueError("Unexpected board count.")

    board_df.to_csv(
        TABLE_DIR / "stage2_board_difficulty.csv",
        index=False,
    )

    # ==================================================
    # 2. GROUP DISTRIBUTION
    # ==================================================

    distribution = (
        board_df.groupby(["variant_id", "difficulty_group"])
        .size()
        .unstack(fill_value=0)
    )

    distribution.to_csv(TABLE_DIR / "stage2_board_difficulty_distribution.csv")

    print("\n========== BOARD DIFFICULTY DISTRIBUTION ==========")

    print(distribution.to_string())

    # ==================================================
    # 3. FEATURE COMPARISON
    # ==================================================

    feature_summary = board_df.groupby(["variant_id", "difficulty_group"])[
        BOARD_FEATURES
    ].agg(["mean", "std", "count"])

    feature_summary.to_csv(TABLE_DIR / "stage2_board_difficulty_features.csv")

    print("\n========== FEATURE MEANS ==========")

    means = board_df.groupby(["variant_id", "difficulty_group"])[BOARD_FEATURES].mean()

    print(means.round(4).to_string())

    # ==================================================
    # 4. VISUALIZE GROUP DISTRIBUTION
    # ==================================================

    order = [
        "Both Lose",
        "Random Only",
        "Greedy Only",
        "Both Win",
    ]

    distribution = distribution.reindex(
        columns=order,
        fill_value=0,
    )

    ax = distribution.plot(
        kind="bar",
        stacked=True,
        figsize=(11, 6),
    )

    ax.set_title("Board Outcomes by Level Configuration")

    ax.set_xlabel("Level Configuration")
    ax.set_ylabel("Number of Boards")

    plt.xticks(rotation=20, ha="right")

    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / "stage2_board_difficulty_distribution.png",
        dpi=160,
    )

    plt.close()

    # ==================================================
    # 5. LEGAL MOVES BY BOARD GROUP
    # ==================================================

    fig, ax = plt.subplots(figsize=(9, 5))

    data = [
        board_df.loc[
            board_df["difficulty_group"] == group,
            "legal_moves_count",
        ]
        for group in order
    ]

    ax.boxplot(
        data,
        tick_labels=order,
    )

    ax.set_title("Initial Legal Moves by Board Outcome")

    ax.set_ylabel("Initial Legal Moves")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "stage2_board_difficulty_legal_moves.png",
        dpi=160,
    )

    plt.close(fig)

    # ==================================================
    # 6. ENTROPY BY BOARD GROUP
    # ==================================================

    fig, ax = plt.subplots(figsize=(9, 5))

    data = [
        board_df.loc[
            board_df["difficulty_group"] == group,
            "color_entropy",
        ]
        for group in order
    ]

    ax.boxplot(
        data,
        tick_labels=order,
    )

    ax.set_title("Color Entropy by Board Outcome")

    ax.set_ylabel("Color Entropy")

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "stage2_board_difficulty_entropy.png",
        dpi=160,
    )

    plt.close(fig)

    print("\nBoard difficulty analysis completed.")


if __name__ == "__main__":
    main()
