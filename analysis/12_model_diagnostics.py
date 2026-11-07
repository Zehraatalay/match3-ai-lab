from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "analysis/data/stage2_ml_dataset.csv"

TABLE_DIR = ROOT / "analysis/tables"
FIGURE_DIR = ROOT / "analysis/figures"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

BOARD_FEATURES = [
    "color_entropy",
    "dominant_color_ratio",
    "legal_moves_count",
]

CATEGORICAL_FEATURES = [
    "variant_id",
    "agent_name",
]


def make_model(include_board_features):
    numeric = BOARD_FEATURES if include_board_features else []

    transformers = []

    if numeric:
        transformers.append(("numeric", StandardScaler(), numeric))

    transformers.append(
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        )
    )

    return Pipeline(
        [
            (
                "preprocessor",
                ColumnTransformer(transformers),
            ),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def split_data(df):
    boards = df[["variant_id", "seed"]].drop_duplicates()

    train_parts = []
    test_parts = []

    for _, group in boards.groupby("variant_id"):
        train_board, test_board = train_test_split(
            group,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

        train_parts.append(train_board)
        test_parts.append(test_board)

    train_boards = pd.concat(train_parts)
    test_boards = pd.concat(test_parts)

    train_keys = set(zip(train_boards["variant_id"], train_boards["seed"]))

    test_keys = set(zip(test_boards["variant_id"], test_boards["seed"]))

    df = df.copy()

    df["board_key"] = list(zip(df["variant_id"], df["seed"]))

    train_df = df[df["board_key"].isin(train_keys)].copy()

    test_df = df[df["board_key"].isin(test_keys)].copy()

    assert len(train_df) == 960
    assert len(test_df) == 240

    return train_df, test_df


def reliability_table(y_true, probabilities, n_bins=8):
    table = pd.DataFrame(
        {
            "win": np.asarray(y_true),
            "probability": probabilities,
        }
    )

    table["bin"] = pd.cut(
        table["probability"],
        bins=np.linspace(0, 1, n_bins + 1),
        include_lowest=True,
    )

    return (
        table.groupby("bin", observed=True)
        .agg(
            count=("win", "size"),
            mean_predicted=("probability", "mean"),
            observed_win_rate=("win", "mean"),
        )
        .reset_index()
    )


def main():
    df = pd.read_csv(DATA_PATH)

    df = df[~df["variant_id"].str.endswith("__original")].copy()

    train_df, test_df = split_data(df)

    models = {
        "Level_Agent": {
            "include_board": False,
            "features": CATEGORICAL_FEATURES,
        },
        "Level_Agent_Board": {
            "include_board": True,
            "features": BOARD_FEATURES + CATEGORICAL_FEATURES,
        },
    }

    prediction_table = test_df[
        [
            "variant_id",
            "seed",
            "agent_name",
            "win",
            "legal_moves_count",
            "color_entropy",
            "dominant_color_ratio",
        ]
    ].copy()

    level_results = []
    reliability_results = []
    coefficient_results = []

    for model_name, specification in models.items():
        features = specification["features"]

        model = make_model(specification["include_board"])

        model.fit(
            train_df[features],
            train_df["win"],
        )

        probabilities = model.predict_proba(test_df[features])[:, 1]

        prediction_table[model_name] = probabilities

        # ------------------------------------------
        # Level-level diagnostics
        # ------------------------------------------

        for variant_id, group in prediction_table.groupby("variant_id"):
            y = group["win"]
            p = group[model_name]

            level_results.append(
                {
                    "model": model_name,
                    "variant_id": variant_id,
                    "episodes": len(group),
                    "win_rate": y.mean(),
                    "roc_auc": (roc_auc_score(y, p) if y.nunique() == 2 else np.nan),
                    "brier_score": brier_score_loss(y, p),
                    "log_loss": log_loss(
                        y,
                        p,
                        labels=[0, 1],
                    ),
                }
            )

        # ------------------------------------------
        # Reliability table
        # ------------------------------------------

        reliability = reliability_table(
            test_df["win"],
            probabilities,
        )

        reliability["model"] = model_name

        reliability_results.append(reliability)

        # ------------------------------------------
        # Coefficients
        # ------------------------------------------

        names = model.named_steps["preprocessor"].get_feature_names_out()

        coefficients = model.named_steps["classifier"].coef_[0]

        for feature, coefficient in zip(
            names,
            coefficients,
        ):
            coefficient_results.append(
                {
                    "model": model_name,
                    "feature": feature,
                    "coefficient": coefficient,
                }
            )

    # ==================================================
    # SAVE TABLES
    # ==================================================

    level_df = pd.DataFrame(level_results)

    reliability_df = pd.concat(
        reliability_results,
        ignore_index=True,
    )

    coefficients_df = pd.DataFrame(coefficient_results)

    level_df.to_csv(
        TABLE_DIR / "stage2_model_level_diagnostics.csv",
        index=False,
    )

    reliability_df.to_csv(
        TABLE_DIR / "stage2_model_reliability.csv",
        index=False,
    )

    coefficients_df.to_csv(
        TABLE_DIR / "stage2_model_coefficients.csv",
        index=False,
    )

    prediction_table.to_csv(
        TABLE_DIR / "stage2_diagnostic_predictions.csv",
        index=False,
    )

    # ==================================================
    # HIGH-CONFIDENCE ERRORS
    # ==================================================

    errors = prediction_table.copy()

    errors["predicted_win"] = (errors["Level_Agent_Board"] >= 0.5).astype(int)

    errors = errors[errors["predicted_win"] != errors["win"]].copy()

    errors["confidence"] = np.maximum(
        errors["Level_Agent_Board"],
        1 - errors["Level_Agent_Board"],
    )

    errors = errors.sort_values(
        "confidence",
        ascending=False,
    )

    errors.to_csv(
        TABLE_DIR / "stage2_high_confidence_errors.csv",
        index=False,
    )

    # ==================================================
    # RELIABILITY DIAGRAM
    # ==================================================

    fig, ax = plt.subplots(figsize=(7, 6))

    for model_name, group in reliability_df.groupby("model"):
        ax.plot(
            group["mean_predicted"],
            group["observed_win_rate"],
            marker="o",
            label=model_name,
        )

    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        label="Perfect calibration",
    )

    ax.set_xlabel("Mean Predicted Win Probability")
    ax.set_ylabel("Observed Win Rate")
    ax.set_title("Reliability Diagram")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()

    fig.tight_layout()

    fig.savefig(
        FIGURE_DIR / "stage2_reliability_diagram.png",
        dpi=160,
    )

    plt.close(fig)

    # ==================================================
    # PRINT RESULTS
    # ==================================================

    print("\n========== LEVEL DIAGNOSTICS ==========")

    print(level_df.round(4).to_string(index=False))

    print("\n========== MODEL COEFFICIENTS ==========")

    print(coefficients_df.round(4).to_string(index=False))

    print("\n========== HIGH-CONFIDENCE ERRORS ==========")

    print(errors.head(10).round(4).to_string(index=False))

    print("\nModel diagnostics completed.")


if __name__ == "__main__":
    main()
