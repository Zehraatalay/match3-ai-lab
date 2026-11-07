from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import (
    StratifiedGroupKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    OneHotEncoder,
    StandardScaler,
)

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "analysis/data/stage2_ml_dataset.csv"

OUTPUT_DIR = ROOT / "analysis/tables"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

FEATURE_SETS = {
    "Level + Agent": {
        "numeric": [],
        "categorical": ["variant_id", "agent_name"],
    },
    "Board Only": {
        "numeric": [
            "color_entropy",
            "dominant_color_ratio",
            "legal_moves_count",
        ],
        "categorical": [],
    },
    "Level + Agent + Board": {
        "numeric": [
            "color_entropy",
            "dominant_color_ratio",
            "legal_moves_count",
        ],
        "categorical": ["variant_id", "agent_name"],
    },
}


def make_pipeline(numeric, categorical):
    transformers = []

    if numeric:
        transformers.append(
            (
                "numeric",
                StandardScaler(),
                numeric,
            )
        )

    if categorical:
        transformers.append(
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical,
            )
        )

    preprocessor = ColumnTransformer(transformers)

    return Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


def calculate_metrics(y_true, probabilities):
    return {
        "roc_auc": roc_auc_score(
            y_true,
            probabilities,
        ),
        "average_precision": average_precision_score(
            y_true,
            probabilities,
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities,
        ),
        "log_loss": log_loss(
            y_true,
            probabilities,
            labels=[0, 1],
        ),
    }


def main():
    df = pd.read_csv(DATA_PATH)

    df = df[~df["variant_id"].str.endswith("__original")].copy()

    df["board_key"] = df["variant_id"].astype(str) + "::" + df["seed"].astype(str)

    # Reproduce the previous board-level split.
    board_table = df[["variant_id", "seed", "board_key"]].drop_duplicates()

    train_keys = []
    test_keys = []

    for _, group in board_table.groupby("variant_id"):
        train_group, test_group = train_test_split(
            group,
            test_size=0.20,
            random_state=RANDOM_STATE,
        )

        train_keys.extend(train_group["board_key"])
        test_keys.extend(test_group["board_key"])

    train_df = df[df["board_key"].isin(train_keys)].copy()

    test_df = df[df["board_key"].isin(test_keys)].copy()

    if len(train_df) != 960 or len(test_df) != 240:
        raise ValueError("Unexpected train/test sizes.")

    if set(train_df["board_key"]) & set(test_df["board_key"]):
        raise ValueError("Board leakage detected.")

    y_train = train_df["win"]
    y_test = test_df["win"]

    groups = train_df["board_key"]

    cv = StratifiedGroupKFold(
        n_splits=5,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    cv_results = []
    test_results = []

    for experiment_name, feature_spec in FEATURE_SETS.items():
        numeric = feature_spec["numeric"]
        categorical = feature_spec["categorical"]

        features = numeric + categorical

        print(f"\nEvaluating: {experiment_name}")

        # ------------------------------------------
        # Grouped cross-validation
        # ------------------------------------------

        for fold, (train_idx, val_idx) in enumerate(
            cv.split(
                train_df[features],
                y_train,
                groups,
            ),
            start=1,
        ):
            fold_train = train_df.iloc[train_idx]
            fold_val = train_df.iloc[val_idx]

            model = make_pipeline(numeric, categorical)

            model.fit(
                fold_train[features],
                fold_train["win"],
            )

            probabilities = model.predict_proba(fold_val[features])[:, 1]

            metrics = calculate_metrics(
                fold_val["win"],
                probabilities,
            )

            cv_results.append(
                {
                    "experiment": experiment_name,
                    "fold": fold,
                    **metrics,
                }
            )

        # ------------------------------------------
        # Existing development holdout
        # ------------------------------------------

        model = make_pipeline(numeric, categorical)

        model.fit(
            train_df[features],
            y_train,
        )

        probabilities = model.predict_proba(test_df[features])[:, 1]

        metrics = calculate_metrics(
            y_test,
            probabilities,
        )

        test_results.append(
            {
                "experiment": experiment_name,
                **metrics,
            }
        )

    cv_df = pd.DataFrame(cv_results)
    test_df_results = pd.DataFrame(test_results)

    cv_summary = (
        cv_df.groupby("experiment")
        .agg(
            roc_auc_mean=("roc_auc", "mean"),
            roc_auc_std=("roc_auc", "std"),
            ap_mean=("average_precision", "mean"),
            brier_mean=("brier_score", "mean"),
            log_loss_mean=("log_loss", "mean"),
        )
        .reset_index()
    )

    cv_df.to_csv(
        OUTPUT_DIR / "stage2_ablation_cv_folds.csv",
        index=False,
    )

    cv_summary.to_csv(
        OUTPUT_DIR / "stage2_ablation_cv_summary.csv",
        index=False,
    )

    test_df_results.to_csv(
        OUTPUT_DIR / "stage2_ablation_holdout.csv",
        index=False,
    )

    print("\n========== CROSS-VALIDATION SUMMARY ==========")
    print(cv_summary.round(4).to_string(index=False))

    print("\n========== DEVELOPMENT HOLDOUT ==========")
    print(test_df_results.round(4).to_string(index=False))

    print("\nAblation study completed.")


if __name__ == "__main__":
    main()
