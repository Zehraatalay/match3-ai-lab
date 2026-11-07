from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ==================================================
# CONFIGURATION
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = ROOT / "analysis/data/stage2_ml_dataset.csv"

TABLE_DIR = ROOT / "analysis/tables"
TABLE_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42

NUMERIC_FEATURES = [
    "color_entropy",
    "dominant_color_ratio",
    "legal_moves_count",
]

CATEGORICAL_FEATURES = [
    "variant_id",
    "agent_name",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


# ==================================================
# 1. LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH)

df = df[~df["variant_id"].str.endswith("__original")].copy()

assert len(df) == 1200

if df[FEATURES + ["win"]].isna().any().any():
    raise ValueError("Missing model inputs or targets.")

if df.duplicated(["variant_id", "seed", "agent_name"]).any():
    raise ValueError("Duplicate episode records.")


# ==================================================
# 2. BOARD-LEVEL TRAIN / TEST SPLIT
# ==================================================

boards = df[["variant_id", "seed"]].drop_duplicates()

train_parts = []
test_parts = []

for variant_id, group in boards.groupby("variant_id"):
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

assert train_keys.isdisjoint(test_keys)

df["_board_key"] = list(zip(df["variant_id"], df["seed"]))

train_df = df[df["_board_key"].isin(train_keys)].copy()
test_df = df[df["_board_key"].isin(test_keys)].copy()

assert len(train_df) == 960
assert len(test_df) == 240

X_train = train_df[FEATURES]
X_test = test_df[FEATURES]

y_train = train_df["win"]
y_test = test_df["win"]


print("\n========== DATA SPLIT ==========")
print("Train episodes:", len(train_df))
print("Test episodes:", len(test_df))
print("Train win rate:", round(y_train.mean(), 3))
print("Test win rate:", round(y_test.mean(), 3))


# ==================================================
# 3. PREPROCESSING
# ==================================================

numeric_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

preprocessor = ColumnTransformer(
    [
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL_FEATURES,
        ),
    ]
)


# ==================================================
# 4. MODELS
# ==================================================

models = {
    "Logistic Regression": Pipeline(
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
    ),
    "Random Forest": Pipeline(
        [
            ("preprocessor", preprocessor),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=300,
                    min_samples_leaf=5,
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                ),
            ),
        ]
    ),
}


# ==================================================
# 5. HISTORICAL WIN RATE BASELINE
# ==================================================

baseline_rates = train_df.groupby(["variant_id", "agent_name"])["win"].mean()

global_rate = float(y_train.mean())

baseline_probs = np.array(
    [
        baseline_rates.get(
            (row.variant_id, row.agent_name),
            global_rate,
        )
        for row in test_df.itertuples()
    ]
)


# ==================================================
# 6. EVALUATION
# ==================================================

results = []
predictions = test_df[["variant_id", "seed", "agent_name", "win"]].copy()


def evaluate(name, probabilities):

    probabilities = np.asarray(probabilities)

    predicted = (probabilities >= 0.5).astype(int)

    results.append(
        {
            "model": name,
            "accuracy": accuracy_score(y_test, predicted),
            "roc_auc": roc_auc_score(y_test, probabilities),
            "pr_auc": average_precision_score(
                y_test,
                probabilities,
            ),
            "brier_score": brier_score_loss(
                y_test,
                probabilities,
            ),
            "log_loss": log_loss(
                y_test,
                probabilities,
                labels=[0, 1],
            ),
        }
    )

    predictions[name] = probabilities


evaluate("Historical Baseline", baseline_probs)


for name, model in models.items():
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]

    evaluate(name, probabilities)


# ==================================================
# 7. SAVE RESULTS
# ==================================================

results_df = pd.DataFrame(results)

results_df.to_csv(
    TABLE_DIR / "stage2_model_comparison.csv",
    index=False,
)

predictions.to_csv(
    TABLE_DIR / "stage2_model_predictions.csv",
    index=False,
)

print("\n========== MODEL COMPARISON ==========")

print(results_df.round(4).to_string(index=False))

print("\nSaved:")
print(TABLE_DIR / "stage2_model_comparison.csv")
print(TABLE_DIR / "stage2_model_predictions.csv")

print("\nPredictive modeling completed.")
