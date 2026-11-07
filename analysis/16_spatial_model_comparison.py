"""Training-only grouped CV of fixed Logistic Regression feature sets.

The original ML CSV ordering determines the historical development split.
Holdout identities are checked, but holdout outcomes are never scored. Earlier
exploratory analyses inspected all 600 boards, so this is a development study,
not an independent confirmation. No tuning or significance tests are performed.
"""

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis/data"
OUTPUT_DIR = ROOT / "analysis/tables"
KEYS = ["variant_id", "seed"]
EPISODE_KEYS = [*KEYS, "agent_name"]
CATEGORICAL = ["variant_id", "agent_name"]
BOARD = ["color_entropy", "dominant_color_ratio", "legal_moves_count"]
SPATIAL = [
    "ordinary_damage_move_count",
    "ordinary_damage_move_ratio",
    "mean_ordinary_hp_loss",
    "max_ordinary_hp_loss",
    "max_immediate_blockers_destroyed",
]
FEATURE_SETS = {
    "Level + Agent": [],
    "Level + Agent + Board": BOARD,
    "Level + Agent + Spatial": SPATIAL,
    "Level + Agent + Board + Spatial": [*BOARD, *SPATIAL],
}
METRICS = ["log_loss", "brier_score", "roc_auc", "average_precision"]
BASELINE = "Level + Agent"
COMBINED = "Level + Agent + Board + Spatial"
RANDOM_STATE = 42


def board_keys(frame: pd.DataFrame) -> set[tuple]:
    return set(frame[KEYS].itertuples(index=False, name=None))


def validate_merge(episodes: pd.DataFrame, spatial: pd.DataFrame) -> pd.DataFrame:
    """Check board identities one-to-one, then merge without changing episode order."""
    for frame, required in (
        (episodes, [*EPISODE_KEYS, "win", "initial_fingerprint", *BOARD]),
        (spatial, [*KEYS, "initial_fingerprint", *SPATIAL]),
    ):
        if set(required) - set(frame.columns):
            raise ValueError("Required dataset columns are missing")
        if frame.variant_id.isna().any():
            raise ValueError("Missing variant identity")
    episodes = episodes.loc[~episodes.variant_id.str.endswith("__original")].copy()
    spatial = spatial.loc[~spatial.variant_id.str.endswith("__original")].copy()
    if episodes.isna().any().any() or spatial.isna().any().any():
        raise ValueError("Missing values in calibrated data")
    if len(episodes) != 1200 or len(spatial) != 600:
        raise ValueError("Expected 1200 episodes and 600 spatial boards")
    if episodes.duplicated(EPISODE_KEYS).any() or spatial.duplicated(KEYS).any():
        raise ValueError("Duplicate board or board-agent identity")
    pairs = episodes.groupby(KEYS).agent_name.agg(set)
    if len(pairs) != 600 or any(pair != {"Random", "Greedy"} for pair in pairs):
        raise ValueError("Expected exactly two paired agents per board")
    if not episodes.win.isin([0, 1]).all():
        raise ValueError("Win must be binary")
    for frame, features in ((episodes, BOARD), (spatial, SPATIAL)):
        if not np.isfinite(frame[features].to_numpy(dtype=float)).all():
            raise ValueError("Nonfinite model features")
        seeds = frame.seed.to_numpy(dtype=float)
        if not np.isfinite(seeds).all() or not (seeds == np.floor(seeds)).all():
            raise ValueError("Seeds must be finite integers")
    counts = spatial.groupby("variant_id").size()
    if len(counts) != 3 or not counts.eq(200).all():
        raise ValueError("Expected three variants with 200 boards each")
    if episodes.groupby(KEYS).initial_fingerprint.nunique().ne(1).any():
        raise ValueError("Paired fingerprints disagree")
    if episodes.groupby(KEYS)[BOARD].nunique().ne(1).any().any():
        raise ValueError("Paired initial board features disagree")
    identities = episodes[[*KEYS, "initial_fingerprint"]].drop_duplicates(KEYS)
    checked = identities.merge(
        spatial[[*KEYS, "initial_fingerprint"]],
        on=KEYS,
        how="outer",
        validate="one_to_one",
        indicator=True,
        suffixes=("_ml", "_spatial"),
    )
    if not checked["_merge"].eq("both").all():
        raise ValueError("Unmatched board identities")
    if not checked.initial_fingerprint_ml.eq(checked.initial_fingerprint_spatial).all():
        raise ValueError("Dataset fingerprints disagree")
    merged = episodes.merge(
        spatial[[*KEYS, *SPATIAL]],
        on=KEYS,
        how="left",
        sort=False,
        validate="many_to_one",
    )
    if list(merged[EPISODE_KEYS].itertuples(index=False, name=None)) != list(
        episodes[EPISODE_KEYS].itertuples(index=False, name=None)
    ):
        raise RuntimeError("Merge changed original episode ordering")
    return merged


def historical_split(episodes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce script 10; return training episodes and holdout identities only."""
    boards = episodes[KEYS].drop_duplicates()
    train_parts, holdout_parts = [], []
    for _, group in boards.groupby("variant_id"):
        train, holdout = train_test_split(
            group, test_size=0.20, random_state=RANDOM_STATE
        )
        train_parts.append(train)
        holdout_parts.append(holdout)
    train_keys = board_keys(pd.concat(train_parts))
    holdout_keys = board_keys(pd.concat(holdout_parts))
    if len(train_keys) != 480 or len(holdout_keys) != 120 or train_keys & holdout_keys:
        raise ValueError("Unexpected historical split or board leakage")
    mask = [
        key in train_keys for key in episodes[KEYS].itertuples(index=False, name=None)
    ]
    training = episodes.loc[mask].copy().reset_index(drop=True)
    holdout = (
        episodes.loc[np.logical_not(mask), EPISODE_KEYS].copy().reset_index(drop=True)
    )
    if len(training) != 960 or len(holdout) != 240:
        raise ValueError("Unexpected historical episode counts")
    return training, holdout


def check_historical_holdout(holdout: pd.DataFrame, path: Path) -> None:
    """Read historical identity columns only, never probabilities or outcomes."""
    recorded = pd.read_csv(path, usecols=EPISODE_KEYS)
    if recorded.duplicated(EPISODE_KEYS).any():
        raise ValueError("Duplicate historical holdout identities")
    if list(recorded.itertuples(index=False, name=None)) != list(
        holdout[EPISODE_KEYS].itertuples(index=False, name=None)
    ):
        raise ValueError("Historical development-holdout ordering/identities differ")


def make_board_folds(training: pd.DataFrame) -> pd.DataFrame:
    boards = training[KEYS].drop_duplicates().reset_index(drop=True)
    counts = boards.groupby("variant_id").size()
    if len(boards) != 480 or len(counts) != 3 or not counts.eq(160).all():
        raise ValueError("Expected 160 training boards per variant")
    folds = np.zeros(len(boards), dtype=int)
    splitter = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for fold, (train_idx, val_idx) in enumerate(
        splitter.split(boards[KEYS], boards.variant_id), start=1
    ):
        if board_keys(boards.iloc[train_idx]) & board_keys(boards.iloc[val_idx]):
            raise RuntimeError("Cross-validation board leakage")
        folds[val_idx] = fold
    boards["fold"] = folds
    if not boards.groupby(["fold", "variant_id"]).size().eq(32).all():
        raise RuntimeError("Unbalanced cross-validation level coverage")
    return boards


def make_pipeline(numeric: list[str]) -> Pipeline:
    transformers = [
        ("categorical", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)
    ]
    if numeric:
        transformers.insert(0, ("numeric", StandardScaler(), numeric))
    return Pipeline(
        [
            ("preprocessor", ColumnTransformer(transformers)),
            (
                "classifier",
                LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
            ),
        ]
    )


def cross_validate(
    training: pd.DataFrame, folds: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if folds.duplicated(KEYS).any() or board_keys(training) != board_keys(folds):
        raise ValueError("Fold assignments do not match training boards")
    data = training.merge(folds, on=KEYS, validate="many_to_one", sort=False)
    if (
        set(data.fold) != set(range(1, 6))
        or data.groupby(KEYS).fold.nunique().ne(1).any()
    ):
        raise ValueError("Invalid grouped fold assignment")
    metrics, predictions = [], []
    for fold in range(1, 6):
        fit = data.loc[data.fold.ne(fold)]
        validation = data.loc[data.fold.eq(fold)]
        if board_keys(fit) & board_keys(validation):
            raise RuntimeError("Training/validation board overlap")
        if fit.win.nunique() != 2 or validation.win.nunique() != 2:
            raise ValueError("Both outcome classes are required for these CV metrics")
        for experiment, numeric in FEATURE_SETS.items():
            features = [*CATEGORICAL, *numeric]
            model = make_pipeline(numeric)
            model.fit(fit[features], fit.win)
            probabilities = model.predict_proba(validation[features])[:, 1]
            metrics.append(
                {
                    "experiment": experiment,
                    "fold": fold,
                    "train_boards": len(board_keys(fit)),
                    "validation_boards": len(board_keys(validation)),
                    "train_episodes": len(fit),
                    "validation_episodes": len(validation),
                    "log_loss": log_loss(validation.win, probabilities, labels=[0, 1]),
                    "brier_score": brier_score_loss(validation.win, probabilities),
                    "roc_auc": roc_auc_score(validation.win, probabilities),
                    "average_precision": average_precision_score(
                        validation.win, probabilities
                    ),
                }
            )
            predicted = validation[
                [*EPISODE_KEYS, "initial_fingerprint", "win", "fold"]
            ].copy()
            predicted["experiment"] = experiment
            predicted["win_probability"] = probabilities
            predictions.append(predicted)
    oof = pd.concat(predictions, ignore_index=True)
    if len(oof) != 3840 or oof.duplicated(["experiment", *EPISODE_KEYS]).any():
        raise RuntimeError("Incomplete or duplicate out-of-fold predictions")
    return pd.DataFrame(metrics), oof


def paired_differences(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    comparisons = [(name, BASELINE) for name in FEATURE_SETS if name != BASELINE]
    comparisons.append((COMBINED, "Level + Agent + Board"))
    for experiment, reference in comparisons:
        paired = metrics.loc[metrics.experiment.eq(experiment)].merge(
            metrics.loc[metrics.experiment.eq(reference)],
            on="fold",
            validate="one_to_one",
            suffixes=("", "_reference"),
        )
        for row in paired.to_dict("records"):
            for metric in METRICS:
                delta = row[metric] - row[f"{metric}_reference"]
                improvement = -delta if metric in ("log_loss", "brier_score") else delta
                rows.append(
                    {
                        "experiment": experiment,
                        "reference": reference,
                        "fold": row["fold"],
                        "metric": metric,
                        "delta_model_minus_reference": delta,
                        "improvement": improvement,
                    }
                )
    differences = pd.DataFrame(rows)
    consistency = (
        differences.groupby(["experiment", "reference", "metric"], sort=False)
        .improvement.agg(
            mean_improvement="mean",
            std_improvement="std",
            improved_folds=lambda values: (values > 0).sum(),
            tied_folds=lambda values: (values == 0).sum(),
            worsened_folds=lambda values: (values < 0).sum(),
        )
        .reset_index()
    )
    return differences, consistency


def main() -> None:
    names = [
        "fold_metrics",
        "summary",
        "oof_predictions",
        "board_folds",
        "paired_differences",
        "consistency",
        "manifest",
    ]
    paths = {
        name: OUTPUT_DIR
        / f"stage2_spatial_model_{name}.{'json' if name == 'manifest' else 'csv'}"
        for name in names
    }
    for path in paths.values():
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing results: {path}")
    ml_path = DATA_DIR / "stage2_ml_dataset.csv"
    spatial_path = DATA_DIR / "stage2_spatial_features.csv"
    merged = validate_merge(pd.read_csv(ml_path), pd.read_csv(spatial_path))
    training, holdout = historical_split(merged)
    check_historical_holdout(holdout, OUTPUT_DIR / "stage2_model_predictions.csv")
    folds = make_board_folds(training)
    metrics, oof = cross_validate(training, folds)
    summary = metrics.groupby("experiment", sort=False)[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    differences, consistency = paired_differences(metrics)
    assignments = pd.concat(
        [
            folds.assign(split="training"),
            holdout[KEYS].drop_duplicates().assign(fold=0, split="development_holdout"),
        ],
        ignore_index=True,
    )
    manifest = {
        "evaluation": "training-only grouped development cross-validation",
        "random_state": RANDOM_STATE,
        "feature_sets": FEATURE_SETS,
        "categorical_features": CATEGORICAL,
        "logistic_regression": {
            "max_iter": 1000,
            "random_state": 42,
            "C": 1.0,
            "solver": "lbfgs",
        },
        "cv": "StratifiedKFold on unique boards, stratified by variant; 32 boards per variant per validation fold",
        "holdout_evaluated": False,
        "holdout_fold_code": 0,
        "historical_holdout_identities_verified": True,
        "input_sha256": {
            str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in (ml_path, spatial_path)
        },
        "versions": {
            "sklearn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "limitations": [
            "Earlier exploratory analyses inspected all 600 boards",
            "Fold SD is descriptive, not a confidence interval",
            "Paired episodes are grouped for splitting but contribute equally to prediction metrics",
            "Fixed feature sets and default regularization; no tuning or significance tests",
        ],
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in {
        "fold_metrics": metrics,
        "summary": summary.reset_index(),
        "oof_predictions": oof,
        "board_folds": assignments,
        "paired_differences": differences,
        "consistency": consistency,
    }.items():
        with paths[name].open("x") as handle:
            frame.to_csv(handle, index=False)
    with paths["manifest"].open("x") as handle:
        json.dump(manifest, handle, indent=2)
    print(
        "Historical split verified: 480 training boards / 960 episodes; 120 holdout boards / 240 episodes"
    )
    print(
        "Holdout outcomes not evaluated; 5 shared folds, each 96 boards / 192 episodes"
    )
    print("\nCV mean and sample SD:\n" + summary.to_string())
    print(
        "\nPaired fold improvements (positive = better):\n"
        + consistency.to_string(index=False)
    )
    print(
        "\nExploratory development comparison; no tuning, significance tests, or independent confirmation"
    )


if __name__ == "__main__":
    main()
