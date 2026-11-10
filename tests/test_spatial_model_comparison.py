"""Split, identity, grouping, and preprocessing checks for training-only CV."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "spatial_model_comparison", ROOT / "analysis/16_spatial_model_comparison.py"
)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


@pytest.fixture
def datasets():
    rows, spatial = [], []
    order = np.random.default_rng(9).permutation(200)
    for variant in ("wooden__calibrated", "double__calibrated", "frozen__calibrated"):
        for seed in order:
            identity = {
                "variant_id": variant,
                "seed": seed,
                "initial_fingerprint": f"{variant}:{seed}",
            }
            spatial.append(
                {
                    **identity,
                    **{
                        name: seed % (i + 3)
                        for i, name in enumerate(comparison.SPATIAL)
                    },
                }
            )
            for agent, win in (("Random", seed % 2), ("Greedy", (seed // 2) % 2)):
                rows.append(
                    {
                        **identity,
                        "agent_name": agent,
                        "win": win,
                        **{
                            name: float(seed % (i + 4))
                            for i, name in enumerate(comparison.BOARD)
                        },
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(spatial)


def test_merge_preserves_original_order_and_source_data(datasets):
    episodes, spatial = datasets
    before = episodes.copy(deep=True)
    merged = comparison.validate_merge(
        episodes, spatial.sample(frac=1, random_state=42)
    )
    pd.testing.assert_frame_equal(merged[episodes.columns], episodes)
    pd.testing.assert_frame_equal(episodes, before)


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate_board",
        "duplicate_agent",
        "unmatched",
        "fingerprint",
        "paired_fingerprint",
        "missing",
        "wrong_agent",
        "paired_feature",
    ],
)
def test_merge_rejects_invalid_identities(datasets, problem):
    episodes, spatial = datasets
    if problem == "duplicate_board":
        spatial.iloc[1] = spatial.iloc[0]
    elif problem == "duplicate_agent":
        episodes.iloc[1] = episodes.iloc[0]
    elif problem == "unmatched":
        spatial.loc[0, "seed"] = 999
    elif problem == "fingerprint":
        spatial.loc[0, "initial_fingerprint"] = "wrong"
    elif problem == "paired_fingerprint":
        episodes.loc[0, "initial_fingerprint"] = "wrong"
    elif problem == "missing":
        spatial.loc[0, "initial_fingerprint"] = None
    elif problem == "wrong_agent":
        episodes.loc[0, "agent_name"] = "Other"
    elif problem == "paired_feature":
        episodes.loc[0, "color_entropy"] = 999
    with pytest.raises(ValueError):
        comparison.validate_merge(episodes, spatial)


def test_historical_split_exactly_matches_original_algorithm(datasets):
    episodes, spatial = datasets
    merged = comparison.validate_merge(episodes, spatial)
    expected_train, expected_holdout = [], []
    for _, group in episodes[comparison.KEYS].drop_duplicates().groupby("variant_id"):
        train, holdout = train_test_split(group, test_size=0.20, random_state=42)
        expected_train.append(train)
        expected_holdout.append(holdout)
    training, holdout = comparison.historical_split(merged)
    assert comparison.board_keys(training) == comparison.board_keys(
        pd.concat(expected_train)
    )
    assert comparison.board_keys(holdout) == comparison.board_keys(
        pd.concat(expected_holdout)
    )
    assert len(training) == 960 and len(holdout) == 240
    assert list(holdout.columns) == comparison.EPISODE_KEYS
    assert comparison.board_keys(training).isdisjoint(comparison.board_keys(holdout))
    changed = merged.copy()
    holdout_keys = comparison.board_keys(holdout)
    mask = [
        key in holdout_keys
        for key in changed[comparison.KEYS].itertuples(index=False, name=None)
    ]
    changed.loc[mask, "win"] = 1 - changed.loc[mask, "win"]
    changed_train, changed_ids = comparison.historical_split(changed)
    pd.testing.assert_frame_equal(changed_train, training)
    pd.testing.assert_frame_equal(changed_ids, holdout)


def test_real_historical_holdout_identity_reproduction():
    episodes = pd.read_csv(ROOT / "analysis/data/stage2_ml_dataset.csv")
    spatial = pd.read_csv(ROOT / "tests/fixtures/spatial/stage2_spatial_features.csv")
    training, holdout = comparison.historical_split(
        comparison.validate_merge(episodes, spatial)
    )
    comparison.check_historical_holdout(
        holdout, ROOT / "tests/fixtures/spatial/stage2_model_predictions.csv"
    )
    assert len(comparison.board_keys(training)) == 480


def test_historical_artifact_reads_identity_columns_only(
    datasets, tmp_path, monkeypatch
):
    episodes, _ = datasets
    _, holdout = comparison.historical_split(episodes)
    path = tmp_path / "historical.csv"
    holdout.assign(win="DO NOT USE", probability="DO NOT USE").to_csv(path, index=False)
    original = pd.read_csv
    calls = []

    def read(*args, **kwargs):
        calls.append(kwargs)
        return original(*args, **kwargs)

    monkeypatch.setattr(comparison.pd, "read_csv", read)
    comparison.check_historical_holdout(holdout, path)
    assert calls == [{"usecols": comparison.EPISODE_KEYS}]


def test_balanced_grouping_is_deterministic_and_without_leakage(datasets):
    episodes, spatial = datasets
    training, holdout = comparison.historical_split(
        comparison.validate_merge(episodes, spatial)
    )
    folds = comparison.make_board_folds(training)
    pd.testing.assert_frame_equal(folds, comparison.make_board_folds(training))
    assert folds.groupby(["fold", "variant_id"]).size().eq(32).all()
    expanded = training.merge(folds, on=comparison.KEYS, validate="many_to_one")
    assert expanded.groupby(comparison.KEYS).fold.nunique().eq(1).all()
    for fold in range(1, 6):
        fit = comparison.board_keys(expanded.loc[expanded.fold.ne(fold)])
        val = comparison.board_keys(expanded.loc[expanded.fold.eq(fold)])
        assert len(fit) == 384 and len(val) == 96
        assert fit.isdisjoint(val)
        assert (fit | val).isdisjoint(comparison.board_keys(holdout))


def test_cv_preprocessing_fits_only_training_folds_and_oof_is_complete(
    datasets, monkeypatch
):
    episodes, spatial = datasets
    training, _ = comparison.historical_split(
        comparison.validate_merge(episodes, spatial)
    )
    folds = comparison.make_board_folds(training)
    data = training.merge(folds, on=comparison.KEYS, validate="many_to_one", sort=False)
    original = comparison.make_pipeline
    fits = []

    def make(numeric):
        pipeline = original(numeric)
        fit = pipeline.fit

        def checked_fit(x, y):
            fold = len(fits) // len(comparison.FEATURE_SETS) + 1
            assert set(x.index) == set(data.index[data.fold.ne(fold)])
            result = fit(x, y)
            if numeric:
                scaler = pipeline.named_steps["preprocessor"].named_transformers_[
                    "numeric"
                ]
                np.testing.assert_allclose(scaler.mean_, x[numeric].mean().to_numpy())
            fits.append(fold)
            return result

        pipeline.fit = checked_fit
        return pipeline

    monkeypatch.setattr(comparison, "make_pipeline", make)
    metrics, oof = comparison.cross_validate(training, folds)
    assert len(fits) == 20 and len(metrics) == 20
    assert len(oof) == 3840
    assert not oof.duplicated(["experiment", *comparison.EPISODE_KEYS]).any()
    assert oof.groupby("experiment").size().eq(960).all()
    assert oof.win_probability.between(0, 1).all()
    assert metrics[comparison.METRICS].notna().all().all()
    _, consistency = comparison.paired_differences(metrics)
    assert (
        (
            consistency.improved_folds
            + consistency.tied_folds
            + consistency.worsened_folds
        )
        .eq(5)
        .all()
    )


def test_invalid_fold_identities_rejected(datasets):
    episodes, spatial = datasets
    training, _ = comparison.historical_split(
        comparison.validate_merge(episodes, spatial)
    )
    folds = comparison.make_board_folds(training)
    folds.loc[0, "seed"] = 999
    with pytest.raises(ValueError, match="Fold assignments"):
        comparison.cross_validate(training, folds)
