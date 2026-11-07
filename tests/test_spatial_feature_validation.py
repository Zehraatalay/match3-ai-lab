"""Integrity and descriptive-analysis checks without gameplay or model fitting."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PATH = Path(__file__).resolve().parents[1] / "analysis/15_spatial_feature_validation.py"
SPEC = importlib.util.spec_from_file_location("spatial_feature_validation", PATH)
validation = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validation
SPEC.loader.exec_module(validation)


@pytest.fixture
def datasets():
    rows, episodes = [], []
    for level in ("a__calibrated", "b__calibrated", "c__calibrated"):
        for seed in range(200):
            identity = {
                "variant_id": level,
                "seed": seed,
                "initial_fingerprint": f"{level}:{seed}",
            }
            rows.append(
                {
                    **identity,
                    **{
                        feature: seed % (i + 2)
                        for i, feature in enumerate(validation.FEATURES)
                    },
                }
            )
            for agent, win in (("Random", seed % 2), ("Greedy", (seed // 2) % 2)):
                episodes.append(
                    {
                        **identity,
                        "agent_name": agent,
                        "win": win,
                        "outcome": "WON" if win else "LOST",
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(episodes)


def test_valid_merge_uses_unique_boards_and_separate_agents(datasets):
    spatial, episodes = datasets
    spatial_before, episodes_before = spatial.copy(deep=True), episodes.copy(deep=True)
    boards, expanded = validation.validate_datasets(spatial, episodes)
    assert len(boards) == 600
    assert len(expanded) == 1200
    assert boards.groupby("difficulty_group").size().to_dict() == dict.fromkeys(
        validation.OUTCOME_GROUPS, 150
    )
    associations = validation.association_table(
        boards, expanded, [validation.FEATURES[0]]
    )
    assert len(associations) == 9
    assert associations.n.eq(200).all()
    assert set(associations.loc[associations.scope.eq("episode"), "agent_name"]) == {
        "Random",
        "Greedy",
    }
    pd.testing.assert_frame_equal(spatial, spatial_before)
    pd.testing.assert_frame_equal(episodes, episodes_before)


@pytest.mark.parametrize(
    "problem",
    [
        "duplicate_board",
        "duplicate_agent",
        "missing_board",
        "missing_episode",
        "wrong_agent",
        "missing_value",
        "paired_fingerprint",
        "cross_fingerprint",
        "board_keys",
        "binary_win",
        "outcome",
        "infinite_feature",
        "level_counts",
    ],
)
def test_integrity_errors_fail_loudly(datasets, problem):
    spatial, episodes = datasets
    if problem == "duplicate_board":
        spatial.iloc[1] = spatial.iloc[0]
    elif problem == "duplicate_agent":
        episodes.iloc[1] = episodes.iloc[0]
    elif problem == "missing_board":
        spatial = spatial.iloc[:-1]
    elif problem == "missing_episode":
        episodes = episodes.iloc[:-1]
    elif problem == "wrong_agent":
        episodes.loc[0, "agent_name"] = "Other"
    elif problem == "missing_value":
        spatial.loc[0, "initial_fingerprint"] = None
    elif problem == "paired_fingerprint":
        episodes.loc[0, "initial_fingerprint"] = "wrong"
    elif problem == "cross_fingerprint":
        spatial.loc[0, "initial_fingerprint"] = "wrong"
    elif problem == "board_keys":
        spatial.loc[0, "seed"] = 999
    elif problem == "binary_win":
        episodes.loc[0, "win"] = 2
    elif problem == "outcome":
        episodes.loc[0, "outcome"] = "WON"
    elif problem == "infinite_feature":
        spatial[validation.FEATURES[0]] = spatial[validation.FEATURES[0]].astype(float)
        spatial.loc[0, validation.FEATURES[0]] = np.inf
    elif problem == "level_counts":
        spatial.loc[0, "variant_id"] = "extra__calibrated"
    with pytest.raises(ValueError):
        validation.validate_datasets(spatial, episodes)


def test_original_variants_are_excluded_before_counts(datasets):
    spatial, episodes = datasets
    original_board = spatial.iloc[[0]].assign(variant_id="a__original")
    original_episodes = episodes.iloc[:2].assign(variant_id="a__original")
    boards, expanded = validation.validate_datasets(
        pd.concat([spatial, original_board]), pd.concat([episodes, original_episodes])
    )
    assert len(boards) == 600 and len(expanded) == 1200


def test_optional_episode_fingerprint(datasets):
    spatial, episodes = datasets
    boards, _ = validation.validate_datasets(
        spatial, episodes.drop(columns="initial_fingerprint")
    )
    assert boards.initial_fingerprint.equals(spatial.initial_fingerprint)


def test_duplicate_detection_is_exact_and_preserves_source():
    frame = pd.DataFrame(
        {
            "ordinary": [0.0, 1.0, 2.0],
            "immediate": [0.0, 1.0, 2.0],
            "third": [0, 1, 2],
            "near": [0.0, 1.0, 2.000001],
            "scaled": [0, 2, 4],
        }
    )
    before = frame.copy(deep=True)
    retained, groups = validation.duplicate_features(frame, list(frame.columns))
    assert retained == ["ordinary", "near", "scaled"]
    assert groups == [["ordinary", "immediate", "third"]]
    pd.testing.assert_frame_equal(frame, before)


def test_variability_reports_constant_and_zero_values(datasets):
    spatial, episodes = datasets
    boards, _ = validation.validate_datasets(spatial, episodes)
    boards[validation.FEATURES[0]] = 0
    table = validation.variability_table(boards, [validation.FEATURES[0]])
    assert table.constant.all()
    assert table.zero_values.eq(200).all()
    assert table.unique_values.eq(1).all()
    assert table["std"].eq(0).all()


@pytest.mark.parametrize(
    "left,right,status",
    [
        ([1, 1, 1], [0, 1, 0], "constant_feature"),
        ([1, 2, 3], [0, 0, 0], "constant_target"),
        ([1, 1, 1], [0, 0, 0], "constant_both"),
        ([1], [0], "insufficient_observations"),
    ],
)
def test_undefined_correlations_are_explicit(left, right, status):
    rho, actual = validation.spearman(pd.Series(left), pd.Series(right))
    assert np.isnan(rho) and actual == status


def test_spearman_average_ranks_with_ties():
    rho, status = validation.spearman(pd.Series([1, 1, 2, 3]), pd.Series([4, 4, 2, 1]))
    assert rho == pytest.approx(-1.0)
    assert status == "defined"


def test_empty_outcome_groups_are_explicit(datasets):
    spatial, episodes = datasets
    episodes["win"], episodes["outcome"] = 0, "LOST"
    boards, expanded = validation.validate_datasets(spatial, episodes)
    groups = validation.outcome_group_table(boards, [validation.FEATURES[0]])
    assert (
        groups.loc[groups.difficulty_group.ne("Both Lose"), "status"]
        .eq("empty_group")
        .all()
    )
    associations = validation.association_table(
        boards, expanded, [validation.FEATURES[0]]
    )
    assert associations.spearman_rho.isna().all()
    assert associations.status.eq("constant_target").all()
