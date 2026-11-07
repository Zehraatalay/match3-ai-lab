"""Descriptive spatial-feature validation on Stage 2 development observations.

Board analyses use one row per unique board. Episode associations condition on
both level and agent. Spearman coefficients use average ranks for ties; no
significance tests are performed. Undefined coefficients remain missing with an
explicit reason. Source datasets and their duplicate columns remain intact.
"""

from itertools import combinations
from pathlib import Path
from textwrap import fill

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "analysis/data"
TABLE_DIR = ROOT / "analysis/tables"
FIGURE_DIR = ROOT / "analysis/figures"
KEYS = ["variant_id", "seed"]
FEATURES = [
    "ordinary_damage_move_count",
    "ordinary_damage_move_ratio",
    "mean_ordinary_hp_loss",
    "max_ordinary_hp_loss",
    "immediate_damage_move_count",
    "immediate_damage_move_ratio",
    "mean_immediate_hp_loss",
    "max_immediate_hp_loss",
    "max_immediate_blockers_destroyed",
]
OUTCOME_GROUPS = ["Both Lose", "Random Only", "Greedy Only", "Both Win"]
HIGH_CORRELATION = 0.8


def validate_datasets(
    spatial: pd.DataFrame, episodes: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Validate board identities before expanding matched features to episodes."""
    for name, frame, required in (
        ("spatial", spatial, [*KEYS, "initial_fingerprint", *FEATURES]),
        ("episodes", episodes, [*KEYS, "agent_name", "win", "outcome"]),
    ):
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"Missing {name} columns: {sorted(missing)}")
        if frame["variant_id"].isna().any():
            raise ValueError(f"Missing {name} variant identities")
    spatial = spatial.loc[~spatial.variant_id.str.endswith("__original")].copy()
    episodes = episodes.loc[~episodes.variant_id.str.endswith("__original")].copy()
    if spatial.isna().any().any() or episodes.isna().any().any():
        raise ValueError("Missing values in calibrated datasets")
    if len(spatial) != 600 or spatial.duplicated(KEYS).any():
        raise ValueError("Expected exactly 600 unique spatial boards")
    if len(episodes) != 1200:
        raise ValueError("Expected exactly 1200 calibrated episodes")
    if episodes.duplicated([*KEYS, "agent_name"]).any():
        raise ValueError("Duplicate board-agent combinations")
    pairs = episodes.groupby(KEYS).agent_name.agg(set)
    if len(pairs) != 600 or any(value != {"Random", "Greedy"} for value in pairs):
        raise ValueError("Expected Random and Greedy exactly once per board")
    counts = spatial.groupby("variant_id").size()
    if len(counts) != 3 or not counts.eq(200).all():
        raise ValueError("Expected three calibrated variants with 200 boards each")
    if not episodes.win.isin([0, 1]).all():
        raise ValueError("Win outcomes must be binary")
    if not episodes.win.eq(episodes.outcome.eq("WON").astype(int)).all():
        raise ValueError("Win values disagree with episode outcomes")
    numeric = spatial[FEATURES].to_numpy(dtype=float)
    if not np.isfinite(numeric).all():
        raise ValueError("Spatial features must be finite numeric values")
    for frame in (spatial, episodes):
        seeds = frame.seed.to_numpy(dtype=float)
        if not np.isfinite(seeds).all() or not np.equal(seeds, np.floor(seeds)).all():
            raise ValueError("Board seeds must be finite integers")

    identity_columns = [*KEYS]
    if "initial_fingerprint" in episodes:
        if episodes.groupby(KEYS).initial_fingerprint.nunique().ne(1).any():
            raise ValueError("Paired episode fingerprints disagree")
        identity_columns.append("initial_fingerprint")
    identities = episodes[identity_columns].drop_duplicates(KEYS)
    matched = spatial.merge(
        identities,
        on=KEYS,
        how="outer",
        validate="one_to_one",
        suffixes=("", "_episode"),
        indicator=True,
    )
    if not matched["_merge"].eq("both").all():
        raise ValueError("Board identity sets differ between datasets")
    if "initial_fingerprint_episode" in matched:
        if not matched.initial_fingerprint.eq(
            matched.initial_fingerprint_episode
        ).all():
            raise ValueError("Initial fingerprints disagree between datasets")
        matched = matched.drop(columns="initial_fingerprint_episode")
    boards = matched.drop(columns="_merge")
    outcomes = episodes.pivot(
        index=KEYS, columns="agent_name", values="win"
    ).reset_index()
    boards = boards.merge(outcomes, on=KEYS, validate="one_to_one")
    boards["wins_out_of_two"] = boards.Random + boards.Greedy
    boards["difficulty_group"] = np.select(
        [
            boards.Random.eq(1) & boards.Greedy.eq(1),
            boards.Random.eq(1) & boards.Greedy.eq(0),
            boards.Random.eq(0) & boards.Greedy.eq(1),
        ],
        ["Both Win", "Random Only", "Greedy Only"],
        default="Both Lose",
    )
    expanded = episodes[[*KEYS, "agent_name", "win"]].merge(
        boards[[*KEYS, *FEATURES]], on=KEYS, validate="many_to_one"
    )
    return boards, expanded


def duplicate_features(
    boards: pd.DataFrame, features: list[str] = FEATURES
) -> tuple[list[str], list[list[str]]]:
    """Group exactly equal values, retaining the first feature in each group."""
    groups: list[list[str]] = []
    for feature in features:
        for group in groups:
            if np.array_equal(boards[feature].to_numpy(), boards[group[0]].to_numpy()):
                group.append(feature)
                break
        else:
            groups.append([feature])
    return [group[0] for group in groups], [group for group in groups if len(group) > 1]


def variability_table(boards: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for level, group in boards.groupby("variant_id"):
        for feature in features:
            values = group[feature]
            rows.append(
                {
                    "variant_id": level,
                    "feature": feature,
                    "boards": len(group),
                    "unique_values": values.nunique(),
                    "mean": values.mean(),
                    "std": values.std(),
                    "min": values.min(),
                    "max": values.max(),
                    "zero_values": values.eq(0).sum(),
                    "constant": values.nunique() <= 1,
                }
            )
    return pd.DataFrame(rows)


def outcome_group_table(boards: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for level, level_boards in boards.groupby("variant_id"):
        for outcome in OUTCOME_GROUPS:
            group = level_boards.loc[level_boards.difficulty_group.eq(outcome)]
            for feature in features:
                values = group[feature]
                rows.append(
                    {
                        "variant_id": level,
                        "difficulty_group": outcome,
                        "feature": feature,
                        "boards": len(group),
                        "mean": values.mean(),
                        "std": values.std(),
                        "min": values.min(),
                        "max": values.max(),
                        "median": values.median(),
                        "q25": values.quantile(0.25),
                        "q75": values.quantile(0.75),
                        "status": "empty_group" if group.empty else "observed",
                    }
                )
    return pd.DataFrame(rows)


def spearman(left: pd.Series, right: pd.Series) -> tuple[float, str]:
    """Compute rank correlation without significance testing or a SciPy dependency."""
    if len(left) < 2:
        return float("nan"), "insufficient_observations"
    constant_left, constant_right = left.nunique() <= 1, right.nunique() <= 1
    if constant_left or constant_right:
        reason = (
            "constant_both"
            if constant_left and constant_right
            else ("constant_feature" if constant_left else "constant_target")
        )
        return float("nan"), reason
    return float(
        np.corrcoef(left.rank(method="average"), right.rank(method="average"))[0, 1]
    ), "defined"


def association_table(
    boards: pd.DataFrame, episodes: pd.DataFrame, features: list[str]
) -> pd.DataFrame:
    rows = []
    for level, group in boards.groupby("variant_id"):
        subsets = [("board", "paired", "wins_out_of_two", group)]
        for agent, agent_group in episodes.loc[episodes.variant_id.eq(level)].groupby(
            "agent_name"
        ):
            subsets.append(("episode", agent, "win", agent_group))
        for scope, agent, target, subset in subsets:
            for feature in features:
                rho, status = spearman(subset[feature], subset[target])
                rows.append(
                    {
                        "variant_id": level,
                        "scope": scope,
                        "agent_name": agent,
                        "target": target,
                        "feature": feature,
                        "n": len(subset),
                        "spearman_rho": rho,
                        "status": status,
                    }
                )
    return pd.DataFrame(rows)


def feature_correlations(boards: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    rows = []
    for scope, group in [
        ("pooled_calibrated", boards),
        *list(boards.groupby("variant_id")),
    ]:
        for left, right in combinations(features, 2):
            rho, status = spearman(group[left], group[right])
            rows.append(
                {
                    "scope": scope,
                    "feature_a": left,
                    "feature_b": right,
                    "n": len(group),
                    "spearman_rho": rho,
                    "status": status,
                    "high_correlation": bool(abs(rho) >= HIGH_CORRELATION),
                }
            )
    return pd.DataFrame(rows)


def plot_correlations(boards: pd.DataFrame, features: list[str], path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 13), layout="constrained")
    labels = [fill(feature.replace("_", " "), width=20) for feature in features]
    for ax, (scope, group) in zip(
        axes.flat,
        [("Pooled calibrated boards", boards), *list(boards.groupby("variant_id"))],
        strict=True,
    ):
        matrix = np.full((len(features), len(features)), np.nan)
        for i, left in enumerate(features):
            for j, right in enumerate(features):
                matrix[i, j] = spearman(group[left], group[right])[0]
        palette = plt.get_cmap("coolwarm").copy()
        palette.set_bad("lightgray")
        image = ax.imshow(np.ma.masked_invalid(matrix), cmap=palette, vmin=-1, vmax=1)
        ax.set_xticks(range(len(features)), labels, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(range(len(features)), labels, fontsize=8)
        title = scope.replace("__", "\n")
        ax.set_title(f"{title}\nn={len(group)} boards", fontsize=9)
        for i in range(len(features)):
            for j in range(len(features)):
                ax.text(
                    j,
                    i,
                    "NA" if np.isnan(matrix[i, j]) else f"{matrix[i, j]:.2f}",
                    ha="center",
                    va="center",
                    fontsize=7,
                )
        fig.colorbar(image, ax=ax, shrink=0.65)
    fig.suptitle("Exploratory Spearman correlations; gray/NA = undefined")
    fig.savefig(path, dpi=160)
    plt.close(fig)


def plot_outcomes(boards: pd.DataFrame, features: list[str], path: Path) -> None:
    levels = list(boards.groupby("variant_id"))
    fig, axes = plt.subplots(
        len(features), 3, figsize=(17, 3 * len(features)), squeeze=False
    )
    for col, (level, level_boards) in enumerate(levels):
        for row, feature in enumerate(features):
            ax = axes[row, col]
            labels = []
            for position, outcome in enumerate(OUTCOME_GROUPS, start=1):
                values = level_boards.loc[
                    level_boards.difficulty_group.eq(outcome), feature
                ]
                labels.append(f"{outcome}\nn={len(values)}")
                if not values.empty:
                    ax.boxplot([values], positions=[position], widths=0.55)
            ax.set_xticks(range(1, 5), labels, fontsize=7)
            ax.set_xlim(0.5, 4.5)
            ax.set_ylabel(feature, fontsize=8)
            if row == 0:
                ax.set_title(level, fontsize=9)
    fig.suptitle(
        "Initial damage features by board outcomes; exploratory development analysis"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def main() -> None:
    boards, episodes = validate_datasets(
        pd.read_csv(DATA_DIR / "stage2_spatial_features.csv"),
        pd.read_csv(DATA_DIR / "stage2_ml_dataset.csv"),
    )
    features, duplicates = duplicate_features(boards)
    variability = variability_table(boards, features)
    associations = association_table(boards, episodes, features)
    correlations = feature_correlations(boards, features)
    duplicate_rows = [
        {"retained_feature": group[0], "excluded_duplicate": name}
        for group in duplicates
        for name in group[1:]
    ]
    tables = {
        "duplicates": pd.DataFrame(
            duplicate_rows, columns=["retained_feature", "excluded_duplicate"]
        ),
        "variability": variability,
        "outcome_groups": outcome_group_table(boards, features),
        "associations": associations,
        "feature_correlations": correlations,
    }
    paths = {
        name: TABLE_DIR / f"stage2_spatial_validation_{name}.csv" for name in tables
    }
    heatmap_path = FIGURE_DIR / "stage2_spatial_validation_correlations.png"
    outcome_path = FIGURE_DIR / "stage2_spatial_validation_outcome_groups.png"
    for path in [*paths.values(), heatmap_path, outcome_path]:
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        with paths[name].open("x") as handle:
            frame.to_csv(handle, index=False)
    plot_correlations(boards, features, heatmap_path)
    plot_outcomes(boards, features, outcome_path)
    print(
        "Validated: 600 unique boards, 200 per level; 1200 episodes; Random/Greedy paired"
    )
    print("No missing values, duplicate identities, or fingerprint mismatches")
    print(f"Analysis features: {features}\nExact duplicate groups: {duplicates}")
    print("\nWithin-level variability:\n" + variability.to_string(index=False))
    print(
        "\nConstant features:\n"
        + variability.loc[variability.constant].to_string(index=False)
    )
    undefined = associations.loc[associations.status.ne("defined")]
    print("\nUndefined outcome correlations:\n" + undefined.to_string(index=False))
    print("\nStrongest exploratory association per level and scope/agent:")
    defined = associations.loc[associations.status.eq("defined")].copy()
    defined["absolute_rho"] = defined.spearman_rho.abs()
    strongest = (
        defined.sort_values("absolute_rho", ascending=False)
        .groupby(["variant_id", "scope", "agent_name"])
        .head(1)
    )
    print(strongest.drop(columns="absolute_rho").to_string(index=False))
    print(f"\nHighly correlated feature pairs (absolute rho >= {HIGH_CORRELATION}):")
    print(correlations.loc[correlations.high_correlation].to_string(index=False))
    print(
        "\nUndefined feature correlations:\n"
        + correlations.loc[correlations.status.ne("defined")].to_string(index=False)
    )
    print(
        "\nDescriptive development associations only; no significance tests or causal claims"
    )


if __name__ == "__main__":
    main()
