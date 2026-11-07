from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

# ==================================================
# CONFIGURATION
# ==================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_PATH = PROJECT_ROOT / "analysis" / "data" / "stage2_episodes_flat.csv"

TABLE_DIR = PROJECT_ROOT / "analysis" / "tables"
REPORT_DIR = PROJECT_ROOT / "analysis" / "reports"

TABLE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

ALPHA = 0.05
BOOTSTRAP_ITERATIONS = 10_000
RANDOM_SEED = 42

LEVELS = {
    "wooden-trouble__colors5-double-moves6": "Wooden Trouble v2",
    "double-trouble__colors5-moves6": "Double Trouble v2",
    "frozen-garden__colors5-moves10": "Frozen Garden v2",
}


# ==================================================
# 1. LOAD DATA
# ==================================================

df = pd.read_csv(DATA_PATH)

df["win"] = df["outcome"].eq("WON").astype(int)

df = df[df["variant_id"].isin(LEVELS)].copy()

if df.empty:
    raise ValueError("No calibrated-level episodes found.")

if df["outcome"].isna().any():
    raise ValueError("Missing episode outcomes.")

if not df["outcome"].isin(["WON", "LOST"]).all():
    raise ValueError("Unexpected episode outcomes found.")


# ==================================================
# 2. PAIRED BOOTSTRAP
# ==================================================


def paired_bootstrap_ci(
    greedy_wins,
    random_wins,
    iterations=BOOTSTRAP_ITERATIONS,
    seed=RANDOM_SEED,
):
    rng = np.random.default_rng(seed)

    differences = np.asarray(greedy_wins, dtype=float) - np.asarray(
        random_wins, dtype=float
    )

    n = len(differences)

    sample_indices = rng.integers(
        low=0,
        high=n,
        size=(iterations, n),
    )

    bootstrap_differences = differences[sample_indices].mean(axis=1)

    lower, upper = np.percentile(
        bootstrap_differences,
        [2.5, 97.5],
    )

    return float(lower), float(upper)


# ==================================================
# 3. MATCH EPISODES AND RUN TESTS
# ==================================================

results = []

for variant_id, level_name in LEVELS.items():
    level_df = df[df["variant_id"] == variant_id].copy()

    if level_df.duplicated(["seed", "agent_name"]).any():
        raise ValueError(f"Duplicate seed-agent combinations: {level_name}")

    expected_agents = {"Random", "Greedy"}

    if set(level_df["agent_name"]) != expected_agents:
        raise ValueError(f"Unexpected agent set for {level_name}")

    random_df = level_df[level_df["agent_name"] == "Random"].set_index("seed")

    greedy_df = level_df[level_df["agent_name"] == "Greedy"].set_index("seed")

    if set(random_df.index) != set(greedy_df.index):
        raise ValueError(f"Unmatched seeds for {level_name}")

    paired = greedy_df[
        [
            "win",
            "initial_fingerprint",
            "initial_board_seed",
            "initial_engine_seed",
        ]
    ].join(
        random_df[
            [
                "win",
                "initial_fingerprint",
                "initial_board_seed",
                "initial_engine_seed",
            ]
        ],
        suffixes=("_greedy", "_random"),
        validate="one_to_one",
    )

    for field in [
        "initial_fingerprint",
        "initial_board_seed",
        "initial_engine_seed",
    ]:
        if not (paired[f"{field}_greedy"] == paired[f"{field}_random"]).all():
            raise ValueError(f"Initial identity mismatch: {level_name}, {field}")

    if len(paired) != 200:
        raise ValueError(f"Expected 200 pairs, found {len(paired)}: {level_name}")

    greedy = paired["win_greedy"]
    random = paired["win_random"]

    both_win = int(((greedy == 1) & (random == 1)).sum())
    greedy_only = int(((greedy == 1) & (random == 0)).sum())
    random_only = int(((greedy == 0) & (random == 1)).sum())
    both_lose = int(((greedy == 0) & (random == 0)).sum())

    discordant = greedy_only + random_only

    # Exact two-sided McNemar test.
    # Conditional on discordant pairs, the count
    # of Greedy-only wins follows Binomial(n, 0.5)
    # under the null hypothesis.
    p_value = (
        binomtest(
            greedy_only,
            n=discordant,
            p=0.5,
            alternative="two-sided",
        ).pvalue
        if discordant > 0
        else 1.0
    )

    greedy_rate = greedy.mean()
    random_rate = random.mean()

    difference = greedy_rate - random_rate

    ci_lower, ci_upper = paired_bootstrap_ci(
        greedy.to_numpy(),
        random.to_numpy(),
    )

    results.append(
        {
            "level": level_name,
            "pairs": len(paired),
            "both_win": both_win,
            "greedy_only_win": greedy_only,
            "random_only_win": random_only,
            "both_lose": both_lose,
            "greedy_win_rate": greedy_rate,
            "random_win_rate": random_rate,
            "difference_pp": difference * 100,
            "ci_lower_pp": ci_lower * 100,
            "ci_upper_pp": ci_upper * 100,
            "mcnemar_p_value": p_value,
        }
    )


# ==================================================
# 4. HOLM MULTIPLE TESTING CORRECTION
# ==================================================

results_df = pd.DataFrame(results)

# Holm step-down adjustment.
sorted_indices = np.argsort(results_df["mcnemar_p_value"].to_numpy())

sorted_p = results_df["mcnemar_p_value"].to_numpy()[sorted_indices]

m = len(sorted_p)

adjusted_sorted = np.maximum.accumulate(sorted_p * (m - np.arange(m)))

adjusted_sorted = np.clip(adjusted_sorted, 0, 1)

adjusted = np.empty(m)
adjusted[sorted_indices] = adjusted_sorted

results_df["holm_adjusted_p_value"] = adjusted

results_df["significant_after_holm"] = results_df["holm_adjusted_p_value"] < ALPHA


# ==================================================
# 5. SAVE RESULTS
# ==================================================

csv_path = TABLE_DIR / "stage2_mcnemar_results.csv"

results_df.to_csv(csv_path, index=False)

print("\n========== STATISTICAL TEST RESULTS ==========")

print(
    results_df[
        [
            "level",
            "pairs",
            "greedy_only_win",
            "random_only_win",
            "difference_pp",
            "ci_lower_pp",
            "ci_upper_pp",
            "mcnemar_p_value",
            "holm_adjusted_p_value",
            "significant_after_holm",
        ]
    ].to_string(index=False)
)


# ==================================================
# 6. GENERATE MARKDOWN REPORT
# ==================================================

report_lines = [
    "# Stage 2: Paired Agent Comparison",
    "",
    "## Method",
    "",
    "- Dataset: Stage 2 calibration confirmation.",
    "- Agents: Greedy and Random.",
    "- Sample size: 200 matched boards per level.",
    "- Test: exact two-sided McNemar.",
    "- Multiple comparisons: Holm correction.",
    "- Effect size: Greedy minus Random win rate.",
    "- Confidence intervals: paired percentile bootstrap (95%).",
    "- Bootstrap resamples: 10,000.",
    "- Evaluation: development data, not independent final test.",
    "",
    "## Results",
    "",
    (
        "| Level | Difference (pp) | 95% bootstrap CI (pp) | "
        "Exact p | Holm-adjusted p | Significant? |"
    ),
    "|---|---:|---:|---:|---:|---|",
]

for row in results_df.itertuples():
    report_lines.append(
        f"| {row.level} "
        f"| {row.difference_pp:+.1f} "
        f"| [{row.ci_lower_pp:+.1f}, {row.ci_upper_pp:+.1f}] "
        f"| {row.mcnemar_p_value:.4f} "
        f"| {row.holm_adjusted_p_value:.4f} "
        f"| {'Yes' if row.significant_after_holm else 'No'} |"
    )

report_lines += [
    "",
    "## Interpretation notes",
    "",
    "- A non-significant result does not establish equivalence.",
    "- Confidence intervals quantify uncertainty in observed differences.",
    "- The paired bootstrap resamples boards, not individual agents.",
    (
        "- Percentile bootstrap CIs are approximate and may not align "
        "exactly with exact-test decisions."
    ),
    "- Comparisons are within each fixed level configuration.",
    "- These results do not establish causal effects of gameplay mechanics.",
]

report_path = REPORT_DIR / "stage2_statistical_analysis.md"

report_path.write_text(
    "\n".join(report_lines) + "\n",
    encoding="utf-8",
)

print("\nSaved:")
print(csv_path)
print(report_path)
