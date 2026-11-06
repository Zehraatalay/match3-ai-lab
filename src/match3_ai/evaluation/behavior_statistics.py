"""Paired and environment-seed-clustered descriptive behavior statistics."""

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np

from match3_ai.evaluation.analysis import exact_mcnemar, wilson_interval


def paired_difference(left, right, *, seed=51000, samples=5000):
    if type(samples) is not int or samples <= 0:
        raise ValueError("samples must be a positive integer")
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if left.shape != right.shape or left.ndim != 1 or not left.size:
        raise ValueError("paired vectors must have identical nonempty shapes")
    values = left - right
    if not np.isfinite(values).all():
        raise ValueError("paired values must be finite")
    rng = np.random.default_rng(seed)
    means = np.concatenate(
        [
            values[
                rng.integers(len(values), size=(min(250, samples - start), len(values)))
            ].mean(axis=1)
            for start in range(0, samples, 250)
        ]
    )
    reference = float(right.mean())
    return {
        "n": len(values),
        "left_mean": float(left.mean()),
        "right_mean": reference,
        "absolute_difference": float(values.mean()),
        "relative_change": float(values.mean() / reference) if reference > 0 else None,
        "ci95": np.quantile(means, [0.025, 0.975]).tolist(),
    }


def paired_outcomes(left, right, *, seed=51000, samples=5000):
    if set(left) != set(right):
        raise ValueError("paired outcomes must cover identical seeds")
    table = Counter()
    differences = []
    excluded = []
    for key in sorted(left):
        if left[key] not in ("WON", "LOST") or right[key] not in ("WON", "LOST"):
            excluded.append(key)
            continue
        a, b = left[key] == "WON", right[key] == "WON"
        table[(a, b)] += 1
        differences.append((int(a), int(b)))
    if not differences:
        raise ValueError("no completed outcome pairs")
    counts = {
        "both_win": table[(True, True)],
        "left_only_win": table[(True, False)],
        "right_only_win": table[(False, True)],
        "both_lose": table[(False, False)],
    }
    a, b = np.asarray(differences).T
    return {
        "counts": counts,
        "percentages": {key: value / len(differences) for key, value in counts.items()},
        "excluded_seeds": excluded,
        "difference": paired_difference(a, b, seed=seed, samples=samples),
        "mcnemar_exact_p": exact_mcnemar(
            counts["left_only_win"], counts["right_only_win"]
        ),
    }


def holm_adjust(p_values):
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values))
    previous = 0.0
    for index, position in enumerate(order):
        previous = max(
            previous, min(1.0, float(values[position]) * (len(values) - index))
        )
        adjusted[position] = previous
    return adjusted.tolist()


def expected_random_agreement(action_mask):
    count = int(np.count_nonzero(action_mask))
    if not count:
        raise ValueError("random agreement is undefined for empty masks")
    return 1 / count


def action_preferences(rows, action_masks):
    """Descriptive source-balanced action marginals versus exact masked uniform."""
    if not rows:
        raise ValueError("preferences require observed decisions")
    policies = sorted({row["source_policy"] for row in rows})
    families = {name: ("DQN" if name.startswith("DQN") else name) for name in policies}
    n_families = len(set(families.values()))
    counts = Counter(row["source_policy"] for row in rows)
    names = sorted(rows[0]["actions"])
    actual = {name: np.zeros(144) for name in names}
    expected = np.zeros(144)
    expected_tie_optimal = 0.0
    for row in rows:
        source = row["source_policy"]
        n_policies = sum(families[name] == families[source] for name in policies)
        weight = 1 / (n_families * n_policies * counts[source])
        mask = np.asarray(action_masks[row["state_id"]], dtype=bool)
        k = int(mask.sum())
        if not k:
            raise ValueError("stored action mask cannot be empty")
        expected += weight * mask / k
        expected_tie_optimal += weight * row["greedy_best_ties"] / k
        for name, action in row["actions"].items():
            if not mask[action]:
                raise ValueError("observed deterministic action is masked")
            actual[name][action] += weight

    def entropy(probabilities):
        p = probabilities[probabilities > 0]
        return float(-(p * np.log2(p)).sum())

    return {
        "expected_uniform_action_probabilities": expected.tolist(),
        "expected_uniform_marginal_entropy_bits": entropy(expected),
        "expected_random_tie_optimal": float(expected_tie_optimal),
        "policies": {
            name: {
                "action_probabilities": p.tolist(),
                "total_variation_from_uniform_mask_marginal": float(
                    np.abs(p - expected).sum() / 2
                ),
                "marginal_entropy_bits": entropy(p),
                "top_actions": [
                    {
                        "action_id": int(action),
                        "observed_probability": float(p[action]),
                        "expected_uniform_probability": float(expected[action]),
                    }
                    for action in np.argsort(-p, kind="stable")[:5]
                ],
            }
            for name, p in actual.items()
        },
        "interpretation": "descriptive empirical marginals on identical stored observations; nonuniform preferences do not imply useful learning or improved return",
    }


def export_preferences(input_dir, output_file):
    """Recompute descriptive preferences from frozen data without any environment."""
    root, destination = Path(input_dir), Path(output_file)
    if destination.exists():
        raise FileExistsError(destination)
    paths = (root / "state_occurrences.jsonl", root / "observations.npz")
    hashes = [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]
    rows = [json.loads(line) for line in paths[0].read_text().splitlines()]
    groups = {
        "Random-generated": [r for r in rows if r["source_policy"] == "RandomAgent"],
        "Greedy-generated": [r for r in rows if r["source_policy"] == "GreedyAgent"],
        "DQN-generated": [r for r in rows if r["source_policy"].startswith("DQN")],
        "Pooled": rows,
    }
    with np.load(paths[1], allow_pickle=False) as data:
        results = {
            name: action_preferences(values, data["action_mask"])
            for name, values in groups.items()
            if values
        }
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x") as handle:
        json.dump(results, handle, indent=2, allow_nan=False)
    if hashes != [hashlib.sha256(path.read_bytes()).hexdigest() for path in paths]:
        raise RuntimeError("preference input artifacts changed")
    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Describe frozen-data action preferences"
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    export_preferences(args.input_dir, args.output_file)


def clustered_agreement(rows, key, *, seed=51000, samples=5000):
    if not rows:
        return None
    clusters = sorted({row["environment_seed"] for row in rows})
    policies = sorted({row["source_policy"] for row in rows})
    index = {value: i for i, value in enumerate(clusters)}
    sums = np.zeros((len(policies), len(clusters)))
    counts = np.zeros_like(sums)
    for p, policy in enumerate(policies):
        for row in rows:
            if row["source_policy"] == policy:
                i = index[row["environment_seed"]]
                sums[p, i] += row[key]
                counts[p, i] += 1

    def macro(numerators, denominators):
        ratios = np.divide(
            numerators,
            denominators,
            out=np.full_like(numerators, np.nan),
            where=denominators > 0,
        )
        families = []
        for names in (
            ("RandomAgent",),
            ("GreedyAgent",),
            tuple(name for name in policies if name.startswith("DQN")),
        ):
            positions = [policies.index(name) for name in names if name in policies]
            if positions:
                families.append(ratios[positions].mean(axis=0))
        return np.mean(families, axis=0)

    point = float(macro(sums.sum(axis=1), counts.sum(axis=1)))
    rng = np.random.default_rng(seed)
    weights = rng.multinomial(
        len(clusters), np.full(len(clusters), 1 / len(clusters)), size=samples
    )
    bootstrap = macro(sums @ weights.T, counts @ weights.T)
    valid = bootstrap[np.isfinite(bootstrap)]
    return {
        "rate": point,
        "ci95": np.quantile(valid, [0.025, 0.975]).tolist() if len(valid) else None,
        "n_occurrences": len(rows),
        "n_seed_clusters": len(clusters),
        "valid_bootstrap_draws": len(valid),
        "weighting": "equal source families; equal DQN source policies within family; occurrence-weighted within policy",
    }


def reward_distribution(episodes, moves):
    grouped = {}
    for move in moves:
        grouped.setdefault(move["episode_id"], []).append(move)
    rewards = [row["reward"] for row in moves if row["reward"] is not None]
    if not rewards or any(value not in (-1.0, 0.0, 1.0) for value in rewards):
        raise ValueError("requires observed sparse rewards")
    before_terminal = []
    for episode in episodes:
        rows = sorted(grouped[episode["episode_id"]], key=lambda row: row["move_index"])
        if (
            len(rows) != episode["steps"]
            or sum(row["reward"] for row in rows) != episode["total_reward"]
        ):
            raise ValueError("episode and move totals differ")
        earlier = rows[:-1]
        if any(row["reward"] != 0 for row in earlier) or rows[-1]["reward"] != (
            1 if episode["outcome"] == "WON" else -1
        ):
            raise ValueError("sparse terminal reward contract violated")
        objectives = earlier[-1]["objective_progress"] if earlier else None
        completion = (
            np.mean([min(item["progress"] / item["target"], 1) for item in objectives])
            if objectives
            else 0.0
        )
        before_terminal.append(
            {
                "episode_id": episode["episode_id"],
                "steps": len(rows),
                "objective_completion_before_terminal": float(completion),
                "blocker_hits_before_terminal": sum(
                    row["blocker_damage"]["hits"] for row in earlier
                ),
            }
        )
    return {
        "transitions": len(rewards),
        "fractions": {
            str(value): sum(reward == value for reward in rewards) / len(rewards)
            for value in (-1, 0, 1)
        },
        "episode_lengths": [item["steps"] for item in before_terminal],
        "before_terminal": before_terminal,
        "win_rate": sum(e["outcome"] == "WON" for e in episodes) / len(episodes),
        "win_ci95": wilson_interval(
            sum(e["outcome"] == "WON" for e in episodes), len(episodes)
        ),
    }
