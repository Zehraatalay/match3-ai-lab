"""Preregistered board-paired analysis of frozen policies, without training."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from match3_ai.evaluation.analysis import exact_mcnemar
from match3_ai.evaluation.behavior_statistics import holm_adjust
from match3_ai.evaluation.fresh_protocol import load_protocol, sha256


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def paired_estimate(left, right, *, samples, seed, binary=False):
    left, right = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if (
        left.shape != right.shape
        or left.ndim != 1
        or not left.size
        or not np.isfinite(left - right).all()
    ):
        raise ValueError("paired outcomes must be aligned finite nonempty vectors")
    differences = left - right
    observed = float(differences.mean())
    rng = np.random.default_rng(seed)
    bootstrap = np.concatenate(
        [
            differences[
                rng.integers(len(left), size=(min(250, samples - start), len(left)))
            ].mean(axis=1)
            for start in range(0, samples, 250)
        ]
    )
    tolerance = 1e-12
    p_value = (
        1
        + int(
            np.count_nonzero(np.abs(bootstrap - observed) >= abs(observed) - tolerance)
        )
    ) / (samples + 1)
    result = {
        "n_boards": len(left),
        "left_mean": float(left.mean()),
        "right_mean": float(right.mean()),
        "difference": observed,
        "ci95": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "p_value": p_value,
        "test": "approximate two-sided null-centered board bootstrap",
    }
    if binary:
        if not np.isin(left, [0, 1]).all() or not np.isin(right, [0, 1]).all():
            raise ValueError("McNemar requires binary policy outcomes")
        counts = {
            "both_win": int(np.sum((left == 1) & (right == 1))),
            "left_only": int(np.sum((left == 1) & (right == 0))),
            "right_only": int(np.sum((left == 0) & (right == 1))),
            "both_lose": int(np.sum((left == 0) & (right == 0))),
        }
        result.update(
            counts=counts,
            p_value=exact_mcnemar(counts["left_only"], counts["right_only"]),
            test="exact two-sided McNemar",
        )
    return result


def critic_metrics(values, targets):
    values, targets = np.asarray(values, dtype=float), np.asarray(targets, dtype=float)
    if (
        values.shape != targets.shape
        or values.ndim != 1
        or not values.size
        or not np.isfinite(values - targets).all()
    ):
        raise ValueError("critic diagnostics require matching finite vectors")
    variance = float(np.var(targets))
    return {
        "n_boards": len(values),
        "value_prediction_mean": float(values.mean()),
        "return_target_variance": variance,
        "prediction_bias_V_minus_return": float((values - targets).mean()),
        "rmse": float(np.sqrt(np.mean((values - targets) ** 2))),
        "explained_variance": None
        if variance <= 1e-12
        else float(1 - np.var(targets - values) / variance),
    }


def comparisons(wins, settings):
    final = np.array([wins[f"PPO{seed}_final"] for seed in (5915, 5916, 5917)])
    initial = np.array([wins[f"PPO{seed}_initial"] for seed in (5915, 5916, 5917)])
    dqn = np.array([wins[f"DQN{seed}"] for seed in (5901, 5902, 5903)])
    arguments = {
        "samples": settings["bootstrap_samples"],
        "seed": settings["bootstrap_seed"],
    }
    primary = paired_estimate(final.mean(0), wins["Greedy"], **arguments)
    secondary = {
        "mean_final_minus_Random": paired_estimate(
            final.mean(0), wins["Random"], **arguments
        ),
        "mean_final_minus_mean_DQN": paired_estimate(
            final.mean(0), dqn.mean(0), **arguments
        ),
        "mean_final_minus_mean_initial": paired_estimate(
            final.mean(0), initial.mean(0), **arguments
        ),
    }
    for index, seed in enumerate((5915, 5916, 5917)):
        secondary[f"PPO{seed}_final_minus_initial"] = paired_estimate(
            final[index], initial[index], binary=True, **arguments
        )
    for index, seed in enumerate((5915, 5916, 5917)):
        secondary[f"PPO{seed}_final_minus_Greedy"] = paired_estimate(
            final[index], wins["Greedy"], binary=True, **arguments
        )
    for item, adjusted in zip(
        secondary.values(),
        holm_adjust([item["p_value"] for item in secondary.values()]),
    ):
        item["holm_p"] = adjusted
    return {
        "primary": primary,
        "secondary": secondary,
        "mean_final_PPO": float(final.mean()),
        "final_PPO_training_seed_sample_sd": float(final.mean(1).std(ddof=1)),
        "mean_initial_PPO": float(initial.mean()),
        "initial_PPO_training_seed_sample_sd": float(initial.mean(1).std(ddof=1)),
        "mean_DQN": float(dqn.mean()),
        "DQN_training_seed_sample_sd": float(dqn.mean(1).std(ddof=1)),
    }


def representative_seeds(wins, seeds, limit):
    arrays = np.array(list(wins.values()))
    ppo = (
        np.array([wins[f"PPO{seed}_final"] for seed in (5915, 5916, 5917)]).sum(0) >= 2
    )
    greedy = np.asarray(wins["Greedy"], dtype=bool)
    masks = {
        "PPO majority win / Greedy loss": ppo & ~greedy,
        "Greedy win / PPO majority loss": greedy & ~ppo,
        "all eleven win": arrays.all(0),
        "all eleven lose": ~arrays.any(0),
    }
    return {
        name: {
            "available": int(mask.sum()),
            "selected_seeds": [seed for seed, selected in zip(seeds, mask) if selected][
                :limit
            ],
        }
        for name, mask in masks.items()
    }


def create_figures(results, wins, critic_rows, output, statistics):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output.mkdir(parents=True, exist_ok=False)
    files = []

    def finish(fig, name):
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            fig.savefig(output / filename, dpi=180)
            files.append(filename)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 5), constrained_layout=True)
    names = list(results)
    rates = np.array([results[name]["win_rate"] for name in names])
    intervals = np.array([results[name]["win_rate_ci95"] for name in names]).T
    ax.bar(names, rates, color=[f"C{i % 10}" for i in range(len(names))])
    ax.errorbar(
        names,
        rates,
        yerr=np.vstack((rates - intervals[0], intervals[1] - rates)),
        fmt="none",
        color="black",
        capsize=3,
    )
    ax.set(ylabel="Fresh-board win rate, Wilson 95% CI", ylim=(0, 1))
    ax.tick_params(axis="x", labelrotation=40)
    finish(fig, "policy_win_rates")
    effects = {
        "PRIMARY mean PPO − Greedy": statistics["primary"],
        **statistics["secondary"],
    }
    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    for index, (name, item) in enumerate(effects.items()):
        ax.plot(np.array(item["ci95"]) * 100, [index, index], color=f"C{index % 10}")
        ax.plot(item["difference"] * 100, index, "o", color=f"C{index % 10}")
    ax.set_yticks(range(len(effects)), list(effects))
    ax.axvline(0, color="gray", linestyle="--")
    ax.set(xlabel="Paired win-rate difference, percentage points (pointwise 95% CI)")
    finish(fig, "paired_effects")
    fig, ax = plt.subplots(figsize=(7, 5), constrained_layout=True)
    for seed in (5915, 5916, 5917):
        ax.plot(
            [0, 1],
            [np.mean(wins[f"PPO{seed}_initial"]), np.mean(wins[f"PPO{seed}_final"])],
            "o-",
            label=str(seed),
        )
    ax.set_xticks([0, 1], ["Frozen initial", "Frozen 50k final"])
    ax.set(ylabel="Fresh-board win rate", ylim=(0, 1))
    ax.legend(title="PPO training seed")
    finish(fig, "initial_vs_final_PPO")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7), constrained_layout=True)
    for ax, name in zip(axes.ravel(), critic_rows):
        rows = critic_rows[name]
        ax.scatter(
            [r["target"] for r in rows], [r["value"] for r in rows], s=8, alpha=0.35
        )
        ax.plot([-1, 1], [-1, 1], color="gray", linestyle="--")
        ax.set(
            title=name,
            xlabel="Greedy-behavior discounted return",
            ylabel="Initial critic V",
        )
    fig.suptitle("Behavior-return agreement only; not on-policy critic accuracy")
    finish(fig, "critic_behavior_agreement")
    return files


def analyze(protocol_path, results_root, output_dir):
    protocol = load_protocol(protocol_path)
    root, output = Path(results_root), Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    digest = sha256(protocol_path)
    seeds = protocol["seeds"]
    results, episodes, initial, wins, sources = (
        {},
        {},
        {},
        {},
        {str(protocol_path): digest},
    )
    all_rows = []
    for specification in protocol["policies"]:
        name = specification["name"]
        directory = root / name
        paths = [
            directory / "result.json",
            directory / "evaluation/episodes.csv",
            directory / "evaluation/moves.csv",
            directory / "initial_states.json",
        ]
        sources.update({str(path): sha256(path) for path in paths})
        result = json.loads(paths[0].read_text())
        if result["protocol_sha256"] != digest or result["policy"] != specification:
            raise ValueError("result metadata differs from protocol")
        records = read_csv(paths[1])
        states = json.loads(paths[3].read_text())
        if (
            len(records) != len(seeds)
            or len(states) != len(seeds)
            or {int(r["seed"]) for r in records} != set(seeds)
        ):
            raise ValueError("incomplete or duplicate per-board outcomes")
        episodes[name] = {int(r["seed"]): r for r in records}
        initial[name] = {r["seed"]: r for r in states}
        if set(initial[name]) != set(seeds):
            raise ValueError("initial state alignment differs")
        if any(
            r["outcome"] not in ("WON", "LOST") or int(r["invalid_actions"])
            for r in records
        ):
            raise ValueError(
                "confirmatory analysis requires all planned valid outcomes"
            )
        wins[name] = np.array(
            [episodes[name][seed]["outcome"] == "WON" for seed in seeds], dtype=int
        )
        results[name] = result["summary"]
        all_rows.extend(records)
    for seed in seeds:
        identifiers = {
            (
                initial[name][seed]["fingerprint"],
                episodes[name][seed]["board_seed"],
                episodes[name][seed]["engine_seed"],
                episodes[name][seed]["level_config_id"],
            )
            for name in wins
        }
        if len(identifiers) != 1:
            raise ValueError(
                "policies did not receive identical initial boards/RNG seeds"
            )
    statistics = comparisons(wins, protocol["statistics"])
    if list(statistics["secondary"]) != protocol["secondaries"]:
        raise ValueError("secondary family differs from preregistration")
    representatives = representative_seeds(
        wins, seeds, protocol["representatives"]["per_category"]
    )
    critic_rows, critic = {}, {}
    for specification in protocol["policies"]:
        if specification["kind"] != "ppo":
            continue
        name = specification["name"]
        rows = [
            {
                "policy": name,
                "seed": seed,
                "value": initial[name][seed]["critic_value"],
                "target": 0.99 ** (int(episodes[name][seed]["steps"]) - 1)
                * float(episodes[name][seed]["total_reward"]),
            }
            for seed in seeds
        ]
        critic_rows[name] = rows
        critic[name] = critic_metrics(
            [r["value"] for r in rows], [r["target"] for r in rows]
        )
    output.mkdir(parents=True, exist_ok=False)

    def write_csv(name, rows):
        with (output / name).open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv("episodes.csv", all_rows)
    write_csv(
        "policy_summary.csv",
        [
            {
                "policy": name,
                "wins": r["wins"],
                "boards": r["attempted"],
                "win_rate": r["win_rate"],
                "wilson_low": r["win_rate_ci95"][0],
                "wilson_high": r["win_rate_ci95"][1],
                "mean_episode_reward": r["mean_total_reward"],
                "mean_episode_length": float(
                    np.mean([int(e["steps"]) for e in episodes[name].values()])
                ),
                "mean_objective_completion": r["mean_objective_completion_ratio"],
                "invalid_actions": 0,
            }
            for name, r in results.items()
        ],
    )
    write_csv(
        "comparisons.csv",
        [
            {
                "family": family,
                "comparison": name,
                "difference": item["difference"],
                "ci_low": item["ci95"][0],
                "ci_high": item["ci95"][1],
                "test": item["test"],
                "p_value": item["p_value"],
                "holm_p": item.get("holm_p"),
            }
            for family, comparisons_group in (
                ("primary", {"mean_final_minus_Greedy": statistics["primary"]}),
                ("secondary", statistics["secondary"]),
            )
            for name, item in comparisons_group.items()
        ],
    )
    write_csv(
        "paired_outcomes.csv",
        [
            {
                "seed": seed,
                "initial_fingerprint": initial["Random"][seed]["fingerprint"],
                **{name: int(wins[name][index]) for name in wins},
            }
            for index, seed in enumerate(seeds)
        ],
    )
    write_csv(
        "critic_predictions.csv", [r for rows in critic_rows.values() for r in rows]
    )
    selected = sorted(
        {seed for group in representatives.values() for seed in group["selected_seeds"]}
    )
    traces = {
        seed: {"initial_state": initial["Random"][seed]["observation"], "policies": {}}
        for seed in selected
    }
    for name in wins:
        moves = read_csv(root / name / "evaluation/moves.csv")
        by_episode = {}
        for move in moves:
            by_episode.setdefault(move["episode_id"], []).append(move)
        for seed in selected:
            record = episodes[name][seed]
            ordered = sorted(
                by_episode[record["episode_id"]], key=lambda r: int(r["move_index"])
            )
            traces[seed]["policies"][name] = {
                "actions": [int(r["action_id"]) for r in ordered],
                "outcome": record["outcome"],
                "moves": int(record["moves_used"]),
                "objective_progress": json.loads(record["objective_progress"]),
            }
    with (output / "representative_boards.json").open("x") as handle:
        json.dump(
            {"criteria": representatives, "boards": traces},
            handle,
            indent=2,
            allow_nan=False,
        )
    result = {
        "protocol_sha256": digest,
        "policies": results,
        "statistics": statistics,
        "critic_behavior_agreement": critic,
        "representatives": representatives,
        "source_sha256": sources,
        "episodes": len(all_rows),
        "final_test_used": False,
    }
    result["figures"] = create_figures(
        results, wins, critic_rows, output / "figures", statistics
    )
    write_report(result, traces, output / "report.md")
    with (output / "analysis.json").open("x") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
    if any(sha256(name) != expected for name, expected in sources.items()):
        raise RuntimeError("evaluation inputs changed during analysis")
    load_protocol(protocol_path)
    return result


def write_report(result, traces, destination):
    statistics = result["statistics"]
    lines = [
        "# Sprint 5.16: preregistered fresh-board paired evaluation",
        "",
        f"Frozen protocol SHA-256: `{result['protocol_sha256']}`. {result['episodes']} episodes; no training or final-test evaluation. All policies received matched initial observations, board seeds, engine seeds and level configuration. Identical initial RNG seeds do not force identical future tile sequences after different actions.",
        "",
        "| Frozen policy | Wins / boards | Win rate [Wilson 95% CI] | Reward | Moves | Objective completion | Invalid actions |",
        "|---|---|---|---:|---:|---:|---:|",
    ]
    for name, r in result["policies"].items():
        lo, hi = r["win_rate_ci95"]
        lines.append(
            f"| {name} | {r['wins']} / {r['attempted']} | {r['win_rate']:.2%} [{lo:.2%}, {hi:.2%}] | {r['mean_total_reward']:.4f} | {r['mean_moves_used']:.3f} | {r['mean_objective_completion_ratio']:.2%} | 0 |"
        )
    p = statistics["primary"]
    lines += [
        "",
        "## Single primary comparison",
        "",
        f"Mean final PPO versus Greedy: **{100 * p['difference']:+.3f} percentage points**, paired board-bootstrap 95% CI **[{100 * p['ci95'][0]:+.3f}, {100 * p['ci95'][1]:+.3f}]**; approximate null-centered bootstrap p={p['p_value']:.6f}. This primary comparison is separate from the secondary Holm family.",
        "",
        f"Mean final PPO {statistics['mean_final_PPO']:.2%}; training-seed sample SD {100 * statistics['final_PPO_training_seed_sample_sd']:.3f} pp. Mean initial PPO {statistics['mean_initial_PPO']:.2%}; sample SD {100 * statistics['initial_PPO_training_seed_sample_sd']:.3f} pp. Mean frozen DQN {statistics['mean_DQN']:.2%}; sample SD {100 * statistics['DQN_training_seed_sample_sd']:.3f} pp.",
        "",
        "The bootstrap resamples boards jointly across all fixed models. Its interval conditions on the three observed PPO training seeds and does not quantify uncertainty across the population of training runs. No pooled 1500-independent-episode interval is used. Wilson intervals describe individual fixed policies. Fresh validation is not reserved final-test evaluation.",
        "",
        "## Nine preregistered secondary comparisons",
        "",
        "| Comparison | Difference pp [pointwise 95% CI] | Test | Raw p | Holm p |",
        "|---|---|---|---:|---:|",
    ]
    for name, r in statistics["secondary"].items():
        lines.append(
            f"| {name} | {100 * r['difference']:+.3f} [{100 * r['ci95'][0]:+.3f}, {100 * r['ci95'][1]:+.3f}] | {r['test']} | {r['p_value']:.6f} | {r['holm_p']:.6f} |"
        )
    lines += [
        "",
        "Aggregate mean tests use a null-centered board bootstrap: approximate inference for means, not an exact label-exchangeability test. Individual binary-policy tests use exact McNemar. All nine secondary p values receive one Holm correction. Intervals are pointwise, not multiplicity-adjusted; nonsignificance is not equivalence. Paired tables are retained in analysis.json and all per-board binary outcomes in paired_outcomes.csv.",
        "",
        "## Held-out critic behavior-return agreement",
        "",
        "The critic was trained under stochastic masked PPO. These rollouts use the deterministic greedy actor, so the following diagnostics are off-policy behavior-return agreement, not estimates of on-policy critic accuracy. Targets are gamma-discounted terminal rewards from one trajectory per initial board (gamma .99). Different policies induce different target trajectories; EV values are not a common fixed-target benchmark and must not be ranked as critic superiority. Training-time GAE diagnostics are separate and unchanged.",
        "",
        "| PPO policy | Mean V | Return variance | Bias V−return | RMSE | EV |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, r in result["critic_behavior_agreement"].items():
        lines.append(
            f"| {name} | {r['value_prediction_mean']:.4f} | {r['return_target_variance']:.4f} | {r['prediction_bias_V_minus_return']:.4f} | {r['rmse']:.4f} | {r['explained_variance']} |"
        )
    lines += [
        "",
        "## Predefined behavioral examples",
        "",
        "PPO denotes majority outcome across the three final policies. Each category selects the three lowest seed IDs; empty categories are not replaced. Full initial observations, all eleven action sequences, moves, outcomes and objective progress are preserved in representative_boards.json.",
        "",
        "| Category | Available boards | Selected seeds |",
        "|---|---:|---|",
    ]
    for name, group in result["representatives"].items():
        lines.append(f"| {name} | {group['available']} | {group['selected_seeds']} |")
    lines += [
        "",
        "| Example seed | Policy | Outcome / moves | Action prefix |",
        "|---:|---|---|---|",
    ]
    for seed, board in traces.items():
        for name in ("Greedy", "PPO5915_final", "PPO5916_final", "PPO5917_final"):
            r = board["policies"][name]
            lines.append(
                f"| {seed} | {name} | {r['outcome']} / {r['moves']} | {r['actions'][:8]} |"
            )
    lines += [
        "",
        "## Interpretation and limits",
        "",
        "Observed fresh-board differences and confidence intervals concern these frozen models on this one level/generator distribution. They do not isolate algorithm effects, equalize optimizer/wall budgets, prove equivalence or establish superiority across training seeds. Checkpoints were fixed at initialization or the absolute 50000-step budget, never selected by fresh-board outcomes or historical validation maxima. No hyperparameter, architecture, reward or weight was changed.",
        "",
        "Review the primary effect, per-seed changes and behavioral cases before approving any longer PPO experiment. A longer-budget hypothesis requires a separate preregistration and checkpoint/continuation plan; exact mid-episode training resume is unsupported. No longer training, tuning, PBRS, CNN-PPO or Sprint 5.17 work was started.",
        "",
        "## Figures",
        "",
    ]
    for name in result["figures"]:
        if name.endswith(".png"):
            lines += [f"![{name}](figures/{name})", ""]
    with destination.open("x") as handle:
        handle.write("\n".join(lines) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--results-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.protocol, args.results_root, args.output_dir)


if __name__ == "__main__":
    main()
