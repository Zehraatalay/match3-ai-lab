"""Standalone figures and research report for frozen-agent behavior analysis."""

import json
from pathlib import Path

import numpy as np

MASTERS = (5901, 5902, 5903)


def create_figures(result, history, occurrences, moves, output_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=False)
    filenames = []

    def save(fig, name):
        for suffix in ("png", "pdf"):
            filename = f"{name}.{suffix}"
            fig.savefig(output / filename, dpi=180, bbox_inches="tight")
            filenames.append(filename)
        plt.close(fig)

    def axes(ylabel):
        fig, ax = plt.subplots(figsize=(9, 5), constrained_layout=True)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.2)
        return fig, ax

    fig, ax = axes("Win rate on paired validation seeds")
    x = np.arange(3)
    for offset, step, label in ((-0.18, 0, "Step zero"), (0.18, 50000, "Final")):
        points = [result["history"][f"DQN{master}_{step}"] for master in MASTERS]
        y = np.array([p["win_rate"] for p in points])
        low, high = np.array([p["ci95"] for p in points]).T
        ax.bar(
            x + offset, y, width=0.35, label=label, yerr=[y - low, high - y], capsize=3
        )
    for name, style in (("RandomAgent", ":"), ("GreedyAgent", "--")):
        ax.axhline(
            result["history"][name]["win_rate"],
            linestyle=style,
            label=name,
            color="gray",
        )
    ax.set(xticks=x, xticklabels=MASTERS, ylim=(0, 1), xlabel="Training seed")
    ax.legend()
    save(fig, "step_zero_final")

    fig, ax = axes("Validation win rate")
    for master in MASTERS:
        steps = list(range(0, 50001, 5000))
        ax.plot(
            steps,
            [result["history"][f"DQN{master}_{step}"]["win_rate"] for step in steps],
            marker="o",
            label=str(master),
        )
    ax.set(ylim=(0, 1), xlabel="Environment training steps")
    ax.legend()
    save(fig, "learning_progression")

    fig, ax = axes("Fraction of paired episodes")
    bottom = np.zeros(3)
    pairs = [
        next(
            item
            for item in result["paired_comparisons"]
            if item["master_seed"] == master and item["reference"] == "GreedyAgent"
        )
        for master in MASTERS
    ]
    for key, label in (
        ("both_win", "Both win"),
        ("left_only_win", "DQN only wins"),
        ("right_only_win", "Greedy only wins"),
        ("both_lose", "Both lose"),
    ):
        values = np.array([item["outcomes"]["percentages"][key] for item in pairs])
        ax.bar(x, values, bottom=bottom, label=label)
        bottom += values
    ax.set(xticks=x, xticklabels=MASTERS, ylim=(0, 1), xlabel="Training seed")
    ax.legend(loc="upper left", bbox_to_anchor=(1, 1))
    save(fig, "paired_outcomes")

    fig, ax = axes("Exact DQN–Greedy agreement")
    groups = list(result["shared_state"]["agreement"])
    for master in MASTERS:
        estimates = [
            result["shared_state"]["agreement"][group][f"agree_greedy_{master}"]
            for group in groups
        ]
        line = ax.plot(
            range(len(groups)),
            [item["rate"] for item in estimates],
            marker="o",
            label=str(master),
        )[0]
        ax.vlines(
            range(len(groups)),
            [item["ci95"][0] for item in estimates],
            [item["ci95"][1] for item in estimates],
            color=line.get_color(),
            alpha=0.5,
        )
    ax.plot(
        range(len(groups)),
        [
            result["shared_state"]["agreement"][group]["expected_random_agreement"][
                "rate"
            ]
            for group in groups
        ],
        linestyle="--",
        label="Uniform random expectation",
    )
    ax.set(xticks=range(len(groups)), xticklabels=groups, ylim=(0, 1))
    ax.legend()
    save(fig, "action_agreement")
    for dimension, filename, xlabel in (
        ("moves", "agreement_moves", "Moves remaining"),
        ("blocker_hp", "agreement_hp", "Remaining blocker HP"),
    ):
        fig, ax = axes("Exact DQN–Greedy agreement (balanced source mixture)")
        groups = sorted(result["shared_state"]["strata"][dimension], key=int)
        for master in MASTERS:
            ax.plot(
                [int(group) for group in groups],
                [
                    result["shared_state"]["strata"][dimension][group][
                        f"agree_greedy_{master}"
                    ]["rate"]
                    for group in groups
                ],
                marker="o",
                label=str(master),
            )
        ax.plot(
            [int(group) for group in groups],
            [
                result["shared_state"]["strata"][dimension][group][
                    "expected_random_agreement"
                ]["rate"]
                for group in groups
            ],
            linestyle="--",
            label="Random expectation",
        )
        ax.set(xlabel=xlabel, ylim=(0, 1))
        ax.legend()
        save(fig, filename)

    fig, ax = axes("Objective completion ratio")
    labels = [
        "RandomAgent",
        "GreedyAgent",
        *[f"DQN{master}_{step}" for master in MASTERS for step in (0, 50000)],
    ]
    ax.boxplot(
        [
            [row["objective_completion_ratio"] for row in history[name]]
            for name in labels
        ],
        tick_labels=[
            name.replace("Agent", "").replace("_50000", " F").replace("_0", " Z")
            for name in labels
        ],
    )
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", rotation=40)
    save(fig, "objective_distributions")

    fig, ax = axes("Mean HP among episodes still at a decision state")
    sources = sorted({row["source_policy"] for row in occurrences})
    for source in sources:
        groups = sorted(
            {row["move_index"] for row in occurrences if row["source_policy"] == source}
        )
        ax.plot(
            [step - 1 for step in groups],
            [
                np.mean(
                    [
                        row["blocker_hp"]
                        for row in occurrences
                        if row["source_policy"] == source and row["move_index"] == step
                    ]
                )
                for step in groups
            ],
            label=source,
        )
    ax.set_xlabel("Moves already used (conditional survivor population)")
    ax.legend(fontsize=8)
    save(fig, "blocker_hp_progression")

    fig, ax = axes("Fraction of observed transitions")
    names = list(result["reward_sparsity"])
    bottom = np.zeros(len(names))
    for reward, label in (
        ("-1", "Loss reward"),
        ("0", "Zero reward"),
        ("1", "Win reward"),
    ):
        values = np.array(
            [result["reward_sparsity"][name]["fractions"][reward] for name in names]
        )
        ax.bar(range(len(names)), values, bottom=bottom, label=label)
        bottom += values
    ax.set(xticks=range(len(names)), xticklabels=names, ylim=(0, 1))
    ax.tick_params(axis="x", rotation=25)
    ax.legend()
    save(fig, "reward_distribution")

    fig, axes_ = plt.subplots(1, 3, figsize=(13, 4.5), constrained_layout=True)
    for ax, key, title in zip(
        axes_,
        ("cleared_tiles", "objective_delta", "blocker_hits"),
        ("Tiles cleared", "Objective completion gain", "Blocker hits"),
    ):
        estimates = [
            result["immediate_quality"]["Pooled"][str(master)][key]
            for master in MASTERS
        ]
        ax.bar(range(3), [item["rate"] for item in estimates])
        ax.vlines(
            range(3),
            [item["ci95"][0] for item in estimates],
            [item["ci95"][1] for item in estimates],
            color="black",
        )
        ax.axhline(0, color="gray", linewidth=1)
        ax.set(
            xticks=range(3), xticklabels=MASTERS, title=title, xlabel="Training seed"
        )
    fig.suptitle(
        "Exact one-action counterfactual: DQN minus Greedy (cluster bootstrap 95% CI)"
    )
    save(fig, "immediate_quality")
    return filenames


def write_report(
    result, history, occurrences, counterfactuals, episodes, moves, output
):
    def pct(value):
        return f"{100 * value:.2f}%"

    def estimate(item):
        return f"{pct(item['rate'])} [{pct(item['ci95'][0])}, {pct(item['ci95'][1])}]"

    shared = result["shared_state"]
    lines = [
        "# Sprint 5.10 — Agent behavior and learning analysis",
        "",
        "## Executive summary",
        "",
        "This study asks whether training changes DQN behavior and improves outcomes relative to its own initialization and a random policy. Greedy is a meaningful immediate-match heuristic, not an optimal policy or an imitation target. Agreement/disagreement alone does not establish decision quality.",
        "",
        f"Frozen policies were evaluated on {len(result['validation_seeds'])} paired initial validation seeds. Stored data contain {shared['occurrences']} decision occurrences and {shared['unique_observations']} unique inference observations. No training, weight updates, reward shaping, objective-aware agent or engine changes were performed.",
        "",
        "## Research questions",
        "",
        "Does DQN improve relative to its own initialization? Does its action distribution differ from uniform random? How often do DQN and Greedy agree on identical observations, including deterministic tie-breaking? What immediate consequences follow from different actions under an identical complete state and RNG? Which observations justify future controlled experiments?",
        "",
        "## Experimental protocol",
        "",
        f"Validation seeds are {result['validation_seeds'][0]}–{result['validation_seeds'][-1]}; reserved final-test seeds 400000–400999 were never accessed. RandomAgent uses independent agent seeds beginning at 600000. Existing Sprint 5.9 checkpoint trajectories are reused only after checkpoint/configuration checks; fresh Random, Greedy and three final DQN trajectory results must match those historical outcomes and totals exactly.",
        "",
        f"Checkpoint compatibility was verified for {len(result['checkpoint_compatibility'])} files, including all 33 periodic checkpoints and three full final checkpoints. The real step-zero files were loaded and compared with freshly initialized networks from their saved seeds/configuration; no missing initialization was fabricated. Source hashes are in analysis.json and rechecked after analysis. Frozen model weights and exploration RNG are also checked for mutation.",
        "",
        "Same initial episode seed means the same initial board and engine RNG seed. It does not imply identical later boards: different actions and special effects can consume refill RNG differently. Fresh complete episodes are observational paired-start comparisons; only the explicitly cloned one-action comparisons hold complete internal state fixed.",
        "",
        "## Dataset construction and integrity",
        "",
        "observations.npz stores each full typed inference observation once, with allow_pickle=False-compatible arrays. Fingerprints include tiles, specials, blocker types/HP, cell mask, shape, moves, objective types/colors/progress/targets/masks and the action mask. state_occurrences.jsonl preserves source policy, episode/seed, move index, shape, progress, HP, legal count, fingerprint, chosen actions, full-state fingerprint and engine RNG state. Existing gameplay CSV supplies action prefixes for independently verified state restoration.",
        "",
        "| Source policy | Decision occurrences |",
        "|---|---:|",
    ]
    for name, count in shared["by_source"].items():
        lines.append(f"| {name} | {count} |")
    lines += [
        "",
        "Deduplication saves inference storage; statistical analyses retain occurrences and their provenance. Within each policy, moves are occurrence-weighted. DQN-generated results give each of the three generating DQN policies equal weight. The pooled mixture gives Random-generated, Greedy-generated and DQN-generated families equal 1/3 weights; consequently each DQN generating policy has pooled weight 1/9. This is a documented sample mixture, not the distribution of all possible game states.",
        "",
        "## Learning progression",
        "",
        "| Training seed | Step-zero win rate | Final win rate | Change [paired bootstrap 95% CI] | Exact McNemar p / Holm p |",
        "|---:|---:|---:|---|---|",
    ]
    for master in MASTERS:
        item = next(
            item
            for item in result["paired_comparisons"]
            if item["master_seed"] == master and item["reference"] == f"DQN{master}_0"
        )
        diff = item["outcomes"]["difference"]
        lines.append(
            f"| {master} | {pct(diff['right_mean'])} | {pct(diff['left_mean'])} | {100 * diff['absolute_difference']:+.2f} pp [{100 * diff['ci95'][0]:+.2f}, {100 * diff['ci95'][1]:+.2f}] | {item['outcomes']['mcnemar_exact_p']:.5f} / {item['holm_p']:.5f} |"
        )
    lines += [
        "",
        "Parameters changed in training, and shared-state action preferences can change; neither alone establishes useful learning. Learning evidence is assessed using paired outcomes and objective/HP/reward consequences. Mean changes, relative changes (undefined at a zero reference), confidence intervals and counts are all retained in analysis.json.",
        "",
        "| Policy checkpoint | Win rate [Wilson 95% CI] | Completion | HP | Moves | Reward | Specials | Cascades | Legal actions after step |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["history"].items():
        metrics = item["metrics"]
        lines.append(
            f"| {name} | {pct(item['win_rate'])} [{pct(item['ci95'][0])}, {pct(item['ci95'][1])}] | "
            + " | ".join(
                f"{metrics[key]['mean']:.4f}"
                for key in (
                    "objective_completion_ratio",
                    "blocker_hp_remaining",
                    "moves_used",
                    "total_reward",
                    "special_activations",
                    "total_cascades",
                    "mean_legal_actions_after_step",
                )
            )
            + " |"
        )
    lines += [
        "",
        "Historical legal-action means refer to post-step masks, including the final returned board; shared-state counts refer to pre-action decision masks. They are different populations and are labeled separately.",
        "",
        "## Comparison with Random and Greedy",
        "",
        "| Training seed / reference | Final minus reference win rate [paired 95% CI] | Exact p / Holm p |",
        "|---|---|---|",
    ]
    for item in result["paired_comparisons"]:
        if item["reference"].endswith("_0"):
            continue
        diff = item["outcomes"]["difference"]
        lines.append(
            f"| {item['master_seed']} / {item['reference']} | {100 * diff['absolute_difference']:+.2f} pp [{100 * diff['ci95'][0]:+.2f}, {100 * diff['ci95'][1]:+.2f}] | {item['outcomes']['mcnemar_exact_p']:.5f} / {item['holm_p']:.5f} |"
        )
    lines += [
        "",
        "## Exact shared-state action agreement",
        "",
        "Intervals use environment-seed cluster resampling, retaining all repeated states/episodes from each sampled seed and recomputing source-balanced ratios. Random agreement is analytical 1/K for K legal actions, not a sampled RandomAgent decision. Greedy breaks tied immediate match scores by lowest canonical action ID. Tie-optimal agreement additionally counts DQN actions with the same immediate heuristic score as Greedy; it is not a long-term optimality claim.",
        "",
        "| State source | DQN seed | DQN–Greedy exact [cluster 95% CI] | Final–zero exact | Zero–Greedy exact | Random expectation | Greedy-score tie-optimal |",
        "|---|---:|---|---:|---:|---:|---:|",
    ]
    for group, items in shared["agreement"].items():
        for master in MASTERS:
            lines.append(
                f"| {group} | {master} | {estimate(items[f'agree_greedy_{master}'])} | {pct(items[f'agree_zero_{master}']['rate'])} | {pct(items[f'agree_zero_greedy_{master}']['rate'])} | {pct(items['expected_random_agreement']['rate'])} | {pct(items[f'tie_optimal_{master}']['rate'])} |"
            )
    lines += [
        "",
        "Agreement by remaining moves, remaining blocker HP, objective completion and forced-vs-choice masks is in the figures and analysis.json. One-action masks force 100% agreement and do not indicate learned preferences. Excess agreement over 1/K and clustered confidence intervals are reported without a broad family of hypothesis tests.",
        "",
        "## Immediate action quality",
        "",
        "Exact counterfactuals are supported by the current in-memory architecture: deepcopy includes BoardState arrays, session raw/capped objective counters, moves/status, engine settings and local engine/Gym RNG. Every action is applied to a distinct full clone. Full-state fingerprints and memory-sharing checks validate independence; the original state/RNG must remain unchanged. For every sampled trajectory decision, the source action's cloned next observation must reproduce the actual next observation. Prefix restoration is available in analysis code and verifies full state/RNG, not just observation arrays. No Match3Engine or LevelSession modifications were needed.",
        "",
        "counterfactuals.jsonl records actual engine clears, capped objective gain, blocker hits/destruction, remaining HP, special activations, cascades, immediate reward and terminal status. These are causal one-action effects conditional on this exact state/RNG. They do not measure long-term return under continuation policies or prove that a trajectory's first disagreement caused its final win/loss.",
        f"Counterfactual integrity: {result['counterfactual_integrity']}. Errors are recorded explicitly and failed pairs are excluded from immediate-effect estimates, not converted to zero effects.",
        "",
        "| State source / DQN | Tiles difference | Objective-gain difference | Blocker-hit difference | HP difference | Specials difference | Cascade difference |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for group, items in result["immediate_quality"].items():
        for master, metrics in items.items():
            lines.append(
                f"| {group} / {master} | "
                + " | ".join(
                    f"{metrics[key]['rate']:+.5f} [{metrics[key]['ci95'][0]:+.5f}, {metrics[key]['ci95'][1]:+.5f}]"
                    for key in (
                        "cleared_tiles",
                        "objective_delta",
                        "blocker_hits",
                        "blocker_hp_remaining",
                        "special_activations",
                        "cascades",
                    )
                )
                + " |"
            )
    lines += [
        "",
        "### Disagreement-only immediate effects",
        "",
        "These estimates condition on DQN and Greedy selecting different actions. Agreement cases are excluded; they measure local consequences on this sampled subset, not overall policy value.",
        "",
        "| Source / DQN | Differing occurrences | Tiles difference | Objective-gain difference | Blocker-hit difference | Immediate terminal-win difference |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for group, items in result["disagreement_quality"].items():
        for master, metrics in items.items():
            if metrics["cleared_tiles"] is None:
                continue
            lines.append(
                f"| {group} / {master} | {metrics['cleared_tiles']['n_occurrences']} | "
                + " | ".join(
                    f"{metrics[key]['rate']:+.5f} [{metrics[key]['ci95'][0]:+.5f}, {metrics[key]['ci95'][1]:+.5f}]"
                    for key in (
                        "cleared_tiles",
                        "objective_delta",
                        "blocker_hits",
                        "terminal_win_difference",
                    )
                )
                + " |"
            )
    lines += [
        "",
        "## Paired episode outcomes",
        "",
        "| DQN seed | Both win | DQN only wins | Greedy only wins | Both lose |",
        "|---:|---:|---:|---:|---:|",
    ]
    cases = []
    by_occurrence = {}
    for row in counterfactuals:
        if row["policy"] == "source_action":
            by_occurrence[row["occurrence_id"]] = row
    for master in MASTERS:
        pair = next(
            item
            for item in result["paired_comparisons"]
            if item["master_seed"] == master and item["reference"] == "GreedyAgent"
        )
        counts, percentages = (
            pair["outcomes"]["counts"],
            pair["outcomes"]["percentages"],
        )
        lines.append(
            f"| {master} | "
            + " | ".join(
                f"{counts[key]} ({pct(percentages[key])})"
                for key in ("both_win", "left_only_win", "right_only_win", "both_lose")
            )
            + " |"
        )
        dqn = {
            row["seed"]: row
            for row in episodes
            if row["agent_name"] == f"DQN{master}_final"
        }
        greedy = {
            row["seed"]: row for row in episodes if row["agent_name"] == "GreedyAgent"
        }
        for category, a, b in (
            ("DQN wins, Greedy loses", "WON", "LOST"),
            ("Greedy wins, DQN loses", "LOST", "WON"),
        ):
            eligible = sorted(
                seed
                for seed in dqn
                if dqn[seed]["outcome"] == a and greedy[seed]["outcome"] == b
            )
            if not eligible:
                continue
            seed = eligible[0]
            case = {
                "master_seed": master,
                "environment_seed": seed,
                "category": category,
                "selection": "earliest seed in discordant category",
                "trajectories": {},
            }
            lines += [
                "",
                f"### Case {master} / {seed}: {category}",
                "",
                "Selected deterministically as the earliest seed in this discordant category, rather than for dramatic effects. Later boards may differ after the first divergent action.",
                "",
                "| Policy | Move | Action | Pre-state fingerprint | Completion after | HP after | Specials | Cascades |",
                "|---|---:|---:|---|---:|---:|---:|---:|",
            ]
            for label in (f"DQN{master}_final", "GreedyAgent"):
                rows = [
                    row
                    for row in occurrences
                    if row["source_policy"] == label and row["environment_seed"] == seed
                ]
                trajectory = []
                for row in rows:
                    after = by_occurrence[row["occurrence_id"]]
                    trajectory.append(
                        {
                            "move": row["move_index"],
                            "action": row["source_action"],
                            "pre_fingerprint": row["fingerprint"],
                            "completion_after": row["objective_completion"]
                            + after["objective_delta"],
                            "hp_after": after["blocker_hp_remaining"],
                            "specials": after["special_activations"],
                            "cascades": after["cascades"],
                        }
                    )
                    item = trajectory[-1]
                    lines.append(
                        f"| {label} | {item['move']} | {item['action']} | {item['pre_fingerprint'][:10]} | {item['completion_after']:.3f} | {item['hp_after']} | {item['specials']} | {item['cascades']} |"
                    )
                case["trajectories"][label] = trajectory
            cases.append(case)
    with (output / "representative_cases.json").open("x") as handle:
        json.dump(cases, handle, indent=2)
    lines += [
        "",
        "## Reward and learning signal",
        "",
        "| Policy | Transitions | Zero reward | Positive terminal | Negative terminal | Median length | Mean completion before terminal | Mean earlier blocker hits |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, item in result["reward_sparsity"].items():
        lines.append(
            f"| {name} | {item['transitions']} | {pct(item['fractions']['0'])} | {pct(item['fractions']['1'])} | {pct(item['fractions']['-1'])} | {np.median(item['episode_lengths']):.1f} | {np.mean([row['objective_completion_before_terminal'] for row in item['before_terminal']]):.3f} | {np.mean([row['blocker_hits_before_terminal'] for row in item['before_terminal']]):.3f} |"
        )
    lines += [
        "",
        "Actual historical replay contents are available in the full final checkpoints, so their reward distribution is measured directly, not reconstructed or invented:",
        "",
        "| Training seed | Replay size | Zero | Positive | Negative | Terminated / truncated |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for master, item in result["actual_replay_statistics"].items():
        lines.append(
            f"| {master} | {item['transitions']} | {pct(item['fractions']['0'])} | {pct(item['fractions']['1'])} | {pct(item['fractions']['-1'])} | {item['terminated']} / {item['truncated']} |"
        )
    lines += [
        "",
        "Most intermediate progress receives no immediate scalar reward, so terminal credit must propagate through bootstrapped targets. This makes credit assignment a plausible challenge, not proof that sparse reward caused the observed performance. Reward shaping is not implemented.",
        "",
        "## Statistical interpretation",
        "",
        f"Bootstrap draws: {result['bootstrap_samples']}; deterministic seed-derived streams. Episode-level paired differences resample complete initial-seed pairs. Exact two-sided McNemar tests condition on discordant outcomes. Holm correction covers three primary final-vs-zero tests and separately six exploratory baseline comparisons. Continuous differences have descriptive percentile bootstrap intervals, without numerous extra p-values. Relative changes are withheld at a zero reference; ratios involving signed sparse returns need caution.",
        "",
        "Shared-state uncertainty resamples initial environment-seed clusters, preserving within-episode dependence and cross-policy reuse. Unique observations are not treated as independent samples. The same validation boards are reused across checkpoints and the earlier pilot. Wilson intervals are conditional on each fixed policy. Nonsignificance is not equivalence; three training seeds do not establish a population-wide or superior strategy. Exact agreement is distinct from expected return and is not inherently desirable.",
        "",
        "## Limitations and unsupported conclusions",
        "",
        "Only one level, 100 familiar validation seeds, three training initializations and sampled trajectory states are covered. The pooled source weighting is a research convention. Greedy's immediate score ignores objectives, specials and future cascades, and is not an optimality certificate. Counterfactuals fix one particular RNG realization rather than integrating all future randomness; favorable immediate effects do not establish long-term quality. Episode paths after disagreement are descriptive, not isolated causal experiments. There is no reserved-test evaluation, no retraining and no new reward/agent architecture.",
        "",
        "## Recommendations",
        "",
        "Use the paired learning intervals and shared-state preferences to decide whether useful learning evidence is stronger than raw final win-rate comparisons suggest. Inspect discordant cases and disagreement-only immediate effects before selecting an intervention. A future approved experiment can isolate one hypothesis, such as objective-progress credit assignment, value calibration or representation, with the same frozen evaluation protocol; avoid simultaneously changing rewards, architecture and heuristic policy. Greedy imitation is not a research objective by default.",
        "",
        "## Figures",
        "",
    ]
    for filename in result["figures"]:
        if filename.endswith(".png"):
            lines += [f"![{filename}](figures/{filename})", ""]
    with (output / "report.md").open("x") as handle:
        handle.write("\n".join(lines) + "\n")
