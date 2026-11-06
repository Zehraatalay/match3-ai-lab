"""Sprint 5.10: frozen-policy trajectories, shared-state decisions and outcomes."""

import argparse
import hashlib
import json
import platform
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from match3_ai.agents.dqn_agent import DQNAgent
from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.evaluation.analysis import (
    EPISODE_FLOATS,
    EPISODE_INTS,
    EPISODE_JSON,
    MOVE_INTS,
    MOVE_JSON,
    _read_csv,
    numeric_summary,
    wilson_interval,
)
from match3_ai.evaluation.analytics import (
    AnalyticsCollector,
    MoveRecord,
    export_evaluation,
)
from match3_ai.evaluation.behavior_data import (
    ObservationStore,
    complete_state_fingerprint,
    complete_state_metadata,
    counterfactual_outcomes,
    greedy_scores,
    objective_completion,
    observation_fingerprint,
)
from match3_ai.evaluation.behavior_statistics import (
    clustered_agreement,
    expected_random_agreement,
    holm_adjust,
    paired_difference,
    paired_outcomes,
    reward_distribution,
)
from match3_ai.evaluation.runner import EpisodeResult, run_episode
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.training.seed_policy import validate_partition

MASTER_SEEDS = (5901, 5902, 5903)
METRICS = (
    "objective_completion_ratio",
    "blocker_hp_remaining",
    "moves_used",
    "total_reward",
    "special_activations",
    "total_cascades",
    "mean_legal_actions_after_step",
)


def _seed(label):
    return int.from_bytes(hashlib.sha256(label.encode()).digest()[:4], "little")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _weights(agent):
    return hashlib.sha256(
        b"".join(
            value.detach().numpy().tobytes() for value in agent.online.parameters()
        )
    ).hexdigest()


def verify_step_zero(agent, payload):
    if payload["training_step"] != 0:
        raise ValueError("requires a true step-zero checkpoint")
    fresh = DQNAgent(
        payload["seed"], tuple(payload["hidden_dims"]), payload["learning_rate"]
    )
    if _weights(agent) != _weights(fresh) or agent.optimizer.state:
        raise ValueError("step-zero initialization is not reproducible")


def checkpoint_config_matches(saved, current):
    return json.loads(json.dumps(saved, sort_keys=True)) == json.loads(
        json.dumps(current, sort_keys=True)
    )


def read_evaluation(directory):
    path = Path(directory)
    episodes = _read_csv(
        path / "episodes.csv",
        set(EpisodeResult.__dataclass_fields__),
        EPISODE_INTS,
        EPISODE_FLOATS,
        {"completed"},
        EPISODE_JSON,
    )
    moves = _read_csv(
        path / "moves.csv",
        set(MoveRecord.__dataclass_fields__),
        MOVE_INTS,
        {"reward"},
        {"valid_action", "resolution_completed"},
        MOVE_JSON,
    )
    grouped = defaultdict(list)
    for move in moves:
        grouped[move["episode_id"]].append(move)
    for episode in episodes:
        rows = grouped[episode["episode_id"]]
        if (
            len(rows) != episode["steps"]
            or sum(row["reward"] or 0 for row in rows) != episode["total_reward"]
        ):
            raise ValueError("historical episode/move totals do not reconcile")
        episode["mean_legal_actions_after_step"] = (
            float(np.mean([row["legal_action_count"] for row in rows]))
            if rows
            else None
        )
        episode["action_sequence"] = [
            row["action_id"] for row in sorted(rows, key=lambda row: row["move_index"])
        ]
    return episodes, moves


def load_artifacts(input_root, seeds):
    policies, history, hashes, compatibility, replay_statistics = {}, {}, {}, [], {}
    expected_level = None
    for master in MASTER_SEEDS:
        directory = Path(input_root) / f"sprint5_9_seed_{master}"
        summary_path = directory / "training_summary.json"
        summary = json.loads(summary_path.read_text())
        config = summary["training_config"]
        if (
            config["seed"] != master
            or config["level_id"] != "blocker-mayhem"
            or config["environment_steps"] != 50000
        ):
            raise ValueError("unexpected historical training configuration")
        expected_protocol = {
            "max_cascades": 200,
            "max_initialization_attempts": 1000,
            "evaluation_seed_start": 300000,
            "evaluation_agent_seed_start": 600000,
            "evaluation_episodes": 100,
            "max_episode_steps": 50,
        }
        if any(config[key] != value for key, value in expected_protocol.items()):
            raise ValueError("historical evaluation protocol differs")
        if expected_level is not None and expected_level != summary["level_config_id"]:
            raise ValueError("historical level configurations differ")
        expected_level = summary["level_config_id"]
        for path in (
            summary_path,
            directory / "evaluation/episodes.csv",
            directory / "evaluation/moves.csv",
            directory / "evaluation/experiment_metadata.json",
        ):
            hashes[str(path)] = _hash(path)
        episodes, _moves = read_evaluation(directory / "evaluation")
        for step in range(0, 50001, 5000):
            checkpoint = directory / f"policy_step_{step}.pt"
            hashes[str(checkpoint)] = _hash(checkpoint)
            agent, payload = DQNAgent.load_checkpoint(checkpoint)
            if payload["training_step"] != step or not checkpoint_config_matches(
                payload["training_config"], config
            ):
                raise ValueError("checkpoint metadata mismatch")
            if payload["encoder_config"] != {
                "version": 1,
                "max_colors": 5,
                "max_hp": 3,
                "max_moves": 15,
            }:
                raise ValueError("unsupported checkpoint encoder")
            if step == 0:
                verify_step_zero(agent, payload)
                policies[f"DQN{master}_zero"] = agent
            compatibility.append(
                {
                    "path": str(checkpoint),
                    "step": step,
                    "master_seed": master,
                    "weights": _weights(agent),
                    "encoder_compatible": True,
                }
            )
            label = f"DQN{master}_{step}"
            selected = [
                dict(e)
                for e in episodes
                if e["agent_name"] == "DQNAgent"
                and int(e["episode_id"].split(":")[-3]) == step
                and e["seed"] in seeds
            ]
            if {e["seed"] for e in selected} != set(seeds) or len(selected) != len(
                seeds
            ):
                raise ValueError("missing historical validation seeds")
            history[label] = selected
        final_path = directory / "checkpoint_final.pt"
        hashes[str(final_path)] = _hash(final_path)
        agent, payload = DQNAgent.load_checkpoint(final_path)
        if (
            payload["training_step"] != 50000
            or _weights(agent) != summary["final_weights"]
        ):
            raise ValueError("final checkpoint weights/step differ")
        policies[f"DQN{master}_final"] = agent
        replay = payload["training_state"]["replay"]
        rewards = replay["rewards"].numpy()
        terminated = replay["terminated"].numpy()
        truncated = replay["truncated"].numpy()
        replay_statistics[str(master)] = {
            "population": "actual final replay contents, not inferred historical samples",
            "transitions": int(replay["size"]),
            "fractions": {
                str(value): float(np.mean(rewards == value)) for value in (-1, 0, 1)
            },
            "terminated": int(terminated.sum()),
            "truncated": int(truncated.sum()),
            "nonterminal_nonzero_rewards": int(
                np.count_nonzero((rewards != 0) & ~terminated)
            ),
        }
        compatibility.append(
            {
                "path": str(final_path),
                "step": 50000,
                "master_seed": master,
                "weights": _weights(agent),
                "full_replay_available": True,
            }
        )
        for name in ("RandomAgent", "GreedyAgent"):
            selected = [
                dict(e)
                for e in episodes
                if e["agent_name"] == name and e["seed"] in seeds
            ]
            if name not in history:
                history[name] = selected
    env = Match3LevelEnv("blocker-mayhem")
    current = hashlib.sha256(
        json.dumps(asdict(env.config), sort_keys=True).encode()
    ).hexdigest()
    env.close()
    if current != expected_level:
        raise ValueError("live level differs from saved checkpoint evaluation")
    for agent in policies.values():
        agent.online.eval()
    return policies, history, hashes, compatibility, replay_statistics, current


class StateObserver:
    def __init__(
        self,
        env,
        policy,
        source,
        seed,
        episode_id,
        store,
        policies,
        cache,
        occurrences,
        counterfactuals,
    ):
        self.env, self.policy, self.source, self.seed, self.episode_id = (
            env,
            policy,
            source,
            seed,
            episode_id,
        )
        self.store, self.policies, self.cache = store, policies, cache
        self.occurrences, self.counterfactuals = occurrences, counterfactuals
        self.move_index = 0

    def select_action(self, observation):
        self.move_index += 1
        state_id = self.store.intern(observation)
        if state_id not in self.cache:
            scores = greedy_scores(observation)
            best = max(scores.values())
            actions = {
                name: agent.select_action(observation)
                for name, agent in self.policies.items()
            }
            actions["GreedyAgent"] = GreedyAgent().select_action(observation)
            if actions["GreedyAgent"] != min(
                action for action, score in scores.items() if score == best
            ):
                raise RuntimeError(
                    "heuristic scores do not reproduce Greedy tie-breaking"
                )
            if any(
                not observation["action_mask"][action] for action in actions.values()
            ):
                raise RuntimeError("frozen policy selected an illegal action")
            self.cache[state_id] = {
                "actions": actions,
                "scores": scores,
                "best_score": best,
                "best_ties": sum(score == best for score in scores.values()),
            }
        decisions = self.cache[state_id]
        action = int(self.policy.select_action(observation))
        full_state = complete_state_metadata(self.env)
        occurrence_id = len(self.occurrences)
        row = {
            "occurrence_id": occurrence_id,
            "state_id": state_id,
            "fingerprint": self.store.fingerprints[state_id],
            "source_policy": self.source,
            "environment_seed": self.seed,
            "episode_id": self.episode_id,
            "move_index": self.move_index,
            "board_shape": np.asarray(observation["board_shape"]).tolist(),
            "moves_remaining": int(observation["moves_remaining"]),
            "objective_progress": np.asarray(
                observation["objective_progress"]
            ).tolist(),
            "objective_completion": objective_completion(observation),
            "blocker_hp": int(observation["blocker_hp"].sum()),
            "legal_actions": int(np.count_nonzero(observation["action_mask"])),
            "expected_random_agreement": expected_random_agreement(
                observation["action_mask"]
            ),
            "source_action": action,
            "actions": decisions["actions"],
            "greedy_match_score": decisions["best_score"],
            "greedy_best_ties": decisions["best_ties"],
            "full_state_fingerprint": complete_state_fingerprint(self.env),
            "engine_rng_state": full_state["engine_rng"],
            "session_objectives": full_state["objective_progress"],
        }
        for master in MASTER_SEEDS:
            final = decisions["actions"][f"DQN{master}_final"]
            zero = decisions["actions"][f"DQN{master}_zero"]
            row[f"agree_greedy_{master}"] = float(
                final == decisions["actions"]["GreedyAgent"]
            )
            row[f"agree_zero_{master}"] = float(final == zero)
            row[f"agree_zero_greedy_{master}"] = float(
                zero == decisions["actions"]["GreedyAgent"]
            )
            row[f"tie_optimal_{master}"] = float(
                decisions["scores"][final] == decisions["best_score"]
            )
            row[f"heuristic_difference_{master}"] = (
                decisions["scores"][final] - decisions["best_score"]
            )
        candidate = {
            name: value
            for name, value in decisions["actions"].items()
            if name.endswith("_final") or name == "GreedyAgent"
        }
        candidate["source_action"] = action
        outcomes = counterfactual_outcomes(self.env, candidate.values())
        for name, chosen in candidate.items():
            self.counterfactuals.append(
                {"occurrence_id": occurrence_id, "policy": name, **outcomes[chosen]}
            )
        self.occurrences.append(row)
        return action


def collect_dataset(policies, seeds, output, experiment_id):
    store, cache, occurrences, counterfactuals = ObservationStore(), {}, [], []
    collector = AnalyticsCollector()
    sources = [
        "RandomAgent",
        "GreedyAgent",
        *[f"DQN{master}_final" for master in MASTER_SEEDS],
    ]
    for source in sources:
        for index, seed in enumerate(seeds):
            validate_partition("validation", seed, 1)
            env = Match3LevelEnv(
                "blocker-mayhem", max_cascades=200, max_initialization_attempts=1000
            )
            policy = (
                RandomAgent(600000 + seed - 300000)
                if source == "RandomAgent"
                else GreedyAgent()
                if source == "GreedyAgent"
                else policies[source]
            )
            episode_id = f"{experiment_id}:{source}:{seed}"
            observer = StateObserver(
                env,
                policy,
                source,
                seed,
                episode_id,
                store,
                policies,
                cache,
                occurrences,
                counterfactuals,
            )
            try:
                episode = run_episode(
                    env,
                    observer,
                    seed=seed,
                    max_steps=50,
                    experiment_id=experiment_id,
                    episode_id=episode_id,
                    agent_name=source,
                    agent_seed=600000 + seed - 300000
                    if source == "RandomAgent"
                    else policies[source].seed
                    if source.startswith("DQN")
                    else None,
                    collector=collector,
                )
                if episode.outcome not in ("WON", "LOST") or episode.invalid_actions:
                    raise RuntimeError(
                        f"episode failed: {source}/{seed}: {episode.error_message}"
                    )
                actual_rows = [
                    row for row in occurrences if row["episode_id"] == episode_id
                ]
                source_results = {
                    row["occurrence_id"]: row
                    for row in counterfactuals
                    if row["policy"] == "source_action"
                }
                for i, row in enumerate(actual_rows):
                    expected = (
                        actual_rows[i + 1]["fingerprint"]
                        if i + 1 < len(actual_rows)
                        else observation_fingerprint(env._observation())
                    )
                    if (
                        source_results[row["occurrence_id"]]["after_fingerprint"]
                        != expected
                    ):
                        raise RuntimeError(
                            "counterfactual source action does not reproduce actual transition"
                        )
            finally:
                env.close()
            if (index + 1) % 10 == 0:
                print(
                    f"source={source} episodes={index + 1}/{len(seeds)} unique_states={len(store.observations)}",
                    flush=True,
                )
    store.export(output / "observations.npz")
    with (output / "state_occurrences.jsonl").open("x") as handle:
        for row in occurrences:
            handle.write(json.dumps(row) + "\n")
    with (output / "counterfactuals.jsonl").open("x") as handle:
        for row in counterfactuals:
            handle.write(json.dumps(row) + "\n")
    return collector, store, occurrences, counterfactuals


def summarize_history(history, samples):
    descriptions, comparisons = {}, []
    for name, episodes in history.items():
        wins = sum(row["outcome"] == "WON" for row in episodes)
        descriptions[name] = {
            "attempted": len(episodes),
            "wins": wins,
            "win_rate": wins / len(episodes),
            "ci95": wilson_interval(wins, len(episodes)),
            "metrics": {
                metric: numeric_summary(row[metric] for row in episodes)
                for metric in METRICS
            },
        }
    for master in MASTER_SEEDS:
        left = {row["seed"]: row for row in history[f"DQN{master}_50000"]}
        for reference in (f"DQN{master}_0", "RandomAgent", "GreedyAgent"):
            right = {row["seed"]: row for row in history[reference]}
            paired = paired_outcomes(
                {key: r["outcome"] for key, r in left.items()},
                {key: r["outcome"] for key, r in right.items()},
                seed=_seed(f"outcome{master}{reference}"),
                samples=samples,
            )
            eligible = sorted(
                key
                for key in left
                if left[key]["outcome"] in ("WON", "LOST")
                and right[key]["outcome"] in ("WON", "LOST")
            )
            continuous = {}
            for metric in METRICS:
                valid = [
                    key
                    for key in eligible
                    if left[key][metric] is not None and right[key][metric] is not None
                ]
                continuous[metric] = paired_difference(
                    [left[key][metric] for key in valid],
                    [right[key][metric] for key in valid],
                    seed=_seed(f"{master}{reference}{metric}"),
                    samples=samples,
                )
                if metric == "total_reward":
                    continuous[metric]["relative_change"] = None
            comparisons.append(
                {
                    "master_seed": master,
                    "reference": reference,
                    "outcomes": paired,
                    "continuous": continuous,
                }
            )
    primary = [item for item in comparisons if item["reference"].endswith("_0")]
    secondary = [item for item in comparisons if not item["reference"].endswith("_0")]
    for family in (primary, secondary):
        corrected = holm_adjust(
            [item["outcomes"]["mcnemar_exact_p"] for item in family]
        )
        for item, p in zip(family, corrected):
            item["holm_p"] = p
            item["test_family"] = (
                "three final-vs-zero tests"
                if family is primary
                else "six exploratory baseline comparisons"
            )
    return descriptions, comparisons


def analyze_shared(occurrences, counterfactuals, samples):
    source_groups = {
        "Random-generated": [
            row for row in occurrences if row["source_policy"] == "RandomAgent"
        ],
        "Greedy-generated": [
            row for row in occurrences if row["source_policy"] == "GreedyAgent"
        ],
        "DQN-generated": [
            row for row in occurrences if row["source_policy"].startswith("DQN")
        ],
        "Pooled": occurrences,
    }
    metrics = [
        "expected_random_agreement",
        *[
            f"{prefix}_{master}"
            for master in MASTER_SEEDS
            for prefix in (
                "agree_greedy",
                "agree_zero",
                "agree_zero_greedy",
                "tie_optimal",
                "heuristic_difference",
            )
        ],
    ]
    summaries = {}
    for group, rows in source_groups.items():
        summaries[group] = {
            key: clustered_agreement(rows, key, seed=51010, samples=samples)
            for key in metrics
        }
        for master in MASTER_SEEDS:
            key = f"above_random_{master}"
            adjusted = [
                {
                    **row,
                    key: row[f"agree_greedy_{master}"]
                    - row["expected_random_agreement"],
                }
                for row in rows
            ]
            summaries[group][key] = clustered_agreement(
                adjusted, key, seed=51010, samples=samples
            )
    strata = {}
    rules = {
        "moves": {
            str(value): lambda row, value=value: row["moves_remaining"] == value
            for value in range(1, 16)
        },
        "blocker_hp": {
            str(value): lambda row, value=value: row["blocker_hp"] == value
            for value in range(10)
        },
        "objective_completion": {
            "0-.5": lambda r: r["objective_completion"] < 0.5,
            ".5-.9": lambda r: 0.5 <= r["objective_completion"] < 0.9,
            ".9-1": lambda r: r["objective_completion"] >= 0.9,
        },
        "legal_actions": {
            "forced": lambda r: r["legal_actions"] == 1,
            "choice": lambda r: r["legal_actions"] > 1,
        },
    }
    for dimension, groups in rules.items():
        strata[dimension] = {}
        for label, predicate in groups.items():
            rows = [row for row in occurrences if predicate(row)]
            if rows:
                strata[dimension][label] = {
                    key: clustered_agreement(rows, key, seed=51010, samples=samples)
                    for key in (
                        "expected_random_agreement",
                        *[f"agree_greedy_{master}" for master in MASTER_SEEDS],
                    )
                }
    by_occurrence = defaultdict(dict)
    for row in counterfactuals:
        by_occurrence[row["occurrence_id"]][row["policy"]] = row
    immediate = {}
    disagreement = {}
    for group, rows in source_groups.items():
        immediate[group] = {}
        disagreement[group] = {}
        for master in MASTER_SEEDS:
            differences = []
            for row in rows:
                outcomes = by_occurrence[row["occurrence_id"]]
                a, b = outcomes[f"DQN{master}_final"], outcomes["GreedyAgent"]
                if "error_type" in a or "error_type" in b:
                    continue
                differences.append(
                    {
                        "source_policy": row["source_policy"],
                        "environment_seed": row["environment_seed"],
                        **{
                            key: a[key] - b[key]
                            for key in (
                                "cleared_tiles",
                                "objective_delta",
                                "blocker_hits",
                                "blocker_hp_remaining",
                                "special_activations",
                                "cascades",
                                "reward",
                            )
                        },
                        "terminal_win_difference": int(a["level_status"] == "WON")
                        - int(b["level_status"] == "WON"),
                        "disagrees": a["action_id"] != b["action_id"],
                    }
                )
            immediate[group][str(master)] = {
                key: clustered_agreement(differences, key, seed=51010, samples=samples)
                for key in (
                    "cleared_tiles",
                    "objective_delta",
                    "blocker_hits",
                    "blocker_hp_remaining",
                    "special_activations",
                    "cascades",
                    "reward",
                    "terminal_win_difference",
                )
            }
            differing = [row for row in differences if row["disagrees"]]
            disagreement[group][str(master)] = {
                key: clustered_agreement(differing, key, seed=51010, samples=samples)
                for key in immediate[group][str(master)]
            }
    return summaries, strata, immediate, disagreement


def run_analysis(input_root, output_dir, *, episodes=100, bootstrap_samples=5000):
    if type(bootstrap_samples) is not int or bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be a positive integer")
    if type(episodes) is not int or not 1 <= episodes <= 100:
        raise ValueError(
            "primary subset must be within the 100 historical validation seeds"
        )
    validate_partition("validation", 300000, episodes)
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    seeds = tuple(range(300000, 300000 + episodes))
    started = perf_counter()
    old_threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    try:
        policies, history, hashes, compatibility, replay_statistics, level_id = (
            load_artifacts(input_root, seeds)
        )
        initial_weights = {name: _weights(agent) for name, agent in policies.items()}
        rng_before = {
            name: json.dumps(agent.rng.bit_generator.state, sort_keys=True)
            for name, agent in policies.items()
        }
        output.mkdir(parents=True, exist_ok=False)
        experiment_id = hashlib.sha256(
            json.dumps(
                {
                    "inputs": hashes,
                    "seeds": seeds,
                    "bootstrap_samples": bootstrap_samples,
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()[:24]
        collector, store, occurrences, counterfactuals = collect_dataset(
            policies, seeds, output, experiment_id
        )
        metadata = {
            "experiment_id": experiment_id,
            "level_config_id": level_id,
            "validation_seeds": seeds,
            "agent_seed_start": 600000,
            "final_test_used": False,
        }
        export_evaluation(collector, metadata, output / "fresh_episodes")
        fresh_episodes, fresh_moves = read_evaluation(output / "fresh_episodes")
        for name in (
            "RandomAgent",
            "GreedyAgent",
            *[f"DQN{master}_final" for master in MASTER_SEEDS],
        ):
            old = history[
                name
                if name in ("RandomAgent", "GreedyAgent")
                else name.replace("_final", "_50000")
            ]
            current = [row for row in fresh_episodes if row["agent_name"] == name]
            previous = {row["seed"]: row for row in old}
            for row in current:
                for key in (
                    "outcome",
                    "total_reward",
                    "total_cleared",
                    "total_cascades",
                    "special_activations",
                    "blocker_hp_remaining",
                    "moves_used",
                    "action_sequence",
                ):
                    if row[key] != previous[row["seed"]][key]:
                        raise RuntimeError(
                            f"fresh trajectory differs from historical result: {name}/{row['seed']}/{key}"
                        )
        descriptions, comparisons = summarize_history(history, bootstrap_samples)
        shared, strata, immediate, disagreement = analyze_shared(
            occurrences, counterfactuals, bootstrap_samples
        )
        sparsity = {
            name: reward_distribution(
                [row for row in fresh_episodes if row["agent_name"] == name],
                [
                    row
                    for row in fresh_moves
                    if row["episode_id"]
                    in {
                        e["episode_id"]
                        for e in fresh_episodes
                        if e["agent_name"] == name
                    }
                ],
            )
            for name in sorted({row["agent_name"] for row in fresh_episodes})
        }
        result = {
            **metadata,
            "checkpoint_compatibility": compatibility,
            "source_sha256": hashes,
            "history": descriptions,
            "paired_comparisons": comparisons,
            "shared_state": {
                "unique_observations": len(store.observations),
                "occurrences": len(occurrences),
                "by_source": {
                    name: sum(row["source_policy"] == name for row in occurrences)
                    for name in sorted({row["source_policy"] for row in occurrences})
                },
                "agreement": shared,
                "strata": strata,
            },
            "immediate_quality": immediate,
            "disagreement_quality": disagreement,
            "counterfactual_integrity": {
                "records": len(counterfactuals),
                "errors": sum("error_type" in row for row in counterfactuals),
                "source_transitions_reproduced": len(occurrences),
                "original_state_rng_preserved": True,
            },
            "reward_sparsity": sparsity,
            "actual_replay_statistics": replay_statistics,
            "bootstrap_samples": bootstrap_samples,
            "bootstrap_seed": 51010,
            "versions": {
                "python": platform.python_version(),
                "numpy": np.__version__,
                "torch": str(torch.__version__),
            },
            "seconds": perf_counter() - started,
            "exact_counterfactuals": True,
        }
        from match3_ai.evaluation.behavior_report import create_figures, write_report

        result["figures"] = create_figures(
            result, history, occurrences, fresh_moves, output / "figures"
        )
        write_report(
            result,
            history,
            occurrences,
            counterfactuals,
            fresh_episodes,
            fresh_moves,
            output,
        )
        with (output / "analysis.json").open("x") as handle:
            json.dump(result, handle, indent=2, allow_nan=False)
        for path, digest in hashes.items():
            if _hash(path) != digest:
                raise RuntimeError("historical artifacts or checkpoint weights changed")
        for name, agent in policies.items():
            if (
                _weights(agent) != initial_weights[name]
                or json.dumps(agent.rng.bit_generator.state, sort_keys=True)
                != rng_before[name]
            ):
                raise RuntimeError("analysis mutated frozen weights or exploration RNG")
        return result
    finally:
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=Path("experiments/results"))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    args = parser.parse_args()
    run_analysis(
        args.input_root,
        args.output_dir,
        episodes=args.episodes,
        bootstrap_samples=args.bootstrap_samples,
    )


if __name__ == "__main__":
    main()
