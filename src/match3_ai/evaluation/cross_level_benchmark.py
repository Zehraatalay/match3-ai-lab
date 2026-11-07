"""Zero-shot, paired development benchmark of frozen DQN policies and baselines."""

import argparse
import csv
import hashlib
import json
import os
import platform
from dataclasses import asdict
from pathlib import Path

from match3_ai.agents.dqn_agent import DQNAgent
from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.evaluation.behavior_data import observation_fingerprint
from match3_ai.evaluation.experiments import (
    _source_fingerprint,
    summarize_level_episodes,
)
from match3_ai.evaluation.fresh_evaluation import policy_state
from match3_ai.evaluation.fresh_protocol import sha256
from match3_ai.evaluation.runner import run_episode
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.loader import load_level
from match3_ai.training.seed_policy import (
    PARTITIONS,
    SPRINT520_TRAINING_PARTITION,
    STUDY_VALIDATION_PARTITIONS,
)

LEVELS = (
    "first-match",
    "wooden-trouble",
    "double-trouble",
    "frozen-garden",
    "stone-breaker",
    "blocker-mayhem",
)
CALIBRATED_LEVELS = (
    "wooden-trouble",
    "wooden-trouble-v2",
    "double-trouble",
    "double-trouble-v2",
    "frozen-garden",
    "frozen-garden-v2",
)
AVAILABLE_LEVELS = tuple(dict.fromkeys((*LEVELS, *CALIBRATED_LEVELS)))
SEEDS = tuple(range(710000, 710200))
RANDOM_AGENT_SEEDS = tuple(range(720000, 720200))
SELECTION = (
    (5990, 125000, "ddqn_seed5990_150k"),
    (5991, 125000, "ddqn_seed5991_150k_retry"),
    (5992, 150000, "ddqn_seed5992_150k"),
)


def validate_seeds(
    seeds,
    results_root,
    *,
    random_agent_seeds=RANDOM_AGENT_SEEDS,
    expected_count=200,
):
    """Reject protected or previously recorded environment and agent seeds."""
    bounds = [
        *PARTITIONS.values(),
        *STUDY_VALIDATION_PARTITIONS.values(),
        SPRINT520_TRAINING_PARTITION,
    ]

    def validate_group(values, name):
        if len(values) != expected_count or len(set(values)) != expected_count:
            raise ValueError(
                f"Exactly {expected_count} distinct {name} seeds are required"
            )
        if any(
            type(seed) is not int
            or seed < 0
            or any(low <= seed <= high for low, high in bounds)
            for seed in values
        ):
            raise ValueError(f"{name.capitalize()} seeds overlap a protected partition")

    validate_group(seeds, "environment")
    validate_group(random_agent_seeds, "Random agent")
    selected = set(seeds)
    selected_agent_seeds = set(random_agent_seeds)
    if selected & selected_agent_seeds:
        raise ValueError("Environment and Random agent seeds must be disjoint")
    selected |= selected_agent_seeds
    seed_keys = (
        "seed",
        "effective_episode_seed",
        "environment_seed",
        "agent_seed",
        "random_seed",
        "random_agent_seed",
    )

    for path in sorted(Path(results_root).rglob("*.csv")):
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for key in seed_keys:
                    value = row.get(key)
                    if value and int(value) in selected:
                        raise ValueError(f"Benchmark seeds already used: {path}")

    for path in sorted(Path(results_root).rglob("*.jsonl")):
        with path.open() as handle:
            for line in handle:
                row = json.loads(line)
                for record in (row, row.get("info") or {}, row.get("record") or {}):
                    if any(
                        record.get(key) is not None and int(record[key]) in selected
                        for key in seed_keys
                    ):
                        raise ValueError(f"Benchmark seeds already used: {path}")


def prepare(
    results_root,
    *,
    levels=LEVELS,
    environment_seeds=SEEDS,
    random_agent_seeds=RANDOM_AGENT_SEEDS,
):
    """Inspect saved inputs without constructing or resetting an environment."""
    root = Path(results_root)
    levels = tuple(levels)
    if not levels or len(set(levels)) != len(levels):
        raise ValueError("Levels must be a non-empty sequence without duplicates")
    if any(level not in AVAILABLE_LEVELS for level in levels):
        raise ValueError("Unknown benchmark level")
    validate_seeds(environment_seeds, root, random_agent_seeds=random_agent_seeds)
    policies, specifications = {}, []
    for seed, step, directory in SELECTION:
        path = (
            root / "sprint5_20_rl_optimization" / directory / f"policy_step_{step}.pt"
        )
        summary_path = path.parent / "training_summary.json"
        summary = json.loads(summary_path.read_text())
        policy, payload = DQNAgent.load_checkpoint(path)
        config = payload["training_config"]
        if (
            json.loads(json.dumps(config)) != summary["training_config"]
            or config["seed"] != seed
            or config["level_id"] != "blocker-mayhem"
            or payload["training_step"] != step
            or policy.architecture != "dueling"
            or policy.algorithm != "double"
        ):
            raise ValueError(f"Checkpoint selection mismatch: {path}")
        start = config["train_seed_start"]
        if any(
            start <= value < start + config["environment_steps"]
            for value in environment_seeds
        ):
            raise ValueError("Environment seeds overlap checkpoint training range")
        policy.online.eval()
        policy.target.eval()
        policy.online.requires_grad_(False)
        name = f"DDQN_{seed}_{step}"
        policies[name] = policy
        specifications.append(
            {
                "name": name,
                "checkpoint": str(path.resolve()),
                "sha256": sha256(path),
                "training_config": config,
                "training_config_sha256": hashlib.sha256(
                    json.dumps(config, sort_keys=True).encode()
                ).hexdigest(),
                "training_summary_sha256": sha256(summary_path),
                "training_seed": seed,
                "checkpoint_step": step,
            }
        )
    level_root = Path(__file__).resolve().parents[1] / "levels" / "data"
    selected_levels = levels
    levels = {}
    for name in selected_levels:
        path = level_root / f"{name}.json"
        config = load_level(path)
        levels[name] = {"sha256": sha256(path), "move_limit": config.move_limit}
    configuration = {
        "levels": list(levels),
        "environment_seeds": list(environment_seeds),
        "random_agent_seeds": list(random_agent_seeds),
        "checkpoint_hashes": {item["name"]: item["sha256"] for item in specifications},
        "level_hashes": {name: item["sha256"] for name, item in levels.items()},
    }
    manifest = {
        "schema_version": 1,
        "kind": "zero_shot_development_generalization",
        "evaluation_label": "development evaluation; not final-test evaluation",
        "environment_seeds": list(environment_seeds),
        "random_agent_seeds": list(random_agent_seeds),
        "levels": levels,
        "policies": specifications,
        "benchmark_configuration_sha256": hashlib.sha256(
            json.dumps(configuration, sort_keys=True).encode()
        ).hexdigest(),
        "checkpoint_selection": "User-selected development-validation checkpoints",
        "final_test_used": False,
        "max_cascades": 200,
        "max_initialization_attempts": 1000,
        "planned_episodes": len(levels) * len(environment_seeds) * 5,
        "failure_policy": "Record failures without replacement; abort on integrity failure",
        "win_rate_denominator": "All attempted episodes, including failures",
        "source_sha256": sha256(__file__),
        "package_source_fingerprint": _source_fingerprint(),
        "python_version": platform.python_version(),
        "inference": "CPU, deterministic masked argmax, epsilon=0, one torch thread",
    }
    return manifest, policies


class PairedEnvironment(Match3LevelEnv):
    """Verify the actual reset observation before any policy can act."""

    def __init__(self, level_id, identities, seed):
        super().__init__(level_id, max_cascades=200, max_initialization_attempts=1000)
        self.identities = identities
        self.identity_key = (level_id, seed)
        self.initial_fingerprint = None
        self.pairing_error = False

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.initial_fingerprint = observation_fingerprint(observation)
        previous = self.identities.setdefault(
            self.identity_key, self.initial_fingerprint
        )
        if previous != self.initial_fingerprint:
            self.pairing_error = True
            raise RuntimeError("Initial boards differ across agents")
        return observation, info


def append_record(handle, value):
    handle.write(json.dumps(value, allow_nan=False) + "\n")
    handle.flush()
    os.fsync(handle.fileno())


def write_report(output, records, status, *, expected_count=200):
    summaries = summarize_level_episodes(records)
    lines = [
        "# Cross-level development evaluation",
        "",
        f"Status: {status}",
        "",
        (
            "Zero-shot transfer; selected checkpoints use prior development validation. "
            "These are development results, not independent final-test results."
        ),
        "Failures count in attempted-episode win rates; incomplete groups are not comparable.",
        "",
        f"| Level | Agent | n/{expected_count} | Win rate [Wilson 95% CI] | Mean moves | Objective completion | Blockers / HP remaining | Errors / time limits |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]

    def number(value):
        return "NA" if value is None else f"{value:.3f}"

    by_group = {(row["level_id"], row["agent_name"]): row for row in summaries}
    for row in summaries:
        interval = row["win_rate_ci95"]
        lines.append(
            f"| {row['level_id']} | {row['agent_name']} | "
            f"{row['attempted']}/{expected_count} | "
            f"{row['win_rate']:.1%} [{interval[0]:.1%}, {interval[1]:.1%}] | "
            f"{number(row['mean_moves_used'])} | "
            f"{number(row['mean_objective_completion_ratio'])} | "
            f"{number(row['mean_blockers_remaining'])} / {number(row['mean_blocker_hp_remaining'])} | "
            f"{row['errors']} / {row['time_limits']} |"
        )

    def complete(level, agent):
        row = by_group.get((level, agent))
        return row if row and row["attempted"] == expected_count else None

    def delta(left, right, key, scale=1):
        if left is None or right is None:
            return "NA"
        left_value, right_value = left[key], right[key]
        if left_value is None or right_value is None:
            return "NA"
        return f"{(right_value - left_value) * scale:+.3f}"

    lines.extend(
        [
            "",
            "## Original-v2 comparisons",
            "",
            "Descriptive v2 minus original differences; not paired inferential estimates.",
            "",
            "| Level pair | Agent | Win-rate difference (pp) | Mean-moves difference | Objective-completion difference |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for original in ("wooden-trouble", "double-trouble", "frozen-garden"):
        versioned = f"{original}-v2"
        for agent in sorted({name for level, name in by_group if level == original}):
            left = complete(original, agent)
            right = complete(versioned, agent)
            lines.append(
                f"| {original} -> {versioned} | {agent} | "
                f"{delta(left, right, 'win_rate', 100):s} | "
                f"{delta(left, right, 'mean_moves_used')} | "
                f"{delta(left, right, 'mean_objective_completion_ratio')} |"
            )

    lines.extend(
        [
            "",
            "## DDQN advantages",
            "",
            "Win-rate differences are DDQN minus baseline, in percentage points.",
            "",
            "| Level | DDQN checkpoint | vs Random (pp) | vs Greedy (pp) |",
            "|---|---|---:|---:|",
        ]
    )
    for level in dict.fromkeys(row["level_id"] for row in summaries):
        for agent in sorted(
            name
            for level_id, name in by_group
            if level_id == level and name.startswith("DDQN_")
        ):
            ddqn = complete(level, agent)
            random = complete(level, "Random")
            greedy = complete(level, "Greedy")
            lines.append(
                f"| {level} | {agent} | "
                f"{delta(random, ddqn, 'win_rate', 100)} | "
                f"{delta(greedy, ddqn, 'win_rate', 100)} |"
            )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    (output / "summary.json").write_text(json.dumps(summaries, indent=2))


def run_benchmark(
    results_root,
    output,
    *,
    levels=LEVELS,
    environment_seed_start=710000,
    random_agent_seed_start=720000,
):
    import torch

    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    environment_seeds = tuple(
        range(environment_seed_start, environment_seed_start + 200)
    )
    random_agent_seeds = tuple(
        range(random_agent_seed_start, random_agent_seed_start + 200)
    )
    manifest, policies = prepare(
        results_root,
        levels=levels,
        environment_seeds=environment_seeds,
        random_agent_seeds=random_agent_seeds,
    )
    output.mkdir(parents=True, exist_ok=False)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    before = {name: policy_state(policy) for name, policy in policies.items()}
    identities, records = {}, []
    status = "interrupted"
    active_episode = None
    manifest["torch_version"] = str(torch.__version__)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2))
    experiment_id = sha256(output / "manifest.json")[:24]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        with (output / "episodes.jsonl").open("x") as journal:
            for level in levels:
                for index, seed in enumerate(environment_seeds):
                    agents = {
                        **policies,
                        "Random": RandomAgent(random_agent_seeds[index]),
                        "Greedy": GreedyAgent(),
                    }
                    for name, agent in agents.items():
                        active_episode = {"level": level, "agent": name, "seed": seed}
                        env = PairedEnvironment(level, identities, seed)
                        try:
                            record = run_episode(
                                env,
                                agent,
                                seed=seed,
                                max_steps=manifest["levels"][level]["move_limit"],
                                experiment_id=experiment_id,
                                episode_id=f"{level}:{name}:{seed}",
                                agent_name=name,
                                agent_seed=(
                                    random_agent_seeds[index]
                                    if name == "Random"
                                    else None
                                ),
                            )
                        finally:
                            env.close()
                        records.append(record)
                        append_record(
                            journal,
                            {
                                **asdict(record),
                                "initial_fingerprint": env.initial_fingerprint,
                            },
                        )
                        if env.pairing_error:
                            raise RuntimeError("Initial-board integrity check failed")
                        if name in policies and policy_state(agent) != before[name]:
                            raise RuntimeError("Frozen policy state changed")
            for spec in manifest["policies"]:
                if sha256(spec["checkpoint"]) != spec["sha256"]:
                    raise RuntimeError("Checkpoint file changed")
        status = "complete"
    except BaseException as error:
        (output / "failure.json").write_text(
            json.dumps(
                {
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "completed_records": len(records),
                    "active_episode": active_episode,
                },
                indent=2,
            )
        )
        raise
    finally:
        try:
            write_report(output, records, status)
        finally:
            torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-root", type=Path, default=Path("experiments/results")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--levels", nargs="+", choices=AVAILABLE_LEVELS, default=LEVELS)
    parser.add_argument("--environment-seed-start", type=int, default=710000)
    parser.add_argument("--random-agent-seed-start", type=int, default=720000)
    args = parser.parse_args()
    run_benchmark(
        args.results_root,
        args.output_dir,
        levels=args.levels,
        environment_seed_start=args.environment_seed_start,
        random_agent_seed_start=args.random_agent_seed_start,
    )


if __name__ == "__main__":
    main()
