"""Approved preregistered PPO budget profiles; no algorithm or rollout changes."""

import argparse
import hashlib
import json
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from match3_ai.evaluation.fresh_protocol import historical_inventory, sha256
from match3_ai.training.ppo import PPOTrainingConfig, fingerprint, train_ppo

MASTER_SEEDS = (5918, 5919, 5920)
MILESTONES = (0, 25000, 50000, 75000, 100000)
VALIDATION_SEEDS = tuple(range(300600, 300800))


def experiment_config(seed):
    if type(seed) is not int or seed not in MASTER_SEEDS:
        raise ValueError("master seed must be one of 5918, 5919, 5920")
    return replace(
        PPOTrainingConfig(),
        seed=seed,
        environment_steps=100000,
        evaluation_every=25000,
        evaluation_seed_start=300600,
        evaluation_episodes=200,
    )


def optimized_progress(
    collected, *, final_budget=100000, rollout=256, minibatch=64, epochs=4
):
    full, remainder = divmod(collected, rollout)
    final = collected == final_budget
    optimized = collected if final else full * rollout
    return {
        "collected": collected,
        "optimized": optimized,
        "pending": collected - optimized,
        "rollouts": full + int(final and remainder > 0),
        "optimizer_updates": full * ((rollout + minibatch - 1) // minibatch) * epochs
        + (
            ((remainder + minibatch - 1) // minibatch) * epochs
            if final and remainder
            else 0
        ),
    }


def freeze_protocol(history_root, output_dir):
    from match3_ai.agents.ppo_agent import PPOAgent
    from match3_ai.games.match3.environment import Match3LevelEnv

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    history = historical_inventory(history_root)
    if set(VALIDATION_SEEDS) & set(history["excluded_validation_values"]):
        raise ValueError(
            "approved development-validation seeds have historical overlap; approval required for any change"
        )
    configurations, streams = {}, {}
    for seed in MASTER_SEEDS:
        config = experiment_config(seed)
        env_stream, agent_stream = np.random.SeedSequence(seed).spawn(2)
        agent_seed = int(agent_stream.generate_state(1)[0])
        agent = PPOAgent(agent_seed, config.hidden_dims, config.learning_rate)
        configurations[str(seed)] = asdict(config)
        streams[str(seed)] = {
            "agent_seed": agent_seed,
            "environment_stream": env_stream.state,
            "environment_signature": int(env_stream.generate_state(1)[0]),
            **agent.stream_seeds,
            "initial_weights": fingerprint(agent),
        }
    identifiers = [
        value[name]
        for value in streams.values()
        for name in ("environment_signature", "model", "policy", "shuffle")
    ]
    if (
        len(set(identifiers)) != 12
        or len({s["initial_weights"] for s in streams.values()}) != 3
    ):
        raise ValueError("initial models or seed streams are not distinct")
    env = Match3LevelEnv("blocker-mayhem")
    level_config = asdict(env.config)
    env.close()
    protocol = {
        "version": 1,
        "study": "sprint5_17_ppo_training_budget",
        "created_utc": datetime.now(UTC).isoformat(),
        "master_seeds": list(MASTER_SEEDS),
        "training_partition": [200000, 299999],
        "validation_seeds": list(VALIDATION_SEEDS),
        "reserved_final_test_partition_not_accessed": [400000, 400999],
        "configurations": configurations,
        "streams": streams,
        "level_config": level_config,
        "level_config_id": hashlib.sha256(
            json.dumps(level_config, sort_keys=True).encode()
        ).hexdigest(),
        "milestones": list(MILESTONES),
        "axis": "collected single-environment transitions; optimized and pending transitions separately reported; evaluation not counted as training",
        "expected_progress": [optimized_progress(step) for step in MILESTONES],
        "checkpoint_selection": "absolute 100000-transition final checkpoint is final endpoint; best development-validation checkpoint descriptive only, earliest tie, includes step zero; no external historical 50k cohort as paired control",
        "hypotheses": [
            "mean_75k_minus_50k",
            "mean_100k_minus_50k",
            *[f"seed{seed}_75k_minus_50k" for seed in MASTER_SEEDS],
            *[f"seed{seed}_100k_minus_50k" for seed in MASTER_SEEDS],
        ],
        "statistics": {
            "bootstrap_samples": 20000,
            "bootstrap_seed": 51717,
            "resampling": "one shared validation-board index vector per replicate applied jointly across all policies, training seeds and checkpoints; no training-seed resampling",
            "ci": "two-sided pointwise percentile paired board bootstrap 95%",
            "aggregate_p": "approximate two-sided null-centered joint board bootstrap, plus-one correction",
            "individual_p": "exact two-sided McNemar on paired binary outcomes",
            "multiplicity": "one Holm family of all eight win-rate hypotheses; objective/HP/diagnostic changes descriptive only",
            "scope": "conditional on three observed training seeds; across-training-seed sample SD reported separately, not a large-sample population CI",
        },
        "continuation": "fresh initialization, no exact checkpoint continuation; active engine state and unfinished rollouts are not serialized",
        "failure_policy": "preserve every completed and partial run; no automatic restart, seed replacement, protocol adjustment or silent increase of the 300000-transition budget; unexpected protocol-changing correctness issue requires approval",
        "history_inventory_sha256": hashlib.sha256(
            json.dumps(history, sort_keys=True).encode()
        ).hexdigest(),
        "source_sha256": {
            str(path): sha256(path)
            for path in sorted(Path("src/match3_ai").rglob("*.py"))
        },
    }
    protocol = json.loads(json.dumps(protocol))
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("history_inventory.json", history),
        ("protocol.json", protocol),
    ):
        with (output / name).open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
    with (output / "protocol.sha256").open("x") as handle:
        handle.write(sha256(output / "protocol.json") + "\n")
    return protocol


def load_protocol(path):
    path = Path(path)
    if sha256(path) != path.with_name("protocol.sha256").read_text().strip():
        raise ValueError("frozen budget protocol changed")
    protocol = json.loads(path.read_text())
    history = json.loads(path.with_name("history_inventory.json").read_text())
    if hashlib.sha256(
        json.dumps(history, sort_keys=True).encode()
    ).hexdigest() != protocol["history_inventory_sha256"] or set(
        protocol["validation_seeds"]
    ) & set(history["excluded_validation_values"]):
        raise ValueError("seed provenance changed or overlaps historical data")
    if (
        protocol["master_seeds"] != list(MASTER_SEEDS)
        or protocol["milestones"] != list(MILESTONES)
        or protocol["validation_seeds"] != list(VALIDATION_SEEDS)
    ):
        raise ValueError(
            "budget protocol differs from approved seed/milestone schedule"
        )
    for seed in MASTER_SEEDS:
        if protocol["configurations"][str(seed)] != json.loads(
            json.dumps(asdict(experiment_config(seed)))
        ):
            raise ValueError("hyperparameters differ from approved budget profile")
    if any(
        sha256(name) != digest for name, digest in protocol["source_sha256"].items()
    ):
        raise ValueError("implementation differs from frozen protocol")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--freeze-dir", type=Path)
    parser.add_argument(
        "--history-root", type=Path, default=Path("experiments/results")
    )
    parser.add_argument("--protocol", type=Path)
    parser.add_argument("--master-seed", type=int, choices=MASTER_SEEDS)
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()
    if args.freeze_dir:
        protocol = freeze_protocol(args.history_root, args.freeze_dir)
        print(
            f"Frozen budget protocol: {sha256(args.freeze_dir / 'protocol.json')}; validation {protocol['validation_seeds'][0]}–{protocol['validation_seeds'][-1]}"
        )
    else:
        if args.protocol is None or args.master_seed is None or args.output_dir is None:
            parser.error("training requires --protocol, --master-seed and --output-dir")
        load_protocol(args.protocol)
        summary = train_ppo(experiment_config(args.master_seed), args.output_dir)
        load_protocol(args.protocol)
        print(
            f"Completed seed {args.master_seed}: {summary['environment_steps']} collected transitions",
            flush=True,
        )


if __name__ == "__main__":
    main()
