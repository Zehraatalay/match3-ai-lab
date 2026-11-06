"""Freeze seed provenance, checkpoint identities and a paired evaluation protocol."""

import argparse
import csv
import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def historical_inventory(root):
    """Conservatively exclude validation-range integers and declared seed ranges."""
    used, files = set(), {}

    def integer(value):
        if type(value) is int and 300000 <= value <= 300999:
            used.add(value)

    def walk(value):
        if isinstance(value, dict):
            for start_key, count_key in (
                ("evaluation_seed_start", "evaluation_episodes"),
                ("base_seed", "num_episodes"),
                ("seed_start", "num_episodes"),
            ):
                start, count = value.get(start_key), value.get(count_key)
                if type(start) is int and type(count) is int and 0 < count <= 1000000:
                    used.update(range(max(start, 300000), min(start + count, 301000)))
            for item in value.values():
                walk(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                walk(item)
        else:
            integer(value)

    for path in sorted(Path(root).rglob("*")):
        if not path.is_file() or path.suffix not in (
            ".csv",
            ".json",
            ".jsonl",
            ".npz",
            ".pt",
        ):
            continue
        if path.suffix == ".csv":
            with path.open(newline="") as handle:
                for row in csv.reader(handle):
                    for cell in row:
                        if cell.isdecimal():
                            integer(int(cell))
        elif path.suffix in (".json", ".jsonl"):
            with path.open() as handle:
                if path.suffix == ".json":
                    walk(json.load(handle))
                else:
                    for line in handle:
                        if line.strip():
                            walk(json.loads(line))
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as arrays:
                for name in arrays.files:
                    array = arrays[name]
                    if array.dtype.kind in "iu":
                        used.update(
                            map(
                                int,
                                np.unique(array[(array >= 300000) & (array <= 300999)]),
                            )
                        )
        else:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=True)
            for name in (
                "training_config",
                "training_progress",
                "seed_metadata",
                "validation_metrics",
                "training_state",
            ):
                walk(payload.get(name, {}))
        files[str(path)] = sha256(path)
    return {
        "root": str(root),
        "files_sha256": files,
        "excluded_validation_values": sorted(used),
        "scope": "all local historical JSON/JSONL/CSV/NPZ/checkpoints; validation-range numeric values conservatively excluded, including non-seed values; no claim about unrecorded/external runs",
    }


def policy_descriptors(results_root):
    root = Path(results_root)
    policies = [
        {"name": "Random", "kind": "random"},
        {"name": "Greedy", "kind": "greedy"},
    ]
    for seed in (5901, 5902, 5903):
        policies.append(
            {
                "name": f"DQN{seed}",
                "kind": "dqn",
                "training_seed": seed,
                "step": 50000,
                "checkpoint": str(root / f"sprint5_9_seed_{seed}/checkpoint_final.pt"),
            }
        )
    for seed in (5915, 5916, 5917):
        for phase, step, filename in (
            ("initial", 0, "policy_step_0.pt"),
            ("final", 50000, "checkpoint_final.pt"),
        ):
            policies.append(
                {
                    "name": f"PPO{seed}_{phase}",
                    "kind": "ppo",
                    "training_seed": seed,
                    "step": step,
                    "checkpoint": str(root / f"sprint5_15_ppo_seed_{seed}/{filename}"),
                }
            )
    for policy in policies:
        if "checkpoint" in policy:
            policy["sha256"] = sha256(policy["checkpoint"])
    return policies


def freeze_protocol(history_root, output_dir, *, count=500):
    from match3_ai.games.match3.environment import Match3LevelEnv

    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    inventory = historical_inventory(history_root)
    seeds = sorted(
        set(range(300000, 301000)) - set(inventory["excluded_validation_values"])
    )[:count]
    if len(seeds) != count or count <= 0:
        raise ValueError("insufficient unused validation seeds")
    policies = policy_descriptors(history_root)
    env = Match3LevelEnv(
        "blocker-mayhem", max_cascades=200, max_initialization_attempts=1000
    )
    config = asdict(env.config)
    env.close()
    protocol = {
        "version": 1,
        "study": "sprint5_16_fresh_paired_evaluation",
        "created_utc": datetime.now(UTC).isoformat(),
        "seeds": seeds,
        "policies": policies,
        "level_config": config,
        "level_config_id": hashlib.sha256(
            json.dumps(config, sort_keys=True).encode()
        ).hexdigest(),
        "environment": {
            "level_id": "blocker-mayhem",
            "max_cascades": 200,
            "max_initialization_attempts": 1000,
            "max_steps": 50,
            "reward": "sparse",
        },
        "random_agent_seeds": [615160000 + index for index in range(count)],
        "selection": "all three PPO initial and absolute-budget final checkpoints; all three saved MLP-DQN 50k checkpoints; no best-validation selection",
        "primary": "mean final PPO minus Greedy, per-board difference",
        "secondaries": [
            "mean_final_minus_Random",
            "mean_final_minus_mean_DQN",
            "mean_final_minus_mean_initial",
            *[f"PPO{seed}_final_minus_initial" for seed in (5915, 5916, 5917)],
            *[f"PPO{seed}_final_minus_Greedy" for seed in (5915, 5916, 5917)],
        ],
        "statistics": {
            "bootstrap_samples": 20000,
            "bootstrap_seed": 51616,
            "ci": "two-sided percentile paired board bootstrap 95%, pointwise",
            "primary_test": "null-centered board bootstrap two-sided mean test; approximate, not an exact randomization test",
            "aggregate_secondary_test": "null-centered board bootstrap two-sided mean test",
            "binary_secondary_test": "exact two-sided McNemar",
            "secondary_multiplicity": "Holm across all nine secondary hypotheses only; primary separate",
            "conditional_scope": "board uncertainty conditional on observed training seeds; training-seed sample SD reported separately",
        },
        "critic": "initial-state V versus gamma-discounted terminal return of deterministic greedy actor, gamma=.99; report mean V, target variance, mean(V-return), RMSE and variance-based EV; off-policy behavior agreement, not on-policy critic accuracy; one trajectory per board, no cross-policy EV ranking",
        "representatives": {
            "per_category": 3,
            "selection": "lowest seed in each category; never relax empty categories",
            "ppo_definition": "majority of three final PPO outcomes",
            "categories": [
                "PPO majority win / Greedy loss",
                "Greedy win / PPO majority loss",
                "all eleven win",
                "all eleven lose",
            ],
        },
        "failures": "preserve errors and partial outputs; no seed replacements or exclusions; stop confirmatory analysis unless all 5500 planned episodes complete with valid outcomes and zero invalid actions",
        "source_sha256": {
            str(path): sha256(path)
            for path in sorted(Path("src/match3_ai").rglob("*.py"))
        },
        "history_inventory_sha256": hashlib.sha256(
            json.dumps(inventory, sort_keys=True).encode()
        ).hexdigest(),
    }
    protocol = json.loads(json.dumps(protocol))
    output.mkdir(parents=True, exist_ok=False)
    for name, value in (
        ("history_inventory.json", inventory),
        ("protocol.json", protocol),
    ):
        with (output / name).open("x") as handle:
            json.dump(value, handle, indent=2, allow_nan=False)
    digest = sha256(output / "protocol.json")
    with (output / "protocol.sha256").open("x") as handle:
        handle.write(digest + "\n")
    return protocol


def load_protocol(path, *, verify_sources=True):
    path = Path(path)
    if sha256(path) != path.with_name("protocol.sha256").read_text().strip():
        raise ValueError("frozen protocol digest differs")
    protocol = json.loads(path.read_text())
    seeds = protocol["seeds"]
    if len(set(seeds)) != len(seeds) or any(
        type(s) is not int or not 300000 <= s <= 300999 for s in seeds
    ):
        raise ValueError("evaluation seeds must be unique validation seeds")
    inventory = json.loads(path.with_name("history_inventory.json").read_text())
    if hashlib.sha256(
        json.dumps(inventory, sort_keys=True).encode()
    ).hexdigest() != protocol["history_inventory_sha256"] or set(seeds) & set(
        inventory["excluded_validation_values"]
    ):
        raise ValueError(
            "history inventory differs or seeds overlap historical records"
        )
    if verify_sources and any(
        sha256(name) != digest for name, digest in protocol["source_sha256"].items()
    ):
        raise ValueError("implementation differs from preregistered source")
    for policy in protocol["policies"]:
        if "checkpoint" in policy and sha256(policy["checkpoint"]) != policy["sha256"]:
            raise ValueError("frozen checkpoint digest differs")
    return protocol


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--history-root", type=Path, default=Path("experiments/results")
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = freeze_protocol(args.history_root, args.output_dir)
    print(
        f"Frozen {len(protocol['seeds'])} seeds, {protocol['seeds'][0]}–{protocol['seeds'][-1]}; no evaluation performed"
    )


if __name__ == "__main__":
    main()
