"""Sequential, bounded DDQN/PPO gameplay research on archived calibrated levels.

Workers adapt only their own imported trainer environment factories and seed
metadata. Production trainers, rules, rewards, and packaged levels are unchanged.
PPO continuation is unsupported: larger PPO budgets are independent fresh runs.
"""

import argparse
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import torch

from match3_ai.agents.dqn_agent import DQNAgent
from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.ppo_agent import PPOAgent
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.evaluation.behavior_data import observation_fingerprint
from match3_ai.evaluation.cross_level_benchmark import validate_seeds
from match3_ai.evaluation.experiments import summarize_level_episodes
from match3_ai.evaluation.fresh_protocol import sha256
from match3_ai.evaluation.runner import run_episode
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.loader import load_level
from match3_ai.training import dqn, ppo, seed_policy
from match3_ai.training.overnight import atomic_json

REPO = Path(__file__).resolve().parents[3]
DEFAULT_ROOT = REPO / "experiments/results/overnight_rl_training"
ARCHIVE = REPO / "experiments/results/level_calibration_stage2_confirmation"
VARIANTS = (
    "wooden-trouble__colors5-double-moves6",
    "double-trouble__colors5-moves6",
    "frozen-garden__colors5-moves10",
)
MODULE = "match3_ai.training.calibrated_overnight"
VALIDATION = list(range(810000, 810040))
FINAL = list(range(820000, 820100))
PARTITION = "calibrated_overnight_validation"


def utc() -> str:
    return datetime.now(UTC).isoformat()


def identifier(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()[:20]


def append(path: Path, value: dict) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def seed_plan_valid(training, validation, final) -> bool:
    low, high = training
    return (
        low <= high
        and bool(validation)
        and bool(final)
        and len(set(validation)) == len(validation)
        and len(set(final)) == len(final)
        and not set(validation) & set(final)
        and all(not low <= seed <= high for seed in [*validation, *final])
        and not any(400000 <= seed <= 400999 for seed in [*validation, *final])
    )


def prepare(root: Path, hours: float = 8) -> dict:
    if not 0 < hours <= 8:
        raise ValueError("Wall-clock budget must be positive and at most eight hours")
    if root.exists():
        raise FileExistsError(root)
    if not seed_plan_valid((200000, 299999), VALIDATION, FINAL):
        raise ValueError("Invalid seed separation")
    results = REPO / "experiments/results"
    validate_seeds(
        VALIDATION,
        results,
        random_agent_seeds=tuple(range(830000, 830040)),
        expected_count=40,
    )
    validate_seeds(
        FINAL,
        results,
        random_agent_seeds=tuple(range(831000, 831100)),
        expected_count=100,
    )
    archived = json.loads((ARCHIVE / "manifest.json").read_text())
    configurations = {}
    for variant in VARIANTS:
        spec = archived["configurations"][variant]
        path = ARCHIVE / spec["path"]
        if sha256(path) != spec["sha256"]:
            raise ValueError(f"Archived configuration changed: {variant}")
        configurations[variant] = {**spec, "archived_path": str(path)}
    references = []
    for seed, step, directory in (
        (5990, 125000, "ddqn_seed5990_150k"),
        (5991, 125000, "ddqn_seed5991_150k_retry"),
        (5992, 150000, "ddqn_seed5992_150k"),
    ):
        path = (
            results
            / "sprint5_20_rl_optimization"
            / directory
            / f"policy_step_{step}.pt"
        )
        agent, payload = DQNAgent.load_checkpoint(path)
        if agent.algorithm != "double" or payload["training_step"] != step:
            raise ValueError("Historical DDQN checkpoint provenance mismatch")
        references.append(
            {
                "name": f"historical_DDQN_{seed}_{step}",
                "kind": "ddqn",
                "checkpoint": str(path),
                "sha256": sha256(path),
                "training_config": payload["training_config"],
            }
        )
    path = results / "sprint5_17_ppo_seed_5918/checkpoint_final.pt"
    _, payload = PPOAgent.load_checkpoint(path)
    references.append(
        {
            "name": "historical_PPO_5918",
            "kind": "ppo",
            "checkpoint": str(path),
            "sha256": sha256(path),
            "training_config": payload["training_config"],
        }
    )
    history = {}
    for name in (
        "level_calibration_stage2_confirmation",
        "calibrated_ddqn_development_790000",
        "overnight_rl_research/final_analysis",
    ):
        path = results / name / "summary.json"
        if path.exists():
            history[name] = {
                "path": str(path),
                "sha256": sha256(path),
                "summary": json.loads(path.read_text()),
            }
    curves = []
    for path in sorted(results.glob("sprint5_*/training_summary.json")):
        value = json.loads(path.read_text())
        curves.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "training_config": value.get("training_config"),
                "environment_steps": value.get("environment_steps"),
                "training_seconds": value.get("training_seconds"),
                "evaluation_rounds": value.get("evaluation_rounds"),
            }
        )
    manifest = {
        "schema_version": 1,
        "created_utc": utc(),
        "created_epoch": time.time(),
        "wall_seconds": hours * 3600,
        "evaluation_reserve_seconds": min(3600, hours * 3600 / 4),
        "configurations": configurations,
        "references": references,
        "training_partition": [200000, 299999],
        "validation_seeds": VALIDATION,
        "final_seeds": FINAL,
        "validation_random_start": 830000,
        "final_random_start": 831000,
        "reserved_final_test_used": False,
        "plan": {
            "smoke_steps": 32,
            "screen_steps": 4096,
            "optimization_steps": 8192,
            "extended_steps": 65536,
            "screen": "DDQN and PPO per level, lr=3e-4",
            "optimization": "one lr=1e-4 alternative for each level's validation-leading family",
            "selection": "best validation win rate, then objective completion, then fewer transitions, then deterministic job id",
            "plateau": "skip extension only after >=8192 steps and >=3 validation points when win improvement <=.025, objective improvement <=.01, and best win rate is >=.10 below Greedy",
            "extended": "best candidate per family and level; compatible DDQN full-state continuation, PPO fresh larger run",
        },
        "resources": {
            "workers": 1,
            "torch_threads": 1,
            "nice": 10,
            "replay_capacity": 10000,
        },
        "reward": "unchanged sparse terminal reward: win +1, loss -1, other transitions 0",
        "shared_policy": "unsupported by existing single-level trainers; no explicit level-identity input; level-specific only",
        "a2c": "not implemented; omitted",
        "ppo_resume": "unsupported; fresh runs only",
        "comparability": "Archived Stage 2 and packaged v2 results use different seeds/config identities; historical figures are contextual only. All final comparisons are rerun on identical archived configurations and fresh paired seeds.",
        "source_sha256": {
            str(p.relative_to(REPO)): sha256(p)
            for p in sorted((REPO / "src/match3_ai").rglob("*.py"))
        },
    }
    manifest["experiment_id"] = identifier(
        {
            "configs": configurations,
            "plan": manifest["plan"],
            "validation": VALIDATION,
            "final": FINAL,
        }
    )
    root.mkdir(parents=True)
    for variant, spec in configurations.items():
        destination = root / "configs" / f"{variant}.json"
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(Path(spec["archived_path"]).read_bytes())
    atomic_json(root / "manifest.json", manifest)
    (root / "manifest.sha256").write_text(sha256(root / "manifest.json") + "\n")
    atomic_json(
        root / "history_inventory.json",
        {"baselines": history, "training_curves": curves},
    )
    return manifest


def verify(root: Path) -> dict:
    if sha256(root / "manifest.json") != (root / "manifest.sha256").read_text().strip():
        raise ValueError("Experiment manifest hash mismatch")
    manifest = json.loads((root / "manifest.json").read_text())
    expected_sources = dict(manifest["source_sha256"])
    amendment_path = root / "driver_amendment.json"
    if amendment_path.exists():
        amendment = json.loads(amendment_path.read_text())
        name = "src/match3_ai/training/calibrated_overnight.py"
        if (
            amendment["parent_manifest_sha256"] != sha256(root / "manifest.json")
            or amendment["previous_driver_sha256"] != expected_sources[name]
            or sha256(root / "frozen_driver_before_amendment.py")
            != expected_sources[name]
        ):
            raise ValueError("Driver amendment provenance mismatch")
        expected_sources[name] = amendment["replacement_driver_sha256"]
    for name, expected in expected_sources.items():
        if sha256(REPO / name) != expected:
            raise ValueError(f"Source changed since preparation: {name}")
    for variant, spec in manifest["configurations"].items():
        if sha256(root / "configs" / f"{variant}.json") != spec["sha256"]:
            raise ValueError("Frozen calibrated level changed")
    for ref in manifest["references"]:
        if sha256(ref["checkpoint"]) != ref["sha256"]:
            raise ValueError("Historical checkpoint changed")
    return manifest


def training_spec(
    variant: str, kind: str, stage: str, steps: int, lr=3e-4, resume=None
) -> dict:
    level = variant.split("__")[0]
    limit = 10 if level == "frozen-garden" else 6
    common = {
        "level_id": level,
        "environment_steps": steps,
        "seed": 8600 + VARIANTS.index(variant),
        "evaluation_seed_start": VALIDATION[0],
        "evaluation_episodes": 2 if stage == "smoke" else 40,
        "evaluation_every": 16 if stage == "smoke" else 2048,
        "evaluation_partition": PARTITION,
        "max_episode_steps": limit,
        "learning_rate": lr,
    }
    if kind == "ddqn":
        config = asdict(
            dqn.TrainingConfig(
                **common,
                train_seed_start=200000,
                evaluation_agent_seed_start=830000,
                architecture="dueling",
                algorithm="double",
                replay_capacity=10000,
                batch_size=16 if stage == "smoke" else 64,
                warmup_steps=8 if stage == "smoke" else 256,
                target_sync_updates=250,
                epsilon_decay_steps=25000,
                evaluate_step_zero=True,
                baseline_every_checkpoint=False,
            )
        )
    else:
        seed_policy.STUDY_VALIDATION_PARTITIONS[PARTITION] = (810000, 810039)
        config = asdict(
            ppo.PPOTrainingConfig(
                **common,
                rollout_steps=16 if stage == "smoke" else 256,
                minibatch_size=16 if stage == "smoke" else 64,
            )
        )
    spec = {"variant": variant, "kind": kind, "stage": stage, "config": config}
    if resume:
        spec["resume"] = resume
    spec["id"] = f"{stage}_{kind}_{identifier(spec)}"
    return spec


def resume_compatible(payload: dict, config: dict, level_hash: str) -> bool:
    previous, current = dict(payload.get("training_config", {})), dict(config)
    previous.pop("environment_steps", None)
    current.pop("environment_steps", None)
    state = payload.get("training_state", {})
    return (
        json.loads(json.dumps(previous)) == json.loads(json.dumps(current))
        and bool(payload.get("optimizer", {}).get("state"))
        and bool(state.get("replay"))
        and state.get("level_config_id") == level_hash
        and payload.get("training_step", 0) < config["environment_steps"]
    )


def adapted_environment(config, level_config):
    return Match3LevelEnv(
        config.level_id,
        level_config=level_config,
        max_cascades=config.max_cascades,
        max_initialization_attempts=config.max_initialization_attempts,
    )


def worker(root: Path, attempt: Path) -> None:
    lock = (root / "training_worker.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = verify(root)
    spec = json.loads((attempt / "spec.json").read_text())
    level = load_level(root / "configs" / f"{spec['variant']}.json")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    os.nice(10)
    signal.signal(
        signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("graceful stop")),
    )
    signal.signal(
        signal.SIGALRM,
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("worker wall-clock limit")),
    )
    limit = manifest["created_epoch"] + manifest["wall_seconds"]
    if spec["kind"] != "final":
        limit -= manifest["evaluation_reserve_seconds"]
    if time.time() >= limit:
        raise RuntimeError("Worker deadline expired")
    signal.setitimer(signal.ITIMER_REAL, max(0.01, limit - time.time()))
    start = time.monotonic()

    def observe(event, data):
        clean = {k: v for k, v in data.items() if k != "collector"}
        append(attempt / "events.jsonl", {"utc": utc(), "event": event, **clean})
        atomic_json(
            attempt / "progress.json",
            {
                "utc": utc(),
                "event": event,
                "environment_step": data.get("environment_step"),
                "optimizer_updates": data.get("optimizer_updates"),
            },
        )

    try:
        if spec["kind"] in ("baseline", "final"):
            result = evaluate(root, attempt, spec, manifest, level)
        else:
            seed_policy.STUDY_VALIDATION_PARTITIONS[PARTITION] = (810000, 810039)
            dqn._environment = lambda config: adapted_environment(config, level)
            ppo.environment = lambda config: adapted_environment(config, level)
            config = dict(spec["config"])
            config["hidden_dims"] = tuple(config["hidden_dims"])
            probe_env = adapted_environment(type("Config", (), config), level)
            try:
                observation, info = probe_env.reset(seed=VALIDATION[0])
                if not probe_env.observation_space.contains(observation):
                    raise ValueError("Observation is outside Gymnasium space")
                atomic_json(
                    attempt / "environment_contract.json",
                    {
                        "config_sha256": manifest["configurations"][spec["variant"]][
                            "sha256"
                        ],
                        "level_config_id": info["level_config_id"],
                        "action_space": str(probe_env.action_space),
                        "observation_space": str(probe_env.observation_space),
                        "reward": manifest["reward"],
                    },
                )
            finally:
                probe_env.close()
            resume = spec.get("resume")
            resume_decision = "fresh run"
            if resume:
                if spec["kind"] != "ddqn" or sha256(resume["path"]) != resume["sha256"]:
                    raise ValueError("Unsupported or changed resume checkpoint")
                _, payload = DQNAgent.load_checkpoint(resume["path"])
                if not resume_compatible(payload, config, info["level_config_id"]):
                    observe(
                        "resume_rejected",
                        {
                            "reason": "optimizer/replay/configuration mismatch; starting fresh"
                        },
                    )
                    resume = None
                    resume_decision = "incompatible continuation rejected; fresh run"
                else:
                    resume_decision = "verified full-state DDQN continuation"
            if spec["kind"] == "ddqn":
                summary = dqn.train_dqn(
                    dqn.TrainingConfig(**config),
                    attempt / "training",
                    resume_from=resume["path"] if resume else None,
                    observer=observe,
                )
                rounds = [
                    {
                        "step": row["environment_step"],
                        **next(
                            r for r in row["summary"] if r["agent_name"] == "DQNAgent"
                        ),
                    }
                    for row in summary["evaluation_rounds"]
                ]
            else:
                summary = ppo.train_ppo(
                    ppo.PPOTrainingConfig(**config),
                    attempt / "training",
                    observer=observe,
                )
                rounds = [
                    {"step": row["environment_step"], **row}
                    for row in summary["evaluation_rounds"]
                ]
            best = max(
                rounds,
                key=lambda row: (
                    row["win_rate"],
                    row["mean_objective_completion_ratio"] or 0,
                    -row["step"],
                ),
            )
            checkpoint = attempt / "training" / f"policy_step_{best['step']}.pt"
            if any(row["errors"] or row["time_limits"] for row in rounds):
                raise ValueError("Training validation contains invalid episodes")
            result = {
                "best_validation": best,
                "curve": rounds,
                "checkpoint": str(checkpoint),
                "checkpoint_sha256": sha256(checkpoint),
                "final_checkpoint": str(attempt / "training/checkpoint_final.pt"),
                "environment_steps": summary["environment_steps"],
                "optimizer_updates": summary["optimizer_updates"],
                "checkpoint_inference_valid": summary["checkpoint_inference_valid"],
                "resume_decision": resume_decision,
                "stability": {
                    key: summary[key]
                    for key in (
                        "finite_losses",
                        "invalid_actions",
                        "evaluation_invalid_actions",
                        "optimization_statistics",
                    )
                    if key in summary
                },
            }
        atomic_json(
            attempt / "result.json",
            {
                "status": "complete",
                "spec": spec,
                "seconds": time.monotonic() - start,
                "finished_utc": utc(),
                **result,
            },
        )
    except BaseException as error:
        atomic_json(
            attempt / "failure.json",
            {
                "status": "interrupted"
                if isinstance(error, KeyboardInterrupt)
                else "failed",
                "error": str(error),
                "type": type(error).__name__,
                "traceback": traceback.format_exc(),
                "seconds": time.monotonic() - start,
                "utc": utc(),
            },
        )
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        lock.close()


def evaluate(root, attempt, spec, manifest, level):
    final = spec["kind"] == "final"
    seeds = manifest["final_seeds"] if final else manifest["validation_seeds"]
    random_start = (
        manifest["final_random_start"] if final else manifest["validation_random_start"]
    )
    agents = [("Random", "random", None), ("Greedy", "greedy", None)]
    if final:
        agents += [(row["name"], row["kind"], row) for row in spec["agents"]]
    records, identities = [], {}
    for name, kind, ref in agents:
        policy = None
        if ref:
            if sha256(ref["checkpoint"]) != ref["sha256"]:
                raise ValueError("Evaluation checkpoint hash mismatch")
            policy, _ = (DQNAgent if kind == "ddqn" else PPOAgent).load_checkpoint(
                ref["checkpoint"]
            )
        for index, seed in enumerate(seeds):
            agent = policy or (
                RandomAgent(random_start + index) if kind == "random" else GreedyAgent()
            )
            env = Match3LevelEnv(level.level_id, level_config=level)
            initial = env.reset

            def reset(*, seed=None, options=None, original=initial):
                observation, info = original(seed=seed, options=options)
                identity = (
                    observation_fingerprint(observation),
                    info["board_seed"],
                    info["engine_seed"],
                )
                if identities.setdefault(seed, identity) != identity:
                    raise ValueError("Paired evaluation board mismatch")
                return observation, info

            env.reset = reset
            try:
                record = run_episode(
                    env,
                    agent,
                    seed=seed,
                    max_steps=level.move_limit,
                    experiment_id=manifest["experiment_id"],
                    agent_name=name,
                    agent_seed=random_start + index if kind == "random" else None,
                )
                records.append(record)
                append(
                    attempt / "episodes.jsonl",
                    {
                        **asdict(record),
                        "initial_identity": identities.get(seed),
                        "variant_id": spec["variant"],
                    },
                )
                if record.invalid_actions or record.outcome not in ("WON", "LOST"):
                    raise ValueError("Evaluation integrity failure; record preserved")
            finally:
                env.close()
            atomic_json(
                attempt / "progress.json",
                {"utc": utc(), "agent": name, "episodes": len(records)},
            )
    return {
        "summary": summarize_level_episodes(records),
        "episodes": len(records),
        "paired_boards_verified": True,
    }


def cached_result(root: Path, spec: dict):
    for path in sorted((root / "jobs" / spec["id"]).glob("attempt_*/result.json")):
        result = json.loads(path.read_text())
        if result["status"] == "complete":
            if json.loads(json.dumps(result["spec"])) != json.loads(json.dumps(spec)):
                raise ValueError("Job id reused with different specification")
            return result
    return None


def can_start(now: float, deadline: float, reserve: float, estimate: float) -> bool:
    return now + reserve + estimate < deadline


def execute(root: Path, spec: dict, cutoff: float, estimate=60, retry=True):
    previous = cached_result(root, spec)
    if previous:
        return previous
    attempts = root / "jobs" / spec["id"]
    attempts.mkdir(parents=True, exist_ok=True)
    for _ in range(2 if retry else 1):
        if time.time() + estimate >= cutoff or (root / "STOP").exists():
            return None
        number = len(list(attempts.glob("attempt_*"))) + 1
        attempt = attempts / f"attempt_{number:03d}"
        attempt.mkdir(exist_ok=False)
        atomic_json(attempt / "spec.json", spec)
        with (attempt / "worker.log").open("x") as log:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-u",
                    "-m",
                    MODULE,
                    "worker",
                    "--root",
                    str(root),
                    "--attempt",
                    str(attempt),
                ],
                cwd=REPO,
                stdout=log,
                stderr=subprocess.STDOUT,
                env={
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                },
            )
            atomic_json(
                root / "active_job.json",
                {
                    "pid": process.pid,
                    "spec": spec,
                    "attempt": str(attempt),
                    "utc": utc(),
                },
            )
            append(
                root / "scheduler.jsonl",
                {
                    "event": "started",
                    "job": spec["id"],
                    "pid": process.pid,
                    "utc": utc(),
                },
            )
            while process.poll() is None:
                if time.time() >= cutoff or (root / "STOP").exists():
                    process.terminate()
                    try:
                        process.wait(timeout=15)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    break
                time.sleep(1)
        result = cached_result(root, spec)
        if result is None and not (attempt / "failure.json").exists():
            atomic_json(
                attempt / "failure.json",
                {
                    "status": "failed",
                    "utc": utc(),
                    "error": "Worker exited without a completed result",
                    "exit_code": process.returncode,
                },
            )
        append(
            root / "scheduler.jsonl",
            {
                "event": "finished" if result else "failed",
                "job": spec["id"],
                "code": process.returncode,
                "utc": utc(),
            },
        )
        if result:
            return result
        if (root / "STOP").exists():
            return None
    return None


def selection_key(result):
    score = result["best_validation"]
    return (
        score["win_rate"],
        score["mean_objective_completion_ratio"] or 0,
        -result["environment_steps"],
        result["spec"]["id"],
    )


def plateau(result, greedy_rate):
    curve = result["curve"]
    if result["environment_steps"] < 8192 or len(curve) < 3:
        return False
    return (
        max(r["win_rate"] for r in curve) <= curve[0]["win_rate"] + 0.025
        and max(r["mean_objective_completion_ratio"] or 0 for r in curve)
        <= (curve[0]["mean_objective_completion_ratio"] or 0) + 0.01
        and result["best_validation"]["win_rate"] <= greedy_rate - 0.10
    )


def aggregate(root: Path, status: str) -> None:
    results, failures = [], []
    for path in sorted((root / "jobs").glob("*/attempt_*/result.json")):
        results.append({"path": str(path), **json.loads(path.read_text())})
    for path in sorted((root / "jobs").glob("*/attempt_*/failure.json")):
        failures.append({"path": str(path), **json.loads(path.read_text())})
    atomic_json(
        root / "summary.json",
        {
            "status": status,
            "updated_utc": utc(),
            "results": results,
            "failures": failures,
        },
    )
    lines = [
        "# Calibrated overnight gameplay training",
        "",
        f"Status: {status}",
        "",
        "Validation selects policies; held-out evaluation does not feed selection. Reserved final-test seeds 400000–400999 are untouched.",
        "",
        "| Variant | Run | Transitions | Seconds | Best validation win rate | Checkpoint |",
        "|---|---|---:|---:|---:|---|",
    ]
    for result in results:
        if "best_validation" in result:
            lines.append(
                f"| {result['spec']['variant']} | {result['spec']['id']} | {result['environment_steps']} | {result['seconds']:.1f} | {result['best_validation']['win_rate']:.1%} | {result['checkpoint']} |"
            )
    lines += [
        "",
        "## Paired held-out evaluation",
        "",
        "| Variant | Agent | n | Win rate [Wilson 95%] | Reward | Moves | Objective completion |",
        "|---|---|---:|---|---:|---:|---:|",
    ]
    final_rows = []
    for result in results:
        if result["spec"]["kind"] == "final":
            for row in result["summary"]:
                interval = row["win_rate_ci95"]
                final_rows.append({"variant_id": result["spec"]["variant"], **row})
                lines.append(
                    f"| {result['spec']['variant']} | {row['agent_name']} | {row['attempted']} | {row['win_rate']:.1%} [{interval[0]:.1%}, {interval[1]:.1%}] | {row['mean_total_reward']:.3f} | {row['mean_moves_used']:.2f} | {row['mean_objective_completion_ratio']:.3f} |"
                )
    atomic_json(root / "final_comparison.json", final_rows)
    lines += [
        "",
        f"Failed/interrupted attempts: {len(failures)}. Detailed records and training diagnostics remain in each attempt directory.",
        "",
        "One training seed per level/family; selected development checkpoints and small validation samples limit conclusions. PPO larger runs restart rather than resume. No shared agent or new architecture was introduced.",
    ]
    (root / "report.md").write_text("\n".join(lines) + "\n")


def supervise(root: Path, smoke_only=False) -> None:
    manifest = verify(root)
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        deadline = manifest["created_epoch"] + manifest["wall_seconds"]
        cutoff = deadline - manifest["evaluation_reserve_seconds"]
        atomic_json(
            root / "controller.json",
            {
                "pid": os.getpid(),
                "utc": utc(),
                "deadline_epoch": deadline,
                "status": "running",
            },
        )
        signal.signal(signal.SIGTERM, lambda *_: (root / "STOP").touch())
        signal.signal(signal.SIGINT, lambda *_: (root / "STOP").touch())
        status = "interrupted"
        try:
            for variant in VARIANTS:
                for kind in ("ddqn", "ppo"):
                    if not execute(
                        root,
                        training_spec(variant, kind, "smoke", 32),
                        cutoff,
                        retry=False,
                    ):
                        raise RuntimeError(
                            "Critical DDQN/PPO smoke failed; full experiment blocked"
                        )
            if smoke_only:
                atomic_json(root / "smoke_passed.json", {"utc": utc(), "jobs": 6})
                status = "smoke_passed"
                return
            if not (root / "smoke_passed.json").exists():
                raise RuntimeError(
                    "Separate reviewed smoke gate is required before launch"
                )
            candidates, baselines = [], {}
            for variant in VARIANTS:
                spec = {
                    "id": f"baseline_{identifier(variant)}",
                    "kind": "baseline",
                    "variant": variant,
                }
                baseline = execute(root, spec, cutoff)
                if baseline is None:
                    raise RuntimeError("Paired validation baselines failed")
                baselines[variant] = baseline
                for kind in ("ddqn", "ppo"):
                    result = execute(
                        root, training_spec(variant, kind, "screen", 4096), cutoff
                    )
                    if result:
                        candidates.append(result)
                aggregate(root, "screening")
            for variant in VARIANTS:
                choices = [r for r in candidates if r["spec"]["variant"] == variant]
                if not choices:
                    continue
                winner = max(choices, key=selection_key)
                result = execute(
                    root,
                    training_spec(
                        variant, winner["spec"]["kind"], "optimize", 8192, lr=1e-4
                    ),
                    cutoff,
                    estimate=max(60, winner["seconds"] * 2.5),
                )
                if result:
                    candidates.append(result)
            for variant in VARIANTS:
                greedy = next(
                    row["win_rate"]
                    for row in baselines[variant]["summary"]
                    if row["agent_name"] == "Greedy"
                )
                for kind in ("ddqn", "ppo"):
                    choices = [
                        r
                        for r in candidates
                        if r["spec"]["variant"] == variant and r["spec"]["kind"] == kind
                    ]
                    if not choices:
                        continue
                    winner = max(choices, key=selection_key)
                    if plateau(winner, greedy):
                        append(
                            root / "scheduler.jsonl",
                            {
                                "event": "validation_plateau",
                                "job": winner["spec"]["id"],
                                "utc": utc(),
                            },
                        )
                        continue
                    estimate = (
                        winner["seconds"] / winner["environment_steps"] * 65536 * 1.3
                    )
                    if not can_start(time.time(), cutoff, 0, estimate):
                        continue
                    resume = None
                    if kind == "ddqn":
                        path = winner["final_checkpoint"]
                        resume = {"path": path, "sha256": sha256(path)}
                    spec = training_spec(
                        variant,
                        kind,
                        "extended",
                        65536,
                        lr=winner["spec"]["config"]["learning_rate"],
                        resume=resume,
                    )
                    result = execute(root, spec, cutoff, estimate=estimate)
                    if result:
                        candidates.append(result)
                    aggregate(root, "extended_training")
            if (root / "STOP").exists():
                status = "stopped"
                return
            selection_path = root / "frozen_final_selection.json"
            if selection_path.exists():
                selected = json.loads(selection_path.read_text())
            else:
                selected = {}
                for variant in VARIANTS:
                    rows = []
                    for kind in ("ddqn", "ppo"):
                        choices = [
                            r
                            for r in candidates
                            if r["spec"]["variant"] == variant
                            and r["spec"]["kind"] == kind
                        ]
                        if choices:
                            best = max(choices, key=selection_key)
                            rows.append(
                                {
                                    "name": f"selected_{kind}",
                                    "kind": kind,
                                    "checkpoint": best["checkpoint"],
                                    "sha256": best["checkpoint_sha256"],
                                    "selected_job": best["spec"]["id"],
                                }
                            )
                    selected[variant] = rows + manifest["references"]
                atomic_json(selection_path, selected)
            completed_final = 0
            for variant in VARIANTS:
                final_spec = {
                    "id": f"final_{identifier([variant, selected[variant]])}",
                    "kind": "final",
                    "variant": variant,
                    "agents": selected[variant],
                }
                if execute(root, final_spec, deadline - 30, retry=False):
                    completed_final += 1
                aggregate(root, "held_out_evaluation")
            status = (
                "stopped"
                if (root / "STOP").exists()
                else "complete"
                if completed_final == 3
                else "incomplete_evaluation"
            )
        except BaseException as error:
            atomic_json(
                root / f"controller_failure_{time.time_ns()}.json",
                {
                    "utc": utc(),
                    "error": str(error),
                    "traceback": traceback.format_exc(),
                },
            )
            raise
        finally:
            aggregate(root, status)
            atomic_json(
                root / "controller.json",
                {
                    "pid": os.getpid(),
                    "utc": utc(),
                    "deadline_epoch": deadline,
                    "status": status,
                },
            )


def launch(root: Path) -> dict:
    launch_lock = (root / "launch.lock").open("a")
    fcntl.flock(launch_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    manifest = verify(root)
    if not (root / "smoke_passed.json").exists():
        raise RuntimeError("Smoke tests must pass before detached launch")
    if time.time() >= manifest["created_epoch"] + manifest["wall_seconds"]:
        raise RuntimeError("Original eight-hour budget has expired")
    with (root / "controller.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (root / "training_worker.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (root / "STOP").exists():
        (root / "STOP").unlink()
    with (root / "controller.log").open("a") as log:
        process = subprocess.Popen(
            [sys.executable, "-u", "-m", MODULE, "run", "--root", str(root)],
            cwd=REPO,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
    caffeinate = subprocess.Popen(
        ["/usr/bin/caffeinate", "-i", "-w", str(process.pid)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    value = {
        "pid": process.pid,
        "caffeinate_pid": caffeinate.pid,
        "launched_utc": utc(),
        "log": str(root / "controller.log"),
        "root": str(root),
    }
    atomic_json(root / "launch.json", value)
    for _ in range(100):
        path = root / "controller.json"
        if path.exists() and json.loads(path.read_text()).get("pid") == process.pid:
            break
        if process.poll() is not None:
            raise RuntimeError("Detached controller failed; inspect controller.log")
        time.sleep(0.1)
    else:
        raise RuntimeError("Detached controller did not acknowledge startup")
    launch_lock.close()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=(
            "prepare",
            "smoke",
            "launch",
            "resume",
            "run",
            "worker",
            "status",
            "stop",
        ),
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--hours", type=float, default=8)
    parser.add_argument("--attempt", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command == "prepare":
        print(json.dumps(prepare(root, args.hours), indent=2))
    elif args.command == "smoke":
        supervise(root, smoke_only=True)
    elif args.command in ("launch", "resume"):
        print(json.dumps(launch(root), indent=2))
    elif args.command == "run":
        supervise(root)
    elif args.command == "worker":
        worker(root, args.attempt)
    elif args.command == "stop":
        (root / "STOP").touch()
        print("Graceful stop requested; current worker will preserve partial artifacts")
    else:
        for name in ("launch.json", "controller.json", "active_job.json"):
            path = root / name
            if path.exists():
                print(name, path.read_text())


if __name__ == "__main__":
    main()
