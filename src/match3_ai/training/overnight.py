"""Frozen, bounded overnight research execution with persistent process tracking."""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from match3_ai.agents.dqn_agent import DQNAgent
from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.ppo_agent import PPOAgent
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.evaluation.analytics import AnalyticsCollector, export_evaluation
from match3_ai.evaluation.experiments import summarize_level_episodes
from match3_ai.evaluation.fresh_protocol import sha256
from match3_ai.evaluation.runner import run_episode
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.training.dqn import TrainingConfig, train_dqn
from match3_ai.training.ppo import PPOTrainingConfig, train_ppo

ROOT = Path("experiments/results/overnight_rl_research")
BOARDS = list(range(300800, 300999))


@dataclass(frozen=True)
class OvernightPPOConfig(PPOTrainingConfig):
    learning_rate_schedule: str = "constant"


def late_learning_rate(collected):
    return 0.0003 * (1 - 0.9 * min(1.0, max(0.0, (collected - 50000) / 50000)))


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(temporary, path)


def verify_manifest(root):
    root = Path(root)
    path = root / "manifest.json"
    if sha256(path) != (root / "manifest.sha256").read_text().strip():
        raise ValueError("frozen overnight manifest changed")
    manifest = json.loads(path.read_text())
    expected = dict(manifest["source_sha256"])
    amendment_path = root / "operational_amendment_001.json"
    if amendment_path.exists():
        if (
            sha256(amendment_path)
            != amendment_path.with_suffix(".sha256").read_text().strip()
        ):
            raise ValueError("operational amendment changed")
        amendment = json.loads(amendment_path.read_text())
        if amendment["parent_manifest_sha256"] != sha256(path):
            raise ValueError("amendment does not identify frozen manifest")
        for name, replacement in amendment["replacement_source_sha256"].items():
            if (
                name != "src/match3_ai/training/overnight.py"
                or amendment["original_source_sha256"][name] != expected[name]
            ):
                raise ValueError("amendment exceeds logging-only driver scope")
            if sha256(root / "frozen_source" / name) != expected[name]:
                raise ValueError("original driver archive differs from frozen source")
            expected[name] = replacement
    if any(sha256(n) != d for n, d in expected.items()):
        raise ValueError("production source differs from frozen manifest")
    for spec in manifest["runs"]:
        if (
            spec.get("checkpoint")
            and sha256(spec["checkpoint"]) != spec["checkpoint_sha256"]
        ):
            raise ValueError("frozen historical checkpoint changed")
    return manifest


def freeze(root=ROOT):
    root = Path(root)
    if (root / "manifest.json").exists():
        raise FileExistsError(root / "manifest.json")
    history = json.loads((root / "history_inventory.json").read_text())
    if set(BOARDS) & set(history["excluded_validation_values"]):
        raise ValueError("overnight boards have recorded provenance conflicts")
    if any(sha256(n) != d for n, d in history["files_sha256"].items()):
        raise ValueError("historical artifacts changed at preflight")
    runs = []
    for mode in ("constant", "late_linear"):
        for seed in (5930, 5931, 5932):
            config = OvernightPPOConfig(
                seed=seed,
                environment_steps=100000,
                evaluation_every=50000,
                evaluation_seed_start=BOARDS[0],
                evaluation_episodes=len(BOARDS),
                learning_rate_schedule=mode,
            )
            runs.append(
                {
                    "id": f"a_ppo_{mode}_{seed}",
                    "stage": "A",
                    "kind": "ppo",
                    "config": asdict(config),
                }
            )
    order = [
        ("constant", 5930),
        ("late_linear", 5931),
        ("constant", 5932),
        ("late_linear", 5930),
        ("constant", 5931),
        ("late_linear", 5932),
    ]
    runs.sort(
        key=lambda r: order.index(
            (r["config"]["learning_rate_schedule"], r["config"]["seed"])
        )
    )
    for stage, architecture in (("B", "mlp"), ("C", "dueling")):
        smoke = TrainingConfig(
            seed=5960 if stage == "B" else 5961,
            environment_steps=512,
            train_seed_start=200000,
            warmup_steps=64,
            evaluation_seed_start=BOARDS[0],
            evaluation_episodes=len(BOARDS),
            evaluation_every=512,
            evaluation_agent_seed_start=619000000,
            evaluate_step_zero=True,
            baseline_every_checkpoint=False,
            seed_policy="sprint5_8",
            algorithm="double",
            architecture=architecture,
        )
        runs.append(
            {
                "id": f"{stage.lower()}_smoke",
                "stage": stage + "_smoke",
                "kind": "dqn",
                "config": asdict(smoke),
            }
        )
        for index, seed in enumerate((5950, 5951, 5952)):
            config = TrainingConfig(
                seed=seed,
                environment_steps=50000,
                train_seed_start=200000 + index * 20000,
                evaluation_seed_start=BOARDS[0],
                evaluation_episodes=len(BOARDS),
                evaluation_every=25000,
                evaluation_agent_seed_start=619000000,
                evaluate_step_zero=True,
                baseline_every_checkpoint=False,
                seed_policy="sprint5_8",
                algorithm="double",
                architecture=architecture,
            )
            runs.append(
                {
                    "id": f"{stage.lower()}_{architecture}_{seed}",
                    "stage": stage,
                    "kind": "dqn",
                    "config": asdict(config),
                }
            )
    for name in ("random", "greedy"):
        runs.append(
            {
                "id": f"baseline_{name}",
                "stage": "baselines",
                "kind": name,
                "config": {"environment_steps": 0},
            }
        )
    for family, seeds in (
        ("vanilla", (5901, 5902, 5903)),
        ("historical_ppo", (5918, 5919, 5920)),
    ):
        for seed in seeds:
            path = Path(
                f"experiments/results/sprint5_{9 if family == 'vanilla' else 17}_{'seed' if family == 'vanilla' else 'ppo_seed'}_{seed}/checkpoint_final.pt"
            )
            runs.append(
                {
                    "id": f"baseline_{family}_{seed}",
                    "stage": "baselines",
                    "kind": family,
                    "config": {"environment_steps": 0},
                    "checkpoint": str(path),
                    "checkpoint_sha256": sha256(path),
                }
            )
    manifest = {
        "study": "overnight_rl_research",
        "created_utc": datetime.now(UTC).isoformat(),
        "deadline_utc": "2026-11-06T04:18:34+00:00",
        "training_cutoff_utc": "2026-11-06T03:18:34+00:00",
        "runs": runs,
        "validation_seeds": BOARDS,
        "training_partition": [200000, 299999],
        "final_test_partition_not_accessed": [400000, 400999],
        "reward": "unchanged sparse terminal -1/0/+1; PBRS disabled",
        "ppo_lr_formula": ".0003*(1-.9*clip((collected_rollout_completion-50000)/50000,0,1)); same rate for every epoch in rollout",
        "dqn_target": "masked online argmax, target-network evaluation; termination reward-only; truncation bootstraps final observation",
        "dueling": "same MLP trunk, scalar value and 144 advantages; subtract mean of all 144 advantages, mask decisions and targets externally; 129 extra parameters",
        "primary": {
            "comparison": "mean paired PPO late_linear minus constant at fixed 100k",
            "family_size": 1,
            "test": "approximate null-centered paired shared-board bootstrap, plus-one p; Holm singleton",
            "alpha": 0.05,
            "scope": "conditional on three observed training pairs, not population-level algorithm superiority",
            "ci": "pointwise paired board percentile 95%; across-training-pair SD and all seed effects separate",
            "practical_gain": 0.05,
        },
        "secondary": {
            "family": "three per-seed PPO effects, mean dueling minus standard DDQN and its three per-seed effects; Holm seven exploratory p-values",
            "individual_test": "exact paired McNemar",
            "cohort_test": "null-centered paired board bootstrap",
            "historical": "historically trained Vanilla DQN and PPO re-evaluated on same boards, descriptive only; unmatched training cohorts",
        },
        "bootstrap": {
            "samples": 20000,
            "seed": 51919,
            "resampling": "one common board vector across all models and checkpoints",
        },
        "seed_uncertainty": "three pairs cannot establish population improvement; exact paired sign-flip sensitivity p minimum .25 reported separately; no significance claim from optimizer records",
        "runtime": {
            "max_parallel": 3,
            "torch_threads": 1,
            "memory_limit_rationale": "16GiB host; bounded replay 50000; at most three workers",
            "stage_A_estimate_minutes": 74,
            "stage_B_estimate_minutes": 25,
            "stage_C_estimate_minutes": 25,
            "smokes_baselines_estimate_minutes": 20,
            "implementation_tests_allowance_minutes": 60,
            "finalization_reserved_minutes": 60,
            "historical_ppo_process_seconds_mean": 2195.2,
            "historical_dqn_process_seconds_mean": 1056.7,
            "budgets_reduced": False,
        },
        "failure_policy": "preserve failed/partial runs and logs; no automatic retries or seed replacements; diagnose first; major correctness/methodology changes require approval",
        "stage_C_pilot_gate": "run only if B completes validly and estimated wave fits before training cutoff, independent of observed win rates",
        "stage_D": "not preregistered for execution; optional intervention deferred unless A-C complete, diagnostics support it and time permits a separate frozen exploratory addendum before any D training",
        "partial_progress": "status JSON atomically replaced; append-only fsynced JSONL events; live PPO optimizer/rollout records and milestone episode/move exports; DQN flushed metrics and milestone exports",
        "source_sha256": {
            str(p): sha256(p) for p in sorted(Path("src/match3_ai").rglob("*.py"))
        },
        "history_inventory_sha256": sha256(root / "history_inventory.json"),
        "seed_provenance_sha256": sha256(root / "seed_provenance.json"),
        "environment_config_sha256": sha256(
            "src/match3_ai/levels/data/blocker-mayhem.json"
        ),
    }
    manifest = json.loads(json.dumps(manifest))
    with (root / "manifest.json").open("x") as f:
        json.dump(manifest, f, indent=2, allow_nan=False)
    (root / "manifest.sha256").write_text(sha256(root / "manifest.json") + "\n")
    return manifest


class Journal:
    def __init__(self, root, spec):
        self.path = Path(root) / "records" / spec["id"]
        self.path.mkdir(parents=True, exist_ok=False)
        self.events = (self.path / "events.jsonl").open("x")
        self.state = {
            "id": spec["id"],
            "status": "started",
            "pid": os.getpid(),
            "started_utc": datetime.now(UTC).isoformat(),
            "environment_step": 0,
        }
        atomic_json(self.path / "config.json", spec)
        self("started", {})

    def __call__(self, event, data):
        collector = data.get("collector")
        clean = {k: v for k, v in data.items() if k != "collector"}
        if collector is not None:
            export_evaluation(
                collector,
                {
                    "step": data["environment_step"],
                    "experiment_id": collector.episodes[0].experiment_id,
                },
                self.path / "evaluation" / f"step_{data['environment_step']}",
            )
        self.events.write(
            json.dumps(
                {"event": event, "utc": datetime.now(UTC).isoformat(), **clean},
                allow_nan=False,
            )
            + "\n"
        )
        self.events.flush()
        os.fsync(self.events.fileno())
        if "environment_step" in data:
            self.state["environment_step"] = data["environment_step"]
        if "optimizer_updates" in data:
            self.state["optimizer_updates"] = data["optimizer_updates"]
        self.state["last_event"] = event
        self.state["updated_utc"] = datetime.now(UTC).isoformat()
        atomic_json(self.path / "status.json", self.state)

    def finish(self, status, **data):
        self.state.update(status=status, **data)
        self(status, data)
        self.events.close()


def baseline(spec, output, manifest, observer):
    output.mkdir(parents=True, exist_ok=False)
    name = spec["kind"]
    agent = None
    if name == "vanilla":
        agent, _ = DQNAgent.load_checkpoint(spec["checkpoint"])
    elif name == "historical_ppo":
        agent, _ = PPOAgent.load_checkpoint(spec["checkpoint"])
    collector = AnalyticsCollector()
    started = time.perf_counter()
    for i, seed in enumerate(manifest["validation_seeds"]):
        policy = agent or (
            RandomAgent(619000000 + i) if name == "random" else GreedyAgent()
        )
        env = Match3LevelEnv(
            "blocker-mayhem", max_cascades=200, max_initialization_attempts=1000
        )
        try:
            result = run_episode(
                env,
                policy,
                seed=seed,
                max_steps=50,
                experiment_id=spec["id"],
                episode_id=f"{spec['id']}:final:{i}",
                agent_name=spec["id"],
                collector=collector,
            )
        finally:
            env.close()
        observer("evaluation_episode", asdict(result))
        if result.outcome not in ("WON", "LOST") or result.invalid_actions:
            export_evaluation(
                collector,
                {"failed": True, "experiment_id": spec["id"]},
                output / "partial_evaluation",
            )
            raise ValueError("baseline evaluation failed; partial records retained")
    summary = {
        "experiment_id": spec["id"],
        "summary": summarize_level_episodes(collector.episodes),
        "evaluation_seconds": time.perf_counter() - started,
        "historical_training": name in ("vanilla", "historical_ppo"),
        "new_training": False,
    }
    export_evaluation(collector, summary, output / "evaluation")
    atomic_json(output / "baseline_summary.json", summary)
    return summary


def worker(root, run_id):
    root = Path(root)
    manifest = verify_manifest(root)
    if (
        datetime.now(UTC).timestamp()
        >= datetime.fromisoformat(manifest["training_cutoff_utc"]).timestamp()
    ):
        raise RuntimeError(
            "execution window elapsed; no training or evaluation restart allowed"
        )
    spec = next(r for r in manifest["runs"] if r["id"] == run_id)
    journal = Journal(root, spec)
    atomic_json(
        journal.path / "provenance.json",
        {
            "protocol_sha256": sha256(root / "manifest.json"),
            "source_sha256": manifest["source_sha256"],
            "validation_seeds": manifest["validation_seeds"],
            "reward": manifest["reward"],
            "environment_config_sha256": manifest["environment_config_sha256"],
        },
    )
    output = root / "runs" / run_id
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    signal.signal(
        signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("supervisor time limit")),
    )
    try:
        config = dict(spec["config"])
        if "hidden_dims" in config:
            config["hidden_dims"] = tuple(config["hidden_dims"])
        if spec["kind"] == "ppo":
            cfg = OvernightPPOConfig(**config)
            result = train_ppo(
                cfg,
                output,
                learning_rate_schedule=late_learning_rate
                if cfg.learning_rate_schedule == "late_linear"
                else None,
                observer=journal,
            )
        elif spec["kind"] == "dqn":
            result = train_dqn(TrainingConfig(**config), output, observer=journal)
        else:
            result = baseline(spec, output, manifest, journal)
        verify_manifest(root)
        if (
            result.get("evaluation_invalid_actions", 0)
            or result.get("evaluation_errors", 0)
            or result.get("evaluation_time_limits", 0)
        ):
            raise ValueError("run has invalid evaluation records")
        journal.finish(
            "completed",
            final_summary=str(
                output
                / (
                    "training_summary.json"
                    if spec["kind"] in ("ppo", "dqn")
                    else "baseline_summary.json"
                )
            ),
        )
    except BaseException as error:
        journal.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
        )
        raise


def supervise(root):
    root = Path(root)
    manifest = verify_manifest(root)
    events = (root / "supervisor_events.jsonl").open("x")
    cutoff = datetime.fromisoformat(manifest["training_cutoff_utc"]).timestamp()
    stage_estimates = {
        "A": 85 * 60,
        "B_smoke": 5 * 60,
        "B": 35 * 60,
        "C_smoke": 5 * 60,
        "C": 35 * 60,
        "baselines": 20 * 60,
    }
    statuses = {}
    for stage in ("A", "B_smoke", "B", "C_smoke", "C", "baselines"):
        jobs = [s for s in manifest["runs"] if s["stage"] == stage]
        dependencies = {"B": ["B_smoke"], "C_smoke": ["B"], "C": ["B", "C_smoke"]}.get(
            stage, []
        )
        dependency_failed = any(
            statuses.get(r["id"]) != "completed"
            for r in manifest["runs"]
            if r["stage"] in dependencies
        )
        if dependency_failed or time.time() + stage_estimates[stage] >= cutoff:
            for spec in jobs:
                statuses[spec["id"]] = "skipped"
                atomic_json(
                    root / f"skipped_{spec['id']}.json",
                    {
                        "status": "skipped",
                        "reason": "dependency failure"
                        if dependency_failed
                        else "insufficient time before reserved finalization",
                        "id": spec["id"],
                    },
                )
            continue
        queue, active = list(jobs), []
        while queue or active:
            while queue and len(active) < 3:
                spec = queue.pop(0)
                log_dir = root / "logs"
                log_dir.mkdir(exist_ok=True)
                handle = (log_dir / f"{spec['id']}.log").open("x")
                env = {
                    **os.environ,
                    "OMP_NUM_THREADS": "1",
                    "OPENBLAS_NUM_THREADS": "1",
                    "MKL_NUM_THREADS": "1",
                    "MPLCONFIGDIR": "/private/tmp/match3-ai-matplotlib",
                }
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "match3_ai.training.overnight",
                        "worker",
                        "--root",
                        str(root),
                        "--run-id",
                        spec["id"],
                    ],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=env,
                )
                active.append((spec, process, handle))
                events.write(
                    json.dumps(
                        {
                            "event": "launched",
                            "id": spec["id"],
                            "pid": process.pid,
                            "utc": datetime.now(UTC).isoformat(),
                        }
                    )
                    + "\n"
                )
                events.flush()
            for spec, process, handle in active[:]:
                if process.poll() is not None:
                    handle.close()
                    state_path = root / "records" / spec["id"] / "status.json"
                    state = (
                        json.loads(state_path.read_text())
                        if state_path.exists()
                        else {
                            "status": "failed",
                            "reason": "worker exited before journal creation",
                        }
                    )
                    statuses[spec["id"]] = state["status"]
                    if process.returncode and state["status"] == "started":
                        statuses[spec["id"]] = "interrupted"
                    events.write(
                        json.dumps(
                            {
                                "event": "exited",
                                "id": spec["id"],
                                "returncode": process.returncode,
                                "status": statuses[spec["id"]],
                            }
                        )
                        + "\n"
                    )
                    events.flush()
                    active.remove((spec, process, handle))
            if time.time() >= cutoff:
                for _, process, _ in active:
                    process.terminate()
                for spec in queue:
                    statuses[spec["id"]] = "skipped"
                queue.clear()
            atomic_json(
                root / "execution_status.json",
                {
                    "stage": stage,
                    "completed_statuses": statuses,
                    "active": [{"id": s["id"], "pid": p.pid} for s, p, _ in active],
                    "queued": [s["id"] for s in queue],
                    "updated_utc": datetime.now(UTC).isoformat(),
                },
            )
            time.sleep(2)
    events.close()
    atomic_json(
        root / "execution_complete.json",
        {
            "statuses": statuses,
            "stage_D": "skipped; no separately frozen entropy pilot; prioritizing analysis and single-intervention interpretation",
            "utc": datetime.now(UTC).isoformat(),
        },
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("freeze", "worker", "supervise"))
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id")
    args = parser.parse_args()
    if args.mode == "freeze":
        freeze(args.root)
    elif args.mode == "worker":
        worker(args.root, args.run_id)
    else:
        supervise(args.root)


if __name__ == "__main__":
    main()
