"""Bounded, preregistered Sprint 5.19 execution using existing PPO and dueling DDQN."""

import argparse
import csv
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

import numpy as np
import torch

from match3_ai.agents.dqn_agent import DQNAgent
from match3_ai.agents.ppo_agent import PPOAgent
from match3_ai.evaluation.fresh_protocol import sha256
from match3_ai.training.dqn import TrainingConfig, train_dqn
from match3_ai.training.overnight import Journal, atomic_json, late_learning_rate
from match3_ai.training.ppo import PPOTrainingConfig, train_ppo

ROOT = Path("experiments/results/sprint5_19_rl_optimization")
BOARDS = list(range(310000, 310200))
PPO_SEEDS = (5970, 5971, 5972)
DQN_SEEDS = (5980, 5981, 5982)


@dataclass(frozen=True)
class StudyPPOConfig(PPOTrainingConfig):
    learning_rate_schedule: str = "constant"
    checkpoint_steps: tuple[int, ...] = (0, 50000, 75000, 100000)


def source_hashes():
    return {str(p): sha256(p) for p in sorted(Path("src/match3_ai").rglob("*.py"))}


def provenance(root):
    """Inventory recorded artifacts and conservatively reject any candidate integer."""
    root = Path(root)
    before = json.loads((root / "workspace_before.json").read_text())
    files = {
        name: d for name, d in before.items() if name.startswith("experiments/results/")
    }
    occurrences, masters = [], set()

    def walk(value, path):
        if type(value) is int and value in BOARDS:
            occurrences.append({"path": path, "value": value})
        elif isinstance(value, dict):
            for key, item in value.items():
                if key in ("seed", "master_seed") and type(item) is int:
                    masters.add(item)
                walk(item, path)
            for start_key, count_key in (
                ("evaluation_seed_start", "evaluation_episodes"),
                ("base_seed", "num_episodes"),
                ("seed_start", "num_episodes"),
            ):
                start, count = value.get(start_key), value.get(count_key)
                if type(start) is int and type(count) is int and count > 0:
                    overlap = set(BOARDS) & set(
                        range(max(310000, start), min(310200, start + count))
                    )
                    if overlap:
                        occurrences.append(
                            {"path": path, "declared_range_overlap": sorted(overlap)}
                        )
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item, path)

    for name, digest in files.items():
        if sha256(name) != digest:
            raise ValueError(f"historical artifact changed: {name}")
        path = Path(name)
        if path.suffix == ".csv":
            with path.open(newline="") as f:
                for row in csv.reader(f):
                    for cell in row:
                        if cell.isdecimal() and int(cell) in BOARDS:
                            occurrences.append({"path": name, "value": int(cell)})
        elif path.suffix in (".json", ".jsonl"):
            if path.suffix == ".json":
                walk(json.loads(path.read_text()), name)
            else:
                with path.open() as f:
                    for line in f:
                        if line.strip():
                            walk(json.loads(line), name)
        elif path.suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                for key in data.files:
                    a = data[key]
                    if a.dtype.kind in "iu":
                        values = np.unique(a[(a >= 310000) & (a <= 310199)])
                        if values.size:
                            occurrences.append(
                                {"path": name, "array": key, "values": values.tolist()}
                            )
        elif path.suffix == ".pt":
            payload = torch.load(path, map_location="cpu", weights_only=True)
            for key in (
                "training_config",
                "training_progress",
                "seed_metadata",
                "validation_metrics",
                "training_state",
            ):
                walk(payload.get(key, {}), name)
    ledger = {
        "files_sha256": files,
        "candidate_validation_seeds": BOARDS,
        "candidate_occurrences": occurrences,
        "used_seed_values_in_recorded_metadata": sorted(masters),
        "scope": "all pre-existing artifact files protected; all JSON/JSONL/CSV numeric values and NPZ/checkpoint metadata audited; unrecorded/external use unknown",
    }
    with (root / "history_inventory.json").open("x") as f:
        json.dump(ledger, f, indent=2)
    if occurrences or masters & {*PPO_SEEDS, *DQN_SEEDS}:
        raise ValueError(
            "candidate evaluation or master seeds have historical provenance conflicts"
        )
    return ledger


def configurations():
    runs = []
    order = (
        (5970, "constant"),
        (5971, "late_linear"),
        (5972, "constant"),
        (5970, "late_linear"),
        (5971, "constant"),
        (5972, "late_linear"),
    )
    for seed, mode in order:
        config = StudyPPOConfig(
            seed=seed,
            environment_steps=100000,
            evaluation_every=None,
            evaluation_seed_start=310000,
            evaluation_episodes=200,
            evaluation_partition="sprint5_19_validation",
            learning_rate_schedule=mode,
        )
        runs.append(
            {
                "id": f"a_{mode}_{seed}",
                "experiment": "A",
                "kind": "ppo",
                "config": asdict(config),
                "estimated_seconds": 3000,
            }
        )
    for seed in DQN_SEEDS:
        config = TrainingConfig(
            seed=seed,
            environment_steps=100000,
            train_seed_start=200000,
            evaluation_seed_start=310000,
            evaluation_episodes=200,
            evaluation_every=25000,
            evaluation_agent_seed_start=621900000,
            evaluation_partition="sprint5_19_validation",
            seed_policy="sprint5_8",
            evaluate_step_zero=True,
            baseline_every_checkpoint=False,
            algorithm="double",
            architecture="dueling",
        )
        runs.append(
            {
                "id": f"b_dueling_{seed}",
                "experiment": "B",
                "kind": "dqn",
                "config": asdict(config),
                "estimated_seconds": 3900,
            }
        )
    return json.loads(json.dumps(runs))


def execute(root, spec, protocol_path):
    root = Path(root)
    journal = Journal(root, spec)
    atomic_json(
        journal.path / "provenance.json",
        {
            "protocol": str(protocol_path),
            "protocol_sha256": sha256(protocol_path),
            "source_sha256": source_hashes(),
            "reward": "sparse unchanged",
        },
    )
    output = root / "runs" / spec["id"]
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    signal.signal(
        signal.SIGTERM,
        lambda *_: (_ for _ in ()).throw(KeyboardInterrupt("bounded supervisor stop")),
    )
    try:
        config = dict(spec["config"])
        config["hidden_dims"] = tuple(config["hidden_dims"])
        if spec["kind"] == "ppo":
            config["checkpoint_steps"] = tuple(config["checkpoint_steps"])
            cfg = StudyPPOConfig(**config)
            result = train_ppo(
                cfg,
                output,
                learning_rate_schedule=late_learning_rate
                if cfg.learning_rate_schedule == "late_linear"
                else None,
                evaluation_steps=cfg.checkpoint_steps,
                observer=journal,
            )
            agent, payload = PPOAgent.load_checkpoint(output / "checkpoint_final.pt")
        else:
            result = train_dqn(TrainingConfig(**config), output, observer=journal)
            agent, payload = DQNAgent.load_checkpoint(output / "checkpoint_final.pt")
        network = agent.network if spec["kind"] == "ppo" else agent.online
        if not all(torch.isfinite(v).all() for v in network.parameters()):
            raise ValueError("nonfinite reloaded checkpoint")
        if (
            result["environment_steps"] != config["environment_steps"]
            or result["optimizer_updates"] <= 0
        ):
            raise ValueError("run did not collect and optimize its budget")
        if any(
            result.get(k, 0)
            for k in (
                "invalid_actions",
                "evaluation_invalid_actions",
                "evaluation_errors",
                "evaluation_time_limits",
            )
        ):
            raise ValueError("invalid or incomplete evaluation records")
        atomic_json(
            journal.path / "completion_audit.json",
            {
                "checkpoint_reloaded": True,
                "experiment_id": result["experiment_id"],
                "environment_steps": result["environment_steps"],
                "optimizer_updates": result["optimizer_updates"],
                "checkpoint_config": payload["training_config"],
            },
        )
        journal.finish(
            "completed",
            experiment_id=result["experiment_id"],
            final_summary=str(output / "training_summary.json"),
        )
        return result
    except BaseException as error:
        journal.finish(
            "interrupted" if isinstance(error, KeyboardInterrupt) else "failed",
            error_type=type(error).__name__,
            error_message=str(error),
            traceback=traceback.format_exc(),
        )
        raise


def smoke(root, name="smoke"):
    study_root = Path(root)
    root = study_root / name
    root.mkdir(exist_ok=False)
    ppo = StudyPPOConfig(
        seed=5990,
        environment_steps=512,
        evaluation_every=None,
        evaluation_seed_start=300000,
        evaluation_episodes=2,
        checkpoint_steps=(0, 256, 512),
    )
    dqn = TrainingConfig(
        seed=5991,
        environment_steps=512,
        train_seed_start=200000,
        evaluation_seed_start=300000,
        evaluation_episodes=2,
        evaluation_every=256,
        evaluation_agent_seed_start=621900000,
        warmup_steps=64,
        evaluate_step_zero=True,
        baseline_every_checkpoint=False,
        seed_policy="sprint5_8",
        algorithm="double",
        architecture="dueling",
    )
    specs = [
        {"id": "ppo_e2e", "kind": "ppo", "config": asdict(ppo)},
        {"id": "dueling_e2e", "kind": "dqn", "config": asdict(dqn)},
    ]
    path = root / "manifest.json"
    with path.open("x") as f:
        json.dump(
            {
                "scope": "preflight only; excluded from research",
                "runs": specs,
                "source_sha256": source_hashes(),
            },
            f,
            indent=2,
        )
    results = {spec["id"]: execute(root, spec, path) for spec in specs}
    for spec in specs:
        record = root / "records" / spec["id"]
        assert json.loads((record / "status.json").read_text())["status"] == "completed"
        if spec["kind"] == "ppo":
            assert (record / "evaluation/step_512/experiment_metadata.json").is_file()
        assert (
            root / "runs" / spec["id"] / "evaluation/experiment_metadata.json"
        ).is_file()
    with (root / "summary.json").open("x") as f:
        json.dump(results, f, indent=2, allow_nan=False)
    with (study_root / "smoke_evidence.json").open("x") as f:
        json.dump(
            {
                "summary_path": str(root / "summary.json"),
                "summary_sha256": sha256(root / "summary.json"),
            },
            f,
            indent=2,
        )
    return results


def freeze(root):
    root = Path(root)
    if not (root / "smoke_evidence.json").is_file():
        raise ValueError("actual end-to-end smoke must complete before registration")
    smoke_evidence = json.loads((root / "smoke_evidence.json").read_text())
    history = json.loads((root / "history_inventory.json").read_text())
    if history["candidate_occurrences"]:
        raise ValueError("fresh evaluation seeds have provenance conflicts")
    runs = configurations()
    window = json.loads((root / "window.json").read_text())
    if (
        datetime.now(UTC).timestamp() + 9900
        >= datetime.fromisoformat(window["training_cutoff_utc"]).timestamp()
    ):
        raise ValueError(
            "insufficient time for planned research waves and reserved finalization"
        )
    protocol = {
        "study": "sprint5_19_rl_optimization",
        **window,
        "frozen_utc": datetime.now(UTC).isoformat(),
        "runs": runs,
        "validation_seeds": BOARDS,
        "validation_partition": "explicit new study block310000–310199; legacy partitions unchanged",
        "reward": "unchanged sparse terminal -1/0/+1; PBRS disabled",
        "level": "blocker-mayhem",
        "training_partition": [200000, 299999],
        "reserved_final_test_not_accessed": [400000, 400999],
        "ppo_schedule": ".0003 through50k; .0003*(1-.9*clip((rollout_completion-50000)/50000,0,1)); all epochs share rollout LR; final .00003",
        "dueling_exploration": "unchanged epsilon1 to.05 over50000 collected transitions, then stays.05; no restart at50k",
        "dueling_training_board_stream": "same deterministic sequential episode-seed curriculum starting200000 per private run; independent model/exploration/replay RNGs from three new masters; inference about training variability conditional on this curriculum",
        "fresh_initialization": True,
        "exact_checkpoint_continuation": False,
        "primary_hypotheses": [
            "mean PPO decay minus constant at100k",
            "mean within-run dueling100k minus50k",
        ],
        "primary_statistics": {
            "samples": 20000,
            "seed": 519519,
            "test": "two-sided approximate null-centered paired shared-board bootstrap with plus-one p",
            "multiplicity": "Holm across both primary hypotheses; missing hypothesis non-rejection slot1 only for correction",
            "ci": "pointwise95% paired board percentile; conditional on three trained pairs; not simultaneous",
            "joint_resampling": "one common board vector across all models, checkpoints, metrics and contrasts",
            "training_variation": "individual effects and training-seed SD reported separately; sign-flip sensitivity with three pairs cannot give p<.25",
        },
        "secondary_statistics": "six individual endpoint comparisons (three A, three B) exact McNemar; separate six-slot exploratory Holm family; all other checkpoints, late regression, absolute change, diagnostics and historical comparisons descriptive",
        "late_stability": "75k→100k signed win change, negative-part regression max(0,win75-win100), absolute change; no causal claims from diagnostic alignment",
        "practical_improvement_pp": 5,
        "selection": "fixed100k primary endpoint; best checkpoint descriptive only, never substituted",
        "parallelism": {"workers": 3, "torch_threads": 1},
        "runtime_estimates_seconds": {
            "A_two_waves": 6000,
            "B_one_wave": 3900,
            "finalization_reserved": 3600,
        },
        "failure_policy": "fail fast; preserve every completed/partial run, stop queue on unexpected worker failure; no automatic retries, replacements, exclusions or manifest edits",
        "source_sha256": source_hashes(),
        "environment_config_sha256": sha256(
            "src/match3_ai/levels/data/blocker-mayhem.json"
        ),
        "history_inventory_sha256": sha256(root / "history_inventory.json"),
        "smoke_summary_sha256": smoke_evidence["summary_sha256"],
        "smoke_summary_path": smoke_evidence["summary_path"],
        "smoke_evidence_sha256": sha256(root / "smoke_evidence.json"),
    }
    for name, d in history["files_sha256"].items():
        if sha256(name) != d:
            raise ValueError("historical artifact changed before freeze")
    with (root / "manifest.json").open("x") as f:
        json.dump(protocol, f, indent=2, allow_nan=False)
    with (root / "manifest.sha256").open("x") as f:
        f.write(sha256(root / "manifest.json") + "\n")
    archive = root / "frozen_source"
    for name, d in protocol["source_sha256"].items():
        destination = archive / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("xb") as f:
            f.write(Path(name).read_bytes())
    return protocol


def verify(root):
    root = Path(root)
    path = root / "manifest.json"
    if sha256(path) != (root / "manifest.sha256").read_text().strip():
        raise ValueError("frozen Sprint5.19 manifest changed")
    protocol = json.loads(path.read_text())
    if any(sha256(name) != d for name, d in protocol["source_sha256"].items()):
        raise ValueError("source differs from frozen protocol")
    if (
        sha256(root / "history_inventory.json") != protocol["history_inventory_sha256"]
        or sha256(protocol["smoke_summary_path"]) != protocol["smoke_summary_sha256"]
        or sha256(root / "smoke_evidence.json") != protocol["smoke_evidence_sha256"]
    ):
        raise ValueError("preflight evidence changed")
    return protocol


def worker(root, run_id):
    protocol = verify(root)
    if (
        datetime.now(UTC).timestamp()
        >= datetime.fromisoformat(protocol["training_cutoff_utc"]).timestamp()
    ):
        raise RuntimeError("training cutoff reached")
    spec = next(r for r in protocol["runs"] if r["id"] == run_id)
    execute(root, spec, Path(root) / "manifest.json")
    verify(root)


def supervise(root):
    root = Path(root)
    protocol = verify(root)
    cutoff = datetime.fromisoformat(protocol["training_cutoff_utc"]).timestamp()
    deadline = datetime.fromisoformat(protocol["deadline_utc"]).timestamp()
    event_handle = (root / "supervisor_events.jsonl").open("x")

    def event(kind, **data):
        event_handle.write(
            json.dumps({"event": kind, "utc": datetime.now(UTC).isoformat(), **data})
            + "\n"
        )
        event_handle.flush()
        os.fsync(event_handle.fileno())

    queue = list(protocol["runs"])
    active = []
    statuses = {}
    stopping = False
    termination_times = {}
    logs = root / "logs"
    logs.mkdir(exist_ok=False)
    environment = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "MPLCONFIGDIR": "/private/tmp/match3-ai-matplotlib",
    }
    event("supervisor_started", pid=os.getpid())
    while queue or active:
        for spec, process, handle in active[:]:
            if process.poll() is not None:
                handle.close()
                path = root / "records" / spec["id"] / "status.json"
                state = (
                    json.loads(path.read_text())
                    if path.is_file()
                    else {"status": "failed"}
                )
                status = (
                    state["status"]
                    if process.returncode == 0
                    else (
                        state["status"]
                        if state["status"] in ("failed", "interrupted")
                        else "failed"
                    )
                )
                if process.returncode and state["status"] == "completed":
                    state.update(
                        status="failed", post_completion_exit_code=process.returncode
                    )
                    atomic_json(path, state)
                if state["status"] not in ("completed", "failed", "interrupted"):
                    state.update(
                        status="interrupted"
                        if process.pid in termination_times
                        else "failed",
                        supervisor_exit_code=process.returncode,
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    atomic_json(path, state)
                    status = state["status"]
                statuses[spec["id"]] = status
                event(
                    "worker_exited",
                    id=spec["id"],
                    returncode=process.returncode,
                    status=status,
                )
                active.remove((spec, process, handle))
                if status != "completed":
                    stopping = True
        if time.time() >= cutoff:
            stopping = True
        if stopping:
            for spec in queue:
                statuses[spec["id"]] = "not_started_safety_stop"
            queue.clear()
            for spec, process, _ in active:
                if process.poll() is None and process.pid not in termination_times:
                    process.terminate()
                    termination_times[process.pid] = time.time()
                    event("termination_requested", id=spec["id"], pid=process.pid)
                elif (
                    process.poll() is None
                    and time.time() - termination_times[process.pid] > 30
                ):
                    process.kill()
                    event("bounded_kill_requested", id=spec["id"], pid=process.pid)
        while not stopping and queue and len(active) < 3:
            spec = queue[0]
            if time.time() + spec["estimated_seconds"] >= cutoff:
                statuses[spec["id"]] = "not_started_insufficient_time"
                queue.pop(0)
                continue
            queue.pop(0)
            handle = (logs / f"{spec['id']}.log").open("x")
            try:
                process = subprocess.Popen(
                    [
                        sys.executable,
                        "-u",
                        "-m",
                        "match3_ai.training.sprint519",
                        "worker",
                        "--root",
                        str(root),
                        "--run-id",
                        spec["id"],
                    ],
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    env=environment,
                )
            except OSError as error:
                handle.write(f"Worker launch failed: {type(error).__name__}: {error}\n")
                handle.close()
                statuses[spec["id"]] = "failed_to_launch"
                event("worker_launch_failed", id=spec["id"], error=str(error))
                stopping = True
                break
            active.append((spec, process, handle))
            event("worker_started", id=spec["id"], pid=process.pid)
        atomic_json(
            root / "execution_status.json",
            {
                "supervisor_pid": os.getpid(),
                "statuses": statuses,
                "active": [{"id": s["id"], "pid": p.pid} for s, p, _ in active],
                "queued": [s["id"] for s in queue],
                "stopping": stopping,
                "updated_utc": datetime.now(UTC).isoformat(),
            },
        )
        if active:
            time.sleep(2)
    atomic_json(
        root / "execution_complete.json",
        {
            "statuses": statuses,
            "completed_utc": datetime.now(UTC).isoformat(),
            "safety_stop": stopping,
        },
    )
    event("training_queue_finished", statuses=statuses)
    tasks = [
        (
            "analysis",
            [
                sys.executable,
                "-m",
                "match3_ai.training.sprint519_analysis",
                "analyze",
                "--root",
                str(root),
            ],
        ),
        ("full_pytest", [sys.executable, "-m", "pytest", "-q"]),
        (
            "focused_pytest",
            [sys.executable, "-m", "pytest", "-q", "tests/test_sprint519.py"],
        ),
        ("ruff", [sys.executable, "-m", "ruff", "check", "."]),
        (
            "format",
            [
                sys.executable,
                "-m",
                "ruff",
                "format",
                "--check",
                "src/match3_ai/training/sprint519.py",
                "src/match3_ai/training/sprint519_analysis.py",
                "src/match3_ai/training/ppo.py",
                "src/match3_ai/training/dqn.py",
                "src/match3_ai/training/seed_policy.py",
                "tests/test_sprint519.py",
            ],
        ),
        ("diff_check", ["git", "diff", "--check"]),
    ]
    validation = {}
    for name, command in tasks:
        remaining = deadline - time.time()
        if remaining < 90:
            validation[name] = {"status": "not_started_deadline", "command": command}
            continue
        with (logs / f"final_{name}.log").open("x") as handle:
            process = subprocess.Popen(
                command, stdout=handle, stderr=subprocess.STDOUT, env=environment
            )
            event("finalization_started", task=name, pid=process.pid)
            try:
                code = process.wait(timeout=min(1800, remaining - 60))
                validation[name] = {
                    "status": "passed" if code == 0 else "failed",
                    "returncode": code,
                    "command": command,
                }
            except subprocess.TimeoutExpired:
                process.terminate()
                try:
                    process.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                validation[name] = {"status": "timeout", "command": command}
        atomic_json(root / "validation.json", validation)
    if validation.get("analysis", {}).get("status") == "passed":
        with (logs / "final_report.log").open("x") as handle:
            process = subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "match3_ai.training.sprint519_analysis",
                    "finalize",
                    "--root",
                    str(root),
                ],
                stdout=handle,
                stderr=subprocess.STDOUT,
                env=environment,
            )
            event("report_started", pid=process.pid)
            try:
                code = process.wait(timeout=max(1, min(300, deadline - time.time())))
            except subprocess.TimeoutExpired:
                process.terminate()
                process.wait()
                code = -1
            validation["report"] = {
                "status": "passed" if code == 0 else "failed",
                "returncode": code,
            }
    atomic_json(root / "validation.json", validation)
    atomic_json(
        root / "supervisor_complete.json",
        {
            "statuses": statuses,
            "validation": validation,
            "finished_utc": datetime.now(UTC).isoformat(),
            "active_children": [],
        },
    )
    event("supervisor_finished", validation=validation)
    event_handle.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode", choices=("provenance", "smoke", "freeze", "worker", "supervise")
    )
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--run-id")
    parser.add_argument("--smoke-name", default="smoke")
    args = parser.parse_args()
    if args.mode == "worker":
        worker(args.root, args.run_id)
    elif args.mode == "smoke":
        smoke(args.root, args.smoke_name)
    else:
        {
            "provenance": provenance,
            "smoke": smoke,
            "freeze": freeze,
            "supervise": supervise,
        }[args.mode](args.root)


if __name__ == "__main__":
    main()
