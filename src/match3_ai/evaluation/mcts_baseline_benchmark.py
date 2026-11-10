"""Paired development evaluation of standalone baselines and stochastic MCTS."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import platform
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.mcts_simulator import MCTSSimulator
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.agents.stochastic_mcts_agent import StochasticMCTSAgent
from match3_ai.evaluation.analytics import AnalyticsCollector, _csv_value
from match3_ai.evaluation.behavior_data import (
    complete_state_fingerprint,
    observation_fingerprint,
)
from match3_ai.evaluation.experiments import (
    _source_fingerprint,
    summarize_level_episodes,
)
from match3_ai.evaluation.mcts_provenance import (
    ALGORITHM_SEMANTICS,
    BENCHMARK_SCHEMA_VERSION,
    IMPLEMENTATION_VERSION,
    semantic_source_hashes,
)
from match3_ai.evaluation.runner import EpisodeResult, run_episode
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.training.seed_policy import PARTITIONS

LEVELS = ("double-trouble-v2", "wooden-trouble-v2", "frozen-garden-v2")
AGENTS = {
    "Random": None,
    "Greedy": None,
    "MCTS_random_terminal": ("random", "terminal"),
    "MCTS_greedy_outcome": ("greedy", "outcome"),
}
AGENT_REGISTRY = {
    **AGENTS,
    "MCTS_random_outcome": ("random", "outcome"),
    "MCTS_greedy_terminal": ("greedy", "terminal"),
}
MCTS_ABLATION = (
    "MCTS_random_terminal",
    "MCTS_random_outcome",
    "MCTS_greedy_terminal",
    "MCTS_greedy_outcome",
)
EXTRA_FIELDS = (
    "initial_fingerprint",
    "pairing_verified",
    "action_selection_calls",
    "decision_time_seconds",
    "episode_time_seconds",
    "simulations",
    "implementation_version",
    "record_origin",
    "origin_episode_id",
    "origin_manifest_sha256",
    "origin_episodes_sha256",
)
SUMMARY_FIELDS = (
    "level_id",
    "agent_name",
    "attempted",
    "wins",
    "losses",
    "errors",
    "time_limits",
    "invalid_actions",
    "win_rate",
    "win_rate_ci95_low",
    "win_rate_ci95_high",
    "mean_objective_completion",
    "n_objective_completion",
    "mean_winning_moves",
    "n_winning_moves",
    "action_selection_calls",
    "mean_decision_time_seconds",
    "mean_episode_time_seconds",
    "n_episode_time",
    "implementation_versions",
    "reused_episodes",
)


@dataclass(frozen=True)
class BenchmarkConfig:
    episodes: int = 20
    simulations: int = 30
    env_seed_start: int = 10_000_000
    agent_seed_start: int = 20_000_000
    agents: tuple[str, ...] = tuple(AGENTS)

    def __post_init__(self):
        if (
            not self.agents
            or len(set(self.agents)) != len(self.agents)
            or any(name not in AGENT_REGISTRY for name in self.agents)
        ):
            raise ValueError(
                "agents must be a nonempty selection of unique registered names"
            )
        for name in ("episodes", "simulations"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("env_seed_start", "agent_seed_start"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 0:
                raise ValueError(f"{name} must be a non-negative integer")
        env_end = self.env_seed_start + self.episodes - 1
        agent_end = max(
            self.agent_seed(
                len(LEVELS) - 1, self.episodes - 1, list(AGENT_REGISTRY).index(name)
            )
            for name in self.agents
        )
        if max(self.env_seed_start, self.agent_seed_start) <= min(env_end, agent_end):
            raise ValueError("Environment and agent seed ranges must be disjoint")
        low, high = PARTITIONS["final_test"]
        if self.env_seed_start <= high and env_end >= low:
            raise ValueError("Reserved final-test environment seeds are excluded")

    def agent_seed(self, level_index: int, episode_index: int, agent_index: int) -> int:
        board_index = level_index * self.episodes + episode_index
        if agent_index < 4:
            return self.agent_seed_start + board_index * 4 + agent_index
        return (
            self.agent_seed_start
            + len(LEVELS) * self.episodes * 4
            + board_index * 2
            + agent_index
            - 4
        )


class BenchmarkEnvironment(Match3LevelEnv):
    """Capture reset identity without attaching shared records to MCTS clones."""

    def __init__(self, level_id: str):
        super().__init__(level_id, max_cascades=200, max_initialization_attempts=1000)
        self.initial_identity = None

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.initial_identity = (
            observation_fingerprint(observation),
            info["board_seed"],
            info["engine_seed"],
            info["level_config_id"],
        )
        return observation, info


def make_agent(name: str, env: Match3LevelEnv, seed: int, simulations: int):
    if name == "Random":
        return RandomAgent(seed)
    if name == "Greedy":
        return GreedyAgent()
    if name not in AGENT_REGISTRY:
        raise ValueError(f"Unknown benchmark agent: {name}")
    policy, value = AGENT_REGISTRY[name]
    return StochasticMCTSAgent(
        MCTSSimulator(env),
        simulations=simulations,
        rollout_policy=policy,
        value_mode=value,
        seed=seed,
        exploration_constant=math.sqrt(2),
    )


class EpisodeAgent:
    """Bind after runner reset; isolate agent timing from integrity checks."""

    def __init__(self, name, env, seed, simulations, identities, environment_seed):
        self.name, self.env, self.seed, self.simulations = name, env, seed, simulations
        self.identities = identities
        self.key = (env.level_id, environment_seed)
        self.agent = None
        self.calls = 0
        self.decision_seconds = 0.0
        self.pairing_verified = False
        self.integrity_error = None

    def select_action(self, observation):
        if self.agent is None:
            identity = self.env.initial_identity
            if (
                identity is None
                or self.identities.setdefault(self.key, identity) != identity
            ):
                self.integrity_error = "Initial-state pairing mismatch"
                raise RuntimeError(self.integrity_error)
            self.pairing_verified = True
            self.agent = make_agent(self.name, self.env, self.seed, self.simulations)
        before = (
            complete_state_fingerprint(self.env)
            if self.name.startswith("MCTS_")
            else None
        )
        self.calls += 1
        started = perf_counter()
        try:
            return self.agent.select_action(observation)
        finally:
            self.decision_seconds += perf_counter() - started
            if before is not None and complete_state_fingerprint(self.env) != before:
                self.integrity_error = "MCTS mutated the real environment or RNG"
                raise RuntimeError(self.integrity_error)


def atomic_json(path: Path, value: dict) -> None:
    temporary = path.with_suffix(".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def summarize(records: list[EpisodeResult], extras: list[dict]) -> list[dict]:
    extra_by_id = {row["episode_id"]: row for row in extras}
    rows = []
    for group in summarize_level_episodes(records):
        selected = [
            record
            for record in records
            if (
                record.level_id == group["level_id"]
                and record.agent_name == group["agent_name"]
            )
        ]
        timing = [extra_by_id[record.episode_id] for record in selected]
        winning_moves = [
            record.moves_used
            for record in selected
            if record.outcome == "WON" and record.moves_used is not None
        ]
        episode_times = [
            row["episode_time_seconds"]
            for row in timing
            if row["episode_time_seconds"] is not None
        ]
        calls = sum(row["action_selection_calls"] for row in timing)
        rows.append(
            {
                **{
                    name: group[name]
                    for name in (
                        "level_id",
                        "agent_name",
                        "attempted",
                        "wins",
                        "losses",
                        "errors",
                        "time_limits",
                        "win_rate",
                    )
                },
                "invalid_actions": sum(record.invalid_actions for record in selected),
                "win_rate_ci95_low": group["win_rate_ci95"][0],
                "win_rate_ci95_high": group["win_rate_ci95"][1],
                "mean_objective_completion": group["mean_objective_completion_ratio"],
                "n_objective_completion": group["n_objective_completion_ratio"],
                "mean_winning_moves": sum(winning_moves) / len(winning_moves)
                if winning_moves
                else None,
                "n_winning_moves": len(winning_moves),
                "action_selection_calls": calls,
                "mean_decision_time_seconds": sum(
                    row["decision_time_seconds"] for row in timing
                )
                / calls
                if calls
                else None,
                "mean_episode_time_seconds": sum(episode_times) / len(episode_times)
                if episode_times
                else None,
                "n_episode_time": len(episode_times),
                "implementation_versions": ";".join(
                    sorted(
                        {
                            row.get("implementation_version", IMPLEMENTATION_VERSION)
                            for row in timing
                        }
                    )
                ),
                "reused_episodes": sum(
                    row.get("record_origin") == "imported" for row in timing
                ),
            }
        )
    return rows


def write_summary(output: Path, records, extras) -> list[dict]:
    rows = summarize(records, extras)
    temporary = output / "summary.tmp"
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(
            {key: _csv_value(value) for key, value in row.items()} for row in rows
        )
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, output / "summary.csv")
    return rows


def build_manifest(config: BenchmarkConfig) -> dict:
    level_root = Path(__file__).resolve().parents[1] / "levels/data"
    levels = {
        name: hashlib.sha256((level_root / f"{name}.json").read_bytes()).hexdigest()
        for name in LEVELS
    }
    configurations = {
        name: {
            "rollout_policy": value[0],
            "value_mode": value[1],
            "simulations": config.simulations,
            "exploration_constant": math.sqrt(2),
            "win_efficiency_weight": 0.2,
            "loss_progress_weight": 0.2,
        }
        if value
        else {"policy": name}
        for name, value in AGENT_REGISTRY.items()
        if name in config.agents
    }
    identity = {
        "config": asdict(config),
        "level_sha256": levels,
        "agents": configurations,
        "package_source_sha256": _source_fingerprint(),
        "semantic_source_sha256": semantic_source_hashes(),
        "algorithm_semantics": ALGORITHM_SEMANTICS,
        "implementation_version": IMPLEMENTATION_VERSION,
        "schema_version": BENCHMARK_SCHEMA_VERSION,
    }
    return {
        **identity,
        "experiment_id": hashlib.sha256(
            json.dumps(identity, sort_keys=True).encode()
        ).hexdigest()[:24],
        "label": "paired development evaluation; not reserved final-test evaluation",
        "status": "running",
        "started_utc": datetime.now(UTC).isoformat(),
        "planned_episodes": len(LEVELS) * len(config.agents) * config.episodes,
        "attempted_episodes": 0,
        "records_written": 0,
        "active_episode": None,
        "environment_seeds": list(
            range(config.env_seed_start, config.env_seed_start + config.episodes)
        ),
        "agent_seed_rule": "Legacy Random/Greedy/RT/GO retain indices 0/1/2/3 and stride 4. RO/GT use a disjoint extension after all legacy slots, stride 2. Selection order never changes seeds; Greedy seed is unused.",
        "max_cascades": 200,
        "max_initialization_attempts": 1000,
        "denominators": {
            "win_rate": "all returned episode records, including ERROR and TIME_LIMIT; interrupted in-flight episodes have no returned record",
            "objective_completion": "available runner values, including last completed state on ERROR; n_objective_completion is recorded",
            "winning_moves": "WON records with available moves_used; no wins produces null",
            "decision_time": "total underlying agent select_action time / actual underlying select_action calls, including calls that raise; excludes lazy setup and state-integrity checks",
            "episode_time": "available wall durations around run_episode, including reset, setup and state-integrity checks; excludes construction, export and close",
        },
        "execution": "sequential; a fresh agent and live environment per episode",
        "versions": {
            "python": platform.python_version(),
            "numpy": version("numpy"),
            "gymnasium": version("gymnasium"),
        },
        "platform": platform.platform(),
        "cli_sha256": hashlib.sha256(
            (
                Path(__file__).resolve().parents[3]
                / "scripts/benchmark_mcts_baselines.py"
            ).read_bytes()
        ).hexdigest(),
    }


def run_benchmark(
    config: BenchmarkConfig,
    output_dir: str | Path,
    *,
    reuse_from: str | Path | None = None,
) -> list[dict]:
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    manifest = build_manifest(config)
    reused = {}
    if reuse_from is not None:
        from match3_ai.evaluation.mcts_ablation import validate_reuse

        reused, provenance = validate_reuse(config, Path(reuse_from), manifest)
        manifest["reuse"] = provenance
    manifest["records_reused"] = 0
    manifest["records_new"] = 0
    output.mkdir(parents=True, exist_ok=False)
    atomic_json(output / "manifest.json", manifest)
    collector = AnalyticsCollector()
    records, extras, identities = collector.episodes, [], {}
    started = perf_counter()
    try:
        with (output / "episodes.csv").open(
            "x", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle, fieldnames=[*EpisodeResult.__dataclass_fields__, *EXTRA_FIELDS]
            )
            writer.writeheader()
            handle.flush()
            os.fsync(handle.fileno())
            write_summary(output, records, extras)

            def persist(record, extra):
                extras.append(extra)
                collector.record_episode(record)
                writer.writerow(
                    {
                        key: _csv_value(value)
                        for key, value in {
                            **asdict(record),
                            **{key: extra[key] for key in EXTRA_FIELDS},
                        }.items()
                    }
                )
                handle.flush()
                os.fsync(handle.fileno())
                counter = (
                    "records_reused"
                    if extra["record_origin"] == "imported"
                    else "records_new"
                )
                manifest[counter] += 1
                manifest["records_written"] = len(records)
                manifest["active_episode"] = None
                write_summary(output, records, extras)
                atomic_json(output / "manifest.json", manifest)

            for level_index, level in enumerate(LEVELS):
                for index in range(config.episodes):
                    seed = config.env_seed_start + index
                    for name in config.agents:
                        agent_index = list(AGENT_REGISTRY).index(name)
                        agent_seed = config.agent_seed(level_index, index, agent_index)
                        episode_id = (
                            f"{manifest['experiment_id']}:{level}:{index}:{name}"
                        )
                        manifest["active_episode"] = {
                            "level_id": level,
                            "agent_name": name,
                            "seed": seed,
                            "agent_seed": agent_seed,
                        }
                        manifest["attempted_episodes"] += 1
                        atomic_json(output / "manifest.json", manifest)
                        if (level, seed, name) in reused:
                            original, extra = reused[(level, seed, name)]
                            record = replace(
                                original,
                                experiment_id=manifest["experiment_id"],
                                episode_id=episode_id,
                            )
                            extra = {**extra, "episode_id": episode_id}
                            identity = (
                                extra["initial_fingerprint"],
                                record.board_seed,
                                record.engine_seed,
                                record.level_config_id,
                            )
                            if (
                                identities.setdefault((level, seed), identity)
                                != identity
                            ):
                                raise ValueError(
                                    "Imported initial-state pairing mismatch"
                                )
                            persist(record, extra)
                            print(
                                f"Imported {level} seed={seed} {name}: {record.outcome}",
                                flush=True,
                            )
                            continue
                        env = BenchmarkEnvironment(level)
                        adapter = EpisodeAgent(
                            name, env, agent_seed, config.simulations, identities, seed
                        )
                        try:
                            episode_start = perf_counter()
                            record = run_episode(
                                env,
                                adapter,
                                seed=seed,
                                max_steps=env.config.move_limit,
                                experiment_id=manifest["experiment_id"],
                                episode_id=episode_id,
                                agent_name=name,
                                agent_seed=agent_seed,
                            )
                            episode_seconds = perf_counter() - episode_start
                        finally:
                            env.close()
                        extra = {
                            "episode_id": episode_id,
                            "initial_fingerprint": env.initial_identity[0]
                            if env.initial_identity
                            else None,
                            "pairing_verified": adapter.pairing_verified,
                            "action_selection_calls": adapter.calls,
                            "decision_time_seconds": adapter.decision_seconds,
                            "episode_time_seconds": episode_seconds,
                            "simulations": config.simulations
                            if AGENT_REGISTRY[name]
                            else None,
                            "implementation_version": IMPLEMENTATION_VERSION,
                            "record_origin": "new",
                            "origin_episode_id": None,
                            "origin_manifest_sha256": None,
                            "origin_episodes_sha256": None,
                        }
                        persist(record, extra)
                        print(
                            f"{level} seed={seed} {name}: {record.outcome} moves={record.moves_used} episode={episode_seconds:.2f}s",
                            flush=True,
                        )
                        if adapter.integrity_error:
                            raise RuntimeError(adapter.integrity_error)
        manifest["status"] = (
            "complete_with_errors"
            if any(
                record.outcome not in ("WON", "LOST") or record.invalid_actions
                for record in records
            )
            else "complete"
        )
    except BaseException as error:
        manifest["status"] = (
            "interrupted"
            if isinstance(error, (KeyboardInterrupt, SystemExit))
            else "failed"
        )
        manifest["failure"] = {"type": type(error).__name__, "message": str(error)}
        raise
    finally:
        manifest["records_written"] = len(records)
        manifest["finished_utc"] = datetime.now(UTC).isoformat()
        manifest["elapsed_seconds"] = perf_counter() - started
        write_summary(output, records, extras)
        atomic_json(output / "manifest.json", manifest)
    return summarize(records, extras)
