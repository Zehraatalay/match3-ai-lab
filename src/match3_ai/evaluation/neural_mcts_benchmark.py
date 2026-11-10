"""Checkpoint-only, paired development evaluation of neural and classic MCTS.

No trainer, optimizer, or replay is restored for inference. Profile artifacts
are explicit inputs; incompatible configurations are never silently altered.
Classic MCTS means the existing greedy-rollout/outcome-value stochastic agent.
Equal simulations do not imply equal simulated steps or equal decision time.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.mcts_simulator import MCTSSimulator
from match3_ai.agents.neural_mcts_agent import NeuralMCTSAgent
from match3_ai.agents.policy_value_encoder import PolicyValueContext, PolicyValueEncoder
from match3_ai.agents.policy_value_network import PolicyValueNetwork
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.agents.stochastic_mcts_agent import StochasticMCTSAgent
from match3_ai.evaluation.analytics import _csv_value
from match3_ai.evaluation.behavior_data import (
    complete_state_fingerprint,
    observation_fingerprint,
)
from match3_ai.evaluation.experiments import (
    _source_fingerprint,
    summarize_level_episodes,
)
from match3_ai.evaluation.mcts_baseline_benchmark import atomic_json, summarize
from match3_ai.evaluation.runner import EpisodeResult, run_episode
from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.games.match3.specials import SpecialType
from match3_ai.levels.config import (
    BlockerPlacement,
    InitialSpecial,
    LevelConfig,
    LevelObjective,
    ObjectiveType,
)
from match3_ai.levels.generation import GenerationSpec, LevelGenerator
from match3_ai.levels.loader import load_level
from match3_ai.training.neural_mcts import NeuralMCTSTrainer
from match3_ai.training.seed_policy import PARTITIONS

ROUNDS = (1, 100, 200, 400, 600, 669)
BASELINES = ("Random", "Greedy", "Classic_MCTS_GO", "Neural_untrained")


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def model_sha256(model: PolicyValueNetwork) -> str:
    return hashlib.sha256(
        b"".join(
            name.encode() + tensor.detach().cpu().contiguous().numpy().tobytes()
            for name, tensor in model.state_dict().items()
        )
    ).hexdigest()


def level_from_checkpoint(raw: dict) -> LevelConfig:
    data = dict(raw)
    data["blockers"] = tuple(
        BlockerPlacement(
            tuple(b["position"]), BlockerType(b["blocker_type"]), b["variant"]
        )
        for b in data["blockers"]
    )
    data["initial_specials"] = tuple(
        InitialSpecial(tuple(s["position"]), SpecialType(s["special_type"]))
        for s in data["initial_specials"]
    )
    data["objectives"] = tuple(
        LevelObjective(ObjectiveType(o["objective_type"]), o["target"], o["color"])
        for o in data["objectives"]
    )
    if data["cell_mask"] is not None:
        data["cell_mask"] = tuple(tuple(row) for row in data["cell_mask"])
    return LevelConfig(**data)


def load_policy_value_checkpoint(path: Path) -> tuple[PolicyValueNetwork, dict]:
    """Validate and freeze only inference weights, without restoring a trainer."""
    path = Path(path)
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if (
        path.stem.startswith("round_")
        and path.stem[6:].isdigit()
        and int(path.stem[6:]) != payload["rounds_completed"]
    ):
        raise ValueError("checkpoint filename and round counter disagree")
    if (
        payload["format_version"] != 1
        or payload["algorithm"] != "neural_mcts_policy_value"
        or payload["architecture"] != NeuralMCTSTrainer.architecture
        or payload["encoder_version"] != PolicyValueEncoder.version
    ):
        raise ValueError("incompatible policy-value checkpoint")
    level = level_from_checkpoint(payload["level_config"])
    context = PolicyValueContext.from_level_config(level)
    if (
        asdict(context) != payload["context"]
        or hashlib.sha256(
            json.dumps(asdict(level), sort_keys=True).encode()
        ).hexdigest()
        != payload["level_sha256"]
    ):
        raise ValueError("checkpoint level/context identity mismatch")
    model = PolicyValueNetwork(payload["stream_seeds"]["network"])
    model.load_state_dict(payload["network"], strict=True)
    if not all(torch.isfinite(p).all() for p in model.parameters()):
        raise ValueError("checkpoint has nonfinite weights")
    model.eval().requires_grad_(False)
    metadata = {
        key: payload[key]
        for key in (
            "rounds_completed",
            "updates",
            "attempts",
            "level_sha256",
            "level_config",
            "config",
            "search_config",
            "stream_seeds",
        )
    }
    metadata.update(
        path=str(path),
        sha256=file_sha256(path),
        replay_outcomes=dict(
            Counter(str(float(e["outcome"])) for e in payload["replay"]["examples"])
        ),
    )
    return model, metadata


def load_profiles(
    final_metadata: dict, root: Path = Path(".")
) -> tuple[dict[str, LevelConfig], dict]:
    """Use saved profiles and explicitly reconstruct a previously confirmed candidate."""
    fixtures = root / "experiments/fixtures/az4_profiles"
    if fixtures.is_dir():
        easy_path = fixtures / "easy/level.json"
        medium_path = fixtures / "medium/level.json"
        hard_path = fixtures / "hard/level.json"
    else:
        easy_path = root / "results/plg/targeted_smoke/easy/level.json"
        medium_path = root / "results/plg/final_acceptance/medium/level.json"
        hard_path = root / "results/plg/final_acceptance/hard/level.json"
    report_path = medium_path.with_name("generation_report.json")
    easy, medium, original_hard = map(load_level, (easy_path, medium_path, hard_path))
    report = json.loads(report_path.read_text())
    generator = LevelGenerator(GenerationSpec(**report["generation_spec"]))
    if asdict(generator.generate(1)) != asdict(medium):
        raise ValueError("generator does not reproduce the confirmed Medium artifact")
    hard = generator.generate(0)
    confirmation = next(
        row
        for row in report["confirmation_attempts"]
        if row["level_id"] == hard.level_id
    )
    if not 0.10 <= confirmation["ci_lower"] <= confirmation["ci_upper"] <= 0.40:
        raise ValueError(
            "compatible candidate lacks confirmation inside the Hard range"
        )
    profiles = {
        "easy": easy,
        "medium": medium,
        "hard_compatible": hard,
        "training_control": level_from_checkpoint(final_metadata["level_config"]),
    }
    for level in profiles.values():
        PolicyValueContext.from_level_config(level)
    provenance = {
        "files": {
            str(p): file_sha256(p)
            for p in (easy_path, medium_path, hard_path, report_path)
        },
        "hard_note": f"Accepted Hard artifact has {original_hard.num_colors} colors, unsupported by AZ-1. hard_compatible is candidate 0 from the saved Medium generation search; original artifacts unchanged.",
        "hard_confirmation": confirmation,
        "confirmation_search": report["target_generation_config"],
    }
    return profiles, provenance


@dataclass(frozen=True)
class EvaluationConfig:
    episodes: int = 10
    easy_episodes: int = 5
    control_episodes: int = 5
    simulations: int = 8
    env_seed_start: int = 31000000
    agent_seed_start: int = 41000000

    def __post_init__(self):
        for name, value in asdict(self).items():
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        count = max(self.episodes, self.easy_episodes, self.control_episodes)
        for start, size in (
            (self.env_seed_start, count),
            (self.agent_seed_start, 4000000 + 16 * count),
        ):
            for low, high in PARTITIONS.values():
                if start <= high and start + size - 1 >= low:
                    raise ValueError("evaluation seeds overlap reserved RL partitions")
        if (
            self.env_seed_start <= self.agent_seed_start + 4000000 + 16 * count
            and self.agent_seed_start <= self.env_seed_start + count - 1
        ):
            raise ValueError("environment and agent seed ranges overlap")

    def count(self, profile: str) -> int:
        return {
            "easy": self.easy_episodes,
            "training_control": self.control_episodes,
        }.get(profile, self.episodes)


class EvaluationAgent:
    """Bind after runner reset, verify pairing, and time only action selection."""

    def __init__(self, name, env, model, seed, simulations, identities, key):
        self.name, self.env, self.model = name, env, model
        self.seed, self.simulations, self.identities, self.key = (
            seed,
            simulations,
            identities,
            key,
        )
        self.agent = None
        self.calls, self.seconds = 0, 0.0
        self.pairing_verified, self.integrity_error = False, None
        self.neural_evaluations = self.sampled_transitions = 0
        self.initial_value = None

    def select_action(self, observation):
        if self.agent is None:
            info = self.env._reset_info()
            identity = (
                observation_fingerprint(observation),
                info["board_seed"],
                info["engine_seed"],
                info["level_config_id"],
            )
            if self.identities.setdefault(self.key, identity) != identity:
                self.integrity_error = "initial-state pairing mismatch"
                raise ValueError(self.integrity_error)
            self.pairing_verified = True
            if self.name == "Random":
                self.agent = RandomAgent(self.seed)
            elif self.name == "Greedy":
                self.agent = GreedyAgent()
            elif self.name == "Classic_MCTS_GO":
                self.agent = StochasticMCTSAgent(
                    MCTSSimulator(self.env),
                    simulations=self.simulations,
                    rollout_policy="greedy",
                    value_mode="outcome",
                    seed=self.seed,
                )
            else:
                context = PolicyValueContext.from_level_config(self.env.config)
                self.agent = NeuralMCTSAgent(
                    MCTSSimulator(self.env),
                    self.model,
                    context,
                    simulations=self.simulations,
                    seed=self.seed,
                    evaluation=True,
                )
                encoded = self.agent.encoder.encode(observation, context)
                with torch.inference_mode():
                    self.initial_value = float(
                        self.model(
                            torch.from_numpy(encoded.spatial[None]),
                            torch.from_numpy(encoded.global_features[None]),
                        )[1].item()
                    )
        search = self.name not in ("Random", "Greedy")
        before = complete_state_fingerprint(self.env) if search else None
        self.calls += 1
        started = perf_counter()
        try:
            action = self.agent.select_action(observation)
            if isinstance(self.agent, NeuralMCTSAgent):
                self.neural_evaluations += (
                    self.agent.last_search_stats.neural_evaluations
                )
                self.sampled_transitions += (
                    self.agent.last_search_stats.sampled_transitions
                )
            return action
        finally:
            self.seconds += perf_counter() - started
            if search and complete_state_fingerprint(self.env) != before:
                self.integrity_error = "search mutated the live environment"
                raise RuntimeError(self.integrity_error)


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        writer.writerows({k: _csv_value(v) for k, v in row.items()} for row in rows)


def comparison_rows(records, extras) -> list[dict]:
    rows = summarize(records, extras)
    by_key = {
        (r["level_id"], r["agent_name"]): r for r in summarize_level_episodes(records)
    }
    for row in rows:
        group = by_key[row["level_id"], row["agent_name"]]
        selected = [
            e
            for e in records
            if e.level_id == row["level_id"] and e.agent_name == row["agent_name"]
        ]
        row["mean_moves_used"] = group["mean_moves_used"]
        row["mean_score_tiles_cleared"] = sum(e.total_cleared for e in selected) / len(
            selected
        )
        row["mean_return"] = group["mean_total_reward"]
    return rows


def run_benchmark(
    config: EvaluationConfig,
    profiles: dict[str, LevelConfig],
    checkpoints: dict[str, Path],
    output: Path,
    *,
    profile_provenance: dict | None = None,
    training_metrics: Path | None = None,
) -> list[dict]:
    if output.exists():
        raise FileExistsError(output)
    models, metadata = {}, {}
    for name, path in checkpoints.items():
        models[name], metadata[name] = load_policy_value_checkpoint(path)
    if not models:
        raise ValueError("at least one checkpoint is required")
    reference = next(iter(metadata.values()))
    if (
        len({m["level_sha256"] for m in metadata.values()}) != 1
        or len({m["stream_seeds"]["network"] for m in metadata.values()}) != 1
    ):
        raise ValueError(
            "checkpoint progression must share training level and initialization"
        )
    models["Neural_untrained"] = (
        PolicyValueNetwork(reference["stream_seeds"]["network"])
        .eval()
        .requires_grad_(False)
    )
    model_hashes = {name: model_sha256(model) for name, model in models.items()}
    for level in profiles.values():
        PolicyValueContext.from_level_config(level)
    names = (*BASELINES, *checkpoints)
    output.mkdir(parents=True, exist_ok=False)
    manifest = {
        "label": "held-out development evaluation; not final test",
        "status": "running",
        "config": asdict(config),
        "profiles": {name: asdict(level) for name, level in profiles.items()},
        "profile_provenance": profile_provenance,
        "checkpoints": metadata,
        "model_sha256": model_hashes,
        "package_source_sha256": _source_fingerprint(),
        "versions": {"numpy": np.__version__, "torch": str(torch.__version__)},
        "seed_rule": "same environment seeds within profile; neural checkpoints and untrained share agent stream; Random/Greedy/classic have separate slots",
        "denominators": {
            "win_rate": "all attempted records; errors and time limits reported separately",
            "moves": "available moves_used, including losses",
            "decision_time": "total action selection seconds / selection calls; excludes model load/binding/integrity checks",
            "score": "tiles cleared, a score proxy; environment has no separate points field",
            "return": "sparse environment reward sum",
        },
        "planned_episodes": sum(config.count(p) for p in profiles) * len(names),
        "records_written": 0,
    }
    if training_metrics is not None:
        metrics = json.loads(training_metrics.read_text())
        episodes = [e for r in metrics for e in r["episodes"]]
        manifest["training_audit"] = {
            "metrics_path": str(training_metrics),
            "metrics_sha256": file_sha256(training_metrics),
            "rounds": len(metrics),
            "episodes": len(episodes),
            "outcomes": dict(Counter(e["outcome"] for e in episodes)),
            "last_policy_loss": metrics[-1]["policy_loss"],
            "last_value_loss": metrics[-1]["value_loss"],
            "interpretation": "Blocker-free collect-any-color task; all observed training targets +1. Near-zero value MSE is consistent with a constant saturated critic, not evidence of discrimination or gameplay improvement.",
        }
    atomic_json(output / "manifest.json", manifest)
    identities, records, extras = {}, [], []
    started = perf_counter()
    try:
        with (output / "episodes.csv").open("x", newline="") as handle:
            writer = None
            for profile_index, (profile, level) in enumerate(profiles.items()):
                for index in range(config.count(profile)):
                    seed = config.env_seed_start + index
                    for name in names:
                        slot = BASELINES.index(name) if name in BASELINES else 3
                        agent_seed = (
                            config.agent_seed_start
                            + profile_index * 1000000
                            + index * 16
                            + slot
                        )
                        env = Match3LevelEnv(
                            level.level_id, level_config=level, max_cascades=200
                        )
                        bound = EvaluationAgent(
                            name,
                            env,
                            models.get(name),
                            agent_seed,
                            config.simulations,
                            identities,
                            (profile, seed),
                        )
                        episode_started = perf_counter()
                        try:
                            result = run_episode(
                                env,
                                bound,
                                seed=seed,
                                max_steps=level.move_limit + 1,
                                agent_name=name,
                                agent_seed=agent_seed,
                                experiment_id=output.name,
                                episode_id=f"{profile}:{seed}:{name}",
                            )
                        finally:
                            env.close()
                        result = EpisodeResult(
                            **{**asdict(result), "level_id": profile}
                        )
                        extra = {
                            "episode_id": result.episode_id,
                            "initial_fingerprint": identities.get(
                                (profile, seed), (None,)
                            )[0],
                            "pairing_verified": bound.pairing_verified,
                            "action_selection_calls": bound.calls,
                            "decision_time_seconds": bound.seconds,
                            "episode_time_seconds": perf_counter() - episode_started,
                            "implementation_version": "az4-evaluation-v1",
                            "neural_evaluations": bound.neural_evaluations,
                            "sampled_transitions": bound.sampled_transitions
                            if slot == 3
                            else None,
                            "initial_neural_value": bound.initial_value,
                            "checkpoint_sha256": metadata.get(name, {}).get("sha256"),
                            "simulations": config.simulations if slot >= 2 else 0,
                        }
                        row = {**asdict(result), **extra}
                        if writer is None:
                            writer = csv.DictWriter(handle, fieldnames=list(row))
                            writer.writeheader()
                        writer.writerow({k: _csv_value(v) for k, v in row.items()})
                        handle.flush()
                        os.fsync(handle.fileno())
                        records.append(result)
                        extras.append(extra)
                        manifest["records_written"] = len(records)
                        atomic_json(output / "manifest.json", manifest)
                        if bound.integrity_error:
                            raise RuntimeError(bound.integrity_error)
                    print(
                        f"{profile}: board {index + 1}/{config.count(profile)} complete",
                        flush=True,
                    )
        rows = comparison_rows(records, extras)
        write_csv(output / "summary.csv", rows)
        write_csv(
            output / "checkpoint_progression.csv",
            [r for r in rows if r["agent_name"] in checkpoints],
        )
        manifest["status"] = "complete"
        manifest["elapsed_seconds"] = perf_counter() - started
        for name, meta in metadata.items():
            if file_sha256(checkpoints[name]) != meta["sha256"]:
                raise RuntimeError("checkpoint changed during evaluation")
        for name, model in models.items():
            if model.training or any(p.requires_grad for p in model.parameters()):
                raise RuntimeError(f"evaluation model is not frozen: {name}")
            if model_sha256(model) != model_hashes[name]:
                raise RuntimeError(f"evaluation mutated model weights: {name}")
        report = render_report(rows, manifest, records, extras)
        (output / "report.md").write_text(report)
        return rows
    except BaseException:
        manifest["status"] = "interrupted_or_failed"
        raise
    finally:
        atomic_json(output / "manifest.json", manifest)


def render_report(rows, manifest, records, extras) -> str:
    lines = [
        "# Neural MCTS development evaluation",
        "",
        "Paired fresh seeds; exploratory small-sample evaluation, not final testing. No significance or causal claims.",
        "",
        "| Profile | Agent | Wins/n | Win % [Wilson 95%] | Moves | Tiles (score proxy) | Decision ms |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        decision = (
            "NA"
            if row["mean_decision_time_seconds"] is None
            else f"{1000 * row['mean_decision_time_seconds']:.2f}"
        )
        lines.append(
            f"| {row['level_id']} | {row['agent_name']} | {row['wins']}/{row['attempted']} | {100 * row['win_rate']:.0f} [{100 * row['win_rate_ci95_low']:.0f}, {100 * row['win_rate_ci95_high']:.0f}] | {row['mean_moves_used']} | {row['mean_score_tiles_cleared']:.1f} | {decision} |"
        )
    lines += [
        "",
        "## Training audit",
        "",
        json.dumps(manifest.get("training_audit", {}), indent=2),
        "",
        "Critic probes (initial-state values averaged over boards):",
    ]
    for profile in manifest["profiles"]:
        for name in manifest["checkpoints"]:
            values = [
                extra["initial_neural_value"]
                for result, extra in zip(records, extras)
                if result.level_id == profile
                and result.agent_name == name
                and extra["initial_neural_value"] is not None
            ]
            if values:
                lines.append(f"- {profile}, {name}: {np.mean(values):.8f}")
    lines += ["", "## Paired training comparison", ""]
    final_name = max(
        manifest["checkpoints"],
        key=lambda name: manifest["checkpoints"][name]["rounds_completed"],
    )
    for profile in manifest["profiles"]:

        def outcomes(name, profile=profile):
            return {
                r.seed: r.outcome
                for r in records
                if r.level_id == profile
                and r.agent_name == name
                and r.outcome in ("WON", "LOST")
            }

        trained, untrained = outcomes(final_name), outcomes("Neural_untrained")
        paired = set(trained) & set(untrained)
        gained = sum(trained[s] == "WON" and untrained[s] == "LOST" for s in paired)
        lost = sum(trained[s] == "LOST" and untrained[s] == "WON" for s in paired)
        lines.append(
            f"- {profile}: n={len(paired)}, trained-only wins={gained}, untrained-only wins={lost}, net wins={gained - lost:+d}"
        )
    lines += [
        "",
        "## Limits",
        "",
        str((manifest.get("profile_provenance") or {}).get("hard_note", "")),
        "",
        "One configuration per profile; profile labels are relative to historical classic MCTS and its recorded budget. Fresh boards do not establish generalization across layouts. Equal simulation counts do not equalize computation: classic MCTS rolls out to termination; neural MCTS evaluates new leaves. Timing is measured sequentially on this machine. Checkpoint progression is exploratory and must not be used to select a winner and call these same boards final tests.",
        "",
        "The observed paired differences above describe this sample only. Training loss reductions and a saturated +1 critic do not establish improved gameplay.",
    ]
    return "\n".join(lines) + "\n"
