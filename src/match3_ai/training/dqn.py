"""Bounded CPU DQN training with separate gameplay and optimization records."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from match3_ai.agents.dqn_agent import DQNAgent, checkpoint_to_numpy
from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.agents.replay_buffer import ReplayBuffer
from match3_ai.evaluation.analytics import AnalyticsCollector, export_evaluation
from match3_ai.evaluation.experiments import summarize_level_episodes
from match3_ai.evaluation.runner import run_episode
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.training.reward_shaping import training_reward
from match3_ai.training.seed_policy import (
    PARTITIONS,
    seed_policy_metadata,
    validate_evaluation_partition,
    validate_partition,
)


@dataclass(frozen=True)
class TrainingConfig:
    level_id: str = "blocker-mayhem"
    environment_steps: int = 1000
    seed: int = 5700
    train_seed_start: int = 1000000
    evaluation_seed_start: int = 110000
    evaluation_agent_seed_start: int = 210000
    evaluation_episodes: int = 5
    evaluation_every: int = 500
    max_episode_steps: int = 50
    replay_capacity: int = 50000
    batch_size: int = 64
    warmup_steps: int = 1000
    train_every: int = 4
    target_sync_updates: int = 1000
    epsilon_start: float = 1.0
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 50000
    gamma: float = 0.99
    learning_rate: float = 3e-4
    gradient_clip: float = 10.0
    hidden_dims: tuple[int, ...] = (256, 128)
    max_cascades: int = 200
    max_initialization_attempts: int = 1000
    seed_policy: str | None = None
    evaluate_step_zero: bool = False
    baseline_every_checkpoint: bool = True
    reward_mode: str = "sparse"
    shaping_beta: float = 0.5
    architecture: str = "mlp"
    algorithm: str = "vanilla"
    evaluation_partition: str = "validation"

    def __post_init__(self):
        if self.architecture not in ("mlp", "cnn", "dueling"):
            raise ValueError("architecture must be mlp, cnn or dueling")
        if self.algorithm not in ("vanilla", "double"):
            raise ValueError("algorithm must be vanilla or double")
        if self.reward_mode not in ("sparse", "pbrs"):
            raise ValueError("reward_mode must be sparse or pbrs")
        if (
            isinstance(self.shaping_beta, (bool, np.bool_))
            or not np.isfinite(self.shaping_beta)
            or self.shaping_beta < 0
        ):
            raise ValueError("shaping_beta must be finite and non-negative")
        if (
            type(self.evaluate_step_zero) is not bool
            or type(self.baseline_every_checkpoint) is not bool
        ):
            raise ValueError("evaluation switches must be booleans")
        if not self.baseline_every_checkpoint and not self.evaluate_step_zero:
            raise ValueError("baseline reuse requires step-zero evaluation")
        for key in (
            "environment_steps",
            "evaluation_episodes",
            "evaluation_every",
            "max_episode_steps",
            "replay_capacity",
            "batch_size",
            "warmup_steps",
            "train_every",
            "target_sync_updates",
            "epsilon_decay_steps",
            "max_cascades",
            "max_initialization_attempts",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be a positive integer")
        for key in (
            "seed",
            "train_seed_start",
            "evaluation_seed_start",
            "evaluation_agent_seed_start",
        ):
            if type(getattr(self, key)) is not int or getattr(self, key) < 0:
                raise ValueError(f"{key} must be a non-negative integer")
        for key in ("epsilon_start", "epsilon_end", "gamma"):
            if not np.isfinite(getattr(self, key)) or not 0 <= getattr(self, key) <= 1:
                raise ValueError(f"{key} must be in [0,1]")
        for key in ("learning_rate", "gradient_clip"):
            if not np.isfinite(getattr(self, key)) or getattr(self, key) <= 0:
                raise ValueError(f"{key} must be positive and finite")
        if (
            self.epsilon_end > self.epsilon_start
            or self.batch_size > self.replay_capacity
        ):
            raise ValueError("invalid epsilon schedule or replay batch size")
        if (
            not isinstance(self.hidden_dims, tuple)
            or not self.hidden_dims
            or any(type(x) is not int or x <= 0 for x in self.hidden_dims)
        ):
            raise ValueError("hidden_dims must contain positive integers")
        train_end = self.train_seed_start + self.environment_steps + 1
        eval_end = self.evaluation_seed_start + self.evaluation_episodes
        if max(self.train_seed_start, self.evaluation_seed_start) < min(
            train_end, eval_end
        ):
            raise ValueError(
                "training and evaluation environment seeds must be disjoint"
            )
        if self.seed_policy not in (None, "sprint5_8"):
            raise ValueError("unknown seed policy")
        if self.seed_policy == "sprint5_8":
            validate_partition(
                "training", self.train_seed_start, self.environment_steps
            )
            validate_evaluation_partition(
                self.evaluation_partition,
                self.evaluation_seed_start,
                self.evaluation_episodes,
            )
            agent_end = self.evaluation_agent_seed_start + self.evaluation_episodes - 1
            if any(
                self.evaluation_agent_seed_start <= high and agent_end >= low
                for low, high in PARTITIONS.values()
            ):
                raise ValueError(
                    "baseline agent seeds must be separate from environment seeds"
                )

    def epsilon(self, step):
        fraction = min(max(step, 0) / self.epsilon_decay_steps, 1)
        return self.epsilon_start + fraction * (self.epsilon_end - self.epsilon_start)


def _fingerprint(agent):
    return hashlib.sha256(
        b"".join(
            parameter.detach().numpy().tobytes()
            for parameter in agent.online.parameters()
        )
    ).hexdigest()


def _environment(config):
    return Match3LevelEnv(
        config.level_id,
        max_cascades=config.max_cascades,
        max_initialization_attempts=config.max_initialization_attempts,
    )


def evaluate_agents(
    config,
    agent,
    *,
    step,
    run_id,
    collector=None,
    names=("DQNAgent", "RandomAgent", "GreedyAgent"),
):
    """Evaluate fixed validation seeds without learning or exploration RNG use."""
    if config.seed_policy == "sprint5_8":
        validate_evaluation_partition(
            config.evaluation_partition,
            config.evaluation_seed_start,
            config.evaluation_episodes,
        )
    if (
        not names
        or len(set(names)) != len(names)
        or set(names) - {"DQNAgent", "RandomAgent", "GreedyAgent"}
    ):
        raise ValueError("unknown or duplicate evaluation agents")
    before = _fingerprint(agent)
    rng_before = deepcopy(agent.rng.bit_generator.state)
    was_training = agent.online.training
    agent.online.eval()
    records = []
    try:
        for name in names:
            for index in range(config.evaluation_episodes):
                agent_seed = config.evaluation_agent_seed_start + index
                policy = (
                    agent
                    if name == "DQNAgent"
                    else RandomAgent(agent_seed)
                    if name == "RandomAgent"
                    else GreedyAgent()
                )
                env = _environment(config)
                try:
                    records.append(
                        run_episode(
                            env,
                            policy,
                            seed=config.evaluation_seed_start + index,
                            max_steps=config.max_episode_steps,
                            experiment_id=run_id,
                            episode_id=f"{run_id}:{step}:{name}:{index}",
                            agent_name=name,
                            agent_seed=agent.seed if name == "DQNAgent" else agent_seed,
                            collector=collector,
                        )
                    )
                finally:
                    env.close()
    finally:
        agent.online.train(was_training)
    if _fingerprint(agent) != before or agent.rng.bit_generator.state != rng_before:
        raise RuntimeError("evaluation modified policy parameters or exploration RNG")
    return records


def train_dqn(config: TrainingConfig, output_dir, *, resume_from=None, observer=None):
    """Train to an absolute step budget; final checkpoints end collection episodes.

    Budget/safety truncations bootstrap from the returned final observation. Resume
    begins a new episode, restoring optimizer, replay, target and local RNG states;
    it does not restore an unrecorded mid-episode engine state.
    """
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    streams = np.random.SeedSequence(config.seed).spawn(2)
    agent_seed, replay_seed = (int(stream.generate_state(1)[0]) for stream in streams)
    agent = DQNAgent(
        agent_seed,
        config.hidden_dims,
        config.learning_rate,
        architecture=config.architecture,
        algorithm=config.algorithm,
    )
    replay = ReplayBuffer(config.replay_capacity, replay_seed)
    step, updates, synchronizations, episode_index = 0, 0, 0, 0
    if resume_from is not None:
        agent, payload = DQNAgent.load_checkpoint(resume_from)
        previous = dict(payload["training_config"])
        previous.setdefault("seed_policy", None)
        previous.setdefault("evaluate_step_zero", False)
        previous.setdefault("baseline_every_checkpoint", True)
        previous.setdefault("reward_mode", "sparse")
        previous.setdefault("shaping_beta", 0.5)
        previous.setdefault("architecture", "mlp")
        previous.setdefault("algorithm", "vanilla")
        previous.setdefault("evaluation_partition", "validation")
        current = asdict(config)
        previous.pop("environment_steps", None)
        current.pop("environment_steps")
        if (
            previous != current
            or not payload["training_state"]
            or agent.architecture != config.architecture
        ):
            raise ValueError("resume requires a compatible full training checkpoint")
        state = checkpoint_to_numpy(payload["training_state"])
        replay.load_state_dict(state["replay"])
        step, updates, synchronizations, episode_index = (
            state[key]
            for key in ("step", "updates", "synchronizations", "episode_index")
        )
        if step >= config.environment_steps:
            raise ValueError("resume budget must exceed checkpoint step")
    start_step = step
    initial_fingerprint = _fingerprint(agent)
    env = _environment(config)
    expected_level_id = hashlib.sha256(
        json.dumps(asdict(env.config), sort_keys=True).encode()
    ).hexdigest()
    run_id = hashlib.sha256(
        json.dumps(
            {
                "config": asdict(config),
                "initial_weights": initial_fingerprint,
                "resume_step": step,
                "level_config_id": expected_level_id,
            },
            sort_keys=True,
        ).encode()
    ).hexdigest()[:24]
    output.mkdir(parents=True, exist_ok=False)
    collector = AnalyticsCollector()
    evaluation_records = []
    episode_records = []
    evaluation_rounds = []
    old_threads = torch.get_num_threads()
    old_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = perf_counter()
    final_checkpoint = output / "checkpoint_final.pt"
    losses = []
    training_budget_truncations = 0
    evaluation_seconds = 0.0
    completed_episodes = training_wins = 0
    completed_reward = 0.0
    completed_shaped_reward = 0.0
    action = None
    fields = [
        "environment_step",
        "action_id",
        "valid_action",
        "optimizer_updates",
        "epsilon",
        "replay_size",
        "td_loss",
        "q_mean",
        "q_min",
        "q_max",
        "gradient_norm",
        "episode_reward",
        "episode_outcome",
        "evaluation_win_rate",
        "training_seconds",
        "evaluation_seconds",
        "completed_episodes",
        "training_win_rate",
        "mean_episode_reward",
        "target_synchronizations",
        "sparse_reward",
        "shaping_bonus",
        "shaped_reward",
        "potential_current",
        "potential_next",
        "episode_sparse_return",
        "episode_shaped_return",
    ]

    seeds = {
        "master_seed": config.seed,
        "agent_seed": agent_seed,
        "replay_seed": replay_seed,
        "training_environment_seed_start": config.train_seed_start,
    }

    def evaluate_milestone(milestone):
        nonlocal evaluation_seconds
        evaluation_started = perf_counter()
        round_collector = AnalyticsCollector()
        names = (
            ("DQNAgent", "RandomAgent", "GreedyAgent")
            if (config.baseline_every_checkpoint or milestone == 0)
            else ("DQNAgent",)
        )
        round_records = evaluate_agents(
            config,
            agent,
            step=milestone,
            run_id=run_id,
            collector=round_collector,
            names=names,
        )
        evaluation_seconds += perf_counter() - evaluation_started
        evaluation_records.extend(round_records)
        for record in round_collector.episodes:
            collector.record_episode(record)
        for record in round_collector.moves:
            collector.record_move(record)
        round_summary = summarize_level_episodes(round_records)
        evaluation_rounds.append(
            {
                "environment_step": milestone,
                "summary": round_summary,
                "baseline_reference_step": None
                if config.baseline_every_checkpoint
                else 0,
            }
        )
        round_metadata = {
            "experiment_id": run_id,
            "training_step": milestone,
            "training_config": asdict(config),
            "seed_metadata": seeds,
            "level_config_id": level_config_id,
            "summary": round_summary,
            "seed_policy": seed_policy_metadata(config.evaluation_partition)
            if config.seed_policy
            else None,
            "evaluation_seconds_cumulative": evaluation_seconds,
            "training_seconds_cumulative": perf_counter() - started,
        }
        export_evaluation(
            round_collector, round_metadata, output / "validation" / f"step_{milestone}"
        )
        agent.save_checkpoint(
            output / f"policy_step_{milestone}.pt",
            training_config=asdict(config),
            training_step=milestone,
            validation_metrics=round_summary,
            seed_metadata=seeds,
        )
        with (output / f"training_episodes_step_{milestone}.json").open("x") as handle:
            json.dump(episode_records, handle)
        if observer is not None:
            observer(
                "milestone",
                {"environment_step": milestone, "optimizer_updates": updates},
            )
        rate = next(
            group["win_rate"]
            for group in round_summary
            if group["agent_name"] == "DQNAgent"
        )
        print(
            f"step={milestone} updates={updates} epsilon={config.epsilon(milestone):.4f} "
            f"validation_win_rate={rate:.3f} elapsed_seconds={perf_counter() - started:.2f}",
            flush=True,
        )
        return rate

    try:
        with (output / "training_metrics.csv").open(
            "x", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            observation, reset_info = env.reset(
                seed=config.train_seed_start + episode_index
            )
            checkpoint_probe = observation
            level_config_id = reset_info["level_config_id"]
            if resume_from is not None and state["level_config_id"] != level_config_id:
                raise ValueError("level configuration changed since checkpoint")
            episode_reward, episode_steps = 0.0, 0
            episode_shaped_return = episode_bonus_return = 0.0
            discounted_sparse = discounted_shaped = 0.0
            if config.evaluate_step_zero and start_step == 0:
                evaluate_milestone(0)
            while step < config.environment_steps:
                epsilon = config.epsilon(step)
                action = agent.select_action(observation, epsilon)
                if not observation["action_mask"][action]:
                    raise RuntimeError("DQN selected a masked action")
                next_observation, reward, terminated, truncated, info = env.step(action)
                if not info["valid_action"]:
                    raise RuntimeError(
                        "environment rejected a mask-valid training action"
                    )
                if terminated and info["level_status"] not in ("WON", "LOST"):
                    raise RuntimeError(
                        info.get("termination_reason") or "unknown level termination"
                    )
                step += 1
                episode_steps += 1
                episode_reward += reward
                at_budget = step == config.environment_steps
                safety_bound = episode_steps >= config.max_episode_steps
                artificial_truncation = (at_budget or safety_bound) and not (
                    terminated or truncated
                )
                truncated = truncated or artificial_truncation
                if artificial_truncation and at_budget:
                    training_budget_truncations += 1
                components = training_reward(
                    observation,
                    next_observation,
                    reward,
                    terminated=terminated,
                    truncated=truncated,
                    mode=config.reward_mode,
                    beta=config.shaping_beta,
                    gamma=config.gamma,
                )
                episode_shaped_return += components.shaped_reward
                episode_bonus_return += components.shaping_bonus
                discounted_sparse += config.gamma ** (episode_steps - 1) * reward
                discounted_shaped += (
                    config.gamma ** (episode_steps - 1) * components.shaped_reward
                )
                replay.add(
                    observation,
                    action,
                    components.shaped_reward,
                    next_observation,
                    terminated,
                    truncated,
                )
                observation = next_observation
                metrics = {}
                if (
                    step >= config.warmup_steps
                    and len(replay) >= config.batch_size
                    and step % config.train_every == 0
                ):
                    metrics = agent.update(
                        replay.sample(config.batch_size),
                        gamma=config.gamma,
                        gradient_clip=config.gradient_clip,
                    )
                    losses.append(metrics["td_loss"])
                    updates += 1
                    if updates % config.target_sync_updates == 0:
                        agent.synchronize_target()
                        synchronizations += 1
                done = terminated or truncated
                episode_outcome = None
                if done:
                    episode_outcome = (
                        info["level_status"] if terminated else "TIME_LIMIT"
                    )
                    episode_records.append(
                        {
                            "episode_index": episode_index,
                            "environment_seed": config.train_seed_start + episode_index,
                            "environment_step": step,
                            "reward": episode_reward,
                            "sparse_return": episode_reward,
                            "shaped_return": episode_shaped_return,
                            "shaping_bonus_return": episode_bonus_return,
                            "discounted_sparse_return": discounted_sparse,
                            "discounted_shaped_return": discounted_shaped,
                            "final_potential": components.potential_next,
                            "outcome": episode_outcome,
                            "steps": episode_steps,
                            "terminated": terminated,
                            "truncated": truncated,
                        }
                    )
                    episode_index += 1
                    if observer is not None:
                        observer("episode", episode_records[-1])
                    if terminated:
                        completed_episodes += 1
                        training_wins += info["level_status"] == "WON"
                        completed_reward += episode_reward
                        completed_shaped_reward += episode_shaped_return
                evaluation_win_rate = None
                if step % config.evaluation_every == 0 or at_budget:
                    evaluation_win_rate = evaluate_milestone(step)
                writer.writerow(
                    {
                        "environment_step": step,
                        "action_id": action,
                        "valid_action": True,
                        "optimizer_updates": updates,
                        "epsilon": epsilon,
                        "replay_size": len(replay),
                        **metrics,
                        "episode_reward": episode_reward if done else None,
                        "episode_outcome": episode_outcome,
                        "evaluation_win_rate": evaluation_win_rate,
                        "training_seconds": perf_counter() - started,
                        "evaluation_seconds": evaluation_seconds,
                        "completed_episodes": completed_episodes,
                        "training_win_rate": training_wins / completed_episodes
                        if completed_episodes
                        else None,
                        "mean_episode_reward": completed_reward / completed_episodes
                        if completed_episodes
                        else None,
                        "target_synchronizations": synchronizations,
                        "sparse_reward": components.sparse_reward,
                        "shaping_bonus": components.shaping_bonus,
                        "shaped_reward": components.shaped_reward,
                        "potential_current": components.potential_current,
                        "potential_next": components.potential_next,
                        "episode_sparse_return": episode_reward if done else None,
                        "episode_shaped_return": episode_shaped_return
                        if done
                        else None,
                    }
                )
                if evaluation_win_rate is not None:
                    handle.flush()
                if observer is not None and (step % 256 == 0 or at_budget):
                    handle.flush()
                    observer(
                        "progress",
                        {"environment_step": step, "optimizer_updates": updates},
                    )
                if done and not at_budget:
                    action = None
                    observation, _ = env.reset(
                        seed=config.train_seed_start + episode_index
                    )
                    episode_reward, episode_steps = 0.0, 0
                    episode_shaped_return = episode_bonus_return = 0.0
                    discounted_sparse = discounted_shaped = 0.0
            agent.save_checkpoint(
                final_checkpoint,
                training_config=asdict(config),
                training_step=step,
                validation_metrics=evaluation_rounds[-1]["summary"],
                seed_metadata=seeds,
                training_state={
                    "replay": replay.state_dict(),
                    "step": step,
                    "updates": updates,
                    "synchronizations": synchronizations,
                    "episode_index": episode_index,
                    "level_config_id": level_config_id,
                },
            )
            loaded, payload = DQNAgent.load_checkpoint(final_checkpoint)
            checkpoint_valid = loaded.select_action(
                checkpoint_probe
            ) == agent.select_action(checkpoint_probe)
            if not checkpoint_valid or _fingerprint(loaded) != _fingerprint(agent):
                raise RuntimeError("checkpoint inference round trip failed")
            metadata = {
                "experiment_id": run_id,
                "training_config": asdict(config),
                "seed_metadata": seeds,
                "level_config_id": level_config_id,
                "training_start_step": start_step,
                "environment_steps": step,
                "optimizer_updates": updates,
                "target_synchronizations": synchronizations,
                "replay_size": len(replay),
                "invalid_actions": 0,
                "evaluation_invalid_actions": sum(
                    record.invalid_actions for record in evaluation_records
                ),
                "evaluation_errors": sum(
                    record.outcome == "ERROR" for record in evaluation_records
                ),
                "evaluation_time_limits": sum(
                    record.outcome == "TIME_LIMIT" for record in evaluation_records
                ),
                "training_budget_truncations": training_budget_truncations,
                "initial_weights": initial_fingerprint,
                "final_weights": _fingerprint(agent),
                "parameters_changed": _fingerprint(agent) != initial_fingerprint,
                "finite_losses": bool(losses) and bool(np.isfinite(losses).all()),
                "mean_td_loss": float(np.mean(losses)) if losses else None,
                "checkpoint_inference_valid": checkpoint_valid,
                "checkpoint": str(final_checkpoint),
                "training_seconds": perf_counter() - started,
                "evaluation_seconds": evaluation_seconds,
                "collection_optimization_seconds": perf_counter()
                - started
                - evaluation_seconds,
                "completed_training_episodes": completed_episodes,
                "training_wins": training_wins,
                "training_win_rate": training_wins / completed_episodes
                if completed_episodes
                else None,
                "mean_training_episode_reward": completed_reward / completed_episodes
                if completed_episodes
                else None,
                "mean_training_episode_shaped_return": completed_shaped_reward
                / completed_episodes
                if completed_episodes
                else None,
                "training_reward_mode": config.reward_mode,
                "evaluation_reward_mode": "sparse",
                "seed_policy": seed_policy_metadata(config.evaluation_partition)
                if config.seed_policy == "sprint5_8"
                else None,
                "torch_version": str(torch.__version__),
                "numpy_version": np.__version__,
                "python_version": platform.python_version(),
                "device": "cpu",
                "architecture": agent.architecture,
                "parameter_count": sum(
                    parameter.numel() for parameter in agent.online.parameters()
                ),
                "torch_threads": 1,
                "evaluation_rounds": evaluation_rounds,
                "limitations": "bounded smoke/pilot; evaluation seeds reused across checkpoints; no improvement or convergence claim",
            }
            export_evaluation(collector, metadata, output / "evaluation")
            (output / "training_episodes.json").write_text(
                json.dumps(episode_records, indent=2) + "\n", encoding="utf-8"
            )
            (output / "training_summary.json").write_text(
                json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
            )
    except BaseException as error:
        (output / "training_error.json").write_text(
            json.dumps(
                {
                    "environment_step": step,
                    "episode_index": episode_index,
                    "environment_seed": config.train_seed_start + episode_index,
                    "action_id": action,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                    "context": getattr(error, "context", {}),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        raise
    finally:
        env.close()
        torch.set_num_threads(old_threads)
        torch.use_deterministic_algorithms(old_determinism)
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--level-id", default="blocker-mayhem")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=5700)
    parser.add_argument("--resume-from", type=Path)
    args = parser.parse_args()
    config = TrainingConfig(
        level_id=args.level_id,
        environment_steps=args.steps,
        seed=args.seed,
        replay_capacity=2048,
        batch_size=32,
        warmup_steps=64,
        target_sync_updates=50,
        evaluation_every=500,
        evaluation_episodes=10,
        epsilon_decay_steps=1000,
        epsilon_end=0.1,
    )
    result = train_dqn(config, args.output_dir, resume_from=args.resume_from)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
