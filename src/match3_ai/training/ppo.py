"""Bounded masked PPO smoke training using existing level evaluation contracts."""

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

from match3_ai.agents.ppo_agent import PPOAgent
from match3_ai.evaluation.analytics import AnalyticsCollector, export_evaluation
from match3_ai.evaluation.experiments import summarize_level_episodes
from match3_ai.evaluation.runner import run_episode
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.training.ppo_update import (
    generalized_advantages,
    optimize_ppo,
    rollout_diagnostics,
)
from match3_ai.training.seed_policy import (
    PARTITIONS,
    seed_policy_metadata,
    validate_evaluation_partition,
    validate_partition,
)


@dataclass(frozen=True)
class PPOTrainingConfig:
    level_id: str = "blocker-mayhem"
    environment_steps: int = 5000
    seed: int = 5913
    evaluation_seed_start: int = 300000
    evaluation_episodes: int = 100
    rollout_steps: int = 256
    optimization_epochs: int = 4
    minibatch_size: int = 64
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_ratio: float = 0.2
    value_coefficient: float = 0.5
    entropy_coefficient: float = 0.01
    learning_rate: float = 3e-4
    gradient_clip: float = 0.5
    hidden_dims: tuple[int, ...] = (256, 128)
    max_episode_steps: int = 50
    max_cascades: int = 200
    max_initialization_attempts: int = 1000
    evaluation_every: int | None = None
    evaluation_partition: str = "validation"

    def __post_init__(self):
        if self.evaluation_every is not None and (
            type(self.evaluation_every) is not int or self.evaluation_every < 1
        ):
            raise ValueError("evaluation_every must be a positive integer or None")
        for name in (
            "environment_steps",
            "rollout_steps",
            "optimization_epochs",
            "minibatch_size",
            "evaluation_episodes",
            "max_episode_steps",
            "max_cascades",
            "max_initialization_attempts",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in ("gamma", "gae_lambda"):
            if (
                not np.isfinite(getattr(self, name))
                or not 0 <= getattr(self, name) <= 1
            ):
                raise ValueError(f"{name} must be in [0,1]")
        for name in ("learning_rate", "gradient_clip", "clip_ratio"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive and finite")
        if self.clip_ratio >= 1:
            raise ValueError("clip_ratio must be less than one")
        for name in ("value_coefficient", "entropy_coefficient"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            not isinstance(self.hidden_dims, tuple)
            or not self.hidden_dims
            or any(type(n) is not int or n < 1 for n in self.hidden_dims)
        ):
            raise ValueError("hidden_dims must be a tuple of positive integers")
        validate_evaluation_partition(
            self.evaluation_partition,
            self.evaluation_seed_start,
            self.evaluation_episodes,
        )


def fingerprint(agent):
    return hashlib.sha256(
        b"".join(p.detach().numpy().tobytes() for p in agent.network.parameters())
    ).hexdigest()


def environment(config):
    return Match3LevelEnv(
        config.level_id,
        max_cascades=config.max_cascades,
        max_initialization_attempts=config.max_initialization_attempts,
    )


def evaluate_ppo(config, agent, *, step, run_id, collector):
    """Use the existing runner without sampling or changing training RNG streams."""
    validate_evaluation_partition(
        config.evaluation_partition,
        config.evaluation_seed_start,
        config.evaluation_episodes,
    )
    before = fingerprint(agent)
    policy_state = agent.policy_rng.get_state().clone()
    shuffle_state = deepcopy(agent.shuffle_rng.bit_generator.state)
    was_training = agent.network.training
    records = []
    agent.network.eval()
    try:
        for index in range(config.evaluation_episodes):
            env = environment(config)
            try:
                records.append(
                    run_episode(
                        env,
                        agent,
                        seed=config.evaluation_seed_start + index,
                        max_steps=config.max_episode_steps,
                        experiment_id=run_id,
                        episode_id=f"{run_id}:{step}:PPOAgent:{index}",
                        agent_name="PPOAgent",
                        agent_seed=agent.seed,
                        collector=collector,
                    )
                )
            finally:
                env.close()
    finally:
        agent.network.train(was_training)
    if (
        fingerprint(agent) != before
        or not torch.equal(policy_state, agent.policy_rng.get_state())
        or agent.shuffle_rng.bit_generator.state != shuffle_state
    ):
        raise RuntimeError("evaluation mutated PPO weights or RNG streams")
    return records


def train_ppo(
    config: PPOTrainingConfig,
    output_dir,
    *,
    learning_rate_schedule=None,
    observer=None,
    evaluation_steps=None,
):
    """Train one sparse-reward CPU policy; rollouts are discarded after optimization."""
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    if evaluation_steps is not None:
        evaluation_steps = tuple(evaluation_steps)
        if (
            not evaluation_steps
            or any(type(s) is not int for s in evaluation_steps)
            or evaluation_steps != tuple(sorted(set(evaluation_steps)))
            or evaluation_steps[0] != 0
            or evaluation_steps[-1] != config.environment_steps
        ):
            raise ValueError("evaluation steps must increase from zero to final budget")
    env_stream, agent_stream = np.random.SeedSequence(config.seed).spawn(2)
    agent_seed = int(agent_stream.generate_state(1)[0])
    env_rng = np.random.default_rng(env_stream)
    agent = PPOAgent(agent_seed, config.hidden_dims, config.learning_rate)
    env = environment(config)
    initial_weights = fingerprint(agent)
    level_config_id = hashlib.sha256(
        json.dumps(asdict(env.config), sort_keys=True).encode()
    ).hexdigest()
    run_id = hashlib.sha256(
        json.dumps(asdict(config), sort_keys=True).encode()
    ).hexdigest()[:24]
    output.mkdir(parents=True, exist_ok=False)
    previous_threads = torch.get_num_threads()
    previous_determinism = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    started = perf_counter()
    evaluation_seconds = 0.0
    step = updates = rollouts = episode_steps = 0
    episode_reward = 0.0
    episodes, metrics, evaluation_rounds, rollout_records = [], [], [], []
    last_optimized_step = 0
    collector = AnalyticsCollector()
    seed_metadata = {
        "master_seed": config.seed,
        "agent_seed": agent_seed,
        "environment_stream": env_stream.state,
        **agent.stream_seeds,
        "environment_seed_policy": "local RNG draws within training partition",
    }

    def reset_episode():
        low, high = PARTITIONS["training"]
        seed = int(env_rng.integers(low, high + 1))
        validate_partition("training", seed, 1)
        observation, _ = env.reset(seed=seed)
        return observation, seed

    def evaluate(milestone):
        nonlocal evaluation_seconds
        start = perf_counter()
        records = evaluate_ppo(
            config, agent, step=milestone, run_id=run_id, collector=collector
        )
        evaluation_seconds += perf_counter() - start
        summary = summarize_level_episodes(records)[0]
        evaluation_rounds.append({"environment_step": milestone, **summary})
        if (
            summary["errors"]
            or summary["time_limits"]
            or any(r.invalid_actions for r in records)
        ):
            raise RuntimeError(
                "PPO validation has errors, time limits or invalid actions"
            )
        agent.save_checkpoint(
            output / f"policy_step_{milestone}.pt",
            training_config=asdict(config),
            training_progress=progress(),
            validation_metrics=summary,
        )
        loaded, _ = PPOAgent.load_checkpoint(output / f"policy_step_{milestone}.pt")
        if fingerprint(loaded) != fingerprint(agent):
            raise RuntimeError("milestone PPO checkpoint weights differ")
        if observer is not None:
            observer(
                "milestone",
                {
                    "environment_step": milestone,
                    "optimizer_updates": updates,
                    "evaluation": evaluation_rounds[-1],
                    "collector": collector,
                },
            )
        print(
            f"step={milestone} updates={updates} validation_win_rate={summary['win_rate']:.3f} elapsed_seconds={perf_counter() - started:.2f}",
            flush=True,
        )

    def progress():
        return {
            "environment_step": step,
            "last_optimized_step": last_optimized_step,
            "pending_rollout_steps": step - last_optimized_step,
            "optimizer_updates": updates,
            "rollouts": rollouts,
            "episodes_recorded": len(episodes),
            "environment_rng": deepcopy(env_rng.bit_generator.state),
            "training_seconds": perf_counter() - started,
            "evaluation_seconds": evaluation_seconds,
            "level_config_id": level_config_id,
            "seed_metadata": seed_metadata,
        }

    try:
        evaluate(0)
        observation, episode_seed = reset_episode()
        probe = deepcopy(observation)
        fields = (
            "environment_step",
            "episode_seed",
            "action",
            "reward",
            "log_prob",
            "value",
            "next_value",
            "terminated",
            "truncated",
            "valid_action",
        )
        with (output / "transitions.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            while step < config.environment_steps:
                data = {
                    name: []
                    for name in (
                        "features",
                        "masks",
                        "actions",
                        "log_probs",
                        "values",
                        "next_values",
                        "rewards",
                        "terminated",
                        "truncated",
                    )
                }
                for _ in range(
                    min(config.rollout_steps, config.environment_steps - step)
                ):
                    features, mask = agent.encoder.encode(observation)
                    action, log_prob, value = agent.act(observation)
                    if not mask[action]:
                        raise RuntimeError("PPO selected a masked action")
                    after, reward, terminated, truncated, info = env.step(action)
                    if not info["valid_action"] or reward not in (-1, 0, 1):
                        raise RuntimeError(
                            "PPO requires mask-valid actions and sparse rewards"
                        )
                    if terminated and info["level_status"] not in ("WON", "LOST"):
                        raise RuntimeError(
                            info.get("termination_reason") or "unknown termination"
                        )
                    step += 1
                    episode_steps += 1
                    episode_reward += reward
                    truncated = bool(
                        truncated
                        or (
                            not terminated
                            and (
                                episode_steps >= config.max_episode_steps
                                or step == config.environment_steps
                            )
                        )
                    )
                    next_value = 0.0 if terminated else agent.value(after)
                    for name, item in zip(
                        data,
                        (
                            features,
                            mask,
                            action,
                            log_prob,
                            value,
                            next_value,
                            reward,
                            bool(terminated),
                            truncated,
                        ),
                    ):
                        data[name].append(item)
                    writer.writerow(
                        dict(
                            zip(
                                fields,
                                (
                                    step,
                                    episode_seed,
                                    action,
                                    reward,
                                    log_prob,
                                    value,
                                    next_value,
                                    bool(terminated),
                                    truncated,
                                    True,
                                ),
                            )
                        )
                    )
                    observation = after
                    if terminated or truncated:
                        episodes.append(
                            {
                                "environment_step": step,
                                "environment_seed": episode_seed,
                                "steps": episode_steps,
                                "reward": episode_reward,
                                "outcome": info["level_status"]
                                if terminated
                                else "TIME_LIMIT",
                                "terminated": bool(terminated),
                                "truncated": truncated,
                            }
                        )
                        if observer is not None:
                            observer("episode", episodes[-1])
                        episode_steps, episode_reward = 0, 0.0
                        if step < config.environment_steps:
                            observation, episode_seed = reset_episode()
                    if step < config.environment_steps and (
                        (evaluation_steps is not None and step in evaluation_steps)
                        or (
                            evaluation_steps is None
                            and config.evaluation_every
                            and step % config.evaluation_every == 0
                        )
                    ):
                        evaluate(step)
                advantages, returns = generalized_advantages(
                    np.array(data["rewards"]),
                    np.array(data["values"]),
                    np.array(data["next_values"]),
                    np.array(data["terminated"], dtype=bool),
                    np.array(data["truncated"], dtype=bool),
                    gamma=config.gamma,
                    gae_lambda=config.gae_lambda,
                )
                batch = {
                    name: torch.from_numpy(np.asarray(data[name], dtype=dtype))
                    for name, dtype in (
                        ("features", np.float32),
                        ("masks", bool),
                        ("actions", np.int64),
                        ("log_probs", np.float32),
                    )
                }
                batch.update(
                    advantages=torch.from_numpy(advantages),
                    returns=torch.from_numpy(returns),
                )
                terminal_episodes = [
                    episode for episode in episodes if episode["terminated"]
                ]
                if learning_rate_schedule is not None:
                    for group in agent.optimizer.param_groups:
                        group["lr"] = learning_rate_schedule(step)
                rollout_records.append(
                    {
                        "environment_step": step,
                        "rollout": rollouts + 1,
                        **rollout_diagnostics(data["values"], returns, advantages),
                        "learning_rate": agent.optimizer.param_groups[0]["lr"],
                        "completed_episodes": len(terminal_episodes),
                        "training_win_rate": float(
                            np.mean(
                                [
                                    episode["outcome"] == "WON"
                                    for episode in terminal_episodes
                                ]
                            )
                        )
                        if terminal_episodes
                        else None,
                        "mean_training_episode_reward": float(
                            np.mean(
                                [episode["reward"] for episode in terminal_episodes]
                            )
                        )
                        if terminal_episodes
                        else None,
                    }
                )
                rows = optimize_ppo(
                    agent,
                    batch,
                    epochs=config.optimization_epochs,
                    minibatch_size=config.minibatch_size,
                    clip_ratio=config.clip_ratio,
                    value_coefficient=config.value_coefficient,
                    entropy_coefficient=config.entropy_coefficient,
                    gradient_clip=config.gradient_clip,
                )
                rollouts += 1
                last_optimized_step = step
                for row in rows:
                    updates += 1
                    metrics.append(
                        {
                            "environment_step": step,
                            "rollout": rollouts,
                            "optimizer_update": updates,
                            **row,
                        }
                    )
                if observer is not None:
                    handle.flush()
                    observer(
                        "rollout",
                        {
                            "environment_step": step,
                            "optimizer_updates": updates,
                            "rollout": rollout_records[-1],
                            "updates": metrics[-len(rows) :],
                        },
                    )
        evaluate(step)
        final_checkpoint = output / "checkpoint_final.pt"
        agent.save_checkpoint(
            final_checkpoint,
            training_config=asdict(config),
            training_progress=progress(),
            validation_metrics=evaluation_rounds[-1],
        )
        loaded, _ = PPOAgent.load_checkpoint(final_checkpoint)
        if fingerprint(loaded) != fingerprint(agent) or loaded.select_action(
            probe
        ) != agent.select_action(probe):
            raise RuntimeError("PPO checkpoint inference differs")
        summary = {
            "experiment_id": run_id,
            "algorithm": "masked_ppo",
            "training_config": asdict(config),
            "seed_metadata": seed_metadata,
            "seed_policy": seed_policy_metadata(config.evaluation_partition),
            "level_config_id": level_config_id,
            "environment_steps": step,
            "rollouts": rollouts,
            "optimizer_updates": updates,
            "parameter_count": sum(p.numel() for p in agent.network.parameters()),
            "initial_weights": initial_weights,
            "final_weights": fingerprint(agent),
            "invalid_actions": 0,
            "evaluation_invalid_actions": sum(
                e.invalid_actions for e in collector.episodes
            ),
            "evaluation_rounds": evaluation_rounds,
            "checkpoint_inference_valid": True,
            "checkpoint": str(final_checkpoint),
            "training_seconds": perf_counter() - started,
            "evaluation_seconds": evaluation_seconds,
            "collection_optimization_seconds": perf_counter()
            - started
            - evaluation_seconds,
            "device": "cpu",
            "torch_threads": 1,
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "python_version": platform.python_version(),
            "optimization_statistics": {
                key: {
                    "min": min(r[key] for r in metrics),
                    "max": max(r[key] for r in metrics),
                    "mean": float(np.mean([r[key] for r in metrics])),
                }
                for key in (
                    "policy_loss",
                    "value_loss",
                    "entropy",
                    "approximate_kl",
                    "clip_fraction",
                    "gradient_norm",
                )
            },
        }
        with (output / "optimization.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
            writer.writeheader()
            writer.writerows(metrics)
        with (output / "rollouts.csv").open("x", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rollout_records[0]))
            writer.writeheader()
            writer.writerows(rollout_records)
        for name, value in (
            ("training_summary.json", summary),
            ("training_episodes.json", episodes),
        ):
            with (output / name).open("x") as handle:
                json.dump(value, handle, indent=2, allow_nan=False)
        export_evaluation(collector, summary, output / "evaluation")
        return summary
    except BaseException as error:
        with (output / "training_error.json").open("x") as handle:
            json.dump(
                {
                    "environment_step": step,
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                handle,
                indent=2,
            )
        raise
    finally:
        env.close()
        torch.set_num_threads(previous_threads)
        torch.use_deterministic_algorithms(previous_determinism)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    config = PPOTrainingConfig()
    if args.review_only:
        print(json.dumps(asdict(config), indent=2))
    else:
        train_ppo(config, args.output_dir)


if __name__ == "__main__":
    main()
