"""Frozen-policy execution through the existing episode runner and analytics."""

import argparse
import hashlib
import json
import platform
from pathlib import Path
from time import perf_counter

import numpy as np

from match3_ai.evaluation.analytics import AnalyticsCollector, export_evaluation
from match3_ai.evaluation.behavior_data import observation_fingerprint
from match3_ai.evaluation.experiments import summarize_level_episodes
from match3_ai.evaluation.fresh_protocol import load_protocol, sha256
from match3_ai.evaluation.runner import run_episode


def load_policy(specification, protocol, index=0):
    from match3_ai.agents.dqn_agent import DQNAgent
    from match3_ai.agents.greedy_agent import GreedyAgent
    from match3_ai.agents.ppo_agent import PPOAgent
    from match3_ai.agents.random_agent import RandomAgent

    kind = specification["kind"]
    if kind == "random":
        return RandomAgent(protocol["random_agent_seeds"][index])
    if kind == "greedy":
        return GreedyAgent()
    if sha256(specification["checkpoint"]) != specification["sha256"]:
        raise ValueError("checkpoint differs from protocol")
    policy, payload = (PPOAgent if kind == "ppo" else DQNAgent).load_checkpoint(
        specification["checkpoint"]
    )
    if (
        payload["training_config"]["level_id"] != protocol["environment"]["level_id"]
        or payload["training_config"]["seed"] != specification["training_seed"]
    ):
        raise ValueError("checkpoint training configuration differs from protocol")
    actual_step = (
        payload["training_progress"]["environment_step"]
        if kind == "ppo"
        else payload["training_step"]
    )
    if actual_step != specification["step"]:
        raise ValueError("checkpoint step differs from selection rule")
    if kind == "ppo":
        if (
            payload["training_progress"]["level_config_id"]
            != protocol["level_config_id"]
        ):
            raise ValueError("PPO checkpoint level configuration differs")
        policy.network.eval()
    else:
        if policy.architecture != "mlp":
            raise ValueError("DQN reference must use MLP")
        policy.online.eval()
    return policy


def policy_state(policy):
    """Fingerprint weights, local RNGs and optimizer state without advancing them."""
    import torch

    digest = hashlib.sha256()

    def walk(value):
        if isinstance(value, torch.Tensor):
            digest.update(str((value.dtype, tuple(value.shape))).encode())
            digest.update(value.detach().cpu().numpy().tobytes())
        elif isinstance(value, dict):
            for key in sorted(value, key=str):
                digest.update(str(key).encode())
                walk(value[key])
        elif isinstance(value, (tuple, list)):
            for item in value:
                walk(item)
        else:
            digest.update(repr(value).encode())

    if hasattr(policy, "network"):
        walk(policy.network.state_dict())
        walk(policy.policy_rng.get_state())
        walk(policy.shuffle_rng.bit_generator.state)
    elif hasattr(policy, "online"):
        walk(policy.online.state_dict())
        walk(policy.target.state_dict())
        walk(policy.rng.bit_generator.state)
    if hasattr(policy, "optimizer"):
        walk(policy.optimizer.state_dict())
    return digest.hexdigest()


class InitialObserver:
    def __init__(self, policy, *, critic=False):
        self.policy = policy
        self.critic = critic
        self.initial = None

    def select_action(self, observation):
        if self.initial is None:
            self.initial = {
                "fingerprint": observation_fingerprint(observation),
                "observation": {
                    key: np.asarray(value).tolist()
                    for key, value in observation.items()
                },
                "critic_value": self.policy.value(observation) if self.critic else None,
            }
        return self.policy.select_action(observation)


def run_policy(protocol_path, policy_name, output_dir):
    import torch

    from match3_ai.games.match3.environment import Match3LevelEnv

    protocol = load_protocol(protocol_path)
    specification = next(
        (s for s in protocol["policies"] if s["name"] == policy_name), None
    )
    if specification is None:
        raise ValueError("policy not in frozen protocol")
    output = Path(output_dir)
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True, exist_ok=False)
    collector, initial_states = AnalyticsCollector(), []
    digest = sha256(protocol_path)
    run_id = digest[:24]
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    started = perf_counter()
    try:
        policy = load_policy(specification, protocol)
        before = policy_state(policy)
        for index, seed in enumerate(protocol["seeds"]):
            if specification["kind"] == "random":
                policy = load_policy(specification, protocol, index)
            observer = InitialObserver(policy, critic=specification["kind"] == "ppo")
            config = protocol["environment"]
            env = Match3LevelEnv(
                config["level_id"],
                max_cascades=config["max_cascades"],
                max_initialization_attempts=config["max_initialization_attempts"],
            )
            try:
                record = run_episode(
                    env,
                    observer,
                    seed=seed,
                    max_steps=config["max_steps"],
                    experiment_id=run_id,
                    episode_id=f"{run_id}:{policy_name}:{seed}",
                    agent_name=policy_name,
                    agent_seed=protocol["random_agent_seeds"][index]
                    if specification["kind"] == "random"
                    else getattr(policy, "seed", None),
                    collector=collector,
                )
            finally:
                env.close()
            if (
                record.outcome not in ("WON", "LOST")
                or record.invalid_actions
                or record.level_config_id != protocol["level_config_id"]
            ):
                raise RuntimeError(
                    f"unusable evaluation episode: {policy_name}, seed {seed}, {record.outcome}, {record.error_message}"
                )
            initial_states.append({"seed": seed, **observer.initial})
            if (index + 1) % 100 == 0:
                print(
                    f"{policy_name}: {index + 1}/{len(protocol['seeds'])} episodes complete",
                    flush=True,
                )
        if (
            specification["kind"] not in ("random", "greedy")
            and policy_state(policy) != before
        ):
            raise RuntimeError("frozen model, optimizer or RNG state changed")
        if (
            "checkpoint" in specification
            and sha256(specification["checkpoint"]) != specification["sha256"]
        ):
            raise RuntimeError("checkpoint file changed")
        load_protocol(protocol_path)
        metadata = {
            "experiment_id": run_id,
            "protocol_sha256": digest,
            "policy": specification,
            "summary": summarize_level_episodes(collector.episodes)[0],
            "episodes": len(collector.episodes),
            "elapsed_seconds": perf_counter() - started,
            "model_optimizer_rng_unchanged": True
            if specification["kind"] in ("ppo", "dqn")
            else None,
            "python_version": platform.python_version(),
            "torch_version": str(torch.__version__),
            "numpy_version": np.__version__,
            "torch_threads": 1,
        }
        export_evaluation(collector, metadata, output / "evaluation")
        with (output / "initial_states.json").open("x") as handle:
            json.dump(initial_states, handle, allow_nan=False)
        with (output / "result.json").open("x") as handle:
            json.dump(metadata, handle, indent=2, allow_nan=False)
        return metadata
    except BaseException as error:
        with (output / "failure.json").open("x") as handle:
            json.dump(
                {
                    "policy": policy_name,
                    "episodes_completed": len(collector.episodes),
                    "error_type": type(error).__name__,
                    "error": str(error),
                    "protocol_sha256": digest,
                },
                handle,
                indent=2,
            )
        if collector.episodes:
            export_evaluation(
                collector,
                {"experiment_id": run_id, "protocol_sha256": digest, "failure": True},
                output / "partial_evaluation",
            )
        raise
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--policy", action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    for name in args.policy:
        run_policy(args.protocol, name, args.output_root / name)


if __name__ == "__main__":
    main()
