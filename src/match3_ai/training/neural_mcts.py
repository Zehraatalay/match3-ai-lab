"""CPU search-guided AZ training; collection and optimization alternate by round.

Only naturally completed episodes enter replay. Checkpoints are supported at
completed round boundaries, including replay and all local RNG streams. The
default config is the requested unexecuted smoke configuration. No search depth
cap is available in AZ-2, and this module does not change its search behavior.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import os
from copy import deepcopy
from dataclasses import asdict, dataclass
from enum import Enum
from numbers import Real
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import ClassVar

import numpy as np
import torch

from match3_ai.agents.mcts_simulator import MCTSSimulator
from match3_ai.agents.neural_mcts_agent import NeuralMCTSAgent
from match3_ai.agents.policy_value_encoder import PolicyValueContext, PolicyValueEncoder
from match3_ai.agents.policy_value_network import (
    PolicyValueNetwork,
    masked_policy_logits,
)
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.config import LevelConfig
from match3_ai.training.search_replay import (
    SearchExample,
    SearchReplayBuffer,
    validate_targets,
)
from match3_ai.training.seed_policy import validate_partition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class NeuralMCTSTrainingConfig:
    device: str = "cpu"
    collection_rounds: int = 2
    episodes_per_round: int = 4
    simulations: int = 8
    replay_capacity: int = 256
    batch_size: int = 16
    updates_per_round: int = 10
    learning_rate: float = 3e-4
    value_loss_weight: float = 1.0
    seed: int = 42
    environment_seed_start: int = 200000
    environment_seed_count: int = 100000
    c_puct: float = 1.5
    temperature: float = 1.0
    dirichlet_alpha: float = 0.3
    dirichlet_epsilon: float = 0.1
    checkpoint_dir: str | None = None

    def __post_init__(self) -> None:
        if self.device != "cpu":
            raise ValueError("AZ-3 supports CPU training only")
        for name in (
            "collection_rounds",
            "episodes_per_round",
            "simulations",
            "replay_capacity",
            "batch_size",
            "updates_per_round",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        if self.batch_size > self.replay_capacity:
            raise ValueError("batch_size cannot exceed replay_capacity")
        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        for name in (
            "learning_rate",
            "value_loss_weight",
            "c_puct",
            "temperature",
            "dirichlet_alpha",
            "dirichlet_epsilon",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if (
            self.learning_rate == 0
            or self.dirichlet_alpha == 0
            or self.dirichlet_epsilon > 1
        ):
            raise ValueError(
                "learning rate and alpha must be positive; epsilon must be <= 1"
            )
        validate_partition(
            "training", self.environment_seed_start, self.environment_seed_count
        )
        if self.checkpoint_dir is not None and (
            not isinstance(self.checkpoint_dir, str) or not self.checkpoint_dir.strip()
        ):
            raise ValueError("checkpoint_dir must be a nonempty string or None")


def policy_value_loss(
    logits, values, masks, policies, outcomes, *, value_loss_weight=1.0
):
    """Masked soft-target cross-entropy plus terminal-outcome MSE, without shaping."""
    if (
        isinstance(value_loss_weight, bool)
        or not isinstance(value_loss_weight, Real)
        or not math.isfinite(value_loss_weight)
        or value_loss_weight < 0
    ):
        raise ValueError("value_loss_weight must be finite and non-negative")
    validate_targets(
        masks.detach().cpu().numpy(),
        policies.detach().cpu().numpy(),
        outcomes.detach().cpu().numpy(),
    )
    if (
        values.shape != outcomes.shape
        or not values.numel()
        or not torch.isfinite(values).all()
        or (values.abs() > 1).any()
    ):
        raise ValueError("values must be finite, in [-1,1], and match outcomes")
    log_probs = (
        masked_policy_logits(logits, masks).log_softmax(-1).masked_fill(~masks, 0)
    )
    policy = -(policies * log_probs).sum(-1).mean()
    value = (values - outcomes).square().mean()
    total = policy + value_loss_weight * value
    if not all(torch.isfinite(loss) for loss in (policy, value, total)):
        raise ValueError("nonfinite policy-value loss")
    return total, policy, value


@dataclass(frozen=True)
class CollectedEpisode:
    outcome: str
    examples: tuple[SearchExample, ...]
    provenance: dict
    error: str | None = None


def _checkpoint_value(value):
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {k: _checkpoint_value(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_checkpoint_value(v) for v in value]
    return value


class NeuralMCTSTrainer:
    """One fixed level, CPU Adam, fresh collection environments, round checkpoints.

    Construction and checkpoint loading do not reset or play environments. run()
    must be called explicitly. Failed/truncated episodes consume an attempt but
    contribute no positions. Updates wait until a full minibatch is available.
    """

    architecture: ClassVar[dict] = {
        "version": 1,
        "encoder_version": 1,
        "spatial_shape": [20, 9, 8],
        "global_features": 30,
        "conv_channels": [16, 32],
        "global_hidden": 16,
        "policy_hidden": 16,
        "value_hidden": 64,
        "actions": 144,
    }

    def __init__(self, config: NeuralMCTSTrainingConfig, level_config: LevelConfig):
        self.config, self.level_config = config, level_config
        self.context = PolicyValueContext.from_level_config(level_config)
        self.level_hash = hashlib.sha256(
            json.dumps(asdict(level_config), sort_keys=True).encode()
        ).hexdigest()
        names = ("network", "environment", "search", "exploration", "replay", "updates")
        self.stream_seeds = {
            name: int(stream.generate_state(1)[0])
            for name, stream in zip(
                names, np.random.SeedSequence(config.seed).spawn(len(names))
            )
        }
        self.rngs = {
            name: np.random.default_rng(self.stream_seeds[name])
            for name in ("environment", "search", "exploration", "updates")
        }
        self.network = PolicyValueNetwork(self.stream_seeds["network"])
        self.optimizer = torch.optim.Adam(
            self.network.parameters(), lr=config.learning_rate
        )
        self.replay = SearchReplayBuffer(
            config.replay_capacity, self.stream_seeds["replay"]
        )
        self.rounds_completed = self.attempts = self.updates = 0
        self.metrics: list[dict] = []
        self._at_boundary = True

    @property
    def search_config(self) -> dict:
        return {
            name: getattr(self.config, name)
            for name in (
                "simulations",
                "c_puct",
                "temperature",
                "dirichlet_alpha",
                "dirichlet_epsilon",
            )
        }

    def collect_episode(self) -> CollectedEpisode:
        """Generate one isolated episode, keeping pending positions out of replay."""
        seed = int(
            self.rngs["environment"].integers(
                self.config.environment_seed_start,
                self.config.environment_seed_start + self.config.environment_seed_count,
            )
        )
        search_seed = int(self.rngs["search"].integers(0, 2**32))
        exploration_seed = int(self.rngs["exploration"].integers(0, 2**32))
        provenance = {
            "attempt": self.attempts,
            "round": self.rounds_completed,
            "environment_seed": seed,
            "search_seed": search_seed,
            "exploration_seed": exploration_seed,
            "level_sha256": self.level_hash,
            "network_updates": self.updates,
            "network_sha256": hashlib.sha256(
                b"".join(
                    name.encode() + tensor.detach().cpu().contiguous().numpy().tobytes()
                    for name, tensor in self.network.state_dict().items()
                )
            ).hexdigest(),
            "encoder_version": PolicyValueEncoder.version,
            "search_config": self.search_config,
        }
        self.attempts += 1
        env = None
        try:
            env = Match3LevelEnv(
                self.level_config.level_id, level_config=self.level_config
            )
            observation, info = env.reset(seed=seed)
            provenance.update(
                {
                    name: info.get(name)
                    for name in ("board_seed", "engine_seed", "level_config_id")
                }
            )
            agent = NeuralMCTSAgent(
                MCTSSimulator(env),
                self.network,
                self.context,
                seed=search_seed,
                exploration_seed=exploration_seed,
                evaluation=False,
                **self.search_config,
            )
            pending = []
            for step in range(self.level_config.move_limit):
                encoded = agent.encoder.encode(observation, self.context)
                action = agent.select_action(observation)
                stats = agent.last_search_stats
                if stats is None:
                    raise RuntimeError("search returned no policy target")
                pending.append(
                    (
                        encoded,
                        stats.policy_target.copy(),
                        {**deepcopy(provenance), "position": step, "action": action},
                    )
                )
                observation, reward, terminated, truncated, info = env.step(action)
                if not info.get("valid_action", False):
                    raise RuntimeError("collection selected an invalid action")
                if terminated:
                    status = info.get("level_status")
                    if (
                        reward == -1
                        and info.get("termination_reason") == "no_legal_actions"
                    ):
                        status = "LOST"
                    if (status, reward) not in (("WON", 1), ("LOST", -1)):
                        raise RuntimeError(
                            "terminal status and sparse reward do not describe a natural outcome"
                        )
                    examples = tuple(
                        SearchExample(
                            *encoded,
                            policy,
                            float(reward),
                            {
                                **metadata,
                                "terminal_status": status,
                                "termination_reason": info.get("termination_reason"),
                            },
                        )
                        for encoded, policy, metadata in pending
                    )
                    return CollectedEpisode(status, examples, provenance)
                if truncated:
                    return CollectedEpisode("TRUNCATED", (), provenance)
                if reward != 0:
                    raise RuntimeError("collection requires zero nonterminal reward")
            return CollectedEpisode(
                "TRUNCATED",
                (),
                provenance,
                "move bound reached without a terminal outcome",
            )
        except Exception as error:
            logger.exception("Episode collection failed")
            return CollectedEpisode(
                "ERROR", (), provenance, f"{type(error).__name__}: {error}"
            )
        finally:
            if env is not None:
                env.close()

    def update(self) -> dict:
        batch = self.replay.sample(self.config.batch_size)
        spatial, globals_, masks, policies, outcomes = [
            torch.from_numpy(array) for array in batch[:5]
        ]
        update_seed = int(self.rngs["updates"].integers(0, 2**32))
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(update_seed)
            self.network.train()
            logits, values = self.network(spatial, globals_)
            total, policy, value = policy_value_loss(
                logits,
                values,
                masks,
                policies,
                outcomes,
                value_loss_weight=self.config.value_loss_weight,
            )
            self.optimizer.zero_grad(set_to_none=True)
            total.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in self.network.parameters()
            ):
                self.optimizer.zero_grad(set_to_none=True)
                raise ValueError("nonfinite training gradients")
            norm = torch.nn.utils.clip_grad_norm_(
                self.network.parameters(), 10, error_if_nonfinite=True
            )
            self.optimizer.step()
        self.updates += 1
        return {
            "policy_loss": float(policy.detach()),
            "value_loss": float(value.detach()),
            "total_loss": float(total.detach()),
            "gradient_norm": float(norm),
        }

    def run(self, *, until_round: int | None = None) -> list[dict]:
        stop = self.config.collection_rounds if until_round is None else until_round
        if (
            type(stop) is not int
            or not self.rounds_completed <= stop <= self.config.collection_rounds
        ):
            raise ValueError("until_round is outside the remaining configured rounds")
        if not self._at_boundary:
            raise RuntimeError(
                "resume from a completed-round checkpoint after interrupted work"
            )
        while self.rounds_completed < stop:
            if (
                self.config.checkpoint_dir is not None
                and self._checkpoint_path(self.rounds_completed + 1).exists()
            ):
                raise FileExistsError(self._checkpoint_path(self.rounds_completed + 1))
            self._at_boundary = False
            episodes = []
            for _ in range(self.config.episodes_per_round):
                episode = self.collect_episode()
                self.replay.extend(episode.examples)
                episodes.append(
                    {
                        "outcome": episode.outcome,
                        "positions": len(episode.examples),
                        "provenance": episode.provenance,
                        "error": episode.error,
                    }
                )
            losses = (
                [self.update() for _ in range(self.config.updates_per_round)]
                if len(self.replay) >= self.config.batch_size
                else []
            )
            self.rounds_completed += 1
            row = {
                "round": self.rounds_completed,
                "episodes": episodes,
                "replay_size": len(self.replay),
                "optimizer_updates": self.updates,
                "updates_this_round": len(losses),
                "attempted": len(episodes),
                "completed": sum(e["outcome"] in ("WON", "LOST") for e in episodes),
                "wins": sum(e["outcome"] == "WON" for e in episodes),
                "losses": sum(e["outcome"] == "LOST" for e in episodes),
                "errors": sum(e["outcome"] == "ERROR" for e in episodes),
                "truncations": sum(e["outcome"] == "TRUNCATED" for e in episodes),
                "positions_added": sum(e["positions"] for e in episodes),
                **{
                    key: float(np.mean([loss[key] for loss in losses]))
                    if losses
                    else None
                    for key in (
                        "policy_loss",
                        "value_loss",
                        "total_loss",
                        "gradient_norm",
                    )
                },
            }
            self.metrics.append(row)
            self._at_boundary = True
            if self.config.checkpoint_dir is not None:
                self.save_checkpoint(self._checkpoint_path(self.rounds_completed))
        return deepcopy(self.metrics)

    def _checkpoint_path(self, round_number: int) -> Path:
        return Path(self.config.checkpoint_dir) / f"round_{round_number:04d}.pt"

    def save_checkpoint(self, path) -> None:
        if (
            not self._at_boundary
            or self.attempts != self.rounds_completed * self.config.episodes_per_round
        ):
            raise RuntimeError("checkpoints require a completed round boundary")
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(destination)
        payload = {
            "format_version": 1,
            "algorithm": "neural_mcts_policy_value",
            "config": asdict(self.config),
            "architecture": self.architecture,
            "search_config": self.search_config,
            "encoder_version": PolicyValueEncoder.version,
            "level_sha256": self.level_hash,
            "level_config": asdict(self.level_config),
            "context": asdict(self.context),
            "stream_seeds": self.stream_seeds,
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "replay": self.replay.state_dict(),
            "rng_states": {
                name: deepcopy(rng.bit_generator.state)
                for name, rng in self.rngs.items()
            },
            "rounds_completed": self.rounds_completed,
            "attempts": self.attempts,
            "updates": self.updates,
            "metrics": self.metrics,
            "versions": {"numpy": np.__version__, "torch": str(torch.__version__)},
        }
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                torch.save(_checkpoint_value(payload), handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load_checkpoint(cls, path, level_config: LevelConfig) -> NeuralMCTSTrainer:
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if (
            payload["format_version"] != 1
            or payload["algorithm"] != "neural_mcts_policy_value"
            or payload["architecture"] != cls.architecture
            or payload["encoder_version"] != PolicyValueEncoder.version
        ):
            raise ValueError("unsupported neural MCTS checkpoint")
        trainer = cls(NeuralMCTSTrainingConfig(**payload["config"]), level_config)
        if (
            payload["level_sha256"] != trainer.level_hash
            or payload["context"] != asdict(trainer.context)
            or payload["search_config"] != trainer.search_config
            or payload["stream_seeds"] != trainer.stream_seeds
            or hashlib.sha256(
                json.dumps(payload["level_config"], sort_keys=True).encode()
            ).hexdigest()
            != trainer.level_hash
        ):
            raise ValueError(
                "checkpoint level, context, search configuration, or seed streams mismatch"
            )
        trainer.network.load_state_dict(payload["network"])
        trainer.optimizer.load_state_dict(payload["optimizer"])
        trainer.replay.load_state_dict(payload["replay"])
        if set(payload["rng_states"]) != set(trainer.rngs):
            raise ValueError("checkpoint RNG streams mismatch")
        for name, state in payload["rng_states"].items():
            trainer.rngs[name].bit_generator.state = deepcopy(state)
        trainer.rounds_completed, trainer.attempts, trainer.updates = (
            payload[name] for name in ("rounds_completed", "attempts", "updates")
        )
        trainer.metrics = deepcopy(payload["metrics"])
        if (
            any(
                type(value) is not int or value < 0
                for value in (
                    trainer.rounds_completed,
                    trainer.attempts,
                    trainer.updates,
                )
            )
            or not 0 <= trainer.rounds_completed <= trainer.config.collection_rounds
            or trainer.attempts
            != trainer.rounds_completed * trainer.config.episodes_per_round
            or len(trainer.metrics) != trainer.rounds_completed
            or trainer.updates
            != sum(row["updates_this_round"] for row in trainer.metrics)
            or set(payload["rng_states"]) != set(trainer.rngs)
        ):
            raise ValueError("invalid round-boundary checkpoint counters")
        return trainer
