"""CPU DQN with selectable MLP/CNN, mask-safe actions, targets and checkpoints."""

import os
from copy import deepcopy
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn

from match3_ai.agents.cnn_q_network import CNNQNetwork
from match3_ai.agents.dueling_q_network import (
    DuelingQNetwork,
    masked_double_bellman_targets,
)
from match3_ai.agents.observation_encoder import EncoderConfig, ObservationEncoder


class QNetwork(nn.Module):
    def __init__(self, hidden_dims=(256, 128), seed: int = 0):
        super().__init__()
        if not hidden_dims or any(
            type(value) is not int or value <= 0 for value in hidden_dims
        ):
            raise ValueError("hidden dimensions must be positive integers")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.hidden_dims = tuple(hidden_dims)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layers = []
            dimensions = (ObservationEncoder.feature_dim, *hidden_dims, 144)
            for index, (source, target) in enumerate(pairwise(dimensions)):
                layers.append(nn.Linear(source, target))
                if index < len(dimensions) - 2:
                    layers.append(nn.ReLU())
            self.layers = nn.Sequential(*layers)

    def forward(self, features):
        return self.layers(features)


def masked_bellman_targets(
    rewards, next_q, terminated, truncated, next_masks, gamma=0.99
):
    """Bootstrap truncations from final observations; termination takes precedence."""
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0,1]")
    size = rewards.shape[0]
    if (
        rewards.shape != (size,)
        or next_q.shape != (size, 144)
        or next_masks.shape != (size, 144)
    ):
        raise ValueError("invalid Bellman batch shapes")
    if terminated.shape != (size,) or truncated.shape != (size,):
        raise ValueError("invalid termination flag shapes")
    if any(value.dtype != torch.bool for value in (terminated, truncated, next_masks)):
        raise ValueError("masks and termination flags must be boolean tensors")
    continuing = ~terminated
    if torch.any(continuing & ~next_masks.any(dim=1)):
        raise ValueError("nonterminal next action mask is empty")
    targets = rewards.detach().clone()
    with torch.no_grad():
        if continuing.any():
            maxima = (
                next_q[continuing]
                .masked_fill(~next_masks[continuing], -torch.inf)
                .max(dim=1)
                .values
            )
            targets[continuing] += gamma * maxima
    if not torch.isfinite(targets).all():
        raise ValueError("Bellman targets must be finite")
    return targets.detach()


def _to_checkpoint(value):
    if isinstance(value, np.ndarray):
        return torch.from_numpy(value.copy())
    if isinstance(value, dict):
        return {key: _to_checkpoint(item) for key, item in value.items()}
    return value


def checkpoint_to_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.cpu().numpy().copy()
    if isinstance(value, dict):
        return {key: checkpoint_to_numpy(item) for key, item in value.items()}
    return value


class DQNAgent:
    def __init__(
        self,
        seed=0,
        hidden_dims=(256, 128),
        learning_rate=3e-4,
        encoder_config=None,
        architecture="mlp",
        algorithm="vanilla",
    ):
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        if architecture not in ("mlp", "cnn", "dueling"):
            raise ValueError("architecture must be mlp, cnn or dueling")
        if algorithm not in ("vanilla", "double"):
            raise ValueError("algorithm must be vanilla or double")
        self.algorithm = algorithm
        self.architecture = architecture
        self.seed = seed
        network_stream, explore_stream = np.random.SeedSequence(seed).spawn(2)
        self.network_seed = int(network_stream.generate_state(1)[0])
        self.rng = np.random.default_rng(explore_stream)
        self.encoder = ObservationEncoder(encoder_config)
        network_type = {
            "mlp": QNetwork,
            "cnn": CNNQNetwork,
            "dueling": DuelingQNetwork,
        }[architecture]
        self.online = network_type(hidden_dims, self.network_seed)
        self.target = deepcopy(self.online).eval()
        self.target.requires_grad_(False)
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

    def select_action(self, observation, epsilon: float = 0.0) -> int:
        if not np.isfinite(epsilon) or not 0 <= epsilon <= 1:
            raise ValueError("epsilon must be in [0,1]")
        features, mask = self.encoder.encode(observation)
        valid = np.flatnonzero(mask)
        if not len(valid):
            raise RuntimeError("no valid actions available")
        if epsilon > 0 and self.rng.random() < epsilon:
            return int(self.rng.choice(valid))
        with torch.inference_mode():
            values = self.online(torch.from_numpy(features)[None])[0]
        if not torch.isfinite(values).all():
            raise RuntimeError("nonfinite predicted Q-values")
        return int(
            values.masked_fill(~torch.from_numpy(mask), -torch.inf).argmax().item()
        )

    def synchronize_target(self):
        self.target.load_state_dict(self.online.state_dict())
        self.target.requires_grad_(False)
        self.target.eval()

    def update(self, batch, *, gamma=0.99, gradient_clip=10.0):
        if not np.isfinite(gradient_clip) or gradient_clip <= 0:
            raise ValueError("gradient_clip must be positive and finite")
        features = torch.from_numpy(
            self.encoder.encode_batch(batch.observations, batch.masks)
        )
        next_features = torch.from_numpy(
            self.encoder.encode_batch(batch.next_observations, batch.next_masks)
        )
        q_values = self.online(features)
        chosen = q_values.gather(1, torch.from_numpy(batch.actions)[:, None]).squeeze(1)
        with torch.no_grad():
            target_q = self.target(next_features)
            target_args = (
                torch.from_numpy(batch.rewards),
                target_q,
                torch.from_numpy(batch.terminated),
                torch.from_numpy(batch.truncated),
                torch.from_numpy(batch.next_masks),
                gamma,
            )
            if self.algorithm == "double":
                targets = masked_double_bellman_targets(
                    target_args[0], self.online(next_features), *target_args[1:]
                )
            else:
                targets = masked_bellman_targets(*target_args)
        loss = nn.functional.smooth_l1_loss(chosen, targets)
        if not torch.isfinite(loss):
            raise RuntimeError("nonfinite TD loss")
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_norm = nn.utils.clip_grad_norm_(
            self.online.parameters(), gradient_clip, error_if_nonfinite=True
        )
        self.optimizer.step()
        valid_q = q_values.detach()[torch.from_numpy(batch.masks)]
        return {
            "td_loss": float(loss.detach()),
            "q_mean": float(valid_q.mean()),
            "q_min": float(valid_q.min()),
            "q_max": float(valid_q.max()),
            "gradient_norm": float(gradient_norm),
        }

    def save_checkpoint(
        self,
        path,
        *,
        training_config=None,
        training_step=0,
        training_state=None,
        validation_metrics=None,
        seed_metadata=None,
    ):
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "architecture": self.architecture,
            "algorithm": self.algorithm,
            "network_config": deepcopy(
                getattr(self.online, "architecture_config", {"version": 1})
            ),
            "encoder_config": asdict(self.encoder.config),
            "seed": self.seed,
            "network_seed": self.network_seed,
            "hidden_dims": list(self.online.hidden_dims),
            "learning_rate": self.learning_rate,
            "online": self.online.state_dict(),
            "target": self.target.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "exploration_rng": self.rng.bit_generator.state,
            "training_config": training_config or {},
            "training_step": training_step,
            "training_state": _to_checkpoint(training_state or {}),
            "validation_metrics": validation_metrics or [],
            "seed_metadata": seed_metadata or {},
        }
        temporary = None
        try:
            with NamedTemporaryFile(
                dir=destination.parent,
                prefix=f".{destination.name}.",
                suffix=".tmp",
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                torch.save(payload, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.link(temporary, destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    @classmethod
    def load_checkpoint(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != 1:
            raise ValueError("unsupported checkpoint version")
        agent = cls(
            payload["seed"],
            tuple(payload["hidden_dims"]),
            payload["learning_rate"],
            EncoderConfig(**payload["encoder_config"]),
            architecture=payload.get("architecture", "mlp"),
            algorithm=payload.get("algorithm", "vanilla"),
        )
        expected_config = getattr(agent.online, "architecture_config", {"version": 1})
        if payload.get("network_config", {"version": 1}) != expected_config:
            raise ValueError("unsupported network configuration")
        agent.online.load_state_dict(payload["online"])
        agent.target.load_state_dict(payload["target"])
        agent.target.requires_grad_(False)
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.rng.bit_generator.state = deepcopy(payload["exploration_rng"])
        return agent, payload
