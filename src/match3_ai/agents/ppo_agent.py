"""CPU MLP actor-critic with local sampling RNG and masked categorical actions."""

import os
from copy import deepcopy
from dataclasses import asdict
from itertools import pairwise
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical

from match3_ai.agents.observation_encoder import EncoderConfig, ObservationEncoder


class ActorCritic(nn.Module):
    def __init__(self, hidden_dims=(256, 128), seed=0):
        super().__init__()
        if not hidden_dims or any(type(n) is not int or n <= 0 for n in hidden_dims):
            raise ValueError("hidden dimensions must be positive integers")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.hidden_dims = tuple(hidden_dims)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            layers = []
            for source, target in pairwise(
                (ObservationEncoder.feature_dim, *hidden_dims)
            ):
                layers.extend((nn.Linear(source, target), nn.ReLU()))
            self.backbone = nn.Sequential(*layers)
            self.actor = nn.Linear(hidden_dims[-1], 144)
            self.critic = nn.Linear(hidden_dims[-1], 1)

    def forward(self, features):
        shared = self.backbone(features)
        return self.actor(shared), self.critic(shared).squeeze(-1)


def masked_distribution(logits, masks):
    """Construct the same distribution for collection, optimization and evaluation."""
    if logits.ndim != 2 or logits.shape[-1] != 144 or masks.shape != logits.shape:
        raise ValueError("logits and masks must have shape (batch,144)")
    if masks.dtype != torch.bool or not masks.any(dim=-1).all():
        raise ValueError("each action mask must contain at least one valid action")
    if not torch.isfinite(logits).all():
        raise ValueError("actor logits must be finite")
    return Categorical(logits=logits.masked_fill(~masks, -torch.inf))


class PPOAgent:
    def __init__(
        self, seed=0, hidden_dims=(256, 128), learning_rate=3e-4, encoder_config=None
    ):
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite")
        self.seed, self.learning_rate = seed, learning_rate
        streams = np.random.SeedSequence(seed).spawn(3)
        self.stream_seeds = {
            name: int(stream.generate_state(1)[0])
            for name, stream in zip(("model", "policy", "shuffle"), streams)
        }
        self.encoder = ObservationEncoder(encoder_config)
        self.network = ActorCritic(hidden_dims, self.stream_seeds["model"])
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=learning_rate)
        self.policy_rng = torch.Generator(device="cpu").manual_seed(
            self.stream_seeds["policy"]
        )
        self.shuffle_rng = np.random.default_rng(self.stream_seeds["shuffle"])

    def act(self, observation, *, deterministic=False):
        features, mask = self.encoder.encode(observation)
        with torch.inference_mode():
            logits, values = self.network(torch.from_numpy(features[None]))
            distribution = masked_distribution(logits, torch.from_numpy(mask[None]))
            if not torch.isfinite(values).all():
                raise ValueError("critic values must be finite")
            action = (
                distribution.probs.argmax(dim=-1)
                if deterministic
                else torch.multinomial(
                    distribution.probs, 1, generator=self.policy_rng
                ).squeeze(-1)
            )
            return (
                int(action.item()),
                float(distribution.log_prob(action).item()),
                float(values.item()),
            )

    def select_action(self, observation):
        return self.act(observation, deterministic=True)[0]

    def value(self, observation):
        """Evaluate final observations, including terminal states with empty masks."""
        features, _ = self.encoder.encode(observation)
        with torch.inference_mode():
            _, value = self.network(torch.from_numpy(features[None]))
        if not torch.isfinite(value).all():
            raise ValueError("critic values must be finite")
        return float(value.item())

    def evaluate_actions(self, features, masks, actions):
        logits, values = self.network(features)
        distribution = masked_distribution(logits, masks)
        if actions.dtype != torch.long or actions.shape != values.shape:
            raise ValueError("actions must be an integer vector matching the batch")
        if (
            torch.any((actions < 0) | (actions >= 144))
            or not masks.gather(1, actions[:, None]).all()
        ):
            raise ValueError("rollout actions must be mask-valid")
        if not torch.isfinite(values).all():
            raise ValueError("critic values must be finite")
        return distribution.log_prob(actions), distribution.entropy(), values

    def save_checkpoint(
        self,
        path,
        *,
        training_config=None,
        training_progress=None,
        validation_metrics=None,
    ):
        destination = Path(path)
        if destination.exists():
            raise FileExistsError(destination)
        destination.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format_version": 1,
            "algorithm": "masked_ppo",
            "architecture": self.architecture_metadata(),
            "seed": self.seed,
            "stream_seeds": self.stream_seeds,
            "hidden_dims": self.network.hidden_dims,
            "learning_rate": self.learning_rate,
            "encoder_config": asdict(self.encoder.config),
            "network": self.network.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "policy_rng": self.policy_rng.get_state(),
            "shuffle_rng": deepcopy(self.shuffle_rng.bit_generator.state),
            "training_config": training_config or {},
            "training_progress": training_progress or {},
            "validation_metrics": validation_metrics or {},
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

    def architecture_metadata(self):
        return {
            "kind": "mlp_actor_critic",
            "version": 1,
            "input_dim": 1467,
            "action_count": 144,
            "hidden_dims": list(self.network.hidden_dims),
        }

    @classmethod
    def load_checkpoint(cls, path):
        payload = torch.load(path, map_location="cpu", weights_only=True)
        if payload["format_version"] != 1 or payload["algorithm"] != "masked_ppo":
            raise ValueError("unsupported PPO checkpoint")
        agent = cls(
            payload["seed"],
            tuple(payload["hidden_dims"]),
            payload["learning_rate"],
            EncoderConfig(**payload["encoder_config"]),
        )
        if agent.stream_seeds != payload["stream_seeds"]:
            raise ValueError("unsupported PPO seed streams")
        if (
            payload.get("architecture", agent.architecture_metadata())
            != agent.architecture_metadata()
        ):
            raise ValueError("unsupported PPO architecture")
        agent.network.load_state_dict(payload["network"])
        agent.optimizer.load_state_dict(payload["optimizer"])
        agent.policy_rng.set_state(payload["policy_rng"])
        agent.shuffle_rng.bit_generator.state = deepcopy(payload["shuffle_rng"])
        return agent, payload
