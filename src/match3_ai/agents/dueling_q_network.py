"""Feature-based dueling network and masked Double DQN targets."""

from itertools import pairwise
from typing import ClassVar

import torch
from torch import nn

from match3_ai.agents.observation_encoder import ObservationEncoder


class DuelingQNetwork(nn.Module):
    architecture_config: ClassVar[dict] = {
        "version": 1,
        "aggregation": "value + advantage - mean_all_144_advantages",
    }

    def __init__(self, hidden_dims=(256, 128), seed=0):
        super().__init__()
        if not hidden_dims or any(type(x) is not int or x <= 0 for x in hidden_dims):
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
            self.trunk = nn.Sequential(*layers)
            self.value = nn.Linear(hidden_dims[-1], 1)
            self.advantage = nn.Linear(hidden_dims[-1], 144)

    def forward(self, features):
        hidden = self.trunk(features)
        advantage = self.advantage(hidden)
        return self.value(hidden) + advantage - advantage.mean(-1, keepdim=True)


def masked_double_bellman_targets(
    rewards, online_q, target_q, terminated, truncated, next_masks, gamma=0.99
):
    """Select masked online actions, evaluate target actions, bootstrap truncations."""
    size = rewards.shape[0]
    if (
        not 0 <= gamma <= 1
        or any(x.shape != (size, 144) for x in (online_q, target_q, next_masks))
        or any(x.shape != (size,) for x in (rewards, terminated, truncated))
    ):
        raise ValueError("invalid Double DQN batch shapes or discount")
    if any(x.dtype != torch.bool for x in (terminated, truncated, next_masks)):
        raise ValueError("masks and termination flags must be boolean tensors")
    continuing = ~terminated
    if torch.any(continuing & ~next_masks.any(-1)):
        raise ValueError("nonterminal next action mask is empty")
    with torch.no_grad():
        targets = rewards.detach().clone()
        if continuing.any():
            online = online_q[continuing].masked_fill(
                ~next_masks[continuing], -torch.inf
            )
            if not torch.isfinite(online[next_masks[continuing]]).all():
                raise ValueError("legal online Q-values must be finite")
            actions = online.argmax(-1)
            targets[continuing] += gamma * target_q[continuing].gather(
                1, actions[:, None]
            ).squeeze(1)
    if not torch.isfinite(targets).all():
        raise ValueError("Double DQN targets must be finite")
    return targets.detach()
