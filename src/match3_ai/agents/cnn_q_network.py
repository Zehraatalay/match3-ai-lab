"""Compact spatial/global Q-network using exactly the legacy encoder features."""

from itertools import pairwise
from typing import ClassVar

import torch
from torch import nn

from match3_ai.agents.observation_encoder import ObservationEncoder


class CNNQNetwork(nn.Module):
    architecture_config: ClassVar[dict] = {
        "version": 1,
        "spatial_shape": [20, 9, 8],
        "global_features": 27,
        "conv_channels": [8, 16],
        "kernel_size": 3,
        "padding": 1,
        "global_hidden_dim": 32,
    }

    def __init__(self, hidden_dims=(256, 128), seed=0):
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
            self.spatial = nn.Sequential(
                nn.Conv2d(20, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Flatten(),
            )
            self.global_branch = nn.Sequential(nn.Linear(27, 32), nn.ReLU())
            dimensions = (16 * 9 * 8 + 32, *hidden_dims, 144)
            layers = []
            for index, (source, target) in enumerate(pairwise(dimensions)):
                layers.append(nn.Linear(source, target))
                if index < len(dimensions) - 2:
                    layers.append(nn.ReLU())
            self.head = nn.Sequential(*layers)

    def forward(self, features, global_features=None):
        """Accept the shared flat minibatch or explicit spatial/global batches."""
        if global_features is None:
            if (
                features.ndim != 2
                or features.shape[1] != ObservationEncoder.feature_dim
            ):
                raise ValueError("flat CNN input must have shape (batch,1467)")
            spatial = features[:, :1440].reshape(-1, 20, 9, 8)
            global_features = features[:, 1440:]
        else:
            spatial = features
            if spatial.ndim != 4 or spatial.shape[1:] != (20, 9, 8):
                raise ValueError("spatial CNN input must have shape (batch,20,9,8)")
            if global_features.shape != (spatial.shape[0], 27):
                raise ValueError("global CNN input must have shape (batch,27)")
        return self.head(
            torch.cat(
                (self.spatial(spatial), self.global_branch(global_features)), dim=1
            )
        )
