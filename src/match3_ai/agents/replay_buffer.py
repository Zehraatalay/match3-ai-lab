"""Bounded raw-observation replay with a private sampling RNG."""

from copy import deepcopy
from dataclasses import dataclass

import numpy as np

from match3_ai.agents.observation_encoder import RAW_FIELDS, ObservationEncoder


@dataclass(frozen=True)
class ReplayBatch:
    observations: dict
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: dict
    terminated: np.ndarray
    truncated: np.ndarray
    masks: np.ndarray
    next_masks: np.ndarray


class ReplayBuffer:
    def __init__(self, capacity: int, seed: int = 0):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        self.capacity, self.size, self.position = capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        self.encoder = ObservationEncoder()
        self.observations = {
            key: np.empty((capacity, *shape), dtype=dtype)
            for key, (dtype, shape) in RAW_FIELDS.items()
        }
        self.next_observations = {
            key: np.empty_like(array) for key, array in self.observations.items()
        }
        self.actions = np.empty(capacity, dtype=np.int16)
        self.rewards = np.empty(capacity, dtype=np.float32)
        self.terminated = np.empty(capacity, dtype=bool)
        self.truncated = np.empty(capacity, dtype=bool)
        self.masks = np.empty((capacity, 144), dtype=bool)
        self.next_masks = np.empty_like(self.masks)

    def __len__(self):
        return self.size

    def add(self, observation, action, reward, next_observation, terminated, truncated):
        before, after = (
            self.encoder.pack(observation),
            self.encoder.pack(next_observation),
        )
        mask = np.asarray(observation["action_mask"], dtype=bool)
        next_mask = np.asarray(next_observation["action_mask"], dtype=bool)
        if (
            isinstance(action, (bool, np.bool_))
            or not isinstance(action, (int, np.integer))
            or not 0 <= action < 144
            or not mask[action]
        ):
            raise ValueError("replay action must be mask-valid")
        if not np.isfinite(reward) or not np.isfinite(np.float32(reward)):
            raise ValueError("reward must be finite float32")
        if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
            truncated, (bool, np.bool_)
        ):
            raise TypeError("termination flags must be boolean")
        if not terminated and not next_mask.any():
            raise ValueError("nonterminal next action mask is empty")
        index = self.position
        for key in RAW_FIELDS:
            self.observations[key][index] = before[key]
            self.next_observations[key][index] = after[key]
        self.actions[index], self.rewards[index] = action, reward
        self.terminated[index], self.truncated[index] = terminated, truncated
        self.masks[index], self.next_masks[index] = mask, next_mask
        self.position = (index + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> ReplayBatch:
        if type(batch_size) is not int or not 0 < batch_size <= self.size:
            raise ValueError(
                "batch_size must be positive and no larger than replay size"
            )
        indices = self.rng.choice(self.size, size=batch_size, replace=False)
        return ReplayBatch(
            {key: values[indices] for key, values in self.observations.items()},
            self.actions[indices].astype(np.int64),
            self.rewards[indices],
            {key: values[indices] for key, values in self.next_observations.items()},
            self.terminated[indices],
            self.truncated[indices],
            self.masks[indices],
            self.next_masks[indices],
        )

    def state_dict(self):
        return {
            "capacity": self.capacity,
            "size": self.size,
            "position": self.position,
            "rng_state": deepcopy(self.rng.bit_generator.state),
            "observations": {
                key: array[: self.size].copy()
                for key, array in self.observations.items()
            },
            "next_observations": {
                key: array[: self.size].copy()
                for key, array in self.next_observations.items()
            },
            **{
                key: getattr(self, key)[: self.size].copy()
                for key in (
                    "actions",
                    "rewards",
                    "terminated",
                    "truncated",
                    "masks",
                    "next_masks",
                )
            },
        }

    def load_state_dict(self, state):
        if (
            state["capacity"] != self.capacity
            or not 0 <= state["size"] <= self.capacity
        ):
            raise ValueError("incompatible replay capacity or size")
        if not 0 <= state["position"] < self.capacity:
            raise ValueError("invalid replay position")
        if state["size"] < self.capacity and state["position"] != state["size"]:
            raise ValueError("partially filled replay position must equal size")
        for group in ("observations", "next_observations"):
            if set(state[group]) != set(RAW_FIELDS):
                raise ValueError("checkpoint replay observation fields differ")
            for key, (dtype, shape) in RAW_FIELDS.items():
                values = np.asarray(state[group][key])
                if values.shape != (state["size"], *shape) or values.dtype != np.dtype(
                    dtype
                ):
                    raise ValueError(f"invalid replay field: {key}")
        self.size, self.position = state["size"], state["position"]
        for group in ("observations", "next_observations"):
            for key, values in state[group].items():
                getattr(self, group)[key][: self.size] = values
        for key in (
            "actions",
            "rewards",
            "terminated",
            "truncated",
            "masks",
            "next_masks",
        ):
            getattr(self, key)[: self.size] = state[key]
        self.rng.bit_generator.state = deepcopy(state["rng_state"])
