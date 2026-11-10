"""Bounded AZ search replay with copied examples and a private sampling RNG."""

import json
from copy import deepcopy
from dataclasses import dataclass
from typing import NamedTuple

import numpy as np


def validate_targets(masks, policies, outcomes) -> None:
    """Validate batched legal visit distributions and completed win/loss targets."""
    masks, policies, outcomes = map(np.asarray, (masks, policies, outcomes))
    if masks.ndim != 2 or masks.shape[1] != 144 or policies.shape != masks.shape:
        raise ValueError("masks and policies must have shape (batch,144)")
    if masks.dtype != np.bool_ or not masks.any(axis=1).all():
        raise ValueError("each boolean action mask must contain a legal action")
    if policies.dtype.kind not in "fiu" or not np.isfinite(policies).all():
        raise ValueError("policy targets must be finite numeric values")
    if np.any(policies < 0) or np.any(policies[~masks] != 0):
        raise ValueError(
            "policy targets must be non-negative and zero on illegal actions"
        )
    if not np.allclose(policies.sum(axis=1), 1, rtol=0, atol=1e-6):
        raise ValueError("policy targets must sum to one")
    if (
        outcomes.shape != (len(masks),)
        or outcomes.dtype.kind not in "fiu"
        or not np.isin(outcomes, [-1, 1]).all()
    ):
        raise ValueError("outcomes must be completed terminal targets -1 or +1")


@dataclass(frozen=True)
class SearchExample:
    spatial: np.ndarray
    global_features: np.ndarray
    action_mask: np.ndarray
    policy_target: np.ndarray
    outcome: float
    provenance: dict

    def __post_init__(self) -> None:
        validate_targets(
            np.asarray(self.action_mask)[None],
            np.asarray(self.policy_target)[None],
            np.asarray([self.outcome]),
        )
        for name, shape, dtype in (
            ("spatial", (20, 9, 8), np.float32),
            ("global_features", (30,), np.float32),
            ("action_mask", (144,), np.bool_),
            ("policy_target", (144,), np.float32),
        ):
            raw = np.asarray(getattr(self, name))
            if (
                raw.shape != shape
                or raw.dtype.kind not in "bifu"
                or not np.isfinite(raw).all()
            ):
                raise ValueError(
                    f"{name} must be finite numeric data with shape {shape}"
                )
            if np.any(np.abs(raw.astype(np.float64)) > np.finfo(np.float32).max):
                raise ValueError(f"{name} exceeds float32 bounds")
            array = np.array(raw, dtype=dtype, copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)
        if not isinstance(self.provenance, dict):
            raise TypeError("provenance must be a dictionary")
        try:
            json.dumps(self.provenance, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError("provenance must contain finite JSON data") from error
        object.__setattr__(self, "provenance", deepcopy(self.provenance))
        object.__setattr__(self, "outcome", float(self.outcome))


class SearchBatch(NamedTuple):
    spatial: np.ndarray
    global_features: np.ndarray
    action_mask: np.ndarray
    policy_target: np.ndarray
    outcomes: np.ndarray
    provenance: tuple[dict, ...]


class SearchReplayBuffer:
    """FIFO capacity eviction; sample without replacement and return fresh arrays."""

    def __init__(self, capacity: int, seed: int = 0) -> None:
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        self.capacity = capacity
        self.rng = np.random.default_rng(seed)
        self._examples: list[SearchExample] = []

    def __len__(self) -> int:
        return len(self._examples)

    def extend(self, examples) -> None:
        """Validate/copy the whole episode before adding any of its positions."""
        copied = [SearchExample(**vars(example)) for example in examples]
        self._examples = (self._examples + copied)[-self.capacity :]

    def sample(self, batch_size: int) -> SearchBatch:
        if type(batch_size) is not int or not 0 < batch_size <= len(self):
            raise ValueError("batch_size must be positive and no larger than replay")
        examples = [
            self._examples[int(i)]
            for i in self.rng.choice(len(self), batch_size, replace=False)
        ]
        return SearchBatch(
            *[
                np.stack([getattr(e, name) for e in examples])
                for name in (
                    "spatial",
                    "global_features",
                    "action_mask",
                    "policy_target",
                )
            ],
            np.asarray([e.outcome for e in examples], dtype=np.float32),
            tuple(deepcopy(e.provenance) for e in examples),
        )

    def state_dict(self) -> dict:
        return {
            "version": 1,
            "capacity": self.capacity,
            "examples": [deepcopy(vars(e)) for e in self._examples],
            "rng_state": deepcopy(self.rng.bit_generator.state),
        }

    def load_state_dict(self, state: dict) -> None:
        if (
            state["version"] != 1
            or state["capacity"] != self.capacity
            or len(state["examples"]) > self.capacity
        ):
            raise ValueError("incompatible replay state")
        examples = [SearchExample(**example) for example in state["examples"]]
        rng = np.random.default_rng()
        rng.bit_generator.state = deepcopy(state["rng_state"])
        self._examples, self.rng = examples, rng
