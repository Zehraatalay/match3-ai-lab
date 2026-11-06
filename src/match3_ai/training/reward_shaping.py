"""Observable objective potentials and optional potential-based training rewards."""

from dataclasses import dataclass

import numpy as np


def objective_potential(observation, *, terminated: bool = False) -> float:
    """Equal active-objective weights; invalid ratios contribute zero, not NaN.

    Codes 1/2 are CLEAR_ALL_BLOCKERS/COLLECT_TILES in the fixed level observation.
    Only true termination zeros the potential; truncation is not a terminal state.
    """
    if terminated:
        return 0.0
    mask = np.asarray(observation["objective_mask"])
    kinds = np.asarray(observation["objective_type"])
    progress = np.asarray(observation["objective_progress"], dtype=float)
    targets = np.asarray(observation["objective_target"], dtype=float)
    if mask.ndim != 1 or any(
        values.shape != mask.shape for values in (kinds, progress, targets)
    ):
        raise ValueError("objective arrays must have matching one-dimensional shapes")
    if not np.isin(mask, [0, 1]).all():
        raise ValueError("objective mask must be binary")
    active = mask.astype(bool)
    if not np.isin(kinds[active], [1, 2]).all():
        raise ValueError("unsupported active objective type")
    if not active.any():
        return 0.0
    progress, targets = progress[active], targets[active]
    valid = np.isfinite(progress) & np.isfinite(targets) & (targets > 0)
    ratios = np.zeros_like(progress)
    bounded_progress = np.minimum(np.maximum(progress, 0), targets)
    np.divide(bounded_progress, targets, out=ratios, where=valid)
    return float(np.clip(ratios, 0, 1).mean())


@dataclass(frozen=True)
class RewardComponents:
    sparse_reward: float
    shaping_bonus: float
    shaped_reward: float
    potential_current: float
    potential_next: float


def training_reward(
    observation,
    next_observation,
    sparse_reward,
    *,
    terminated,
    truncated=False,
    mode="sparse",
    beta=0.5,
    gamma=0.99,
):
    """Compute PBRS without modifying observations, environment or RNG state.

    Truncations retain next-state potential for the existing bootstrapping policy.
    When both flags are true, true termination takes precedence.
    """
    if mode not in ("sparse", "pbrs"):
        raise ValueError("reward mode must be sparse or pbrs")
    if isinstance(beta, (bool, np.bool_)) or not np.isfinite(beta) or beta < 0:
        raise ValueError("beta must be finite and non-negative")
    if not np.isfinite(gamma) or not 0 <= gamma <= 1:
        raise ValueError("gamma must be in [0,1]")
    if not np.isfinite(sparse_reward):
        raise ValueError("sparse reward must be finite")
    if not isinstance(terminated, (bool, np.bool_)) or not isinstance(
        truncated, (bool, np.bool_)
    ):
        raise TypeError("termination flags must be booleans")
    current = objective_potential(observation)
    following = objective_potential(next_observation, terminated=terminated)
    bonus = (
        float(beta * (gamma * following - current))
        if mode == "pbrs" and beta != 0
        else 0.0
    )
    return RewardComponents(
        float(sparse_reward), bonus, float(sparse_reward) + bonus, current, following
    )
