"""Episode-aware GAE and the masked PPO clipped optimization objective."""

import numpy as np
import torch
from torch.nn import functional as F


def generalized_advantages(
    rewards, values, next_values, terminated, truncated, *, gamma=0.99, gae_lambda=0.95
):
    arrays = [
        np.asarray(a) for a in (rewards, values, next_values, terminated, truncated)
    ]
    rewards, values, next_values, terminated, truncated = arrays
    if (
        rewards.ndim != 1
        or not rewards.size
        or any(a.shape != rewards.shape for a in arrays)
    ):
        raise ValueError("GAE arrays must be matching nonempty vectors")
    if terminated.dtype != np.bool_ or truncated.dtype != np.bool_:
        raise ValueError("GAE termination flags must be boolean")
    if not all(np.isfinite(a).all() for a in arrays[:3]):
        raise ValueError("GAE values must be finite")
    if not (
        np.isfinite(gamma)
        and 0 <= gamma <= 1
        and np.isfinite(gae_lambda)
        and 0 <= gae_lambda <= 1
    ):
        raise ValueError("gamma and gae_lambda must be in [0,1]")
    advantages = np.zeros(rewards.shape, dtype=np.float32)
    tail = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        delta = (
            float(rewards[index])
            + gamma * (not terminated[index]) * float(next_values[index])
            - float(values[index])
        )
        tail = (
            delta
            + gamma * gae_lambda * (not (terminated[index] or truncated[index])) * tail
        )
        advantages[index] = tail
    return advantages, (advantages + values).astype(np.float32)


def normalize_advantages(advantages):
    if (
        advantages.ndim != 1
        or not advantages.numel()
        or not torch.isfinite(advantages).all()
    ):
        raise ValueError("advantages must be a finite nonempty vector")
    return (advantages - advantages.mean()) / advantages.std(unbiased=False).clamp_min(
        1e-8
    )


def explained_variance(values, returns):
    """Return None when constant targets make explained variance undefined."""
    values, returns = np.asarray(values, dtype=float), np.asarray(returns, dtype=float)
    if values.ndim != 1 or values.shape != returns.shape or not values.size:
        raise ValueError("explained variance requires matching nonempty vectors")
    if not np.isfinite(values).all() or not np.isfinite(returns).all():
        raise ValueError("explained variance inputs must be finite")
    variance = float(np.var(returns))
    return None if variance <= 1e-12 else float(1 - np.var(returns - values) / variance)


def rollout_diagnostics(values, returns, advantages):
    advantages = np.asarray(advantages, dtype=float)
    if (
        advantages.shape != np.asarray(values).shape
        or not np.isfinite(advantages).all()
    ):
        raise ValueError("rollout advantages must be finite and match values")
    return {
        "explained_variance": explained_variance(values, returns),
        "advantage_mean": float(advantages.mean()),
        "advantage_std": float(advantages.std()),
    }


def clipped_objective(
    log_probs,
    old_log_probs,
    advantages,
    values,
    returns,
    entropy,
    *,
    clip_ratio=0.2,
    value_coefficient=0.5,
    entropy_coefficient=0.01,
):
    if (
        any(
            x.shape != log_probs.shape
            for x in (old_log_probs, advantages, values, returns, entropy)
        )
        or log_probs.ndim != 1
    ):
        raise ValueError("PPO loss inputs must be matching vectors")
    log_ratio = log_probs - old_log_probs
    ratio = log_ratio.exp()
    policy_loss = -torch.minimum(
        ratio * advantages, ratio.clamp(1 - clip_ratio, 1 + clip_ratio) * advantages
    ).mean()
    value_loss = 0.5 * F.mse_loss(values, returns)
    mean_entropy = entropy.mean()
    loss = (
        policy_loss
        + value_coefficient * value_loss
        - entropy_coefficient * mean_entropy
    )
    with torch.no_grad():
        metrics = {
            "policy_loss": float(policy_loss.detach()),
            "value_loss": float(value_loss.detach()),
            "entropy": float(mean_entropy.detach()),
            "approximate_kl": float(((ratio - 1) - log_ratio).mean()),
            "clip_fraction": float(((ratio - 1).abs() > clip_ratio).float().mean()),
            "ratio_mean": float(ratio.mean()),
            "ratio_min": float(ratio.min()),
            "ratio_max": float(ratio.max()),
        }
    if not torch.isfinite(loss) or not all(np.isfinite(v) for v in metrics.values()):
        raise ValueError("PPO losses must be finite")
    return loss, metrics


def optimize_ppo(
    agent,
    batch,
    *,
    epochs=4,
    minibatch_size=64,
    clip_ratio=0.2,
    value_coefficient=0.5,
    entropy_coefficient=0.01,
    gradient_clip=0.5,
):
    """Reuse one fixed on-policy rollout for a bounded number of PPO epochs."""
    count = len(batch["actions"])
    if count < 1 or epochs < 1 or minibatch_size < 1:
        raise ValueError("optimization requires a nonempty rollout and positive sizes")
    advantages = normalize_advantages(batch["advantages"])
    rows = []
    for epoch in range(epochs):
        order = agent.shuffle_rng.permutation(count)
        for start in range(0, count, minibatch_size):
            indices = torch.as_tensor(
                order[start : start + minibatch_size], dtype=torch.long
            )
            log_probs, entropy, values = agent.evaluate_actions(
                batch["features"][indices],
                batch["masks"][indices],
                batch["actions"][indices],
            )
            loss, metrics = clipped_objective(
                log_probs,
                batch["log_probs"][indices],
                advantages[indices],
                values,
                batch["returns"][indices],
                entropy,
                clip_ratio=clip_ratio,
                value_coefficient=value_coefficient,
                entropy_coefficient=entropy_coefficient,
            )
            agent.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                agent.network.parameters(), gradient_clip, error_if_nonfinite=True
            )
            if not all(
                p.grad is None or torch.isfinite(p.grad).all()
                for p in agent.network.parameters()
            ):
                raise ValueError("PPO gradients must be finite")
            agent.optimizer.step()
            rows.append(
                {
                    "epoch": epoch + 1,
                    "minibatch_size": len(indices),
                    **metrics,
                    "gradient_norm": float(norm),
                    "learning_rate": agent.optimizer.param_groups[0]["lr"],
                }
            )
    return rows
