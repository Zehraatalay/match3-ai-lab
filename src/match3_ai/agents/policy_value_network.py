"""Lightweight AZ policy-value network with explicit row/column/direction order."""

import torch
from torch import Tensor, nn

from match3_ai.agents.policy_value_encoder import PolicyValueEncoder


def masked_policy_logits(logits: Tensor, masks: Tensor) -> Tensor:
    """Mask illegal actions; reject empty rows instead of creating NaN policies.

    Terminal states can be evaluated through the network without a policy mask.
    """
    if logits.ndim != 2 or logits.shape[1] != 144 or masks.shape != logits.shape:
        raise ValueError("logits and masks must have shape (batch,144)")
    if masks.dtype != torch.bool or masks.device != logits.device:
        raise ValueError("masks must be boolean and on the logits device")
    if not logits.is_floating_point() or not torch.isfinite(logits).all():
        raise ValueError("policy logits must be finite floating-point values")
    if not masks.any(dim=1).all():
        raise ValueError("each policy row must have at least one legal action")
    return logits.masked_fill(~masks, -torch.inf)


class PolicyValueNetwork(nn.Module):
    """Version 1 model: two 3x3 convolutions, contextual policy, tanh value.

    Inputs are batched spatial (B,20,9,8) and global (B,30) float tensors.
    Returns raw logits (B,144) and values (B,); optionally masks the logits.
    """

    version = 1

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            self.spatial = nn.Sequential(
                nn.Conv2d(20, 16, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
            )
            self.global_branch = nn.Sequential(
                nn.Linear(PolicyValueEncoder.global_feature_dim, 16), nn.ReLU()
            )
            self.policy_head = nn.Sequential(
                nn.Conv2d(48, 16, 1), nn.ReLU(), nn.Conv2d(16, 2, 1)
            )
            self.value_head = nn.Sequential(
                nn.Linear(32 * 9 * 8 + 16, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Tanh(),
            )

    @staticmethod
    def flatten_policy(policy_maps: Tensor) -> Tensor:
        """Map (B,direction,row,col) to action_id=((row*8)+col)*2+direction."""
        if policy_maps.ndim != 4 or policy_maps.shape[1:] != (2, 9, 8):
            raise ValueError("policy maps must have shape (batch,2,9,8)")
        return policy_maps.permute(0, 2, 3, 1).reshape(policy_maps.shape[0], 144)

    def forward(
        self,
        spatial: Tensor,
        global_features: Tensor,
        action_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if spatial.ndim != 4 or spatial.shape[1:] != (20, 9, 8):
            raise ValueError("spatial input must have shape (batch,20,9,8)")
        if spatial.shape[0] == 0 or global_features.shape != (spatial.shape[0], 30):
            raise ValueError("global input must have shape (nonempty batch,30)")
        if (
            not torch.isfinite(spatial).all()
            or not torch.isfinite(global_features).all()
        ):
            raise ValueError("network inputs must be finite")
        features = self.spatial(spatial)
        context = self.global_branch(global_features)
        broadcast = context[:, :, None, None].expand(-1, -1, 9, 8)
        logits = self.flatten_policy(
            self.policy_head(torch.cat((features, broadcast), dim=1))
        )
        values = self.value_head(
            torch.cat((features.flatten(1), context), dim=1)
        ).squeeze(-1)
        if action_mask is not None:
            logits = masked_policy_logits(logits, action_mask)
        return logits, values
