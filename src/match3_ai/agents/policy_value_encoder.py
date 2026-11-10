"""AZ encoding version 1: 20 spatial planes, 30 global features, 144 actions.

Spatial planes contain tile categories (empty, colors 1..5, colorless), special
categories 0..5, blocker categories 0..3, HP/3, playable cells, and board bounds.
Globals contain rows/9, columns/8, remaining/32, limit/32, remaining/limit,
colors/5, then two 12-feature objectives: type one-hot (3), color one-hot (6),
log1p(target)/log1p(65535), clipped completion ratio, and active flag.
Level IDs, seeds, engine state, and future outcomes are never encoded.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, NamedTuple

import numpy as np
from numpy.typing import NDArray

from match3_ai.levels.config import LevelConfig


@dataclass(frozen=True, slots=True)
class PolicyValueContext:
    """Observable static task context, independent of level identity and RNG."""

    rows: int
    cols: int
    num_colors: int
    move_limit: int

    def __post_init__(self) -> None:
        for name, low, high in (
            ("rows", 3, 9),
            ("cols", 3, 8),
            ("num_colors", 3, 5),
            ("move_limit", 1, 32),
        ):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError(f"{name} must be an integer in [{low}, {high}]")

    @classmethod
    def from_level_config(cls, config: LevelConfig) -> "PolicyValueContext":
        """Validate the supported level domain without retaining its identity."""
        if not isinstance(config, LevelConfig):
            raise TypeError("config must be a LevelConfig")
        config.__post_init__()
        if not 1 <= len(config.objectives) <= 2:
            raise ValueError("one or two objectives are required")
        if any(
            objective.target is not None and objective.target > 65535
            for objective in config.objectives
        ):
            raise ValueError("objective targets must fit uint16")
        return cls(config.rows, config.cols, config.num_colors, config.move_limit)


class EncodedPolicyValue(NamedTuple):
    spatial: NDArray[np.float32]
    global_features: NDArray[np.float32]
    action_mask: NDArray[np.bool_]


class PolicyValueEncoder:
    """Encode canonical level observations; empty terminal masks are preserved."""

    version: ClassVar[int] = 1
    spatial_shape: ClassVar[tuple[int, int, int]] = (20, 9, 8)
    global_feature_dim: ClassVar[int] = 30
    action_count: ClassVar[int] = 144

    @staticmethod
    def _integers(
        observation: Mapping[str, object],
        key: str,
        shape: tuple[int, ...],
        low: int,
        high: int,
    ) -> np.ndarray:
        if key not in observation:
            raise ValueError(f"missing observation field: {key}")
        values = np.asarray(observation[key])
        if values.shape != shape or values.dtype.kind not in "biu":
            raise ValueError(f"{key} must be an integer array with shape {shape}")
        if np.any(values < low) or np.any(values > high):
            raise ValueError(f"{key} is outside supported bounds [{low}, {high}]")
        return values

    def encode(
        self, observation: Mapping[str, object], context: PolicyValueContext
    ) -> EncodedPolicyValue:
        if not isinstance(context, PolicyValueContext):
            raise TypeError("context must be a PolicyValueContext")
        shape = self._integers(observation, "board_shape", (2,), 3, 9)
        if tuple(shape) != (context.rows, context.cols):
            raise ValueError("board_shape does not match static context")
        tiles = self._integers(observation, "tiles", (9, 8), 0, 255)
        if np.any((tiles > context.num_colors) & (tiles != 255)):
            raise ValueError("tile color exceeds configured palette")
        specials = self._integers(observation, "specials", (9, 8), 0, 5)
        blockers = self._integers(observation, "blockers", (9, 8), 0, 3)
        hp = self._integers(observation, "blocker_hp", (9, 8), 0, 3)
        playable = self._integers(observation, "cell_mask", (9, 8), 0, 1).astype(bool)
        bounds = np.zeros((9, 8), dtype=np.float32)
        bounds[: context.rows, : context.cols] = 1
        if np.any(playable & ~bounds.astype(bool)):
            raise ValueError("playable cells exceed board bounds")
        if any(np.any(array[~playable]) for array in (tiles, specials, blockers, hp)):
            raise ValueError("non-playable cells must contain padding only")
        moves = int(
            self._integers(observation, "moves_remaining", (), 0, context.move_limit)
        )
        kinds = self._integers(observation, "objective_type", (2,), 0, 2)
        colors = self._integers(
            observation, "objective_color", (2,), 0, context.num_colors
        )
        targets = self._integers(observation, "objective_target", (2,), 0, 65535)
        progress = self._integers(observation, "objective_progress", (2,), 0, 65535)
        active = self._integers(observation, "objective_mask", (2,), 0, 1).astype(bool)
        if not active.any() or np.any(active & ((kinds == 0) | (targets == 0))):
            raise ValueError("active objectives require a type and positive target")
        if np.any(active & (kinds == 1) & (colors != 0)):
            raise ValueError("blocker objectives cannot specify a color")
        if any(np.any(array[~active]) for array in (kinds, colors, targets, progress)):
            raise ValueError("inactive objective fields must be zero")
        mask = self._integers(observation, "action_mask", (144,), 0, 1).astype(bool)
        geometric = np.zeros((9, 8, 2), dtype=bool)
        geometric[:, :-1, 0] = playable[:, :-1] & playable[:, 1:]
        geometric[:-1, :, 1] = playable[:-1, :] & playable[1:, :]
        if np.any(mask & ~geometric.reshape(-1)):
            raise ValueError("action_mask contains endpoints outside playable cells")

        tile_indices = np.where(tiles == 255, 6, tiles)
        spatial = np.concatenate(
            (
                np.eye(7, dtype=np.float32)[tile_indices].transpose(2, 0, 1) * playable,
                np.eye(6, dtype=np.float32)[specials].transpose(2, 0, 1) * playable,
                np.eye(4, dtype=np.float32)[blockers].transpose(2, 0, 1) * playable,
                (hp.astype(np.float32) / 3 * playable)[None],
                playable[None].astype(np.float32),
                bounds[None],
            )
        )
        objectives = np.zeros((2, 12), dtype=np.float32)
        for index in np.flatnonzero(active):
            objectives[index, int(kinds[index])] = 1
            objectives[index, 3 + int(colors[index])] = 1
            objectives[index, 9] = np.log1p(float(targets[index])) / np.log1p(65535)
            objectives[index, 10] = min(float(progress[index]) / targets[index], 1)
            objectives[index, 11] = 1
        global_features = np.concatenate(
            (
                np.array(
                    [
                        context.rows / 9,
                        context.cols / 8,
                        moves / 32,
                        context.move_limit / 32,
                        moves / context.move_limit,
                        context.num_colors / 5,
                    ],
                    dtype=np.float32,
                ),
                objectives.reshape(-1),
            )
        )
        return EncodedPolicyValue(spatial, global_features, mask)
