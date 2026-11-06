"""Pure NumPy encoder for the reviewed fixed level observation contract."""

from dataclasses import asdict, dataclass

import numpy as np

RAW_FIELDS = {
    "tiles": (np.uint8, (9, 8)),
    "specials": (np.uint8, (9, 8)),
    "blockers": (np.uint8, (9, 8)),
    "blocker_hp": (np.uint8, (9, 8)),
    "cell_mask": (np.bool_, (9, 8)),
    "board_shape": (np.uint8, (2,)),
    "moves_remaining": (np.uint16, ()),
    "objective_type": (np.uint8, (2,)),
    "objective_color": (np.uint8, (2,)),
    "objective_progress": (np.uint16, (2,)),
    "objective_target": (np.uint16, (2,)),
    "objective_mask": (np.bool_, (2,)),
}


@dataclass(frozen=True)
class EncoderConfig:
    version: int = 1
    max_colors: int = 5
    max_hp: int = 3
    max_moves: int = 15

    def __post_init__(self):
        if any(type(value) is not int for value in asdict(self).values()) or asdict(
            self
        ) != {
            "version": 1,
            "max_colors": 5,
            "max_hp": 3,
            "max_moves": 15,
        }:
            raise ValueError(
                "unsupported encoder configuration; expected reviewed version 1"
            )


class ObservationEncoder:
    feature_dim = 1467
    action_count = 144
    spatial_shape = (20, 9, 8)
    spatial_feature_dim = 1440
    global_feature_dim = 27

    def __init__(self, config: EncoderConfig | None = None):
        self.config = config or EncoderConfig()

    @staticmethod
    def _integers(observation, key, shape, low, high):
        values = np.asarray(observation[key])
        if values.shape != shape or values.dtype.kind not in "biu":
            raise ValueError(f"{key} must be an integer array with shape {shape}")
        if np.any(values < low) or np.any(values > high):
            raise ValueError(f"{key} is outside encoder bounds [{low}, {high}]")
        return values

    def encode(self, observation) -> tuple[np.ndarray, np.ndarray]:
        missing = set(RAW_FIELDS) | {"action_mask"}
        missing -= set(observation)
        if missing:
            raise ValueError(f"missing observation keys: {sorted(missing)}")
        tiles = self._integers(observation, "tiles", (9, 8), 0, 255)
        if np.any((tiles > 5) & (tiles != 255)):
            raise ValueError("unknown tile category")
        specials = self._integers(observation, "specials", (9, 8), 0, 5)
        blockers = self._integers(observation, "blockers", (9, 8), 0, 3)
        hp = self._integers(observation, "blocker_hp", (9, 8), 0, 3)
        playable = self._integers(observation, "cell_mask", (9, 8), 0, 1).astype(bool)
        shape = self._integers(observation, "board_shape", (2,), 1, 9)
        rows, cols = map(int, shape)
        if cols > 8:
            raise ValueError("board shape exceeds padded columns")
        bounds = np.zeros((9, 8), dtype=np.float32)
        bounds[:rows, :cols] = 1
        if np.any(playable & ~bounds.astype(bool)):
            raise ValueError("padded cells cannot be playable")
        moves = self._integers(observation, "moves_remaining", (), 0, 15)
        kinds = self._integers(observation, "objective_type", (2,), 0, 2)
        colors = self._integers(observation, "objective_color", (2,), 0, 5)
        progress = self._integers(observation, "objective_progress", (2,), 0, 65535)
        target = self._integers(observation, "objective_target", (2,), 0, 65535)
        active = self._integers(observation, "objective_mask", (2,), 0, 1).astype(bool)
        if np.any(active & (target == 0)):
            raise ValueError("active objective targets must be positive")
        if np.any(active & (kinds == 0)):
            raise ValueError("active objective types cannot be padding")
        mask = self._integers(observation, "action_mask", (144,), 0, 1).astype(bool)
        tile_indices = np.where(tiles == 255, 6, tiles)
        planes = np.concatenate(
            [
                np.eye(7, dtype=np.float32)[tile_indices].transpose(2, 0, 1) * playable,
                np.eye(6, dtype=np.float32)[specials].transpose(2, 0, 1) * playable,
                np.eye(4, dtype=np.float32)[blockers].transpose(2, 0, 1) * playable,
                (hp.astype(np.float32) / 3 * playable)[None],
                playable[None].astype(np.float32),
                bounds[None],
            ]
        )
        objectives = np.zeros((2, 12), dtype=np.float32)
        for index in np.flatnonzero(active):
            objectives[index, int(kinds[index])] = 1
            objectives[index, 3 + int(colors[index])] = 1
            objectives[index, 9] = np.log1p(float(target[index])) / np.log1p(65535)
            objectives[index, 10] = min(
                float(progress[index]) / float(target[index]), 1
            )
            objectives[index, 11] = 1
        features = np.concatenate(
            [
                planes.reshape(-1),
                np.array([rows / 9, cols / 8, float(moves) / 15], dtype=np.float32),
                objectives.reshape(-1),
            ]
        ).astype(np.float32)
        return features, mask.copy()

    def encode_cnn(self, observation):
        """Reshape the exact MLP features, preserving their ordering and precision."""
        features, mask = self.encode(observation)
        return (
            features[: self.spatial_feature_dim].reshape(self.spatial_shape),
            features[self.spatial_feature_dim :],
            mask,
        )

    def encode_batch(self, observations, masks):
        return np.stack(
            [
                self.encode(
                    {
                        **{key: values[index] for key, values in observations.items()},
                        "action_mask": masks[index],
                    }
                )[0]
                for index in range(len(masks))
            ]
        )

    def pack(self, observation):
        self.encode(observation)
        return {
            key: np.asarray(observation[key], dtype=dtype).copy()
            for key, (dtype, _) in RAW_FIELDS.items()
        }
