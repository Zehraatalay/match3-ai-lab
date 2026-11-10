from __future__ import annotations

import copy

import numpy as np

from match3_ai.games.match3.environment import Match3LevelEnv


class MCTSSimulator:
    """Independent simulation wrapper for Match3LevelEnv."""

    def __init__(self, env: Match3LevelEnv) -> None:
        if env.session is None:
            raise ValueError("Environment must be reset before simulation.")

        self.env = env

    def clone(self) -> Match3LevelEnv:
        """Create an independent copy of the current game state."""
        return copy.deepcopy(self.env)

    def clone_with_seed(self, seed: int) -> Match3LevelEnv:
        """clone the environment and resample future tile randomness."""
        env = self.clone()

        assert env.session is not None
        env.session.engine.rng = np.random.default_rng(seed)

        return env

    @staticmethod
    def legal_actions(env: Match3LevelEnv) -> list[int]:
        """Return legal actions for the current simulated state."""
        observation = env._observation()
        return [int(action) for action in np.flatnonzero(observation["action_mask"])]

    @staticmethod
    def step(
        env: Match3LevelEnv,
        action: int,
    ) -> tuple[float, bool]:
        """Apply an action and return reward and terminal status."""
        _, reward, terminated, truncated, _ = env.step(action)
        return float(reward), bool(terminated or truncated)
