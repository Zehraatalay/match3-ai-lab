from __future__ import annotations

import numpy as np

from match3_ai.games.match3.environment import Match3LevelEnv


class ObjectiveRolloutPolicy:
    """One-step objective-aware rollout policy."""

    def __init__(
        self,
        max_candidates: int = 5,
        seed: int | None = None,
    ) -> None:
        if max_candidates <= 0:
            raise ValueError("max_candidates must be positive")

        self.max_candidates = max_candidates
        self.rng = np.random.default_rng(seed)

    @staticmethod
    def _objective_score(env: Match3LevelEnv) -> float:
        objectives = env._objective_info()

        if not objectives:
            return 0.0

        scores = []

        for objective in objectives:
            target = objective["target"]
            progress = objective["progress"]

            if target <= 0:
                continue

            scores.append(min(progress / target, 1.0))

        return float(np.mean(scores)) if scores else 0.0

    def select_action(self, env: Match3LevelEnv) -> int:
        observation = env._observation()
        available = np.flatnonzero(observation["action_mask"])

        if available.size == 0:
            raise RuntimeError("No legal actions available.")

        candidate_count = min(self.max_candidates, available.size)

        candidates = self.rng.choice(
            available,
            size=candidate_count,
            replace=False,
        )

        best_action = int(candidates[0])
        best_score = float("-inf")

        for action in candidates:
            simulation = None

            try:
                simulation = env.__class__.__new__(env.__class__)

                import copy

                simulation = copy.deepcopy(env)

                before = self._objective_score(env)

                _, reward, terminated, truncated, _ = simulation.step(int(action))

                after = self._objective_score(simulation)

                score = after - before

                if terminated or truncated:
                    if reward > 0:
                        score += 1.0
                    elif reward < 0:
                        score -= 1.0

                if score > best_score:
                    best_score = score
                    best_action = int(action)

            finally:
                if simulation is not None:
                    simulation.close()

        return best_action
