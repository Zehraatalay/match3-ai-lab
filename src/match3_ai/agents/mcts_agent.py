from __future__ import annotations

import numpy as np

from match3_ai.agents.base import Observation
from match3_ai.agents.mcts_simulator import MCTSSimulator


class MonteCarloAgent:
    """Evaluate root actions using independent gameplay rollouts.
    this is a rollout-based planning baseline, not full tree-based MCTS.
    """

    def __init__(
        self,
        simulator: MCTSSimulator,
        simulations_per_action: int = 10,
        seed: int | None = None,
    ) -> None:
        if simulations_per_action <= 0:
            raise ValueError("simulations_per_action must be positive")

        self.simulator = simulator
        self.simulations_per_action = simulations_per_action
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: Observation) -> int:
        legal_actions = [
            int(action) for action in np.flatnonzero(observation["action_mask"])
        ]

        if not legal_actions:
            raise RuntimeError("No legal actions available.")

        best_action = legal_actions[0]
        best_score = float("-inf")

        for action in legal_actions:
            rewards = []

            for _ in range(self.simulations_per_action):
                env = self.simulator.clone()

                try:
                    _, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    while not done:
                        mask = env._observation()["action_mask"]
                        available = np.flatnonzero(mask)

                        if available.size == 0:
                            reward = -1.0
                            break

                        rollout_action = int(self.rng.choice(available))
                        _, reward, terminated, truncated, _ = env.step(rollout_action)
                        done = terminated or truncated

                    rewards.append(float(reward))
                finally:
                    env.close()

            score = float(np.mean(rewards))

            if score > best_score:
                best_score = score
                best_action = action

        return best_action
