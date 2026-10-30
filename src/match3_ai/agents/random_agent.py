from __future__ import annotations

import numpy as np

from match3_ai.agents.base import Observation


class RandomAgent:
    """uniform random policy over valid actions."""

    def __init__(self, seed: int | None = None) -> None:
        # ayri seed olusturuyoruz bc agent'in random karari envin randomlugundan bagimsiz olmali
        self.rng = np.random.default_rng(seed)

    def select_action(self, observation: Observation) -> int:
        action_mask = observation["action_mask"]
        # valid action id'leri bulur
        valid_actions = np.flatnonzero(action_mask)

        if valid_actions.size == 0:
            raise RuntimeError("no valid actions available.")
        # valid choices arasindan birini esit olasilikla seciyor
        return int(self.rng.choice(valid_actions))
