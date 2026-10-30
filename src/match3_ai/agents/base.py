from __future__ import annotations

from typing import Protocol, TypeAlias

import numpy as np
from numpy.typing import NDArray

Observation: TypeAlias = dict[
    str,
    NDArray[np.uint8] | NDArray[np.int8] | int,
]


class Agent(Protocol):
    """common action-selection interface for Match-3 agents."""

    def select_action(self, observation: Observation) -> int:
        """select an action from the current observation."""
        ...
