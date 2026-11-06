from __future__ import annotations

import numpy as np

from match3_ai.agents.base import Observation
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.games.match3.rules import all_find_matches


# board uzerindeki tum yatay ve dikey adjacent pairs olusturuluyor burada mesela 112
def build_action_pairs(
    rows: int,
    cols: int,
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Build adjacent swap pairs in row-major order."""

    pairs = []

    for row in range(rows):
        for col in range(cols):
            if col + 1 < cols:
                pairs.append(((row, col), (row, col + 1)))

            if row + 1 < rows:
                pairs.append(((row, col), (row + 1, col)))

    return pairs


class GreedyAgent:
    """Select the highest immediate match count, breaking ties by lowest action ID.

    Special effects, cascades, blocker damage, and objectives are not scored.
    """

    def select_action(self, observation: Observation) -> int:
        action_mask = observation["action_mask"]
        canonical = "tiles" in observation
        if canonical:
            rows, cols = (int(value) for value in observation["board_shape"])
            board = np.where(
                observation["cell_mask"][:rows, :cols],
                observation["tiles"][:rows, :cols],
                0,
            )
        else:
            board = observation["board"]
            rows, cols = board.shape
            action_pairs = build_action_pairs(rows, cols)

        valid_actions = np.flatnonzero(action_mask)

        if valid_actions.size == 0:
            raise RuntimeError("No valid actions available.")

        best_action = -1
        best_score = -1

        for action in valid_actions:
            action_id = int(action)

            if canonical:
                r1, c1, direction = Match3LevelEnv.decode_action(action_id)
                r2, c2 = (r1, c1 + 1) if direction == 0 else (r1 + 1, c1)
            else:
                (r1, c1), (r2, c2) = action_pairs[action_id]
            # dogrudan observation icindeki boardı degistirseydik agent karar verirken environment stateini de yanlıslıkla degistirebilirdik so copy sart
            simulated_board = board.copy()
            # swap islemi
            simulated_board[r1, c1], simulated_board[r2, c2] = (
                simulated_board[r2, c2],
                simulated_board[r1, c1],
            )

            matches = all_find_matches(simulated_board)
            score = len(matches)

            if score > best_score:
                best_score = score
                best_action = action_id

        return best_action
