from __future__ import annotations

from match3_ai.games.match3.environment import Match3LevelEnv


class MCTSValueFunction:
    """Evaluate terminal outcomes using win/loss and efficiency."""

    def __init__(
        self,
        win_efficiency_weight: float = 0.2,
        loss_progress_weight: float = 0.2,
    ) -> None:
        if not 0.0 <= win_efficiency_weight < 1.0:
            raise ValueError("win_efficiency_weight must be in [0, 1)")

        if not 0.0 <= loss_progress_weight < 1.0:
            raise ValueError("loss_progress_weight must be in [0, 1)")

        self.win_efficiency_weight = win_efficiency_weight
        self.loss_progress_weight = loss_progress_weight

    @staticmethod
    def objective_completion(env: Match3LevelEnv) -> float:
        objectives = env._objective_info()

        if not objectives:
            return 0.0

        values = []

        for objective in objectives:
            target = objective["target"]

            if target <= 0:
                continue

            progress = objective["progress"]

            values.append(min(max(progress / target, 0.0), 1.0))

        return sum(values) / len(values) if values else 0.0

    def evaluate(
        self,
        env: Match3LevelEnv,
        terminal_reward: float,
    ) -> float:
        assert env.session is not None

        if terminal_reward > 0:
            move_limit = env.config.move_limit

            remaining_ratio = env.session.moves_remaining / move_limit

            return 1.0 + self.win_efficiency_weight * remaining_ratio

        if terminal_reward < 0:
            completion = self.objective_completion(env)

            return -1.0 + self.loss_progress_weight * completion

        return 0.0
