from dataclasses import dataclass, replace
from enum import Enum

import numpy as np

from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.engine import (
    DEFAULT_MAX_CASCADES,
    Action,
    Match3Engine,
    StepResult,
)
from match3_ai.levels.config import LevelConfig, ObjectiveType
from match3_ai.levels.initialization import (
    DEFAULT_MAX_INITIALIZATION_ATTEMPTS,
    create_level_board,
)


class LevelStatus(str, Enum):
    IN_PROGRESS = "IN_PROGRESS"
    WON = "WON"
    LOST = "LOST"


@dataclass(frozen=True, slots=True)
class ObjectiveProgress:
    objective_type: ObjectiveType
    target: int
    progress: int
    raw_progress: int
    completed: bool
    initial_blocker_count: int | None = None
    remaining_blocker_count: int | None = None
    cleared_blocker_count: int | None = None


class LevelSession:
    def __init__(
        self,
        config: LevelConfig,
        *,
        board_seed: int | None = None,
        engine_seed: int | None = None,
        max_cascades: int = DEFAULT_MAX_CASCADES,
        max_initialization_attempts: int = DEFAULT_MAX_INITIALIZATION_ATTEMPTS,
    ) -> None:
        if not isinstance(config, LevelConfig):
            raise TypeError("config must be a LevelConfig")
        if not config.objectives:
            raise ValueError("level sessions require at least one objective")
        if type(max_cascades) is not int or max_cascades <= 0:
            raise ValueError("max_cascades must be a positive integer")
        if (
            type(max_initialization_attempts) is not int
            or max_initialization_attempts <= 0
        ):
            raise ValueError("max_initialization_attempts must be a positive integer")
        for name, value in (("board_seed", board_seed), ("engine_seed", engine_seed)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")

        self.config = config
        self._board_seed = config.seed if board_seed is None else board_seed
        self._engine_seed = config.seed if engine_seed is None else engine_seed
        self.max_cascades = max_cascades
        self.max_initialization_attempts = max_initialization_attempts
        self.state = create_level_board(
            config, max_attempts=self.max_initialization_attempts, seed=self._board_seed
        )
        self.engine = Match3Engine(
            num_colors=config.num_colors,
            seed=self._engine_seed,
            max_cascades=self.max_cascades,
        )
        self.moves_remaining = config.move_limit
        self._initial_blocker_count = int(
            np.count_nonzero(self.state.blockers != BlockerType.NONE)
        )
        self.objective_progress = self._make_initial_progress()
        self.status = LevelStatus.IN_PROGRESS

        if all(progress.completed for progress in self.objective_progress):
            raise ValueError("level objectives cannot be complete at initialization")

    @property
    def board(self) -> BoardState:
        return self.state

    def _make_initial_progress(self) -> tuple[ObjectiveProgress, ...]:
        progress_items = []
        for objective in self.config.objectives:
            if objective.objective_type == ObjectiveType.CLEAR_ALL_BLOCKERS:
                remaining = int(
                    np.count_nonzero(self.state.blockers != BlockerType.NONE)
                )
                cleared = max(0, self._initial_blocker_count - remaining)
                target = self._initial_blocker_count
                progress_items.append(
                    ObjectiveProgress(
                        objective_type=objective.objective_type,
                        target=target,
                        progress=min(cleared, target),
                        raw_progress=cleared,
                        completed=remaining == 0,
                        initial_blocker_count=self._initial_blocker_count,
                        remaining_blocker_count=remaining,
                        cleared_blocker_count=cleared,
                    )
                )
            else:
                progress_items.append(
                    ObjectiveProgress(
                        objective_type=objective.objective_type,
                        target=objective.target,
                        progress=0,
                        raw_progress=0,
                        completed=False,
                    )
                )
        return tuple(progress_items)

    def reset(self) -> BoardState:
        self.state = create_level_board(
            self.config,
            max_attempts=self.max_initialization_attempts,
            seed=self._board_seed,
        )
        self.engine = Match3Engine(
            num_colors=self.config.num_colors,
            seed=self._engine_seed,
            max_cascades=self.max_cascades,
        )
        self.moves_remaining = self.config.move_limit
        self.objective_progress = self._make_initial_progress()
        self.status = LevelStatus.IN_PROGRESS
        return self.state

    def step(
        self,
        action: Action,
        *,
        record_events: bool = False,
    ) -> StepResult:
        if self.status != LevelStatus.IN_PROGRESS:
            raise RuntimeError("cannot step a terminal level session; call reset()")

        result = self.engine.step(
            self.state,
            action,
            record_events=record_events,
        )
        if not result.valid:
            return result

        self.state = result.state
        self.moves_remaining -= 1
        self._update_objective_progress(result)

        if all(progress.completed for progress in self.objective_progress):
            self.status = LevelStatus.WON
        elif self.moves_remaining == 0:
            self.status = LevelStatus.LOST

        return result

    def _update_objective_progress(self, result: StepResult) -> None:
        updated = []
        remaining_blockers = int(
            np.count_nonzero(self.state.blockers != BlockerType.NONE)
        )

        for current, objective in zip(
            self.objective_progress,
            self.config.objectives,
        ):
            if objective.objective_type == ObjectiveType.CLEAR_ALL_BLOCKERS:
                cleared = max(
                    current.raw_progress,
                    self._initial_blocker_count - remaining_blockers,
                )
                updated.append(
                    replace(
                        current,
                        progress=min(cleared, current.target),
                        raw_progress=cleared,
                        completed=remaining_blockers == 0,
                        remaining_blocker_count=remaining_blockers,
                        cleared_blocker_count=cleared,
                    )
                )
                continue

            if objective.color is None:
                increment = result.cleared_tiles
            else:
                color_index = objective.color - 1
                if color_index >= len(result.cleared_by_color):
                    raise RuntimeError(
                        "engine result is missing per-color clear metadata"
                    )
                increment = result.cleared_by_color[color_index]

            raw_progress = current.raw_progress + increment
            updated.append(
                replace(
                    current,
                    progress=min(raw_progress, current.target),
                    raw_progress=raw_progress,
                    completed=raw_progress >= current.target,
                )
            )

        self.objective_progress = tuple(updated)
