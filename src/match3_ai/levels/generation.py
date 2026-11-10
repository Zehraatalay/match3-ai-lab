from dataclasses import dataclass

import numpy as np

from match3_ai.games.match3.blockers import BlockerType
from match3_ai.levels.config import (
    BlockerPlacement,
    LevelConfig,
    LevelObjective,
    ObjectiveType,
)


@dataclass(frozen=True, slots=True)
class GenerationSpec:
    rows: int = 8
    cols: int = 8
    num_colors: int = 5

    min_moves: int = 8
    max_moves: int = 20

    min_blockers: int = 0
    max_blockers: int = 8

    seed: int = 42

    def __post_init__(self) -> None:
        if type(self.rows) is not int or self.rows < 3:
            raise ValueError("rows must be an integer >= 3")

        if type(self.cols) is not int or self.cols < 3:
            raise ValueError("cols must be an integer >= 3")

        if type(self.num_colors) is not int or not 3 <= self.num_colors <= 254:
            raise ValueError("num_colors must be between 3 and 254")

        for name in ("min_moves", "max_moves"):
            value = getattr(self, name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if self.min_moves > self.max_moves:
            raise ValueError("min_moves cannot exceed max_moves")

        for name in ("min_blockers", "max_blockers"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        if self.min_blockers > self.max_blockers:
            raise ValueError("min_blockers cannot exceed max_blockers")

        if self.max_blockers >= self.rows * self.cols:
            raise ValueError("too many blockers for the board")

        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")
        if self.rows > 9 or self.cols > 8:
            raise ValueError(
                "generated level exceeds Match3LevelEnv observation capacity"
            )


class LevelGenerator:
    def __init__(self, spec: GenerationSpec) -> None:
        self.spec = spec

    def generate(self, candidate_id: int) -> LevelConfig:
        if type(candidate_id) is not int or candidate_id < 0:
            raise ValueError("candidate_id must be a non-negative integer")

        spec = self.spec

        rng = np.random.default_rng(np.random.SeedSequence([spec.seed, candidate_id]))

        move_limit = int(rng.integers(spec.min_moves, spec.max_moves + 1))

        blocker_count = int(rng.integers(spec.min_blockers, spec.max_blockers + 1))

        total_cells = spec.rows * spec.cols

        positions = rng.choice(
            total_cells,
            size=blocker_count,
            replace=False,
        )

        blockers = tuple(
            BlockerPlacement(
                position=(
                    int(position) // spec.cols,
                    int(position) % spec.cols,
                ),
                blocker_type=BlockerType.CRATE,
                variant="SINGLE",
            )
            for position in sorted(positions)
        )

        if blockers:
            objectives = (
                LevelObjective(
                    objective_type=ObjectiveType.CLEAR_ALL_BLOCKERS,
                ),
            )
        else:
            objectives = (
                LevelObjective(
                    objective_type=ObjectiveType.COLLECT_TILES,
                    target=10,
                ),
            )

        return LevelConfig(
            rows=spec.rows,
            cols=spec.cols,
            num_colors=spec.num_colors,
            move_limit=move_limit,
            level_id=f"generated-{spec.seed}-{candidate_id}",
            name=f"Generated Level {candidate_id}",
            seed=int(rng.integers(0, 2**32)),
            blockers=blockers,
            objectives=objectives,
        )
