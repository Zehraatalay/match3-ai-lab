from dataclasses import dataclass

from match3_ai.evaluation.difficulty_evaluator import DifficultyEvaluator
from match3_ai.levels.config import LevelConfig
from match3_ai.levels.difficulty import (
    Difficulty,
    DifficultyClassification,
    DifficultyMatch,
    classify_difficulty,
)
from match3_ai.levels.generation import GenerationSpec, LevelGenerator
from match3_ai.levels.screening import (
    CandidateScreener,
    ScreeningConfig,
    ScreeningResult,
)


@dataclass(frozen=True, slots=True)
class TargetGenerationConfig:
    target: Difficulty
    max_candidates: int = 50

    screening_episodes: int = 10
    confirmation_episodes: int = 100

    screening_seed: int = 1000
    confirmation_seed: int = 100_000

    mcts_simulations: int = 30

    def __post_init__(self) -> None:
        if not isinstance(self.target, Difficulty):
            raise TypeError("target must be a Difficulty")

        for name in (
            "max_candidates",
            "screening_episodes",
            "confirmation_episodes",
            "mcts_simulations",
        ):
            value = getattr(self, name)

            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        for name in ("screening_seed", "confirmation_seed"):
            value = getattr(self, name)

            if type(value) is not int or value < 0:
                raise ValueError(f"{name} must be a non-negative integer")

        screening_end = self.screening_seed + self.screening_episodes
        confirmation_end = self.confirmation_seed + self.confirmation_episodes

        if (
            self.screening_seed < confirmation_end
            and self.confirmation_seed < screening_end
        ):
            raise ValueError("Screening and confirmation seed ranges must not overlap")


@dataclass(frozen=True, slots=True)
class TargetGenerationResult:
    target: Difficulty
    selected_level: LevelConfig | None
    screening: ScreeningResult
    confirmation: DifficultyClassification | None
    confirmation_attempts: tuple[DifficultyClassification, ...] = ()

    @property
    def found(self) -> bool:
        return self.selected_level is not None


class TargetDifficultyGenerator:
    def __init__(
        self,
        *,
        generation_spec: GenerationSpec,
        config: TargetGenerationConfig,
    ) -> None:
        self.generation_spec = generation_spec
        self.config = config

        self.generator = LevelGenerator(generation_spec)

        self.screener = CandidateScreener(
            generation_spec=generation_spec,
            screening_config=ScreeningConfig(
                target=config.target,
                max_candidates=config.max_candidates,
                episodes_per_candidate=config.screening_episodes,
                evaluation_seed=config.screening_seed,
                mcts_simulations=config.mcts_simulations,
                stop_on_first_match=False,
            ),
        )

        self.confirmation_evaluator = DifficultyEvaluator(
            episodes_per_agent=config.confirmation_episodes,
            seed=config.confirmation_seed,
            agents=("MCTS_greedy_outcome",),
            mcts_simulations=config.mcts_simulations,
        )

    def generate(self) -> TargetGenerationResult:
        screening = self.screener.screen()

        candidates = [
            candidate
            for candidate in screening.candidates
            if (
                candidate.valid
                and candidate.classification is not None
                and candidate.classification.status
                in (
                    DifficultyMatch.MATCH,
                    DifficultyMatch.UNCERTAIN,
                )
            )
        ]

        # Prioritize screening matches, then uncertain candidates.
        # Preserve candidate ID ordering for deterministic behavior.
        candidates.sort(
            key=lambda candidate: (
                candidate.classification.status != DifficultyMatch.MATCH,
                candidate.candidate_id,
            )
        )

        confirmation_attempts = []

        for candidate in candidates:
            level = self.generator.generate(candidate.candidate_id)

            report = self.confirmation_evaluator.evaluate(level)

            confirmation = classify_difficulty(
                report,
                self.config.target,
            )

            confirmation_attempts.append(confirmation)

            if confirmation.status == DifficultyMatch.MATCH:
                return TargetGenerationResult(
                    target=self.config.target,
                    selected_level=level,
                    screening=screening,
                    confirmation=confirmation,
                    confirmation_attempts=tuple(confirmation_attempts),
                )

        return TargetGenerationResult(
            target=self.config.target,
            selected_level=None,
            screening=screening,
            confirmation=None,
            confirmation_attempts=tuple(confirmation_attempts),
        )
