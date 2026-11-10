from dataclasses import dataclass

from match3_ai.evaluation.difficulty_evaluator import (
    DifficultyEvaluator,
)
from match3_ai.levels.difficulty import (
    Difficulty,
    DifficultyClassification,
    DifficultyMatch,
    classify_difficulty,
)
from match3_ai.levels.generation import GenerationSpec, LevelGenerator
from match3_ai.levels.validation import LevelValidator


@dataclass(frozen=True, slots=True)
class ScreeningConfig:
    target: Difficulty
    max_candidates: int = 50
    episodes_per_candidate: int = 30
    evaluation_seed: int = 1000
    mcts_simulations: int = 30
    stop_on_first_match: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.target, Difficulty):
            raise TypeError("target must be a Difficulty")

        for name in (
            "max_candidates",
            "episodes_per_candidate",
            "mcts_simulations",
        ):
            value = getattr(self, name)

            if type(value) is not int or value <= 0:
                raise ValueError(f"{name} must be a positive integer")

        if type(self.evaluation_seed) is not int or self.evaluation_seed < 0:
            raise ValueError("evaluation_seed must be non-negative")

        if type(self.stop_on_first_match) is not bool:
            raise TypeError("stop_on_first_match must be a boolean")


@dataclass(frozen=True, slots=True)
class CandidateResult:
    candidate_id: int
    level_id: str
    valid: bool
    validation_reason: str | None
    classification: DifficultyClassification | None


@dataclass(frozen=True, slots=True)
class ScreeningResult:
    target: Difficulty
    candidates: tuple[CandidateResult, ...]
    selected_candidate_id: int | None
    selected_level_id: str | None

    @property
    def found(self) -> bool:
        return self.selected_candidate_id is not None


class CandidateScreener:
    def __init__(
        self,
        *,
        generation_spec: GenerationSpec,
        screening_config: ScreeningConfig,
    ) -> None:
        self.generation_spec = generation_spec
        self.screening_config = screening_config

        self.generator = LevelGenerator(generation_spec)
        self.validator = LevelValidator()

        self.evaluator = DifficultyEvaluator(
            episodes_per_agent=screening_config.episodes_per_candidate,
            seed=screening_config.evaluation_seed,
            agents=("MCTS_greedy_outcome",),
            mcts_simulations=screening_config.mcts_simulations,
        )

    def screen(self) -> ScreeningResult:
        candidates = []
        selected_candidate_id = None
        selected_level_id = None

        for candidate_id in range(self.screening_config.max_candidates):
            level = self.generator.generate(candidate_id)

            validation = self.validator.validate(level)

            if not validation.valid:
                candidates.append(
                    CandidateResult(
                        candidate_id=candidate_id,
                        level_id=level.level_id,
                        valid=False,
                        validation_reason=validation.reason,
                        classification=None,
                    )
                )
                continue

            report = self.evaluator.evaluate(level)

            classification = classify_difficulty(
                report,
                self.screening_config.target,
            )

            candidates.append(
                CandidateResult(
                    candidate_id=candidate_id,
                    level_id=level.level_id,
                    valid=True,
                    validation_reason=None,
                    classification=classification,
                )
            )

            if (
                classification.status == DifficultyMatch.MATCH
                and self.screening_config.stop_on_first_match
            ):
                selected_candidate_id = candidate_id
                selected_level_id = level.level_id
                break

        return ScreeningResult(
            target=self.screening_config.target,
            candidates=tuple(candidates),
            selected_candidate_id=selected_candidate_id,
            selected_level_id=selected_level_id,
        )
