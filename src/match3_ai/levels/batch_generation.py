from dataclasses import dataclass

from match3_ai.levels.config import LevelConfig
from match3_ai.levels.generation import GenerationSpec, LevelGenerator
from match3_ai.levels.validation import LevelValidator


@dataclass(frozen=True, slots=True)
class RejectedCandidate:
    candidate_id: int
    reason: str


@dataclass(frozen=True, slots=True)
class GenerationBatch:
    accepted: tuple[LevelConfig, ...]
    rejected: tuple[RejectedCandidate, ...]
    attempted: int

    @property
    def acceptance_rate(self) -> float:
        if self.attempted == 0:
            return 0.0
        return len(self.accepted) / self.attempted


def generate_valid_levels(
    spec: GenerationSpec,
    *,
    target_count: int,
    max_candidates: int,
) -> GenerationBatch:
    if type(target_count) is not int or target_count <= 0:
        raise ValueError("target_count must be a positive integer")

    if type(max_candidates) is not int or max_candidates <= 0:
        raise ValueError("max_candidates must be a positive integer")

    generator = LevelGenerator(spec)
    validator = LevelValidator()

    accepted = []
    rejected = []
    attempted = 0

    for candidate_id in range(max_candidates):
        if len(accepted) >= target_count:
            break

        attempted += 1

        config = generator.generate(candidate_id)
        result = validator.validate(config)

        if result.valid:
            accepted.append(config)
        else:
            rejected.append(
                RejectedCandidate(
                    candidate_id=candidate_id,
                    reason=result.reason or "Unknown validation failure",
                )
            )

    return GenerationBatch(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        attempted=attempted,
    )
