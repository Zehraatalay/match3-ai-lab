from unittest.mock import Mock

import pytest

from match3_ai.levels.difficulty import (
    Difficulty,
    DifficultyClassification,
    DifficultyMatch,
)
from match3_ai.levels.generation import GenerationSpec
from match3_ai.levels.screening import CandidateResult, ScreeningResult
from match3_ai.levels.targeted_generation import (
    TargetDifficultyGenerator,
    TargetGenerationConfig,
)


def classification(level_id, status):
    return DifficultyClassification(
        level_id=level_id,
        target=Difficulty.HARD,
        status=status,
        win_rate=0.25,
        ci_lower=0.15,
        ci_upper=0.35,
        episodes=100,
    )


def make_generator():
    return TargetDifficultyGenerator(
        generation_spec=GenerationSpec(seed=42),
        config=TargetGenerationConfig(
            target=Difficulty.HARD,
            max_candidates=3,
            screening_episodes=5,
            confirmation_episodes=20,
        ),
    )


def test_accepts_confirmed_candidate(monkeypatch):
    generator = make_generator()

    candidate = CandidateResult(
        candidate_id=1,
        level_id="generated-42-1",
        valid=True,
        validation_reason=None,
        classification=classification(
            "generated-42-1",
            DifficultyMatch.UNCERTAIN,
        ),
    )

    generator.screener.screen = Mock(
        return_value=ScreeningResult(
            target=Difficulty.HARD,
            candidates=(candidate,),
            selected_candidate_id=None,
            selected_level_id=None,
        )
    )

    generator.confirmation_evaluator.evaluate = Mock(
        side_effect=lambda level: Mock(level_id=level.level_id)
    )

    monkeypatch.setattr(
        "match3_ai.levels.targeted_generation.classify_difficulty",
        lambda report, target: classification(
            report.level_id,
            DifficultyMatch.MATCH,
        ),
    )

    result = generator.generate()

    assert result.found
    assert result.selected_level.level_id == "generated-42-1"
    assert result.confirmation.status == DifficultyMatch.MATCH


def test_rejects_unconfirmed_candidate(monkeypatch):
    generator = make_generator()

    candidate = CandidateResult(
        candidate_id=0,
        level_id="generated-42-0",
        valid=True,
        validation_reason=None,
        classification=classification(
            "generated-42-0",
            DifficultyMatch.UNCERTAIN,
        ),
    )

    generator.screener.screen = Mock(
        return_value=ScreeningResult(
            target=Difficulty.HARD,
            candidates=(candidate,),
            selected_candidate_id=None,
            selected_level_id=None,
        )
    )

    generator.confirmation_evaluator.evaluate = Mock(
        side_effect=lambda level: Mock(level_id=level.level_id)
    )

    monkeypatch.setattr(
        "match3_ai.levels.targeted_generation.classify_difficulty",
        lambda report, target: classification(
            report.level_id,
            DifficultyMatch.REJECT,
        ),
    )

    result = generator.generate()

    assert not result.found
    assert result.selected_level is None


def test_no_candidates():
    generator = make_generator()

    generator.screener.screen = Mock(
        return_value=ScreeningResult(
            target=Difficulty.HARD,
            candidates=(),
            selected_candidate_id=None,
            selected_level_id=None,
        )
    )

    generator.confirmation_evaluator.evaluate = Mock()

    result = generator.generate()

    assert not result.found
    generator.confirmation_evaluator.evaluate.assert_not_called()


def test_overlapping_seed_ranges():
    with pytest.raises(ValueError, match="must not overlap"):
        TargetGenerationConfig(
            target=Difficulty.HARD,
            screening_seed=1000,
            screening_episodes=100,
            confirmation_seed=1050,
            confirmation_episodes=100,
        )
