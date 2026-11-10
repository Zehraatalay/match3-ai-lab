from unittest.mock import Mock

import pytest

from match3_ai.levels.difficulty import (
    Difficulty,
    DifficultyClassification,
    DifficultyMatch,
)
from match3_ai.levels.generation import GenerationSpec
from match3_ai.levels.screening import (
    CandidateScreener,
    ScreeningConfig,
)
from match3_ai.levels.validation import ValidationResult


def make_classification(level_id, status):
    return DifficultyClassification(
        level_id=level_id,
        target=Difficulty.HARD,
        status=status,
        win_rate=0.25,
        ci_lower=0.15,
        ci_upper=0.35,
        episodes=30,
    )


def test_screening_stops_on_first_match(monkeypatch):
    screener = CandidateScreener(
        generation_spec=GenerationSpec(seed=42),
        screening_config=ScreeningConfig(
            target=Difficulty.HARD,
            max_candidates=5,
            episodes_per_candidate=5,
        ),
    )

    screener.validator.validate = Mock(
        return_value=ValidationResult(
            valid=True,
            legal_move_count=10,
        )
    )

    statuses = iter(
        [
            DifficultyMatch.REJECT,
            DifficultyMatch.UNCERTAIN,
            DifficultyMatch.MATCH,
        ]
    )

    def fake_classify(report, target):
        return make_classification(
            report.level_id,
            next(statuses),
        )

    monkeypatch.setattr(
        "match3_ai.levels.screening.classify_difficulty",
        fake_classify,
    )

    screener.evaluator.evaluate = Mock(
        side_effect=lambda level: Mock(level_id=level.level_id)
    )

    result = screener.screen()

    assert result.found
    assert result.selected_candidate_id == 2
    assert len(result.candidates) == 3


def test_screening_returns_not_found(monkeypatch):
    screener = CandidateScreener(
        generation_spec=GenerationSpec(seed=42),
        screening_config=ScreeningConfig(
            target=Difficulty.HARD,
            max_candidates=3,
            episodes_per_candidate=5,
        ),
    )

    screener.validator.validate = Mock(return_value=ValidationResult(valid=True))

    screener.evaluator.evaluate = Mock(
        side_effect=lambda level: Mock(level_id=level.level_id)
    )

    monkeypatch.setattr(
        "match3_ai.levels.screening.classify_difficulty",
        lambda report, target: make_classification(
            report.level_id,
            DifficultyMatch.REJECT,
        ),
    )

    result = screener.screen()

    assert not result.found
    assert result.selected_candidate_id is None
    assert len(result.candidates) == 3


def test_invalid_candidates_are_skipped():
    screener = CandidateScreener(
        generation_spec=GenerationSpec(seed=42),
        screening_config=ScreeningConfig(
            target=Difficulty.HARD,
            max_candidates=2,
            episodes_per_candidate=5,
        ),
    )

    screener.validator.validate = Mock(
        return_value=ValidationResult(
            valid=False,
            reason="No legal moves",
        )
    )

    screener.evaluator.evaluate = Mock()

    result = screener.screen()

    assert not result.found
    assert len(result.candidates) == 2
    assert all(not candidate.valid for candidate in result.candidates)

    screener.evaluator.evaluate.assert_not_called()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("max_candidates", 0),
        ("episodes_per_candidate", 0),
        ("mcts_simulations", 0),
        ("evaluation_seed", -1),
    ],
)
def test_invalid_screening_config(field, value):
    kwargs = {
        "target": Difficulty.HARD,
        field: value,
    }

    with pytest.raises(ValueError):
        ScreeningConfig(**kwargs)
