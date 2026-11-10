import pytest

from match3_ai.levels.generation import GenerationSpec, LevelGenerator
from match3_ai.levels.validation import LevelValidator


def test_generation_is_deterministic():
    spec = GenerationSpec(seed=42)
    generator = LevelGenerator(spec)

    first = generator.generate(candidate_id=5)
    second = generator.generate(candidate_id=5)

    assert first == second


def test_generation_is_order_independent():
    spec = GenerationSpec(seed=42)
    generator = LevelGenerator(spec)

    first = generator.generate(candidate_id=7)

    generator.generate(candidate_id=2)
    generator.generate(candidate_id=3)

    second = generator.generate(candidate_id=7)

    assert first == second


def test_generated_configs_satisfy_requested_ranges():
    spec = GenerationSpec(
        min_moves=8,
        max_moves=15,
        min_blockers=2,
        max_blockers=6,
    )

    generator = LevelGenerator(spec)

    for candidate_id in range(20):
        config = generator.generate(candidate_id)

        assert 8 <= config.move_limit <= 15
        assert 2 <= len(config.blockers) <= 6


def test_generated_levels_can_initialize():
    spec = GenerationSpec(
        rows=8,
        cols=8,
        min_blockers=1,
        max_blockers=4,
    )

    generator = LevelGenerator(spec)
    validator = LevelValidator()

    for candidate_id in range(10):
        config = generator.generate(candidate_id)
        result = validator.validate(config)

        assert result.valid, f"{config.level_id}: {result.reason}"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 2},
        {"cols": 2},
        {"min_moves": 0},
        {"min_moves": 20, "max_moves": 10},
        {"min_blockers": -1},
        {"min_blockers": 8, "max_blockers": 3},
        {"seed": -1},
    ],
)
def test_invalid_generation_spec(kwargs):
    with pytest.raises(ValueError):
        GenerationSpec(**kwargs)


import pytest


@pytest.mark.parametrize(
    "rows, cols",
    [
        (10, 8),
        (9, 9),
        (10, 10),
    ],
)
def test_generation_spec_rejects_unsupported_board_shapes(rows, cols):
    with pytest.raises(ValueError, match="observation capacity"):
        GenerationSpec(rows=rows, cols=cols)
