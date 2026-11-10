import pytest

from match3_ai.levels.batch_generation import generate_valid_levels
from match3_ai.levels.generation import GenerationSpec


def test_batch_generation_returns_requested_count():
    spec = GenerationSpec(
        rows=8,
        cols=8,
        min_blockers=1,
        max_blockers=4,
        seed=42,
    )

    batch = generate_valid_levels(
        spec,
        target_count=5,
        max_candidates=20,
    )

    assert len(batch.accepted) == 5
    assert batch.attempted <= 20


def test_batch_generation_is_deterministic():
    spec = GenerationSpec(seed=42)

    first = generate_valid_levels(
        spec,
        target_count=5,
        max_candidates=20,
    )

    second = generate_valid_levels(
        spec,
        target_count=5,
        max_candidates=20,
    )

    assert first == second


def test_batch_generation_tracks_attempts():
    spec = GenerationSpec(seed=42)

    batch = generate_valid_levels(
        spec,
        target_count=3,
        max_candidates=10,
    )

    assert batch.attempted == len(batch.accepted) + len(batch.rejected)


@pytest.mark.parametrize(
    "target_count,max_candidates",
    [
        (0, 10),
        (-1, 10),
        (5, 0),
        (5, -1),
        (True, 10),
    ],
)
def test_invalid_batch_parameters(target_count, max_candidates):
    with pytest.raises(ValueError):
        generate_valid_levels(
            GenerationSpec(),
            target_count=target_count,
            max_candidates=max_candidates,
        )
