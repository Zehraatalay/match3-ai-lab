from dataclasses import replace

import pytest

from match3_ai.training.dqn import TrainingConfig
from match3_ai.training.seed_policy import (
    SPRINT520_TRAINING_PARTITION,
    seed_policy_metadata,
    validate_sprint520_training,
)


def test_sprint520_training_partition():
    assert SPRINT520_TRAINING_PARTITION == (500000, 699999)

    validate_sprint520_training(500000, 150000)

    with pytest.raises(ValueError):
        validate_sprint520_training(499999, 150000)

    with pytest.raises(ValueError):
        validate_sprint520_training(600000, 150000)


def test_sprint520_configuration():
    config = TrainingConfig(
        seed=5990,
        environment_steps=150000,
        train_seed_start=500000,
        evaluation_seed_start=310000,
        evaluation_episodes=200,
        evaluation_every=25000,
        evaluation_agent_seed_start=621900000,
        evaluation_partition="sprint5_19_validation",
        seed_policy="sprint5_20",
        evaluate_step_zero=True,
        baseline_every_checkpoint=False,
        algorithm="double",
        architecture="dueling",
    )

    assert config.environment_steps == 150000
    assert config.algorithm == "double"
    assert config.architecture == "dueling"


def test_sprint520_metadata():
    metadata = seed_policy_metadata(
        "sprint5_19_validation",
        training_policy="sprint5_20",
    )

    assert metadata["name"] == "sprint5_20"
    assert metadata["final_test_used"] is False
    assert metadata["environment_partitions_inclusive"]["sprint5_20_training"] == [
        500000,
        699999,
    ]


def test_legacy_metadata_unchanged():
    assert seed_policy_metadata()["name"] == "sprint5_8"
    assert seed_policy_metadata("sprint5_19_validation")["name"] == "sprint5_19"


@pytest.fixture
def training_config():
    return TrainingConfig(
        environment_steps=150000,
        train_seed_start=500000,
        evaluation_seed_start=310000,
        evaluation_episodes=200,
        evaluation_agent_seed_start=621900000,
        evaluation_partition="sprint5_19_validation",
        seed_policy="sprint5_20",
    )


@pytest.mark.parametrize("policy", ["sprint5_8", "sprint5_20"])
@pytest.mark.parametrize("start", [309801, 309999, 310000, 310100, 310199])
def test_baseline_rng_rejects_study_partition_overlap(training_config, policy, start):
    with pytest.raises(ValueError, match="baseline agent seeds"):
        replace(
            training_config,
            seed_policy=policy,
            train_seed_start=200000 if policy == "sprint5_8" else 500000,
            environment_steps=50000 if policy == "sprint5_8" else 150000,
            evaluation_agent_seed_start=start,
        )


@pytest.mark.parametrize("policy", ["sprint5_8", "sprint5_20"])
@pytest.mark.parametrize("start", [309800, 310200, 621900000])
def test_baseline_rng_accepts_disjoint_boundary_ranges(training_config, policy, start):
    config = replace(
        training_config,
        seed_policy=policy,
        train_seed_start=200000 if policy == "sprint5_8" else 500000,
        environment_steps=50000 if policy == "sprint5_8" else 150000,
        evaluation_agent_seed_start=start,
    )
    assert config.evaluation_agent_seed_start == start


@pytest.mark.parametrize("start", [200000, 300000, 400000, 499801, 500000, 699999])
def test_baseline_rng_still_rejects_other_reserved_ranges(training_config, start):
    with pytest.raises(ValueError, match="baseline agent seeds"):
        replace(training_config, evaluation_agent_seed_start=start)


@pytest.mark.parametrize(
    "start,count", [(500000, 200000), (550000, 150000), (699999, 1)]
)
def test_extended_training_partition_inclusive_boundaries(start, count):
    validate_sprint520_training(start, count)


@pytest.mark.parametrize(
    "start,count",
    [(499999, 1), (700000, 1), (550001, 150000), (500000, 0), (500000, -1)],
)
def test_extended_training_partition_rejects_invalid_ranges(start, count):
    with pytest.raises(ValueError, match="training seeds"):
        validate_sprint520_training(start, count)
