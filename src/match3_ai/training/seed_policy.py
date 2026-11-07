"""Reserved environment seed partitions for the Sprint 5.8 pilot."""

PARTITIONS = {
    "training": (200000, 299999),
    "validation": (300000, 300999),
    "final_test": (400000, 400999),
}

STUDY_VALIDATION_PARTITIONS = {"sprint5_19_validation": (310000, 310199)}
SPRINT520_TRAINING_PARTITION = (500000, 699999)


def validate_partition(split: str, start: int, count: int) -> None:
    """Validate an inclusive contiguous seed interval without generating boards."""
    if split not in PARTITIONS:
        raise ValueError(f"unknown seed partition: {split}")
    low, high = PARTITIONS[split]
    if (
        type(start) is not int
        or type(count) is not int
        or count <= 0
        or start < low
        or start + count - 1 > high
    ):
        raise ValueError(f"{split} seeds must remain in [{low}, {high}]")


def validate_evaluation_partition(split: str, start: int, count: int) -> None:
    if split == "validation":
        validate_partition(split, start, count)
        return
    if split not in STUDY_VALIDATION_PARTITIONS:
        raise ValueError("unknown development-validation partition")
    low, high = STUDY_VALIDATION_PARTITIONS[split]
    if (
        type(start) is not int
        or type(count) is not int
        or count <= 0
        or start < low
        or start + count - 1 > high
    ):
        raise ValueError(f"{split} seeds must remain in [{low}, {high}]")


def seed_policy_metadata(
    evaluation_partition="validation",
    training_policy="sprint5_8",
) -> dict:
    partitions = dict(PARTITIONS)

    if training_policy == "sprint5_20":
        partitions["sprint5_20_training"] = SPRINT520_TRAINING_PARTITION
    elif training_policy != "sprint5_8":
        raise ValueError("unknown training seed policy")

    if evaluation_partition != "validation":
        if evaluation_partition not in STUDY_VALIDATION_PARTITIONS:
            raise ValueError("unknown development-validation partition")

        partitions[evaluation_partition] = STUDY_VALIDATION_PARTITIONS[
            evaluation_partition
        ]

    if training_policy == "sprint5_20":
        policy_name = "sprint5_20"
    elif evaluation_partition == "validation":
        policy_name = "sprint5_8"
    else:
        policy_name = "sprint5_19"

    return {
        "name": policy_name,
        "environment_partitions_inclusive": {
            key: list(bounds) for key, bounds in partitions.items()
        },
        "evaluation_split": evaluation_partition,
        "final_test_used": False,
        "fixed_validation_seeds_across_checkpoints": True,
        "agent_rng": (
            "independent spawned training streams and separate baseline seeds"
        ),
    }


def validate_sprint520_training(start: int, count: int) -> None:
    """validate the reserved Sprint 5.20 training seed interval."""
    low, high = SPRINT520_TRAINING_PARTITION

    if (
        type(start) is not int
        or type(count) is not int
        or count <= 0
        or start < low
        or start + count - 1 > high
    ):
        raise ValueError(f"Sprint 5.20 training seeds must remain in [{low}, {high}]")
