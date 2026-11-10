import json
from pathlib import Path

from match3_ai.levels.config import (
    LevelConfig,
    ObjectiveType,
)


def level_to_dict(level: LevelConfig) -> dict:
    """Convert a LevelConfig into the existing loader JSON schema."""
    if not isinstance(level, LevelConfig):
        raise TypeError("level must be a LevelConfig")

    objectives = []

    for objective in level.objectives:
        item = {"type": objective.objective_type.value}

        if objective.objective_type == ObjectiveType.COLLECT_TILES:
            item["target"] = objective.target
            if objective.color is not None:
                item["color"] = objective.color

        objectives.append(item)

    return {
        "level_id": level.level_id,
        "name": level.name,
        "rows": level.rows,
        "cols": level.cols,
        "num_colors": level.num_colors,
        "move_limit": level.move_limit,
        "seed": level.seed,
        "cell_mask": (
            [list(row) for row in level.cell_mask]
            if level.cell_mask is not None
            else None
        ),
        "blockers": [
            {
                "position": list(blocker.position),
                "type": blocker.blocker_type.name,
                "variant": blocker.variant,
            }
            for blocker in level.blockers
        ],
        "initial_specials": [
            {
                "position": list(special.position),
                "type": special.special_type.name,
            }
            for special in level.initial_specials
        ],
        "objectives": objectives,
    }


def export_level(level: LevelConfig, path: str | Path) -> Path:
    """Export a level that can be loaded by load_level()."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = level_to_dict(level)

    # The loader requires cell_mask to be absent when it is not defined.
    if data["cell_mask"] is None:
        del data["cell_mask"]

    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return path


from dataclasses import asdict

from match3_ai.levels.generation import GenerationSpec
from match3_ai.levels.targeted_generation import (
    TargetGenerationConfig,
    TargetGenerationResult,
)


def export_generation_report(
    result: TargetGenerationResult,
    generation_spec: GenerationSpec,
    config: TargetGenerationConfig,
    path: str | Path,
) -> Path:
    """Export screening and confirmation results."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    candidates = result.screening.candidates

    valid_candidates = sum(candidate.valid for candidate in candidates)

    data = {
        "schema_version": 1,
        "target": result.target.value,
        "found": result.found,
        "selected_level_id": (
            result.selected_level.level_id
            if result.selected_level is not None
            else None
        ),
        "generation_spec": asdict(generation_spec),
        "target_generation_config": {
            **asdict(config),
            "target": config.target.value,
        },
        "statistics": {
            "attempted_candidates": len(candidates),
            "valid_candidates": valid_candidates,
            "invalid_candidates": len(candidates) - valid_candidates,
            "confirmation_attempts": len(result.confirmation_attempts),
        },
        "screening": [
            {
                **asdict(candidate),
                "classification": (
                    {
                        **asdict(candidate.classification),
                        "target": candidate.classification.target.value,
                        "status": candidate.classification.status.value,
                    }
                    if candidate.classification is not None
                    else None
                ),
            }
            for candidate in candidates
        ],
        "confirmation_attempts": [
            {
                **asdict(attempt),
                "target": attempt.target.value,
                "status": attempt.status.value,
            }
            for attempt in result.confirmation_attempts
        ],
    }

    path.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return path
