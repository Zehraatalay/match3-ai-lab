import json
from pathlib import Path
from typing import Any

from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.specials import SpecialType
from match3_ai.levels.config import (
    BlockerPlacement,
    InitialSpecial,
    LevelConfig,
    LevelObjective,
    ObjectiveType,
)

_REQUIRED_FIELDS = {
    "level_id",
    "name",
    "rows",
    "cols",
    "num_colors",
    "move_limit",
    "seed",
    "objectives",
}
_OPTIONAL_FIELDS = {"cell_mask", "blockers", "initial_specials"}


def load_level(path: str | Path) -> LevelConfig:
    """Load and validate a JSON level configuration from a filesystem path."""

    level_path = Path(path)
    try:
        raw_json = level_path.read_text(encoding="utf-8")
    except FileNotFoundError as error:
        raise FileNotFoundError(f"level file not found: {level_path}") from error
    except UnicodeDecodeError as error:
        raise ValueError(
            f"invalid JSON encoding in level file: {level_path}"
        ) from error

    try:
        data = json.loads(raw_json)
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise ValueError(f"invalid JSON in level file: {level_path}") from error

    if not isinstance(data, dict):
        raise TypeError("level JSON root must be an object")

    unknown_fields = set(data) - _REQUIRED_FIELDS - _OPTIONAL_FIELDS
    if unknown_fields:
        raise ValueError(f"unknown level fields: {', '.join(sorted(unknown_fields))}")

    missing_fields = _REQUIRED_FIELDS - set(data)
    if missing_fields:
        raise ValueError(f"missing level fields: {', '.join(sorted(missing_fields))}")

    for field in ("level_id", "name"):
        if not isinstance(data[field], str):
            raise TypeError(f"{field} must be a string")
    for field in ("rows", "cols", "num_colors", "move_limit", "seed"):
        if type(data[field]) is not int:
            raise TypeError(f"{field} must be an integer")

    rows = data["rows"]
    cols = data["cols"]
    cell_mask = (
        _parse_cell_mask(data["cell_mask"], rows, cols) if "cell_mask" in data else None
    )
    blockers = _parse_blockers(data.get("blockers", []))
    initial_specials = _parse_initial_specials(data.get("initial_specials", []))
    objectives = _parse_objectives(data["objectives"], data["num_colors"])
    if not objectives:
        raise ValueError("objectives must be a non-empty array")

    try:
        return LevelConfig(
            level_id=data["level_id"],
            name=data["name"],
            rows=rows,
            cols=cols,
            num_colors=data["num_colors"],
            move_limit=data["move_limit"],
            seed=data["seed"],
            cell_mask=cell_mask,
            blockers=blockers,
            initial_specials=initial_specials,
            objectives=objectives,
        )
    except (TypeError, ValueError, KeyError) as error:
        raise ValueError(f"invalid level configuration: {error}") from error


def _parse_cell_mask(value: Any, rows: int, cols: int):
    if not isinstance(value, list) or len(value) != rows:
        raise ValueError("cell_mask must be an array with exactly rows entries")

    mask_rows = []
    for row in value:
        if not isinstance(row, list) or len(row) != cols:
            raise ValueError("cell_mask rows must be arrays with exactly cols entries")
        if any(type(cell) is not bool for cell in row):
            raise ValueError("cell_mask values must be booleans")
        mask_rows.append(tuple(row))
    return tuple(mask_rows)


def _parse_position(value: Any, label: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(type(coordinate) is not int for coordinate in value)
    ):
        raise ValueError(f"{label} position must be a [row, col] integer pair")
    return value[0], value[1]


def _reject_unknown_fields(
    value: dict[str, Any],
    allowed: set[str],
    label: str,
) -> None:
    unknown_fields = set(value) - allowed
    if unknown_fields:
        raise ValueError(f"unknown {label} fields: {', '.join(sorted(unknown_fields))}")


def _parse_blockers(value: Any) -> tuple[BlockerPlacement, ...]:
    if not isinstance(value, list):
        raise TypeError("blockers must be an array")

    placements = []
    for index, item in enumerate(value):
        label = f"blockers[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{label} must be an object")
        _reject_unknown_fields(item, {"position", "type", "variant"}, label)
        missing = {"position", "type", "variant"} - set(item)
        if missing:
            raise ValueError(f"{label} is missing fields: {', '.join(sorted(missing))}")
        if not isinstance(item["type"], str):
            raise TypeError(f"{label}.type must be a string")
        if not isinstance(item["variant"], str):
            raise TypeError(f"{label}.variant must be a string")
        try:
            blocker_type = BlockerType[item["type"]]
        except KeyError as error:
            raise ValueError(f"invalid blocker type: {item['type']}") from error
        if blocker_type == BlockerType.NONE:
            raise ValueError("blocker type cannot be NONE")
        placements.append(
            BlockerPlacement(
                position=_parse_position(item["position"], label),
                blocker_type=blocker_type,
                variant=item["variant"],
            )
        )
    return tuple(placements)


def _parse_initial_specials(value: Any) -> tuple[InitialSpecial, ...]:
    if not isinstance(value, list):
        raise TypeError("initial_specials must be an array")

    specials = []
    for index, item in enumerate(value):
        label = f"initial_specials[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{label} must be an object")
        _reject_unknown_fields(item, {"position", "type"}, label)
        missing = {"position", "type"} - set(item)
        if missing:
            raise ValueError(f"{label} is missing fields: {', '.join(sorted(missing))}")
        if not isinstance(item["type"], str):
            raise TypeError(f"{label}.type must be a string")
        try:
            special_type = SpecialType[item["type"]]
        except KeyError as error:
            raise ValueError(f"invalid special type: {item['type']}") from error
        if special_type == SpecialType.NONE:
            raise ValueError("initial special type cannot be NONE")
        specials.append(
            InitialSpecial(
                position=_parse_position(item["position"], label),
                special_type=special_type,
            )
        )
    return tuple(specials)


def _parse_objectives(value: Any, num_colors: int) -> tuple[LevelObjective, ...]:
    if not isinstance(value, list):
        raise TypeError("objectives must be an array")

    objectives = []
    for index, item in enumerate(value):
        label = f"objectives[{index}]"
        if not isinstance(item, dict):
            raise TypeError(f"{label} must be an object")
        if not isinstance(item.get("type"), str):
            raise TypeError(f"{label}.type must be a string")

        objective_name = item["type"]
        try:
            objective_type = ObjectiveType(objective_name)
        except ValueError as error:
            if objective_name == "REACH_SCORE":
                raise ValueError(
                    "REACH_SCORE objectives are not supported yet"
                ) from error
            raise ValueError(f"unsupported objective type: {objective_name}") from error

        if objective_type == ObjectiveType.CLEAR_ALL_BLOCKERS:
            _reject_unknown_fields(item, {"type"}, label)
            objectives.append(LevelObjective(objective_type))
            continue

        _reject_unknown_fields(item, {"type", "target", "color"}, label)
        if "target" not in item or type(item["target"]) is not int:
            raise ValueError(f"{label}.target must be an integer")
        if item["target"] <= 0:
            raise ValueError(f"{label}.target must be a positive integer")
        color = item.get("color")
        if color is not None and (
            type(color) is not int or not 1 <= color <= num_colors
        ):
            raise ValueError(f"{label}.color must be between 1 and num_colors")
        objectives.append(
            LevelObjective(
                objective_type=objective_type,
                target=item["target"],
                color=color,
            )
        )
    return tuple(objectives)
