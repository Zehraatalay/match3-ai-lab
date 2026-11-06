from dataclasses import dataclass
from enum import Enum

from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.specials import SpecialType

Position = tuple[int, int]
CellMask = tuple[tuple[bool, ...], ...]

BLOCKER_VARIANT_HP: dict[BlockerType, dict[str, int]] = {
    BlockerType.CRATE: {"SINGLE": 1, "DOUBLE": 2, "TRIPLE": 3},
    BlockerType.ICE: {"THIN": 1, "THICK": 2},
    BlockerType.STONE: {"SINGLE": 1, "DOUBLE": 2},
}


class ObjectiveType(str, Enum):
    CLEAR_ALL_BLOCKERS = "CLEAR_ALL_BLOCKERS"
    COLLECT_TILES = "COLLECT_TILES"


@dataclass(frozen=True, slots=True)
class BlockerPlacement:
    position: Position
    blocker_type: BlockerType
    variant: str

    @property
    def hp(self) -> int:
        return BLOCKER_VARIANT_HP[self.blocker_type][self.variant]


@dataclass(frozen=True, slots=True)
class InitialSpecial:
    position: Position
    special_type: SpecialType


@dataclass(frozen=True, slots=True)
class LevelObjective:
    objective_type: ObjectiveType
    target: int | None = None
    color: int | None = None


@dataclass(frozen=True, slots=True)
class LevelConfig:
    rows: int
    cols: int
    num_colors: int
    move_limit: int
    level_id: str = "custom"
    name: str = "Custom Level"
    seed: int = 0
    cell_mask: CellMask | None = None
    blockers: tuple[BlockerPlacement, ...] = ()
    initial_specials: tuple[InitialSpecial, ...] = ()
    objectives: tuple[LevelObjective, ...] = ()

    def __post_init__(self) -> None:
        if type(self.rows) is not int or self.rows < 3:
            raise ValueError("rows must be an integer >= 3")

        if type(self.cols) is not int or self.cols < 3:
            raise ValueError("cols must be an integer >= 3")

        if type(self.num_colors) is not int or not 3 <= self.num_colors <= 254:
            raise ValueError("num_colors must be an integer between 3 and 254")

        if type(self.move_limit) is not int or self.move_limit <= 0:
            raise ValueError("move_limit must be a positive integer")

        if type(self.seed) is not int or self.seed < 0:
            raise ValueError("seed must be a non-negative integer")

        if not isinstance(self.level_id, str) or not self.level_id.strip():
            raise ValueError("level_id must be a non-empty string")

        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("name must be a non-empty string")

        if self.cell_mask is not None:
            if not isinstance(self.cell_mask, tuple) or len(self.cell_mask) != self.rows:
                raise ValueError("cell_mask must contain exactly rows tuple rows")
            for row in self.cell_mask:
                if not isinstance(row, tuple) or len(row) != self.cols:
                    raise ValueError("cell_mask rows must contain exactly cols values")
                if any(type(value) is not bool for value in row):
                    raise ValueError("cell_mask values must be booleans")

        if not isinstance(self.blockers, tuple):
            raise TypeError("blockers must be an immutable tuple")
        if not isinstance(self.initial_specials, tuple):
            raise TypeError("initial_specials must be an immutable tuple")
        if not isinstance(self.objectives, tuple):
            raise TypeError("objectives must be an immutable tuple")

        blocker_positions: set[Position] = set()
        blockers_by_position: dict[Position, BlockerType] = {}
        for blocker in self.blockers:
            if not isinstance(blocker, BlockerPlacement):
                raise TypeError("blockers must contain BlockerPlacement values")
            self._validate_position(blocker.position, "blocker")
            if blocker.position in blocker_positions:
                raise ValueError("duplicate blocker position")
            blocker_positions.add(blocker.position)
            blockers_by_position[blocker.position] = blocker.blocker_type

            if not isinstance(blocker.blocker_type, BlockerType):
                raise TypeError("unknown blocker type")
            if not isinstance(blocker.variant, str):
                raise TypeError("blocker variant must be a string")
            variants = BLOCKER_VARIANT_HP.get(blocker.blocker_type)
            if variants is None or blocker.variant not in variants:
                raise ValueError(
                    f"invalid variant {blocker.variant!r} for "
                    f"{blocker.blocker_type.name}"
                )
            self._validate_playable(blocker.position, "blocker")

        special_positions: set[Position] = set()
        for special in self.initial_specials:
            if not isinstance(special, InitialSpecial):
                raise TypeError("initial_specials must contain InitialSpecial values")
            self._validate_position(special.position, "special")
            if special.position in special_positions:
                raise ValueError("duplicate special position")
            special_positions.add(special.position)
            self._validate_playable(special.position, "special")

            if not isinstance(special.special_type, SpecialType):
                raise TypeError("unknown special type")
            if special.special_type == SpecialType.NONE:
                raise ValueError("initial special type cannot be NONE")

            blocker_type = blockers_by_position.get(special.position)
            if blocker_type in (BlockerType.CRATE, BlockerType.STONE):
                raise ValueError(
                    "specials cannot be placed on CRATE or STONE blockers"
                )

        clear_all_count = 0
        for objective in self.objectives:
            if not isinstance(objective, LevelObjective):
                raise TypeError("objectives must contain LevelObjective values")

            if objective.objective_type == ObjectiveType.CLEAR_ALL_BLOCKERS:
                clear_all_count += 1
                if objective.target is not None or objective.color is not None:
                    raise ValueError(
                        "CLEAR_ALL_BLOCKERS does not accept target or color"
                    )
            elif objective.objective_type == ObjectiveType.COLLECT_TILES:
                if type(objective.target) is not int or objective.target <= 0:
                    raise ValueError("COLLECT_TILES target must be a positive integer")
                if objective.color is not None and (
                    type(objective.color) is not int
                    or not 1 <= objective.color <= self.num_colors
                ):
                    raise ValueError("COLLECT_TILES color is outside the level colors")
            else:
                raise ValueError("unsupported objective type")

        if clear_all_count > 1:
            raise ValueError("duplicate CLEAR_ALL_BLOCKERS objective")
        if self.blockers and clear_all_count != 1:
            raise ValueError(
                "levels containing blockers require CLEAR_ALL_BLOCKERS"
            )
        if clear_all_count and not self.blockers:
            raise ValueError(
                "CLEAR_ALL_BLOCKERS is only valid when the level contains blockers"
            )

    def _validate_position(self, position: Position, label: str) -> None:
        if (
            not isinstance(position, tuple)
            or len(position) != 2
            or any(type(value) is not int for value in position)
        ):
            raise ValueError(f"{label} position must be a [row, col] integer pair")
        row, col = position
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            raise ValueError(f"{label} position is outside the board")

    def _validate_playable(self, position: Position, label: str) -> None:
        if self.cell_mask is not None and not self.cell_mask[position[0]][position[1]]:
            raise ValueError(f"{label} cannot be placed on a masked cell")
