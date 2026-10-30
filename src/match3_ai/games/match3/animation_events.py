from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto

from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.specials import SpecialType

Position = tuple[int, int]


class AnimationEventType(Enum):
    SWAP = auto()
    MATCH = auto()
    SPECIAL_CREATION = auto()
    SPECIAL_ACTIVATION = auto()
    CLEAR = auto()
    GRAVITY = auto()
    REFILL = auto()
    RESHUFFLE = auto()


@dataclass(frozen=True, slots=True)
class SpecialActivationInfo:
    position: Position
    special_type: SpecialType
    effect_cells: frozenset[Position] = frozenset()


@dataclass(frozen=True, slots=True)
class SpecialCombinationInfo:
    source: Position
    destination: Position
    source_type: SpecialType
    destination_type: SpecialType
    effect_cells: frozenset[Position]

@dataclass(frozen=True, slots=True)
class AnimationEvent:
    event_type: AnimationEventType
    state: BoardState
    affected_cells: frozenset[Position] = frozenset()
    special_activations: tuple[SpecialActivationInfo, ...] = ()
    special_combination: SpecialCombinationInfo | None = None