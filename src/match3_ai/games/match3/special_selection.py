from __future__ import annotations

from dataclasses import dataclass

from match3_ai.games.match3.pattern_detection import (
    MatchGroup,
    MatchPattern,
)
from match3_ai.games.match3.specials import SpecialType

Position = tuple[int, int]


@dataclass(frozen=True, slots=True)
class SpecialCreation:
    special_type: SpecialType
    position: Position


# bomb uretimini kontrol ediyor
# hem yatay hem dikey eslesmeleri buluyor ve kesisim var mi diye bakiyor
def _has_intersecting_lines(
    patterns: tuple[MatchPattern, ...],
) -> bool:
    """
    check whether a horizontal and vertical line pattern intersect.
    """
    horizontal_patterns = [
        pattern for pattern in patterns if pattern.kind == "horizontal"
    ]

    vertical_patterns = [pattern for pattern in patterns if pattern.kind == "vertical"]

    for horizontal in horizontal_patterns:
        for vertical in vertical_patterns:
            if horizontal.cells & vertical.cells:
                return True

    return False


# burada priority-based rule evaluation kullaniyoruz
# mesela light ball tespit edildi diyelim return ediliyor direkt diger special typesa bakilmiyor
def _select_special_type(group: MatchGroup) -> SpecialType:
    """
    select the highest-priority special for a match group.
    priority:
    Light Ball > Bomb > Rocket > Propeller
    """

    # 1. light Ball: straight line of 5+
    for pattern in group.patterns:
        if pattern.kind in ("horizontal", "vertical") and len(pattern.cells) >= 5:
            return SpecialType.LIGHT_BALL

    # 2. bomb: intersecting horizontal and vertical lines
    if _has_intersecting_lines(group.patterns):
        return SpecialType.BOMB

    # 3. rocket: straight line of 4
    for pattern in group.patterns:
        if len(pattern.cells) >= 4:
            if pattern.kind == "horizontal":
                return SpecialType.ROCKET_HORIZONTAL

            if pattern.kind == "vertical":
                return SpecialType.ROCKET_VERTICAL

    # 4. propeller: 2x2 square
    for pattern in group.patterns:
        if pattern.kind == "square":
            return SpecialType.PROPELLER

    return SpecialType.NONE


# special konumunu belirliyoruz
# special konumunu belirliyoruz
def select_special_creations(
    groups: list[MatchGroup],
    destination: Position | None = None,
) -> list[SpecialCreation]:
    """
    select at most one special per independent match group.
    prefer the swap destination when provided and part of the group.
    otherwise use the lexicographically smallest cell.
    """
    creations = []

    for group in groups:
        special_type = _select_special_type(group)

        if special_type == SpecialType.NONE:
            continue

        # min() tuplesda lexicographical ordering kullaniyor
        if destination is not None and destination in group.cells:
            position = destination
        else:
            position = min(group.cells)

        creations.append(
            SpecialCreation(
                special_type=special_type,
                position=position,
            )
        )

    return creations
