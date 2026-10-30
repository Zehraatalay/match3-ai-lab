from dataclasses import dataclass

import numpy as np

from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.constants import EMPTY
from match3_ai.games.match3.specials import SpecialType

Position = tuple[int, int]


def rocket_cells(
    state: BoardState,
    position: Position,
    horizontal: bool,
) -> set[Position]:
    """return the cells affected by a horizontal or vertical rocket."""
    row, col = position

    if horizontal:
        return {(row, c) for c in range(state.cols) if state.cell_mask[row, c]}

    return {(r, col) for r in range(state.rows) if state.cell_mask[r, col]}


def bomb_cells(
    state: BoardState,
    position: Position,
    radius: int = 1,
) -> set[Position]:
    """
    return cells affected by a bomb.
    radius=1 -> 3x3
    radius=2 -> 5x5 bomb-bomb icin kullanilacak
    """
    row, col = position

    return {
        (r, c)
        for r in range(max(0, row - radius), min(state.rows, row + radius + 1))
        for c in range(max(0, col - radius), min(state.cols, col + radius + 1))
        if state.cell_mask[r, c]
    }


# propeller activite edildiginde temizlenecek celleri hesapliyor
def propeller_launch_cells(
    state: BoardState,
    position: Position,
) -> set[Position]:
    """return the propeller's launch cell and four orthogonal neighbors."""
    row, col = position

    candidates = {
        (row, col),
        (row - 1, col),
        (row + 1, col),
        (row, col - 1),
        (row, col + 1),
    }

    return {
        (r, c)
        for r, c in candidates
        if 0 <= r < state.rows and 0 <= c < state.cols and state.cell_mask[r, c]
    }


# random hedef seciyor bu islem RNG kullaniyor
def select_propeller_target(
    state: BoardState,
    rng: np.random.Generator,
    excluded_positions: set[Position],
) -> Position | None:
    """
    select a random remaining tile for the propeller.
    the excluded positions are usually the launch area
    and cells already affected by other activations.
    """
    candidates = [
        (row, col)
        for row in range(state.rows)
        for col in range(state.cols)
        if state.cell_mask[row, col]
        and state.tiles[row, col] != EMPTY
        and (row, col) not in excluded_positions
    ]

    if not candidates:
        return None

    index = int(rng.integers(len(candidates)))

    return candidates[index]


def propeller_cells(
    state: BoardState,
    position: Position,
    rng: np.random.Generator,
    excluded_positions: set[Position] | None = None,
) -> set[Position]:
    """
    return the propeller's launch area and one random target.
    """
    affected = propeller_launch_cells(state, position)

    excluded = affected | (
        excluded_positions if excluded_positions is not None else set()
    )

    target = select_propeller_target(
        state=state,
        rng=rng,
        excluded_positions=excluded,
    )

    if target is not None:
        affected.add(target)

    return affected


# sadece hedef renkteki taslari return ediyor light-ball bu kumeye dahil degil
# light ball'ın hangi hucrede bulundugunu engine.step() biliyor so ileride activation sirasinda
# affected = light_ball_cells(state, target_color=1)
# affected.add(light_ball_position) yapınca hem light ball hem hedef renkteki taslar temizlenmis olacak
def light_ball_cells(
    state: BoardState,
    target_color: int,
) -> set[Position]:
    """return all playable cells containing the selected color."""
    return {
        (row, col)
        for row in range(state.rows)
        for col in range(state.cols)
        if state.cell_mask[row, col] and int(state.tiles[row, col]) == target_color
    }


@dataclass(frozen=True, slots=True)
class TileTransformation:
    position: Position
    special_type: SpecialType


def get_special_effect_cells(
    state: BoardState,
    position: Position,
    special_type: SpecialType,
    rng: np.random.Generator,
    target_color: int | None = None,
    excluded_positions: set[Position] | None = None,
) -> set[Position]:
    """
    calculate the cells affected by a single special.
    this function does not mutate the board or trigger
    other specials.
    """
    if special_type == SpecialType.ROCKET_HORIZONTAL:
        return rocket_cells(
            state,
            position,
            horizontal=True,
        )

    if special_type == SpecialType.ROCKET_VERTICAL:
        return rocket_cells(
            state,
            position,
            horizontal=False,
        )

    if special_type == SpecialType.BOMB:
        return bomb_cells(
            state,
            position,
            radius=1,
        )

    if special_type == SpecialType.PROPELLER:
        return propeller_cells(
            state,
            position,
            rng=rng,
            excluded_positions=excluded_positions,
        )

    if special_type == SpecialType.LIGHT_BALL:
        if target_color is None:
            raise ValueError("light Ball activation requires a target color")

        return light_ball_cells(
            state,
            target_color=target_color,
        ) | {position}

    raise ValueError(f"unsupported special type: {special_type}")


def rocket_rocket_cells(
    state: BoardState,
    position: Position,
) -> set[Position]:
    """
    clear one horizontal row and one vertical column.
    """
    return rocket_cells(state, position, horizontal=True) | rocket_cells(
        state, position, horizontal=False
    )


def rocket_bomb_cells(
    state: BoardState,
    position: Position,
) -> set[Position]:
    """
    clear three horizontal rows and three vertical columns.
    """
    row, col = position
    affected = set()

    for offset in (-1, 0, 1):
        target_row = row + offset
        target_col = col + offset

        if 0 <= target_row < state.rows:
            affected.update(
                rocket_cells(
                    state,
                    position=(target_row, col),
                    horizontal=True,
                )
            )

        if 0 <= target_col < state.cols:
            affected.update(
                rocket_cells(
                    state,
                    position=(row, target_col),
                    horizontal=False,
                )
            )

    return affected


def is_rocket(special_type: SpecialType) -> bool:
    return special_type in (
        SpecialType.ROCKET_HORIZONTAL,
        SpecialType.ROCKET_VERTICAL,
    )


def get_geometric_combination_cells(
    state: BoardState,
    position: Position,
    special_a: SpecialType,
    special_b: SpecialType,
) -> set[Position]:
    """
    calculate the effect of geometric special combinations.
    the order of the two specials does not matter.
    """
    if is_rocket(special_a) and is_rocket(special_b):
        return rocket_rocket_cells(state, position)

    if (is_rocket(special_a) and special_b == SpecialType.BOMB) or (
        special_a == SpecialType.BOMB and is_rocket(special_b)
    ):
        return rocket_bomb_cells(state, position)

    if special_a == SpecialType.BOMB and special_b == SpecialType.BOMB:
        return bomb_cells(state, position, radius=2)

    raise ValueError(
        f"unsupported geometric combination: {special_a.name} + {special_b.name}"
    )


def propeller_combination_cells(
    state: BoardState,
    position: Position,
    special_a: SpecialType,
    special_b: SpecialType,
    rng: np.random.Generator,
    consumed_positions: set[Position] | None = None,
) -> set[Position]:
    """
    calculate cells affected by a Propeller combination.
    supported combinations:
        Propeller + Propeller: 3 random targets.
        Propeller + Rocket: 2 targets, each with a cross effect.
        Propeller + Bomb: 2 targets, each with a 3x3 effect.
    no board mutation or chain reaction occurs here.
    """
    is_propeller_a = special_a == SpecialType.PROPELLER
    is_propeller_b = special_b == SpecialType.PROPELLER

    if not (is_propeller_a or is_propeller_b):
        raise ValueError("propeller combination requires at least one Propeller")

    if is_propeller_a and is_propeller_b:
        combination_type = "propeller_propeller"
        target_count = 3

    elif is_rocket(special_a) or is_rocket(special_b):
        combination_type = "propeller_rocket"
        target_count = 2

    elif special_a == SpecialType.BOMB or special_b == SpecialType.BOMB:
        combination_type = "propeller_bomb"
        target_count = 2

    else:
        raise ValueError(
            f"unsupported Propeller combination: {special_a.name} + {special_b.name}"
        )

    # all Propeller combinations begin with a launch effect.
    affected_cells = propeller_launch_cells(
        state,
        position,
    )

    # both swapped specials must be consumed.
    if consumed_positions is not None:
        affected_cells.update(consumed_positions)

    # track previously selected targets separately.
    # a previous explosion may cover a future target,
    # but we still want distinct target centers.
    selected_targets: set[Position] = set()

    for _ in range(target_count):
        excluded = affected_cells | selected_targets

        target = select_propeller_target(
            state=state,
            rng=rng,
            excluded_positions=excluded,
        )

        if target is None:
            break

        selected_targets.add(target)

        if combination_type == "propeller_propeller":
            affected_cells.add(target)

        elif combination_type == "propeller_rocket":
            affected_cells.update(
                rocket_rocket_cells(
                    state,
                    position=target,
                )
            )

        elif combination_type == "propeller_bomb":
            affected_cells.update(
                bomb_cells(
                    state,
                    position=target,
                    radius=1,
                )
            )

    return affected_cells


def light_ball_combination_transformations(
    state: BoardState,
    special_type: SpecialType,
    target_color: int,
    rng: np.random.Generator,
    consumed_positions: set[Position] | None = None,
) -> tuple[TileTransformation, ...]:
    """
    calculate tile transformations for Light Ball + Special combinations.
    rocket directions are selected independently using the provided RNG.
    this function does not mutate the board.
    """
    if special_type not in (
        SpecialType.ROCKET_HORIZONTAL,
        SpecialType.ROCKET_VERTICAL,
        SpecialType.BOMB,
        SpecialType.PROPELLER,
    ):
        raise ValueError(f"unsupported Light Ball combination: {special_type.name}")

    consumed = consumed_positions if consumed_positions is not None else set()
    # light ball + x oldugunda x ile ayni renge sahip tum taslar seciliyor
    target_positions = sorted(
        light_ball_cells(
            state,
            target_color=target_color,
        )
        - consumed
    )

    transformations = []

    for position in target_positions:
        if is_rocket(special_type):
            direction = int(rng.integers(2))
            # burada rocket yatay mi dikey mi onu seciyoruz
            transformed_type = (
                SpecialType.ROCKET_HORIZONTAL
                if direction == 0
                else SpecialType.ROCKET_VERTICAL
            )

        else:
            transformed_type = special_type

        transformations.append(
            TileTransformation(
                position=position,
                special_type=transformed_type,
            )
        )

    return tuple(transformations)


def light_ball_light_ball_cells(
    state: BoardState,
) -> set[Position]:
    """
    return all occupied playable cells.
    """
    return {
        (row, col)
        for row in range(state.rows)
        for col in range(state.cols)
        if state.cell_mask[row, col] and state.tiles[row, col] != EMPTY
    }
