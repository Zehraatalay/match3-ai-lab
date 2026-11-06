from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from match3_ai.games.match3.blockers import (
    BlockerDamageEvent,
    DamageSource,
)
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.special_activation import (
    TileTransformation,
    get_geometric_combination_passes,
    get_special_effect_passes,
    light_ball_combination_transformations,
    light_ball_light_ball_cells,
    propeller_combination_effect_passes,
)
from match3_ai.games.match3.specials import SpecialType

Position = tuple[int, int]

@dataclass(frozen=True, slots=True)
class SpecialActivationStep:
    position: Position
    special_type: SpecialType
    effect_cells: frozenset[Position]

@dataclass(frozen=True, slots=True)
class ActivationResult:
    affected_cells: frozenset[Position]
    activated_positions: frozenset[Position]
    transformations: tuple[TileTransformation, ...] = ()
    effect_cells: frozenset[Position] = frozenset()
    activation_order: tuple[Position, ...] = ()
    activation_steps: tuple[SpecialActivationStep, ...] = ()
    damage_events: tuple[BlockerDamageEvent, ...] = ()


def _damage_source_for_special(special_type: SpecialType) -> DamageSource:
    if special_type in (
        SpecialType.ROCKET_HORIZONTAL,
        SpecialType.ROCKET_VERTICAL,
    ):
        return DamageSource.ROCKET

    if special_type == SpecialType.BOMB:
        return DamageSource.BOMB

    if special_type == SpecialType.PROPELLER:
        return DamageSource.PROPELLER

    if special_type == SpecialType.LIGHT_BALL:
        return DamageSource.LIGHT_BALL

    raise ValueError(f"unsupported special damage source: {special_type.name}")


def _damage_events_for_passes(
    special_types: set[SpecialType],
    effect_passes: tuple[frozenset[Position], ...],
) -> tuple[BlockerDamageEvent, ...]:
    sources = frozenset(
        _damage_source_for_special(special_type)
        for special_type in special_types
    )
    return tuple(
        BlockerDamageEvent(sources=sources, cells=effect_cells)
        for effect_cells in effect_passes
    )

def resolve_special_activation(
    state: BoardState,
    position: Position,
    rng: np.random.Generator,
    target_color: int | None = None,
) -> ActivationResult:
    """
    resolve one directly activated special and its chain reactions.
    """
    row, col = position

    if not (0 <= row < state.rows and 0 <= col < state.cols):
        raise ValueError("activation position is outside the board")

    initial_special = SpecialType(int(state.specials[row, col]))

    if initial_special == SpecialType.NONE:
        raise ValueError("the selected position does not contain a special")

    if initial_special == SpecialType.LIGHT_BALL and target_color is None:
        raise ValueError("light Ball activation requires a target color")

    return _resolve_activation_queue(
        state=state,
        activation_queue=deque([position]),
        activated_positions={position},
        affected_cells=set(),
        rng=rng,
        initial_position=position,
        target_color=target_color,
    )


def select_chain_light_ball_color(
    state: BoardState,
    rng: np.random.Generator,
    affected_cells: set[Position],
) -> int | None:
    """
    select a random color from normal tiles not yet affected.
    returns None when no eligible tiles remain.
    """
    available_colors = sorted(
        {
            int(state.tiles[row, col])
            for row in range(state.rows)
            for col in range(state.cols)
            if state.cell_mask[row, col]
            and (row, col) not in affected_cells
            and 1 <= int(state.tiles[row, col]) <= 254
        }
    )

    if not available_colors:
        return None

    return int(rng.choice(available_colors))


def _resolve_activation_queue(
    state: BoardState,
    activation_queue: deque[Position],
    activated_positions: set[Position],
    affected_cells: set[Position],
    rng: np.random.Generator,
    initial_position: Position | None = None,
    target_color: int | None = None,
    initial_effect_cells: set[Position] | None = None,
    initial_damage_events: tuple[BlockerDamageEvent, ...] = (),
) -> ActivationResult:
    """
    resolve queued special activations and their chain reactions.
    initial_position identifies a directly activated special.
    when None, all queued specials are chain-triggered.
    """
    all_effect_cells = (
        set(initial_effect_cells) if initial_effect_cells is not None else set()
    )
    activation_steps: list[SpecialActivationStep] = []
    activation_order: list[Position] = []
    damage_events = list(initial_damage_events)
    while activation_queue:
        current_position = activation_queue.popleft()
        activation_order.append(current_position)
        current_row, current_col = current_position

        current_special = SpecialType(int(state.specials[current_row, current_col]))

        current_target_color = None

        if current_special == SpecialType.LIGHT_BALL:
            if current_position == initial_position:
                current_target_color = target_color

            else:
                current_target_color = select_chain_light_ball_color(
                    state=state,
                    rng=rng,
                    affected_cells=affected_cells,
                )

                if current_target_color is None:
                    affected_cells.add(current_position)
                    continue

        effect_passes = get_special_effect_passes(
            state=state,
            position=current_position,
            special_type=current_special,
            rng=rng,
            target_color=current_target_color,
            excluded_positions=affected_cells,
        )
        effect_cells = set().union(*effect_passes)
        activation_steps.append(
            SpecialActivationStep(
                position=current_position,
                special_type=current_special,
                effect_cells=frozenset(effect_cells),
            )
        )
        damage_events.extend(
            _damage_events_for_passes(
                {current_special},
                effect_passes,
            )
        )

        all_effect_cells.update(effect_cells)
        affected_cells.update(effect_cells)
        for affected_position in sorted(effect_cells):
            affected_row, affected_col = affected_position

            special = SpecialType(int(state.specials[affected_row, affected_col]))

            if special == SpecialType.NONE:
                continue

            if affected_position in activated_positions:
                continue

            activated_positions.add(affected_position)
            activation_queue.append(affected_position)

    return ActivationResult(
        affected_cells=frozenset(affected_cells),
        activated_positions=frozenset(activated_positions),
        effect_cells=frozenset(all_effect_cells),
        activation_order=tuple(activation_order),
        activation_steps=tuple(activation_steps),
        damage_events=tuple(damage_events),
    )


def _resolve_combination_effect(
    state: BoardState,
    affected_cells: set[Position],
    consumed_positions: set[Position],
    rng: np.random.Generator,
    initial_damage_events: tuple[BlockerDamageEvent, ...],
) -> ActivationResult:
    """
    resolve chain reactions caused by a combination effect.
    consumed combination partners do not activate individually.
    """
    affected_cells = set(affected_cells)
    affected_cells.update(consumed_positions)

    activated_positions = set(consumed_positions)
    activation_queue: deque[Position] = deque()

    for position in sorted(affected_cells):
        if position in consumed_positions:
            continue

        row, col = position

        special = SpecialType(int(state.specials[row, col]))

        if special == SpecialType.NONE:
            continue

        activated_positions.add(position)
        activation_queue.append(position)

    return _resolve_activation_queue(
        state=state,
        activation_queue=activation_queue,
        activated_positions=activated_positions,
        affected_cells=affected_cells,
        rng=rng,
        initial_effect_cells=set(affected_cells),
        initial_damage_events=initial_damage_events,
    )


def resolve_geometric_combination(
    state: BoardState,
    source: Position,
    destination: Position,
    rng: np.random.Generator,
) -> ActivationResult:
    """
    resolve Rocket+Rocket, Rocket+Bomb, or Bomb+Bomb.
    the board must already contain the swapped tiles.
    """
    source_row, source_col = source
    destination_row, destination_col = destination

    special_a = SpecialType(int(state.specials[source_row, source_col]))

    special_b = SpecialType(int(state.specials[destination_row, destination_col]))

    consumed_positions = {source, destination}

    # calculate the combined effect at the swap destination.
    effect_passes = get_geometric_combination_passes(
        state=state,
        position=destination,
        special_a=special_a,
        special_b=special_b,
    )
    affected_cells = set().union(*effect_passes)
    initial_damage_events = _damage_events_for_passes(
        {special_a, special_b},
        effect_passes,
    )

    return _resolve_combination_effect(
        state=state,
        affected_cells=affected_cells,
        consumed_positions=consumed_positions,
        rng=rng,
        initial_damage_events=initial_damage_events,
    )


def resolve_propeller_combination(
    state: BoardState,
    source: Position,
    destination: Position,
    rng: np.random.Generator,
) -> ActivationResult:
    """
    Resolve Propeller+Propeller, Propeller+Rocket,
    or Propeller+Bomb combinations and chain reactions.

    The board must already contain the swapped tiles.
    """
    source_row, source_col = source
    destination_row, destination_col = destination

    special_a = SpecialType(int(state.specials[source_row, source_col]))

    special_b = SpecialType(int(state.specials[destination_row, destination_col]))

    consumed_positions = {source, destination}

    # Calculate the combined Propeller effect.
    effect_passes = propeller_combination_effect_passes(
        state=state,
        position=destination,
        special_a=special_a,
        special_b=special_b,
        rng=rng,
        consumed_positions=consumed_positions,
    )
    affected_cells = set().union(*effect_passes)
    initial_damage_events = _damage_events_for_passes(
        {special_a, special_b},
        effect_passes,
    )

    return _resolve_combination_effect(
        state=state,
        affected_cells=affected_cells,
        consumed_positions=consumed_positions,
        rng=rng,
        initial_damage_events=initial_damage_events,
    )


def resolve_light_ball_special_combination(
    state: BoardState,
    source: Position,
    destination: Position,
    rng: np.random.Generator,
) -> ActivationResult:
    """
    resolve Light Ball + Rocket/Bomb/Propeller combinations.
    the board must already contain the swapped tiles.
    the original board is not mutated.
    """
    source_row, source_col = source
    destination_row, destination_col = destination

    special_a = SpecialType(int(state.specials[source_row, source_col]))

    special_b = SpecialType(int(state.specials[destination_row, destination_col]))

    if special_a == SpecialType.LIGHT_BALL:
        partner_position = destination
        partner_special = special_b

    elif special_b == SpecialType.LIGHT_BALL:
        partner_position = source
        partner_special = special_a

    else:
        raise ValueError("light Ball combination requires a Light Ball")

    partner_row, partner_col = partner_position

    target_color = int(state.tiles[partner_row, partner_col])

    consumed_positions = {source, destination}

    transformations = light_ball_combination_transformations(
        state=state,
        special_type=partner_special,
        target_color=target_color,
        rng=rng,
        consumed_positions=consumed_positions,
    )

    # work on a copy so the original board remains unchanged.
    working_state = state.copy()

    # apply transformations to the temporary board.
    for transformation in transformations:
        row, col = transformation.position

        working_state.specials[row, col] = transformation.special_type

    # both swapped specials are consumed.
    affected_cells = set(consumed_positions)
    initial_damage_events = _damage_events_for_passes(
        {SpecialType.LIGHT_BALL, partner_special},
        (frozenset(consumed_positions),),
    )

    # prevent the combination partners from activating separately.
    activated_positions = set(consumed_positions)

    # all transformed specials must activate.
    activation_queue = deque()

    for transformation in transformations:
        position = transformation.position

        activated_positions.add(position)
        activation_queue.append(position)

    result = _resolve_activation_queue(
        state=working_state,
        activation_queue=activation_queue,
        activated_positions=activated_positions,
        affected_cells=affected_cells,
        rng=rng,
        initial_effect_cells=set(affected_cells),
        initial_damage_events=initial_damage_events,
    )

    return ActivationResult(
        affected_cells=result.affected_cells,
        activated_positions=result.activated_positions,
        transformations=transformations,
        effect_cells=result.effect_cells,
        activation_order=result.activation_order,
        activation_steps=result.activation_steps,
        damage_events=result.damage_events,
    )


def resolve_light_ball_light_ball_combination(
    state: BoardState,
    source: Position,
    destination: Position,
) -> ActivationResult:
    """
    clear all occupied playable cells without chain reactions.
    """
    source_special = SpecialType(int(state.specials[source]))
    destination_special = SpecialType(int(state.specials[destination]))

    if (
        source_special != SpecialType.LIGHT_BALL
        or destination_special != SpecialType.LIGHT_BALL
    ):
        raise ValueError("Combination requires two Light Balls")

    affected_cells = light_ball_light_ball_cells(state)
    affected_cells.update({source, destination})

    return ActivationResult(
        affected_cells=frozenset(affected_cells),
        activated_positions=frozenset({source, destination}),
        effect_cells=frozenset(affected_cells),
        damage_events=(
            BlockerDamageEvent(
                sources=frozenset({DamageSource.LIGHT_BALL}),
                cells=frozenset(
                    (row, col)
                    for row in range(state.rows)
                    for col in range(state.cols)
                    if state.cell_mask[row, col]
                ),
            ),
        ),
    )


def resolve_special_swap(
    state: BoardState,
    source: Position,
    destination: Position,
    rng: np.random.Generator,
) -> ActivationResult:
    """
    resolve a swap involving at least one special tile.
    the board must already contain the swapped tiles.
    """
    source_special = SpecialType(int(state.specials[source]))
    destination_special = SpecialType(int(state.specials[destination]))

    source_is_special = source_special != SpecialType.NONE
    destination_is_special = destination_special != SpecialType.NONE

    if not source_is_special and not destination_is_special:
        raise ValueError("Swap does not contain a special tile")

    # special + normal tile
    if source_is_special != destination_is_special:
        special_position = source if source_is_special else destination

        special_type = source_special if source_is_special else destination_special

        target_color = None

        if special_type == SpecialType.LIGHT_BALL:
            normal_position = destination if source_is_special else source
            target_color = int(state.tiles[normal_position])

        return resolve_special_activation(
            state=state,
            position=special_position,
            rng=rng,
            target_color=target_color,
        )

    # light Ball + Light Ball
    if (
        source_special == SpecialType.LIGHT_BALL
        and destination_special == SpecialType.LIGHT_BALL
    ):
        return resolve_light_ball_light_ball_combination(
            state=state,
            source=source,
            destination=destination,
        )

    # light Ball + another special
    if (
        source_special == SpecialType.LIGHT_BALL
        or destination_special == SpecialType.LIGHT_BALL
    ):
        return resolve_light_ball_special_combination(
            state=state,
            source=source,
            destination=destination,
            rng=rng,
        )

    # propeller combinations
    if (
        source_special == SpecialType.PROPELLER
        or destination_special == SpecialType.PROPELLER
    ):
        return resolve_propeller_combination(
            state=state,
            source=source,
            destination=destination,
            rng=rng,
        )

    # rocket/Bomb combinations
    return resolve_geometric_combination(
        state=state,
        source=source,
        destination=destination,
        rng=rng,
    )


def resolve_matched_specials(
    state: BoardState,
    matched_cells: set[Position],
    rng: np.random.Generator,
) -> ActivationResult:
    """
    resolve specials included in a normal match.
    all matched specials activate once.
    chain-triggered specials are resolved through the same queue.
    """
    special_positions = sorted(
        position
        for position in matched_cells
        if SpecialType(int(state.specials[position])) != SpecialType.NONE
    )

    if not special_positions:
        return ActivationResult(
            affected_cells=frozenset(matched_cells),
            activated_positions=frozenset(),
        )

    return _resolve_activation_queue(
        state=state,
        activation_queue=deque(special_positions),
        activated_positions=set(special_positions),
        affected_cells=set(matched_cells),
        rng=rng,
    )
