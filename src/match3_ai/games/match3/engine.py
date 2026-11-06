from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from match3_ai.games.match3.animation_events import (
    AnimationEvent,
    AnimationEventType,
    SpecialActivationInfo,
    SpecialCombinationInfo,
)
from match3_ai.games.match3.blockers import (
    BlockerDamageEvent,
    BlockerDamageRecord,
    DamageSource,
)
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.board_mechanics import (
    apply_blocker_damage,
    apply_board_gravity,
    clear_board_matches,
    place_special_creations,
    refill_board_state,
)
from match3_ai.games.match3.moves import is_valid_swap
from match3_ai.games.match3.pattern_detection import (
    detect_patterns,
    group_patterns,
)
from match3_ai.games.match3.recovery import (
    is_dead_board,
    recover_dead_board,
)
from match3_ai.games.match3.special_resolver import (
    ActivationResult,
    resolve_matched_specials,
    resolve_special_swap,
)
from match3_ai.games.match3.special_selection import (
    select_special_creations,
)
from match3_ai.games.match3.specials import SpecialType

Position = tuple[int, int]
Action = tuple[Position, Position]

DEFAULT_MAX_CASCADES = 200


class CascadeLimitExceeded(RuntimeError):
    """A bounded resolution failed to settle; context contains no board snapshots."""

    def __init__(
        self,
        limit: int,
        cascades_processed: int,
        action: Action,
        remaining_patterns: int,
    ) -> None:
        self.limit = limit
        self.cascades_processed = cascades_processed
        self.context = {
            "max_cascades": limit,
            "cascades_processed": cascades_processed,
            "action": action,
            "remaining_patterns": remaining_patterns,
        }
        super().__init__(
            f"Cascade limit exceeded: limit={limit}, "
            f"cascades_processed={cascades_processed}, action={action}, "
            f"remaining_patterns={remaining_patterns}"
        )


def build_match_damage_events(
    match_groups: tuple[frozenset[Position], ...],
    rows: int,
    cols: int,
) -> tuple[BlockerDamageEvent, ...]:
    events: list[BlockerDamageEvent] = []

    for group_cells in match_groups:
        events.append(
            BlockerDamageEvent(
                sources=frozenset({DamageSource.DIRECT_MATCH}),
                cells=group_cells,
            )
        )
        adjacent_cells = {
            neighbor
            for row, col in group_cells
            for neighbor in (
                (row - 1, col),
                (row + 1, col),
                (row, col - 1),
                (row, col + 1),
            )
            if 0 <= neighbor[0] < rows and 0 <= neighbor[1] < cols
        } - set(group_cells)

        if adjacent_cells:
            events.append(
                BlockerDamageEvent(
                    sources=frozenset({DamageSource.ADJACENT_MATCH}),
                    cells=frozenset(adjacent_cells),
                )
            )

    return tuple(events)


def count_occupied_cells(
    state: BoardState,
    positions: set[Position],
) -> int:
    return sum(
        1
        for position in positions
        if state.cell_mask[position] and state.tiles[position] != 0
    )


def count_cleared_by_color(
    state: BoardState,
    positions: set[Position],
    num_colors: int,
) -> tuple[int, ...]:
    counts = [0] * num_colors
    for position in positions:
        if not state.cell_mask[position]:
            continue
        tile = int(state.tiles[position])
        if 1 <= tile <= num_colors:
            counts[tile - 1] += 1
    return tuple(counts)


@dataclass(frozen=True, slots=True)
class StepResult:
    """Move result; cleared_by_color indexes color 1 at 0 and excludes colorless tiles."""

    state: BoardState
    valid: bool
    cleared_tiles: int
    cascade_count: int
    reshuffled: bool = False
    activation_result: ActivationResult | None = None
    activation_history: tuple[ActivationResult, ...] = ()
    animation_events: tuple[AnimationEvent, ...] = ()
    blocker_damage: tuple[BlockerDamageRecord, ...] = ()
    cleared_by_color: tuple[int, ...] = ()


class Match3Engine:
    def __init__(
        self,
        num_colors: int,
        seed: int | None = None,
        max_cascades: int = DEFAULT_MAX_CASCADES,
    ) -> None:
        if type(num_colors) is not int or not 3 <= num_colors <= 254:
            raise ValueError("num_colors must be an integer between 3 and 254")

        if type(max_cascades) is not int or max_cascades <= 0:
            raise ValueError("max_cascades must be a positive integer")

        self.num_colors = num_colors
        self.max_cascades = max_cascades
        self.rng = np.random.default_rng(seed)

    def step(
        self,
        state: BoardState,
        action: Action,
        record_events: bool = False,
    ) -> StepResult:
        source, destination = action

        animation_events: list[AnimationEvent] = []

        # --------------------------------------------------
        # Animation event recording
        # --------------------------------------------------

        def record_event(
            event_type: AnimationEventType,
            current_state: BoardState,
            affected_cells: set[Position] | None = None,
            special_activations: tuple[SpecialActivationInfo, ...] = (),
            special_combination: SpecialCombinationInfo | None = None,
            blocker_damage: tuple[BlockerDamageRecord, ...] = (),
        ) -> None:
            if not record_events:
                return

            animation_events.append(
                AnimationEvent(
                    event_type=event_type,
                    state=current_state.copy(),
                    affected_cells=frozenset(
                        affected_cells if affected_cells is not None else set()
                    ),
                    special_activations=special_activations,
                    special_combination=special_combination,
                    blocker_damage=blocker_damage,
                )
            )

        # --------------------------------------------------
        # Build special activation metadata
        # --------------------------------------------------

        def build_special_activations(
            result: ActivationResult,
        ) -> tuple[SpecialActivationInfo, ...]:
            if not record_events:
                return ()

            return tuple(
                SpecialActivationInfo(
                    position=step.position,
                    special_type=step.special_type,
                    effect_cells=step.effect_cells,
                )
                for step in result.activation_steps
            )

        def order_damage_records(
            records: tuple[BlockerDamageRecord, ...],
        ) -> tuple[BlockerDamageRecord, ...]:
            start = len(blocker_damage_history)
            return tuple(
                replace(record, hit_order=start + index)
                for index, record in enumerate(records)
            )

        # --------------------------------------------------
        # Validate swap
        # --------------------------------------------------

        if not is_valid_swap(state, source, destination):
            return StepResult(
                state=state.copy(),
                valid=False,
                cleared_tiles=0,
                cascade_count=0,
            )

        is_special_swap = (
            state.specials[source] != SpecialType.NONE
            or state.specials[destination] != SpecialType.NONE
        )

        # --------------------------------------------------
        # Swap tiles and specials
        # --------------------------------------------------

        tiles = state.tiles.copy()
        specials = state.specials.copy()

        tiles[source], tiles[destination] = (
            tiles[destination],
            tiles[source],
        )

        specials[source], specials[destination] = (
            specials[destination],
            specials[source],
        )

        current_state = BoardState(
            tiles=tiles,
            specials=specials,
            cell_mask=state.cell_mask,
            blockers=state.blockers,
            blocker_hp=state.blocker_hp,
        )

        record_event(
            AnimationEventType.SWAP,
            current_state,
            {source, destination},
        )

        total_cleared = 0
        total_cleared_by_color = [0] * self.num_colors
        blocker_damage_history: list[BlockerDamageRecord] = []
        cascade_count = 0
        is_initial_match = True

        activation_result: ActivationResult | None = None
        activation_history: list[ActivationResult] = []

        # --------------------------------------------------
        # Resolve special swaps
        # --------------------------------------------------

        if is_special_swap:
            activation_result = resolve_special_swap(
                state=current_state,
                source=source,
                destination=destination,
                rng=self.rng,
            )

            activation_history.append(activation_result)

            affected_cells = set(activation_result.affected_cells)

            # --------------------------------------------------
            # Prepare animation state
            # --------------------------------------------------

            # Keep the actual game state unchanged.
            animation_state = current_state

            # Show Light Ball transformations before activation.
            if record_events and activation_result.transformations:
                transformed_state = current_state.copy()
                transformed_positions: set[Position] = set()

                for transformation in activation_result.transformations:
                    transformed_state.specials[transformation.position] = (
                        transformation.special_type
                    )
                    transformed_positions.add(transformation.position)

                record_event(
                    AnimationEventType.SPECIAL_CREATION,
                    transformed_state,
                    transformed_positions,
                )

                # Use transformed board for the activation animation.
                animation_state = transformed_state

            # --------------------------------------------------
            # Detect special combinations
            # --------------------------------------------------

            special_combination = None

            if record_events:
                source_type = SpecialType(
                    int(current_state.specials[source])
                )
                destination_type = SpecialType(
                    int(current_state.specials[destination])
                )

                if (
                    source_type != SpecialType.NONE
                    and destination_type != SpecialType.NONE
                ):
                    special_combination = SpecialCombinationInfo(
                        source=source,
                        destination=destination,
                        source_type=source_type,
                        destination_type=destination_type,
                        effect_cells=activation_result.effect_cells,
                    )

            # --------------------------------------------------
            # Record special activation
            # --------------------------------------------------

            record_event(
                AnimationEventType.SPECIAL_ACTIVATION,
                animation_state,
                affected_cells,
                special_activations=build_special_activations(
                    activation_result,
                ),
                special_combination=special_combination,
            )

            current_state, damage_records = apply_blocker_damage(
                current_state,
                activation_result.damage_events,
            )
            damage_records = order_damage_records(damage_records)
            blocker_damage_history.extend(damage_records)
            if damage_records:
                record_event(
                    AnimationEventType.BLOCKER_DAMAGE,
                    current_state,
                    {record.position for record in damage_records},
                    blocker_damage=damage_records,
                )

            # --------------------------------------------------
            # Clear affected cells
            # --------------------------------------------------

            cleared_count = count_occupied_cells(
                current_state,
                affected_cells,
            )
            color_counts = count_cleared_by_color(
                current_state,
                affected_cells,
                self.num_colors,
            )
            for index, count in enumerate(color_counts):
                total_cleared_by_color[index] += count

            current_state = clear_board_matches(
                current_state,
                affected_cells,
            )

            record_event(
                AnimationEventType.CLEAR,
                current_state,
                affected_cells,
            )

            total_cleared += cleared_count

            # --------------------------------------------------
            # Gravity
            # --------------------------------------------------

            current_state = apply_board_gravity(current_state)

            record_event(
                AnimationEventType.GRAVITY,
                current_state,
            )

            # --------------------------------------------------
            # Refill
            # --------------------------------------------------

            current_state = refill_board_state(
                current_state,
                num_colors=self.num_colors,
                rng=self.rng,
            )

            record_event(
                AnimationEventType.REFILL,
                current_state,
            )

            is_initial_match = False

        # --------------------------------------------------
        # Detect normal matches and cascades
        # --------------------------------------------------

        patterns = detect_patterns(current_state.tiles)

        while patterns:
            if cascade_count >= self.max_cascades:
                raise CascadeLimitExceeded(
                    self.max_cascades, cascade_count, action, len(patterns)
                )

            groups = group_patterns(
                current_state.tiles,
                patterns,
            )

            matched_cells: set[Position] = set()

            for group in groups:
                matched_cells.update(group.cells)

            # Exclude groups containing existing specials.
            creation_groups = [
                group
                for group in groups
                if not any(
                    current_state.specials[position] != SpecialType.NONE
                    for position in group.cells
                )
            ]

            creations = select_special_creations(
                creation_groups,
                destination=destination if is_initial_match else None,
            )

            original_tiles = current_state.tiles.copy()

            # --------------------------------------------------
            # Resolve matched specials
            # --------------------------------------------------

            match_activation = resolve_matched_specials(
                state=current_state,
                matched_cells=matched_cells,
                rng=self.rng,
            )

            match_damage_events = build_match_damage_events(
                tuple(group.cells for group in groups),
                current_state.rows,
                current_state.cols,
            )

            if match_activation.activated_positions:
                activation_history.append(match_activation)

            affected_cells = set(match_activation.affected_cells)

            record_event(
                AnimationEventType.MATCH,
                current_state,
                matched_cells,
            )

            # Cancel creations affected by special effects.
            creations = [
                creation
                for creation in creations
                if creation.position not in match_activation.effect_cells
            ]

            if match_activation.activated_positions:
                record_event(
                    AnimationEventType.SPECIAL_ACTIVATION,
                    current_state,
                    affected_cells,
                    special_activations=build_special_activations(
                        match_activation,
                    ),
                )

            current_state, match_damage_records = apply_blocker_damage(
                current_state,
                match_damage_events + match_activation.damage_events,
            )
            match_damage_records = order_damage_records(match_damage_records)
            blocker_damage_history.extend(match_damage_records)

            if match_damage_records:
                record_event(
                    AnimationEventType.BLOCKER_DAMAGE,
                    current_state,
                    {record.position for record in match_damage_records},
                    blocker_damage=match_damage_records,
                )

            # --------------------------------------------------
            # Clear matched cells
            # --------------------------------------------------

            cleared_count = count_occupied_cells(
                current_state,
                affected_cells,
            )

            preserved_creation_positions = {
                creation.position
                for creation in creations
                if current_state.cell_mask[creation.position]
                and current_state.tiles[creation.position] != 0
            }
            preserved_creations = len(preserved_creation_positions)
            color_counts = count_cleared_by_color(
                current_state,
                affected_cells - preserved_creation_positions,
                self.num_colors,
            )
            for index, count in enumerate(color_counts):
                total_cleared_by_color[index] += count

            current_state = clear_board_matches(
                current_state,
                affected_cells,
            )

            record_event(
                AnimationEventType.CLEAR,
                current_state,
                affected_cells,
            )

            # --------------------------------------------------
            # Create new specials
            # --------------------------------------------------

            if creations:
                current_state = place_special_creations(
                    current_state,
                    creations,
                    original_tiles,
                )

                record_event(
                    AnimationEventType.SPECIAL_CREATION,
                    current_state,
                    {creation.position for creation in creations},
                )

            total_cleared += cleared_count - preserved_creations
            cascade_count += 1

            # --------------------------------------------------
            # Gravity
            # --------------------------------------------------

            current_state = apply_board_gravity(current_state)

            record_event(
                AnimationEventType.GRAVITY,
                current_state,
            )

            # --------------------------------------------------
            # Refill
            # --------------------------------------------------

            current_state = refill_board_state(
                current_state,
                num_colors=self.num_colors,
                rng=self.rng,
            )

            record_event(
                AnimationEventType.REFILL,
                current_state,
            )

            # --------------------------------------------------
            # Check for cascades
            # --------------------------------------------------

            patterns = detect_patterns(current_state.tiles)
            is_initial_match = False

        # --------------------------------------------------
        # Recover dead boards
        # --------------------------------------------------

        reshuffled = False

        if is_dead_board(current_state):
            current_state = recover_dead_board(
                state=current_state,
                rng=self.rng,
            )

            reshuffled = True

            record_event(
                AnimationEventType.RESHUFFLE,
                current_state,
            )

        # --------------------------------------------------
        # Return result
        # --------------------------------------------------

        return StepResult(
            state=current_state,
            valid=True,
            cleared_tiles=total_cleared,
            cascade_count=cascade_count,
            reshuffled=reshuffled,
            activation_result=activation_result,
            activation_history=tuple(activation_history),
            animation_events=tuple(animation_events),
            blocker_damage=tuple(blocker_damage_history),
            cleared_by_color=tuple(total_cleared_by_color),
        )
