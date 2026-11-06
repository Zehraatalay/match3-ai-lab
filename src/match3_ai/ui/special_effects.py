from __future__ import annotations

import math

import pygame

from match3_ai.games.match3.animation_events import (
    SpecialActivationInfo,
    SpecialCombinationInfo,
)
from match3_ai.games.match3.specials import SpecialType

Position = tuple[int, int]


def cell_center(
    position: Position,
    cell_size: int,
    padding: int,
    header_height: int,
) -> tuple[int, int]:
    row, col = position

    return (
        padding + col * cell_size + cell_size // 2,
        header_height + row * cell_size + cell_size // 2,
    )


def draw_activation_effect(
    surface: pygame.Surface,
    affected_cells: frozenset[Position],
    progress: float,
    cell_size: int,
    padding: int,
    header_height: int,
) -> None:
    """Draw a fading highlight over affected cells."""

    progress = max(0.0, min(1.0, progress))

    overlay = pygame.Surface(
        surface.get_size(),
        pygame.SRCALPHA,
    )

    alpha = int(120 * (1.0 - progress))

    pulse = math.sin(progress * math.pi)
    border_width = max(2, int(3 + 4 * pulse))

    for row, col in affected_cells:
        x = padding + col * cell_size
        y = header_height + row * cell_size

        rect = pygame.Rect(
            x + 4,
            y + 4,
            cell_size - 8,
            cell_size - 8,
        )

        pygame.draw.rect(
            overlay,
            (255, 225, 100, alpha),
            rect,
            border_radius=10,
        )

        pygame.draw.rect(
            overlay,
            (255, 255, 255, min(255, alpha + 70)),
            rect,
            width=border_width,
            border_radius=10,
        )

    surface.blit(overlay, (0, 0))


def draw_rocket_effect(
    overlay: pygame.Surface,
    position: Position,
    horizontal: bool,
    progress: float,
    cell_size: int,
    padding: int,
    header_height: int,
    rows: int,
    cols: int,
) -> None:
    """Draw an expanding rocket beam."""

    center_x, center_y = cell_center(
        position,
        cell_size,
        padding,
        header_height,
    )

    alpha = int(230 * (1.0 - progress))
    thickness = max(2, int(12 * (1.0 - progress) + 3))

    if horizontal:
        left = padding
        right = padding + cols * cell_size

        start_x = int(center_x + (left - center_x) * progress)
        end_x = int(center_x + (right - center_x) * progress)

        pygame.draw.line(
            overlay,
            (255, 245, 150, alpha),
            (start_x, center_y),
            (end_x, center_y),
            thickness,
        )

    else:
        top = header_height
        bottom = header_height + rows * cell_size

        start_y = int(center_y + (top - center_y) * progress)
        end_y = int(center_y + (bottom - center_y) * progress)

        pygame.draw.line(
            overlay,
            (255, 245, 150, alpha),
            (center_x, start_y),
            (center_x, end_y),
            thickness,
        )


def draw_bomb_effect(
    overlay: pygame.Surface,
    position: Position,
    progress: float,
    cell_size: int,
    padding: int,
    header_height: int,
) -> None:
    """Draw an expanding explosion ring."""

    center = cell_center(
        position,
        cell_size,
        padding,
        header_height,
    )

    max_radius = cell_size * 1.8

    radius = max(
        1,
        int(max_radius * progress),
    )

    alpha = int(240 * (1.0 - progress))

    pygame.draw.circle(
        overlay,
        (255, 150, 50, alpha),
        center,
        radius,
        width=8,
    )

    inner_radius = max(1, int(radius * 0.6))

    pygame.draw.circle(
        overlay,
        (255, 230, 100, alpha),
        center,
        inner_radius,
        width=4,
    )


def draw_propeller_effect(
    overlay: pygame.Surface,
    position: Position,
    progress: float,
    cell_size: int,
    padding: int,
    header_height: int,
) -> None:
    """Draw a rotating propeller effect."""

    center_x, center_y = cell_center(
        position,
        cell_size,
        padding,
        header_height,
    )

    alpha = int(230 * (1.0 - progress))

    angle = progress * math.pi * 6

    blade_length = int(cell_size * 0.45)

    for index in range(4):
        blade_angle = angle + index * math.pi / 2

        end_x = int(center_x + math.cos(blade_angle) * blade_length)
        end_y = int(center_y + math.sin(blade_angle) * blade_length)

        pygame.draw.line(
            overlay,
            (170, 240, 255, alpha),
            (center_x, center_y),
            (end_x, end_y),
            width=7,
        )

    radius = max(
        1,
        int(cell_size * 0.7 * progress),
    )

    pygame.draw.circle(
        overlay,
        (200, 250, 255, alpha),
        (center_x, center_y),
        radius,
        width=3,
    )


def draw_light_ball_effect(
    overlay: pygame.Surface,
    position: Position,
    progress: float,
    cell_size: int,
    padding: int,
    header_height: int,
) -> None:
    """Draw expanding rainbow energy rings."""

    center = cell_center(
        position,
        cell_size,
        padding,
        header_height,
    )

    colors = [
        (255, 100, 100),
        (255, 210, 80),
        (100, 220, 140),
        (100, 180, 255),
        (190, 130, 255),
    ]

    for index, color in enumerate(colors):
        ring_progress = (progress + index * 0.15) % 1.0

        radius = max(
            1,
            int(cell_size * 2.5 * ring_progress),
        )

        alpha = int(220 * (1.0 - ring_progress))

        pygame.draw.circle(
            overlay,
            (*color, alpha),
            center,
            radius,
            width=4,
        )


def draw_special_effects(
    surface: pygame.Surface,
    activations: tuple[SpecialActivationInfo, ...],
    progress: float,
    cell_size: int,
    padding: int,
    header_height: int,
    rows: int,
    cols: int,
) -> None:
    """Render type-specific effects for activated specials."""

    progress = max(0.0, min(1.0, progress))

    overlay = pygame.Surface(
        surface.get_size(),
        pygame.SRCALPHA,
    )

    for activation in activations:
        special_type = activation.special_type
        position = activation.position

        if special_type == SpecialType.ROCKET_HORIZONTAL:
            draw_rocket_effect(
                overlay,
                position,
                horizontal=True,
                progress=progress,
                cell_size=cell_size,
                padding=padding,
                header_height=header_height,
                rows=rows,
                cols=cols,
            )

        elif special_type == SpecialType.ROCKET_VERTICAL:
            draw_rocket_effect(
                overlay,
                position,
                horizontal=False,
                progress=progress,
                cell_size=cell_size,
                padding=padding,
                header_height=header_height,
                rows=rows,
                cols=cols,
            )

        elif special_type == SpecialType.BOMB:
            draw_bomb_effect(
                overlay,
                position,
                progress,
                cell_size,
                padding,
                header_height,
            )

        elif special_type == SpecialType.PROPELLER:
            draw_propeller_effect(
                overlay,
                position,
                progress,
                cell_size,
                padding,
                header_height,
            )

        elif special_type == SpecialType.LIGHT_BALL:
            draw_light_ball_effect(
                overlay,
                position,
                progress,
                cell_size,
                padding,
                header_height,
            )

    surface.blit(overlay, (0, 0))


def get_active_special(
    activations: tuple[SpecialActivationInfo, ...],
    progress: float,
) -> tuple[SpecialActivationInfo | None, float]:
    """
    Return the currently active special and its local progress.

    Global progress is divided equally among activations.
    """

    if not activations:
        return None, 0.0

    progress = max(0.0, min(1.0, progress))

    count = len(activations)

    scaled_progress = progress * count

    index = min(
        int(scaled_progress),
        count - 1,
    )

    local_progress = min(
        1.0,
        scaled_progress - index,
    )

    return activations[index], local_progress


def draw_combination_effect(
    surface: pygame.Surface,
    combination: SpecialCombinationInfo,
    progress: float,
    cell_size: int,
    padding: int,
    header_height: int,
    rows: int,
    cols: int,
) -> None:
    """Render visual effects for special combinations."""

    progress = max(0.0, min(1.0, progress))

    special_types = {
        combination.source_type,
        combination.destination_type,
    }

    rocket_types = {
        SpecialType.ROCKET_HORIZONTAL,
        SpecialType.ROCKET_VERTICAL,
    }

    overlay = pygame.Surface(
        surface.get_size(),
        pygame.SRCALPHA,
    )

    # Use the destination as the combination's visual origin.
    position = combination.destination

    # --------------------------------------------------
    # Rocket + Rocket
    # --------------------------------------------------

    if (
        combination.source_type in rocket_types
        and combination.destination_type in rocket_types
    ):
        draw_rocket_effect(
            overlay,
            position,
            horizontal=True,
            progress=progress,
            cell_size=cell_size,
            padding=padding,
            header_height=header_height,
            rows=rows,
            cols=cols,
        )

        draw_rocket_effect(
            overlay,
            position,
            horizontal=False,
            progress=progress,
            cell_size=cell_size,
            padding=padding,
            header_height=header_height,
            rows=rows,
            cols=cols,
        )

    # --------------------------------------------------
    # Rocket + Bomb
    # --------------------------------------------------

    elif (
        SpecialType.BOMB in special_types
        and bool(special_types & rocket_types)
    ):
        center_row, center_col = position

        # Three horizontal beams.
        for row in range(
            max(0, center_row - 1),
            min(rows, center_row + 2),
        ):
            draw_rocket_effect(
                overlay,
                (row, center_col),
                horizontal=True,
                progress=progress,
                cell_size=cell_size,
                padding=padding,
                header_height=header_height,
                rows=rows,
                cols=cols,
            )

        # Three vertical beams.
        for col in range(
            max(0, center_col - 1),
            min(cols, center_col + 2),
        ):
            draw_rocket_effect(
                overlay,
                (center_row, col),
                horizontal=False,
                progress=progress,
                cell_size=cell_size,
                padding=padding,
                header_height=header_height,
                rows=rows,
                cols=cols,
            )

    # --------------------------------------------------
    # Bomb + Bomb
    # --------------------------------------------------

    elif special_types == {SpecialType.BOMB}:
        center = cell_center(
            position,
            cell_size,
            padding,
            header_height,
        )

        max_radius = cell_size * 3.5
        radius = max(1, int(max_radius * progress))
        alpha = int(240 * (1.0 - progress))

        pygame.draw.circle(
            overlay,
            (255, 120, 40, alpha),
            center,
            radius,
            width=12,
        )

        pygame.draw.circle(
            overlay,
            (255, 220, 100, alpha),
            center,
            max(1, int(radius * 0.7)),
            width=6,
        )

    # --------------------------------------------------
    # Light Ball combinations
    # --------------------------------------------------

    elif SpecialType.LIGHT_BALL in special_types:
        light_ball_position = (
            combination.source
            if combination.source_type == SpecialType.LIGHT_BALL
            else combination.destination
        )

        draw_light_ball_effect(
            overlay,
            light_ball_position,
            progress,
            cell_size,
            padding,
            header_height,
        )

        center = cell_center(
            light_ball_position,
            cell_size,
            padding,
            header_height,
        )

        alpha = int(180 * (1.0 - progress))
        beam_progress = min(1.0, progress * 2)

        # Draw energy beams towards affected cells.
        for target_position in combination.effect_cells:
            if target_position == light_ball_position:
                continue

            target = cell_center(
                target_position,
                cell_size,
                padding,
                header_height,
            )

            end_x = int(
                center[0] + (target[0] - center[0]) * beam_progress
            )
            end_y = int(
                center[1] + (target[1] - center[1]) * beam_progress
            )

            pygame.draw.line(
                overlay,
                (180, 220, 255, alpha),
                center,
                (end_x, end_y),
                width=3,
            )

    # --------------------------------------------------
    # Propeller combinations
    # --------------------------------------------------

    elif SpecialType.PROPELLER in special_types:
        propeller_position = (
            combination.source
            if combination.source_type == SpecialType.PROPELLER
            else combination.destination
        )

        draw_propeller_effect(
            overlay,
            propeller_position,
            progress,
            cell_size,
            padding,
            header_height,
        )

        origin = cell_center(
            propeller_position,
            cell_size,
            padding,
            header_height,
        )
        alpha = max(70, int(220 * (1.0 - progress)))
        beam_progress = min(1.0, progress * 1.5)

        for target_position in sorted(combination.effect_cells):
            if target_position == propeller_position:
                continue
            target = cell_center(
                target_position,
                cell_size,
                padding,
                header_height,
            )
            endpoint = (
                int(origin[0] + (target[0] - origin[0]) * beam_progress),
                int(origin[1] + (target[1] - origin[1]) * beam_progress),
            )
            pygame.draw.line(
                overlay,
                (170, 240, 255, alpha),
                origin,
                endpoint,
                width=2,
            )
            if progress >= 0.45:
                pygame.draw.circle(
                    overlay,
                    (205, 248, 255, alpha),
                    endpoint,
                    max(2, int(5 * (1.0 - progress))),
                )

    surface.blit(overlay, (0, 0))
