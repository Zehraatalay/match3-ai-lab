from match3_ai.games.match3.pattern_detection import (
    MatchGroup,
    MatchPattern,
)
from match3_ai.games.match3.special_selection import (
    select_special_creations,
)
from match3_ai.games.match3.specials import SpecialType


def make_pattern(kind, cells):
    return MatchPattern(
        cells=frozenset(cells),
        kind=kind,
    )


def make_group(patterns):
    cells = frozenset().union(*(pattern.cells for pattern in patterns))

    return MatchGroup(
        cells=cells,
        patterns=tuple(patterns),
        color=1,
    )


def test_five_line_creates_light_ball():
    pattern = make_pattern(
        "horizontal",
        {(2, c) for c in range(5)},
    )

    group = make_group([pattern])

    creations = select_special_creations(
        [group],
        destination=(2, 3),
    )

    assert len(creations) == 1
    assert creations[0].special_type == SpecialType.LIGHT_BALL
    assert creations[0].position == (2, 3)


def test_intersecting_lines_create_bomb():
    horizontal = make_pattern(
        "horizontal",
        {(2, c) for c in range(3)},
    )

    vertical = make_pattern(
        "vertical",
        {(r, 1) for r in range(1, 4)},
    )

    group = make_group([horizontal, vertical])

    creations = select_special_creations(
        [group],
        destination=(2, 1),
    )

    assert creations[0].special_type == SpecialType.BOMB


def test_horizontal_four_creates_horizontal_rocket():
    pattern = make_pattern(
        "horizontal",
        {(1, c) for c in range(4)},
    )

    group = make_group([pattern])

    creations = select_special_creations(
        [group],
        destination=(1, 2),
    )

    assert creations[0].special_type == SpecialType.ROCKET_HORIZONTAL


def test_vertical_four_creates_vertical_rocket():
    pattern = make_pattern(
        "vertical",
        {(r, 2) for r in range(4)},
    )

    group = make_group([pattern])

    creations = select_special_creations(
        [group],
        destination=(2, 2),
    )

    assert creations[0].special_type == SpecialType.ROCKET_VERTICAL


def test_square_creates_propeller():
    pattern = make_pattern(
        "square",
        {
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        },
    )

    group = make_group([pattern])

    creations = select_special_creations(
        [group],
        destination=(1, 1),
    )

    assert creations[0].special_type == SpecialType.PROPELLER


def test_three_line_creates_no_special():
    pattern = make_pattern(
        "horizontal",
        {
            (0, 0),
            (0, 1),
            (0, 2),
        },
    )

    group = make_group([pattern])

    creations = select_special_creations(
        [group],
        destination=(0, 1),
    )

    assert creations == []


def test_two_independent_groups_create_two_specials():
    first_pattern = make_pattern(
        "horizontal",
        {(0, c) for c in range(5)},
    )

    second_pattern = make_pattern(
        "vertical",
        {(r, 6) for r in range(4)},
    )

    first_group = make_group([first_pattern])
    second_group = make_group([second_pattern])

    creations = select_special_creations(
        [first_group, second_group],
        destination=(0, 3),
    )

    assert len(creations) == 2

    assert creations[0].special_type == SpecialType.LIGHT_BALL
    assert creations[0].position == (0, 3)

    assert creations[1].special_type == SpecialType.ROCKET_VERTICAL
    assert creations[1].position == (0, 6)


def test_light_ball_has_priority_over_bomb():
    horizontal = make_pattern(
        "horizontal",
        {(2, c) for c in range(5)},
    )

    vertical = make_pattern(
        "vertical",
        {(r, 2) for r in range(1, 4)},
    )

    group = make_group([horizontal, vertical])

    creations = select_special_creations(
        [group],
        destination=(2, 2),
    )

    assert creations[0].special_type == SpecialType.LIGHT_BALL
