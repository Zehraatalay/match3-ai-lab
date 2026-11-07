import json
from collections import Counter
from pathlib import Path

import pytest

from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.specials import SpecialType
from match3_ai.levels import (
    BlockerPlacement,
    InitialSpecial,
    LevelConfig,
    LevelObjective,
    ObjectiveType,
    load_level,
)

LEVEL_DATA = Path(__file__).parents[1] / "src/match3_ai/levels/data"


def base_level(**overrides):
    data = {
        "level_id": "test-level",
        "name": "Test Level",
        "rows": 4,
        "cols": 4,
        "num_colors": 4,
        "move_limit": 12,
        "seed": 17,
        "objectives": [{"type": "COLLECT_TILES", "target": 10}],
    }
    data.update(overrides)
    return data


def write_level(tmp_path: Path, data) -> Path:
    path = tmp_path / "level.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    return path


def test_existing_level_config_constructor_remains_compatible():
    config = LevelConfig(rows=8, cols=8, num_colors=5, move_limit=20)

    assert config.level_id == "custom"
    assert config.name == "Custom Level"
    assert config.seed == 0
    assert config.cell_mask is None
    assert config.blockers == ()
    assert config.initial_specials == ()
    assert config.objectives == ()


def test_load_minimal_json_level(tmp_path):
    config = load_level(write_level(tmp_path, base_level()))

    assert config.level_id == "test-level"
    assert config.objectives == (LevelObjective(ObjectiveType.COLLECT_TILES, 10),)
    assert config.cell_mask is None
    assert config.blockers == ()
    assert config.initial_specials == ()


def test_load_full_json_with_mask_blockers_specials_and_objectives(tmp_path):
    data = base_level(
        cell_mask=[
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, True],
            [False, True, True, False],
        ],
        blockers=[
            {"position": [0, 0], "type": "CRATE", "variant": "DOUBLE"},
            {"position": [1, 1], "type": "ICE", "variant": "THICK"},
        ],
        initial_specials=[{"position": [1, 1], "type": "BOMB"}],
        objectives=[
            {"type": "CLEAR_ALL_BLOCKERS"},
            {"type": "COLLECT_TILES", "target": 8, "color": 2},
        ],
    )

    config = load_level(write_level(tmp_path, data))

    assert config.cell_mask is not None
    assert config.cell_mask[3] == (False, True, True, False)
    assert config.blockers == (
        BlockerPlacement((0, 0), BlockerType.CRATE, "DOUBLE"),
        BlockerPlacement((1, 1), BlockerType.ICE, "THICK"),
    )
    assert [blocker.hp for blocker in config.blockers] == [2, 2]
    assert config.initial_specials == (InitialSpecial((1, 1), SpecialType.BOMB),)
    assert config.objectives == (
        LevelObjective(ObjectiveType.CLEAR_ALL_BLOCKERS),
        LevelObjective(ObjectiveType.COLLECT_TILES, target=8, color=2),
    )


@pytest.mark.parametrize(
    ("filename", "blocker_counts", "hp_counts"),
    [
        ("first-match.json", {}, {}),
        ("wooden-trouble.json", {BlockerType.CRATE: 4}, {(BlockerType.CRATE, 1): 4}),
        (
            "double-trouble.json",
            {BlockerType.CRATE: 5},
            {(BlockerType.CRATE, 1): 2, (BlockerType.CRATE, 2): 3},
        ),
        (
            "frozen-garden.json",
            {BlockerType.ICE: 8},
            {(BlockerType.ICE, 1): 4, (BlockerType.ICE, 2): 4},
        ),
        (
            "stone-breaker.json",
            {BlockerType.STONE: 2},
            {(BlockerType.STONE, 1): 1, (BlockerType.STONE, 2): 1},
        ),
        (
            "blocker-mayhem.json",
            {BlockerType.CRATE: 2, BlockerType.ICE: 2, BlockerType.STONE: 2},
            {
                (BlockerType.CRATE, 1): 1,
                (BlockerType.CRATE, 2): 1,
                (BlockerType.ICE, 1): 1,
                (BlockerType.ICE, 2): 1,
                (BlockerType.STONE, 1): 1,
                (BlockerType.STONE, 2): 1,
            },
        ),
    ],
)
def test_six_level_definitions_have_expected_blocker_and_hp_counts(
    filename,
    blocker_counts,
    hp_counts,
):
    config = load_level(LEVEL_DATA / filename)
    actual_blockers = Counter(item.blocker_type for item in config.blockers)
    actual_hp = Counter((item.blocker_type, item.hp) for item in config.blockers)

    assert dict(actual_blockers) == blocker_counts
    assert dict(actual_hp) == hp_counts
    assert config.seed > 0
    assert config.objectives
    if config.blockers:
        assert config.objectives[0].objective_type == ObjectiveType.CLEAR_ALL_BLOCKERS
    assert all(len(item.position) == 2 for item in config.blockers)


def test_six_level_ids_names_and_seeds_are_distinct():
    filenames = (
        "first-match.json",
        "wooden-trouble.json",
        "double-trouble.json",
        "frozen-garden.json",
        "stone-breaker.json",
        "blocker-mayhem.json",
    )
    configs = [load_level(LEVEL_DATA / filename) for filename in filenames]

    assert [config.level_id for config in configs] == [
        "first-match",
        "wooden-trouble",
        "double-trouble",
        "frozen-garden",
        "stone-breaker",
        "blocker-mayhem",
    ]
    assert len({config.name for config in configs}) == 6
    assert len({config.seed for config in configs}) == 6


@pytest.mark.parametrize(
    ("filename", "num_colors", "move_limit", "expected_objectives"),
    [
        (
            "first-match.json",
            5,
            5,
            ((ObjectiveType.COLLECT_TILES, 30, None),),
        ),
        (
            "wooden-trouble.json",
            3,
            10,
            ((ObjectiveType.CLEAR_ALL_BLOCKERS, None, None),),
        ),
        (
            "double-trouble.json",
            4,
            15,
            ((ObjectiveType.CLEAR_ALL_BLOCKERS, None, None),),
        ),
        (
            "frozen-garden.json",
            4,
            15,
            ((ObjectiveType.CLEAR_ALL_BLOCKERS, None, None),),
        ),
        (
            "stone-breaker.json",
            4,
            15,
            ((ObjectiveType.CLEAR_ALL_BLOCKERS, None, None),),
        ),
        (
            "blocker-mayhem.json",
            4,
            15,
            (
                (ObjectiveType.CLEAR_ALL_BLOCKERS, None, None),
                (ObjectiveType.COLLECT_TILES, 30, None),
            ),
        ),
    ],
)
def test_six_levels_match_requested_limits_colors_and_objectives(
    filename,
    num_colors,
    move_limit,
    expected_objectives,
):
    config = load_level(LEVEL_DATA / filename)

    expected_dimensions = {
        "first-match.json": (6, 6),
        "wooden-trouble.json": (7, 7),
        "double-trouble.json": (8, 7),
        "frozen-garden.json": (8, 8),
        "stone-breaker.json": (8, 8),
        "blocker-mayhem.json": (9, 8),
    }
    assert (config.rows, config.cols) == expected_dimensions[filename]
    assert config.num_colors == num_colors
    assert config.move_limit == move_limit
    assert (
        tuple(
            (objective.objective_type, objective.target, objective.color)
            for objective in config.objectives
        )
        == expected_objectives
    )


@pytest.mark.parametrize(
    "blockers,objectives,error",
    [
        (
            [{"position": [1, 1], "type": "CRATE", "variant": "SINGLE"}],
            [{"type": "COLLECT_TILES", "target": 10}],
            "require CLEAR_ALL_BLOCKERS",
        ),
        (
            [],
            [{"type": "CLEAR_ALL_BLOCKERS"}],
            "only valid when.*blocker",
        ),
        (
            [],
            [
                {"type": "CLEAR_ALL_BLOCKERS"},
                {"type": "CLEAR_ALL_BLOCKERS"},
            ],
            "duplicate CLEAR_ALL_BLOCKERS",
        ),
    ],
)
def test_clear_all_blockers_objective_constraints(
    tmp_path, blockers, objectives, error
):
    path = write_level(tmp_path, base_level(blockers=blockers, objectives=objectives))

    with pytest.raises((TypeError, ValueError), match=error):
        load_level(path)


@pytest.mark.parametrize(
    "blocker",
    [
        {"position": [1, 1], "type": "CRATE", "variant": "THIN"},
        {"position": [1, 1], "type": "ICE", "variant": "TRIPLE"},
        {"position": [1, 1], "type": "STONE", "variant": "THICK"},
        {"position": [1, 1], "type": "NONE", "variant": "SINGLE"},
        {"position": [1, 1], "type": "CRATE", "variant": "SINGLE", "hp": 9},
    ],
)
def test_invalid_blocker_types_variants_and_custom_hp_are_rejected(tmp_path, blocker):
    path = write_level(
        tmp_path,
        base_level(
            blockers=[blocker],
            objectives=[{"type": "CLEAR_ALL_BLOCKERS"}],
        ),
    )

    with pytest.raises(ValueError):
        load_level(path)


@pytest.mark.parametrize(
    "mask",
    [
        [[True] * 4] * 3,
        [[True] * 3 for _ in range(4)],
        [[True, True, True, 1] for _ in range(4)],
        "1111",
        None,
    ],
)
def test_invalid_masks_are_rejected(tmp_path, mask):
    with pytest.raises(ValueError, match="cell_mask"):
        load_level(write_level(tmp_path, base_level(cell_mask=mask)))


@pytest.mark.parametrize("position", [[4, 0], [-1, 0], [1], [True, 1], "1,1"])
def test_invalid_blocker_coordinates_are_rejected(tmp_path, position):
    data = base_level(
        blockers=[{"position": position, "type": "CRATE", "variant": "SINGLE"}],
        objectives=[{"type": "CLEAR_ALL_BLOCKERS"}],
    )

    with pytest.raises(ValueError, match="position"):
        load_level(write_level(tmp_path, data))


def test_duplicate_blocker_positions_are_rejected(tmp_path):
    blockers = [
        {"position": [1, 1], "type": "CRATE", "variant": "SINGLE"},
        {"position": [1, 1], "type": "STONE", "variant": "DOUBLE"},
    ]
    path = write_level(
        tmp_path,
        base_level(blockers=blockers, objectives=[{"type": "CLEAR_ALL_BLOCKERS"}]),
    )

    with pytest.raises(ValueError, match="duplicate blocker"):
        load_level(path)


@pytest.mark.parametrize(
    "specials,blockers,mask,error",
    [
        ([{"position": [0, 0], "type": "BAD"}], [], None, "invalid special type"),
        ([{"position": [0, 0], "type": "NONE"}], [], None, "cannot be NONE"),
        (
            [
                {"position": [0, 0], "type": "BOMB"},
                {"position": [0, 0], "type": "PROPELLER"},
            ],
            [],
            None,
            "duplicate special",
        ),
        (
            [{"position": [0, 0], "type": "BOMB"}],
            [{"position": [0, 0], "type": "CRATE", "variant": "SINGLE"}],
            None,
            "specials cannot be placed",
        ),
        (
            [{"position": [0, 0], "type": "BOMB"}],
            [{"position": [0, 0], "type": "STONE", "variant": "DOUBLE"}],
            None,
            "specials cannot be placed",
        ),
        ([{"position": [4, 0], "type": "BOMB"}], [], None, "outside the board"),
        (
            [{"position": [0, 0], "type": "BOMB"}],
            [],
            [[False] + [True] * 3] + [[True] * 4 for _ in range(3)],
            "masked cell",
        ),
    ],
)
def test_invalid_special_placements_are_rejected(
    tmp_path,
    specials,
    blockers,
    mask,
    error,
):
    objectives = (
        [{"type": "CLEAR_ALL_BLOCKERS"}]
        if blockers
        else [{"type": "COLLECT_TILES", "target": 1}]
    )
    data = base_level(
        blockers=blockers,
        initial_specials=specials,
        objectives=objectives,
    )
    if mask is not None:
        data["cell_mask"] = mask
    path = write_level(tmp_path, data)

    with pytest.raises(ValueError, match=error):
        load_level(path)


def test_blocker_on_masked_cell_is_rejected(tmp_path):
    mask = [[True] * 4 for _ in range(4)]
    mask[2][2] = False
    path = write_level(
        tmp_path,
        base_level(
            cell_mask=mask,
            blockers=[{"position": [2, 2], "type": "ICE", "variant": "THIN"}],
            objectives=[{"type": "CLEAR_ALL_BLOCKERS"}],
        ),
    )

    with pytest.raises(ValueError, match="masked cell"):
        load_level(path)


@pytest.mark.parametrize(
    "objective,error",
    [
        ({"type": "REACH_SCORE", "target": 10}, "REACH_SCORE.*not supported"),
        ({"type": "COLLECT_TILES", "target": 0}, "positive integer"),
        ({"type": "COLLECT_TILES", "target": True}, "target must be an integer"),
        ({"type": "COLLECT_TILES", "target": 2, "color": 0}, "between 1 and"),
        ({"type": "COLLECT_TILES", "target": 2, "color": 5}, "between 1 and"),
        ({"type": "COLLECT_TILES", "target": 2, "score": 4}, "unknown objectives"),
        ({"type": "CLEAR_ALL_BLOCKERS", "target": 1}, "unknown objectives"),
        ({"type": "OTHER"}, "unsupported objective type"),
    ],
)
def test_invalid_objectives_are_rejected(tmp_path, objective, error):
    path = write_level(tmp_path, base_level(objectives=[objective]))

    with pytest.raises(ValueError, match=error):
        load_level(path)


@pytest.mark.parametrize(
    "field,value,error",
    [
        ("rows", True, "rows must be an integer"),
        ("cols", 2.5, "cols must be an integer"),
        ("num_colors", 255, "between 3 and 254"),
        ("move_limit", 0, "positive integer"),
        ("seed", -1, "non-negative integer"),
        ("seed", False, "seed must be an integer"),
        ("name", 4, "name must be a string"),
        ("level_id", "", "non-empty string"),
    ],
)
def test_invalid_primitive_values_are_rejected(tmp_path, field, value, error):
    path = write_level(tmp_path, base_level(**{field: value}))

    with pytest.raises((TypeError, ValueError), match=error):
        load_level(path)


def test_unknown_level_fields_and_missing_required_fields_are_rejected(tmp_path):
    unknown_path = write_level(tmp_path, base_level(unused=True))
    with pytest.raises(ValueError, match="unknown level fields"):
        load_level(unknown_path)

    missing = base_level()
    del missing["seed"]
    missing_path = write_level(tmp_path, missing)
    with pytest.raises(ValueError, match="missing level fields.*seed"):
        load_level(missing_path)


@pytest.mark.parametrize("root", [[], None, "level", 7])
def test_non_object_json_roots_are_rejected(tmp_path, root):
    path = write_level(tmp_path, root)

    with pytest.raises(TypeError, match="root must be an object"):
        load_level(path)


def test_invalid_json_and_missing_files_have_clear_errors(tmp_path):
    invalid_path = tmp_path / "broken.json"
    invalid_path.write_text("{broken", encoding="utf-8")
    with pytest.raises(ValueError, match="invalid JSON"):
        load_level(invalid_path)

    with pytest.raises(FileNotFoundError, match="level file not found"):
        load_level(tmp_path / "missing.json")

    invalid_encoding = tmp_path / "encoding.json"
    invalid_encoding.write_bytes(b"\xff")
    with pytest.raises(ValueError, match="invalid JSON encoding"):
        load_level(invalid_encoding)


def test_loaded_nested_configuration_is_immutable(tmp_path):
    config = load_level(
        write_level(
            tmp_path,
            base_level(
                cell_mask=[[True] * 4 for _ in range(4)],
                blockers=[{"position": [1, 1], "type": "CRATE", "variant": "SINGLE"}],
                objectives=[{"type": "CLEAR_ALL_BLOCKERS"}],
            ),
        )
    )

    assert isinstance(config.cell_mask, tuple)
    assert all(isinstance(row, tuple) for row in config.cell_mask)
    assert isinstance(config.blockers, tuple)
    assert isinstance(config.objectives, tuple)
    with pytest.raises((AttributeError, TypeError)):
        config.name = "Changed"
    with pytest.raises(TypeError):
        config.cell_mask[0][0] = False
