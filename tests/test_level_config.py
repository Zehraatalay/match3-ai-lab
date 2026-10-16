import pytest

from match3_ai.levels.config import LevelConfig


def test_valid_level_config():
    config = LevelConfig(
        rows=8,
        cols=8,
        num_colors=5,
        move_limit=20
    )

    assert config.rows == 8
    assert config.cols == 8
    assert config.num_colors == 5
    assert config.move_limit == 20


@pytest.mark.parametrize(
    "field, value",
    [
        ("rows", 2),
        ("cols", 1),
        ("num_colors", 2),
        ("move_limit", 0),
        ("move_limit", -5),
        ("rows", 3.5),
        ("cols", True),
        ("num_colors", "5"),
    ]
)
def test_invalid_level_config(field, value):
    params = {
        "rows": 8,
        "cols": 8,
        "num_colors": 5,
        "move_limit": 20
    }

    params[field] = value

    with pytest.raises(ValueError):
        LevelConfig(**params)


def test_config_is_immutable():
    config = LevelConfig(
        rows=8,
        cols=8,
        num_colors=5,
        move_limit=20
    )

    with pytest.raises(AttributeError):
        config.rows = 10