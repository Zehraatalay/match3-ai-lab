import json

import pytest

from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.specials import SpecialType
from match3_ai.levels.config import (
    BlockerPlacement,
    InitialSpecial,
    LevelConfig,
    LevelObjective,
    ObjectiveType,
)
from match3_ai.levels.export import export_level
from match3_ai.levels.generation import GenerationSpec, LevelGenerator
from match3_ai.levels.loader import load_level


def test_generated_level_json_round_trip(tmp_path):
    generator = LevelGenerator(GenerationSpec(seed=42))

    for candidate_id in range(10):
        original = generator.generate(candidate_id)

        path = export_level(
            original,
            tmp_path / f"level-{candidate_id}.json",
        )

        restored = load_level(path)

        assert restored == original


def test_complex_level_json_round_trip(tmp_path):
    level = LevelConfig(
        level_id="complex-test",
        name="Complex Test",
        rows=4,
        cols=4,
        num_colors=4,
        move_limit=12,
        seed=123,
        cell_mask=(
            (True, True, True, True),
            (True, True, True, True),
            (True, True, True, True),
            (True, True, True, False),
        ),
        blockers=(
            BlockerPlacement(
                position=(0, 0),
                blocker_type=BlockerType.CRATE,
                variant="DOUBLE",
            ),
        ),
        initial_specials=(
            InitialSpecial(
                position=(1, 1),
                special_type=next(
                    special for special in SpecialType if special != SpecialType.NONE
                ),
            ),
        ),
        objectives=(
            LevelObjective(
                objective_type=ObjectiveType.CLEAR_ALL_BLOCKERS,
            ),
            LevelObjective(
                objective_type=ObjectiveType.COLLECT_TILES,
                target=10,
                color=2,
            ),
        ),
    )

    path = export_level(level, tmp_path / "complex.json")

    restored = load_level(path)

    assert restored == level


def test_export_rejects_non_level(tmp_path):
    with pytest.raises(TypeError):
        export_level({"rows": 8}, tmp_path / "invalid.json")


def test_exported_json_uses_loader_schema(tmp_path):
    level = LevelGenerator(GenerationSpec(seed=42)).generate(0)

    path = export_level(level, tmp_path / "level.json")

    data = json.loads(path.read_text(encoding="utf-8"))

    assert "objectives" in data
    assert "blockers" in data
    assert "initial_specials" in data
    assert "cell_mask" not in data
    assert all("type" in item for item in data["objectives"])
    assert all("type" in item for item in data["blockers"])


from match3_ai.levels.difficulty import Difficulty
from match3_ai.levels.export import export_generation_report
from match3_ai.levels.screening import ScreeningResult
from match3_ai.levels.targeted_generation import (
    TargetGenerationConfig,
    TargetGenerationResult,
)


def test_generation_report_export(tmp_path):
    generation_spec = GenerationSpec(seed=42)

    config = TargetGenerationConfig(
        target=Difficulty.HARD,
        max_candidates=3,
    )

    result = TargetGenerationResult(
        target=Difficulty.HARD,
        selected_level=None,
        screening=ScreeningResult(
            target=Difficulty.HARD,
            candidates=(),
            selected_candidate_id=None,
            selected_level_id=None,
        ),
        confirmation=None,
    )

    path = export_generation_report(
        result,
        generation_spec,
        config,
        tmp_path / "generation_report.json",
    )

    data = json.loads(path.read_text(encoding="utf-8"))

    assert data["schema_version"] == 1
    assert data["target"] == "HARD"
    assert data["found"] is False
    assert data["statistics"]["attempted_candidates"] == 0
    assert data["statistics"]["confirmation_attempts"] == 0
