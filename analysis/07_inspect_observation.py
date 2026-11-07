import json
from pathlib import Path

import numpy as np

from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.loader import load_level

PROJECT_ROOT = Path(__file__).resolve().parents[1]

EXPERIMENT_DIR = (
    PROJECT_ROOT / "experiments" / "results" / "level_calibration_stage2_confirmation"
)

VARIANT_ID = "wooden-trouble__colors5-double-moves6"
SEED = 761000


def inspect_value(name, value):
    print(f"\n--- {name} ---")
    print("Type:", type(value))

    if isinstance(value, np.ndarray):
        print("Shape:", value.shape)
        print("Dtype:", value.dtype)
        print("Values:")
        print(value)

    elif isinstance(value, (list, tuple)):
        print("Length:", len(value))
        print("Values:", value)

    elif isinstance(value, dict):
        print("Keys:", list(value.keys()))

        for key, item in value.items():
            inspect_value(f"{name}.{key}", item)

    else:
        print("Value:", value)


def main():
    manifest = json.loads((EXPERIMENT_DIR / "manifest.json").read_text())

    specification = manifest["configurations"][VARIANT_ID]

    config_path = EXPERIMENT_DIR / specification["path"]

    config = load_level(config_path)

    env = Match3LevelEnv(
        config.level_id,
        level_config=config,
    )

    try:
        observation, info = env.reset(seed=SEED)

        print("\n========== LEVEL CONFIG ==========")

        print("Level ID:", config.level_id)
        print("Move limit:", config.move_limit)
        print("Number of colors:", config.num_colors)
        print("Blockers:", config.blockers)

        print("\n========== OBSERVATION ==========")

        inspect_value("observation", observation)

        print("\n========== RESET INFO ==========")

        inspect_value("info", info)

        print("\n========== OBSERVATION SPACE ==========")

        print(env.observation_space)

        print("\n========== ACTION SPACE ==========")

        print(env.action_space)

    finally:
        env.close()


if __name__ == "__main__":
    main()
