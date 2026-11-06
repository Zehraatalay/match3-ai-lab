import numpy as np

from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.board import BoardState
from match3_ai.games.match3.constants import COLORLESS, EMPTY
from match3_ai.games.match3.generator import generate_random_board
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.pattern_detection import detect_patterns
from match3_ai.games.match3.specials import SpecialType
from match3_ai.levels.config import LevelConfig

DEFAULT_MAX_INITIALIZATION_ATTEMPTS = 1000


class LevelGenerationError(RuntimeError):
    pass


def create_level_board(
    config: LevelConfig,
    max_attempts: int = DEFAULT_MAX_INITIALIZATION_ATTEMPTS,
    *,
    seed: int | None = None,
) -> BoardState:
    if not isinstance(config, LevelConfig):
        raise TypeError("config must be a LevelConfig")
    if type(max_attempts) is not int or max_attempts <= 0:
        raise ValueError("max_attempts must be a positive integer")
    if seed is not None and (type(seed) is not int or seed < 0):
        raise ValueError("seed must be a non-negative integer")

    shape = (config.rows, config.cols)
    cell_mask = (
        np.ones(shape, dtype=np.bool_)
        if config.cell_mask is None
        else np.asarray(config.cell_mask, dtype=np.bool_)
    )
    blockers = np.zeros(shape, dtype=np.uint8)
    blocker_hp = np.zeros(shape, dtype=np.uint8)

    for placement in config.blockers:
        blockers[placement.position] = placement.blocker_type
        blocker_hp[placement.position] = placement.hp

    fixed_blockers = np.isin(blockers, [BlockerType.CRATE, BlockerType.STONE])
    rng = np.random.default_rng(config.seed if seed is None else seed)

    for _ in range(max_attempts):
        board_seed = int(rng.integers(0, 2**32))
        tiles = generate_random_board(
            rows=config.rows,
            cols=config.cols,
            num_colors=config.num_colors,
            seed=board_seed,
        )
        tiles[~cell_mask] = EMPTY
        tiles[fixed_blockers] = EMPTY

        specials = np.zeros(shape, dtype=np.uint8)
        for initial_special in config.initial_specials:
            position = initial_special.position
            special_type = initial_special.special_type
            specials[position] = special_type
            if special_type == SpecialType.LIGHT_BALL:
                tiles[position] = COLORLESS

        state = BoardState(
            tiles=tiles,
            cell_mask=cell_mask,
            specials=specials,
            blockers=blockers,
            blocker_hp=blocker_hp,
        )

        if detect_patterns(state.tiles):
            continue
        if not find_valid_moves(state):
            continue

        return state

    raise LevelGenerationError(
        f"could not generate a playable initial board for level "
        f"{config.level_id!r} after {max_attempts} attempts"
    )
