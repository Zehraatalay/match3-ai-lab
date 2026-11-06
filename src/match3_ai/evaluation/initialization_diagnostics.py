"""Read-only, bounded investigation of the First Match initialization failure."""

from collections import Counter
from dataclasses import replace
from unittest.mock import patch

import numpy as np

from match3_ai.games.match3.board import BoardState, generate_random_board
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.pattern_detection import detect_patterns
from match3_ai.levels import initialization


def investigate_initialization(sample_size: int = 10000) -> dict:
    """Replay the failed seed, audit rejection reasons, and estimate acceptance."""
    if type(sample_size) is not int or sample_size <= 0:
        raise ValueError("sample_size must be a positive integer")
    env = Match3LevelEnv("first-match", max_initialization_attempts=1000)
    config = replace(env.config, num_colors=3)
    children = np.random.SeedSequence(
        [52028, env.level_ids.index("first-match")]
    ).spawn(2)
    board_seed, engine_seed = (
        int(child.generate_state(1, dtype=np.uint32)[0]) for child in children
    )
    repetitions = []
    for _ in range(2):
        counts = Counter()

        def counted_patterns(board, counts=counts):
            patterns = detect_patterns(board)
            counts["attempts"] += 1
            counts["match_free"] += not bool(patterns)
            for kind in {pattern.kind for pattern in patterns}:
                counts[f"has_{kind}"] += 1
            return patterns

        def counted_moves(board, counts=counts):
            moves = find_valid_moves(board)
            counts["legal_checks"] += 1
            counts["dead_boards"] += not bool(moves)
            return moves

        with (
            patch.object(initialization, "detect_patterns", counted_patterns),
            patch.object(initialization, "find_valid_moves", counted_moves),
        ):
            try:
                initialization.create_level_board(
                    config, seed=board_seed, max_attempts=1000
                )
            except initialization.LevelGenerationError as error:
                repetitions.append({"counts": dict(counts), "error": str(error)})
            else:
                raise RuntimeError(
                    "the recorded initialization failure no longer reproduces"
                )

    attempts = 0

    def count_continuation(board):
        nonlocal attempts
        attempts += 1
        return detect_patterns(board)

    with patch.object(initialization, "detect_patterns", count_continuation):
        continued = initialization.create_level_board(
            config, max_attempts=10000, seed=board_seed
        )
    continuation = {
        "diagnostic_attempt_budget": 10000,
        "first_playable_attempt": attempts,
        "final_patterns": len(detect_patterns(continued.tiles)),
        "legal_moves": len(find_valid_moves(continued)),
    }

    rng = np.random.default_rng(55028)
    accepted = matched = dead = 0
    for child_seed in rng.integers(0, 2**32, size=sample_size):
        tiles = generate_random_board(
            config.rows,
            config.cols,
            config.num_colors,
            seed=int(child_seed),
        )
        if detect_patterns(tiles):
            matched += 1
        elif find_valid_moves(BoardState(tiles, np.ones_like(tiles, dtype=bool))):
            accepted += 1
        else:
            dead += 1
    probability = accepted / sample_size
    env.close()
    return {
        "level_id": "first-match",
        "environment_seed": 52028,
        "historical_configuration": True,
        "board_seed": board_seed,
        "engine_seed": engine_seed,
        "repetitions": repetitions,
        "continuation": continuation,
        "sampling_seed": 55028,
        "sample_size": sample_size,
        "accepted": accepted,
        "matched": matched,
        "dead_match_free": dead,
        "estimated_acceptance_probability": probability,
        "approximate_1000_attempt_failure_probability": (1 - probability) ** 1000,
        "configuration": {
            "rows": config.rows,
            "cols": config.cols,
            "num_colors": config.num_colors,
        },
        "limitations": "Monte Carlo acceptance estimate; attempt-independence approximation; no rule changes or benchmark reruns",
    }
