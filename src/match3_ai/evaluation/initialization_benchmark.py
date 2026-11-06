"""Profile bounded level resets without running gameplay episodes."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from time import perf_counter
from unittest.mock import patch

import numpy as np

from match3_ai.games.match3.environment import (
    DEFAULT_LEVEL_INITIALIZATION_ATTEMPTS,
    Match3LevelEnv,
)
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.pattern_detection import detect_patterns
from match3_ai.levels import LevelSession, initialization


def compare_first_match_colors(
    seeds: tuple[int, ...], *, max_initialization_attempts: int = 1000
) -> dict:
    """Compare frozen 3/5-color configurations on identical environment seed streams."""
    if not seeds or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be nonempty non-negative integers")
    env = Match3LevelEnv("first-match")
    base = env.config
    level_index = env.level_ids.index("first-match")
    records, summaries, configurations = [], [], {}
    real_generate = initialization.generate_random_board
    attempts = 0

    def counted_generate(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return real_generate(*args, **kwargs)

    for num_colors in (3, 5):
        config = replace(base, num_colors=num_colors)
        configurations[str(num_colors)] = asdict(config)
        for seed in seeds:
            board_seed, engine_seed = (
                int(child.generate_state(1, dtype=np.uint32)[0])
                for child in np.random.SeedSequence([seed, level_index]).spawn(2)
            )
            attempts = 0
            with patch.object(
                initialization, "generate_random_board", counted_generate
            ):
                started = perf_counter()
                try:
                    session = LevelSession(
                        config,
                        board_seed=board_seed,
                        engine_seed=engine_seed,
                        max_initialization_attempts=max_initialization_attempts,
                    )
                except initialization.LevelGenerationError as error:
                    elapsed = perf_counter() - started
                    records.append(
                        {
                            "num_colors": num_colors,
                            "environment_seed": seed,
                            "board_seed": board_seed,
                            "engine_seed": engine_seed,
                            "success": False,
                            "attempts": attempts,
                            "seconds": elapsed,
                            "initial_patterns": None,
                            "legal_moves": None,
                            "error": str(error),
                        }
                    )
                else:
                    elapsed = perf_counter() - started
                    patterns = detect_patterns(session.board.tiles)
                    legal_moves = find_valid_moves(session.board)
                    if patterns or not legal_moves:
                        raise RuntimeError(
                            "generator returned an invalid initial board"
                        )
                    records.append(
                        {
                            "num_colors": num_colors,
                            "environment_seed": seed,
                            "board_seed": board_seed,
                            "engine_seed": engine_seed,
                            "success": True,
                            "attempts": attempts,
                            "seconds": elapsed,
                            "initial_patterns": len(patterns),
                            "legal_moves": len(legal_moves),
                            "error": None,
                        }
                    )
        group = [row for row in records if row["num_colors"] == num_colors]
        successful = [row for row in group if row["success"]]
        summaries.append(
            {
                "num_colors": num_colors,
                "resets": len(group),
                "successes": len(successful),
                "success_rate": len(successful) / len(group),
                "mean_attempts_all": float(np.mean([row["attempts"] for row in group])),
                "mean_attempts_successful": float(
                    np.mean([row["attempts"] for row in successful])
                )
                if successful
                else None,
                "max_attempts_all": max(row["attempts"] for row in group),
                "mean_seconds": float(np.mean([row["seconds"] for row in group])),
                "valid_boards": sum(
                    row["initial_patterns"] == 0 and bool(row["legal_moves"])
                    for row in successful
                ),
            }
        )
    env.close()
    return {
        "environment_seeds": list(seeds),
        "max_initialization_attempts": max_initialization_attempts,
        "configurations": configurations,
        "summary": summaries,
        "records": records,
        "limitations": "finite paired seed sample; failed attempts are capped at the budget; no gameplay balance conclusions",
    }


def benchmark_initialization(
    level_ids: tuple[str, ...],
    seeds: tuple[int, ...],
    *,
    max_initialization_attempts: int = DEFAULT_LEVEL_INITIALIZATION_ATTEMPTS,
) -> dict:
    """Count unchanged generator calls and measure complete reset latency.

    Instrumentation is serial and consumes no RNG. Constructor time is excluded;
    reset timing includes generation, observation/mask construction and the counter.
    """
    if not level_ids or not seeds:
        raise ValueError("level_ids and seeds must be nonempty")
    if any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("seeds must be non-negative integers")
    records = []
    real_generate = initialization.generate_random_board
    attempts = 0

    def counted_generate(*args, **kwargs):
        nonlocal attempts
        attempts += 1
        return real_generate(*args, **kwargs)

    for level in level_ids:
        env = Match3LevelEnv(
            level, max_initialization_attempts=max_initialization_attempts
        )
        try:
            for seed in seeds:
                attempts = 0
                with patch.object(
                    initialization, "generate_random_board", counted_generate
                ):
                    started = perf_counter()
                    try:
                        observation, info = env.reset(seed=seed)
                    except initialization.LevelGenerationError as error:
                        elapsed = perf_counter() - started
                        records.append(
                            {
                                "level_id": level,
                                "environment_seed": seed,
                                "attempts": attempts,
                                "success": False,
                                "seconds": elapsed,
                                "board_fingerprint": None,
                                "board_seed": None,
                                "engine_seed": None,
                                "error_type": type(error).__name__,
                                "error_message": str(error),
                            }
                        )
                    else:
                        elapsed = perf_counter() - started
                        digest = hashlib.sha256(
                            b"".join(
                                observation[key].tobytes()
                                for key in (
                                    "tiles",
                                    "specials",
                                    "cell_mask",
                                    "blockers",
                                    "blocker_hp",
                                )
                            )
                        ).hexdigest()
                        records.append(
                            {
                                "level_id": level,
                                "environment_seed": seed,
                                "attempts": attempts,
                                "success": True,
                                "seconds": elapsed,
                                "board_fingerprint": digest,
                                "board_seed": info["board_seed"],
                                "engine_seed": info["engine_seed"],
                                "error_type": None,
                                "error_message": None,
                            }
                        )
        finally:
            env.close()
    summary = []
    for level in level_ids:
        group = [row for row in records if row["level_id"] == level]
        summary.append(
            {
                "level_id": level,
                "resets": len(group),
                "successes": sum(row["success"] for row in group),
                "failures": sum(not row["success"] for row in group),
                "success_rate": sum(row["success"] for row in group) / len(group),
                "mean_attempts": float(np.mean([row["attempts"] for row in group])),
                "p95_attempts": float(
                    np.percentile([row["attempts"] for row in group], 95)
                ),
                "max_attempts": max(row["attempts"] for row in group),
                "mean_seconds": float(np.mean([row["seconds"] for row in group])),
                "p95_seconds": float(
                    np.percentile([row["seconds"] for row in group], 95)
                ),
            }
        )
    return {
        "max_initialization_attempts": max_initialization_attempts,
        "environment_seeds": list(seeds),
        "summary": summary,
        "records": records,
        "timing_scope": "serial reset including generator call counter and observation/action-mask construction; constructor excluded",
        "limitations": "finite seed sample; timing is hardware dependent; a higher bounded budget does not guarantee initialization",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/results/sprint5_6_initialization"),
    )
    parser.add_argument("--seed", type=int, default=52000)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--compare-first-match-colors", action="store_true")
    parser.add_argument(
        "--max-initialization-attempts",
        type=int,
        default=DEFAULT_LEVEL_INITIALIZATION_ATTEMPTS,
    )
    args = parser.parse_args()
    if args.samples <= 0:
        parser.error("samples must be positive")
    if args.output_dir.exists():
        parser.error("output directory already exists")
    seeds = tuple(range(args.seed, args.seed + args.samples))
    if args.compare_first_match_colors:
        result = compare_first_match_colors(
            seeds,
            max_initialization_attempts=args.max_initialization_attempts,
        )
    else:
        result = benchmark_initialization(
            Match3LevelEnv().level_ids,
            seeds,
            max_initialization_attempts=args.max_initialization_attempts,
        )
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "initialization_benchmark.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    with (args.output_dir / "initialization.csv").open(
        "x", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(result["records"][0]))
        writer.writeheader()
        writer.writerows(result["records"])
    for row in result["summary"]:
        print(json.dumps(row), flush=True)
    print(f"Results: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
