import argparse
import csv
import time
from pathlib import Path

from match3_ai.evaluation.difficulty_evaluator import DifficultyEvaluator
from match3_ai.levels.difficulty import Difficulty, classify_difficulty
from match3_ai.levels.generation import GenerationSpec, LevelGenerator
from match3_ai.levels.validation import LevelValidator

PROFILES = {
    "baseline": {
        "min_moves": 8,
        "max_moves": 20,
        "min_blockers": 2,
        "max_blockers": 8,
        "num_colors": 5,
    },
    "restricted": {
        "min_moves": 5,
        "max_moves": 10,
        "min_blockers": 6,
        "max_blockers": 12,
        "num_colors": 5,
    },
    "restricted_colors": {
        "min_moves": 5,
        "max_moves": 10,
        "min_blockers": 6,
        "max_blockers": 12,
        "num_colors": 6,
    },
}


def run_analysis(
    *,
    output_dir: Path,
    candidates_per_profile: int,
    episodes: int,
    mcts_simulations: int,
    generation_seed: int,
    evaluation_seed: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    validator = LevelValidator()

    evaluator = DifficultyEvaluator(
        episodes_per_agent=episodes,
        seed=evaluation_seed,
        agents=("MCTS_greedy_outcome",),
        mcts_simulations=mcts_simulations,
    )

    rows = []
    start_total = time.perf_counter()

    for profile_name, parameters in PROFILES.items():
        print(f"\nPROFILE: {profile_name}", flush=True)

        spec = GenerationSpec(
            rows=8,
            cols=8,
            num_colors=parameters["num_colors"],
            min_moves=parameters["min_moves"],
            max_moves=parameters["max_moves"],
            min_blockers=parameters["min_blockers"],
            max_blockers=parameters["max_blockers"],
            seed=generation_seed,
        )

        generator = LevelGenerator(spec)

        for candidate_id in range(candidates_per_profile):
            level = generator.generate(candidate_id)

            validation = validator.validate(level)

            row = {
                "profile": profile_name,
                "candidate_id": candidate_id,
                "level_id": level.level_id,
                "move_limit": level.move_limit,
                "blocker_count": len(level.blockers),
                "num_colors": level.num_colors,
                "valid": validation.valid,
                "validation_reason": validation.reason or "",
                "win_rate": "",
                "ci_lower": "",
                "ci_upper": "",
                "easy": "",
                "medium": "",
                "hard": "",
                "very_hard": "",
                "elapsed_seconds": "",
            }

            if not validation.valid:
                rows.append(row)
                print(
                    f"  Candidate {candidate_id}: INVALID ({validation.reason})",
                    flush=True,
                )
                continue

            start = time.perf_counter()

            report = evaluator.evaluate(level)

            elapsed = time.perf_counter() - start

            metrics = report.agent_metrics[0]

            row["win_rate"] = metrics.win_rate
            row["ci_lower"] = metrics.ci_lower
            row["ci_upper"] = metrics.ci_upper
            row["elapsed_seconds"] = round(elapsed, 3)

            for difficulty in Difficulty:
                classification = classify_difficulty(report, difficulty)
                row[difficulty.value.lower()] = classification.status.value

            rows.append(row)

            print(
                f"  Candidate {candidate_id}: "
                f"moves={level.move_limit}, "
                f"blockers={len(level.blockers)}, "
                f"colors={level.num_colors}, "
                f"win_rate={metrics.win_rate:.1%}, "
                f"CI=[{metrics.ci_lower:.1%}, {metrics.ci_upper:.1%}], "
                f"time={elapsed:.2f}s",
                flush=True,
            )

    output_path = output_dir / "difficulty_distribution.csv"

    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    total_elapsed = time.perf_counter() - start_total

    print(f"\nTotal elapsed: {total_elapsed:.2f}s")
    print(f"Results: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plg/difficulty_analysis"),
    )
    parser.add_argument("--candidates-per-profile", type=int, default=5)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--mcts-simulations", type=int, default=10)
    parser.add_argument("--generation-seed", type=int, default=42)
    parser.add_argument("--evaluation-seed", type=int, default=1000)

    args = parser.parse_args()

    if args.candidates_per_profile <= 0:
        parser.error("--candidates-per-profile must be positive")
    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    if args.mcts_simulations <= 0:
        parser.error("--mcts-simulations must be positive")

    run_analysis(
        output_dir=args.output_dir,
        candidates_per_profile=args.candidates_per_profile,
        episodes=args.episodes,
        mcts_simulations=args.mcts_simulations,
        generation_seed=args.generation_seed,
        evaluation_seed=args.evaluation_seed,
    )


if __name__ == "__main__":
    main()
