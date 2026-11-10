import argparse
import csv
import time
from pathlib import Path

from match3_ai.levels.difficulty import Difficulty
from match3_ai.levels.export import (
    export_generation_report,
    export_level,
)
from match3_ai.levels.generation import GenerationSpec
from match3_ai.levels.targeted_generation import (
    TargetDifficultyGenerator,
    TargetGenerationConfig,
)


def run_benchmark(
    *,
    output_dir: Path,
    max_candidates: int,
    screening_episodes: int,
    confirmation_episodes: int,
    mcts_simulations: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for target in (
        Difficulty.EASY,
        Difficulty.MEDIUM,
        Difficulty.HARD,
    ):
        print(f"\nRunning target: {target.value}", flush=True)

        generation_spec = GenerationSpec(
            rows=8,
            cols=8,
            num_colors=5,
            min_moves=8,
            max_moves=20,
            min_blockers=2,
            max_blockers=8,
            seed=42,
        )

        config = TargetGenerationConfig(
            target=target,
            max_candidates=max_candidates,
            screening_episodes=screening_episodes,
            confirmation_episodes=confirmation_episodes,
            screening_seed=1000,
            confirmation_seed=100_000,
            mcts_simulations=mcts_simulations,
        )

        generator = TargetDifficultyGenerator(
            generation_spec=generation_spec,
            config=config,
        )

        start = time.perf_counter()
        result = generator.generate()
        elapsed = time.perf_counter() - start

        target_dir = output_dir / target.value.lower()
        target_dir.mkdir(parents=True, exist_ok=True)

        export_generation_report(
            result,
            generation_spec,
            config,
            target_dir / "generation_report.json",
        )

        if result.selected_level is not None:
            export_level(
                result.selected_level,
                target_dir / "level.json",
            )

        candidates = result.screening.candidates

        valid_count = sum(candidate.valid for candidate in candidates)

        screened_count = sum(
            candidate.classification is not None for candidate in candidates
        )

        eligible_count = sum(
            candidate.classification is not None
            and candidate.classification.status.value in ("MATCH", "UNCERTAIN")
            for candidate in candidates
        )

        row = {
            "target": target.value,
            "found": result.found,
            "selected_level_id": (
                result.selected_level.level_id
                if result.selected_level is not None
                else ""
            ),
            "attempted_candidates": len(candidates),
            "valid_candidates": valid_count,
            "screened_candidates": screened_count,
            "eligible_candidates": eligible_count,
            "confirmation_attempts": len(result.confirmation_attempts),
            "elapsed_seconds": round(elapsed, 3),
        }

        rows.append(row)

        print(
            f"Found={result.found} | "
            f"Valid={valid_count}/{len(candidates)} | "
            f"Eligible={eligible_count} | "
            f"Confirmations={len(result.confirmation_attempts)} | "
            f"Time={elapsed:.2f}s",
            flush=True,
        )

    summary_path = output_dir / "summary.csv"

    with summary_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nBenchmark completed: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/plg/targeted_benchmark"),
    )
    parser.add_argument("--max-candidates", type=int, default=5)
    parser.add_argument("--screening-episodes", type=int, default=10)
    parser.add_argument("--confirmation-episodes", type=int, default=30)
    parser.add_argument("--mcts-simulations", type=int, default=10)

    args = parser.parse_args()

    run_benchmark(
        output_dir=args.output_dir,
        max_candidates=args.max_candidates,
        screening_episodes=args.screening_episodes,
        confirmation_episodes=args.confirmation_episodes,
        mcts_simulations=args.mcts_simulations,
    )


if __name__ == "__main__":
    main()
