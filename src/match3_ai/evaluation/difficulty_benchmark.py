from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from match3_ai.evaluation.difficulty_comparison import compare_agents
from match3_ai.evaluation.difficulty_evaluator import (
    SUPPORTED_AGENTS,
    DifficultyEvaluator,
)
from match3_ai.levels.generation import GenerationSpec, LevelGenerator
from match3_ai.levels.validation import LevelValidator


@dataclass(frozen=True, slots=True)
class BenchmarkConfig:
    level_count: int = 20
    max_candidates: int = 200
    episodes_per_agent: int = 100

    generation_seed: int = 42
    evaluation_seed: int = 1000
    bootstrap_seed: int = 42
    bootstrap_samples: int = 5000

    agents: tuple[str, ...] = ("Random", "Greedy")
    mcts_simulations: int = 30


def run_difficulty_benchmark(
    *,
    config: BenchmarkConfig,
    output_dir: Path,
) -> None:
    # --------------------------------------------------
    # 1. Configuration validation
    # --------------------------------------------------

    for name in (
        "level_count",
        "max_candidates",
        "episodes_per_agent",
        "bootstrap_samples",
        "mcts_simulations",
    ):
        value = getattr(config, name)

        if type(value) is not int or value <= 0:
            raise ValueError(f"{name} must be a positive integer")

    if config.max_candidates < config.level_count:
        raise ValueError("max_candidates must be >= level_count")

    if not isinstance(config.agents, tuple) or not config.agents:
        raise ValueError("agents must be a nonempty tuple")

    if len(set(config.agents)) != len(config.agents):
        raise ValueError("agents must be unique")

    if any(agent not in SUPPORTED_AGENTS for agent in config.agents):
        raise ValueError("unsupported benchmark agent")

    if "Random" not in config.agents or "Greedy" not in config.agents:
        raise ValueError(
            "Random and Greedy are required for paired benchmark comparisons"
        )

    # --------------------------------------------------
    # 2. Generation and evaluation setup
    # --------------------------------------------------

    spec = GenerationSpec(
        rows=8,
        cols=8,
        num_colors=5,
        min_moves=8,
        max_moves=20,
        min_blockers=2,
        max_blockers=8,
        seed=config.generation_seed,
    )

    generator = LevelGenerator(spec)
    validator = LevelValidator()

    evaluator = DifficultyEvaluator(
        episodes_per_agent=config.episodes_per_agent,
        seed=config.evaluation_seed,
        agents=config.agents,
        mcts_simulations=config.mcts_simulations,
    )

    summary_rows = []
    episode_rows = []
    agent_metric_rows = []
    comparison_rows = []

    accepted_count = 0
    rejected_count = 0
    attempted_count = 0

    # --------------------------------------------------
    # 3. Generate and evaluate valid levels
    # --------------------------------------------------

    for candidate_id in range(config.max_candidates):
        if accepted_count >= config.level_count:
            break

        attempted_count += 1

        level = generator.generate(candidate_id)
        validation = validator.validate(level)

        if not validation.valid:
            rejected_count += 1
            continue

        report = evaluator.evaluate(level)

        # Fail fast if evaluation did not complete reliably.
        for metric in report.agent_metrics:
            if metric.errors or metric.time_limits:
                raise RuntimeError(
                    f"Evaluation failed for {level.level_id}, "
                    f"agent={metric.agent_name}: "
                    f"errors={metric.errors}, "
                    f"time_limits={metric.time_limits}"
                )

        metrics = {metric.agent_name: metric for metric in report.agent_metrics}

        random = metrics["Random"]
        greedy = metrics["Greedy"]

        # --------------------------------------------------
        # 4. Paired comparisons
        # --------------------------------------------------

        agent_pairs = [("Random", "Greedy")]

        if "MCTS_greedy_outcome" in config.agents:
            agent_pairs.extend(
                [
                    ("Random", "MCTS_greedy_outcome"),
                    ("Greedy", "MCTS_greedy_outcome"),
                ]
            )

        comparisons = {}

        for agent_a, agent_b in agent_pairs:
            comparison = compare_agents(
                report,
                agent_a=agent_a,
                agent_b=agent_b,
                bootstrap_samples=config.bootstrap_samples,
                seed=config.bootstrap_seed,
            )

            comparisons[(agent_a, agent_b)] = comparison

            comparison_rows.append(
                {
                    "level_id": level.level_id,
                    "candidate_id": candidate_id,
                    "agent_a": comparison.agent_a,
                    "agent_b": comparison.agent_b,
                    "paired_episodes": comparison.paired_episodes,
                    "win_rate_a": comparison.win_rate_a,
                    "win_rate_b": comparison.win_rate_b,
                    "difference": comparison.difference,
                    "ci_lower": comparison.ci_lower,
                    "ci_upper": comparison.ci_upper,
                }
            )

        baseline_comparison = comparisons[("Random", "Greedy")]

        # --------------------------------------------------
        # 5. Legacy summary (Random vs Greedy)
        # --------------------------------------------------

        summary_rows.append(
            {
                "level_id": level.level_id,
                "candidate_id": candidate_id,
                "move_limit": level.move_limit,
                "blocker_count": len(level.blockers),
                "random_win_rate": random.win_rate,
                "greedy_win_rate": greedy.win_rate,
                "win_rate_difference": baseline_comparison.difference,
                "difference_ci_lower": baseline_comparison.ci_lower,
                "difference_ci_upper": baseline_comparison.ci_upper,
                "random_mean_moves": random.mean_moves_used,
                "greedy_mean_moves": greedy.mean_moves_used,
                "random_mean_completion": random.mean_objective_completion,
                "greedy_mean_completion": greedy.mean_objective_completion,
                "random_errors": random.errors,
                "greedy_errors": greedy.errors,
                "random_time_limits": random.time_limits,
                "greedy_time_limits": greedy.time_limits,
            }
        )

        # --------------------------------------------------
        # 6. Agent-level metrics
        # --------------------------------------------------

        for metric in report.agent_metrics:
            agent_metric_rows.append(
                {
                    "level_id": level.level_id,
                    "candidate_id": candidate_id,
                    "move_limit": level.move_limit,
                    "blocker_count": len(level.blockers),
                    **asdict(metric),
                }
            )

        # --------------------------------------------------
        # 7. Episode-level records
        # --------------------------------------------------

        for episode in report.episodes:
            episode_rows.append(
                {
                    "level_id": level.level_id,
                    "candidate_id": candidate_id,
                    "agent_name": episode.agent_name,
                    "episode_seed": episode.seed,
                    "agent_seed": episode.agent_seed,
                    "outcome": episode.outcome,
                    "moves_used": episode.moves_used,
                    "objective_completion_ratio": (episode.objective_completion_ratio),
                    "board_seed": episode.board_seed,
                    "engine_seed": episode.engine_seed,
                    "level_config_id": episode.level_config_id,
                }
            )

        accepted_count += 1

        agent_results = " | ".join(
            f"{metric.agent_name}={metric.win_rate:.1%}"
            for metric in report.agent_metrics
        )

        print(
            f"[{accepted_count}/{config.level_count}] "
            f"{level.level_id} | {agent_results}"
        )

    # --------------------------------------------------
    # 8. Validate candidate budget
    # --------------------------------------------------

    if accepted_count < config.level_count:
        raise RuntimeError(
            f"Only {accepted_count}/{config.level_count} valid levels "
            f"found after {attempted_count} candidates"
        )

    # --------------------------------------------------
    # 9. Export benchmark results
    # --------------------------------------------------

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _write_csv(output_dir / "summary.csv", summary_rows)
    _write_csv(output_dir / "episodes.csv", episode_rows)
    _write_csv(output_dir / "agent_metrics.csv", agent_metric_rows)
    _write_csv(output_dir / "paired_comparisons.csv", comparison_rows)

    metadata = {
        "benchmark_config": asdict(config),
        "generation_spec": asdict(spec),
        "agents": list(config.agents),
        "mcts_configuration": (
            {
                "agent": "MCTS_greedy_outcome",
                "simulations": config.mcts_simulations,
                "rollout_policy": "greedy",
                "value_mode": "outcome",
                "exploration_constant": 2**0.5,
            }
            if "MCTS_greedy_outcome" in config.agents
            else None
        ),
        "level_count": len(summary_rows),
        "total_episodes": len(episode_rows),
        "attempted_candidates": attempted_count,
        "accepted_levels": accepted_count,
        "rejected_candidates": rejected_count,
        "acceptance_rate": accepted_count / attempted_count,
    }

    (output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )
        writer.writeheader()
        writer.writerows(rows)
