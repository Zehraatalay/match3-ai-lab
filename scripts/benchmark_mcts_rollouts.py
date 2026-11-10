from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np

from match3_ai.agents.mcts_simulator import MCTSSimulator
from match3_ai.agents.stochastic_mcts_agent import StochasticMCTSAgent
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.session import LevelStatus

LEVELS = (
    "double-trouble-v2",
    "wooden-trouble-v2",
    "frozen-garden-v2",
)

POLICIES = ("random", "greedy", "objective")

VALUE_MODES = ("terminal", "outcome")


def run_episode(
    level_id: str,
    policy: str,
    episode_seed: int,
    value_mode: str,
    simulations: int,
) -> dict:
    env = Match3LevelEnv(level_id=level_id)

    try:
        observation, _ = env.reset(seed=episode_seed)

        agent = StochasticMCTSAgent(
            simulator=MCTSSimulator(env),
            simulations=simulations,
            rollout_policy=policy,
            value_mode=value_mode,
            seed=episode_seed + 10_000,
        )

        done = False
        decision_times = []
        search_depths = []
        moves = 0
        final_info = {}

        start_episode = time.perf_counter()

        while not done:
            start_decision = time.perf_counter()

            action = agent.select_action(observation)

            decision_times.append(time.perf_counter() - start_decision)

            stats = agent.last_search_stats

            if stats is not None:
                search_depths.append(stats.max_depth)

            observation, _reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            final_info = info
            moves += 1

        elapsed = time.perf_counter() - start_episode

        objectives = final_info.get("objective_progress", [])
        completion_scores = []

        for objective in objectives:
            target = objective["target"]
            progress = objective["progress"]

            if target > 0:
                completion_scores.append(min(max(progress / target, 0.0), 1.0))

        objective_completion = (
            float(np.mean(completion_scores)) if completion_scores else 0.0
        )

        return {
            "level_id": level_id,
            "policy": policy,
            "value_mode": value_mode,
            "episode_seed": episode_seed,
            "simulations": simulations,
            "won": int(final_info["level_status"] == LevelStatus.WON.value),
            "moves": moves,
            "objective_completion": objective_completion,
            "avg_decision_time": float(np.mean(decision_times)),
            "avg_search_depth": (
                float(np.mean(search_depths)) if search_depths else 0.0
            ),
            "episode_time": elapsed,
        }

    finally:
        env.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark MCTS rollout policies and value functions."
    )

    parser.add_argument(
        "--episodes",
        type=int,
        default=2,
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=20,
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/mcts_value_benchmark.csv"),
    )

    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")

    if args.simulations <= 0:
        parser.error("--simulations must be positive")

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    rows = []

    # --------------------------------------------------
    # Run benchmark
    # --------------------------------------------------

    for level_id in LEVELS:
        for episode_seed in range(args.episodes):
            for policy in POLICIES:
                for value_mode in VALUE_MODES:
                    print(
                        f"Running {level_id} | "
                        f"seed={episode_seed} | "
                        f"policy={policy} | "
                        f"value={value_mode}",
                        flush=True,
                    )

                    result = run_episode(
                        level_id=level_id,
                        policy=policy,
                        episode_seed=episode_seed,
                        value_mode=value_mode,
                        simulations=args.simulations,
                    )

                    rows.append(result)

                    print(
                        f"  won={result['won']} | "
                        f"moves={result['moves']} | "
                        f"completion={result['objective_completion']:.2%} | "
                        f"avg_decision={result['avg_decision_time']:.2f}s | "
                        f"episode={result['episode_time']:.2f}s",
                        flush=True,
                    )

    # --------------------------------------------------
    # Save CSV
    # --------------------------------------------------

    with args.output.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=list(rows[0]),
        )

        writer.writeheader()
        writer.writerows(rows)

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    print("\nBENCHMARK SUMMARY")
    print("-" * 85)

    for level_id in LEVELS:
        print(f"\n{level_id}")

        for policy in POLICIES:
            for value_mode in VALUE_MODES:
                selected = [
                    row
                    for row in rows
                    if row["level_id"] == level_id
                    and row["policy"] == policy
                    and row["value_mode"] == value_mode
                ]

                win_rate = float(np.mean([row["won"] for row in selected]))

                completion = float(
                    np.mean([row["objective_completion"] for row in selected])
                )

                avg_moves = float(np.mean([row["moves"] for row in selected]))

                winning_moves = [row["moves"] for row in selected if row["won"] == 1]

                avg_winning_moves = (
                    float(np.mean(winning_moves)) if winning_moves else float("nan")
                )

                decision_time = float(
                    np.mean([row["avg_decision_time"] for row in selected])
                )

                print(
                    f"  {policy:9s} | "
                    f"{value_mode:8s} | "
                    f"win={win_rate:6.1%} | "
                    f"completion={completion:6.1%} | "
                    f"moves={avg_moves:5.1f} | "
                    f"win_moves={avg_winning_moves:5.1f} | "
                    f"decision={decision_time:6.2f}s"
                )

    print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()
