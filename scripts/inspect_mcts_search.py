from __future__ import annotations

import argparse
import time

import numpy as np

from match3_ai.agents.mcts_simulator import MCTSSimulator
from match3_ai.agents.stochastic_mcts_agent import StochasticMCTSAgent
from match3_ai.games.match3.environment import Match3LevelEnv


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--level",
        type=str,
        default="frozen-garden",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=4,
    )

    parser.add_argument(
        "--simulations",
        type=int,
        default=100,
    )

    parser.add_argument(
        "--value-mode",
        choices=("terminal", "outcome"),
        default="terminal",
    )

    args = parser.parse_args()

    for policy in ("random", "greedy", "objective"):
        env = Match3LevelEnv(level_id=args.level)

        try:
            observation, _ = env.reset(seed=args.seed)

            legal_actions = np.flatnonzero(observation["action_mask"])

            agent = StochasticMCTSAgent(
                simulator=MCTSSimulator(env),
                simulations=args.simulations,
                rollout_policy=policy,
                value_mode=args.value_mode,
                seed=args.seed + 10_000,
            )

            start = time.perf_counter()
            selected_action = agent.select_action(observation)
            elapsed = time.perf_counter() - start

            stats = agent.last_search_stats
            assert stats is not None

            print(f"\n{'=' * 60}")
            print(f"POLICY: {policy.upper()}")
            print(f"VALUE MODE: {args.value_mode.upper()}")
            print(f"{'=' * 60}")

            print(f"Level: {args.level}")
            print(f"Seed: {args.seed}")
            print(f"Simulations: {args.simulations}")
            print(f"Legal actions: {len(legal_actions)}")
            print(f"Root actions explored: {len(stats.action_visits)}")
            print(f"Root visits: {stats.root_visits}")
            print(f"Total nodes: {stats.node_count}")
            print(f"Max tree depth: {stats.max_depth}")
            print(f"Selected action: {selected_action}")
            print(f"Decision time: {elapsed:.2f}s")

            ranked_actions = sorted(
                stats.action_visits,
                key=lambda action: (
                    stats.action_visits[action],
                    stats.action_values[action],
                ),
                reverse=True,
            )

            print("\nTop 10 root actions:")

            for action in ranked_actions[:10]:
                visits = stats.action_visits[action]
                value = stats.action_values[action]

                print(f"  action={action:3d} | visits={visits:4d} | value={value:+.3f}")

        finally:
            env.close()


if __name__ == "__main__":
    main()
