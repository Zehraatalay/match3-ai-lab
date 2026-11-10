from __future__ import annotations

import math
from dataclasses import dataclass, field
from numbers import Integral, Real

import numpy as np

from match3_ai.agents.base import Observation
from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.mcts_simulator import MCTSSimulator
from match3_ai.agents.mcts_value import MCTSValueFunction
from match3_ai.agents.objective_rollout import ObjectiveRolloutPolicy


@dataclass
class MCTSNode:
    """An action-history node in an open-loop MCTS tree."""

    parent: MCTSNode | None = None
    action: int | None = None
    visits: int = 0
    value_sum: float = 0.0
    children: dict[int, MCTSNode] = field(default_factory=dict)

    @property
    def mean_value(self) -> float:
        if self.visits == 0:
            return 0.0

        return self.value_sum / self.visits

    def uct(
        self,
        parent_visits: int,
        exploration_constant: float,
    ) -> float:
        """Calculate the Upper Confidence Bound for Trees."""

        if self.visits == 0:
            return float("inf")

        exploitation = self.mean_value

        exploration = exploration_constant * math.sqrt(
            math.log(max(parent_visits, 1)) / self.visits
        )

        return exploitation + exploration


@dataclass(frozen=True)
class SearchStats:
    """Diagnostics collected from one MCTS decision."""

    simulations: int
    root_visits: int
    max_depth: int
    node_count: int
    action_visits: dict[int, int]
    action_values: dict[int, float]


class StochasticMCTSAgent:
    """Open-loop MCTS with stochastic future tile generation."""

    def __init__(
        self,
        simulator: MCTSSimulator,
        simulations: int = 100,
        exploration_constant: float = math.sqrt(2),
        rollout_policy: str = "greedy",
        seed: int | None = None,
        value_mode: str = "terminal",
    ) -> None:
        if (
            isinstance(simulations, (bool, np.bool_))
            or not isinstance(simulations, Integral)
            or simulations <= 0
        ):
            raise ValueError(
                "simulations must be a positive integer, excluding booleans"
            )

        if (
            isinstance(exploration_constant, (bool, np.bool_))
            or not isinstance(exploration_constant, Real)
            or not math.isfinite(exploration_constant)
            or exploration_constant < 0
        ):
            raise ValueError("exploration_constant must be finite and non-negative")

        if rollout_policy not in {"random", "greedy", "objective"}:
            raise ValueError(
                "rollout_policy must be 'random', 'greedy', or 'objective'"
            )

        if value_mode not in {"terminal", "outcome"}:
            raise ValueError("value_mode must be 'terminal' or 'outcome'")

        self.simulator = simulator
        self.simulations = int(simulations)
        self.exploration_constant = exploration_constant
        self.rollout_policy = rollout_policy
        self.value_mode = value_mode

        self.rng = np.random.default_rng(seed)
        self.greedy_agent = GreedyAgent()

        self.objective_policy = ObjectiveRolloutPolicy(
            max_candidates=5,
            seed=seed,
        )

        self.value_function = MCTSValueFunction()

        self.last_search_stats: SearchStats | None = None

    @staticmethod
    def _count_nodes(node: MCTSNode) -> int:
        """Count all nodes in the search tree."""

        return 1 + sum(
            StochasticMCTSAgent._count_nodes(child) for child in node.children.values()
        )

    def select_action(self, observation: Observation) -> int:
        """Select an action using stochastic open-loop MCTS."""

        legal_actions = [
            int(action) for action in np.flatnonzero(observation["action_mask"])
        ]

        if not legal_actions:
            raise RuntimeError("No legal actions available.")

        self.last_search_stats = None

        root = MCTSNode()
        max_depth = 0

        for _ in range(self.simulations):
            # Independent RNG stream for stochastic tile generation
            simulation_seed = int(self.rng.integers(0, 2**32, dtype=np.uint64))

            env = self.simulator.clone_with_seed(simulation_seed)

            node = root
            path = [root]

            done = False
            reward = 0.0

            try:
                simulation_observation = env._observation()
                # -----------------------------------------
                # 1. Selection and Expansion
                # -----------------------------------------

                while not done:
                    available = [
                        int(action)
                        for action in np.flatnonzero(
                            simulation_observation["action_mask"]
                        )
                    ]

                    if not available:
                        reward = -1.0
                        done = True
                        break

                    unexplored = [
                        action for action in available if action not in node.children
                    ]

                    if unexplored:
                        action = int(self.rng.choice(unexplored))

                        child = MCTSNode(
                            parent=node,
                            action=action,
                        )

                        node.children[action] = child

                        simulation_observation, reward, terminated, truncated, _ = (
                            env.step(action)
                        )

                        done = terminated or truncated

                        node = child
                        path.append(node)

                        max_depth = max(
                            max_depth,
                            len(path) - 1,
                        )

                        break

                    # UCT selection
                    action = max(
                        available,
                        key=lambda candidate: node.children[candidate].uct(
                            node.visits,
                            self.exploration_constant,
                        ),
                    )

                    node = node.children[action]
                    path.append(node)

                    max_depth = max(
                        max_depth,
                        len(path) - 1,
                    )

                    simulation_observation, reward, terminated, truncated, _ = env.step(
                        action
                    )

                    done = terminated or truncated

                # -----------------------------------------
                # 2. Simulation (Policy-guided Rollout)
                # -----------------------------------------

                while not done:
                    available = [
                        int(action)
                        for action in np.flatnonzero(
                            simulation_observation["action_mask"]
                        )
                    ]

                    if not available:
                        reward = -1.0
                        break

                    if self.rollout_policy == "greedy":
                        action = self.greedy_agent.select_action(simulation_observation)

                    elif self.rollout_policy == "objective":
                        action = self.objective_policy.select_action(env)

                    else:
                        action = int(self.rng.choice(available))

                    simulation_observation, reward, terminated, truncated, _ = env.step(
                        action
                    )

                    done = terminated or truncated

                # -----------------------------------------
                # 3. Backpropagation
                # -----------------------------------------

                if self.value_mode == "outcome":
                    simulation_value = self.value_function.evaluate(
                        env,
                        terminal_reward=float(reward),
                    )

                else:
                    simulation_value = float(reward)
                for visited_node in path:
                    visited_node.visits += 1
                    visited_node.value_sum += simulation_value
            finally:
                env.close()

        if not root.children:
            raise RuntimeError("MCTS did not expand any root actions.")

        # -----------------------------------------
        # 4. Search Diagnostics
        # -----------------------------------------

        self.last_search_stats = SearchStats(
            simulations=self.simulations,
            root_visits=root.visits,
            max_depth=max_depth,
            node_count=self._count_nodes(root),
            action_visits={
                action: child.visits for action, child in root.children.items()
            },
            action_values={
                action: child.mean_value for action, child in root.children.items()
            },
        )

        # -----------------------------------------
        # 5. Final Action Selection
        # -----------------------------------------

        best_action = max(
            root.children,
            key=lambda action: (
                root.children[action].visits,
                root.children[action].mean_value,
                -action,
            ),
        )

        return best_action
