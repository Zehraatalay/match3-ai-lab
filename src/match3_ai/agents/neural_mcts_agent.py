"""Single-player stochastic PUCT using the exact simulator and AZ-1 network.

Decision nodes describe observable states, not action histories. Each action
edge owns separate sampled successors. The simulator samples chance outcomes;
search never selects a favorable outcome. Edge values average sampled returns.
New nonterminal successors use neural leaf values, without rollout policies.
Backups use undiscounted sparse terminal rewards without player sign changes.
Trees and neural evaluations are local to one decision and one network snapshot.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import astuple, dataclass, field
from numbers import Integral, Real
from time import perf_counter

import numpy as np
import torch
from numpy.typing import NDArray

from match3_ai.agents.base import Observation
from match3_ai.agents.mcts_simulator import MCTSSimulator
from match3_ai.agents.policy_value_encoder import PolicyValueContext, PolicyValueEncoder
from match3_ai.agents.policy_value_network import (
    PolicyValueNetwork,
    masked_policy_logits,
)

OBSERVABLE_FIELDS = (
    "tiles",
    "specials",
    "blockers",
    "blocker_hp",
    "cell_mask",
    "board_shape",
    "moves_remaining",
    "objective_type",
    "objective_color",
    "objective_progress",
    "objective_target",
    "objective_mask",
    "action_mask",
)


@dataclass
class SuccessorStats:
    node: DecisionNode | None
    visits: int = 0
    value_sum: float = 0.0


@dataclass
class ActionStats:
    prior: float
    visits: int = 0
    value_sum: float = 0.0
    successors: dict[tuple[str, bool, float], SuccessorStats] = field(
        default_factory=dict
    )

    @property
    def mean_value(self) -> float:
        return self.value_sum / self.visits if self.visits else 0.0

    def puct(self, parent_visits: int, c_puct: float) -> float:
        """Q + c_puct * P * sqrt(1 + N(parent)) / (1 + N(edge))."""
        return self.mean_value + c_puct * self.prior * math.sqrt(1 + parent_visits) / (
            1 + self.visits
        )


@dataclass
class DecisionNode:
    actions: dict[int, ActionStats]
    visits: int = 0
    value_sum: float = 0.0


@dataclass(frozen=True)
class NeuralSearchStats:
    simulations: int
    root_visits: int
    action_visits: dict[int, int]
    action_values: dict[int, float]
    action_value_sums: dict[int, float]
    root_priors: dict[int, float]
    policy_target: NDArray[np.float64]
    neural_evaluations: int
    sampled_transitions: int
    unique_successor_states: int
    root_successor_counts: dict[int, int]
    max_depth: int
    elapsed_seconds: float


class NeuralMCTSAgent:
    """Environment-bound policy-value search implementing select_action.

    Evaluation ignores noise/temperature and resolves visit ties by lowest ID.
    Exploration optionally mixes legal root priors with Dirichlet noise and
    samples visits**(1/temperature); temperature zero uses evaluation selection.
    Policy targets always contain untempered normalized visits, including zeros.
    External truncations are errors, not terminal outcomes. The network must
    remain fixed during a search; inference temporarily disables training mode.
    """

    def __init__(
        self,
        simulator: MCTSSimulator,
        network: PolicyValueNetwork,
        context: PolicyValueContext,
        *,
        simulations: int = 30,
        c_puct: float = 1.5,
        seed: int = 0,
        evaluation: bool = True,
        exploration_seed: int | None = None,
        temperature: float = 1.0,
        dirichlet_alpha: float = 0.3,
        dirichlet_epsilon: float = 0.0,
    ) -> None:
        if (
            isinstance(simulations, (bool, np.bool_))
            or not isinstance(simulations, Integral)
            or simulations <= 0
        ):
            raise ValueError(
                "simulations must be a positive integer excluding booleans"
            )
        for name, value in (
            ("c_puct", c_puct),
            ("temperature", temperature),
            ("dirichlet_alpha", dirichlet_alpha),
            ("dirichlet_epsilon", dirichlet_epsilon),
        ):
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be a finite non-negative real")
        if dirichlet_alpha == 0 or dirichlet_epsilon > 1:
            raise ValueError("dirichlet_alpha must be positive and epsilon at most one")
        if type(evaluation) is not bool:
            raise ValueError("evaluation must be a boolean")
        for name, value in (("seed", seed), ("exploration_seed", exploration_seed)):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError(f"{name} must be a non-negative integer")
        if seed is None:
            raise ValueError("seed must be a non-negative integer")
        if not isinstance(context, PolicyValueContext):
            raise TypeError("context must be a PolicyValueContext")
        if PolicyValueContext.from_level_config(simulator.env.config) != context:
            raise ValueError("context does not match simulator level configuration")
        self.simulator, self.network, self.context = simulator, network, context
        self.encoder = PolicyValueEncoder()
        self.simulations, self.c_puct = int(simulations), float(c_puct)
        self.evaluation, self.temperature = evaluation, float(temperature)
        self.dirichlet_alpha, self.dirichlet_epsilon = (
            float(dirichlet_alpha),
            float(dirichlet_epsilon),
        )
        simulation_stream, exploration_stream = np.random.SeedSequence(seed).spawn(2)
        self.simulation_rng = np.random.default_rng(simulation_stream)
        self.exploration_rng = np.random.default_rng(
            exploration_stream if exploration_seed is None else exploration_seed
        )
        self.last_search_stats: NeuralSearchStats | None = None
        self.last_root: DecisionNode | None = None

    def _state_key(self, observation: Observation) -> str:
        """Hash raw observable counts/context, excluding identity and all RNGs.

        Raw progress is retained even when the neural encoding clips completion.
        Canonical int64 bytes make identity independent of incoming array dtype.
        """
        self.encoder.encode(observation, self.context)
        digest = hashlib.sha256(str(astuple(self.context)).encode())
        for name in OBSERVABLE_FIELDS:
            digest.update(name.encode())
            digest.update(np.asarray(observation[name], dtype="<i8").tobytes())
        return digest.hexdigest()

    def _evaluate(self, observation: Observation) -> tuple[DecisionNode, float]:
        encoded = self.encoder.encode(observation, self.context)
        if not encoded.action_mask.any():
            raise RuntimeError("Nonterminal state has no legal actions")
        parameter = next(self.network.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        dtype = parameter.dtype if parameter is not None else torch.float32
        spatial = torch.from_numpy(encoded.spatial[None]).to(device=device, dtype=dtype)
        globals_ = torch.from_numpy(encoded.global_features[None]).to(
            device=device, dtype=dtype
        )
        logits, values = self.network(spatial, globals_)
        if logits.shape != (1, 144) or values.shape != (1,):
            raise ValueError("network must return logits (1,144) and values (1,)")
        value = float(values.item())
        if not math.isfinite(value) or not -1 <= value <= 1:
            raise ValueError("neural value must be finite and in [-1,1]")
        logits = masked_policy_logits(
            logits.cpu().double(), torch.from_numpy(encoded.action_mask[None])
        )
        priors = logits.softmax(-1)[0].numpy()
        return DecisionNode(
            {
                int(a): ActionStats(float(priors[a]))
                for a in np.flatnonzero(encoded.action_mask)
            }
        ), value

    @staticmethod
    def _choose_action(node: DecisionNode, c_puct: float) -> int:
        return max(
            sorted(node.actions),
            key=lambda a: node.actions[a].puct(node.visits, c_puct),
        )

    def _root_action(self, root: DecisionNode) -> int:
        visited = sorted(a for a, edge in root.actions.items() if edge.visits)
        if not visited:
            raise RuntimeError("Search produced no root visits")
        if self.evaluation or self.temperature == 0:
            return max(visited, key=lambda a: root.actions[a].visits)
        log_counts = np.log([root.actions[a].visits for a in visited])
        weights = np.exp((log_counts - log_counts.max()) / self.temperature)
        return int(self.exploration_rng.choice(visited, p=weights / weights.sum()))

    def select_action(self, observation: Observation) -> int:
        self.last_search_stats = None
        self.last_root = None
        started = perf_counter()
        if (
            PolicyValueContext.from_level_config(self.simulator.env.config)
            != self.context
        ):
            raise ValueError("simulator context changed")
        if self.simulator.env._episode_done:
            raise RuntimeError("Cannot search a terminal live environment")
        root_key = self._state_key(observation)
        if root_key != self._state_key(self.simulator.env._observation()):
            raise ValueError("observation does not match the live simulator state")
        modes = [(module, module.training) for module in self.network.modules()]
        try:
            self.network.eval()
            with torch.inference_mode():
                return self._search(observation, root_key, started)
        finally:
            for module, training in modes:
                module.training = training

    def _search(self, observation: Observation, root_key: str, started: float) -> int:
        root, _ = self._evaluate(observation)
        if not self.evaluation and self.dirichlet_epsilon:
            noise = self.exploration_rng.dirichlet(
                [self.dirichlet_alpha] * len(root.actions)
            )
            for edge, component in zip(root.actions.values(), noise):
                edge.prior = (
                    1 - self.dirichlet_epsilon
                ) * edge.prior + self.dirichlet_epsilon * float(component)
        evaluations, transitions, max_depth = 1, 0, 0
        unique_states: set[tuple[str, bool, float]] = set()
        for _ in range(self.simulations):
            seed = int(self.simulation_rng.integers(0, 2**32, dtype=np.uint64))
            env = self.simulator.clone_with_seed(seed)
            try:
                if self._state_key(env._observation()) != root_key:
                    raise ValueError("simulation clone does not match the root state")
                node = root
                nodes, edges, outcomes = [root], [], []
                depth = 0
                while True:
                    action = self._choose_action(node, self.c_puct)
                    edge = node.actions[action]
                    after, reward, terminated, truncated, info = env.step(action)
                    transitions += 1
                    depth += 1
                    max_depth = max(max_depth, depth)
                    if info.get("valid_action") is False:
                        raise RuntimeError("Simulator rejected a masked legal action")
                    if truncated and not terminated:
                        raise RuntimeError(
                            "External truncation has no terminal outcome"
                        )
                    if reward not in (-1, 0, 1) or (not terminated and reward != 0):
                        raise ValueError(
                            "Search requires sparse terminal rewards in [-1,1]"
                        )
                    if depth > self.context.move_limit:
                        raise RuntimeError(
                            "Simulation exceeded the configured move limit"
                        )
                    key = (self._state_key(after), bool(terminated), float(reward))
                    unique_states.add(key)
                    successor = edge.successors.get(key)
                    is_new = successor is None
                    if is_new:
                        if terminated:
                            child, value = None, float(reward)
                        else:
                            child, value = self._evaluate(after)
                            evaluations += 1
                        successor = SuccessorStats(child)
                        edge.successors[key] = successor
                    edges.append(edge)
                    outcomes.append(successor)
                    if terminated:
                        value = float(reward)
                        break
                    node = successor.node
                    assert node is not None
                    nodes.append(node)
                    if is_new:
                        break
                for visited in (*nodes, *edges, *outcomes):
                    visited.visits += 1
                    visited.value_sum += value
            finally:
                env.close()
        action = self._root_action(root)
        target = np.zeros(144, dtype=np.float64)
        for a, edge in root.actions.items():
            target[a] = edge.visits / root.visits
        target.setflags(write=False)
        self.last_root = root
        self.last_search_stats = NeuralSearchStats(
            self.simulations,
            root.visits,
            {a: e.visits for a, e in root.actions.items()},
            {a: e.mean_value for a, e in root.actions.items()},
            {a: e.value_sum for a, e in root.actions.items()},
            {a: e.prior for a, e in root.actions.items()},
            target,
            evaluations,
            transitions,
            len(unique_states),
            {a: len(e.successors) for a, e in root.actions.items()},
            max_depth,
            perf_counter() - started,
        )
        return action
