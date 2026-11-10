"""Deterministic open-loop search accounting and pre-optimization equivalence."""

import copy
import json
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from match3_ai.agents.mcts_simulator import MCTSSimulator
from match3_ai.agents.mcts_value import MCTSValueFunction
from match3_ai.agents.stochastic_mcts_agent import MCTSNode, StochasticMCTSAgent
from match3_ai.evaluation.behavior_data import complete_state_fingerprint
from match3_ai.games.match3.environment import Match3LevelEnv


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        0,
        -1,
        1.5,
        1.0,
        "30",
        None,
        float("nan"),
        float("inf"),
        [],
        {},
    ],
)
def test_simulation_budget_validation(value):
    with pytest.raises(ValueError, match="positive integer"):
        StochasticMCTSAgent(None, simulations=value)


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
        np.bool_(True),
        -1,
        float("nan"),
        float("inf"),
        -float("inf"),
        "1",
        None,
        [],
        {},
    ],
)
def test_exploration_validation(value):
    with pytest.raises(ValueError, match="finite and non-negative"):
        StochasticMCTSAgent(None, exploration_constant=value)


def test_valid_numeric_configuration_preserved():
    agent = StochasticMCTSAgent(None, simulations=np.int64(5), exploration_constant=0)
    assert agent.simulations == 5 and agent.exploration_constant == 0
    assert StochasticMCTSAgent(None).exploration_constant == pytest.approx(np.sqrt(2))


class ToyEnv:
    def __init__(
        self, seed, *, one_step=False, rewards=(1.0, -1.0), root_actions=(0, 1)
    ):
        self.branch = seed % 2
        self.one_step, self.rewards, self.root_actions = one_step, rewards, root_actions
        self.depth, self.first = 0, None
        self.actions = []
        self.closed = False
        self.config = SimpleNamespace(move_limit=2)
        self.session = SimpleNamespace(moves_remaining=2)

    def _observation(self):
        mask = np.zeros(144, dtype=np.int8)
        if self.depth == 0:
            mask[list(self.root_actions)] = 1
        elif self.depth == 1 and not self.one_step:
            mask[2 + self.branch] = 1
        return {"action_mask": mask}

    def _objective_info(self):
        return [{"progress": 1, "target": 2}]

    def step(self, action):
        assert self._observation()["action_mask"][action], (
            "illegal or post-terminal action"
        )
        if self.depth == 0:
            self.first = action
        self.depth += 1
        self.session.moves_remaining -= 1
        self.actions.append(action)
        terminal = self.one_step or self.depth == 2
        reward = self.rewards[self.first] if terminal else 0.0
        return self._observation(), reward, terminal, False, {}

    def close(self):
        self.closed = True


class ToySimulator:
    def __init__(self, **options):
        self.options, self.clones = options, []

    def clone_with_seed(self, seed):
        env = ToyEnv(seed, **self.options)
        self.clones.append(env)
        return env


def test_uct_formula_and_zero_visit_handling():
    node = MCTSNode(visits=4, value_sum=2)
    assert node.uct(16, 2) == pytest.approx(0.5 + 2 * np.sqrt(np.log(16) / 4))
    assert MCTSNode().uct(0, 0) == float("inf")


def test_dynamic_actions_expansion_and_exact_backups():
    simulator = ToySimulator()
    agent = StochasticMCTSAgent(
        simulator, simulations=40, seed=42, rollout_policy="random"
    )
    action = agent.select_action(ToyEnv(0)._observation())
    stats = agent.last_search_stats
    assert stats.root_visits == sum(stats.action_visits.values()) == 40
    assert stats.action_values == {0: 1.0, 1: -1.0}
    assert action == 0 and stats.action_visits[0] > stats.action_visits[1]
    assert stats.max_depth == 2 and stats.node_count == 6
    assert {clone.branch for clone in simulator.clones} == {0, 1}
    assert all(clone.closed and len(clone.actions) == 2 for clone in simulator.clones)


def test_uct_ties_and_final_action_ties_choose_lowest_legal_action():
    for simulations in (2, 3):
        simulator = ToySimulator(one_step=True, rewards=(0.0, 0.0))
        agent = StochasticMCTSAgent(simulator, simulations=simulations, seed=42)
        assert agent.select_action(ToyEnv(0)._observation()) == 0
        stats = agent.last_search_stats
        assert stats.action_visits[0] == (1 if simulations == 2 else 2)
        assert stats.action_visits[1] == 1


@pytest.mark.parametrize(
    "mode,values", [("terminal", {0: 1.0, 1: -1.0}), ("outcome", {0: 1.1, 1: -0.9})]
)
def test_terminal_states_stop_and_value_perspective(mode, values):
    simulator = ToySimulator(one_step=True)
    agent = StochasticMCTSAgent(simulator, simulations=20, seed=42, value_mode=mode)
    agent.select_action(ToyEnv(0)._observation())
    assert agent.last_search_stats.action_values == pytest.approx(values)
    assert all(clone.closed and len(clone.actions) == 1 for clone in simulator.clones)


@pytest.mark.parametrize("policy", ["random", "greedy"])
def test_rollout_policy_delegation(policy):
    simulator = ToySimulator(root_actions=(0,))
    agent = StochasticMCTSAgent(
        simulator, simulations=1, seed=42, rollout_policy=policy
    )
    calls = []

    class Greedy:
        def select_action(self, observation):
            calls.append(observation)
            return int(np.flatnonzero(observation["action_mask"])[0])

    agent.greedy_agent = Greedy()
    agent.select_action(ToyEnv(0, root_actions=(0,))._observation())
    assert len(calls) == (1 if policy == "greedy" else 0)
    assert len(simulator.clones[0].actions) == 2


def test_outcome_value_ranges_and_exact_objective_mean():
    env = SimpleNamespace(
        config=SimpleNamespace(move_limit=10),
        session=SimpleNamespace(moves_remaining=3),
        _objective_info=lambda: [
            {"progress": 1, "target": 2},
            {"progress": 2, "target": 2},
        ],
    )
    value = MCTSValueFunction()
    assert value.objective_completion(env) == 0.75
    assert value.evaluate(env, 1) == pytest.approx(1.06)
    assert value.evaluate(env, -1) == pytest.approx(-0.85)
    assert value.evaluate(env, 0) == 0
    env.session.moves_remaining = 0
    assert value.evaluate(env, 1) == 1
    env.session.moves_remaining = 10
    assert value.evaluate(env, 1) == 1.2


@pytest.mark.parametrize("policy", ["random", "greedy"])
@pytest.mark.parametrize("mode", ["terminal", "outcome"])
def test_search_cannot_reveal_real_future_rng(policy, mode):
    live = Match3LevelEnv("double-trouble-v2")
    observation, _ = live.reset(seed=10001000)
    results = []
    try:
        for hidden_seed in (1, 999999):
            env = copy.deepcopy(live)
            try:
                env.session.engine.rng = np.random.default_rng(hidden_seed)
                before = complete_state_fingerprint(env)
                agent = StochasticMCTSAgent(
                    MCTSSimulator(env),
                    simulations=3,
                    seed=777,
                    rollout_policy=policy,
                    value_mode=mode,
                )
                results.append(
                    (agent.select_action(observation), asdict(agent.last_search_stats))
                )
                assert complete_state_fingerprint(env) == before
            finally:
                env.close()
        assert results[0] == results[1]
    finally:
        live.close()


def test_seeded_clones_are_independent():
    live = Match3LevelEnv("double-trouble-v2")
    live.reset(seed=10001000)
    before = complete_state_fingerprint(live)
    simulator = MCTSSimulator(live)
    first, second = simulator.clone_with_seed(7), simulator.clone_with_seed(7)
    try:
        assert complete_state_fingerprint(first) == complete_state_fingerprint(second)
        second_before = complete_state_fingerprint(second)
        first.session.engine.rng.random()
        first.session.moves_remaining -= 1
        assert complete_state_fingerprint(second) == second_before
        assert complete_state_fingerprint(live) == before
    finally:
        first.close()
        second.close()
        live.close()


def test_simulation_exception_closes_clone():
    class BrokenEnv(ToyEnv):
        def step(self, action):
            raise RuntimeError("simulated step failed")

    class BrokenSimulator:
        def __init__(self):
            self.env = BrokenEnv(0)

        def clone_with_seed(self, seed):
            return self.env

    simulator = BrokenSimulator()
    agent = StochasticMCTSAgent(simulator, simulations=1, seed=42)
    with pytest.raises(RuntimeError, match="simulated step failed"):
        agent.select_action(simulator.env._observation())
    assert simulator.env.closed


def test_truncation_stops_rollout_with_neutral_terminal_reward():
    class TruncatedEnv(ToyEnv):
        def step(self, action):
            observation, _, _, _, info = super().step(action)
            return observation, 0.0, False, True, info

    class Simulator:
        def __init__(self):
            self.clones = []

        def clone_with_seed(self, seed):
            env = TruncatedEnv(seed, one_step=True)
            self.clones.append(env)
            return env

    simulator = Simulator()
    agent = StochasticMCTSAgent(simulator, simulations=4, seed=42)
    agent.select_action(ToyEnv(0)._observation())
    assert all(value == 0 for value in agent.last_search_stats.action_values.values())
    assert all(clone.closed and len(clone.actions) == 1 for clone in simulator.clones)


REFERENCE = json.loads(
    (Path(__file__).parent / "fixtures/mcts_sprint3_reference.json").read_text()
)


@pytest.mark.parametrize(
    "case",
    REFERENCE["cases"],
    ids=lambda case: f"{case['level']}-{case['rollout_policy']}-{case['value_mode']}",
)
def test_preoptimization_reference_actions_and_search_statistics(case):
    env = Match3LevelEnv(case["level"])
    try:
        observation, _ = env.reset(seed=case["env_seed"])
        agent = StochasticMCTSAgent(
            MCTSSimulator(env),
            simulations=case["simulations"],
            seed=case["agent_seed"],
            rollout_policy=case["rollout_policy"],
            value_mode=case["value_mode"],
        )
        for expected in case["trace"]:
            before = complete_state_fingerprint(env)
            action = agent.select_action(observation)
            assert complete_state_fingerprint(env) == before
            assert (
                json.loads(
                    json.dumps(
                        {"action": action, "stats": asdict(agent.last_search_stats)}
                    )
                )
                == expected
            )
            observation, *_ = env.step(action)
    finally:
        env.close()
