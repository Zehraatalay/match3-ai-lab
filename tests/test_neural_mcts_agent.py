from copy import deepcopy
from dataclasses import replace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from match3_ai.agents.mcts_simulator import MCTSSimulator
from match3_ai.agents.neural_mcts_agent import (
    ActionStats,
    DecisionNode,
    NeuralMCTSAgent,
)
from match3_ai.agents.policy_value_encoder import PolicyValueContext
from match3_ai.agents.policy_value_network import PolicyValueNetwork
from match3_ai.evaluation.behavior_data import complete_state_fingerprint
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.config import LevelConfig, LevelObjective, ObjectiveType


class ControlledEnv:
    def __init__(
        self,
        seed=0,
        *,
        root_actions=(0, 1),
        horizon=2,
        stochastic=False,
        rewards=(1.0, -1.0),
        chance_rewards=None,
        truncate=False,
        fail=False,
    ):
        self.config = LevelConfig(
            rows=3,
            cols=3,
            num_colors=3,
            move_limit=4,
            objectives=(LevelObjective(ObjectiveType.COLLECT_TILES, target=9),),
        )
        self.root_actions, self.horizon = root_actions, horizon
        self.branch = seed % 2 if stochastic else 0
        self.rewards, self.truncate, self.fail = rewards, truncate, fail
        self.chance_rewards = chance_rewards
        self.depth, self.first = 0, None
        self._episode_done, self.closed = False, False
        self.actions = []

    def _observation(self):
        observation = {
            key: np.zeros((9, 8), dtype=np.uint8)
            for key in ("tiles", "specials", "blockers", "blocker_hp", "cell_mask")
        }
        observation["tiles"][:3, :3] = 1
        observation["cell_mask"][:3, :3] = 1
        if self.depth:
            observation["tiles"][0, 0] = 1 + self.branch
        mask = np.zeros(144, dtype=np.int8)
        if not self._episode_done:
            mask[list(self.root_actions) if not self.depth else [16 + self.branch]] = 1
        observation.update(
            board_shape=np.array([3, 3]),
            moves_remaining=4 - self.depth,
            objective_type=np.array([2, 0]),
            objective_color=np.array([0, 0]),
            objective_target=np.array([9, 0]),
            objective_progress=np.array([self.depth, 0]),
            objective_mask=np.array([1, 0]),
            action_mask=mask,
        )
        return observation

    def step(self, action):
        if self.fail:
            raise RuntimeError("controlled failure")
        assert self._observation()["action_mask"][action], "illegal action"
        if not self.depth:
            self.first = action
        self.actions.append(action)
        self.depth += 1
        self._episode_done = self.depth == self.horizon
        reward = self.rewards[self.first] if self._episode_done else 0.0
        if self._episode_done and self.chance_rewards is not None:
            reward = self.chance_rewards[self.branch]
        return (
            self._observation(),
            reward,
            self._episode_done and not self.truncate,
            self._episode_done and self.truncate,
            {"valid_action": True},
        )

    def close(self):
        self.closed = True


class ControlledSimulator:
    def __init__(self, **options):
        self.options, self.clones, self.seeds = options, [], []
        self.env = ControlledEnv(**options)

    def clone_with_seed(self, seed):
        self.seeds.append(seed)
        clone = ControlledEnv(seed, **self.options)
        self.clones.append(clone)
        return clone


class FixedNetwork(torch.nn.Module):
    def __init__(self, value=0.4):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.tensor(0.0))
        self.value, self.calls = value, 0

    def forward(self, spatial, globals_):
        assert not torch.is_grad_enabled() and not self.training
        self.calls += 1
        logits = torch.zeros((1, 144))
        logits[0, 143] = 1000
        return logits, torch.tensor([self.value])


def make_agent(simulator=None, network=None, **kwargs):
    simulator = simulator or ControlledSimulator()
    network = network or FixedNetwork()
    return NeuralMCTSAgent(
        simulator,
        network,
        PolicyValueContext.from_level_config(simulator.env.config),
        **kwargs,
    )


def test_puct_formula_selection_and_ties():
    edge = ActionStats(prior=0.75, visits=2, value_sum=1)
    assert edge.mean_value == 0.5
    assert edge.puct(4, 2) == pytest.approx(0.5 + 2 * 0.75 * np.sqrt(5) / 3)
    root = DecisionNode({1: ActionStats(0.5), 0: ActionStats(0.5)})
    assert NeuralMCTSAgent._choose_action(root, 1) == 0
    root.actions[1].prior = 0.9
    assert NeuralMCTSAgent._choose_action(root, 1) == 1


def test_exact_backups_leaf_and_terminal_without_sign_changes():
    simulator = ControlledSimulator(root_actions=(0,))
    agent = make_agent(simulator, simulations=3)
    assert agent.select_action(simulator.env._observation()) == 0
    root = agent.last_root
    stats = agent.last_search_stats
    assert root.visits == stats.root_visits == stats.simulations == 3
    assert root.value_sum == pytest.approx(2.4)
    edge = root.actions[0]
    assert edge.visits == 3 and edge.value_sum == pytest.approx(2.4)
    assert edge.mean_value == pytest.approx(0.8)
    successor = next(iter(edge.successors.values()))
    assert successor.visits == successor.node.visits == 3
    assert successor.value_sum == successor.node.value_sum == pytest.approx(2.4)
    terminal_edge = successor.node.actions[16]
    assert terminal_edge.visits == 2 and terminal_edge.value_sum == 2
    assert stats.neural_evaluations == 2
    assert stats.sampled_transitions == 5 and stats.max_depth == 2
    assert stats.unique_successor_states == 2
    assert stats.root_successor_counts == {0: 1}
    assert stats.elapsed_seconds >= 0
    assert all(clone.closed for clone in simulator.clones)
    assert simulator.env.depth == 0 and not simulator.env.actions


@pytest.mark.parametrize("reward", [-1.0, 0.0, 1.0])
def test_terminal_values_ignore_neural_prediction(reward):
    simulator = ControlledSimulator(root_actions=(0,), horizon=1, rewards=(reward, -1))
    network = FixedNetwork(value=-0.7)
    agent = make_agent(simulator, network, simulations=4)
    agent.select_action(simulator.env._observation())
    stats = agent.last_search_stats
    assert stats.action_values[0] == reward
    assert stats.action_value_sums[0] == 4 * reward
    assert stats.neural_evaluations == network.calls == 1
    assert stats.sampled_transitions == 4 and stats.max_depth == 1


def test_legal_actions_zero_visits_and_normalized_policy():
    simulator = ControlledSimulator(horizon=1)
    agent = make_agent(simulator, simulations=4, c_puct=0)
    assert agent.select_action(simulator.env._observation()) == 0
    stats = agent.last_search_stats
    assert stats.action_visits == {0: 4, 1: 0}
    assert stats.action_values == {0: 1, 1: 0}
    assert stats.root_priors == {0: 0.5, 1: 0.5}
    assert stats.policy_target.shape == (144,)
    assert stats.policy_target[0] == stats.policy_target.sum() == 1
    assert not stats.policy_target[1:].any()
    assert not stats.policy_target.flags.writeable
    assert sum(stats.action_visits.values()) == stats.root_visits
    assert (
        agent._root_action(
            DecisionNode({1: ActionStats(0.5, visits=2), 0: ActionStats(0.5, visits=2)})
        )
        == 0
    )


def test_stochastic_outcomes_have_separate_nodes_and_legal_masks():
    simulator = ControlledSimulator(root_actions=(0,), stochastic=True)
    agent = make_agent(simulator, simulations=12, seed=42)
    agent.select_action(simulator.env._observation())
    successors = list(agent.last_root.actions[0].successors.values())
    assert len(successors) == 2
    assert {tuple(s.node.actions) for s in successors} == {(16,), (17,)}
    assert sum(s.visits for s in successors) == 12
    assert sum(s.value_sum for s in successors) == pytest.approx(
        agent.last_root.value_sum
    )
    assert all(s.node.visits == s.visits for s in successors)
    assert agent.last_search_stats.unique_successor_states == 4
    assert agent.last_search_stats.neural_evaluations == 3


def test_chance_outcomes_are_sampled_not_selected_by_value():
    simulator = ControlledSimulator(
        root_actions=(0,), stochastic=True, chance_rewards=(1, -1)
    )
    agent = make_agent(simulator, simulations=12)
    agent.select_action(simulator.env._observation())
    expected = {0: 0, 1: 0}
    for seed in simulator.seeds:
        expected[seed % 2] += 1
    actual = {
        next(iter(s.node.actions)) - 16: s.visits
        for s in agent.last_root.actions[0].successors.values()
    }
    assert actual == expected
    expected_value = sum(
        0.4 + (count - 1) * (1 if branch == 0 else -1)
        for branch, count in expected.items()
        if count
    )
    assert agent.last_root.value_sum == pytest.approx(expected_value)


def test_rng_streams_independent_and_evaluation_ignores_noise():
    first_sim, second_sim = (
        ControlledSimulator(horizon=1),
        ControlledSimulator(horizon=1),
    )
    first = make_agent(first_sim, simulations=8, seed=7)
    second = make_agent(
        second_sim,
        simulations=8,
        seed=7,
        evaluation=False,
        exploration_seed=99,
        dirichlet_epsilon=1,
    )
    first.select_action(first_sim.env._observation())
    second.select_action(second_sim.env._observation())
    assert first_sim.seeds == second_sim.seeds
    assert len(set(first_sim.seeds)) == 8
    noised_eval = make_agent(
        ControlledSimulator(horizon=1),
        simulations=8,
        seed=7,
        exploration_seed=99,
        dirichlet_epsilon=1,
        temperature=7,
    )
    noised_eval.select_action(noised_eval.simulator.env._observation())
    assert (
        noised_eval.last_search_stats.action_visits
        == first.last_search_stats.action_visits
    )
    assert (
        noised_eval.last_search_stats.root_priors == first.last_search_stats.root_priors
    )


def test_seeded_root_noise_is_normalized_legal_and_reproducible():
    agents = [
        make_agent(
            simulations=3,
            evaluation=False,
            exploration_seed=51,
            dirichlet_epsilon=1,
            dirichlet_alpha=0.4,
        )
        for _ in range(2)
    ]
    actions = [a.select_action(a.simulator.env._observation()) for a in agents]
    expected = np.random.default_rng(51).dirichlet([0.4, 0.4])
    np.testing.assert_allclose(
        list(agents[0].last_search_stats.root_priors.values()), expected
    )
    assert sum(agents[0].last_search_stats.root_priors.values()) == pytest.approx(1)
    assert (
        agents[0].last_search_stats.root_priors
        == agents[1].last_search_stats.root_priors
    )
    assert actions[0] == actions[1]


def test_temperature_sampling_and_zero_temperature():
    root = DecisionNode(
        {
            0: ActionStats(0.5, visits=1),
            1: ActionStats(0.5, visits=3),
            16: ActionStats(0),
        }
    )
    agent = make_agent(evaluation=False, exploration_seed=12)
    draws = [agent._root_action(root) for _ in range(2000)]
    assert set(draws) == {0, 1}
    assert draws.count(1) / len(draws) == pytest.approx(0.75, abs=0.04)
    zero = make_agent(evaluation=False, temperature=0)
    assert zero._root_action(root) == 1
    with pytest.raises(RuntimeError, match="no root visits"):
        zero._root_action(DecisionNode({0: ActionStats(1)}))


def test_reproducibility_full_statistics_except_timing():
    agents = [make_agent(simulations=9, seed=23) for _ in range(2)]
    selected = [a.select_action(a.simulator.env._observation()) for a in agents]
    assert selected[0] == selected[1]
    first, second = (a.last_search_stats for a in agents)
    for name in first.__dataclass_fields__:
        if name == "elapsed_seconds":
            continue
        if name == "policy_target":
            np.testing.assert_array_equal(first.policy_target, second.policy_target)
        else:
            assert getattr(first, name) == getattr(second, name)


def test_raw_state_identity_excludes_rng_metadata_and_preserves_counts():
    agent = make_agent()
    observation = agent.simulator.env._observation()
    first = agent._state_key(observation)
    observation.update(level_id="unrelated", rng_state={"future": 999})
    assert agent._state_key(observation) == first
    observation["objective_progress"][0] = 10
    second = agent._state_key(observation)
    observation["objective_progress"][0] = 11
    assert agent._state_key(observation) != second
    other = make_agent()
    other.context = replace(agent.context, move_limit=5)
    assert other._state_key(agent.simulator.env._observation()) != first


def test_no_legal_actions_terminal_root_and_stale_observation():
    agent = make_agent(ControlledSimulator(root_actions=()))
    with pytest.raises(RuntimeError, match="no legal actions"):
        agent.select_action(agent.simulator.env._observation())
    assert agent.network.calls == 0 and agent.last_search_stats is None
    agent = make_agent()
    observation = agent.simulator.env._observation()
    observation["moves_remaining"] = 3
    with pytest.raises(ValueError, match="does not match"):
        agent.select_action(observation)
    agent.simulator.env._episode_done = True
    with pytest.raises(RuntimeError, match="terminal live"):
        agent.select_action(agent.simulator.env._observation())


@pytest.mark.parametrize(
    "options,message",
    [
        ({"fail": True}, "controlled failure"),
        ({"horizon": 1, "truncate": True, "rewards": (0, 0)}, "truncation"),
    ],
)
def test_errors_close_simulations_restore_network_mode_and_clear_stats(
    options, message
):
    simulator = ControlledSimulator(**options)
    network = FixedNetwork()
    assert network.training
    agent = make_agent(simulator, network, simulations=2)
    with pytest.raises(RuntimeError, match=message):
        agent.select_action(simulator.env._observation())
    assert all(clone.closed for clone in simulator.clones)
    assert network.training
    assert agent.last_search_stats is None and agent.last_root is None


@pytest.mark.parametrize(
    "option,value",
    [
        ("simulations", 0),
        ("simulations", True),
        ("simulations", 1.0),
        ("simulations", 1.5),
        ("c_puct", -1),
        ("c_puct", float("nan")),
        ("c_puct", float("inf")),
        ("c_puct", True),
        ("c_puct", "1"),
        ("temperature", -1),
        ("dirichlet_alpha", 0),
        ("dirichlet_epsilon", 1.1),
        ("seed", True),
        ("exploration_seed", -1),
        ("evaluation", 1),
    ],
)
def test_constructor_validation(option, value):
    with pytest.raises(ValueError):
        make_agent(**{option: value})


def test_invalid_neural_values_are_rejected():
    agent = make_agent(network=FixedNetwork(float("nan")))
    with pytest.raises(ValueError, match="neural value"):
        agent.select_action(agent.simulator.env._observation())
    assert agent.network.training


def test_real_simulator_agent_interface_isolation_and_hidden_future_independence():
    env = Match3LevelEnv("first-match")
    try:
        observation, _ = env.reset(seed=200012)
        original = deepcopy(observation)
        network = PolicyValueNetwork(seed=4)
        network.spatial.eval()
        modes = [module.training for module in network.modules()]
        parameters = [p.detach().clone() for p in network.parameters()]
        context = PolicyValueContext.from_level_config(env.config)
        first = NeuralMCTSAgent(
            MCTSSimulator(env), network, context, simulations=2, seed=9
        )
        before = complete_state_fingerprint(env)
        action = first.select_action(observation)
        assert observation["action_mask"][action]
        assert complete_state_fingerprint(env) == before
        env.session.engine.rng = np.random.default_rng(998)
        before = complete_state_fingerprint(env)
        second = NeuralMCTSAgent(
            MCTSSimulator(env), network, context, simulations=2, seed=9
        )
        assert second.select_action(observation) == action
        assert complete_state_fingerprint(env) == before
        assert (
            first.last_search_stats.action_visits
            == second.last_search_stats.action_visits
        )
        assert (
            first.last_search_stats.action_values
            == second.last_search_stats.action_values
        )
        assert modes == [module.training for module in network.modules()]
        for actual, expected in zip(network.parameters(), parameters):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        for key, value in original.items():
            np.testing.assert_array_equal(observation[key], value)
    finally:
        env.close()
