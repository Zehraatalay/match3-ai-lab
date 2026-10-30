from __future__ import annotations

import numpy as np
import pytest

from match3_ai.agents.greedy_agent import GreedyAgent, build_action_pairs
from match3_ai.games.match3.environment import Match3Env
from match3_ai.games.match3.rules import all_find_matches


def test_action_pair_count():
    pairs = build_action_pairs(8, 8)

    assert len(pairs) == 112
    assert len(set(pairs)) == 112


def test_greedy_selects_valid_action():
    env = Match3Env()

    try:
        observation, _ = env.reset(seed=42)

        agent = GreedyAgent()
        action = agent.select_action(observation)

        assert observation["action_mask"][action] == 1
    finally:
        env.close()


def test_greedy_is_deterministic():
    env = Match3Env()

    try:
        observation, _ = env.reset(seed=42)

        agent = GreedyAgent()

        action_a = agent.select_action(observation)
        action_b = agent.select_action(observation)

        assert action_a == action_b
    finally:
        env.close()


def test_greedy_does_not_mutate_observation():
    env = Match3Env()

    try:
        observation, _ = env.reset(seed=42)

        original_board = observation["board"].copy()
        original_mask = observation["action_mask"].copy()

        agent = GreedyAgent()
        agent.select_action(observation)

        assert np.array_equal(observation["board"], original_board)
        assert np.array_equal(observation["action_mask"], original_mask)
    finally:
        env.close()


def test_greedy_rejects_no_valid_actions():
    observation = {
        "board": np.ones((3, 3), dtype=np.uint8),
        "action_mask": np.zeros(12, dtype=np.int8),
        "moves_remaining": 10,
    }

    agent = GreedyAgent()

    with pytest.raises(RuntimeError, match="No valid actions available"):
        agent.select_action(observation)


def test_greedy_selects_highest_immediate_score():
    board = np.array(
        [
            [1, 2, 1, 3, 4],
            [2, 1, 2, 4, 3],
            [3, 2, 3, 1, 2],
            [4, 3, 4, 2, 1],
            [1, 4, 1, 3, 4],
        ],
        dtype=np.uint8,
    )

    action_pairs = build_action_pairs(5, 5)

    scores = {}

    for action_id, ((r1, c1), (r2, c2)) in enumerate(action_pairs):
        simulated_board = board.copy()

        simulated_board[r1, c1], simulated_board[r2, c2] = (
            simulated_board[r2, c2],
            simulated_board[r1, c1],
        )

        score = len(all_find_matches(simulated_board))

        if score > 0:
            scores[action_id] = score

    assert scores, "Test board must contain at least one scoring action."

    action_mask = np.zeros(len(action_pairs), dtype=np.int8)

    for action_id in scores:
        action_mask[action_id] = 1

    observation = {
        "board": board,
        "action_mask": action_mask,
        "moves_remaining": 20,
    }

    agent = GreedyAgent()
    selected_action = agent.select_action(observation)

    max_score = max(scores.values())

    best_actions = [
        action_id for action_id, score in scores.items() if score == max_score
    ]

    assert selected_action in best_actions
    assert selected_action == min(best_actions)


def test_action_pairs_match_environment():
    env = Match3Env(rows=4, cols=5)

    try:
        expected_pairs = build_action_pairs(4, 5)

        assert env.action_space.n == len(expected_pairs)

        for action_id, expected_pair in enumerate(expected_pairs):
            actual_pair = env._decode_action(action_id)

            assert actual_pair == expected_pair

    finally:
        env.close()
