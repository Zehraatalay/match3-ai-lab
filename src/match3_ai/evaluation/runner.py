from __future__ import annotations

from dataclasses import dataclass

from match3_ai.agents.base import Agent
from match3_ai.games.match3.environment import Match3Env


@dataclass(frozen=True, slots=True)
class EpisodeResult:
    seed: int | None
    total_reward: float
    steps: int
    valid_actions: int
    invalid_actions: int
    total_cleared: int
    total_cascades: int
    reshuffles: int
    completed: bool


def run_episode(
    env: Match3Env,
    agent: Agent,
    seed: int | None = None,
    max_steps: int = 1000,
) -> EpisodeResult:
    """run one episode and collect evaluation metrics."""

    if max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    observation, _ = env.reset(seed=seed)

    total_reward = 0.0
    steps = 0
    valid_actions = 0
    invalid_actions = 0
    total_cleared = 0
    total_cascades = 0
    reshuffles = 0
    completed = False

    while steps < max_steps:
        action = agent.select_action(observation)

        # Validate the agent's output before passing it to the environment.
        if not env.action_space.contains(action):
            raise ValueError(f"agent selected an out-of-range action: {action}")

        observation, reward, terminated, truncated, info = env.step(action)

        steps += 1
        total_reward += reward
        total_cleared += info["cleared_tiles"]
        total_cascades += info["cascade_count"]
        reshuffles += int(info["reshuffled"])

        if info["valid_action"]:
            valid_actions += 1
        else:
            invalid_actions += 1

        if terminated or truncated:
            completed = True
            break

    return EpisodeResult(
        seed=seed,
        total_reward=total_reward,
        steps=steps,
        valid_actions=valid_actions,
        invalid_actions=invalid_actions,
        total_cleared=total_cleared,
        total_cascades=total_cascades,
        reshuffles=reshuffles,
        completed=completed,
    )
