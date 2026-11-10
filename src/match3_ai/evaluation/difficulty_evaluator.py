from dataclasses import dataclass
from math import sqrt

import numpy as np

from match3_ai.agents.greedy_agent import GreedyAgent
from match3_ai.agents.mcts_simulator import MCTSSimulator
from match3_ai.agents.random_agent import RandomAgent
from match3_ai.agents.stochastic_mcts_agent import StochasticMCTSAgent
from match3_ai.evaluation.behavior_data import complete_state_fingerprint
from match3_ai.evaluation.runner import EpisodeResult, run_episode
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.levels.config import LevelConfig

SUPPORTED_AGENTS = (
    "Random",
    "Greedy",
    "MCTS_greedy_outcome",
)


@dataclass(frozen=True, slots=True)
class AgentDifficultyMetrics:
    agent_name: str
    episodes: int
    wins: int
    losses: int
    errors: int
    time_limits: int
    win_rate: float
    ci_lower: float
    ci_upper: float
    mean_moves_used: float | None
    mean_winning_moves: float | None
    mean_objective_completion: float | None


@dataclass(frozen=True, slots=True)
class DifficultyReport:
    level_id: str
    agent_metrics: tuple[AgentDifficultyMetrics, ...]
    episodes: tuple[EpisodeResult, ...] = ()


def wilson_interval(
    wins: int,
    total: int,
    z: float = 1.96,
) -> tuple[float, float]:
    if total <= 0:
        raise ValueError("total must be positive")

    if wins < 0 or wins > total:
        raise ValueError("wins must be between zero and total")

    p = wins / total
    z2 = z * z

    denominator = 1 + z2 / total

    center = (p + z2 / (2 * total)) / denominator

    margin = z * sqrt(p * (1 - p) / total + z2 / (4 * total * total)) / denominator

    return max(0.0, center - margin), min(1.0, center + margin)


def _mean(values: list[float]) -> float | None:
    if not values:
        return None

    return float(np.mean(values))


def summarize_agent(
    agent_name: str,
    episodes: list[EpisodeResult],
) -> AgentDifficultyMetrics:
    total = len(episodes)

    if total == 0:
        raise ValueError("episodes cannot be empty")

    wins = [episode for episode in episodes if episode.outcome == "WON"]

    losses = sum(episode.outcome == "LOST" for episode in episodes)

    errors = sum(episode.outcome == "ERROR" for episode in episodes)

    time_limits = sum(episode.outcome == "TIME_LIMIT" for episode in episodes)

    ci_lower, ci_upper = wilson_interval(len(wins), total)

    return AgentDifficultyMetrics(
        agent_name=agent_name,
        episodes=total,
        wins=len(wins),
        losses=losses,
        errors=errors,
        time_limits=time_limits,
        win_rate=len(wins) / total,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        mean_moves_used=_mean(
            [
                float(episode.moves_used)
                for episode in episodes
                if episode.moves_used is not None
            ]
        ),
        mean_winning_moves=_mean(
            [
                float(episode.moves_used)
                for episode in wins
                if episode.moves_used is not None
            ]
        ),
        mean_objective_completion=_mean(
            [
                episode.objective_completion_ratio
                for episode in episodes
                if episode.objective_completion_ratio is not None
            ]
        ),
    )


class LazyMCTSAgent:
    """Initialize MCTS after the real environment has been reset."""

    def __init__(
        self,
        env: Match3LevelEnv,
        seed: int,
        simulations: int,
    ) -> None:
        self.env = env
        self.seed = seed
        self.simulations = simulations
        self.agent: StochasticMCTSAgent | None = None

    def select_action(self, observation) -> int:
        if self.agent is None:
            self.agent = StochasticMCTSAgent(
                MCTSSimulator(self.env),
                simulations=self.simulations,
                rollout_policy="greedy",
                value_mode="outcome",
                seed=self.seed,
                exploration_constant=sqrt(2),
            )

        before = complete_state_fingerprint(self.env)

        try:
            return self.agent.select_action(observation)
        finally:
            after = complete_state_fingerprint(self.env)

            if before != after:
                raise RuntimeError("MCTS mutated the real environment or RNG")


class DifficultyEvaluator:
    def __init__(
        self,
        *,
        episodes_per_agent: int = 30,
        seed: int = 42,
        agents: tuple[str, ...] = ("Random", "Greedy"),
        mcts_simulations: int = 30,
    ) -> None:
        if type(episodes_per_agent) is not int or episodes_per_agent <= 0:
            raise ValueError("episodes_per_agent must be positive")

        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a non-negative integer")

        if not agents:
            raise ValueError("agents cannot be empty")

        if len(set(agents)) != len(agents):
            raise ValueError("agents must be unique")

        if any(agent_name not in SUPPORTED_AGENTS for agent_name in agents):
            raise ValueError("unsupported agent in agents")

        if type(mcts_simulations) is not int or mcts_simulations <= 0:
            raise ValueError("mcts_simulations must be positive")

        self.episodes_per_agent = episodes_per_agent
        self.seed = seed
        self.agents = agents
        self.mcts_simulations = mcts_simulations

    def evaluate(
        self,
        config: LevelConfig,
    ) -> DifficultyReport:
        if not isinstance(config, LevelConfig):
            raise TypeError("config must be a LevelConfig")

        summaries = []
        all_episodes = []

        for agent_index, agent_name in enumerate(self.agents):
            episodes = []

            for episode_index in range(self.episodes_per_agent):
                environment_seed = self.seed + episode_index

                agent_seed = (
                    self.seed + 1_000_000 + agent_index * 100_000 + episode_index
                )

                env = Match3LevelEnv(
                    config.level_id,
                    level_config=config,
                )

                try:
                    if agent_name == "Random":
                        agent = RandomAgent(seed=agent_seed)

                    elif agent_name == "Greedy":
                        agent = GreedyAgent()

                    elif agent_name == "MCTS_greedy_outcome":
                        agent = LazyMCTSAgent(
                            env=env,
                            seed=agent_seed,
                            simulations=self.mcts_simulations,
                        )

                    result = run_episode(
                        env,
                        agent,
                        seed=environment_seed,
                        agent_seed=agent_seed,
                        agent_name=agent_name,
                        experiment_id="plg_difficulty_screening",
                    )

                    episodes.append(result)

                finally:
                    env.close()

            summaries.append(
                summarize_agent(
                    agent_name,
                    episodes,
                )
            )

            all_episodes.extend(episodes)

        return DifficultyReport(
            level_id=config.level_id,
            agent_metrics=tuple(summaries),
            episodes=tuple(all_episodes),
        )
