from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from time import perf_counter

import numpy as np

from match3_ai.agents.base import Agent
from match3_ai.evaluation.analytics import AnalyticsCollector, MoveRecord
from match3_ai.games.match3.environment import Match3Env, Match3LevelEnv


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
    experiment_id: str = "standalone"
    episode_id: str = ""
    agent_name: str | None = None
    level_id: str | None = None
    level_config_id: str | None = None
    agent_seed: int | None = None
    effective_episode_seed: int | None = None
    board_seed: int | None = None
    engine_seed: int | None = None
    outcome: str | None = None
    termination_reason: str | None = None
    moves_used: int | None = None
    objective_progress: list[dict] | None = None
    objective_completion_ratio: float | None = None
    blockers_remaining: int | None = None
    blocker_hp_remaining: int | None = None
    special_activations: int | None = None
    max_cascades_in_move: int = 0
    cascade_limit: int | None = None
    inference_time_seconds: float = field(default=0.0, compare=False)
    error_type: str | None = None
    error_message: str | None = None
    error_context: dict | None = None


def run_episode(
    env: Match3Env | Match3LevelEnv,
    agent: Agent,
    seed: int | None = None,
    max_steps: int = 1000,
    *,
    experiment_id: str = "standalone",
    episode_id: str | None = None,
    agent_name: str | None = None,
    agent_seed: int | None = None,
    collector: AnalyticsCollector | None = None,
) -> EpisodeResult:
    """Run one bounded episode using observations and info only.

    Level errors become explicit ERROR records. Legacy exceptions retain their
    original behavior. Error-state metrics describe the last completed transition;
    cascade-limit diagnostics additionally report work in the failed resolution.
    """

    if type(max_steps) is not int or max_steps <= 0:
        raise ValueError("max_steps must be positive.")

    level_mode = isinstance(env.unwrapped, Match3LevelEnv)
    agent_name = agent_name or type(agent).__name__
    level_id = getattr(env.unwrapped, "level_id", None)
    if episode_id is None:
        episode_id = hashlib.sha256(
            repr((experiment_id, agent_name, level_id, seed, agent_seed)).encode()
        ).hexdigest()[:24]
    reset_info = {}
    info = {}
    observation = None
    total_reward = 0.0
    steps = 0
    valid_actions = 0
    invalid_actions = 0
    total_cleared = 0
    total_cascades = 0
    reshuffles = 0
    completed = False
    outcome = "TIME_LIMIT"
    reason = "max_steps"
    inference_time = 0.0
    special_count = 0 if level_mode else None
    max_cascades = 0
    error_type = error_message = error_context = None
    action = None
    pending_move = False
    try:
        observation, reset_info = env.reset(seed=seed)
        info = reset_info
        level_id = reset_info.get("level_id", level_id)
        while steps < max_steps:
            steps += 1
            pending_move = True
            action = None
            started = perf_counter()
            try:
                action = agent.select_action(observation)
            finally:
                inference_time += perf_counter() - started

            if not env.action_space.contains(action):
                raise ValueError(f"agent selected an out-of-range action: {action}")
            action = int(action)
            observation, reward, terminated, truncated, info = env.step(action)
            cascades = info.get("cascades", info.get("cascade_count", 0))
            total_reward += reward
            total_cleared += info["cleared_tiles"]
            total_cascades += cascades
            max_cascades = max(max_cascades, cascades)
            reshuffles += int(info["reshuffled"])
            if info["valid_action"]:
                valid_actions += 1
            else:
                invalid_actions += 1
            if special_count is not None:
                special_count += info.get("special_activations", {}).get("count", 0)

            if collector is not None:
                collector.record_move(
                    MoveRecord(
                        experiment_id=experiment_id,
                        episode_id=episode_id,
                        move_index=steps,
                        action_id=action,
                        valid_action=info["valid_action"],
                        reward=float(reward),
                        moves_remaining=observation.get("moves_remaining"),
                        cleared_tiles=info.get("cleared_tiles"),
                        cleared_by_color=info.get("cleared_by_color"),
                        cascade_count=cascades,
                        blocker_damage=info.get("blocker_damage"),
                        special_activations=info.get("special_activations"),
                        objective_progress=info.get("objective_progress"),
                        legal_action_count=int(
                            np.count_nonzero(observation["action_mask"])
                        ),
                        resolution_completed=True,
                    )
                )
            pending_move = False

            if terminated or truncated:
                completed = True
                status = info.get("level_status")
                if status in ("WON", "LOST") and terminated:
                    outcome = status
                    reason = "objectives_complete" if status == "WON" else "move_limit"
                elif truncated:
                    outcome, reason = "TIME_LIMIT", "external_truncation"
                else:
                    outcome = "ERROR"
                    reason = info.get("termination_reason") or "unknown_terminal_status"
                if level_mode:
                    completed = outcome in ("WON", "LOST")
                break
    except Exception as error:
        if not level_mode:
            raise
        outcome, reason, completed = "ERROR", "exception", False
        error_type, error_message = type(error).__name__, str(error)
        error_context = dict(getattr(error, "context", {}))
        error_context.update(level_id=level_id, episode_seed=seed, action_id=action)
        processed = error_context.get("cascades_processed", 0)
        total_cascades += processed
        max_cascades = max(max_cascades, processed)
        if pending_move and collector is not None:
            collector.record_move(
                MoveRecord(
                    experiment_id=experiment_id,
                    episode_id=episode_id,
                    move_index=steps,
                    action_id=action if isinstance(action, int) else None,
                    valid_action=None,
                    reward=None,
                    moves_remaining=None
                    if observation is None
                    else observation.get("moves_remaining"),
                    cleared_tiles=None,
                    cleared_by_color=None,
                    cascade_count=processed
                    if "cascades_processed" in error_context
                    else None,
                    blocker_damage=None,
                    special_activations=None,
                    objective_progress=info.get("objective_progress"),
                    legal_action_count=None
                    if observation is None
                    else int(np.count_nonzero(observation["action_mask"])),
                    resolution_completed=False,
                    error_type=error_type,
                    error_message=error_message,
                )
            )

    objectives = info.get("objective_progress")
    ratio = None
    if objectives:
        ratio = float(
            np.mean(
                [
                    min(1.0, item["progress"] / item["target"])
                    if item["target"]
                    else 1.0
                    for item in objectives
                ]
            )
        )
    blockers_remaining = hp_remaining = None
    if observation is not None and "blockers" in observation:
        playable = observation["cell_mask"].astype(bool)
        blockers_remaining = int(np.count_nonzero(observation["blockers"][playable]))
        hp_remaining = int(observation["blocker_hp"][playable].sum())
    move_limit = reset_info.get("move_limit")
    moves_used = None
    if move_limit is not None and observation is not None:
        moves_used = int(move_limit - observation["moves_remaining"])

    result = EpisodeResult(
        seed=seed,
        total_reward=total_reward,
        steps=steps,
        valid_actions=valid_actions,
        invalid_actions=invalid_actions,
        total_cleared=total_cleared,
        total_cascades=total_cascades,
        reshuffles=reshuffles,
        completed=completed,
        experiment_id=experiment_id,
        episode_id=episode_id,
        agent_name=agent_name,
        level_id=level_id,
        level_config_id=reset_info.get("level_config_id"),
        agent_seed=agent_seed,
        effective_episode_seed=reset_info.get("effective_episode_seed"),
        board_seed=reset_info.get("board_seed"),
        engine_seed=reset_info.get("engine_seed"),
        outcome=outcome,
        termination_reason=reason,
        moves_used=moves_used,
        objective_progress=objectives,
        objective_completion_ratio=ratio,
        blockers_remaining=blockers_remaining,
        blocker_hp_remaining=hp_remaining,
        special_activations=special_count,
        max_cascades_in_move=max_cascades,
        cascade_limit=reset_info.get("max_cascades"),
        inference_time_seconds=inference_time,
        error_type=error_type,
        error_message=error_message,
        error_context=error_context,
    )
    if collector is not None:
        collector.record_episode(result)
    return result
