"""Analysis-only observation storage and exact live-environment counterfactuals."""

import hashlib
import json
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path

import numpy as np

from match3_ai.agents.observation_encoder import RAW_FIELDS, ObservationEncoder
from match3_ai.games.match3.environment import Match3LevelEnv
from match3_ai.games.match3.rules import all_find_matches
from match3_ai.training.seed_policy import validate_partition

FIELDS = {**RAW_FIELDS, "action_mask": (np.bool_, (144,))}


def packed_observation(observation):
    values = ObservationEncoder().pack(observation)
    values["action_mask"] = np.asarray(observation["action_mask"], dtype=bool).copy()
    return values


def observation_fingerprint(observation):
    values = packed_observation(observation)
    digest = hashlib.sha256()
    for key, (dtype, shape) in FIELDS.items():
        digest.update(f"{key}:{np.dtype(dtype).str}:{shape}".encode())
        digest.update(values[key].tobytes())
    return digest.hexdigest()


class ObservationStore:
    def __init__(self):
        self.observations = []
        self.fingerprints = []
        self.index = {}

    def intern(self, observation):
        fingerprint = observation_fingerprint(observation)
        if fingerprint not in self.index:
            self.index[fingerprint] = len(self.observations)
            self.observations.append(packed_observation(observation))
            self.fingerprints.append(fingerprint)
        return self.index[fingerprint]

    def export(self, path):
        if not self.observations:
            raise ValueError("cannot export an empty observation store")
        with Path(path).open("xb") as handle:
            np.savez_compressed(
                handle,
                **{
                    key: np.stack(
                        [observation[key] for observation in self.observations]
                    )
                    for key in FIELDS
                },
                fingerprints=np.asarray(self.fingerprints, dtype="U64"),
            )


def objective_completion(observation):
    active = np.asarray(observation["objective_mask"], dtype=bool)
    target = np.asarray(observation["objective_target"])[active]
    progress = np.asarray(observation["objective_progress"])[active]
    if not len(target) or np.any(target <= 0):
        raise ValueError("active objectives need positive targets")
    return float(np.minimum(progress.astype(float) / target, 1).mean())


def complete_state_metadata(env):
    if not isinstance(env, Match3LevelEnv) or env.session is None:
        raise ValueError("requires an initialized Match3LevelEnv")
    session = env.session
    return {
        "observation": observation_fingerprint(env._observation()),
        "level_config": asdict(env.config),
        "objective_progress": [asdict(value) for value in session.objective_progress],
        "moves_remaining": session.moves_remaining,
        "status": session.status.value,
        "initial_blocker_count": session._initial_blocker_count,
        "session_board_seed": session._board_seed,
        "session_engine_seed": session._engine_seed,
        "engine_num_colors": session.engine.num_colors,
        "max_cascades": session.engine.max_cascades,
        "max_initialization_attempts": session.max_initialization_attempts,
        "engine_rng": deepcopy(session.engine.rng.bit_generator.state),
        "gym_rng": deepcopy(env.np_random.bit_generator.state),
        "episode_done": env._episode_done,
        "has_reset": env._has_reset,
        "effective_episode_seed": env._effective_episode_seed,
        "board_seed": env._board_seed,
        "engine_seed": env._engine_seed,
    }


def complete_state_fingerprint(env):
    return hashlib.sha256(
        json.dumps(complete_state_metadata(env), sort_keys=True).encode()
    ).hexdigest()


def clone_environment(env):
    before = complete_state_fingerprint(env)
    clone = deepcopy(env)
    if complete_state_fingerprint(clone) != before:
        raise RuntimeError("clone does not preserve complete state")
    for name in ("tiles", "specials", "blockers", "blocker_hp", "cell_mask"):
        if np.shares_memory(
            getattr(env.session.board, name), getattr(clone.session.board, name)
        ):
            raise RuntimeError("clone shares board arrays")
    if clone.session.engine.rng is env.session.engine.rng:
        raise RuntimeError("clone shares engine RNG")
    return clone


def restore_by_prefix(
    level_id,
    seed,
    actions,
    expected_fingerprint,
    *,
    max_cascades=200,
    max_initialization_attempts=1000,
):
    validate_partition("validation", seed, 1)
    env = Match3LevelEnv(
        level_id,
        max_cascades=max_cascades,
        max_initialization_attempts=max_initialization_attempts,
    )
    env.reset(seed=seed)
    for action in actions:
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated or not info["valid_action"]:
            raise ValueError("prefix must end at a nonterminal valid decision state")
    if complete_state_fingerprint(env) != expected_fingerprint:
        raise ValueError("prefix restoration failed full-state/RNG verification")
    return env


def counterfactual_outcomes(env, actions):
    before = complete_state_fingerprint(env)
    observation = env._observation()
    initial_completion = objective_completion(observation)
    initial_hp = int(np.asarray(observation["blocker_hp"]).sum())
    results = {}
    for action in sorted(set(actions)):
        if not observation["action_mask"][action]:
            raise ValueError("counterfactual action must be legal")
        clone = clone_environment(env)
        try:
            after, reward, terminated, truncated, info = clone.step(action)
            if not info["valid_action"]:
                raise RuntimeError("counterfactual action rejected")
            hp = int(after["blocker_hp"].sum())
            if initial_hp - hp != info["blocker_damage"]["hits"]:
                raise RuntimeError("blocker HP and damage records differ")
            results[int(action)] = {
                "action_id": int(action),
                "cleared_tiles": info["cleared_tiles"],
                "objective_delta": objective_completion(after) - initial_completion,
                "blocker_hits": info["blocker_damage"]["hits"],
                "blockers_destroyed": info["blocker_damage"]["destroyed"],
                "blocker_hp_remaining": hp,
                "special_activations": info["special_activations"]["count"],
                "cascades": info["cascades"],
                "reward": reward,
                "terminated": terminated,
                "truncated": truncated,
                "level_status": info["level_status"],
                "after_fingerprint": observation_fingerprint(after),
            }
        except (RuntimeError, ValueError) as error:
            results[int(action)] = {
                "action_id": int(action),
                "error_type": type(error).__name__,
                "error_message": str(error),
                "error_context": getattr(error, "context", {}),
            }
        finally:
            clone.close()
    if complete_state_fingerprint(env) != before:
        raise RuntimeError("counterfactual evaluation mutated original state/RNG")
    return results


def greedy_scores(observation):
    rows, cols = map(int, observation["board_shape"])
    board = np.where(
        observation["cell_mask"][:rows, :cols], observation["tiles"][:rows, :cols], 0
    )
    scores = {}
    for action in np.flatnonzero(observation["action_mask"]):
        row, col, direction = Match3LevelEnv.decode_action(int(action))
        other = (row, col + 1) if direction == 0 else (row + 1, col)
        tiles = board.copy()
        tiles[row, col], tiles[other] = tiles[other], tiles[row, col]
        scores[int(action)] = len(all_find_matches(tiles))
    return scores
