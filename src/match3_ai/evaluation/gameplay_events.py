"""Incremental gameplay telemetry, independent of policy and reward logic."""

import json
import os
from dataclasses import fields, is_dataclass
from enum import Enum

import numpy as np

from match3_ai.evaluation.analytics import AnalyticsCollector
from match3_ai.evaluation.behavior_data import observation_fingerprint
from match3_ai.games.match3.environment import Match3LevelEnv


def structured(value):
    """Detach game records into deterministic, JSON-compatible values."""
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value):
        return {
            field.name: structured(getattr(value, field.name))
            for field in fields(value)
        }
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: structured(item) for key, item in value.items()}
    if isinstance(value, (set, frozenset)):
        return [structured(item) for item in sorted(value)]
    if isinstance(value, (tuple, list)):
        return [structured(item) for item in value]
    return value


class GameplayCollector(AnalyticsCollector):
    """Stream existing move/episode records and richer ordered engine events."""

    def __init__(self, handle, experiment_id, episode_id):
        super().__init__()
        self.handle = handle
        self.experiment_id = experiment_id
        self.episode_id = episode_id
        self.sequence = 0

    def emit(self, kind, **data):
        record = structured(
            {
                "schema_version": 1,
                "experiment_id": self.experiment_id,
                "episode_id": self.episode_id,
                "sequence": self.sequence,
                "kind": kind,
                **data,
            }
        )
        self.handle.write(json.dumps(record, allow_nan=False) + "\n")
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.sequence += 1

    def record_move(self, record):
        super().record_move(record)
        self.emit("move_summary", move_index=record.move_index, record=record)

    def record_episode(self, record):
        super().record_episode(record)
        self.emit("episode_end", record=record)


class InstrumentedLevelEnv(Match3LevelEnv):
    """Log actual transitions once; never simulate or replay policy decisions."""

    def __init__(self, level_id, collector, *, initial_identities=None, **kwargs):
        super().__init__(level_id, record_events=True, **kwargs)
        self.collector = collector
        self.initial_identities = initial_identities
        self.pairing_error = False
        self.move_index = 0

    def reset(self, **kwargs):
        observation, info = super().reset(**kwargs)
        self.move_index = 0
        self.initial_identity = observation_fingerprint(observation)
        self.collector.emit("episode_start", observation=observation, info=info)
        if self.initial_identities is not None:
            key = (self.level_id, info["effective_episode_seed"])
            previous = self.initial_identities.setdefault(key, self.initial_identity)
            if previous != self.initial_identity:
                self.pairing_error = True
                raise RuntimeError("Initial boards differ across agents")
        return observation, info

    def step(self, action):
        self.move_index += 1
        self.collector.emit(
            "move_start",
            move_index=self.move_index,
            action_id=int(action),
            state=self.session.board,
            moves_remaining=self.session.moves_remaining,
        )
        observation, reward, terminated, truncated, info = super().step(action)
        round_index = 0
        for index, event in enumerate(info["resolution_events"]):
            if event.event_type.name == "MATCH":
                round_index += 1
            self.collector.emit(
                "resolution",
                move_index=self.move_index,
                event_index=index,
                cascade_round=round_index,
                event=event,
            )
        self.collector.emit(
            "move_end",
            move_index=self.move_index,
            state=self.session.board,
            moves_remaining=self.session.moves_remaining,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            cascade_count=info["cascades"],
        )
        return observation, reward, terminated, truncated, info
