from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from importlib import resources
from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from match3_ai.games.match3.blockers import BlockerType
from match3_ai.games.match3.engine import (
    DEFAULT_MAX_CASCADES,
    CascadeLimitExceeded,
    Match3Engine,
    StepResult,
)
from match3_ai.games.match3.generator import generate_playable_board
from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.games.match3.specials import SpecialType
from match3_ai.levels import (
    LevelConfig,
    LevelSession,
    LevelStatus,
    ObjectiveType,
    load_level,
)

Position = tuple[int, int]
Action = tuple[Position, Position]

DEFAULT_LEVEL_INITIALIZATION_ATTEMPTS = 1000


class Match3Env(gym.Env):
    metadata: ClassVar[dict] = {"render_modes": []}

    def __init__(
        self,
        rows: int = 8,
        cols: int = 8,
        num_colors: int = 5,
        move_limit: int = 20,
    ) -> None:
        super().__init__()

        if type(rows) is not int or rows < 3:
            raise ValueError("rows must be an integer >= 3")

        if type(cols) is not int or cols < 3:
            raise ValueError("cols must be an integer >= 3")

        if type(num_colors) is not int or not 3 <= num_colors <= 254:
            raise ValueError("num_colors must be an integer between 3 and 254")

        if type(move_limit) is not int or move_limit <= 0:
            raise ValueError("move_limit must be a positive integer")

        self.rows = rows
        self.cols = cols
        self.num_colors = num_colors
        self.move_limit = move_limit

        # each action represents one unique adjacent cell pair
        self.actions = self._build_actions()
        # her swape integer id atiyoruz
        self.action_to_id = {action: index for index, action in enumerate(self.actions)}

        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Dict(
            {
                "board": spaces.Box(
                    low=0,
                    high=num_colors,
                    shape=(rows, cols),
                    dtype=np.uint8,
                ),
                # her action valid or invalid belirtiyor
                "action_mask": spaces.MultiBinary(len(self.actions)),
                "moves_remaining": spaces.Discrete(move_limit + 1),
            }
        )

        self.state = None
        self.engine = None
        self.moves_remaining = move_limit
        self._episode_done = False

    def _build_actions(self) -> list[Action]:
        """
        build a fixed list of unique adjacent swaps.
        only right and down neighbors are enumerated,
        but all four physical swap directions are represented.
        """
        actions = []

        for row in range(self.rows):
            for col in range(self.cols):
                first = (row, col)

                if col + 1 < self.cols:
                    actions.append((first, (row, col + 1)))

                if row + 1 < self.rows:
                    actions.append((first, (row + 1, col)))

        return actions

    def _decode_action(self, action: int) -> Action:
        """convert an action ID into a pair of positions."""
        return self.actions[action]

    def _encode_action(
        self,
        first: Position,
        second: Position,
    ) -> int:
        """
        convert a swap into an action ID.
        both directions of the same swap map to one ID.
        """
        swap = (first, second)

        if swap in self.action_to_id:
            return self.action_to_id[swap]

        reverse_swap = (second, first)

        if reverse_swap in self.action_to_id:
            return self.action_to_id[reverse_swap]

        raise ValueError("positions must be orthogonally adjacent cells.")

    def _get_action_mask(self) -> np.ndarray:
        # baslangicta full 0 olusturuyoruz
        mask = np.zeros(
            self.action_space.n,
            dtype=np.int8,
        )
        # mevcut board uzerindeki valid swapleri buluyoruz
        for first, second in find_valid_moves(self.state):
            action_id = self._encode_action(first, second)
            mask[action_id] = 1
        # agent bu actionsdan birini secebilir
        return mask

    def _get_observation(self) -> dict[str, Any]:
        return {
            "board": self.state.tiles.copy(),
            "action_mask": self._get_action_mask(),
            "moves_remaining": self.moves_remaining,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ):
        super().reset(seed=seed)
        # 2 ayri seed uretiyoruz bc board generation ile engine icindeki refill islemlerini ayri rastgelelik akısında tutmak istiyoruz
        board_seed = int(self.np_random.integers(0, 2**32))

        engine_seed = int(self.np_random.integers(0, 2**32))
        # baslangic boardini olusturuyoruz
        self.state = generate_playable_board(
            rows=self.rows,
            cols=self.cols,
            num_colors=self.num_colors,
            seed=board_seed,
        )
        # game engine olusturuyoruz
        self.engine = Match3Engine(
            num_colors=self.num_colors,
            seed=engine_seed,
        )
        # episode durumunu sifirliyoruz
        self.moves_remaining = self.move_limit
        self._episode_done = False

        observation = self._get_observation()

        info = {
            "valid_actions": int(np.count_nonzero(observation["action_mask"])),
        }

        return observation, info

    def step(self, action: int):
        # reset() cagrilmadan movement yapilmaz
        if self.state is None or self.engine is None:
            raise RuntimeError("call reset() before step().")
        # episode bittiye movement yapilmaz
        if self._episode_done:
            raise RuntimeError("episode finished call reset().")
        # action id action space icinde mi control edilir
        if not self.action_space.contains(action):
            raise ValueError("action is outside the action space.")
        # integer actionu coordinatese convert ediyoruz
        first, second = self._decode_action(int(action))
        # engine swap -> match detection -> clear -> gravity -> refill -> cascade resolution -> dead board recovery islemlerini gerceklestiriyor
        result = self.engine.step(
            self.state,
            (first, second),
        )
        # just valid movementsda board update and movements -1 azaliyor
        if result.valid:
            self.state = result.state
            self.moves_remaining -= 1
        # reward = cleared tiles
        reward = float(result.cleared_tiles)

        terminated = False
        truncated = self.moves_remaining == 0

        self._episode_done = terminated or truncated

        observation = self._get_observation()
        # for debugging
        info = {
            "valid_action": result.valid,
            "cleared_tiles": result.cleared_tiles,
            "cascade_count": result.cascade_count,
            "reshuffled": result.reshuffled,
        }

        return observation, reward, terminated, truncated, info


class Match3LevelEnv(gym.Env):
    """Fixed-shape level adapter with raw objective counts.

    Agents should select only actions whose observation mask entry is nonzero.
    """

    metadata: ClassVar[dict] = {"render_modes": []}
    # mevcut seed namespace konumlari korunur ve yeni level kimlikleri sona eklenir
    LEVEL_REGISTRY_ORDER: ClassVar[tuple[str, ...]] = (
        "blocker-mayhem",
        "double-trouble",
        "first-match",
        "frozen-garden",
        "stone-breaker",
        "wooden-trouble",
        "wooden-trouble-v2",
        "double-trouble-v2",
        "frozen-garden-v2",
    )
    LEVEL_VERSION_PARENTS: ClassVar[dict[str, str]] = {
        "wooden-trouble-v2": "wooden-trouble",
        "double-trouble-v2": "double-trouble",
        "frozen-garden-v2": "frozen-garden",
    }
    MAX_ROWS: ClassVar[int] = 9
    MAX_COLS: ClassVar[int] = 8
    MAX_OBJECTIVES: ClassVar[int] = 2
    ACTION_COUNT: ClassVar[int] = MAX_ROWS * MAX_COLS * 2
    OBJECTIVE_CODES: ClassVar[dict[ObjectiveType, int]] = {
        ObjectiveType.CLEAR_ALL_BLOCKERS: 1,
        ObjectiveType.COLLECT_TILES: 2,
    }

    def __init__(
        self,
        level_id: str = "first-match",
        *,
        max_cascades: int = DEFAULT_MAX_CASCADES,
        max_initialization_attempts: int = DEFAULT_LEVEL_INITIALIZATION_ATTEMPTS,
        record_events: bool = False,
        level_config: LevelConfig | None = None,
    ) -> None:
        super().__init__()
        if type(max_cascades) is not int or max_cascades <= 0:
            raise ValueError("max_cascades must be a positive integer")
        if (
            type(max_initialization_attempts) is not int
            or max_initialization_attempts <= 0
        ):
            raise ValueError("max_initialization_attempts must be a positive integer")
        if type(record_events) is not bool:
            raise TypeError("record_events must be a boolean")
        self.record_events = record_events
        self.max_cascades = max_cascades
        self.max_initialization_attempts = max_initialization_attempts
        self._configs = self._load_packaged_levels()
        if level_config is not None:
            if not isinstance(level_config, LevelConfig):
                raise TypeError("level_config must be a LevelConfig")

            level_config.__post_init__()

            if level_config.level_id != level_id:
                raise ValueError("level_config must retain the selected level ID")

            if level_config.rows > self.MAX_ROWS or level_config.cols > self.MAX_COLS:
                raise ValueError("level_config exceeds padded board shape")

            if (
                not level_config.objectives
                or len(level_config.objectives) > self.MAX_OBJECTIVES
                or any(
                    item.target is not None and item.target > 65535
                    for item in level_config.objectives
                )
            ):
                raise ValueError("level_config exceeds observation objective capacity")

            self._configs[level_id] = level_config

        elif level_id not in self._configs:
            raise ValueError(f"unknown level_id: {level_id}")

        self.level_id = level_id
        self.config = self._configs[level_id]
        self.session: LevelSession | None = None
        self._episode_done = False
        self._has_reset = False
        self._effective_episode_seed: int | None = None
        self._board_seed: int | None = None
        self._engine_seed: int | None = None

        self.action_space = spaces.Discrete(self.ACTION_COUNT)
        self.observation_space = spaces.Dict(
            {
                "tiles": spaces.Box(0, 255, (self.MAX_ROWS, self.MAX_COLS), np.uint8),
                "specials": spaces.Box(
                    0,
                    max(int(value) for value in SpecialType),
                    (self.MAX_ROWS, self.MAX_COLS),
                    np.uint8,
                ),
                "blockers": spaces.Box(
                    0,
                    max(int(value) for value in BlockerType),
                    (self.MAX_ROWS, self.MAX_COLS),
                    np.uint8,
                ),
                "blocker_hp": spaces.Box(
                    0, 255, (self.MAX_ROWS, self.MAX_COLS), np.uint8
                ),
                "cell_mask": spaces.MultiBinary((self.MAX_ROWS, self.MAX_COLS)),
                "board_shape": spaces.Box(
                    low=np.array([0, 0], dtype=np.int16),
                    high=np.array([self.MAX_ROWS, self.MAX_COLS], dtype=np.int16),
                    dtype=np.int16,
                ),
                "moves_remaining": spaces.Discrete(
                    max(config.move_limit for config in self._configs.values()) + 1
                ),
                "objective_type": spaces.MultiDiscrete(
                    np.full(self.MAX_OBJECTIVES, len(self.OBJECTIVE_CODES) + 1)
                ),
                "objective_color": spaces.MultiDiscrete(
                    np.full(self.MAX_OBJECTIVES, 255)
                ),
                "objective_progress": spaces.Box(
                    0, 65535, (self.MAX_OBJECTIVES,), np.uint16
                ),
                "objective_target": spaces.Box(
                    0, 65535, (self.MAX_OBJECTIVES,), np.uint16
                ),
                "objective_mask": spaces.MultiBinary(self.MAX_OBJECTIVES),
                "action_mask": spaces.MultiBinary(self.ACTION_COUNT),
            }
        )

    @classmethod
    def _load_packaged_levels(cls) -> dict[str, LevelConfig]:
        data_dir = resources.files("match3_ai.levels").joinpath("data")
        configs: dict[str, LevelConfig] = {}
        for resource in sorted(data_dir.iterdir(), key=lambda item: item.name):
            if not resource.name.endswith(".json"):
                continue
            with resources.as_file(resource) as path:
                config = load_level(path)
            if config.level_id in configs:
                raise ValueError(f"duplicate packaged level_id: {config.level_id}")
            if config.rows > cls.MAX_ROWS or config.cols > cls.MAX_COLS:
                raise ValueError(
                    f"level {config.level_id!r} exceeds padded board shape"
                )
            if len(config.objectives) > cls.MAX_OBJECTIVES:
                raise ValueError(
                    f"level {config.level_id!r} exceeds objective capacity "
                    f"({cls.MAX_OBJECTIVES})"
                )
            for objective in config.objectives:
                if objective.target is not None and objective.target > 65535:
                    raise ValueError(
                        f"level {config.level_id!r} objective target exceeds uint16"
                    )
            configs[config.level_id] = config
        if not configs:
            raise ValueError("no packaged JSON levels were found")
        order = (
            *cls.LEVEL_REGISTRY_ORDER,
            *sorted(set(configs).difference(cls.LEVEL_REGISTRY_ORDER)),
        )
        return {
            level_id: configs[level_id] for level_id in order if level_id in configs
        }

    @property
    def level_ids(self) -> tuple[str, ...]:
        return tuple(self._configs)

    @classmethod
    def encode_action(cls, row: int, col: int, direction: int) -> int:
        if type(row) is not int or not 0 <= row < cls.MAX_ROWS:
            raise ValueError("row is outside the fixed observation shape")
        if type(col) is not int or not 0 <= col < cls.MAX_COLS:
            raise ValueError("col is outside the fixed observation shape")
        if type(direction) is not int or direction not in (0, 1):
            raise ValueError("direction must be 0 (RIGHT) or 1 (DOWN)")
        return ((row * cls.MAX_COLS) + col) * 2 + direction

    @classmethod
    def decode_action(cls, action_id: int) -> tuple[int, int, int]:
        if type(action_id) is not int or not 0 <= action_id < cls.ACTION_COUNT:
            raise ValueError("action_id is outside the action space")
        cell, direction = divmod(action_id, 2)
        row, col = divmod(cell, cls.MAX_COLS)
        return row, col, direction

    def _action_mask(self) -> np.ndarray:
        mask = np.zeros(self.ACTION_COUNT, dtype=np.int8)
        assert self.session is not None
        for first, second in find_valid_moves(self.session.board):
            row_a, col_a = first
            row_b, col_b = second
            if row_a == row_b:
                action_id = self.encode_action(row_a, min(col_a, col_b), 0)
            else:
                action_id = self.encode_action(min(row_a, row_b), col_a, 1)
            mask[action_id] = 1
        return mask

    def _observation(self) -> dict[str, Any]:
        assert self.session is not None
        board = self.session.board
        rows, cols = board.shape
        tiles = np.zeros((self.MAX_ROWS, self.MAX_COLS), dtype=np.uint8)
        specials = np.zeros_like(tiles)
        blockers = np.zeros_like(tiles)
        blocker_hp = np.zeros_like(tiles)
        cell_mask = np.zeros((self.MAX_ROWS, self.MAX_COLS), dtype=np.int8)
        target = (slice(0, rows), slice(0, cols))
        tiles[target] = board.tiles
        specials[target] = board.specials
        blockers[target] = board.blockers
        blocker_hp[target] = board.blocker_hp
        cell_mask[target] = board.cell_mask

        objective_type = np.zeros(self.MAX_OBJECTIVES, dtype=np.int64)
        objective_color = np.zeros(self.MAX_OBJECTIVES, dtype=np.int64)
        objective_progress = np.zeros(self.MAX_OBJECTIVES, dtype=np.uint16)
        objective_target = np.zeros(self.MAX_OBJECTIVES, dtype=np.uint16)
        objective_mask = np.zeros(self.MAX_OBJECTIVES, dtype=np.int8)
        for index, progress in enumerate(self.session.objective_progress):
            objective_type[index] = self.OBJECTIVE_CODES[progress.objective_type]
            objective_progress[index] = min(progress.raw_progress, 65535)
            objective_target[index] = min(progress.target, 65535)
            objective_mask[index] = 1
            objective = self.config.objectives[index]
            if objective.objective_type == ObjectiveType.COLLECT_TILES:
                objective_color[index] = objective.color or 0

        return {
            "tiles": tiles,
            "specials": specials,
            "blockers": blockers,
            "blocker_hp": blocker_hp,
            "cell_mask": cell_mask,
            "board_shape": np.array([rows, cols], dtype=np.int16),
            "moves_remaining": int(self.session.moves_remaining),
            "objective_type": objective_type,
            "objective_color": objective_color,
            "objective_progress": objective_progress,
            "objective_target": objective_target,
            "objective_mask": objective_mask,
            "action_mask": self._action_mask(),
        }

    def _reset_info(self) -> dict[str, Any]:
        assert self.session is not None
        return {
            "level_id": self.level_id,
            "board_shape": self.session.board.shape,
            "configured_level_seed": self.config.seed,
            "level_config_id": hashlib.sha256(
                json.dumps(asdict(self.config), sort_keys=True).encode()
            ).hexdigest(),
            "move_limit": self.config.move_limit,
            "max_cascades": self.max_cascades,
            "max_initialization_attempts": self.max_initialization_attempts,
            "effective_episode_seed": self._effective_episode_seed,
            "board_seed": self._board_seed,
            "engine_seed": self._engine_seed,
            "moves_remaining": self.session.moves_remaining,
            "level_status": self.session.status.value,
            "objective_progress": self._objective_info(),
        }

    def _objective_info(self) -> list[dict[str, Any]]:
        assert self.session is not None
        return [
            {
                "type": progress.objective_type.value,
                "color": objective.color,
                "progress": progress.raw_progress,
                "target": progress.target,
                "completed": progress.completed,
            }
            for progress, objective in zip(
                self.session.objective_progress, self.config.objectives
            )
        ]

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        if options is not None:
            if not isinstance(options, dict):
                raise TypeError("options must be a dictionary or None")
            unknown = set(options) - {"level_id"}
            if unknown:
                raise ValueError(f"unknown reset options: {', '.join(sorted(unknown))}")
            selected_level = options.get("level_id", self.level_id)
            if (
                not isinstance(selected_level, str)
                or selected_level not in self._configs
            ):
                raise ValueError(f"unknown level_id: {selected_level}")
            self.level_id = selected_level
        self.config = self._configs[self.level_id]

        first_default_reset = seed is None and not self._has_reset
        root_seed = self.config.seed if first_default_reset else seed
        super().reset(seed=root_seed)
        if root_seed is None:
            root_seed = int(self.np_random.integers(0, 2**32, dtype=np.uint64))
        self._effective_episode_seed = int(root_seed)

        if first_default_reset:
            board_seed = engine_seed = None
            self._board_seed = self.config.seed
            self._engine_seed = self.config.seed
        else:
            children = np.random.SeedSequence(
                [
                    self._effective_episode_seed,
                    self.level_ids.index(
                        self.LEVEL_VERSION_PARENTS.get(self.level_id, self.level_id)
                    ),
                ]
            ).spawn(2)
            board_seed, engine_seed = (
                int(child.generate_state(1, dtype=np.uint32)[0]) for child in children
            )
            self._board_seed = board_seed
            self._engine_seed = engine_seed

        self.session = LevelSession(
            self.config,
            board_seed=board_seed,
            engine_seed=engine_seed,
            max_cascades=self.max_cascades,
            max_initialization_attempts=self.max_initialization_attempts,
        )
        self._episode_done = False
        self._has_reset = True
        observation = self._observation()
        if not np.any(observation["action_mask"]):
            raise RuntimeError("level initialization produced no legal actions")
        return observation, self._reset_info()

    def step(
        self, action: int
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.session is None:
            raise RuntimeError("call reset() before step()")
        if self._episode_done:
            raise RuntimeError("episode finished; call reset()")
        if isinstance(action, (bool, np.bool_)) or not self.action_space.contains(
            action
        ):
            raise ValueError("action is outside the action space")
        action_id = int(action)
        if not self._action_mask()[action_id]:
            return self._observation(), 0.0, False, False, self._step_info(valid=False)

        row, col, direction = self.decode_action(action_id)
        second = (row, col + 1) if direction == 0 else (row + 1, col)
        try:
            result = self.session.step(
                ((row, col), second), record_events=self.record_events
            )
        except CascadeLimitExceeded as error:
            error.context.update(
                level_id=self.level_id,
                effective_episode_seed=self._effective_episode_seed,
                action_id=action_id,
            )
            error.add_note(
                f"level_id={self.level_id}, episode_seed={self._effective_episode_seed}, "
                f"action_id={action_id}"
            )
            raise
        if not result.valid:
            raise RuntimeError(
                "find_valid_moves returned a move rejected by LevelSession"
            )

        terminated = self.session.status != LevelStatus.IN_PROGRESS
        reward = self._terminal_reward() if terminated else 0.0
        observation = self._observation()
        dead_board = not terminated and not np.any(observation["action_mask"])
        if dead_board:
            terminated = True
            reward = -1.0
            self._episode_done = True
        if self.session.status != LevelStatus.IN_PROGRESS:
            self._episode_done = True
        return (
            observation,
            reward,
            terminated,
            False,
            self._step_info(valid=True, result=result, dead_board=dead_board),
        )

    def _terminal_reward(self) -> float:
        assert self.session is not None
        if self.session.status == LevelStatus.WON:
            return 1.0
        if self.session.status == LevelStatus.LOST:
            return -1.0
        return 0.0

    def _step_info(
        self,
        *,
        valid: bool,
        result: StepResult | None = None,
        dead_board: bool = False,
    ) -> dict[str, Any]:
        assert self.session is not None
        damage = () if result is None else result.blocker_damage
        activations = () if result is None else result.activation_history
        activation_steps = [
            step for activation in activations for step in activation.activation_steps
        ]
        by_blocker = {
            blocker.name: sum(record.blocker_type == blocker for record in damage)
            for blocker in BlockerType
            if blocker != BlockerType.NONE
        }
        return {
            **(
                {"resolution_events": () if result is None else result.animation_events}
                if self.record_events
                else {}
            ),
            "level_id": self.level_id,
            "valid_action": valid,
            "moves_remaining": self.session.moves_remaining,
            "level_status": self.session.status.value,
            "objective_progress": self._objective_info(),
            "cleared_tiles": 0 if result is None else result.cleared_tiles,
            "cleared_by_color": () if result is None else result.cleared_by_color,
            "cascades": 0 if result is None else result.cascade_count,
            "blocker_damage": {
                "hits": len(damage),
                "destroyed": sum(record.destroyed for record in damage),
                "by_type": by_blocker,
            },
            "special_activations": {
                "count": len(activation_steps),
                "by_type": {
                    special.name: sum(
                        step.special_type == special for step in activation_steps
                    )
                    for special in SpecialType
                    if special != SpecialType.NONE
                },
            },
            "reshuffled": False if result is None else result.reshuffled,
            "valid_actions": int(np.count_nonzero(self._action_mask())),
            "termination_reason": "no_legal_actions" if dead_board else None,
        }

    def close(self) -> None:
        return None
