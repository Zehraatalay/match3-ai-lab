from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from match3_ai.games.match3.engine import Match3Engine
from match3_ai.games.match3.generator import generate_playable_board
from match3_ai.games.match3.moves import find_valid_moves

Position = tuple[int, int]
Action = tuple[Position, Position]


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

        if type(num_colors) is not int or not 3 <= num_colors <= 255:
            raise ValueError("num_colors must be an integer between 3 and 255")

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
