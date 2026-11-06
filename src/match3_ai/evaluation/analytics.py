"""Optional observation-based records and CSV export for evaluation."""

from __future__ import annotations

import csv
import json
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from match3_ai.evaluation.runner import EpisodeResult


@dataclass(frozen=True, slots=True)
class MoveRecord:
    experiment_id: str
    episode_id: str
    move_index: int
    action_id: int | None
    valid_action: bool | None
    reward: float | None
    moves_remaining: int | None
    cleared_tiles: int | None
    cleared_by_color: tuple[int, ...] | None
    cascade_count: int | None
    blocker_damage: dict | None
    special_activations: dict | None
    objective_progress: list[dict] | None
    legal_action_count: int | None
    resolution_completed: bool
    error_type: str | None = None
    error_message: str | None = None


class AnalyticsCollector:
    """Collect detached records; duplicate episode and move IDs are rejected."""

    def __init__(self) -> None:
        self.episodes: list[EpisodeResult] = []
        self.moves: list[MoveRecord] = []
        self._episode_ids: set[str] = set()
        self._move_keys: set[tuple[str, int]] = set()

    def record_move(self, record: MoveRecord) -> None:
        key = (record.episode_id, record.move_index)
        if key in self._move_keys:
            raise ValueError(f"duplicate move record: {key}")
        if record.move_index <= 0:
            raise ValueError("move_index must be positive")
        self._move_keys.add(key)
        self.moves.append(deepcopy(record))

    def record_episode(self, record: EpisodeResult) -> None:
        if record.episode_id in self._episode_ids:
            raise ValueError(f"duplicate episode ID: {record.episode_id}")
        self._episode_ids.add(record.episode_id)
        self.episodes.append(deepcopy(record))

    def validate(self) -> None:
        episodes = {episode.episode_id: episode for episode in self.episodes}
        for move in self.moves:
            episode = episodes.get(move.episode_id)
            if episode is None or episode.experiment_id != move.experiment_id:
                raise ValueError(
                    f"move references an unknown episode: {move.episode_id}"
                )
        for episode in self.episodes:
            indices = sorted(
                move.move_index
                for move in self.moves
                if move.episode_id == episode.episode_id
            )
            if indices != list(range(1, episode.steps + 1)):
                raise ValueError(
                    f"incomplete move trace for episode: {episode.episode_id}"
                )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True)
    return value


def export_evaluation(
    collector: AnalyticsCollector,
    metadata: dict[str, Any],
    output_dir: str | Path,
) -> Path:
    """Write a new result directory; an existing directory is never overwritten.

    Missing CSV values are blank; nested values use JSON. Nulls remain null in JSON.
    """
    from match3_ai.evaluation.runner import EpisodeResult

    collector.validate()
    if any(
        episode.experiment_id != metadata.get("experiment_id")
        for episode in collector.episodes
    ):
        raise ValueError("episode experiment IDs do not match export metadata")
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=False)
    for filename, records, record_type in (
        ("episodes.csv", collector.episodes, EpisodeResult),
        ("moves.csv", collector.moves, MoveRecord),
    ):
        with (path / filename).open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(record_type.__dataclass_fields__)
            )
            writer.writeheader()
            for record in records:
                writer.writerow(
                    {key: _csv_value(value) for key, value in asdict(record).items()}
                )
    with (path / "experiment_metadata.json").open("x", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return path
