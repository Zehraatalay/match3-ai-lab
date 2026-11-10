from dataclasses import dataclass

from match3_ai.games.match3.moves import find_valid_moves
from match3_ai.levels.config import LevelConfig
from match3_ai.levels.session import LevelSession


@dataclass(frozen=True, slots=True)
class ValidationResult:
    valid: bool
    reason: str | None = None
    legal_move_count: int = 0


class LevelValidator:
    def validate(self, config: LevelConfig) -> ValidationResult:
        if not isinstance(config, LevelConfig):
            raise TypeError("config must be a LevelConfig")

        try:
            session = LevelSession(config)
        except (ValueError, RuntimeError) as error:
            return ValidationResult(
                valid=False,
                reason=str(error),
            )

        legal_moves = find_valid_moves(session.board)

        if not legal_moves:
            return ValidationResult(
                valid=False,
                reason="No legal moves available at initialization",
            )

        return ValidationResult(
            valid=True,
            legal_move_count=len(legal_moves),
        )
