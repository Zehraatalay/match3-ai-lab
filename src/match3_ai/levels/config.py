from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LevelConfig:
    rows: int
    cols: int
    num_colors: int
    move_limit: int

    def __post_init__(self) -> None:
        if isinstance(self.rows, bool) or type(self.rows) is not int or self.rows < 3:
            raise ValueError("rows must be an integer >= 3")

        if isinstance(self.cols, bool) or type(self.cols) is not int or self.cols < 3:
            raise ValueError("cols must be an integer >= 3")

        if (
            isinstance(self.num_colors, bool)
            or type(self.num_colors) is not int
            or self.num_colors < 3
        ):
            raise ValueError("num_colors must be an integer >= 3")

        if (
            isinstance(self.move_limit, bool)
            or type(self.move_limit) is not int
            or self.move_limit <= 0
        ):
            raise ValueError("move_limit must be a positive integer")
