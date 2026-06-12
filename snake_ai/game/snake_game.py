"""Dependency-free Snake environment with relative actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import random
from typing import TypeAlias

Position: TypeAlias = tuple[int, int]


class Action(IntEnum):
    """Actions relative to the snake's current direction."""

    LEFT = 0
    STRAIGHT = 1
    RIGHT = 2


class Direction(IntEnum):
    """Absolute directions, ordered clockwise."""

    UP = 0
    RIGHT = 1
    DOWN = 2
    LEFT = 3


_DIRECTION_VECTORS: dict[Direction, Position] = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}

_ACTION_TURNS: dict[Action, int] = {
    Action.LEFT: -1,
    Action.STRAIGHT: 0,
    Action.RIGHT: 1,
}


@dataclass(frozen=True)
class GameState:
    """Immutable snapshot returned by the environment."""

    width: int
    height: int
    snake: tuple[Position, ...]
    food: Position | None
    direction: Direction
    score: int
    steps: int
    done: bool

    @property
    def head(self) -> Position:
        return self.snake[0]


class SnakeGame:
    """Configurable Snake game suitable for agents and manual inspection."""

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        *,
        seed: int | None = None,
    ) -> None:
        if isinstance(width, bool) or not isinstance(width, int) or width < 3:
            raise ValueError("width must be an integer of at least 3")
        if isinstance(height, bool) or not isinstance(height, int) or height < 3:
            raise ValueError("height must be an integer of at least 3")

        self.width = width
        self.height = height
        self._rng = random.Random(seed)
        self._snake: list[Position] = []
        self._food: Position | None = None
        self._direction = Direction.RIGHT
        self._score = 0
        self._steps = 0
        self._done = False
        self.reset()

    @property
    def state(self) -> GameState:
        """Return the current immutable game snapshot."""
        return GameState(
            width=self.width,
            height=self.height,
            snake=tuple(self._snake),
            food=self._food,
            direction=self._direction,
            score=self._score,
            steps=self._steps,
            done=self._done,
        )

    def reset(self, *, seed: int | None = None) -> GameState:
        """Reset the board and return its initial state."""
        if seed is not None:
            self._rng.seed(seed)

        center_x = max(2, self.width // 2)
        center_y = self.height // 2
        self._snake = [
            (center_x, center_y),
            (center_x - 1, center_y),
            (center_x - 2, center_y),
        ]
        self._direction = Direction.RIGHT
        self._score = 0
        self._steps = 0
        self._done = False
        self._food = self._spawn_food()
        return self.state

    def step(self, action: int | Action) -> tuple[GameState, float, bool, dict[str, object]]:
        """Advance one step using a relative action."""
        if self._done:
            raise RuntimeError("cannot step a finished game; call reset() first")

        parsed_action = self._parse_action(action)
        self._direction = Direction(
            (int(self._direction) + _ACTION_TURNS[parsed_action]) % len(Direction)
        )
        dx, dy = _DIRECTION_VECTORS[self._direction]
        head_x, head_y = self._snake[0]
        new_head = (head_x + dx, head_y + dy)
        ate_food = new_head == self._food
        self._steps += 1

        if self._hits_wall(new_head) or self._hits_self(new_head, ate_food):
            self._done = True
            return self.state, -10.0, True, self._info("collision", parsed_action)

        self._snake.insert(0, new_head)
        reward = 0.0
        event = "moved"

        if ate_food:
            self._score += 1
            reward = 10.0
            event = "ate_food"
            self._food = self._spawn_food()
            if self._food is None:
                self._done = True
                event = "board_filled"
        else:
            self._snake.pop()

        return self.state, reward, self._done, self._info(event, parsed_action)

    def render(self) -> str:
        """Print and return a text representation of the current board."""
        rows = [["." for _ in range(self.width)] for _ in range(self.height)]
        if self._food is not None:
            food_x, food_y = self._food
            rows[food_y][food_x] = "*"

        for body_x, body_y in self._snake[1:]:
            rows[body_y][body_x] = "o"
        head_x, head_y = self._snake[0]
        rows[head_y][head_x] = "H"

        border = "+" + "-" * self.width + "+"
        output = "\n".join(
            [border, *("|" + "".join(row) + "|" for row in rows), border]
        )
        print(output)
        return output

    def _spawn_food(self) -> Position | None:
        occupied = set(self._snake)
        free_cells = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in occupied
        ]
        return self._rng.choice(free_cells) if free_cells else None

    def _hits_wall(self, position: Position) -> bool:
        x, y = position
        return x < 0 or x >= self.width or y < 0 or y >= self.height

    def _hits_self(self, position: Position, growing: bool) -> bool:
        body = self._snake if growing else self._snake[:-1]
        return position in body

    def _info(self, event: str, action: Action) -> dict[str, object]:
        return {
            "event": event,
            "score": self._score,
            "steps": self._steps,
            "action": action,
            "direction": self._direction,
        }

    @staticmethod
    def _parse_action(action: int | Action) -> Action:
        if isinstance(action, bool) or not isinstance(action, int):
            raise ValueError("action must be 0 (left), 1 (straight), or 2 (right)")
        try:
            return Action(action)
        except (TypeError, ValueError) as error:
            raise ValueError(
                "action must be 0 (left), 1 (straight), or 2 (right)"
            ) from error
