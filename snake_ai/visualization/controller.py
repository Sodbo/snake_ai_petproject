"""Simulation controller shared by the Pygame dashboard and tests."""

from __future__ import annotations

from collections import Counter
import random

from snake_ai.agents.manual_agent import action_for_direction
from snake_ai.game import Action, Direction, SnakeGame
from snake_ai.visualization.snapshot import DashboardSnapshot

SPEEDS = (1, 5, 10, 50, 100, 500)
MODES = ("manual", "random")


class DashboardController:
    """Own game execution while keeping rendering concerns separate."""

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        *,
        seed: int | None = None,
        mode: str = "random",
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of: {', '.join(MODES)}")
        self._seed = seed
        self._rng = random.Random(seed)
        self.game = SnakeGame(width, height, seed=seed)
        self.mode = mode
        self.episode = 1
        self.reward = 0.0
        self.action: Action | None = None
        self.paused = True
        self.speed = 1
        self._pending_manual_action = Action.STRAIGHT
        self._action_counts: Counter[Action] = Counter()

    @property
    def snapshot(self) -> DashboardSnapshot:
        metrics = {
            "left actions": self._action_counts[Action.LEFT],
            "straight actions": self._action_counts[Action.STRAIGHT],
            "right actions": self._action_counts[Action.RIGHT],
            "learning status": "No learning - baseline agent",
        }
        return DashboardSnapshot(
            game_state=self.game.state,
            agent_name=self.mode.title(),
            learning_view="Baseline agent",
            episode=self.episode,
            reward=self.reward,
            action=self.action,
            paused=self.paused,
            speed=self.speed,
            metrics=metrics,
        )

    def set_mode(self, mode: str) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of: {', '.join(MODES)}")
        self.mode = mode
        self.reset()

    def set_speed(self, speed: int) -> None:
        if speed not in SPEEDS:
            raise ValueError(f"speed must be one of: {SPEEDS}")
        self.speed = speed
        self.paused = False

    def toggle_pause(self) -> None:
        self.paused = not self.paused

    def reset(self, *, width: int | None = None, height: int | None = None) -> None:
        width = self.game.width if width is None else width
        height = self.game.height if height is None else height
        self.game = SnakeGame(width, height, seed=self._seed)
        self.episode = 1
        self.reward = 0.0
        self.action = None
        self.paused = True
        self._action_counts.clear()

    def request_direction(self, direction: Direction) -> None:
        action = action_for_direction(self.game.state.direction, direction)
        if action is not None:
            self._pending_manual_action = action

    def advance(self, steps: int = 1, *, force: bool = False) -> None:
        if steps < 0:
            raise ValueError("steps cannot be negative")
        if self.paused and not force:
            return
        for _ in range(steps):
            self._advance_once()

    def _advance_once(self) -> None:
        if self.game.state.done:
            self._next_episode()

        if self.mode == "manual":
            action = self._pending_manual_action
            self._pending_manual_action = Action.STRAIGHT
        else:
            action = self._rng.choice(tuple(Action))

        _, self.reward, done, _ = self.game.step(action)
        self.action = action
        self._action_counts[action] += 1
        if done and self.mode == "manual":
            self.paused = True

    def _next_episode(self) -> None:
        self.episode += 1
        self.game.reset()
        self.reward = 0.0
        self.action = None

