"""Simulation controller shared by the Pygame dashboard and tests."""

from __future__ import annotations

from collections import Counter, deque
import random
from typing import Callable

from snake_ai.agents.manual_agent import action_for_direction
from snake_ai.agents.q_learning import EncodedState, QLearningAgent, encode_state
from snake_ai.agents.q_learning_two_step import (
    TwoStepDangerQLearningAgent,
    encode_two_step_state,
)
from snake_ai.game import Action, Direction, GameState, SnakeGame
from snake_ai.visualization.snapshot import DashboardSnapshot

SPEEDS = (1, 5, 10, 50, 100, 500, 1000, 5000)
MODES = ("manual", "random", "q-learning", "q-learning-2step")


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
        self._recent_snake_lengths: deque[int] = deque(maxlen=50)
        self._max_snake_length = len(self.game.state.snake)
        self.q_agent = QLearningAgent(seed=seed)
        self.two_step_q_agent = TwoStepDangerQLearningAgent(seed=seed)

    @property
    def snapshot(self) -> DashboardSnapshot:
        metrics, q_values = self._learning_telemetry()
        names = {
            "q-learning": ("Q-Learning", "Tabular Q-learning"),
            "q-learning-2step": ("Q-Learning 2-Step", "14-bit two-step danger"),
        }
        agent_name, learning_view = names.get(
            self.mode, (self.mode.title(), "Baseline agent")
        )
        return DashboardSnapshot(
            game_state=self.game.state,
            agent_name=agent_name,
            learning_view=learning_view,
            episode=self.episode,
            reward=self.reward,
            action=self.action,
            paused=self.paused,
            speed=self.speed,
            max_snake_length=self._max_snake_length,
            average_snake_length_50=self._average_snake_length_50,
            q_values=q_values,
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
        self._recent_snake_lengths.clear()
        self._max_snake_length = len(self.game.state.snake)
        self.q_agent = QLearningAgent(seed=self._seed)
        self.two_step_q_agent = TwoStepDangerQLearningAgent(seed=self._seed)

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

        q_agent, encoder = self._active_q_learning()
        encoded = encoder(self.game.state) if encoder is not None else None
        exploratory = False
        if self.mode == "manual":
            action = self._pending_manual_action
            self._pending_manual_action = Action.STRAIGHT
        elif q_agent is not None and encoded is not None:
            action, exploratory = q_agent.choose_action(encoded.state_id)
        else:
            action = self._rng.choice(tuple(Action))

        next_state, self.reward, done, _ = self.game.step(action)
        if q_agent is not None and encoder is not None and encoded is not None:
            next_encoded = encoder(next_state)
            q_agent.update(
                encoded.state_id,
                action,
                self.reward,
                next_encoded.state_id,
                done,
                exploratory=exploratory,
            )
        self.action = action
        self._action_counts[action] += 1
        self._max_snake_length = max(self._max_snake_length, len(next_state.snake))
        if done:
            self._recent_snake_lengths.append(len(next_state.snake))
        if done and self.mode == "manual":
            self.paused = True

    def _next_episode(self) -> None:
        q_agent, _ = self._active_q_learning()
        if q_agent is not None:
            q_agent.finish_episode()
        self.episode += 1
        self.game.reset()
        self.reward = 0.0
        self.action = None

    def _learning_telemetry(
        self,
    ) -> tuple[dict[str, str | int | float | bool], tuple[float, float, float] | None]:
        q_agent, encoder = self._active_q_learning()
        if q_agent is None or encoder is None:
            return (
                {
                    "left actions": self._action_counts[Action.LEFT],
                    "straight actions": self._action_counts[Action.STRAIGHT],
                    "right actions": self._action_counts[Action.RIGHT],
                    "learning status": "No learning - baseline agent",
                },
                None,
            )

        encoded = encoder(self.game.state)
        q_values = q_agent.q_table[encoded.state_id]
        metrics: dict[str, str | int | float | bool] = {
            "state bits": "".join(str(bit) for bit in encoded.features),
            "state ID": encoded.state_id,
            "table rows": q_agent.state_count,
            "epsilon": f"{q_agent.epsilon:.3f}",
        }
        update = q_agent.last_update
        if update is not None:
            metrics.update(
                {
                    "selection": "explore" if update.exploratory else "greedy",
                    "updated row": update.state_id,
                    "old Q": f"{update.old_value:.3f}",
                    "update reward": f"{update.reward:.1f}",
                    "target": f"{update.target:.3f}",
                    "new Q": f"{update.new_value:.3f}",
                }
            )
        return metrics, tuple(q_values)

    @property
    def _average_snake_length_50(self) -> float:
        if not self._recent_snake_lengths:
            return 0.0
        return sum(self._recent_snake_lengths) / len(self._recent_snake_lengths)

    def _active_q_learning(
        self,
    ) -> tuple[
        QLearningAgent | None,
        Callable[[GameState], EncodedState] | None,
    ]:
        if self.mode == "q-learning":
            return self.q_agent, encode_state
        if self.mode == "q-learning-2step":
            return self.two_step_q_agent, encode_two_step_state
        return None, None
