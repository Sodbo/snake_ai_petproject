"""Simulation controller shared by the Pygame dashboard and tests."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from pathlib import Path
import random
from typing import Callable

import numpy as np

from snake_ai.agents.cnn_dqn import BOARD_CHANNELS, CNNDQNAgent, encode_board
from snake_ai.agents.dqn import (
    DEFAULT_GRADIENT_STEPS,
    DEFAULT_LEARNING_STARTS,
    DEFAULT_TRAIN_EVERY,
    DQNAgent,
    encode_dqn_state,
    sight_cell_count,
)
from snake_ai.agents.manual_agent import action_for_direction
from snake_ai.agents.q_learning import EncodedState, QLearningAgent, encode_state
from snake_ai.agents.q_learning_two_step import (
    TwoStepDangerQLearningAgent,
    encode_two_step_state,
)
from snake_ai.game import DEFAULT_STEP_PENALTY, Action, Direction, GameState, SnakeGame
from snake_ai.training.metrics import build_length_metrics, save_metrics
from snake_ai.visualization.snapshot import DashboardSnapshot

SPEEDS = (1, 5, 10, 50, 100, 500, 1000, 5000)
MODES = (
    "manual",
    "random",
    "q-learning",
    "q-learning-2step",
    "dqn",
    "dqn-inference",
    "cnn-dqn",
)


class DashboardController:
    """Own game execution while keeping rendering concerns separate."""

    def __init__(
        self,
        width: int = 20,
        height: int = 20,
        *,
        seed: int | None = None,
        mode: str = "random",
        sight_distance: int = 1,
        step_penalty: float = DEFAULT_STEP_PENALTY,
        dqn_checkpoint: str | Path | None = None,
        learning_starts: int = DEFAULT_LEARNING_STARTS,
        train_every: int = DEFAULT_TRAIN_EVERY,
        gradient_steps: int = DEFAULT_GRADIENT_STEPS,
        cnn_device: str = "auto",
    ) -> None:
        if mode not in MODES:
            raise ValueError(f"mode must be one of: {', '.join(MODES)}")
        self._seed = seed
        self._dqn_schedule = {
            "learning_starts": learning_starts,
            "train_every": train_every,
            "gradient_steps": gradient_steps,
        }
        self._dqn_checkpoint = Path(dqn_checkpoint) if dqn_checkpoint else None
        self._cnn_device = cnn_device
        if mode == "dqn-inference" and self._dqn_checkpoint is None:
            raise ValueError("dqn-inference mode requires a DQN checkpoint")
        inference_agent: DQNAgent | None = None
        self._checkpoint_metadata: dict[str, object] = {}
        if self._dqn_checkpoint is not None:
            inference_agent, self._checkpoint_metadata = DQNAgent.load_checkpoint(
                self._dqn_checkpoint, seed=seed
            )
            sight_distance = inference_agent.sight_distance
        self._rng = random.Random(seed)
        self.game = SnakeGame(width, height, seed=seed, step_penalty=step_penalty)
        self.mode = mode
        self.step_penalty = self.game.step_penalty
        sight_cell_count(sight_distance)
        self.sight_distance = sight_distance
        self.episode = 1
        self.reward = 0.0
        self.action: Action | None = None
        self.paused = True
        self.speed = 1
        self._pending_manual_action = Action.STRAIGHT
        self._action_counts: Counter[Action] = Counter()
        self._episode_snake_lengths: list[int] = []
        self._running_max_lengths: list[float] = []
        self._rolling_average_lengths: list[float] = []
        self._coverage_history: list[float] = []
        self._loss_history: list[float] = []
        self._current_episode_losses: list[float] = []
        self._learning_started_episode: int | None = None
        self._last_exploratory = False
        self._max_snake_length = len(self.game.state.snake)
        self.q_agent = QLearningAgent(seed=seed)
        self.two_step_q_agent = TwoStepDangerQLearningAgent(seed=seed)
        self.dqn_agent = inference_agent or DQNAgent(
            sight_distance=sight_distance, seed=seed, **self._dqn_schedule
        )
        self.cnn_agent: CNNDQNAgent | None = (
            CNNDQNAgent(device=cnn_device, seed=seed, **self._dqn_schedule)
            if mode == "cnn-dqn"
            else None
        )

    @property
    def snapshot(self) -> DashboardSnapshot:
        metrics, q_values = self._learning_telemetry()
        q_agent, _ = self._active_q_learning()
        names = {
            "q-learning": ("Q-Learning", "Tabular Q-learning"),
            "q-learning-2step": ("Q-Learning 2-Step", "14-bit two-step danger"),
            "dqn": ("DQN", "NumPy deep Q-network"),
            "dqn-inference": ("DQN Inference", "Loaded deterministic policy"),
            "cnn-dqn": ("CNN DQN", "PyTorch full-board Double DQN"),
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
            length_history=tuple(self._episode_snake_lengths),
            running_max=tuple(self._running_max_lengths),
            rolling_average_50=tuple(self._rolling_average_lengths),
            q_table_coverage=q_agent.q_table_coverage if q_agent is not None else None,
            coverage_history=tuple(self._coverage_history),
            loss_history=tuple(self._loss_history),
            learning_started_episode=self._learning_started_episode,
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

    def dump_stats(self, output: str | Path | None = None) -> Path:
        """Save all completed dashboard episodes as metrics JSON."""
        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output = Path("outputs/stats") / f"{self.mode}_{timestamp}.json"
        algorithm = {
            "q-learning": "Q-Learning",
            "q-learning-2step": "Q-Learning 2-Step",
            "dqn": "DQN (NumPy)",
            "dqn-inference": "DQN Inference",
            "cnn-dqn": "CNN Double DQN (PyTorch)",
        }.get(self.mode, self.mode.title())
        q_agent, _ = self._active_q_learning()
        config: dict[str, object] = {
            "source": "dashboard",
            "mode": self.mode,
            "width": self.game.width,
            "height": self.game.height,
            "completed_episodes": len(self._episode_snake_lengths),
            "step_penalty": self.step_penalty,
        }
        if q_agent is not None:
            config.update(
                {
                    "allocated_state_count": q_agent.state_count,
                    "valid_state_count": q_agent.valid_state_count,
                }
            )
        if self.mode in ("dqn", "dqn-inference"):
            config.update(
                {
                    "network_layers": self.dqn_agent.online_network.layer_sizes,
                    "batch_size": self.dqn_agent.batch_size,
                    "target_sync_interval": self.dqn_agent.target_sync_interval,
                    "learning_starts": self.dqn_agent.learning_starts,
                    "train_every": self.dqn_agent.train_every,
                    "gradient_steps": self.dqn_agent.gradient_steps,
                    "environment_steps": self.dqn_agent.environment_steps,
                    "training_steps": self.dqn_agent.training_steps,
                    "learning_started_episode": self._learning_started_episode,
                    "sight_distance": self.sight_distance,
                    "sight_cells": sight_cell_count(self.sight_distance),
                    "checkpoint": (
                        str(self._dqn_checkpoint) if self._dqn_checkpoint else None
                    ),
                }
            )
        if self.mode == "cnn-dqn":
            config.update(self._cnn_config())
        return save_metrics(
            build_length_metrics(
                algorithm,
                self._episode_snake_lengths,
                q_table_coverage=self._coverage_history or None,
                loss=self._loss_history or None,
                config=config,
            ),
            output,
        )

    def save_dqn_checkpoint(self, output: str | Path | None = None) -> Path:
        """Save the current DQN policy for later inference."""
        if self.mode not in ("dqn", "dqn-inference", "cnn-dqn"):
            raise ValueError("DQN checkpoints are only available in DQN modes")
        if output is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            suffix = ".pt" if self.mode == "cnn-dqn" else ".npz"
            output = Path("outputs/models") / f"{self.mode}_{timestamp}{suffix}"
        if self.mode == "cnn-dqn":
            assert self.cnn_agent is not None
            return self.cnn_agent.save_checkpoint(
                output,
                metadata={
                    "width": self.game.width,
                    "height": self.game.height,
                    "step_penalty": self.step_penalty,
                    "completed_episodes": len(self._episode_snake_lengths),
                    "learning_started_episode": self._learning_started_episode,
                },
            )
        return self.dqn_agent.save_checkpoint(
            output,
            metadata={
                "width": self.game.width,
                "height": self.game.height,
                "step_penalty": self.step_penalty,
                "completed_episodes": len(self._episode_snake_lengths),
                "learning_started_episode": self._learning_started_episode,
            },
        )

    def reset(self, *, width: int | None = None, height: int | None = None) -> None:
        width = self.game.width if width is None else width
        height = self.game.height if height is None else height
        self.game = SnakeGame(
            width, height, seed=self._seed, step_penalty=self.step_penalty
        )
        self.episode = 1
        self.reward = 0.0
        self.action = None
        self.paused = True
        self._action_counts.clear()
        self._episode_snake_lengths.clear()
        self._running_max_lengths.clear()
        self._rolling_average_lengths.clear()
        self._coverage_history.clear()
        self._loss_history.clear()
        self._current_episode_losses.clear()
        self._learning_started_episode = None
        self._last_exploratory = False
        self._max_snake_length = len(self.game.state.snake)
        self.q_agent = QLearningAgent(seed=self._seed)
        self.two_step_q_agent = TwoStepDangerQLearningAgent(seed=self._seed)
        if self.mode == "dqn-inference":
            assert self._dqn_checkpoint is not None
            self.dqn_agent, self._checkpoint_metadata = DQNAgent.load_checkpoint(
                self._dqn_checkpoint, seed=self._seed
            )
        else:
            self.dqn_agent = DQNAgent(
                sight_distance=self.sight_distance,
                seed=self._seed,
                **self._dqn_schedule,
            )
        self.cnn_agent = (
            CNNDQNAgent(
                device=self._cnn_device, seed=self._seed, **self._dqn_schedule
            )
            if self.mode == "cnn-dqn"
            else None
        )

    def set_sight_distance(self, sight_distance: int) -> None:
        """Change DQN sight radius and reset training for the new input shape."""
        if self.mode == "dqn-inference":
            raise ValueError("loaded inference model sight distance cannot be changed")
        sight_cell_count(sight_distance)
        self.sight_distance = sight_distance
        self.reset()

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
        dqn_features = (
            encode_dqn_state(self.game.state, self.sight_distance)
            if self.mode in ("dqn", "dqn-inference")
            else None
        )
        cnn_board = encode_board(self.game.state) if self.mode == "cnn-dqn" else None
        exploratory = False
        if self.mode == "manual":
            action = self._pending_manual_action
            self._pending_manual_action = Action.STRAIGHT
        elif q_agent is not None and encoded is not None:
            action, exploratory = q_agent.choose_action(encoded.state_id)
        elif self.mode in ("dqn", "dqn-inference"):
            assert dqn_features is not None
            action, exploratory = self.dqn_agent.choose_action(dqn_features)
        elif self.mode == "cnn-dqn":
            assert cnn_board is not None
            assert self.cnn_agent is not None
            action, exploratory = self.cnn_agent.choose_action(cnn_board)
        else:
            action = self._rng.choice(tuple(Action))
        self._last_exploratory = exploratory

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
        elif self.mode == "dqn":
            assert dqn_features is not None
            self.dqn_agent.remember(
                dqn_features,
                action,
                self.reward,
                encode_dqn_state(next_state, self.sight_distance),
                done,
            )
            previous_loss_count = len(self.dqn_agent.loss_history)
            previous_training_steps = self.dqn_agent.training_steps
            self.dqn_agent.learn()
            if (
                self._learning_started_episode is None
                and self.dqn_agent.training_steps > previous_training_steps
            ):
                self._learning_started_episode = self.episode
            self._current_episode_losses.extend(
                self.dqn_agent.loss_history[previous_loss_count:]
            )
        elif self.mode == "cnn-dqn":
            assert cnn_board is not None
            assert self.cnn_agent is not None
            self.cnn_agent.remember(
                cnn_board,
                action,
                self.reward,
                encode_board(next_state),
                done,
            )
            previous_loss_count = len(self.cnn_agent.loss_history)
            previous_training_steps = self.cnn_agent.training_steps
            self.cnn_agent.learn()
            if (
                self._learning_started_episode is None
                and self.cnn_agent.training_steps > previous_training_steps
            ):
                self._learning_started_episode = self.episode
            self._current_episode_losses.extend(
                self.cnn_agent.loss_history[previous_loss_count:]
            )
        self.action = action
        self._action_counts[action] += 1
        self._max_snake_length = max(self._max_snake_length, len(next_state.snake))
        if done:
            self._record_episode_length(
                len(next_state.snake),
                q_table_coverage=(
                    q_agent.q_table_coverage if q_agent is not None else None
                ),
                loss=(
                    sum(self._current_episode_losses) / len(self._current_episode_losses)
                    if self._current_episode_losses
                    else (0.0 if self.mode in ("dqn", "cnn-dqn") else None)
                ),
            )
            self._current_episode_losses.clear()
        if done and self.mode == "manual":
            self.paused = True

    def _next_episode(self) -> None:
        q_agent, _ = self._active_q_learning()
        if q_agent is not None:
            q_agent.finish_episode()
        if self.mode == "dqn":
            self.dqn_agent.finish_episode()
        if self.mode == "cnn-dqn":
            assert self.cnn_agent is not None
            self.cnn_agent.finish_episode()
        self.episode += 1
        self.game.reset()
        self.reward = 0.0
        self.action = None

    def _learning_telemetry(
        self,
    ) -> tuple[dict[str, str | int | float | bool], tuple[float, float, float] | None]:
        q_agent, encoder = self._active_q_learning()
        if self.mode in ("dqn", "dqn-inference"):
            features = encode_dqn_state(self.game.state, self.sight_distance)
            q_values = self.dqn_agent.online_network.predict(
                np.asarray(features, dtype=np.float32)
            )[0]
            remaining = max(
                0, self.dqn_agent.learning_starts - self.dqn_agent.environment_steps
            )
            if self.mode == "dqn-inference":
                learning_status = "inference only"
            elif self.dqn_agent.training_steps:
                learning_status = "active"
            else:
                learning_status = f"warm-up ({remaining} steps remaining)"
            metrics: dict[str, str | int | float | bool] = {
                "sight distance": self.sight_distance,
                "sight cells": sight_cell_count(self.sight_distance),
                "network inputs": self.dqn_agent.input_count,
                "epsilon": f"{self.dqn_agent.epsilon:.3f}",
                "selection": "explore" if self._last_exploratory else "greedy",
                "replay size": len(self.dqn_agent.replay_buffer),
                "environment steps": self.dqn_agent.environment_steps,
                "training steps": self.dqn_agent.training_steps,
                "learning starts (steps)": self.dqn_agent.learning_starts,
                "learning status": learning_status,
                "started at episode": self._learning_started_episode or "-",
                "train every": self.dqn_agent.train_every,
                "gradient steps": self.dqn_agent.gradient_steps,
                "learning": self.mode == "dqn",
            }
            if self.mode == "dqn-inference":
                metrics["checkpoint"] = str(self._dqn_checkpoint)
            update = self.dqn_agent.last_update
            if update is not None:
                metrics.update(
                    {
                        "batch size": update.batch_size,
                        "loss": f"{update.loss:.5f}",
                        "target synced": update.target_synced,
                    }
                )
            return metrics, tuple(float(value) for value in q_values)
        if self.mode == "cnn-dqn":
            assert self.cnn_agent is not None
            metrics = self._cnn_telemetry()
            return metrics, self.cnn_agent.q_values(encode_board(self.game.state))
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
            "valid rows": q_agent.valid_state_count,
            "valid-space coverage": f"{q_agent.q_table_coverage:.3f}%",
            "epsilon": f"{q_agent.epsilon:.3f}",
        }
        update = q_agent.last_update
        if update is not None:
            metrics.update(
                {
                    "selection": "explore" if update.exploratory else "greedy",
                    "updated row": update.state_id,
                    "old Q": f"{update.old_value:.3f}",
                    "update reward": f"{update.reward:.2f}",
                    "target": f"{update.target:.3f}",
                    "new Q": f"{update.new_value:.3f}",
                }
            )
        return metrics, tuple(q_values)

    def _cnn_config(self) -> dict[str, object]:
        assert self.cnn_agent is not None
        return {
            "board_channels": BOARD_CHANNELS,
            "device": str(self.cnn_agent.device),
            "batch_size": self.cnn_agent.batch_size,
            "target_sync_interval": self.cnn_agent.target_sync_interval,
            "learning_starts": self.cnn_agent.learning_starts,
            "train_every": self.cnn_agent.train_every,
            "gradient_steps": self.cnn_agent.gradient_steps,
            "environment_steps": self.cnn_agent.environment_steps,
            "training_steps": self.cnn_agent.training_steps,
            "learning_started_episode": self._learning_started_episode,
        }

    def _cnn_telemetry(self) -> dict[str, str | int | float | bool]:
        assert self.cnn_agent is not None
        remaining = max(
            0, self.cnn_agent.learning_starts - self.cnn_agent.environment_steps
        )
        status = (
            "active"
            if self.cnn_agent.training_steps
            else f"warm-up ({remaining} steps remaining)"
        )
        metrics: dict[str, str | int | float | bool] = {
            "board channels": BOARD_CHANNELS,
            "device": str(self.cnn_agent.device),
            "epsilon": f"{self.cnn_agent.epsilon:.3f}",
            "selection": "explore" if self._last_exploratory else "greedy",
            "replay size": len(self.cnn_agent.replay_buffer),
            "environment steps": self.cnn_agent.environment_steps,
            "training steps": self.cnn_agent.training_steps,
            "learning starts (steps)": self.cnn_agent.learning_starts,
            "learning status": status,
            "started at episode": self._learning_started_episode or "-",
            "train every": self.cnn_agent.train_every,
            "gradient steps": self.cnn_agent.gradient_steps,
        }
        if self.cnn_agent.last_update is not None:
            metrics.update(
                {
                    "batch size": self.cnn_agent.last_update.batch_size,
                    "loss": f"{self.cnn_agent.last_update.loss:.5f}",
                    "target synced": self.cnn_agent.last_update.target_synced,
                }
            )
        return metrics

    @property
    def _average_snake_length_50(self) -> float:
        if not self._episode_snake_lengths:
            return 0.0
        recent = tuple(self._episode_snake_lengths)[-50:]
        return sum(recent) / len(recent)

    def _record_episode_length(
        self,
        length: int,
        *,
        q_table_coverage: float | None = None,
        loss: float | None = None,
    ) -> None:
        self._episode_snake_lengths.append(length)
        previous_max = self._running_max_lengths[-1] if self._running_max_lengths else 0
        self._running_max_lengths.append(float(max(previous_max, length)))
        recent = self._episode_snake_lengths[-50:]
        self._rolling_average_lengths.append(sum(recent) / len(recent))
        if q_table_coverage is not None:
            self._coverage_history.append(q_table_coverage)
        if loss is not None:
            self._loss_history.append(loss)

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
