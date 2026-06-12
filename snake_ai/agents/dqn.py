"""Deep Q-network agent implemented with NumPy."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Sequence

import numpy as np

from snake_ai.agents.q_learning import ACTION_COUNT
from snake_ai.agents.results import EpisodeResult, ScoreTracker
from snake_ai.game import DEFAULT_STEP_PENALTY, Action, GameState, SnakeGame
from snake_ai.training.metrics import build_metrics, save_metrics

DEFAULT_SIGHT_DISTANCE = 1
DEFAULT_LEARNING_STARTS = 1_000
DEFAULT_TRAIN_EVERY = 4
DEFAULT_GRADIENT_STEPS = 1
POSITION_FEATURE_COUNT = 8
INPUT_COUNT = (1 + 2 * DEFAULT_SIGHT_DISTANCE) ** 2 - 1 + POSITION_FEATURE_COUNT


def sight_cell_count(sight_distance: int) -> int:
    """Return the number of cells in a square sight area, excluding the head."""
    if (
        isinstance(sight_distance, bool)
        or not isinstance(sight_distance, int)
        or sight_distance < 1
    ):
        raise ValueError("sight_distance must be an integer of at least 1")
    return (1 + 2 * sight_distance) ** 2 - 1


def dqn_input_count(sight_distance: int) -> int:
    """Return sight inputs plus four direction and four food-position inputs."""
    return sight_cell_count(sight_distance) + POSITION_FEATURE_COUNT


def encode_dqn_state(state: GameState, sight_distance: int = 1) -> tuple[float, ...]:
    """Encode blocked surrounding cells, direction, and relative food position."""
    sight_cell_count(sight_distance)
    head_x, head_y = state.head
    blocked = set(state.snake[1:])
    sight: list[float] = []
    for dy in range(-sight_distance, sight_distance + 1):
        for dx in range(-sight_distance, sight_distance + 1):
            if dx == 0 and dy == 0:
                continue
            position = (head_x + dx, head_y + dy)
            x, y = position
            sight.append(
                float(
                    x < 0
                    or x >= state.width
                    or y < 0
                    or y >= state.height
                    or position in blocked
                )
            )
    food_x, food_y = state.food if state.food is not None else state.head
    return (
        *sight,
        float(state.direction.value == 0),
        float(state.direction.value == 1),
        float(state.direction.value == 2),
        float(state.direction.value == 3),
        float(food_x < head_x),
        float(food_x > head_x),
        float(food_y < head_y),
        float(food_y > head_y),
    )


@dataclass(frozen=True)
class Experience:
    """One state transition stored for replay."""

    state: tuple[float, ...]
    action: Action
    reward: float
    next_state: tuple[float, ...]
    done: bool


@dataclass(frozen=True)
class DQNUpdate:
    """Summary of the latest mini-batch learning step."""

    loss: float
    batch_size: int
    replay_size: int
    target_synced: bool


class ReplayBuffer:
    """Fixed-size memory sampled uniformly for mini-batch learning."""

    def __init__(self, capacity: int, *, seed: int | None = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._items: deque[Experience] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, experience: Experience) -> None:
        self._items.append(experience)

    def sample(self, size: int) -> list[Experience]:
        if size < 1:
            raise ValueError("sample size must be at least 1")
        if size > len(self._items):
            raise ValueError("sample size cannot exceed replay buffer size")
        return self._rng.sample(list(self._items), size)


class QNetwork:
    """Small fully connected network ending in one Q-value per action."""

    def __init__(
        self,
        layer_sizes: Sequence[int] = (INPUT_COUNT, 64, 64, ACTION_COUNT),
        *,
        seed: int | None = None,
    ) -> None:
        if len(layer_sizes) < 2 or any(size < 1 for size in layer_sizes):
            raise ValueError("layer_sizes must contain at least two positive sizes")
        self.layer_sizes = tuple(layer_sizes)
        rng = np.random.default_rng(seed)
        self.weights = [
            rng.standard_normal((input_size, output_size)).astype(np.float32)
            * np.sqrt(2.0 / input_size)
            for input_size, output_size in zip(layer_sizes[:-1], layer_sizes[1:])
        ]
        self.biases = [
            np.zeros((1, output_size), dtype=np.float32)
            for output_size in layer_sizes[1:]
        ]

    def predict(self, states: np.ndarray) -> np.ndarray:
        """Return Q-values for a batch of states."""
        activations = np.asarray(states, dtype=np.float32)
        if activations.ndim == 1:
            activations = activations.reshape(1, -1)
        for index, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            activations = activations @ weights + biases
            if index < len(self.weights) - 1:
                activations = np.maximum(activations, 0.0)
        return activations

    def train(self, states: np.ndarray, targets: np.ndarray, learning_rate: float) -> float:
        """Apply one full-batch gradient descent step and return MSE loss."""
        activations = [np.asarray(states, dtype=np.float32)]
        pre_activations: list[np.ndarray] = []
        current = activations[0]
        for index, (weights, biases) in enumerate(zip(self.weights, self.biases)):
            before_activation = current @ weights + biases
            pre_activations.append(before_activation)
            current = (
                np.maximum(before_activation, 0.0)
                if index < len(self.weights) - 1
                else before_activation
            )
            activations.append(current)

        difference = activations[-1] - np.asarray(targets, dtype=np.float32)
        loss = float(np.mean(np.square(difference)))
        gradient = 2.0 * difference / difference.size

        for index in range(len(self.weights) - 1, -1, -1):
            weight_gradient = activations[index].T @ gradient
            bias_gradient = np.sum(gradient, axis=0, keepdims=True)
            previous_gradient = gradient @ self.weights[index].T
            self.weights[index] -= learning_rate * weight_gradient
            self.biases[index] -= learning_rate * bias_gradient
            if index > 0:
                gradient = previous_gradient * (pre_activations[index - 1] > 0)
        return loss

    def copy_from(self, other: QNetwork) -> None:
        """Copy all parameters from another network with the same architecture."""
        if self.layer_sizes != other.layer_sizes:
            raise ValueError("network architectures must match")
        for target, source in zip(self.weights, other.weights):
            target[:] = source
        for target, source in zip(self.biases, other.biases):
            target[:] = source


class DQNAgent:
    """Learn Q-values with a neural network and replayed transitions."""

    def __init__(
        self,
        *,
        hidden_sizes: Sequence[int] = (64, 64),
        learning_rate: float = 0.001,
        discount: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        replay_capacity: int = 10_000,
        batch_size: int = 64,
        target_sync_interval: int = 100,
        learning_starts: int = DEFAULT_LEARNING_STARTS,
        train_every: int = DEFAULT_TRAIN_EVERY,
        gradient_steps: int = DEFAULT_GRADIENT_STEPS,
        sight_distance: int = DEFAULT_SIGHT_DISTANCE,
        seed: int | None = None,
    ) -> None:
        for name, value in (
            ("learning_rate", learning_rate),
            ("discount", discount),
            ("epsilon", epsilon),
            ("epsilon_min", epsilon_min),
            ("epsilon_decay", epsilon_decay),
        ):
            if not 0 <= value <= 1:
                raise ValueError(f"{name} must be between 0 and 1")
        if epsilon_min > epsilon:
            raise ValueError("epsilon_min cannot exceed epsilon")
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if target_sync_interval < 1:
            raise ValueError("target_sync_interval must be at least 1")
        for name, value in (
            ("learning_starts", learning_starts),
            ("train_every", train_every),
            ("gradient_steps", gradient_steps),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be an integer of at least 1")
        if learning_starts > replay_capacity:
            raise ValueError("learning_starts cannot exceed replay_capacity")

        self.sight_distance = sight_distance
        self.input_count = dqn_input_count(sight_distance)
        layer_sizes = (self.input_count, *hidden_sizes, ACTION_COUNT)
        self.online_network = QNetwork(layer_sizes, seed=seed)
        self.target_network = QNetwork(layer_sizes, seed=seed)
        self.target_network.copy_from(self.online_network)
        self.replay_buffer = ReplayBuffer(replay_capacity, seed=seed)
        self.replay_capacity = replay_capacity
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_sync_interval = target_sync_interval
        self.learning_starts = learning_starts
        self.train_every = train_every
        self.gradient_steps = gradient_steps
        self.environment_steps = 0
        self.training_steps = 0
        self.last_update: DQNUpdate | None = None
        self.loss_history: list[float] = []
        self.episode_loss_history: list[float] = []
        self._rng = random.Random(seed)

    def choose_action(self, state: Sequence[float]) -> tuple[Action, bool]:
        """Choose an action using an epsilon-greedy policy."""
        if self._rng.random() < self.epsilon:
            return self._rng.choice(tuple(Action)), True
        q_values = self.online_network.predict(np.asarray(state, dtype=np.float32))[0]
        best_indices = np.flatnonzero(q_values == np.max(q_values))
        return Action(self._rng.choice(best_indices.tolist())), False

    def remember(
        self,
        state: Sequence[float],
        action: Action,
        reward: float,
        next_state: Sequence[float],
        done: bool,
    ) -> None:
        """Store one transition in replay memory."""
        self.replay_buffer.append(
            Experience(tuple(state), action, reward, tuple(next_state), done)
        )
        self.environment_steps += 1

    def learn(self) -> DQNUpdate | None:
        """Train on schedule after replay memory has enough transitions."""
        if (
            len(self.replay_buffer) < max(self.batch_size, self.learning_starts)
            or self.environment_steps % self.train_every != 0
        ):
            return None
        update = None
        for _ in range(self.gradient_steps):
            update = self._learn_once()
        return update

    def _learn_once(self) -> DQNUpdate:
        """Sample replay memory and apply one gradient update."""
        batch = self.replay_buffer.sample(self.batch_size)
        states = np.asarray([item.state for item in batch], dtype=np.float32)
        next_states = np.asarray([item.next_state for item in batch], dtype=np.float32)
        targets = self.online_network.predict(states)
        next_q_values = self.target_network.predict(next_states)

        for index, item in enumerate(batch):
            future = 0.0 if item.done else float(np.max(next_q_values[index]))
            targets[index, item.action] = item.reward + self.discount * future

        loss = self.online_network.train(states, targets, self.learning_rate)
        self.training_steps += 1
        target_synced = self.training_steps % self.target_sync_interval == 0
        if target_synced:
            self.target_network.copy_from(self.online_network)
        self.loss_history.append(loss)
        self.last_update = DQNUpdate(
            loss=loss,
            batch_size=self.batch_size,
            replay_size=len(self.replay_buffer),
            target_synced=target_synced,
        )
        return self.last_update

    def finish_episode(self) -> None:
        """Decay exploration after a completed episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(
        self,
        output: str | Path,
        *,
        metadata: dict[str, object] | None = None,
    ) -> Path:
        """Save the online policy network and inference metadata."""
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_metadata: dict[str, object] = {
            "format_version": 1,
            "sight_distance": self.sight_distance,
            "layer_sizes": self.online_network.layer_sizes,
            "training_steps": self.training_steps,
            "environment_steps": self.environment_steps,
            "learning_starts": self.learning_starts,
            "train_every": self.train_every,
            "gradient_steps": self.gradient_steps,
            "replay_capacity": self.replay_capacity,
        }
        checkpoint_metadata.update(metadata or {})
        arrays: dict[str, np.ndarray] = {
            "metadata": np.asarray(json.dumps(checkpoint_metadata)),
        }
        for index, weights in enumerate(self.online_network.weights):
            arrays[f"weight_{index}"] = weights
        for index, biases in enumerate(self.online_network.biases):
            arrays[f"bias_{index}"] = biases
        with path.open("wb") as file:
            np.savez_compressed(file, **arrays)
        return path

    @classmethod
    def load_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        seed: int | None = None,
    ) -> tuple[DQNAgent, dict[str, object]]:
        """Load a policy checkpoint as a deterministic inference agent."""
        path = Path(checkpoint)
        with np.load(path, allow_pickle=False) as data:
            metadata = json.loads(str(data["metadata"].item()))
            if metadata.get("format_version") != 1:
                raise ValueError("unsupported DQN checkpoint format")
            layer_sizes = tuple(int(size) for size in metadata["layer_sizes"])
            sight_distance = int(metadata["sight_distance"])
            learning_starts = int(
                metadata.get("learning_starts", DEFAULT_LEARNING_STARTS)
            )
            agent = cls(
                hidden_sizes=layer_sizes[1:-1],
                sight_distance=sight_distance,
                epsilon=0.0,
                epsilon_min=0.0,
                replay_capacity=max(
                    int(metadata.get("replay_capacity", 10_000)), learning_starts
                ),
                learning_starts=learning_starts,
                train_every=int(metadata.get("train_every", DEFAULT_TRAIN_EVERY)),
                gradient_steps=int(
                    metadata.get("gradient_steps", DEFAULT_GRADIENT_STEPS)
                ),
                seed=seed,
            )
            if agent.online_network.layer_sizes != layer_sizes:
                raise ValueError("checkpoint architecture does not match sight distance")
            for index, target in enumerate(agent.online_network.weights):
                target[:] = data[f"weight_{index}"]
            for index, target in enumerate(agent.online_network.biases):
                target[:] = data[f"bias_{index}"]
        agent.target_network.copy_from(agent.online_network)
        agent.training_steps = int(metadata.get("training_steps", 0))
        agent.environment_steps = int(metadata.get("environment_steps", 0))
        return agent, metadata


def train_dqn(
    *,
    episodes: int = 1000,
    width: int = 20,
    height: int = 20,
    seed: int | None = None,
    learning_rate: float = 0.001,
    discount: float = 0.9,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    replay_capacity: int = 10_000,
    batch_size: int = 64,
    target_sync_interval: int = 100,
    learning_starts: int = DEFAULT_LEARNING_STARTS,
    train_every: int = DEFAULT_TRAIN_EVERY,
    gradient_steps: int = DEFAULT_GRADIENT_STEPS,
    sight_distance: int = DEFAULT_SIGHT_DISTANCE,
    step_penalty: float = DEFAULT_STEP_PENALTY,
    report_every: int = 100,
) -> tuple[DQNAgent, ScoreTracker]:
    """Train a DQN agent without opening the dashboard."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if report_every < 1:
        raise ValueError("report_every must be at least 1")

    game = SnakeGame(width, height, seed=seed, step_penalty=step_penalty)
    agent = DQNAgent(
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        replay_capacity=replay_capacity,
        batch_size=batch_size,
        target_sync_interval=target_sync_interval,
        learning_starts=learning_starts,
        train_every=train_every,
        gradient_steps=gradient_steps,
        sight_distance=sight_distance,
        seed=seed,
    )
    tracker = ScoreTracker()

    for episode in range(1, episodes + 1):
        state = game.reset()
        total_reward = 0.0
        episode_losses: list[float] = []
        while not state.done:
            features = encode_dqn_state(state, sight_distance)
            action, _ = agent.choose_action(features)
            next_state, reward, done, _ = game.step(action)
            agent.remember(
                features,
                action,
                reward,
                encode_dqn_state(next_state, sight_distance),
                done,
            )
            previous_loss_count = len(agent.loss_history)
            agent.learn()
            episode_losses.extend(agent.loss_history[previous_loss_count:])
            total_reward += reward
            state = next_state

        tracker.add(EpisodeResult(state.score, state.steps, total_reward))
        agent.episode_loss_history.append(
            sum(episode_losses) / len(episode_losses) if episode_losses else 0.0
        )
        agent.finish_episode()
        if episode % report_every == 0 or episode == episodes:
            recent = tracker.results[-report_every:]
            average = sum(result.score for result in recent) / len(recent)
            loss = agent.last_update.loss if agent.last_update else 0.0
            print(
                f"Episode {episode}: recent_average={average:.2f}, "
                f"best={tracker.best_score}, epsilon={agent.epsilon:.3f}, "
                f"loss={loss:.5f}, replay={len(agent.replay_buffer)}"
            )

    return agent, tracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a NumPy DQN agent.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--replay-capacity", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--target-sync-interval", type=int, default=100)
    parser.add_argument("--learning-starts", type=int, default=DEFAULT_LEARNING_STARTS)
    parser.add_argument("--train-every", type=int, default=DEFAULT_TRAIN_EVERY)
    parser.add_argument("--gradient-steps", type=int, default=DEFAULT_GRADIENT_STEPS)
    parser.add_argument("--sight-distance", type=int, default=DEFAULT_SIGHT_DISTANCE)
    parser.add_argument("--step-penalty", type=float, default=DEFAULT_STEP_PENALTY)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--output", help="Write episode metrics to this JSON file.")
    parser.add_argument("--checkpoint", help="Save the trained DQN policy to this file.")
    args = parser.parse_args()
    agent, tracker = train_dqn(
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        seed=args.seed,
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        replay_capacity=args.replay_capacity,
        batch_size=args.batch_size,
        target_sync_interval=args.target_sync_interval,
        learning_starts=args.learning_starts,
        train_every=args.train_every,
        gradient_steps=args.gradient_steps,
        sight_distance=args.sight_distance,
        step_penalty=args.step_penalty,
        report_every=args.report_every,
    )
    print(tracker.summary())
    if args.checkpoint:
        checkpoint = agent.save_checkpoint(
            args.checkpoint,
            metadata={
                "episodes": args.episodes,
                "width": args.width,
                "height": args.height,
                "step_penalty": args.step_penalty,
            },
        )
        print(f"Checkpoint saved to {checkpoint}")
    if args.output:
        output = save_metrics(
            build_metrics(
                "DQN (NumPy)",
                tracker.results,
                loss=agent.episode_loss_history,
                config={
                    "episodes": args.episodes,
                    "width": args.width,
                    "height": args.height,
                    "seed": args.seed,
                    "learning_rate": args.learning_rate,
                    "discount": args.discount,
                    "epsilon": args.epsilon,
                    "epsilon_min": args.epsilon_min,
                    "epsilon_decay": args.epsilon_decay,
                    "replay_capacity": args.replay_capacity,
                    "batch_size": args.batch_size,
                    "target_sync_interval": args.target_sync_interval,
                    "learning_starts": args.learning_starts,
                    "train_every": args.train_every,
                    "gradient_steps": args.gradient_steps,
                    "environment_steps": agent.environment_steps,
                    "sight_distance": args.sight_distance,
                    "input_count": agent.input_count,
                    "step_penalty": args.step_penalty,
                    "training_steps": agent.training_steps,
                },
            ),
            args.output,
        )
        print(f"Metrics saved to {output}")


if __name__ == "__main__":
    main()
