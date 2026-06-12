"""PyTorch CNN-DQN agent that learns directly from full-board spatial inputs."""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from snake_ai.agents.dqn import (
    DEFAULT_GRADIENT_STEPS,
    DEFAULT_LEARNING_STARTS,
    DEFAULT_TRAIN_EVERY,
)
from snake_ai.agents.q_learning import ACTION_COUNT
from snake_ai.agents.results import EpisodeResult, ScoreTracker
from snake_ai.game import DEFAULT_STEP_PENALTY, Action, Direction, GameState, SnakeGame
from snake_ai.training.metrics import build_metrics, save_metrics

BOARD_CHANNELS = 7


def encode_board(state: GameState) -> np.ndarray:
    """Encode head, body, food, and direction as full-board channels."""
    board = np.zeros((BOARD_CHANNELS, state.height, state.width), dtype=np.float32)
    head_x, head_y = state.head
    board[0, head_y, head_x] = 1.0
    for x, y in state.snake[1:]:
        board[1, y, x] = 1.0
    if state.food is not None:
        food_x, food_y = state.food
        board[2, food_y, food_x] = 1.0
    board[3 + int(state.direction), :, :] = 1.0
    return board


class CNNQNetwork(nn.Module):
    """Convolutional Q-network that supports arbitrary board dimensions."""

    def __init__(self, action_count: int = ACTION_COUNT) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(BOARD_CHANNELS, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
        )
        self.values = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 128),
            nn.ReLU(),
            nn.Linear(128, action_count),
        )

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        return self.values(self.features(states))


@dataclass(frozen=True)
class CNNExperience:
    state: np.ndarray
    action: Action
    reward: float
    next_state: np.ndarray
    done: bool


@dataclass(frozen=True)
class CNNUpdate:
    loss: float
    batch_size: int
    replay_size: int
    target_synced: bool


class CNNReplayBuffer:
    def __init__(self, capacity: int, *, seed: int | None = None) -> None:
        if capacity < 1:
            raise ValueError("capacity must be at least 1")
        self._items: deque[CNNExperience] = deque(maxlen=capacity)
        self._rng = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def append(self, experience: CNNExperience) -> None:
        self._items.append(experience)

    def sample(self, size: int) -> list[CNNExperience]:
        return self._rng.sample(list(self._items), size)


class CNNDQNAgent:
    """Double DQN agent backed by a PyTorch convolutional network."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.0005,
        discount: float = 0.95,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.9999,
        replay_capacity: int = 50_000,
        batch_size: int = 128,
        target_sync_interval: int = 1_000,
        learning_starts: int = 5_000,
        train_every: int = DEFAULT_TRAIN_EVERY,
        gradient_steps: int = DEFAULT_GRADIENT_STEPS,
        device: str = "auto",
        seed: int | None = None,
    ) -> None:
        if learning_starts > replay_capacity:
            raise ValueError("learning_starts cannot exceed replay_capacity")
        for name, value in (
            ("batch_size", batch_size),
            ("target_sync_interval", target_sync_interval),
            ("learning_starts", learning_starts),
            ("train_every", train_every),
            ("gradient_steps", gradient_steps),
        ):
            if value < 1:
                raise ValueError(f"{name} must be at least 1")
        if seed is not None:
            torch.manual_seed(seed)
        self.device = torch.device(
            "cuda" if device == "auto" and torch.cuda.is_available() else
            "cpu" if device == "auto" else device
        )
        self.online_network = CNNQNetwork().to(self.device)
        self.target_network = CNNQNetwork().to(self.device)
        self.target_network.load_state_dict(self.online_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(
            self.online_network.parameters(), lr=learning_rate
        )
        self.replay_buffer = CNNReplayBuffer(replay_capacity, seed=seed)
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.replay_capacity = replay_capacity
        self.batch_size = batch_size
        self.target_sync_interval = target_sync_interval
        self.learning_starts = learning_starts
        self.train_every = train_every
        self.gradient_steps = gradient_steps
        self.environment_steps = 0
        self.training_steps = 0
        self.last_update: CNNUpdate | None = None
        self.loss_history: list[float] = []
        self.episode_loss_history: list[float] = []
        self._rng = random.Random(seed)

    def choose_action(self, state: np.ndarray) -> tuple[Action, bool]:
        if self._rng.random() < self.epsilon:
            return self._rng.choice(tuple(Action)), True
        values = self.q_values(state)
        return Action(int(np.argmax(values))), False

    def q_values(self, state: np.ndarray) -> tuple[float, float, float]:
        """Return current action values for one encoded board."""
        with torch.no_grad():
            tensor = torch.from_numpy(state).unsqueeze(0).to(self.device)
            values = self.online_network(tensor)[0]
        return tuple(float(value) for value in values.cpu().tolist())

    def remember(
        self,
        state: np.ndarray,
        action: Action,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.replay_buffer.append(CNNExperience(state, action, reward, next_state, done))
        self.environment_steps += 1

    def learn(self) -> CNNUpdate | None:
        if (
            len(self.replay_buffer) < max(self.batch_size, self.learning_starts)
            or self.environment_steps % self.train_every != 0
        ):
            return None
        for _ in range(self.gradient_steps):
            self._learn_once()
        return self.last_update

    def _learn_once(self) -> None:
        batch = self.replay_buffer.sample(self.batch_size)
        states = torch.from_numpy(np.stack([item.state for item in batch])).to(self.device)
        next_states = torch.from_numpy(
            np.stack([item.next_state for item in batch])
        ).to(self.device)
        actions = torch.tensor(
            [int(item.action) for item in batch], device=self.device
        ).unsqueeze(1)
        rewards = torch.tensor(
            [item.reward for item in batch], dtype=torch.float32, device=self.device
        )
        dones = torch.tensor(
            [item.done for item in batch], dtype=torch.float32, device=self.device
        )

        predicted = self.online_network(states).gather(1, actions).squeeze(1)
        with torch.no_grad():
            next_actions = self.online_network(next_states).argmax(dim=1, keepdim=True)
            future = self.target_network(next_states).gather(1, next_actions).squeeze(1)
            targets = rewards + self.discount * future * (1.0 - dones)
        loss = F.smooth_l1_loss(predicted, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online_network.parameters(), max_norm=10.0)
        self.optimizer.step()
        self.training_steps += 1
        target_synced = self.training_steps % self.target_sync_interval == 0
        if target_synced:
            self.target_network.load_state_dict(self.online_network.state_dict())
        loss_value = float(loss.item())
        self.loss_history.append(loss_value)
        self.last_update = CNNUpdate(
            loss_value, self.batch_size, len(self.replay_buffer), target_synced
        )

    def finish_episode(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save_checkpoint(
        self, output: str | Path, *, metadata: dict[str, object] | None = None
    ) -> Path:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format_version": 1,
                "model_state_dict": self.online_network.state_dict(),
                "metadata": {
                    "algorithm": "CNN Double DQN (PyTorch)",
                    "training_steps": self.training_steps,
                    "environment_steps": self.environment_steps,
                    **(metadata or {}),
                },
            },
            path,
        )
        return path


def train_cnn_dqn(
    *,
    episodes: int = 10_000,
    width: int = 20,
    height: int = 20,
    seed: int | None = None,
    step_penalty: float = DEFAULT_STEP_PENALTY,
    learning_rate: float = 0.0005,
    discount: float = 0.95,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.9999,
    replay_capacity: int = 50_000,
    batch_size: int = 128,
    target_sync_interval: int = 1_000,
    learning_starts: int = 5_000,
    train_every: int = DEFAULT_TRAIN_EVERY,
    gradient_steps: int = DEFAULT_GRADIENT_STEPS,
    device: str = "auto",
    report_every: int = 100,
) -> tuple[CNNDQNAgent, ScoreTracker]:
    game = SnakeGame(width, height, seed=seed, step_penalty=step_penalty)
    agent = CNNDQNAgent(
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
        device=device,
        seed=seed,
    )
    tracker = ScoreTracker()
    for episode in range(1, episodes + 1):
        state = game.reset()
        total_reward = 0.0
        losses: list[float] = []
        while not state.done:
            encoded = encode_board(state)
            action, _ = agent.choose_action(encoded)
            next_state, reward, done, _ = game.step(action)
            agent.remember(encoded, action, reward, encode_board(next_state), done)
            previous = len(agent.loss_history)
            agent.learn()
            losses.extend(agent.loss_history[previous:])
            total_reward += reward
            state = next_state
        tracker.add(EpisodeResult(state.score, state.steps, total_reward))
        agent.episode_loss_history.append(sum(losses) / len(losses) if losses else 0.0)
        agent.finish_episode()
        if episode % report_every == 0 or episode == episodes:
            recent = tracker.results[-report_every:]
            average = sum(result.score for result in recent) / len(recent)
            loss = agent.last_update.loss if agent.last_update else 0.0
            print(
                f"Episode {episode}: recent_average={average:.2f}, "
                f"best={tracker.best_score}, epsilon={agent.epsilon:.3f}, "
                f"loss={loss:.5f}, replay={len(agent.replay_buffer)}, "
                f"device={agent.device}"
            )
    return agent, tracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PyTorch CNN Double DQN.")
    parser.add_argument("--episodes", type=int, default=10_000)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--step-penalty", type=float, default=DEFAULT_STEP_PENALTY)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--discount", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.9999)
    parser.add_argument("--replay-capacity", type=int, default=50_000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--target-sync-interval", type=int, default=1_000)
    parser.add_argument("--learning-starts", type=int, default=5_000)
    parser.add_argument("--train-every", type=int, default=DEFAULT_TRAIN_EVERY)
    parser.add_argument("--gradient-steps", type=int, default=DEFAULT_GRADIENT_STEPS)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--output")
    parser.add_argument("--checkpoint")
    args = parser.parse_args()
    agent, tracker = train_cnn_dqn(**{
        key: value for key, value in vars(args).items()
        if key not in ("output", "checkpoint")
    })
    print(tracker.summary())
    config = {**vars(args), "device_used": str(agent.device)}
    if args.checkpoint:
        print(f"Checkpoint saved to {agent.save_checkpoint(args.checkpoint, metadata=config)}")
    if args.output:
        output = save_metrics(
            build_metrics(
                "CNN Double DQN (PyTorch)",
                tracker.results,
                loss=agent.episode_loss_history,
                config=config,
            ),
            args.output,
        )
        print(f"Metrics saved to {output}")


if __name__ == "__main__":
    main()
