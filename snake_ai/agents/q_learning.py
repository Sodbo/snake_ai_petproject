"""Tabular Q-learning agent with an 11-bit Snake state."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import random

from snake_ai.agents.results import EpisodeResult, ScoreTracker
from snake_ai.game import Action, Direction, GameState, SnakeGame
from snake_ai.training.metrics import build_metrics, save_metrics

STATE_FEATURES = (
    "danger straight",
    "danger right",
    "danger left",
    "direction up",
    "direction right",
    "direction down",
    "direction left",
    "food left",
    "food right",
    "food up",
    "food down",
)
STATE_COUNT = 2 ** len(STATE_FEATURES)
ACTION_COUNT = len(Action)

_DIRECTION_VECTORS = {
    Direction.UP: (0, -1),
    Direction.RIGHT: (1, 0),
    Direction.DOWN: (0, 1),
    Direction.LEFT: (-1, 0),
}
_ACTION_TURNS = {
    Action.LEFT: -1,
    Action.STRAIGHT: 0,
    Action.RIGHT: 1,
}


@dataclass(frozen=True)
class EncodedState:
    """Binary features and their integer Q-table row ID."""

    features: tuple[int, ...]
    state_id: int


@dataclass(frozen=True)
class QLearningUpdate:
    """Values involved in the most recent Q-learning update."""

    state_id: int
    next_state_id: int
    action: Action
    exploratory: bool
    old_value: float
    reward: float
    target: float
    new_value: float


def encode_state(state: GameState) -> EncodedState:
    """Convert a full game state into the compact 11-bit representation."""
    head_x, head_y = state.head
    food_x, food_y = state.food if state.food is not None else state.head
    features = (
        int(_danger(state, Action.STRAIGHT)),
        int(_danger(state, Action.RIGHT)),
        int(_danger(state, Action.LEFT)),
        int(state.direction == Direction.UP),
        int(state.direction == Direction.RIGHT),
        int(state.direction == Direction.DOWN),
        int(state.direction == Direction.LEFT),
        int(food_x < head_x),
        int(food_x > head_x),
        int(food_y < head_y),
        int(food_y > head_y),
    )
    state_id = 0
    for feature in features:
        state_id = (state_id << 1) | feature
    return EncodedState(features, state_id)


def _danger(state: GameState, action: Action) -> bool:
    return danger_at_distance(state, action, distance=1)


def danger_at_distance(state: GameState, action: Action, *, distance: int) -> bool:
    """Return whether a relative-direction cell is currently blocked."""
    if distance < 1:
        raise ValueError("distance must be at least 1")
    direction = Direction(
        (int(state.direction) + _ACTION_TURNS[action]) % len(Direction)
    )
    dx, dy = _DIRECTION_VECTORS[direction]
    new_head = (state.head[0] + dx * distance, state.head[1] + dy * distance)
    x, y = new_head
    if x < 0 or x >= state.width or y < 0 or y >= state.height:
        return True
    growing = distance == 1 and new_head == state.food
    if distance > 1:
        body = state.snake[1:]
    else:
        body = state.snake if growing else state.snake[:-1]
    return new_head in body


class QLearningAgent:
    """Learn action values for each encoded Snake state."""

    def __init__(
        self,
        *,
        learning_rate: float = 0.1,
        discount: float = 0.9,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        state_count: int = STATE_COUNT,
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
        if state_count < 1:
            raise ValueError("state_count must be at least 1")

        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.state_count = state_count
        self.q_table = [
            [0.0 for _ in range(ACTION_COUNT)] for _ in range(state_count)
        ]
        self.last_update: QLearningUpdate | None = None
        self._rng = random.Random(seed)

    def choose_action(self, state_id: int) -> tuple[Action, bool]:
        """Choose an action using an epsilon-greedy policy."""
        if self._rng.random() < self.epsilon:
            return self._rng.choice(tuple(Action)), True
        values = self.q_table[state_id]
        best = max(values)
        choices = [Action(index) for index, value in enumerate(values) if value == best]
        return self._rng.choice(choices), False

    def update(
        self,
        state_id: int,
        action: Action,
        reward: float,
        next_state_id: int,
        done: bool,
        *,
        exploratory: bool = False,
    ) -> QLearningUpdate:
        """Apply one Q-learning update and return its calculation."""
        old_value = self.q_table[state_id][action]
        future_value = 0.0 if done else max(self.q_table[next_state_id])
        target = reward + self.discount * future_value
        new_value = old_value + self.learning_rate * (target - old_value)
        self.q_table[state_id][action] = new_value
        self.last_update = QLearningUpdate(
            state_id=state_id,
            next_state_id=next_state_id,
            action=action,
            exploratory=exploratory,
            old_value=old_value,
            reward=reward,
            target=target,
            new_value=new_value,
        )
        return self.last_update

    def finish_episode(self) -> None:
        """Decay exploration after a completed episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


def train_q_learning(
    *,
    episodes: int = 1000,
    width: int = 20,
    height: int = 20,
    seed: int | None = None,
    learning_rate: float = 0.1,
    discount: float = 0.9,
    epsilon: float = 1.0,
    epsilon_min: float = 0.05,
    epsilon_decay: float = 0.995,
    report_every: int = 100,
) -> tuple[QLearningAgent, ScoreTracker]:
    """Train a tabular agent without opening the dashboard."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if report_every < 1:
        raise ValueError("report_every must be at least 1")

    game = SnakeGame(width, height, seed=seed)
    agent = QLearningAgent(
        learning_rate=learning_rate,
        discount=discount,
        epsilon=epsilon,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay,
        seed=seed,
    )
    tracker = ScoreTracker()

    for episode in range(1, episodes + 1):
        state = game.reset()
        total_reward = 0.0
        while not state.done:
            encoded = encode_state(state)
            action, exploratory = agent.choose_action(encoded.state_id)
            next_state, reward, done, _ = game.step(action)
            next_encoded = encode_state(next_state)
            agent.update(
                encoded.state_id,
                action,
                reward,
                next_encoded.state_id,
                done,
                exploratory=exploratory,
            )
            total_reward += reward
            state = next_state

        tracker.add(EpisodeResult(state.score, state.steps, total_reward))
        agent.finish_episode()
        if episode % report_every == 0 or episode == episodes:
            recent = tracker.results[-report_every:]
            average = sum(result.score for result in recent) / len(recent)
            print(
                f"Episode {episode}: recent_average={average:.2f}, "
                f"best={tracker.best_score}, epsilon={agent.epsilon:.3f}"
            )

    return agent, tracker


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a tabular Q-learning agent.")
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--discount", type=float, default=0.9)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--report-every", type=int, default=100)
    parser.add_argument("--output", help="Write episode metrics to this JSON file.")
    args = parser.parse_args()
    _, tracker = train_q_learning(
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        seed=args.seed,
        learning_rate=args.learning_rate,
        discount=args.discount,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        report_every=args.report_every,
    )
    print(tracker.summary())
    if args.output:
        output = save_metrics(
            build_metrics(
                "Q-Learning",
                tracker.results,
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
                },
            ),
            args.output,
        )
        print(f"Metrics saved to {output}")


if __name__ == "__main__":
    main()
