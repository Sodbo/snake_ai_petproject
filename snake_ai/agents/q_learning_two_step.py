"""Tabular Q-learning agent with one-step and two-step danger inputs."""

from __future__ import annotations

import argparse

from snake_ai.agents.q_learning import (
    EncodedState,
    QLearningAgent,
    danger_at_distance,
)
from snake_ai.agents.results import EpisodeResult, ScoreTracker
from snake_ai.game import Action, Direction, GameState, SnakeGame
from snake_ai.training.metrics import build_metrics, save_metrics

STATE_FEATURES = (
    "danger forward-1",
    "danger left-1",
    "danger right-1",
    "danger forward-2",
    "danger left-2",
    "danger right-2",
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


def encode_two_step_state(state: GameState) -> EncodedState:
    """Encode Snake using immediate and distance-two relative danger."""
    head_x, head_y = state.head
    food_x, food_y = state.food if state.food is not None else state.head
    features = (
        int(danger_at_distance(state, Action.STRAIGHT, distance=1)),
        int(danger_at_distance(state, Action.LEFT, distance=1)),
        int(danger_at_distance(state, Action.RIGHT, distance=1)),
        int(danger_at_distance(state, Action.STRAIGHT, distance=2)),
        int(danger_at_distance(state, Action.LEFT, distance=2)),
        int(danger_at_distance(state, Action.RIGHT, distance=2)),
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


class TwoStepDangerQLearningAgent(QLearningAgent):
    """Q-learning agent using the expanded 14-bit state."""

    def __init__(self, **kwargs: object) -> None:
        super().__init__(state_count=STATE_COUNT, **kwargs)


def train_two_step_q_learning(
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
) -> tuple[TwoStepDangerQLearningAgent, ScoreTracker]:
    """Train the expanded-state agent without opening the dashboard."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if report_every < 1:
        raise ValueError("report_every must be at least 1")

    game = SnakeGame(width, height, seed=seed)
    agent = TwoStepDangerQLearningAgent(
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
            encoded = encode_two_step_state(state)
            action, exploratory = agent.choose_action(encoded.state_id)
            next_state, reward, done, _ = game.step(action)
            next_encoded = encode_two_step_state(next_state)
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
    parser = argparse.ArgumentParser(
        description="Train Q-learning with one-step and two-step danger."
    )
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
    _, tracker = train_two_step_q_learning(
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
                "Q-Learning 2-Step",
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
