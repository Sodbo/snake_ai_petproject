"""Run a random agent against the Snake environment."""

from __future__ import annotations

import argparse
import random
import time

from snake_ai.agents.results import EpisodeResult, ScoreTracker
from snake_ai.game import DEFAULT_STEP_PENALTY, Action, SnakeGame


def run_episode(
    game: SnakeGame,
    *,
    rng: random.Random | None = None,
    render: bool = False,
    delay: float = 0.0,
) -> EpisodeResult:
    """Play one episode by selecting every action uniformly at random."""
    action_rng = rng or random.Random()
    state = game.reset()
    total_reward = 0.0

    if render:
        _render_frame(game, episode=None, action=None, reward=0.0)

    while not state.done:
        action = action_rng.choice(tuple(Action))
        state, reward, _, _ = game.step(action)
        total_reward += reward
        if render:
            _render_frame(game, episode=None, action=action, reward=reward)
            if delay > 0:
                time.sleep(delay)

    return EpisodeResult(state.score, state.steps, total_reward)


def run_random_agent(
    *,
    episodes: int = 10,
    width: int = 20,
    height: int = 20,
    seed: int | None = None,
    render: bool = False,
    delay: float = 0.0,
    step_penalty: float = DEFAULT_STEP_PENALTY,
) -> ScoreTracker:
    """Run random-agent episodes and return their score tracker."""
    if episodes < 1:
        raise ValueError("episodes must be at least 1")
    if delay < 0:
        raise ValueError("delay cannot be negative")

    game = SnakeGame(width, height, seed=seed, step_penalty=step_penalty)
    rng = random.Random(seed)
    tracker = ScoreTracker()

    for episode in range(1, episodes + 1):
        result = run_episode(game, rng=rng, render=render, delay=delay)
        tracker.add(result)
        print(
            f"Episode {episode}: score={result.score}, "
            f"steps={result.steps}, reward={result.total_reward:.2f}"
        )

    print(tracker.summary())
    return tracker


def _render_frame(
    game: SnakeGame,
    *,
    episode: int | None,
    action: Action | None,
    reward: float,
) -> None:
    print("\x1b[2J\x1b[H", end="")
    state = game.state
    game.render()
    action_name = action.name.lower() if action is not None else "-"
    episode_text = str(episode) if episode is not None else "-"
    print(
        f"Episode: {episode_text} | Step: {state.steps} | Score: {state.score} | "
        f"Reward: {reward:.2f} | Action: {action_name}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Snake with random actions.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--step-penalty", type=float, default=DEFAULT_STEP_PENALTY)
    parser.add_argument(
        "--delay",
        type=float,
        default=0.05,
        help="Seconds between rendered steps.",
    )
    args = parser.parse_args()
    run_random_agent(
        episodes=args.episodes,
        width=args.width,
        height=args.height,
        seed=args.seed,
        render=args.render,
        delay=args.delay,
        step_penalty=args.step_penalty,
    )


if __name__ == "__main__":
    main()
