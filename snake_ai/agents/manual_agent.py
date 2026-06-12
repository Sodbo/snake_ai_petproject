"""Play Snake manually in a terminal."""

from __future__ import annotations

import argparse
import os
import sys

from snake_ai.agents.results import EpisodeResult, ScoreTracker
from snake_ai.game import DEFAULT_STEP_PENALTY, Action, Direction, SnakeGame

_ARROW_KEYS = {
    "up": Direction.UP,
    "right": Direction.RIGHT,
    "down": Direction.DOWN,
    "left": Direction.LEFT,
}


def action_for_direction(
    current: Direction, requested: Direction
) -> Action | None:
    """Convert an absolute direction request into a legal relative action."""
    difference = (int(requested) - int(current)) % len(Direction)
    if difference == 0:
        return Action.STRAIGHT
    if difference == 1:
        return Action.RIGHT
    if difference == 3:
        return Action.LEFT
    return None


def action_for_key(key: str, direction: Direction) -> Action | None:
    """Map a manual-play key to a relative Snake action."""
    normalized = key.lower()
    if normalized == "a":
        return Action.LEFT
    if normalized == "d":
        return Action.RIGHT
    if normalized in ("w", "straight", "space"):
        return Action.STRAIGHT
    requested = _ARROW_KEYS.get(normalized)
    return action_for_direction(direction, requested) if requested is not None else None


def play_manual(
    *,
    width: int = 20,
    height: int = 20,
    seed: int | None = None,
    step_penalty: float = DEFAULT_STEP_PENALTY,
) -> ScoreTracker:
    """Play episodes until the player chooses to quit."""
    game = SnakeGame(width, height, seed=seed, step_penalty=step_penalty)
    tracker = ScoreTracker()

    while True:
        result, quit_requested = _play_episode(game, tracker.episodes + 1)
        if result is not None:
            tracker.add(result)
            print(
                f"Game over. Score: {result.score} | Steps: {result.steps} | "
                f"Total reward: {result.total_reward:.2f}"
            )
            print(tracker.summary())
        if quit_requested:
            return tracker
        print("Press R to play again or Q to quit.")
        if _read_key() != "r":
            return tracker


def _play_episode(
    game: SnakeGame, episode: int
) -> tuple[EpisodeResult | None, bool]:
    state = game.reset()
    total_reward = 0.0
    reward = 0.0
    action: Action | None = None

    while not state.done:
        _render_frame(game, episode, action, reward)
        key = _read_key()
        if key == "q":
            return None, True

        action = action_for_key(key, state.direction)
        if action is None:
            continue
        state, reward, _, _ = game.step(action)
        total_reward += reward

    _render_frame(game, episode, action, reward)
    return EpisodeResult(state.score, state.steps, total_reward), False


def _render_frame(
    game: SnakeGame, episode: int, action: Action | None, reward: float
) -> None:
    print("\x1b[2J\x1b[H", end="")
    game.render()
    state = game.state
    action_name = action.name.lower() if action is not None else "-"
    print(
        f"Episode: {episode} | Step: {state.steps} | Score: {state.score} | "
        f"Reward: {reward:.2f} | Action: {action_name}"
    )
    print("Arrow keys: move | A/D: turn | W/Space: straight | Q: quit")


def _read_key() -> str:
    if os.name == "nt":
        return _read_windows_key()
    return _read_posix_key()


def _read_windows_key() -> str:
    import msvcrt

    key = msvcrt.getwch()
    if key in ("\x00", "\xe0"):
        return {
            "H": "up",
            "M": "right",
            "P": "down",
            "K": "left",
        }.get(msvcrt.getwch(), "")
    if key == " ":
        return "space"
    return key.lower()


def _read_posix_key() -> str:
    import termios
    import tty

    file_descriptor = sys.stdin.fileno()
    previous_settings = termios.tcgetattr(file_descriptor)
    try:
        tty.setraw(file_descriptor)
        key = sys.stdin.read(1)
        if key == "\x1b":
            sequence = sys.stdin.read(2)
            return {
                "[A": "up",
                "[C": "right",
                "[B": "down",
                "[D": "left",
            }.get(sequence, "")
        if key == " ":
            return "space"
        return key.lower()
    finally:
        termios.tcsetattr(file_descriptor, termios.TCSADRAIN, previous_settings)


def main() -> None:
    parser = argparse.ArgumentParser(description="Play Snake in the terminal.")
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--step-penalty", type=float, default=DEFAULT_STEP_PENALTY)
    args = parser.parse_args()
    play_manual(
        width=args.width,
        height=args.height,
        seed=args.seed,
        step_penalty=args.step_penalty,
    )


if __name__ == "__main__":
    main()
