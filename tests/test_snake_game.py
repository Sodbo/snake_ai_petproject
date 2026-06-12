import contextlib
from dataclasses import FrozenInstanceError
import io
import unittest

from snake_ai.game import Action, Direction, GameState, SnakeGame


class SnakeGameTests(unittest.TestCase):
    def test_defaults_and_reset_state(self) -> None:
        game = SnakeGame(seed=1)

        state = game.state

        self.assertIsInstance(state, GameState)
        self.assertEqual((state.width, state.height), (20, 20))
        self.assertEqual(state.direction, Direction.RIGHT)
        self.assertEqual(state.score, 0)
        self.assertEqual(state.steps, 0)
        self.assertFalse(state.done)
        self.assertEqual(len(state.snake), 3)
        self.assertNotIn(state.food, state.snake)

    def test_custom_board_size(self) -> None:
        for width, height in ((8, 8), (10, 10), (30, 30), (50, 50)):
            with self.subTest(width=width, height=height):
                state = SnakeGame(width, height).state
                self.assertEqual((state.width, state.height), (width, height))

    def test_relative_actions_change_direction_and_position(self) -> None:
        game = SnakeGame(8, 8, seed=1)

        state, _, _, _ = game.step(Action.LEFT)
        self.assertEqual(state.direction, Direction.UP)
        self.assertEqual(state.head, (4, 3))

        state, _, _, _ = game.step(Action.RIGHT)
        self.assertEqual(state.direction, Direction.RIGHT)
        self.assertEqual(state.head, (5, 3))

        state, _, _, _ = game.step(Action.STRAIGHT)
        self.assertEqual(state.direction, Direction.RIGHT)
        self.assertEqual(state.head, (6, 3))

    def test_collision_ends_episode(self) -> None:
        game = SnakeGame(3, 3, seed=1)

        state, reward, done, info = game.step(Action.STRAIGHT)

        self.assertTrue(done)
        self.assertTrue(state.done)
        self.assertEqual(reward, -10.0)
        self.assertEqual(info["event"], "collision")
        with self.assertRaises(RuntimeError):
            game.step(Action.STRAIGHT)

    def test_food_is_reproducible_with_seed(self) -> None:
        first = SnakeGame(10, 10, seed=42)
        second = SnakeGame(10, 10, seed=42)

        self.assertEqual(first.state.food, second.state.food)
        self.assertEqual(first.reset(seed=7).food, second.reset(seed=7).food)

    def test_eating_food_grows_snake_and_increases_score(self) -> None:
        game = SnakeGame(8, 8, seed=1)
        game._food = (5, 4)

        state, reward, done, info = game.step(Action.STRAIGHT)

        self.assertEqual(state.head, (5, 4))
        self.assertEqual(len(state.snake), 4)
        self.assertEqual(state.score, 1)
        self.assertEqual(reward, 10.0)
        self.assertFalse(done)
        self.assertEqual(info["event"], "ate_food")

    def test_state_snapshot_is_immutable(self) -> None:
        state = SnakeGame(seed=1).state

        with self.assertRaises(FrozenInstanceError):
            state.score = 10

    def test_invalid_configuration_and_action(self) -> None:
        for size in (True, 2, 3.5):
            with self.subTest(size=size):
                with self.assertRaises(ValueError):
                    SnakeGame(size, 10)

        game = SnakeGame()
        for action in (True, -1, 1.0, 3, "left"):
            with self.subTest(action=action):
                with self.assertRaises(ValueError):
                    game.step(action)

    def test_render_returns_and_prints_board(self) -> None:
        game = SnakeGame(8, 8, seed=1)
        stdout = io.StringIO()

        with contextlib.redirect_stdout(stdout):
            rendered = game.render()

        self.assertEqual(stdout.getvalue().rstrip(), rendered)
        self.assertIn("H", rendered)
        self.assertIn("*", rendered)
        self.assertEqual(len(rendered.splitlines()), 10)


if __name__ == "__main__":
    unittest.main()
