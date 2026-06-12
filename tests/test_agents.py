import contextlib
import io
import random
import unittest

from snake_ai.agents.manual_agent import action_for_direction, action_for_key
from snake_ai.agents.random_agent import run_episode, run_random_agent
from snake_ai.agents.results import EpisodeResult, ScoreTracker
from snake_ai.game import Action, Direction, SnakeGame


class ScoreTrackerTests(unittest.TestCase):
    def test_tracks_basic_score_statistics(self) -> None:
        tracker = ScoreTracker()
        tracker.add(EpisodeResult(score=1, steps=5, total_reward=0.0))
        tracker.add(EpisodeResult(score=3, steps=10, total_reward=20.0))

        self.assertEqual(tracker.episodes, 2)
        self.assertEqual(tracker.best_score, 3)
        self.assertEqual(tracker.average_score, 2.0)
        self.assertIn("Average score: 2.00", tracker.summary())


class RandomAgentTests(unittest.TestCase):
    def test_random_episode_runs_to_completion(self) -> None:
        game = SnakeGame(5, 5, seed=1)

        result = run_episode(game, rng=random.Random(1))

        self.assertIsInstance(result, EpisodeResult)
        self.assertTrue(game.state.done)
        self.assertEqual(result.score, game.state.score)
        self.assertEqual(result.steps, game.state.steps)

    def test_random_agent_tracks_requested_episodes(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            tracker = run_random_agent(episodes=3, width=5, height=5, seed=1)

        self.assertEqual(tracker.episodes, 3)

    def test_random_agent_rejects_invalid_options(self) -> None:
        with self.assertRaises(ValueError):
            run_random_agent(episodes=0)
        with self.assertRaises(ValueError):
            run_random_agent(delay=-1)


class ManualAgentTests(unittest.TestCase):
    def test_absolute_directions_become_relative_actions(self) -> None:
        self.assertEqual(
            action_for_direction(Direction.UP, Direction.LEFT), Action.LEFT
        )
        self.assertEqual(
            action_for_direction(Direction.UP, Direction.RIGHT), Action.RIGHT
        )
        self.assertEqual(
            action_for_direction(Direction.UP, Direction.UP), Action.STRAIGHT
        )
        self.assertIsNone(action_for_direction(Direction.UP, Direction.DOWN))

    def test_manual_keys_map_to_actions(self) -> None:
        self.assertEqual(action_for_key("a", Direction.RIGHT), Action.LEFT)
        self.assertEqual(action_for_key("space", Direction.RIGHT), Action.STRAIGHT)
        self.assertEqual(action_for_key("left", Direction.DOWN), Action.RIGHT)
        self.assertEqual(action_for_key("down", Direction.RIGHT), Action.RIGHT)
        self.assertIsNone(action_for_key("up", Direction.DOWN))
        self.assertIsNone(action_for_key("x", Direction.RIGHT))


if __name__ == "__main__":
    unittest.main()
