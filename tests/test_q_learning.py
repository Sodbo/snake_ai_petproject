import contextlib
import io
import unittest

from snake_ai.agents.q_learning import (
    ACTION_COUNT,
    STATE_COUNT,
    QLearningAgent,
    encode_state,
    train_q_learning,
)
from snake_ai.game import Action, Direction, GameState, SnakeGame


class StateEncodingTests(unittest.TestCase):
    def test_initial_state_encodes_direction_food_and_danger(self) -> None:
        game = SnakeGame(8, 8, seed=1)
        game._food = (2, 2)

        encoded = encode_state(game.state)

        self.assertEqual(
            encoded.features,
            (0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0),
        )
        self.assertEqual(encoded.state_id, int("00001001010", 2))

    def test_encoder_detects_relative_wall_danger(self) -> None:
        state = GameState(
            width=5,
            height=5,
            snake=((4, 2), (3, 2), (2, 2)),
            food=(0, 0),
            direction=Direction.RIGHT,
            score=0,
            steps=0,
            done=False,
        )

        encoded = encode_state(state)

        self.assertEqual(encoded.features[:3], (1, 0, 0))
        self.assertGreaterEqual(encoded.state_id, 0)
        self.assertLess(encoded.state_id, STATE_COUNT)


class QLearningAgentTests(unittest.TestCase):
    def test_q_table_is_zero_initialized(self) -> None:
        agent = QLearningAgent()

        self.assertEqual(len(agent.q_table), STATE_COUNT)
        self.assertEqual(len(agent.q_table[0]), ACTION_COUNT)
        self.assertTrue(all(value == 0.0 for row in agent.q_table for value in row))

    def test_q_table_accepts_custom_state_count(self) -> None:
        agent = QLearningAgent(state_count=8)

        self.assertEqual(agent.state_count, 8)
        self.assertEqual(len(agent.q_table), 8)

    def test_terminal_update_uses_reward_as_target(self) -> None:
        agent = QLearningAgent(learning_rate=0.1, discount=0.9)

        update = agent.update(10, Action.STRAIGHT, -10.0, 11, True)

        self.assertEqual(update.target, -10.0)
        self.assertEqual(update.new_value, -1.0)
        self.assertEqual(agent.q_table[10][Action.STRAIGHT], -1.0)

    def test_update_uses_best_next_q_value(self) -> None:
        agent = QLearningAgent(learning_rate=0.5, discount=0.9)
        agent.q_table[11] = [1.0, 4.0, 2.0]

        update = agent.update(10, Action.LEFT, 10.0, 11, False)

        self.assertAlmostEqual(update.target, 13.6)
        self.assertAlmostEqual(update.new_value, 6.8)

    def test_epsilon_greedy_can_exploit_and_decay(self) -> None:
        agent = QLearningAgent(
            epsilon=0.0, epsilon_min=0.0, epsilon_decay=0.5, seed=1
        )
        agent.q_table[5] = [-1.0, 3.0, 1.0]

        action, exploratory = agent.choose_action(5)
        agent.finish_episode()

        self.assertEqual(action, Action.STRAIGHT)
        self.assertFalse(exploratory)
        self.assertEqual(agent.epsilon, 0.0)

    def test_short_training_run_updates_table_and_scores(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            agent, tracker = train_q_learning(
                episodes=10, width=5, height=5, seed=1, report_every=5
            )

        self.assertEqual(tracker.episodes, 10)
        self.assertTrue(any(value != 0.0 for row in agent.q_table for value in row))


if __name__ == "__main__":
    unittest.main()
