import contextlib
import io
import unittest

from snake_ai.agents.q_learning_two_step import (
    STATE_COUNT,
    TwoStepDangerQLearningAgent,
    encode_two_step_state,
    train_two_step_q_learning,
)
from snake_ai.game import Direction, GameState


class TwoStepStateEncodingTests(unittest.TestCase):
    def test_encoder_has_six_relative_danger_bits(self) -> None:
        state = GameState(
            width=5,
            height=5,
            snake=((1, 1), (1, 2), (1, 3)),
            food=(4, 0),
            direction=Direction.UP,
            score=0,
            steps=0,
            done=False,
        )

        encoded = encode_two_step_state(state)

        self.assertEqual(encoded.features[:6], (0, 0, 0, 1, 1, 0))
        self.assertEqual(len(encoded.features), 14)
        self.assertLess(encoded.state_id, STATE_COUNT)

    def test_distance_two_detects_body_without_marking_distance_one(self) -> None:
        state = GameState(
            width=7,
            height=7,
            snake=((3, 3), (3, 4), (3, 5), (3, 1)),
            food=(6, 6),
            direction=Direction.UP,
            score=0,
            steps=0,
            done=False,
        )

        encoded = encode_two_step_state(state)

        self.assertEqual(encoded.features[0], 0)
        self.assertEqual(encoded.features[3], 1)


class TwoStepDangerQLearningAgentTests(unittest.TestCase):
    def test_agent_uses_expanded_q_table(self) -> None:
        agent = TwoStepDangerQLearningAgent()

        self.assertEqual(STATE_COUNT, 16_384)
        self.assertEqual(agent.state_count, STATE_COUNT)
        self.assertEqual(len(agent.q_table), STATE_COUNT)

    def test_short_training_run_updates_expanded_table(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            agent, tracker = train_two_step_q_learning(
                episodes=10, width=5, height=5, seed=1, report_every=5
            )

        self.assertEqual(tracker.episodes, 10)
        self.assertTrue(any(value != 0.0 for row in agent.q_table for value in row))


if __name__ == "__main__":
    unittest.main()
