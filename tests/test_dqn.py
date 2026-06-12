import contextlib
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from snake_ai.agents.dqn import (
    ACTION_COUNT,
    INPUT_COUNT,
    DQNAgent,
    Experience,
    QNetwork,
    ReplayBuffer,
    dqn_input_count,
    encode_dqn_state,
    sight_cell_count,
    train_dqn,
)
from snake_ai.game import Action, Direction, GameState


class DQNStateEncodingTests(unittest.TestCase):
    def test_sight_distance_controls_surrounding_cell_count(self) -> None:
        self.assertEqual(sight_cell_count(1), 8)
        self.assertEqual(sight_cell_count(2), 24)
        self.assertEqual(dqn_input_count(1), 16)
        self.assertEqual(dqn_input_count(2), 32)

    def test_encoder_marks_walls_and_body_in_square_sight(self) -> None:
        state = GameState(
            width=5,
            height=5,
            snake=((0, 0), (1, 0), (1, 1)),
            food=(4, 4),
            direction=Direction.RIGHT,
            score=0,
            steps=0,
            done=False,
        )

        features = encode_dqn_state(state, sight_distance=1)

        self.assertEqual(features[:8], (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0))
        self.assertEqual(len(features), 16)


class QNetworkTests(unittest.TestCase):
    def test_network_outputs_one_q_value_per_action(self) -> None:
        network = QNetwork(seed=1)

        output = network.predict(np.zeros((2, INPUT_COUNT), dtype=np.float32))

        self.assertEqual(output.shape, (2, ACTION_COUNT))

    def test_training_reduces_loss_on_repeated_example(self) -> None:
        network = QNetwork((INPUT_COUNT, 8, ACTION_COUNT), seed=1)
        states = np.ones((1, INPUT_COUNT), dtype=np.float32)
        targets = np.asarray([[1.0, -1.0, 0.5]], dtype=np.float32)

        first_loss = network.train(states, targets, 0.01)
        for _ in range(100):
            final_loss = network.train(states, targets, 0.01)

        self.assertLess(final_loss, first_loss)


class ReplayBufferTests(unittest.TestCase):
    def test_buffer_discards_oldest_experience_at_capacity(self) -> None:
        buffer = ReplayBuffer(2, seed=1)
        state = (0.0,) * INPUT_COUNT
        for reward in (1.0, 2.0, 3.0):
            buffer.append(Experience(state, Action.LEFT, reward, state, False))

        rewards = {item.reward for item in buffer.sample(2)}

        self.assertEqual(rewards, {2.0, 3.0})


class DQNAgentTests(unittest.TestCase):
    def test_agent_learns_after_replay_has_full_batch(self) -> None:
        agent = DQNAgent(
            hidden_sizes=(8,),
            batch_size=2,
            target_sync_interval=1,
            seed=1,
        )
        state = (0.0,) * INPUT_COUNT
        agent.remember(state, Action.LEFT, 1.0, state, False)
        self.assertIsNone(agent.learn())
        agent.remember(state, Action.RIGHT, -1.0, state, True)

        update = agent.learn()

        self.assertIsNotNone(update)
        self.assertTrue(update.target_synced)
        self.assertEqual(agent.training_steps, 1)

    def test_agent_input_layer_matches_sight_distance(self) -> None:
        agent = DQNAgent(sight_distance=2, seed=1)

        self.assertEqual(agent.input_count, 32)
        self.assertEqual(agent.online_network.layer_sizes[0], 32)

    def test_checkpoint_round_trip_preserves_policy_for_inference(self) -> None:
        agent = DQNAgent(hidden_sizes=(8,), sight_distance=2, seed=1)
        state = np.ones(agent.input_count, dtype=np.float32)
        expected = agent.online_network.predict(state)

        with TemporaryDirectory() as directory:
            path = agent.save_checkpoint(
                Path(directory) / "policy.npz",
                metadata={"step_penalty": -0.01},
            )
            loaded, metadata = DQNAgent.load_checkpoint(path, seed=2)

        np.testing.assert_allclose(loaded.online_network.predict(state), expected)
        self.assertEqual(loaded.epsilon, 0.0)
        self.assertEqual(loaded.sight_distance, 2)
        self.assertEqual(metadata["step_penalty"], -0.01)

    def test_short_training_run_collects_scores_and_trains(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            agent, tracker = train_dqn(
                episodes=5,
                width=5,
                height=5,
                seed=1,
                batch_size=4,
                target_sync_interval=2,
                report_every=5,
            )

        self.assertEqual(tracker.episodes, 5)
        self.assertGreater(agent.training_steps, 0)
        self.assertTrue(agent.loss_history)
        self.assertEqual(len(agent.episode_loss_history), 5)


if __name__ == "__main__":
    unittest.main()
