import contextlib
import importlib.util
import io
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np

from snake_ai.game import Direction, GameState

TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None

if TORCH_AVAILABLE:
    import torch

    from snake_ai.agents.cnn_dqn import (
        BOARD_CHANNELS,
        CNNDQNAgent,
        CNNQNetwork,
        encode_board,
        train_cnn_dqn,
    )


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class CNNStateTests(unittest.TestCase):
    def test_encoder_marks_head_body_food_and_direction(self) -> None:
        state = GameState(
            width=5,
            height=4,
            snake=((2, 1), (1, 1), (0, 1)),
            food=(4, 3),
            direction=Direction.RIGHT,
            score=0,
            steps=0,
            done=False,
        )

        board = encode_board(state)

        self.assertEqual(board.shape, (BOARD_CHANNELS, 4, 5))
        self.assertEqual(board[0, 1, 2], 1.0)
        self.assertEqual(board[1].sum(), 2.0)
        self.assertEqual(board[2, 3, 4], 1.0)
        self.assertTrue(np.all(board[4] == 1.0))


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed")
class CNNAgentTests(unittest.TestCase):
    def test_network_supports_different_board_sizes(self) -> None:
        network = CNNQNetwork()

        small = network(torch.zeros(2, BOARD_CHANNELS, 8, 8))
        large = network(torch.zeros(2, BOARD_CHANNELS, 20, 30))

        self.assertEqual(tuple(small.shape), (2, 3))
        self.assertEqual(tuple(large.shape), (2, 3))

    def test_short_training_run_updates_network(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            agent, tracker = train_cnn_dqn(
                episodes=5,
                width=5,
                height=5,
                seed=1,
                batch_size=4,
                learning_starts=4,
                train_every=1,
                target_sync_interval=2,
                device="cpu",
                report_every=5,
            )

        self.assertEqual(tracker.episodes, 5)
        self.assertGreater(agent.training_steps, 0)
        self.assertTrue(agent.loss_history)

    def test_checkpoint_is_written(self) -> None:
        agent = CNNDQNAgent(
            batch_size=2, learning_starts=2, replay_capacity=10, device="cpu"
        )

        with TemporaryDirectory() as directory:
            path = agent.save_checkpoint(Path(directory) / "cnn.pt")

            self.assertTrue(path.exists())
            self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
