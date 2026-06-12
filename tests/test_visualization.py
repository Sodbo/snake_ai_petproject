import unittest
from tempfile import TemporaryDirectory
from pathlib import Path

from snake_ai.game import Action, Direction
from snake_ai.training.metrics import load_metrics
from snake_ai.visualization.controller import DashboardController
from snake_ai.visualization.snapshot import DashboardSnapshot


class DashboardControllerTests(unittest.TestCase):
    def test_snapshot_exposes_game_and_learning_telemetry(self) -> None:
        controller = DashboardController(8, 8, seed=1)

        snapshot = controller.snapshot

        self.assertIsInstance(snapshot, DashboardSnapshot)
        self.assertEqual(snapshot.agent_name, "Random")
        self.assertEqual(snapshot.game_state.width, 8)
        self.assertIn("learning status", snapshot.metrics)
        self.assertEqual(controller.step_penalty, -0.01)

    def test_paused_controller_only_advances_when_forced(self) -> None:
        controller = DashboardController(8, 8, seed=1)

        controller.advance()
        self.assertEqual(controller.game.state.steps, 0)

        controller.advance(force=True)
        self.assertEqual(controller.game.state.steps, 1)

    def test_manual_direction_is_converted_to_relative_action(self) -> None:
        controller = DashboardController(8, 8, seed=1, mode="manual")

        controller.request_direction(Direction.UP)
        controller.advance(force=True)

        self.assertEqual(controller.snapshot.action, Action.LEFT)
        self.assertEqual(controller.game.state.direction, Direction.UP)

    def test_speed_starts_controller_and_reset_changes_board(self) -> None:
        controller = DashboardController()

        controller.set_speed(5000)
        self.assertFalse(controller.paused)
        self.assertEqual(controller.speed, 5000)

        controller.reset(width=10, height=8)
        self.assertTrue(controller.paused)
        self.assertEqual(
            (controller.game.state.width, controller.game.state.height), (10, 8)
        )
        self.assertEqual(controller.snapshot.max_snake_length, 3)
        self.assertEqual(controller.snapshot.average_snake_length_50, 0.0)

    def test_random_agent_continues_across_episodes(self) -> None:
        controller = DashboardController(3, 3, seed=1)
        controller.set_speed(1)

        controller.advance(3)

        self.assertGreaterEqual(controller.episode, 2)
        self.assertFalse(controller.paused)
        self.assertEqual(controller.snapshot.max_snake_length, 3)
        self.assertEqual(controller.snapshot.average_snake_length_50, 3.0)

    def test_snake_length_statistics_use_last_50_completed_episodes(self) -> None:
        controller = DashboardController(3, 3, seed=1)

        for length in range(1, 52):
            controller._record_episode_length(length)
        controller._max_snake_length = 60
        snapshot = controller.snapshot

        self.assertEqual(snapshot.max_snake_length, 60)
        self.assertEqual(snapshot.average_snake_length_50, sum(range(2, 52)) / 50)
        self.assertEqual(snapshot.length_history, tuple(range(1, 52)))
        self.assertEqual(snapshot.running_max[-1], 51.0)
        self.assertEqual(snapshot.rolling_average_50[-1], sum(range(2, 52)) / 50)

    def test_length_history_keeps_full_current_run(self) -> None:
        controller = DashboardController()

        for length in range(250):
            controller._record_episode_length(length)
        snapshot = controller.snapshot

        self.assertEqual(len(snapshot.length_history), 250)
        self.assertEqual(snapshot.length_history[0], 0)
        self.assertEqual(snapshot.length_history[-1], 249)
        self.assertEqual(snapshot.running_max, tuple(float(value) for value in range(250)))

    def test_running_maximum_never_decreases(self) -> None:
        controller = DashboardController()
        for length in (3, 8, 5, 12, 4):
            controller._record_episode_length(length)

        self.assertEqual(controller.snapshot.running_max, (3.0, 8.0, 8.0, 12.0, 12.0))

    def test_dashboard_dump_keeps_all_completed_episode_lengths(self) -> None:
        controller = DashboardController(mode="q-learning")
        for length in range(205):
            controller._record_episode_length(length)

        with TemporaryDirectory() as directory:
            output = controller.dump_stats(Path(directory) / "metrics.json")
            metrics = load_metrics(output)

        self.assertEqual(metrics["algorithm"], "Q-Learning")
        self.assertEqual(len(metrics["episodes"]), 205)
        self.assertEqual(metrics["episodes"][-1]["snake_length"], 204)

    def test_q_learning_mode_updates_table_and_telemetry(self) -> None:
        controller = DashboardController(5, 5, seed=1, mode="q-learning")

        controller.advance(force=True)
        snapshot = controller.snapshot

        self.assertEqual(snapshot.agent_name, "Q-Learning")
        self.assertEqual(snapshot.learning_view, "Tabular Q-learning")
        self.assertIn("state ID", snapshot.metrics)
        self.assertIn("old Q", snapshot.metrics)
        self.assertIn("valid-space coverage", snapshot.metrics)
        self.assertEqual(snapshot.metrics["valid rows"], 256)
        self.assertGreater(snapshot.q_table_coverage, 0.0)
        self.assertEqual(len(snapshot.q_values), 3)
        self.assertIsNotNone(controller.q_agent.last_update)

    def test_q_learning_episode_records_coverage(self) -> None:
        controller = DashboardController(3, 3, seed=1, mode="q-learning")
        controller.set_speed(1)

        while not controller.snapshot.coverage_history:
            controller.advance()

        self.assertEqual(len(controller.snapshot.coverage_history), 1)
        self.assertGreater(controller.snapshot.coverage_history[0], 0.0)

        with TemporaryDirectory() as directory:
            output = controller.dump_stats(Path(directory) / "metrics.json")
            metrics = load_metrics(output)

        self.assertEqual(
            metrics["episodes"][0]["q_table_coverage"],
            controller.snapshot.coverage_history[0],
        )
        self.assertEqual(metrics["config"]["valid_state_count"], 256)

    def test_two_step_q_learning_mode_uses_expanded_table(self) -> None:
        controller = DashboardController(5, 5, seed=1, mode="q-learning-2step")

        controller.advance(force=True)
        snapshot = controller.snapshot

        self.assertEqual(snapshot.agent_name, "Q-Learning 2-Step")
        self.assertEqual(snapshot.learning_view, "14-bit two-step danger")
        self.assertEqual(snapshot.metrics["table rows"], 16_384)
        self.assertEqual(snapshot.metrics["valid rows"], 2_048)
        self.assertEqual(len(snapshot.q_values), 3)
        self.assertIsNotNone(controller.two_step_q_agent.last_update)

    def test_dqn_mode_trains_and_exports_loss_without_q_table_coverage(self) -> None:
        controller = DashboardController(5, 5, seed=1, mode="dqn")
        controller.dqn_agent.batch_size = 2
        controller.dqn_agent.learning_starts = 2
        controller.dqn_agent.train_every = 1
        controller.set_speed(1)

        while not controller.snapshot.loss_history:
            controller.advance()
        snapshot = controller.snapshot

        self.assertEqual(snapshot.agent_name, "DQN")
        self.assertEqual(snapshot.learning_view, "NumPy deep Q-network")
        self.assertIsNone(snapshot.q_table_coverage)
        self.assertFalse(snapshot.coverage_history)
        self.assertEqual(len(snapshot.q_values), 3)
        self.assertIn("replay size", snapshot.metrics)
        self.assertIn("loss", snapshot.metrics)
        self.assertIn("train every", snapshot.metrics)
        self.assertEqual(snapshot.metrics["learning status"], "active")
        self.assertIsNotNone(snapshot.learning_started_episode)

        with TemporaryDirectory() as directory:
            output = controller.dump_stats(Path(directory) / "metrics.json")
            metrics = load_metrics(output)

        self.assertEqual(metrics["algorithm"], "DQN (NumPy)")
        self.assertIn("loss", metrics["episodes"][0])
        self.assertNotIn("q_table_coverage", metrics["episodes"][0])

    def test_dqn_telemetry_distinguishes_transition_warmup_from_episodes(self) -> None:
        controller = DashboardController(
            8, 8, seed=1, mode="dqn", learning_starts=1000
        )

        controller.advance(force=True)
        snapshot = controller.snapshot

        self.assertEqual(snapshot.metrics["learning starts (steps)"], 1000)
        self.assertEqual(snapshot.metrics["learning status"], "warm-up (999 steps remaining)")
        self.assertEqual(snapshot.metrics["started at episode"], "-")
        self.assertIsNone(snapshot.learning_started_episode)

    def test_dqn_sight_distance_rebuilds_input_layer_and_is_exported(self) -> None:
        controller = DashboardController(8, 8, seed=1, mode="dqn")

        controller.set_sight_distance(2)
        snapshot = controller.snapshot

        self.assertEqual(controller.sight_distance, 2)
        self.assertEqual(controller.dqn_agent.input_count, 32)
        self.assertEqual(snapshot.metrics["sight cells"], 24)
        self.assertEqual(snapshot.metrics["network inputs"], 32)
        with TemporaryDirectory() as directory:
            output = controller.dump_stats(Path(directory) / "metrics.json")
            metrics = load_metrics(output)
        self.assertEqual(metrics["config"]["sight_distance"], 2)
        self.assertEqual(metrics["config"]["sight_cells"], 24)
        self.assertEqual(metrics["config"]["step_penalty"], -0.01)

    def test_loaded_dqn_inference_plays_without_learning(self) -> None:
        agent = DashboardController(5, 5, seed=1, mode="dqn").dqn_agent
        with TemporaryDirectory() as directory:
            checkpoint = agent.save_checkpoint(Path(directory) / "policy.npz")
            controller = DashboardController(
                5,
                5,
                seed=1,
                mode="dqn-inference",
                dqn_checkpoint=checkpoint,
            )
            controller.advance(10, force=True)

        self.assertEqual(controller.snapshot.agent_name, "DQN Inference")
        self.assertFalse(controller.snapshot.metrics["learning"])
        self.assertEqual(controller.dqn_agent.epsilon, 0.0)
        self.assertEqual(controller.dqn_agent.training_steps, 0)
        self.assertEqual(len(controller.dqn_agent.replay_buffer), 0)

    def test_cnn_dqn_mode_trains_and_exports_cnn_telemetry(self) -> None:
        controller = DashboardController(
            5,
            5,
            seed=1,
            mode="cnn-dqn",
            learning_starts=2,
            train_every=1,
            cnn_device="cpu",
        )
        assert controller.cnn_agent is not None
        controller.cnn_agent.batch_size = 2
        controller.set_speed(1)

        while not controller.snapshot.loss_history:
            controller.advance()
        snapshot = controller.snapshot

        self.assertEqual(snapshot.agent_name, "CNN DQN")
        self.assertEqual(snapshot.learning_view, "PyTorch full-board Double DQN")
        self.assertEqual(snapshot.metrics["board channels"], 7)
        self.assertEqual(snapshot.metrics["device"], "cpu")
        self.assertEqual(snapshot.metrics["learning status"], "active")
        self.assertEqual(len(snapshot.q_values), 3)
        self.assertIsNone(snapshot.q_table_coverage)
        with TemporaryDirectory() as directory:
            output = controller.dump_stats(Path(directory) / "metrics.json")
            model = controller.save_dqn_checkpoint(Path(directory) / "cnn.pt")
            metrics = load_metrics(output)
            self.assertTrue(model.exists())

        self.assertEqual(metrics["algorithm"], "CNN Double DQN (PyTorch)")
        self.assertEqual(metrics["config"]["board_channels"], 7)
        self.assertEqual(metrics["config"]["device"], "cpu")


if __name__ == "__main__":
    unittest.main()
