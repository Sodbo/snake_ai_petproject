import unittest

from snake_ai.game import Action, Direction
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

        controller._recent_snake_lengths.extend(range(1, 52))
        controller._max_snake_length = 60
        snapshot = controller.snapshot

        self.assertEqual(snapshot.max_snake_length, 60)
        self.assertEqual(snapshot.average_snake_length_50, sum(range(2, 52)) / 50)

    def test_q_learning_mode_updates_table_and_telemetry(self) -> None:
        controller = DashboardController(5, 5, seed=1, mode="q-learning")

        controller.advance(force=True)
        snapshot = controller.snapshot

        self.assertEqual(snapshot.agent_name, "Q-Learning")
        self.assertEqual(snapshot.learning_view, "Tabular Q-learning")
        self.assertIn("state ID", snapshot.metrics)
        self.assertIn("old Q", snapshot.metrics)
        self.assertEqual(len(snapshot.q_values), 3)
        self.assertIsNotNone(controller.q_agent.last_update)

    def test_two_step_q_learning_mode_uses_expanded_table(self) -> None:
        controller = DashboardController(5, 5, seed=1, mode="q-learning-2step")

        controller.advance(force=True)
        snapshot = controller.snapshot

        self.assertEqual(snapshot.agent_name, "Q-Learning 2-Step")
        self.assertEqual(snapshot.learning_view, "14-bit two-step danger")
        self.assertEqual(snapshot.metrics["table rows"], 16_384)
        self.assertEqual(len(snapshot.q_values), 3)
        self.assertIsNotNone(controller.two_step_q_agent.last_update)


if __name__ == "__main__":
    unittest.main()
