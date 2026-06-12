from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from snake_ai.agents.results import EpisodeResult
from snake_ai.training.metrics import (
    build_length_metrics,
    build_metrics,
    load_metrics,
    plot_length_comparison,
    rolling_series,
    running_maximum,
    save_metrics,
)


class TrainingMetricsTests(unittest.TestCase):
    def test_build_save_and_load_metrics(self) -> None:
        data = build_metrics(
            "Q-Learning",
            (
                EpisodeResult(score=0, steps=4, total_reward=-10.0),
                EpisodeResult(score=2, steps=20, total_reward=10.0),
            ),
            config={"episodes": 2},
        )

        with TemporaryDirectory() as directory:
            path = save_metrics(data, Path(directory) / "metrics.json")
            loaded = load_metrics(path)

        self.assertEqual(loaded["algorithm"], "Q-Learning")
        self.assertEqual(loaded["episodes"][0]["snake_length"], 3)
        self.assertEqual(loaded["episodes"][1]["snake_length"], 5)

    def test_rolling_series(self) -> None:
        maximum, average = rolling_series((3, 5, 4, 8), window=3)

        self.assertEqual(maximum, (3.0, 5.0, 5.0, 8.0))
        self.assertEqual(average[-1], (5 + 4 + 8) / 3)

    def test_metrics_include_q_table_coverage_per_episode(self) -> None:
        data = build_length_metrics(
            "Q-Learning",
            (3, 4),
            q_table_coverage=(0.5, 1.25),
        )

        self.assertEqual(data["episodes"][0]["q_table_coverage"], 0.5)
        self.assertEqual(data["episodes"][1]["q_table_coverage"], 1.25)

    def test_metrics_reject_mismatched_q_table_coverage(self) -> None:
        with self.assertRaises(ValueError):
            build_length_metrics("Q-Learning", (3, 4), q_table_coverage=(0.5,))

    def test_running_maximum_never_decreases(self) -> None:
        self.assertEqual(running_maximum((3, 7, 4, 9, 5)), (3.0, 7.0, 7.0, 9.0, 9.0))

    def test_plot_accepts_runs_with_different_episode_counts(self) -> None:
        first = {
            "algorithm": "Q-Learning",
            "config": {},
            "episodes": [
                {"episode": index, "snake_length": index + 3}
                for index in range(1, 6)
            ],
        }
        second = {
            "algorithm": "Q-Learning 2-Step",
            "config": {},
            "episodes": [
                {"episode": index, "snake_length": index + 2}
                for index in range(1, 9)
            ],
        }
        for episode in first["episodes"]:
            episode["q_table_coverage"] = episode["episode"] / 10
        for episode in second["episodes"]:
            episode["q_table_coverage"] = episode["episode"] / 20

        with TemporaryDirectory() as directory:
            output = plot_length_comparison(
                (first, second), Path(directory) / "comparison.png", window=3
            )

            self.assertTrue(output.exists())
            self.assertGreater(output.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()
