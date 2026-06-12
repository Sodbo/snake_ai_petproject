"""Save training metrics and compare runs with different episode counts."""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Iterable, Mapping, Sequence

from snake_ai.agents.results import EpisodeResult

MetricsData = dict[str, object]


def rolling_series(
    values: Sequence[int | float], *, window: int = 50
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Return rolling maximum and average series."""
    if window < 1:
        raise ValueError("window must be at least 1")
    maximums: list[float] = []
    averages: list[float] = []
    for index in range(len(values)):
        recent = values[max(0, index - window + 1) : index + 1]
        maximums.append(float(max(recent)))
        averages.append(float(sum(recent) / len(recent)))
    return tuple(maximums), tuple(averages)


def running_maximum(values: Sequence[int | float]) -> tuple[float, ...]:
    """Return the all-time maximum observed up to each episode."""
    maximum = float("-inf")
    result: list[float] = []
    for value in values:
        maximum = max(maximum, float(value))
        result.append(maximum)
    return tuple(result)


def build_metrics(
    algorithm: str,
    results: Iterable[EpisodeResult],
    *,
    q_table_coverage: Sequence[float] | None = None,
    loss: Sequence[float] | None = None,
    config: Mapping[str, object] | None = None,
) -> MetricsData:
    """Build serializable metrics from completed training episodes."""
    episodes: list[dict[str, int | float]] = [
        {
            "episode": index,
            "score": result.score,
            "snake_length": result.score + 3,
            "steps": result.steps,
            "total_reward": result.total_reward,
        }
        for index, result in enumerate(results, start=1)
    ]
    _add_q_table_coverage(episodes, q_table_coverage)
    _add_optional_series(episodes, "loss", loss)
    return {
        "algorithm": algorithm,
        "config": dict(config or {}),
        "episodes": episodes,
    }


def build_length_metrics(
    algorithm: str,
    lengths: Sequence[int],
    *,
    q_table_coverage: Sequence[float] | None = None,
    loss: Sequence[float] | None = None,
    config: Mapping[str, object] | None = None,
) -> MetricsData:
    """Build serializable metrics when only final snake lengths are available."""
    episodes: list[dict[str, int | float]] = [
        {"episode": index, "snake_length": length}
        for index, length in enumerate(lengths, start=1)
    ]
    _add_q_table_coverage(episodes, q_table_coverage)
    _add_optional_series(episodes, "loss", loss)
    return {
        "algorithm": algorithm,
        "config": dict(config or {}),
        "episodes": episodes,
    }


def _add_q_table_coverage(
    episodes: Sequence[dict[str, int | float]],
    coverage: Sequence[float] | None,
) -> None:
    if coverage is None:
        return
    if len(coverage) != len(episodes):
        raise ValueError("Q-table coverage must contain one value per episode")
    for episode, value in zip(episodes, coverage):
        episode["q_table_coverage"] = float(value)


def _add_optional_series(
    episodes: Sequence[dict[str, int | float]],
    name: str,
    values: Sequence[float] | None,
) -> None:
    if values is None:
        return
    if len(values) != len(episodes):
        raise ValueError(f"{name} must contain one value per episode")
    for episode, value in zip(episodes, values):
        episode[name] = float(value)


def save_metrics(data: MetricsData, output: str | Path) -> Path:
    """Write metrics as readable JSON."""
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return path


def load_metrics(path: str | Path) -> MetricsData:
    """Load a metrics JSON file."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def plot_length_comparison(
    datasets: Sequence[MetricsData],
    output: str | Path,
    *,
    window: int = 500,
) -> Path:
    """Plot all-time maximum and rolling average length for any number of runs."""
    if len(datasets) < 2:
        raise ValueError("at least two metrics datasets are required")

    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "snake_ai_matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    figure, axes = plt.subplots(4, 1, figsize=(12, 14), sharex=False)
    coverage_plotted = False
    loss_plotted = False
    maximum_coverage = 0.0
    for data in datasets:
        algorithm = str(data["algorithm"])
        episodes = data["episodes"]
        if not isinstance(episodes, list) or not episodes:
            raise ValueError(f"{algorithm} has no episode metrics")
        x_values = [int(item["episode"]) for item in episodes]
        lengths = [int(item["snake_length"]) for item in episodes]
        _, rolling_average = rolling_series(lengths, window=window)
        axes[0].plot(x_values, running_maximum(lengths), label=algorithm)
        axes[1].plot(x_values, rolling_average, label=algorithm)
        coverage = [
            float(item["q_table_coverage"])
            for item in episodes
            if "q_table_coverage" in item
        ]
        if len(coverage) == len(episodes):
            axes[2].plot(x_values, coverage, label=algorithm)
            coverage_plotted = True
            maximum_coverage = max(maximum_coverage, max(coverage))
        loss = [float(item["loss"]) for item in episodes if "loss" in item]
        if len(loss) == len(episodes):
            axes[3].plot(x_values, loss, label=algorithm)
            loss_plotted = True

    axes[0].set_title("All-Time Maximum Snake Length")
    axes[1].set_title(f"Rolling Average Snake Length (window={window})")
    axes[2].set_title("Cumulative Valid-Space Q-Table Coverage")
    axes[3].set_title("Episode Average Training Loss")
    axes[2].set_ylim(0, min(100, max(1, maximum_coverage * 1.1)))
    for axis in axes:
        axis.set_xlabel("Episode")
        axis.grid(True, alpha=0.3)
    axes[0].set_ylabel("Snake length")
    axes[1].set_ylabel("Snake length")
    axes[2].set_ylabel("Coverage (%)")
    axes[3].set_ylabel("MSE loss")
    axes[0].legend()
    axes[1].legend()
    if coverage_plotted:
        axes[2].legend()
    else:
        axes[2].text(
            0.5,
            0.5,
            "No Q-table coverage data in these metrics files",
            ha="center",
            va="center",
            transform=axes[2].transAxes,
        )
    if loss_plotted:
        axes[3].legend()
    else:
        axes[3].text(
            0.5,
            0.5,
            "No training loss data in these metrics files",
            ha="center",
            va="center",
            transform=axes[3].transAxes,
        )
    figure.tight_layout()

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
