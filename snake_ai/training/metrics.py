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
    config: Mapping[str, object] | None = None,
) -> MetricsData:
    """Build serializable metrics from completed training episodes."""
    episodes = [
        {
            "episode": index,
            "score": result.score,
            "snake_length": result.score + 3,
            "steps": result.steps,
            "total_reward": result.total_reward,
        }
        for index, result in enumerate(results, start=1)
    ]
    return {
        "algorithm": algorithm,
        "config": dict(config or {}),
        "episodes": episodes,
    }


def build_length_metrics(
    algorithm: str,
    lengths: Sequence[int],
    *,
    config: Mapping[str, object] | None = None,
) -> MetricsData:
    """Build serializable metrics when only final snake lengths are available."""
    return {
        "algorithm": algorithm,
        "config": dict(config or {}),
        "episodes": [
            {"episode": index, "snake_length": length}
            for index, length in enumerate(lengths, start=1)
        ],
    }


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

    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
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

    axes[0].set_title("All-Time Maximum Snake Length")
    axes[1].set_title(f"Rolling Average Snake Length (window={window})")
    for axis in axes:
        axis.set_xlabel("Episode")
        axis.set_ylabel("Snake length")
        axis.grid(True, alpha=0.3)
        axis.legend()
    figure.tight_layout()

    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path
