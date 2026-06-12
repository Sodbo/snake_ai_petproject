"""Data passed from an agent/controller to the visualization dashboard."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping

from snake_ai.game import Action, GameState

MetricValue = str | int | float | bool


@dataclass(frozen=True)
class DashboardSnapshot:
    """Immutable state needed to draw one dashboard frame."""

    game_state: GameState
    agent_name: str
    learning_view: str
    episode: int
    reward: float = 0.0
    action: Action | None = None
    paused: bool = True
    speed: int = 1
    max_snake_length: int = 3
    average_snake_length_50: float = 0.0
    length_history: tuple[int, ...] = ()
    running_max: tuple[float, ...] = ()
    rolling_average_50: tuple[float, ...] = ()
    q_table_coverage: float | None = None
    coverage_history: tuple[float, ...] = ()
    loss_history: tuple[float, ...] = ()
    q_values: tuple[float, float, float] | None = None
    metrics: Mapping[str, MetricValue] = field(default_factory=dict)
