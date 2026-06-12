"""Episode results and basic score tracking."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class EpisodeResult:
    """Summary of one completed game."""

    score: int
    steps: int
    total_reward: float


class ScoreTracker:
    """Collect and summarize results across episodes."""

    def __init__(self) -> None:
        self._results: list[EpisodeResult] = []

    @property
    def results(self) -> tuple[EpisodeResult, ...]:
        return tuple(self._results)

    @property
    def episodes(self) -> int:
        return len(self._results)

    @property
    def best_score(self) -> int:
        return max((result.score for result in self._results), default=0)

    @property
    def average_score(self) -> float:
        if not self._results:
            return 0.0
        return sum(result.score for result in self._results) / self.episodes

    def add(self, result: EpisodeResult) -> None:
        self._results.append(result)

    def summary(self) -> str:
        return (
            f"Episodes: {self.episodes} | "
            f"Average score: {self.average_score:.2f} | "
            f"Best score: {self.best_score}"
        )
