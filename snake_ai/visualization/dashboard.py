"""Pygame dashboard for gameplay and future learning visualization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
from typing import Callable

import pygame

from snake_ai.game import DEFAULT_STEP_PENALTY, Action, Direction, GameState
from snake_ai.visualization.controller import DashboardController, MODES, SPEEDS
from snake_ai.visualization.snapshot import DashboardSnapshot

WINDOW_SIZE = (1440, 960)
FPS = 60
BASE_STEPS_PER_SECOND = 8

BLACK = (0, 0, 0)
PANEL = (20, 20, 20)
PANEL_LIGHT = (31, 31, 31)
GRID = (35, 35, 35)
WHITE = (235, 235, 235)
MUTED = (155, 155, 155)
GREEN = (0, 175, 70)
HEAD_GREEN = (0, 235, 95)
RED = (230, 45, 45)
ACCENT = (60, 130, 235)
SELECTED = (245, 180, 45)
CHART_AVERAGE = (0, 190, 100)
CHART_MAX = (70, 140, 255)
CHART_COVERAGE = (190, 100, 255)
CHART_LOSS = (245, 150, 45)


@dataclass
class Button:
    label: str
    rect: pygame.Rect
    callback: Callable[[], None]
    active: Callable[[], bool] = lambda: False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        color = ACCENT if self.active() else PANEL_LIGHT
        pygame.draw.rect(surface, color, self.rect, border_radius=4)
        pygame.draw.rect(surface, MUTED, self.rect, 1, border_radius=4)
        text = font.render(self.label, True, WHITE)
        surface.blit(text, text.get_rect(center=self.rect.center))


class Dashboard:
    """Interactive dashboard shell used by current and future agents."""

    def __init__(self, controller: DashboardController) -> None:
        pygame.init()
        pygame.display.set_caption("Snake AI Learning Dashboard")
        self.surface = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small_font = pygame.font.SysFont("consolas", 15)
        self.tiny_font = pygame.font.SysFont("consolas", 13)
        self.title_font = pygame.font.SysFont("consolas", 22, bold=True)
        self.controller = controller
        self.running = True
        self.step_accumulator = 0.0
        self.width_text = str(controller.game.width)
        self.height_text = str(controller.game.height)
        self.sight_text = str(controller.sight_distance)
        self.active_input: str | None = None
        self.buttons: list[Button] = []

    def run(self) -> None:
        while self.running:
            elapsed = self.clock.tick(FPS) / 1000.0
            self._handle_events()
            self._advance_simulation(elapsed)
            self._draw()
            pygame.display.flip()
        pygame.quit()

    def _advance_simulation(self, elapsed: float) -> None:
        if self.controller.paused:
            self.step_accumulator = 0.0
            return
        self.step_accumulator += elapsed * BASE_STEPS_PER_SECOND * self.controller.speed
        steps = int(self.step_accumulator)
        if steps:
            self.step_accumulator -= steps
            self.controller.advance(steps)

    def _handle_events(self) -> None:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.running = False
            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                self._handle_click(event.pos)
            elif event.type == pygame.KEYDOWN:
                self._handle_key(event)

    def _handle_click(self, position: tuple[int, int]) -> None:
        self.active_input = None
        for name, rect in self._input_rects().items():
            if rect.collidepoint(position):
                self.active_input = name
                return
        for button in self.buttons:
            if button.rect.collidepoint(position):
                button.callback()
                return

    def _handle_key(self, event: pygame.event.Event) -> None:
        if self.active_input is not None:
            self._edit_input(event)
            return
        direction = {
            pygame.K_UP: Direction.UP,
            pygame.K_RIGHT: Direction.RIGHT,
            pygame.K_DOWN: Direction.DOWN,
            pygame.K_LEFT: Direction.LEFT,
        }.get(event.key)
        if direction is not None:
            self.controller.request_direction(direction)
        elif event.key == pygame.K_SPACE:
            self.controller.toggle_pause()
        elif event.key == pygame.K_n:
            self.controller.advance(force=True)
        elif event.key == pygame.K_r:
            self.controller.reset()

    def _edit_input(self, event: pygame.event.Event) -> None:
        values = {
            "width": self.width_text,
            "height": self.height_text,
            "sight": self.sight_text,
        }
        value = values[self.active_input]
        if event.key == pygame.K_BACKSPACE:
            value = value[:-1]
        elif event.key == pygame.K_RETURN:
            self._apply_board_size()
            self.active_input = None
            return
        elif event.unicode.isdigit() and len(value) < 3:
            value += event.unicode
        if self.active_input == "width":
            self.width_text = value
        elif self.active_input == "height":
            self.height_text = value
        else:
            self.sight_text = value

    def _apply_board_size(self) -> None:
        try:
            width = int(self.width_text)
            height = int(self.height_text)
            sight_distance = int(self.sight_text)
            if sight_distance != self.controller.sight_distance:
                self.controller.set_sight_distance(sight_distance)
            self.controller.reset(width=width, height=height)
        except ValueError:
            self.width_text = str(self.controller.game.width)
            self.height_text = str(self.controller.game.height)
            self.sight_text = str(self.controller.sight_distance)

    def _draw(self) -> None:
        self.surface.fill(BLACK)
        width, height = self.surface.get_size()
        controls = pygame.Rect(24, height - 132, width - 48, 108)
        chart_height = max(170, min(240, height // 4))
        chart = pygame.Rect(24, controls.y - chart_height - 16, width - 48, chart_height)
        upper_height = chart.y - 40
        board_width = min(upper_height, max(360, int(width * 0.46)))
        board_area = pygame.Rect(24, 24, board_width, upper_height)
        info_panel = pygame.Rect(
            board_area.right + 24,
            24,
            width - board_area.right - 48,
            upper_height,
        )
        snapshot = self.controller.snapshot
        self._draw_board(snapshot.game_state, board_area)
        self._draw_info_panel(snapshot, info_panel)
        self._draw_length_chart(snapshot, chart)
        self._draw_controls(controls)

    def _draw_board(self, state: GameState, area: pygame.Rect) -> None:
        pygame.draw.rect(self.surface, BLACK, area)
        cell = max(1, min(area.width // state.width, area.height // state.height))
        board = pygame.Rect(
            area.x + (area.width - cell * state.width) // 2,
            area.y + (area.height - cell * state.height) // 2,
            cell * state.width,
            cell * state.height,
        )
        pygame.draw.rect(self.surface, GRID, board, 1)
        if cell >= 8:
            for x in range(state.width + 1):
                pygame.draw.line(
                    self.surface,
                    GRID,
                    (board.x + x * cell, board.y),
                    (board.x + x * cell, board.bottom),
                )
            for y in range(state.height + 1):
                pygame.draw.line(
                    self.surface,
                    GRID,
                    (board.x, board.y + y * cell),
                    (board.right, board.y + y * cell),
                )
        if state.food is not None:
            self._draw_cell(board, cell, state.food, RED)
        for position in state.snake[1:]:
            self._draw_cell(board, cell, position, GREEN)
        self._draw_cell(board, cell, state.head, HEAD_GREEN)

    def _draw_cell(
        self,
        board: pygame.Rect,
        cell: int,
        position: tuple[int, int],
        color: tuple[int, int, int],
    ) -> None:
        inset = 1 if cell >= 5 else 0
        x, y = position
        rect = pygame.Rect(
            board.x + x * cell + inset,
            board.y + y * cell + inset,
            max(1, cell - inset * 2),
            max(1, cell - inset * 2),
        )
        pygame.draw.rect(self.surface, color, rect)

    def _draw_info_panel(self, snapshot: DashboardSnapshot, area: pygame.Rect) -> None:
        pygame.draw.rect(self.surface, PANEL, area, border_radius=5)
        padding = 18
        left_width = int(area.width * 0.42)
        divider_x = area.x + left_width
        pygame.draw.line(
            self.surface,
            GRID,
            (divider_x, area.y + padding),
            (divider_x, area.bottom - padding),
        )

        x, y = area.x + padding, area.y + padding
        self._text("GAME STATE", x, y, self.title_font)
        y += 34
        state = snapshot.game_state
        action = snapshot.action.name.lower() if snapshot.action is not None else "-"
        status = "Paused" if snapshot.paused else "Running"
        rows = (
            ("Agent", snapshot.agent_name),
            ("Episode", snapshot.episode),
            ("Step", state.steps),
            ("Score", state.score),
            ("Reward", f"{snapshot.reward:.2f}"),
            ("Action", action),
            ("Direction", state.direction.name.lower()),
            ("Snake length", len(state.snake)),
            ("Max length", snapshot.max_snake_length),
            ("Avg length (50)", f"{snapshot.average_snake_length_50:.2f}"),
            (
                "Valid-space coverage",
                (
                    f"{snapshot.q_table_coverage:.3f}%"
                    if snapshot.q_table_coverage is not None
                    else "-"
                ),
            ),
            ("Food", state.food),
            ("Status", status),
            ("Speed", f"{snapshot.speed}x"),
            ("Step penalty", f"{self.controller.step_penalty:.2f}"),
        )
        for label, value in rows:
            self._text(f"{label}: {value}", x, y, self.small_font)
            y += 19

        q_x = divider_x + padding
        q_y = area.y + padding
        self._text("CURRENT ACTION VALUES", q_x, q_y, self.title_font)
        q_y += 34
        self._text(snapshot.learning_view, q_x, q_y, self.font, ACCENT)
        q_y += 28
        if snapshot.q_values is not None:
            q_y = self._draw_q_values(
                snapshot, q_x, q_y, area.right - q_x - padding
            )
        else:
            self._text(
                "Available for learning agents.",
                q_x,
                q_y,
                self.small_font,
                MUTED,
            )

        metrics_y = max(y + 8, q_y + 12)
        pygame.draw.line(
            self.surface,
            GRID,
            (area.x + padding, metrics_y),
            (area.right - padding, metrics_y),
        )
        metrics_y += 12
        self._text("LEARNING UPDATE", area.x + padding, metrics_y, self.title_font)
        metrics_y += 30
        metric_items = list(snapshot.metrics.items())
        column_width = (area.width - padding * 3) // 2
        for index, (label, value) in enumerate(metric_items):
            column = index % 2
            row = index // 2
            metric_x = area.x + padding + column * (column_width + padding)
            metric_y = metrics_y + row * 18
            self._text(f"{label}: {value}", metric_x, metric_y, self.small_font, MUTED)

    def _draw_length_chart(
        self, snapshot: DashboardSnapshot, area: pygame.Rect
    ) -> None:
        pygame.draw.rect(self.surface, PANEL, area, border_radius=5)
        self._text(
            "SNAKE LENGTH - CURRENT RUN",
            area.x + 18,
            area.y + 14,
            self.title_font,
        )
        legend_x = area.right - 540
        self._text("all-time maximum", legend_x, area.y + 18, self.small_font, CHART_MAX)
        self._text(
            "average (rolling 50)",
            legend_x + 155,
            area.y + 18,
            self.small_font,
            CHART_AVERAGE,
        )
        self._text(
            "valid-space coverage",
            legend_x + 355,
            area.y + 18,
            self.small_font,
            CHART_COVERAGE,
        )
        if snapshot.loss_history:
            self._text(
                "DQN loss",
                legend_x + 355,
                area.y + 18,
                self.small_font,
                CHART_LOSS,
            )
        plot = pygame.Rect(area.x + 54, area.y + 48, area.width - 108, area.height - 78)
        pygame.draw.rect(self.surface, GRID, plot, 1)
        if not snapshot.length_history:
            self._text(
                "Chart begins after the first completed episode.",
                plot.x + 18,
                plot.centery - 8,
                self.small_font,
                MUTED,
            )
            return

        series_max = snapshot.running_max
        series_average = snapshot.rolling_average_50
        upper = max(3.0, max(series_max))
        lower = min(3.0, min(series_average))
        span = max(1.0, upper - lower)
        for tick in range(5):
            value = lower + span * tick / 4
            tick_y = plot.bottom - int(plot.height * tick / 4)
            pygame.draw.line(self.surface, GRID, (plot.x, tick_y), (plot.right, tick_y))
            self._text(f"{value:.1f}", area.x + 8, tick_y - 8, self.tiny_font, MUTED)

        self._text("1", plot.x, plot.bottom + 8, self.tiny_font, MUTED)
        self._text(
            str(snapshot.episode - 1),
            plot.right - 34,
            plot.bottom + 8,
            self.tiny_font,
            MUTED,
        )
        self._draw_chart_series(plot, series_max, lower, span, CHART_MAX)
        self._draw_chart_series(plot, series_average, lower, span, CHART_AVERAGE)
        if snapshot.coverage_history:
            coverage_upper = min(100.0, max(1.0, max(snapshot.coverage_history) * 1.1))
            for tick in range(5):
                value = coverage_upper * tick / 4
                tick_y = plot.bottom - int(plot.height * tick / 4)
                self._text(
                    f"{value:.1f}%",
                    plot.right + 6,
                    tick_y - 8,
                    self.tiny_font,
                    CHART_COVERAGE,
                )
            self._draw_chart_series(
                plot,
                snapshot.coverage_history,
                0.0,
                coverage_upper,
                CHART_COVERAGE,
            )
        elif snapshot.loss_history:
            loss_upper = max(1.0, max(snapshot.loss_history) * 1.1)
            for tick in range(5):
                value = loss_upper * tick / 4
                tick_y = plot.bottom - int(plot.height * tick / 4)
                self._text(
                    f"{value:.2f}",
                    plot.right + 6,
                    tick_y - 8,
                    self.tiny_font,
                    CHART_LOSS,
                )
            self._draw_chart_series(
                plot,
                snapshot.loss_history,
                0.0,
                loss_upper,
                CHART_LOSS,
            )

    def _draw_chart_series(
        self,
        plot: pygame.Rect,
        values: tuple[float, ...],
        lower: float,
        span: float,
        color: tuple[int, int, int],
    ) -> None:
        if len(values) < 2:
            return
        stride = max(1, math.ceil(len(values) / plot.width))
        indices = list(range(0, len(values), stride))
        if indices[-1] != len(values) - 1:
            indices.append(len(values) - 1)
        points = [
            (
                plot.x + int(index / (len(values) - 1) * plot.width),
                plot.bottom - int((values[index] - lower) / span * plot.height),
            )
            for index in indices
        ]
        pygame.draw.lines(self.surface, color, False, points, 2)

    def _draw_q_values(
        self, snapshot: DashboardSnapshot, x: int, y: int, width: int
    ) -> int:
        assert snapshot.q_values is not None
        values = snapshot.q_values
        largest = max(max(abs(value) for value in values), 1.0)
        best = max(values)
        label_width = 92
        value_width = 68
        bar_x = x + label_width
        bar_width = max(80, width - label_width - value_width)
        center_x = bar_x + bar_width // 2
        half_width = bar_width // 2 - 4

        for action in Action:
            value = values[action]
            row = pygame.Rect(bar_x, y, bar_width, 18)
            pygame.draw.rect(self.surface, PANEL_LIGHT, row, border_radius=3)
            pygame.draw.line(
                self.surface, MUTED, (center_x, row.y), (center_x, row.bottom)
            )
            extent = int(abs(value) / largest * half_width)
            if extent:
                fill = pygame.Rect(
                    center_x if value >= 0 else center_x - extent,
                    row.y + 2,
                    extent,
                    row.height - 4,
                )
                pygame.draw.rect(
                    self.surface, GREEN if value >= 0 else RED, fill, border_radius=2
                )
            if value == best:
                pygame.draw.rect(self.surface, ACCENT, row, 2, border_radius=3)
            if snapshot.action == action:
                pygame.draw.circle(self.surface, SELECTED, (x + 5, y + 9), 4)
            self._text(action.name.lower(), x + 14, y + 1, self.small_font)
            self._text(f"{value:+.3f}", row.right + 8, y + 1, self.small_font)
            y += 24
        self._text(
            "blue = best | amber = last selected",
            x,
            y,
            self.small_font,
            MUTED,
        )
        return y + 18

    def _draw_controls(self, area: pygame.Rect) -> None:
        pygame.draw.rect(self.surface, PANEL, area, border_radius=5)
        self.buttons = []
        x, y = area.x + 14, area.y + 12

        def add(
            label: str,
            callback: Callable[[], None],
            *,
            width: int = 78,
            active: Callable[[], bool] = lambda: False,
        ) -> None:
            nonlocal x
            button = Button(label, pygame.Rect(x, y, width, 34), callback, active)
            self.buttons.append(button)
            button.draw(self.surface, self.small_font)
            x += width + 8

        add(
            "Manual",
            lambda: self.controller.set_mode("manual"),
            active=lambda: self.controller.mode == "manual",
        )
        add(
            "Random",
            lambda: self.controller.set_mode("random"),
            active=lambda: self.controller.mode == "random",
        )
        add(
            "Q-Learn",
            lambda: self.controller.set_mode("q-learning"),
            width=82,
            active=lambda: self.controller.mode == "q-learning",
        )
        add(
            "Q-2Step",
            lambda: self.controller.set_mode("q-learning-2step"),
            width=88,
            active=lambda: self.controller.mode == "q-learning-2step",
        )
        add(
            "DQN",
            lambda: self.controller.set_mode("dqn"),
            width=64,
            active=lambda: self.controller.mode == "dqn",
        )
        if self.controller.mode == "dqn-inference":
            add(
                "DQN View",
                lambda: None,
                width=86,
                active=lambda: True,
            )
        add(
            "Pause",
            self.controller.toggle_pause,
            active=lambda: self.controller.paused,
        )
        add("Step", lambda: self.controller.advance(force=True), width=64)
        for speed in SPEEDS:
            add(
                f"{speed}x",
                lambda selected=speed: self.controller.set_speed(selected),
                width=64 if speed >= 1000 else 58,
                active=lambda selected=speed: (
                    not self.controller.paused and self.controller.speed == selected
                ),
            )
        add("Reset", self.controller.reset, width=70)
        add("Dump Stats", self._dump_stats, width=96)

        self._draw_board_size_controls(area)

    def _dump_stats(self) -> None:
        output = self.controller.dump_stats()
        print(f"Metrics saved to {output}")

    def _save_model(self) -> None:
        output = self.controller.save_dqn_checkpoint()
        print(f"Checkpoint saved to {output}")

    def _draw_board_size_controls(self, area: pygame.Rect) -> None:
        y = area.y + 60
        self._text("Board:", area.x + 16, y + 7, self.small_font)
        for name, rect in self._input_rects().items():
            active = name == self.active_input
            pygame.draw.rect(self.surface, PANEL_LIGHT, rect, border_radius=3)
            pygame.draw.rect(self.surface, ACCENT if active else MUTED, rect, 1)
            value = {
                "width": self.width_text,
                "height": self.height_text,
                "sight": self.sight_text,
            }[name]
            self._text(value, rect.x + 8, rect.y + 6, self.small_font)
        self._text("x", area.x + 202, y + 7, self.small_font)
        self._text("DQN sight:", area.x + 320, y + 7, self.small_font)
        apply_rect = pygame.Rect(area.x + 500, y, 70, 30)
        button = Button("Apply", apply_rect, self._apply_board_size)
        self.buttons.append(button)
        button.draw(self.surface, self.small_font)
        help_x = area.x + 590
        if self.controller.mode in ("dqn", "dqn-inference"):
            save_rect = pygame.Rect(area.x + 580, y, 96, 30)
            save_button = Button("Save Model", save_rect, self._save_model)
            self.buttons.append(save_button)
            save_button.draw(self.surface, self.small_font)
            help_x = area.x + 690
        self._text(
            "Arrow keys: steer manual | Space: pause | N: step | R: reset",
            help_x,
            y + 7,
            self.small_font,
            MUTED,
        )

    def _input_rects(self) -> dict[str, pygame.Rect]:
        height = self.surface.get_height()
        y = height - 72
        return {
            "width": pygame.Rect(100, y, 74, 30),
            "height": pygame.Rect(230, y, 74, 30),
            "sight": pygame.Rect(435, y, 60, 30),
        }

    def _text(
        self,
        value: object,
        x: int,
        y: int,
        font: pygame.font.Font,
        color: tuple[int, int, int] = WHITE,
    ) -> None:
        self.surface.blit(font.render(str(value), True, color), (x, y))


def main() -> None:
    parser = argparse.ArgumentParser(description="Open the Snake AI dashboard.")
    parser.add_argument("--width", type=int, default=20)
    parser.add_argument("--height", type=int, default=20)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--mode", choices=MODES, default="random")
    parser.add_argument("--sight-distance", type=int, default=1)
    parser.add_argument("--step-penalty", type=float, default=DEFAULT_STEP_PENALTY)
    parser.add_argument("--checkpoint", help="Load a DQN checkpoint for inference.")
    args = parser.parse_args()
    Dashboard(
        DashboardController(
            args.width,
            args.height,
            seed=args.seed,
            mode=args.mode,
            sight_distance=args.sight_distance,
            step_penalty=args.step_penalty,
            dqn_checkpoint=args.checkpoint,
        )
    ).run()


if __name__ == "__main__":
    main()
