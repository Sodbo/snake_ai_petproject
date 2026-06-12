"""Pygame dashboard for gameplay and future learning visualization."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable

import pygame

from snake_ai.game import Direction, GameState
from snake_ai.visualization.controller import DashboardController, MODES, SPEEDS
from snake_ai.visualization.snapshot import DashboardSnapshot

WINDOW_SIZE = (1280, 820)
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
        self.title_font = pygame.font.SysFont("consolas", 22, bold=True)
        self.controller = controller
        self.running = True
        self.step_accumulator = 0.0
        self.width_text = str(controller.game.width)
        self.height_text = str(controller.game.height)
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
        value = self.width_text if self.active_input == "width" else self.height_text
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
        else:
            self.height_text = value

    def _apply_board_size(self) -> None:
        try:
            width = int(self.width_text)
            height = int(self.height_text)
            self.controller.reset(width=width, height=height)
        except ValueError:
            self.width_text = str(self.controller.game.width)
            self.height_text = str(self.controller.game.height)

    def _draw(self) -> None:
        self.surface.fill(BLACK)
        width, height = self.surface.get_size()
        board_area = pygame.Rect(24, 24, min(width - 420, height - 180), height - 180)
        side_panel = pygame.Rect(board_area.right + 24, 24, width - board_area.right - 48, height - 180)
        controls = pygame.Rect(24, height - 132, width - 48, 108)
        self._draw_board(self.controller.snapshot.game_state, board_area)
        self._draw_side_panel(self.controller.snapshot, side_panel)
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

    def _draw_side_panel(self, snapshot: DashboardSnapshot, area: pygame.Rect) -> None:
        pygame.draw.rect(self.surface, PANEL, area, border_radius=5)
        x, y = area.x + 18, area.y + 18
        self._text("GAME STATE", x, y, self.title_font)
        y += 40
        state = snapshot.game_state
        action = snapshot.action.name.lower() if snapshot.action is not None else "-"
        status = "Paused" if snapshot.paused else "Running"
        rows = (
            ("Agent", snapshot.agent_name),
            ("Episode", snapshot.episode),
            ("Step", state.steps),
            ("Score", state.score),
            ("Reward", f"{snapshot.reward:.1f}"),
            ("Action", action),
            ("Direction", state.direction.name.lower()),
            ("Snake length", len(state.snake)),
            ("Food", state.food),
            ("Status", status),
            ("Speed", f"{snapshot.speed}x"),
        )
        for label, value in rows:
            self._text(f"{label}: {value}", x, y, self.font)
            y += 23

        y += 10
        self._text("LEARNING VIEW", x, y, self.title_font)
        y += 28
        self._text(snapshot.learning_view, x, y, self.font, ACCENT)
        y += 25
        for label, value in snapshot.metrics.items():
            self._text(f"{label}: {value}", x, y, self.small_font, MUTED)
            y += 19
        if snapshot.learning_view == "Baseline agent":
            y += 8
            self._text("Future agent panels:", x, y, self.small_font, MUTED)
            y += 19
            self._text("Q-table | network | backprop", x, y, self.small_font, MUTED)

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
            "Pause",
            self.controller.toggle_pause,
            active=lambda: self.controller.paused,
        )
        add("Step", lambda: self.controller.advance(force=True), width=64)
        for speed in SPEEDS:
            add(
                f"{speed}x",
                lambda selected=speed: self.controller.set_speed(selected),
                width=58,
                active=lambda selected=speed: (
                    not self.controller.paused and self.controller.speed == selected
                ),
            )
        add("Reset", self.controller.reset, width=70)

        self._draw_board_size_controls(area)

    def _draw_board_size_controls(self, area: pygame.Rect) -> None:
        y = area.y + 60
        self._text("Board:", area.x + 16, y + 7, self.small_font)
        for name, rect in self._input_rects().items():
            active = name == self.active_input
            pygame.draw.rect(self.surface, PANEL_LIGHT, rect, border_radius=3)
            pygame.draw.rect(self.surface, ACCENT if active else MUTED, rect, 1)
            value = self.width_text if name == "width" else self.height_text
            self._text(value, rect.x + 8, rect.y + 6, self.small_font)
        self._text("x", area.x + 202, y + 7, self.small_font)
        apply_rect = pygame.Rect(area.x + 282, y, 70, 30)
        button = Button("Apply", apply_rect, self._apply_board_size)
        self.buttons.append(button)
        button.draw(self.surface, self.small_font)
        self._text(
            "Arrow keys: steer manual | Space: pause | N: step | R: reset",
            area.x + 380,
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
    args = parser.parse_args()
    Dashboard(
        DashboardController(args.width, args.height, seed=args.seed, mode=args.mode)
    ).run()


if __name__ == "__main__":
    main()
