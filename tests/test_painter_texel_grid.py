"""A paint job on a sign with a real texel grid lands every cell on its texel."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image

from app.input_controller import MockInputController
from app.models import ColorGroup, PaintPlan, ScreenRect, Stroke
from app.painter import Painter, PainterSettings, PainterState
from app.profiles import CalibrationProfile
from test_texel_grid import SimulatedSign


def _profile() -> CalibrationProfile:
    return CalibrationProfile.new(
        "Texel sign",
        canvas=ScreenRect(100, 100, 640, 320),
        color_box=ScreenRect(600, 500, 100, 100),
        hue_bar=ScreenRect(720, 500, 12, 100),
        brush_size_box=ScreenRect(800, 100, 60, 24),
        clear_button=ScreenRect(880, 100, 24, 24),
    )


def _settings(**overrides: object) -> PainterSettings:
    values: dict[str, object] = {
        "countdown_seconds": 0.0,
        "mouse_down_duration_seconds": 0.0,
        "delay_after_hue_seconds": 0.0,
        "delay_after_saturation_value_seconds": 0.0,
        "delay_between_strokes_seconds": 0.0,
        "delay_between_colors_seconds": 0.0,
        "stroke_speed_pixels_per_second": 20_000.0,
        "stroke_interpolation_step_pixels": 4.0,
        "corner_abort_enabled": False,
        "progress_callback_interval_seconds": 0.0,
        "safety_poll_interval_seconds": 0.002,
        "apply_brush_size": True,
        "verify_passes": 0,
    }
    values.update(overrides)
    return PainterSettings(**values)  # type: ignore[arg-type]


class ReplayingTexelSign(SimulatedSign):
    """A texel sign driven by replaying everything the painter did.

    Size numbers typed into the field set the stamp's width in texels, a click
    on the clear control wipes the texture, a click in the hue bar starts a new
    color, and while the button is held every move paints the texels along
    its path - which is what the game does, and what turns a run of cells
    into a line.
    """

    PALETTE = ((200, 30, 160), (30, 200, 60), (40, 90, 230), (240, 180, 20), (20, 20, 20))

    def __init__(self, controller: MockInputController, profile: CalibrationProfile, **kwargs) -> None:
        super().__init__(**kwargs)
        self.controller = controller
        self.profile = profile
        self.canvas = profile.canvas
        self.base = self.texture.copy()
        self.painted: dict[tuple[int, int], tuple[int, int, int]] = {}

    def _on_texture(self, x: float, y: float) -> bool:
        """The game takes paint clicks on the texture, not on a rectangle."""

        ox, oy = self.origin
        return (
            ox <= x < ox + self.columns * self.pitch[0]
            and oy <= y < oy + self.rows * self.pitch[1]
        )

    def _stamp(self, x: float, y: float, size: float, color) -> None:
        ox, oy = self.origin
        px, py = self.pitch
        column = math.floor((x - ox - self.cursor_shift[0]) / px) + self.stamp_offset[0]
        row = math.floor((y - oy - self.cursor_shift[1]) / py) + self.stamp_offset[1]
        reach = int(max(1.0, size) // 2)
        for r in range(row - reach, row + reach + 1):
            for c in range(column - reach, column + reach + 1):
                if 0 <= c < self.columns and 0 <= r < self.rows:
                    self.texture[r, c] = color
                    self.painted[(c, r)] = color

    def _replay(self) -> None:
        self.texture = self.base.copy()
        self.painted = {}
        size = 1.0
        digits = ""
        position = (0, 0)
        down = False
        color_index = -1
        color = self.PALETTE[0]
        for event in self.controller.events:
            if event.kind == "move" and event.x is not None and event.y is not None:
                new_position = (event.x, event.y)
                if down and self._on_texture(*position):
                    steps = max(1, int(math.hypot(new_position[0] - position[0], new_position[1] - position[1])))
                    for step in range(1, steps + 1):
                        t = step / steps
                        self._stamp(
                            position[0] + (new_position[0] - position[0]) * t,
                            position[1] + (new_position[1] - position[1]) * t,
                            size,
                            color,
                        )
                position = new_position
            elif event.kind == "key_down":
                value = event.value
                if value == 0xBE:
                    digits += "."
                elif isinstance(value, str) and len(value) == 1 and value.isdigit():
                    digits += value
                elif value == "ENTER":
                    size = float(digits) if digits else size
                    digits = ""
                else:
                    digits = ""
            elif event.kind == "mouse_down":
                down = True
                if self.profile.clear_button.contains(*position):
                    self.texture = self.base.copy()
                    self.painted = {}
                elif self.profile.hue_bar.contains(*position):
                    color_index += 1
                    color = self.PALETTE[color_index % len(self.PALETTE)]
                elif self._on_texture(*position):
                    self._stamp(position[0], position[1], size, color)
            elif event.kind == "mouse_up":
                down = False

    def capture(self, rect) -> Image.Image:
        canvas = self.canvas
        margin_rect = (
            rect.left < canvas.left
            and rect.top < canvas.top
            and rect.left + rect.width > canvas.left + canvas.width
        )
        if (rect.left, rect.top) != (canvas.left, canvas.top) and not margin_rect:
            return Image.new("RGB", (rect.width, rect.height), (21, 21, 12))
        self._replay()
        return super().capture(rect)


def _impatient(painter: Painter) -> Painter:
    painter._CAPTURE_SETTLE_SECONDS = 0.0  # type: ignore[misc]
    painter._CLEAR_SETTLE_SECONDS = 0.0  # type: ignore[misc]
    painter._KEY_HOLD_SECONDS = 0.0  # type: ignore[misc]
    painter._KEY_GAP_SECONDS = 0.0  # type: ignore[misc]
    return painter


def test_a_paint_job_lands_every_cell_on_its_texel() -> None:
    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    # 128x64 texels on a quad a little larger than, and offset from, the
    # hand-dragged rectangle, with the cursor mapping sitting off the lattice.
    # (A stamp that lands a whole texel from the cursor's is covered by the
    # grid tests; here it would put the far column outside the quad, where
    # the painter rightly never sends the mouse.)
    sign = ReplayingTexelSign(
        controller,
        profile,
        columns=128,
        rows=64,
        origin=(99.4, 100.8),
        pitch=(5.02, 5.01),
        cursor_shift=(1.3, -0.7),
    )
    painter = _impatient(Painter(controller, screen_capture=sign.capture))
    # A native-resolution plan: a run, a lone dot, and both far corners.
    plan = PaintPlan(
        128,
        64,
        (
            ColorGroup((40, 80, 160), (Stroke(10, 10, 30, 10), Stroke(0, 0, 0, 0)), 1),
            ColorGroup((200, 40, 40), (Stroke(64, 40, 64, 40), Stroke(127, 63, 127, 63)), 1),
        ),
    )

    assert painter.start(plan, profile, _settings())
    assert painter.wait(30.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason

    grid = painter.measured_texel_grid
    assert grid is not None
    assert (grid.columns, grid.rows) == (128, 64)
    assert abs(grid.pitch_x - 5.02) < 0.003
    assert abs(grid.origin_x - 99.4) < 0.3

    sign.capture(profile.canvas)  # replay to the end of the job
    expected = {(x, 10) for x in range(10, 31)} | {(0, 0), (64, 40), (127, 63)}
    assert set(sign.painted) == expected
    assert not controller.held_buttons


def _wide_profile() -> CalibrationProfile:
    return CalibrationProfile.new(
        "Non-canonical sign",
        canvas=ScreenRect(100, 100, 900, 450),
        color_box=ScreenRect(1100, 500, 100, 100),
        hue_bar=ScreenRect(1220, 500, 12, 100),
        brush_size_box=ScreenRect(1300, 100, 60, 24),
        clear_button=ScreenRect(1380, 100, 24, 24),
    )


def _paint_non_canonical_sign(measure_texel_grid: bool) -> tuple[set[tuple[int, int]], set[tuple[int, int]], Painter]:
    """Paint a dotted cross on a 300x150 sign, a size no table lists."""

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _wide_profile()
    sign = ReplayingTexelSign(
        controller,
        profile,
        columns=300,
        rows=150,
        origin=(99.2, 100.9),
        pitch=(3.004, 2.998),
        cursor_shift=(0.9, -0.4),
    )
    painter = _impatient(Painter(controller, screen_capture=sign.capture))
    plan = PaintPlan(
        300,
        150,
        (
            ColorGroup((40, 80, 160), tuple(Stroke(x, 75, x, 75) for x in range(0, 300, 7)), 1),
            ColorGroup((200, 40, 40), tuple(Stroke(150, y, 150, y) for y in range(0, 150, 5)), 1),
        ),
    )
    assert painter.start(plan, profile, _settings(measure_texel_grid=measure_texel_grid))
    assert painter.wait(60.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    sign.capture(profile.canvas)
    expected = {(x, 75) for x in range(0, 300, 7)} | {(150, y) for y in range(0, 150, 5)}
    return set(sign.painted), expected, painter


def test_a_sign_of_no_canonical_size_is_painted_exactly_only_with_the_grid() -> None:
    """The brush-derived count snaps 300 to 320; the measured grid does not.

    Both runs paint the same plan on the same sign.  With the grid measured
    every cell lands on its texel; without it the inferred 320-column grid
    stretches the strokes across the 300-column texture and most of them
    land a texel or more off.  The second half is not a behaviour to keep,
    it is the reason the first half exists.
    """

    painted, expected, painter = _paint_non_canonical_sign(True)
    grid = painter.measured_texel_grid
    assert grid is not None and (grid.columns, grid.rows) == (300, 150)
    assert painted == expected

    painted, expected, painter = _paint_non_canonical_sign(False)
    assert painter.measured_texel_grid is None
    assert len(painted & expected) < len(expected) // 2
