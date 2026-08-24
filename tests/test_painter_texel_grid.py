"""A paint job on a sign with a real texel grid lands every cell on its texel."""

from __future__ import annotations

import math
import time

import pytest

from PIL import Image

from app.input_controller import InputEvent, MockInputController
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
        "progress_callback_interval_seconds": 0.0,
        "safety_poll_interval_seconds": 0.002,
        "apply_brush_size": True,
        "verify_passes": 0,
        # Checking each color as it goes down captures the screen; the tests
        # that exercise it turn it on against a simulated sign.
        "confirm_strokes": False,
    }
    values.update(overrides)
    return PainterSettings(**values)  # type: ignore[arg-type]


class ReplayingTexelSign(SimulatedSign):
    """A texel sign driven by replaying everything the painter did.

    Size numbers typed into the field set the stamp's width in texels, a click
    on the clear control wipes the texture, a click in the hue bar starts a new
    color, and while the button is held every move paints the texels along
    its path - which is what the game does, and what turns a run of cells
    into a line.  With ``shift_lines`` the sign also has the game's line
    tool: while Shift is held the drag's path is not painted, and the
    release fills a straight run of texels from the press to the release,
    the way Rust draws it.  A cursor jump wider than ``max_drag_step_px``
    while pressed paints only where the cursor arrives, as the game does
    when it never samples the positions in between.
    """

    PALETTE = ((200, 30, 160), (30, 200, 60), (40, 90, 230), (240, 180, 20), (20, 20, 20))

    def __init__(
        self,
        controller: MockInputController,
        profile: CalibrationProfile,
        *,
        shift_lines: bool = False,
        min_dab_hold: float = 0.0,
        max_drag_step_px: float = 8.0,
        dead_columns: frozenset[int] = frozenset(),
        faithful_colors: bool = False,
        min_gap_seconds: float = 0.0,
        max_drag_texels_per_second: float = float("inf"),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.controller = controller
        self.profile = profile
        self.canvas = profile.canvas
        self.base = self.texture.copy()
        self.shift_lines = shift_lines
        self.shift_lines_drawn = 0
        self.max_drag_step_px = max_drag_step_px
        # Columns where a stamp narrower than Size 1.5 never takes: a sign
        # whose smallest brush is smaller than a texel, seen through the
        # cursor quantization that decides which texels it can reach.
        self.dead_columns = dead_columns
        # Paint the colour the painter asked for (a "select_color" marker the
        # test records after each pick) instead of cycling the palette on hue
        # clicks, so a verification pass can compare the sign with the plan.
        self.faithful_colors = faithful_colors
        # A press that begins sooner than this after the previous release is
        # merged into it by the game and paints nothing (needs event times).
        self.min_gap_seconds = min_gap_seconds
        # A drag commanded faster than this (a "drag_rate" marker the test
        # records before each stroke) paints only where its cursor events
        # arrive, not the path between - the game sampling too rarely.
        self.max_drag_texels_per_second = max_drag_texels_per_second
        # A press-and-release quicker than this paints nothing, like a game
        # frame that never sampled the button down.  Needs a controller that
        # records event times (TimedMockController); zero keeps every dab.
        self.min_dab_hold = min_dab_hold
        self.painted: dict[tuple[int, int], tuple[int, int, int]] = {}

    def _on_texture(self, x: float, y: float) -> bool:
        """The game takes paint clicks on the texture, not on a rectangle."""

        ox, oy = self.origin
        return (
            ox <= x < ox + self.columns * self.pitch[0]
            and oy <= y < oy + self.rows * self.pitch[1]
        )

    def _texel_at(self, x: float, y: float) -> tuple[int, int]:
        ox, oy = self.origin
        px, py = self.pitch
        return (
            math.floor((x - ox - self.cursor_shift[0]) / px) + self.stamp_offset[0],
            math.floor((y - oy - self.cursor_shift[1]) / py) + self.stamp_offset[1],
        )

    def _stamp_texel(self, column: int, row: int, size: float, color) -> None:
        if size < 1.5 and column in self.dead_columns:
            return
        reach = int(max(1.0, size) // 2)
        for r in range(row - reach, row + reach + 1):
            for c in range(column - reach, column + reach + 1):
                if 0 <= c < self.columns and 0 <= r < self.rows:
                    self.texture[r, c] = color
                    self.painted[(c, r)] = color

    def _stamp(self, x: float, y: float, size: float, color) -> None:
        column, row = self._texel_at(x, y)
        self._stamp_texel(column, row, size, color)

    def _replay(self) -> None:
        self.texture = self.base.copy()
        self.painted = {}
        self.shift_lines_drawn = 0
        size = 1.0
        digits = ""
        position = (0, 0)
        down = False
        shift_down = False
        anchor: tuple[int, int] | None = None
        color_index = -1
        color = self.PALETTE[0]
        times = getattr(self.controller, "event_times", None)
        # (column, row, size, color, event index of the press) of a press
        # whose fate depends on how long it stays down.
        pending: tuple[int, int, float, tuple[int, int, int], int] | None = None

        def flush_pending() -> None:
            nonlocal pending
            if pending is not None:
                self._stamp_texel(pending[0], pending[1], pending[2], pending[3])
                pending = None

        line_from: tuple[int, int] | None = None
        last_up_time: float | None = None
        pending_rate: float | None = None
        drag_rate: float | None = None
        unsampled_travel = 0.0
        for index, event in enumerate(self.controller.events):
            if event.kind == "move" and event.x is not None and event.y is not None:
                new_position = (event.x, event.y)
                if down and self._on_texture(*position):
                    # The cursor moved while pressed: this is a drag, whose
                    # press paints however short it was.
                    flush_pending()
                    if self.shift_lines and shift_down and line_from is not None:
                        pass  # the line tool paints nothing until the release
                    else:
                        travel = math.hypot(new_position[0] - position[0], new_position[1] - position[1])
                        too_fast = drag_rate is not None and drag_rate > self.max_drag_texels_per_second
                        if too_fast:
                            # Sampled too rarely: the game sees one position
                            # in several and paints only there.
                            unsampled_travel += travel
                            if unsampled_travel >= 12.0:
                                unsampled_travel = 0.0
                                if self._on_texture(*new_position):
                                    self._stamp(new_position[0], new_position[1], size, color)
                        elif travel > self.max_drag_step_px:
                            # Too far for one frame: the game never saw the
                            # positions between, only where the cursor arrived.
                            if self._on_texture(*new_position):
                                self._stamp(new_position[0], new_position[1], size, color)
                        else:
                            steps = max(1, int(travel))
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
                if value == "SHIFT":
                    shift_down = True
                    digits = ""
                elif value == 0xBE:
                    digits += "."
                elif isinstance(value, str) and len(value) == 1 and value.isdigit():
                    digits += value
                elif value == "ENTER":
                    size = float(digits) if digits else size
                    digits = ""
                else:
                    digits = ""
            elif event.kind == "key_up":
                if event.value == "SHIFT":
                    shift_down = False
            elif event.kind == "select_color":
                color = tuple(int(v) for v in event.value)  # type: ignore[union-attr]
            elif event.kind == "drag_rate":
                pending_rate = None if event.value is None else float(event.value)  # type: ignore[arg-type]
            elif event.kind == "mouse_down":
                down = True
                drag_rate, pending_rate = pending_rate, None
                if (
                    self.min_gap_seconds > 0.0
                    and times is not None
                    and index < len(times)
                    and last_up_time is not None
                    and times[index] - last_up_time < self.min_gap_seconds
                    and self._on_texture(*position)
                ):
                    # Too soon after the last release: the game never saw the
                    # button come up, so this press is part of the previous one.
                    down = False
                    continue
                if self.profile.clear_button.contains(*position):
                    self.texture = self.base.copy()
                    self.painted = {}
                    anchor = None
                elif self.profile.hue_bar.contains(*position):
                    if not self.faithful_colors:
                        color_index += 1
                        color = self.PALETTE[color_index % len(self.PALETTE)]
                elif self._on_texture(*position):
                    landed = self._texel_at(position[0], position[1])
                    if self.shift_lines and shift_down:
                        # The line tool: the press marks one end and paints
                        # it; the release will fill the straight run.
                        self._stamp(position[0], position[1], size, color)
                        line_from = landed
                    elif (
                        self.min_dab_hold > 0.0
                        and times is not None
                        and index < len(times)
                    ):
                        # Held long enough it paints on release; see mouse_up.
                        pending = (landed[0], landed[1], size, color, index)
                    else:
                        self._stamp(position[0], position[1], size, color)
                    anchor = landed
            elif event.kind == "mouse_up":
                if down and times is not None and index < len(times):
                    last_up_time = times[index]
                down = False
                if line_from is not None:
                    if self.shift_lines and shift_down and self._on_texture(*position):
                        # The game draws the straight run itself, texel to
                        # texel from the press to the release, however far
                        # the cursor jumped.
                        landed = self._texel_at(position[0], position[1])
                        steps = max(abs(landed[0] - line_from[0]), abs(landed[1] - line_from[1]), 1)
                        for step in range(steps + 1):
                            t = step / steps
                            self._stamp_texel(
                                round(line_from[0] + (landed[0] - line_from[0]) * t),
                                round(line_from[1] + (landed[1] - line_from[1]) * t),
                                size,
                                color,
                            )
                        self.shift_lines_drawn += 1
                    line_from = None
                if pending is not None:
                    held = (
                        times[index] - times[pending[4]]
                        if times is not None and index < len(times)
                        else self.min_dab_hold
                    )
                    if held >= self.min_dab_hold:
                        flush_pending()
                    else:
                        pending = None

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
    painter._CONFIRM_SETTLE_SECONDS = 0.0  # type: ignore[misc]
    painter._CLEAR_SETTLE_SECONDS = 0.0  # type: ignore[misc]
    painter._KEY_HOLD_SECONDS = 0.0  # type: ignore[misc]
    painter._KEY_GAP_SECONDS = 0.0  # type: ignore[misc]
    painter._SETTLE_FLOOR_SECONDS = 0.0  # type: ignore[misc]
    painter._STROKE_GAP_FLOOR_SECONDS = 0.0  # type: ignore[misc]
    painter._LONG_DRAG_MAX_TEXELS_PER_SECOND = float("inf")  # type: ignore[misc]
    painter._LONG_DRAG_MAX_STEP_TEXELS = float("inf")  # type: ignore[misc]
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


def _frameless(painter: Painter) -> Painter:
    """Strip only the waits that exist for a real sign to redraw.

    Unlike :func:`_impatient` this keeps the timing floors and the long-drag
    cap, which are what the test below is about.
    """

    painter._CAPTURE_SETTLE_SECONDS = 0.0  # type: ignore[misc]
    painter._CONFIRM_SETTLE_SECONDS = 0.0  # type: ignore[misc]
    painter._CLEAR_SETTLE_SECONDS = 0.0  # type: ignore[misc]
    painter._KEY_HOLD_SECONDS = 0.0  # type: ignore[misc]
    painter._KEY_GAP_SECONDS = 0.0  # type: ignore[misc]
    return painter


def test_a_long_drag_at_top_speed_paints_exactly_its_run_on_the_measured_grid() -> None:
    """Turbo as typed - 2200 px/s, an 8 px step, zero delays - on a measured
    sign: the drag is brought down to the texel rate with an event on every
    texel, and paints its cells and nothing past either end."""

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    sign = ReplayingTexelSign(
        controller,
        profile,
        columns=128,
        rows=64,
        origin=(99.4, 100.8),
        pitch=(5.02, 5.01),
        cursor_shift=(1.3, -0.7),
    )
    painter = _frameless(Painter(controller, screen_capture=sign.capture))
    plan = PaintPlan(
        128,
        64,
        (ColorGroup((40, 80, 160), (Stroke(4, 30, 123, 30),), 1),),
    )
    turbo = _settings(
        stroke_speed_pixels_per_second=2200.0,
        stroke_interpolation_step_pixels=8.0,
        mouse_down_duration_seconds=0.012,
    )

    assert painter.start(plan, profile, turbo)
    assert painter.wait(60.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    grid = painter.measured_texel_grid
    assert grid is not None and (grid.columns, grid.rows) == (128, 64)

    sign.capture(profile.canvas)  # replay to the end of the job
    assert set(sign.painted) == {(x, 30) for x in range(4, 124)}

    # The artwork stroke is the last press on the texture: find its moves.
    presses: list[list[tuple[int, int]]] = []
    down = False
    for event in controller.events:
        if event.kind == "mouse_down":
            down = True
            presses.append([])
        elif event.kind == "mouse_up":
            down = False
        elif down and event.kind == "move" and event.x is not None:
            presses[-1].append((event.x, event.y))
    moves = [m for m in presses if len(m) > 50][-1]
    assert all(abs(b[0] - a[0]) <= math.ceil(grid.pitch_x) for a, b in zip(moves, moves[1:]))
    assert len(moves) >= 119


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


def _measure_grid_on(sign_kwargs: dict) -> dict:
    """Run one sizing-on job so a profile can carry the grid it measured."""

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    sign = ReplayingTexelSign(controller, profile, **sign_kwargs)
    painter = _impatient(Painter(controller, screen_capture=sign.capture))
    plan = PaintPlan(
        128, 64, (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),)
    )
    assert painter.start(plan, profile, _settings())
    assert painter.wait(30.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    grid = painter.measured_texel_grid
    assert grid is not None
    return grid.to_dict()


def test_with_sizing_off_a_job_paints_on_the_grid_the_profile_stores() -> None:
    """Automatic sizing off means nothing is typed into the Size field and no
    probe is stamped - but the job still aims by the grid an earlier job
    measured, not by the hand-dragged rectangle, whose half-texel error
    leaves rows bare at native resolution."""

    sign_kwargs = dict(
        columns=128,
        rows=64,
        origin=(99.4, 100.8),
        pitch=(5.02, 5.01),
        cursor_shift=(1.3, -0.7),
    )
    stored = _measure_grid_on(sign_kwargs)

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    profile.metadata["texel_grid"] = stored
    sign = ReplayingTexelSign(controller, profile, **sign_kwargs)
    painter = _impatient(Painter(controller, screen_capture=sign.capture))
    plan = PaintPlan(
        128,
        64,
        (
            ColorGroup((40, 80, 160), (Stroke(10, 10, 30, 10), Stroke(0, 0, 0, 0)), 1),
            ColorGroup((200, 40, 40), (Stroke(64, 40, 64, 40), Stroke(127, 63, 127, 63)), 1),
        ),
    )

    assert painter.start(plan, profile, _settings(apply_brush_size=False))
    assert painter.wait(30.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason

    # Adopted, not measured: the Size field and clear control were never touched.
    assert painter.measured_texel_grid is not None
    assert painter.measured_texel_grid.to_dict() == stored
    assert not any(event.kind == "key_down" for event in controller.events)
    positions = []
    position = (0, 0)
    for event in controller.events:
        if event.kind == "move" and event.x is not None and event.y is not None:
            position = (event.x, event.y)
        elif event.kind == "mouse_down":
            positions.append(position)
    assert profile.clear_button is not None
    assert not any(profile.clear_button.contains(*p) for p in positions)

    sign.capture(profile.canvas)  # replay to the end of the job
    expected = {(x, 10) for x in range(10, 31)} | {(0, 0), (64, 40), (127, 63)}
    assert set(sign.painted) == expected
    assert not controller.held_buttons


def test_a_stored_grid_off_the_rectangle_is_not_painted_on() -> None:
    """A grid measured with the sign framed elsewhere is stale: the rectangle
    re-dragged around the new framing no longer holds it, and the job falls
    back to the rectangle rather than aim at where the sign used to be."""

    from app.texel_grid import TexelGridModel

    controller = MockInputController()
    profile = _profile()
    assert profile.canvas is not None
    stale = TexelGridModel(
        columns=128,
        rows=64,
        pitch_x=5.0,
        pitch_y=5.0,
        origin_x=profile.canvas.left + 0.2 * profile.canvas.width,
        origin_y=profile.canvas.top,
    )
    profile.metadata["texel_grid"] = stale.to_dict()
    painter = Painter(controller)
    plan = PaintPlan(128, 64, (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),))
    painter.configure(plan, profile, _settings(apply_brush_size=False))
    job = painter._job
    assert job is not None
    assert job.target.texel_grid is not None
    painter._adopt_stored_texel_grid(job)
    assert job.texel_grid is None
    assert painter.measured_texel_grid is None


def test_long_straight_runs_go_down_as_shift_lines_when_the_probe_proves_them() -> None:
    """On a sign with the line mechanic, the probe proves it and long rows
    become a press, a jump and a release with Shift held; short runs keep
    the glide."""

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    sign = ReplayingTexelSign(
        controller,
        profile,
        shift_lines=True,
        columns=128,
        rows=64,
        origin=(99.4, 100.8),
        pitch=(5.02, 5.01),
        cursor_shift=(1.3, -0.7),
    )
    painter = _impatient(Painter(controller, screen_capture=sign.capture))
    plan = PaintPlan(
        128,
        64,
        (
            ColorGroup(
                (40, 80, 160), (Stroke(4, 30, 123, 30), Stroke(4, 32, 123, 32)), 1
            ),
            ColorGroup((200, 40, 40), (Stroke(20, 40, 40, 40),), 1),
        ),
    )

    assert painter.start(plan, profile, _settings())
    assert painter.wait(30.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason

    sign.capture(profile.canvas)  # replay to the end of the job
    expected = (
        {(x, 30) for x in range(4, 124)}
        | {(x, 32) for x in range(4, 124)}
        | {(x, 40) for x in range(20, 41)}
    )
    assert set(sign.painted) == expected
    # One line proved the tool, the two long rows each drew one; the
    # 21-texel run is below the threshold and stayed a drag.
    assert sign.shift_lines_drawn == 3
    assert not controller.held_buttons
    assert not controller.held_keys


def test_a_sign_without_the_line_mechanic_keeps_painting_drags() -> None:
    """A Shift-held jump that paints only its two ends fails the probe, so
    every run is dragged exactly as before and Shift is never held again."""

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
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
    plan = PaintPlan(
        128,
        64,
        (ColorGroup((40, 80, 160), (Stroke(4, 30, 123, 30),), 1),),
    )

    assert painter.start(plan, profile, _settings())
    assert painter.wait(30.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason

    sign.capture(profile.canvas)  # replay to the end of the job
    assert set(sign.painted) == {(x, 30) for x in range(4, 124)}
    assert sign.shift_lines_drawn == 0
    holds = [
        event
        for event in controller.events
        if event.kind == "key_down" and event.value == "SHIFT"
    ]
    assert len(holds) == 1  # the probe's one refused stroke
    assert not controller.held_keys


def test_the_line_tool_switch_keeps_every_run_a_drag_without_probing() -> None:
    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    sign = ReplayingTexelSign(
        controller,
        profile,
        shift_lines=True,
        columns=128,
        rows=64,
        origin=(99.4, 100.8),
        pitch=(5.02, 5.01),
        cursor_shift=(1.3, -0.7),
    )
    painter = _impatient(Painter(controller, screen_capture=sign.capture))
    plan = PaintPlan(
        128,
        64,
        (ColorGroup((40, 80, 160), (Stroke(4, 30, 123, 30),), 1),),
    )

    assert painter.start(plan, profile, _settings(use_line_tool=False))
    assert painter.wait(30.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason

    sign.capture(profile.canvas)  # replay to the end of the job
    assert set(sign.painted) == {(x, 30) for x in range(4, 124)}
    assert sign.shift_lines_drawn == 0
    assert not any(
        event.kind == "key_down" and event.value == "SHIFT"
        for event in controller.events
    )


class TimedMockController(MockInputController):
    """A mock controller that also remembers when each event happened.

    The hold-sensitive sign reads a press's duration from these times the
    way the game reads it from its frame clock.
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.event_times: list[float] = []

    def _stamp_times(self) -> None:
        while len(self.event_times) < len(self.events):
            self.event_times.append(time.monotonic())

    def move_mouse(self, x, y):  # type: ignore[override]
        super().move_mouse(x, y)
        self._stamp_times()

    def mouse_down(self, button="left"):  # type: ignore[override]
        super().mouse_down(button)
        self._stamp_times()

    def mouse_up(self, button="left"):  # type: ignore[override]
        super().mouse_up(button)
        self._stamp_times()

    def press_key(self, key, *, hold_seconds=0.01):  # type: ignore[override]
        super().press_key(key, hold_seconds=hold_seconds)
        self._stamp_times()

    def key_down(self, key):  # type: ignore[override]
        super().key_down(key)
        self._stamp_times()

    def key_up(self, key):  # type: ignore[override]
        super().key_up(key)
        self._stamp_times()


def test_the_press_hold_probe_adopts_the_shortest_hold_with_a_margin_step() -> None:
    """All candidates land on a forgiving sign, so the job's dabs run at the
    second-shortest hold - one proven step above the shortest clean one."""

    controller = TimedMockController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
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
    painter._PRESS_HOLD_PROBE_MIN_STROKES = 0  # type: ignore[misc]
    painter._PRESS_HOLD_PROBE_DOTS = 12  # type: ignore[misc]
    painter._PRESS_HOLD_PROBE_CANDIDATES = (0.03, 0.02, 0.012)  # type: ignore[misc]
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
    assert painter.measured_press_hold_seconds == pytest.approx(0.02)

    sign.capture(profile.canvas)  # replay to the end of the job
    expected = {(x, 10) for x in range(10, 31)} | {(0, 0), (64, 40), (127, 63)}
    assert set(sign.painted) == expected


def test_a_sign_that_drops_quick_presses_keeps_a_proven_longer_hold() -> None:
    """The descent stops at the first dropped dot: the adopted hold is the
    shortest clean one with a clean step below it, and painting stays exact."""

    controller = TimedMockController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    sign = ReplayingTexelSign(
        controller,
        profile,
        min_dab_hold=0.055,
        columns=128,
        rows=64,
        origin=(99.4, 100.8),
        pitch=(5.02, 5.01),
        cursor_shift=(1.3, -0.7),
    )
    painter = _impatient(Painter(controller, screen_capture=sign.capture))
    painter._PRESS_HOLD_PROBE_MIN_STROKES = 0  # type: ignore[misc]
    painter._PRESS_HOLD_PROBE_DOTS = 12  # type: ignore[misc]
    # 30 ms cannot reach the sign's 55 ms threshold even with scheduler
    # slack; the two above it always land, because sleeps never undershoot.
    # 80 ms is clean but its step below dropped dots, so it carries no
    # margin: the probe settles on 100 ms, whose step below proved clean.
    painter._PRESS_HOLD_PROBE_CANDIDATES = (0.10, 0.08, 0.03)  # type: ignore[misc]
    plan = PaintPlan(
        128,
        64,
        (ColorGroup((40, 80, 160), (Stroke(10, 10, 30, 10), Stroke(64, 40, 64, 40)), 1),),
    )

    assert painter.start(plan, profile, _settings())
    assert painter.wait(60.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.measured_press_hold_seconds == pytest.approx(0.10)

    sign.capture(profile.canvas)  # replay to the end of the job
    expected = {(x, 10) for x in range(10, 31)} | {(64, 40)}
    assert set(sign.painted) == expected


def test_the_press_hold_switch_keeps_the_floor_without_probing() -> None:
    controller = TimedMockController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
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
    painter._PRESS_HOLD_PROBE_MIN_STROKES = 0  # type: ignore[misc]
    plan = PaintPlan(
        128,
        64,
        (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),),
    )

    assert painter.start(plan, profile, _settings(measure_press_hold=False))
    assert painter.wait(30.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.measured_press_hold_seconds is None

    sign.capture(profile.canvas)  # replay to the end of the job
    assert set(sign.painted) == {(0, 0)}


def _record_color_picks(painter: Painter, controller: MockInputController) -> None:
    """Append a marker event after every colour pick, for a faithful sign."""

    original = painter._select_color

    def select(color, target, settings, epoch, **kwargs):  # type: ignore[no-untyped-def]
        original(color, target, settings, epoch, **kwargs)
        controller.events.append(InputEvent("select_color", value=tuple(color)))  # type: ignore[arg-type]

    painter._select_color = select  # type: ignore[method-assign]


def _record_drag_rates(painter: Painter, controller: MockInputController) -> None:
    """Append a marker with the commanded drag rate before every stroke."""

    original = painter._execute_stroke

    def execute(*args, **kwargs):  # type: ignore[no-untyped-def]
        controller.events.append(InputEvent("drag_rate", value=kwargs.get("drag_rate")))  # type: ignore[arg-type]
        return original(*args, **kwargs)

    painter._execute_stroke = execute  # type: ignore[method-assign]


def _presses_after_last_clear(controller: MockInputController, profile: CalibrationProfile):
    """(press position, moves while held) for every press after the sign was cleared."""

    presses: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    position = (0, 0)
    down = False
    for event in controller.events:
        if event.kind == "move" and event.x is not None and event.y is not None:
            position = (event.x, event.y)
            if down:
                presses[-1][1].append(position)
        elif event.kind == "mouse_down":
            down = True
            if profile.clear_button.contains(*position):
                presses = []
                presses.append((position, []))
            else:
                presses.append((position, []))
        elif event.kind == "mouse_up":
            down = False
    canvas = profile.canvas
    return [
        p
        for p in presses
        if canvas.left - 2 <= p[0][0] <= canvas.left + canvas.width + 2
        and canvas.top - 2 <= p[0][1] <= canvas.top + canvas.height + 2
    ]


def test_a_native_lone_dab_is_one_stationary_press() -> None:
    """On a measured native grid a dab has no sideways reach: no micro-drag
    whose release could land in the neighbouring texel."""

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
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

    presses = _presses_after_last_clear(controller, profile)
    stationary = [p for p in presses if not p[1]]
    micro = [p for p in presses if p[1] and max(abs(m[0] - p[0][0]) + abs(m[1] - p[0][1]) for m in p[1]) <= 3]
    assert len(stationary) >= 3, presses  # the three lone dabs
    assert micro == [], micro
    sign.capture(profile.canvas)  # replay to the end of the job
    expected = {(x, 10) for x in range(10, 31)} | {(0, 0), (64, 40), (127, 63)}
    assert set(sign.painted) == expected


def _dab_plan() -> PaintPlan:
    dabs = tuple(Stroke(x, y, x, y) for y in (12, 30, 50) for x in (20, 25, 30, 61, 100))
    return PaintPlan(
        128,
        64,
        (
            ColorGroup((40, 80, 160), dabs, 1),
            ColorGroup((200, 40, 40), (Stroke(4, 40, 60, 40),), 1),
        ),
    )


def test_the_dab_probe_raises_the_one_cell_brush_until_lone_dabs_land() -> None:
    """Columns 20, 25 and 30 swallow the smallest brush; the probe sees its
    dots go missing there, steps the Size up until they land, and the job's
    dabs and runs then paint exactly."""

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    sign = ReplayingTexelSign(
        controller,
        profile,
        dead_columns=frozenset({20, 25, 30}),
        columns=128,
        rows=64,
        origin=(99.4, 100.8),
        pitch=(5.02, 5.01),
        cursor_shift=(1.3, -0.7),
    )
    painter = _impatient(Painter(controller, screen_capture=sign.capture))
    painter._DAB_PROBE_MIN_DABS = 0  # type: ignore[misc]
    painter._DAB_PROBE_DOTS = 24  # type: ignore[misc]
    plan = _dab_plan()

    assert painter.start(plan, profile, _settings())
    assert painter.wait(60.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.measured_detail_size == pytest.approx(1.5)

    sign.capture(profile.canvas)  # replay to the end of the job
    expected = {(x, y) for y in (12, 30, 50) for x in (20, 25, 30, 61, 100)} | {
        (x, 40) for x in range(4, 61)
    }
    assert set(sign.painted) == expected


def test_the_dab_probe_keeps_the_smallest_brush_when_every_dab_lands() -> None:
    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
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
    painter._DAB_PROBE_MIN_DABS = 0  # type: ignore[misc]
    painter._DAB_PROBE_DOTS = 24  # type: ignore[misc]

    assert painter.start(_dab_plan(), profile, _settings())
    assert painter.wait(60.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.measured_detail_size == pytest.approx(1.0)


def test_the_dab_probe_switch_leaves_the_one_cell_brush_unmeasured() -> None:
    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    sign = ReplayingTexelSign(
        controller,
        profile,
        dead_columns=frozenset({20, 25, 30}),
        columns=128,
        rows=64,
        origin=(99.4, 100.8),
        pitch=(5.02, 5.01),
        cursor_shift=(1.3, -0.7),
    )
    painter = _impatient(Painter(controller, screen_capture=sign.capture))
    painter._DAB_PROBE_MIN_DABS = 0  # type: ignore[misc]

    assert painter.start(_dab_plan(), profile, _settings(measure_dab_size=False))
    assert painter.wait(60.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.measured_detail_size is None
    sign.capture(profile.canvas)  # replay to the end of the job
    # Without the probe the dead columns stay bare - the failure the probe exists for.
    assert not any(column in {20, 25, 30} for column, _ in sign.painted)


def test_a_touch_up_only_job_raises_the_brush_for_holes_that_survive_a_pass() -> None:
    """A finished sign with holes the smallest brush cannot reach, touched up
    as it is: the job starts at its last stroke, so nothing is cleared or
    probed and every pass is a capture and a repaint - and a hole that
    survives a pass is repainted with the next Size up until it fills."""

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    sign = ReplayingTexelSign(
        controller,
        profile,
        dead_columns=frozenset({20, 25, 30}),
        faithful_colors=True,
        columns=128,
        rows=64,
        origin=(99.4, 100.8),
        pitch=(5.02, 5.01),
        cursor_shift=(1.3, -0.7),
    )
    plan = _dab_plan()
    expected = {(x, y) for y in (12, 30, 50) for x in (20, 25, 30, 61, 100)} | {
        (x, 40) for x in range(4, 61)
    }

    # The nine-hour sign: painted without the dab probe, so the dead columns
    # came out bare.
    painter = _impatient(Painter(controller, screen_capture=sign.capture))
    _record_color_picks(painter, controller)
    assert painter.start(plan, profile, _settings(measure_dab_size=False))
    assert painter.wait(60.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    sign.capture(profile.canvas)
    assert not any(column in {20, 25, 30} for column, _ in sign.painted)
    assert set(sign.painted) < expected
    grid = painter.measured_texel_grid
    model = painter.measured_brush_size_model
    assert grid is not None and model is not None

    # Touch it up as it is: the profile carries what the first job measured,
    # the job begins at its last stroke, and three passes are allowed.
    profile.metadata["texel_grid"] = grid.to_dict()
    profile.metadata["brush_size_model"] = model.to_dict()
    touch_up = _impatient(Painter(controller, screen_capture=sign.capture))
    _record_color_picks(touch_up, controller)
    assert touch_up.start(
        plan,
        profile,
        _settings(verify_passes=3),
        start_stroke=plan.stroke_count,
    )
    assert touch_up.wait(120.0)
    assert touch_up.state is PainterState.COMPLETED, touch_up.state_reason
    # Pass 1 repainted at the smallest brush and missed; pass 2 raised it to
    # 1.25 and missed; pass 3 raised it to 1.5, which lands.
    assert touch_up.measured_detail_size == pytest.approx(1.5)

    sign.capture(profile.canvas)  # replay to the end of both jobs
    assert set(sign.painted) == expected
    assert not any(
        profile.clear_button.contains(*_position_at(controller, index))
        for index, event in enumerate(controller.events)
        if event.kind == "mouse_down" and index > _last_clear_index(controller, profile)
    )


def _position_at(controller: MockInputController, index: int) -> tuple[int, int]:
    position = (0, 0)
    for event in controller.events[: index + 1]:
        if event.kind == "move" and event.x is not None and event.y is not None:
            position = (event.x, event.y)
    return position


def _last_clear_index(controller: MockInputController, profile: CalibrationProfile) -> int:
    last = -1
    position = (0, 0)
    for index, event in enumerate(controller.events):
        if event.kind == "move" and event.x is not None and event.y is not None:
            position = (event.x, event.y)
        elif event.kind == "mouse_down" and profile.clear_button.contains(*position):
            last = index
    return last


def test_the_touch_up_brush_rises_only_when_repaints_did_not_take() -> None:
    import numpy as np

    from app.painter import PainterSettings, _Job

    controller = MockInputController()
    painter = Painter(controller, screen_capture=lambda rect: Image.new("RGB", (rect.width, rect.height)))
    plan = _dab_plan()
    job = _Job(plan, object(), PainterSettings(apply_brush_size=True))  # type: ignore[arg-type]
    previous = np.zeros((64, 128), dtype=bool)
    previous[12, 20:60] = True  # 40 cells repainted last pass
    still = np.zeros((64, 128), dtype=bool)
    still[12, 20:25] = True  # 5 of them still wrong: below the floor of 10
    painter._escalate_touch_up_brush(job, 2, still, previous)
    assert painter.measured_detail_size is None
    still[12, 20:32] = True  # 12 still wrong: 30% of the 40, over both bars
    painter._escalate_touch_up_brush(job, 2, still, previous)
    assert painter.measured_detail_size == pytest.approx(1.25)
    painter._escalate_touch_up_brush(job, 3, still, previous)
    assert painter.measured_detail_size == pytest.approx(1.5)
    for _ in range(5):
        painter._escalate_touch_up_brush(job, 4, still, previous)
    assert painter.measured_detail_size == pytest.approx(2.0)  # the ladder's top
    # Off means off, whatever the holes say.
    quiet = _Job(plan, object(), PainterSettings(apply_brush_size=True, measure_dab_size=False))  # type: ignore[arg-type]
    painter._measured_detail_size = None
    painter._escalate_touch_up_brush(quiet, 2, still, previous)
    assert painter.measured_detail_size is None


def test_the_stroke_gap_probe_adopts_the_shortest_gap_with_a_margin_step() -> None:
    """A sign that merges presses closer than 15 ms: the 4 ms batch loses
    dots, so the gap adopted is the shortest clean one with a clean step
    below it - and the artwork's dabs all land at it."""

    controller = TimedMockController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    sign = ReplayingTexelSign(
        controller,
        profile,
        min_gap_seconds=0.015,
        faithful_colors=True,
        columns=128,
        rows=64,
        origin=(99.4, 100.8),
        pitch=(5.02, 5.01),
        cursor_shift=(1.3, -0.7),
    )
    # Frameless, not impatient: the between-strokes floor must stay so the
    # grid probe's own dots are kept apart on this sign.
    painter = _frameless(Painter(controller, screen_capture=sign.capture))
    _record_color_picks(painter, controller)
    painter._PRESS_HOLD_PROBE_MIN_STROKES = 0  # type: ignore[misc]
    painter._PRESS_HOLD_PROBE_DOTS = 12  # type: ignore[misc]
    painter._PRESS_HOLD_PROBE_CANDIDATES = (0.03, 0.02)  # type: ignore[misc]
    # Sleeps never undershoot, so 40 and 25 ms always clear the sign's 15 ms;
    # 4 ms plus any scheduler slack cannot reach it.
    painter._STROKE_GAP_PROBE_CANDIDATES = (0.040, 0.025, 0.004)  # type: ignore[misc]
    plan = PaintPlan(
        128,
        64,
        (ColorGroup((40, 80, 160), (Stroke(10, 10, 30, 10), Stroke(0, 0, 0, 0), Stroke(64, 40, 64, 40)), 1),),
    )

    assert painter.start(plan, profile, _settings(delay_between_strokes_seconds=0.05))
    assert painter.wait(120.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.measured_stroke_gap_seconds == pytest.approx(0.040)

    sign.capture(profile.canvas)  # replay to the end of the job
    expected = {(x, 10) for x in range(10, 31)} | {(0, 0), (64, 40)}
    assert set(sign.painted) == expected


def test_the_drag_rate_probe_adopts_the_second_fastest_clean_rate() -> None:
    """The sign paints drags whole up to 700 texels/s: 250, 400 and 600 are
    clean and 900 is not, so 400 - the fastest with a proven step above
    it - is adopted, and the artwork's long run paints exactly at it."""

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    sign = ReplayingTexelSign(
        controller,
        profile,
        max_drag_texels_per_second=700.0,
        columns=128,
        rows=64,
        origin=(99.4, 100.8),
        pitch=(5.02, 5.01),
        cursor_shift=(1.3, -0.7),
    )
    painter = _impatient(Painter(controller, screen_capture=sign.capture))
    painter._LONG_DRAG_MAX_TEXELS_PER_SECOND = 250.0  # type: ignore[misc]
    _record_drag_rates(painter, controller)
    plan = PaintPlan(
        128,
        64,
        (ColorGroup((40, 80, 160), (Stroke(4, 30, 123, 30),), 1),),
    )

    assert painter.start(plan, profile, _settings())
    assert painter.wait(120.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.measured_drag_rate == pytest.approx(400.0)

    sign.capture(profile.canvas)  # replay to the end of the job
    assert set(sign.painted) == {(x, 30) for x in range(4, 124)}
    rates = [event.value for event in controller.events if event.kind == "drag_rate"]
    assert rates[-1] == pytest.approx(400.0)  # the artwork's run was commanded at it


def test_a_sign_that_only_takes_the_floor_rate_keeps_it() -> None:
    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    sign = ReplayingTexelSign(
        controller,
        profile,
        max_drag_texels_per_second=300.0,
        columns=128,
        rows=64,
        origin=(99.4, 100.8),
        pitch=(5.02, 5.01),
        cursor_shift=(1.3, -0.7),
    )
    painter = _impatient(Painter(controller, screen_capture=sign.capture))
    painter._LONG_DRAG_MAX_TEXELS_PER_SECOND = 250.0  # type: ignore[misc]
    _record_drag_rates(painter, controller)
    plan = PaintPlan(128, 64, (ColorGroup((40, 80, 160), (Stroke(4, 30, 123, 30),), 1),))

    assert painter.start(plan, profile, _settings())
    assert painter.wait(120.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.measured_drag_rate is None
    sign.capture(profile.canvas)
    assert set(sign.painted) == {(x, 30) for x in range(4, 124)}


def test_a_cleared_sign_teaches_the_profile_its_wood_for_later_touch_ups() -> None:
    """A job that clears the sign measures the wood; a later touch-up-only
    job on the same profile reads its holes against that colour."""

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    sign = ReplayingTexelSign(
        controller,
        profile,
        faithful_colors=True,
        columns=128,
        rows=64,
        origin=(99.4, 100.8),
        pitch=(5.02, 5.01),
        cursor_shift=(1.3, -0.7),
    )
    painter = _impatient(Painter(controller, screen_capture=sign.capture))
    _record_color_picks(painter, controller)
    plan = _dab_plan()

    assert painter.start(plan, profile, _settings())
    assert painter.wait(60.0)
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    measured = painter.measured_bare_color
    assert measured is not None and len(measured) == 3

    # What the GUI stores, and what a later job reads back.
    profile.metadata["bare_sign_color"] = list(measured)
    profile.metadata["texel_grid"] = painter.measured_texel_grid.to_dict()
    profile.metadata["brush_size_model"] = painter.measured_brush_size_model.to_dict()
    from app.painter import PaintingTarget

    assert PaintingTarget.from_profile(profile).bare_color == tuple(measured)
