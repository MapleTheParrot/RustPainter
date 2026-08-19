from __future__ import annotations

import logging
import os
import threading
import time

import pytest
from PIL import Image, ImageDraw

from app.color_calibration import ColorCorrectionModel
from app.color_mapping import map_rgb_to_picker
from app.input_controller import DryRunInputController, MockInputController
from app.models import ColorGroup, PaintPlan, ScreenRect, Stroke
from app.painter import Painter, PainterSettings, PainterState, PaintingTarget
from app.screen import VirtualScreen
from app.brush_calibration import fit_brush_size_model
from app.profiles import CalibrationProfile


def _profile(*, canvas_width: int = 400) -> CalibrationProfile:
    return CalibrationProfile.new(
        "Test",
        canvas=ScreenRect(100, 100, canvas_width, 80),
        color_box=ScreenRect(600, 100, 100, 100),
        hue_bar=ScreenRect(720, 100, 12, 100),
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
    }
    values.update(overrides)
    return PainterSettings(**values)  # type: ignore[arg-type]


def _panel_capture(rect) -> Image.Image:
    """A flat capture of Rust's panel colour.

    The mouse-safety tests declare real input so the guards engage, which also
    turns on the picker measurement.  Handing them a stub keeps them off the
    real desktop: a flat region has no widget in it, so the measurement finds
    nothing to trim and the calibration is used exactly as written.
    """

    return Image.new("RGB", (rect.width, rect.height), (21, 21, 12))


def _dot_plan(count: int) -> PaintPlan:
    strokes = tuple(Stroke(x, 0, x, 0) for x in range(count))
    return PaintPlan(
        count,
        1,
        (ColorGroup((220, 40, 20), strokes, count),),
    )


# Shared CI runners are far slower than a dev machine, and these waits are
# wall-clock deadlines rather than assertions about behaviour. Scaling them
# keeps the same checks while tolerating a contended runner.
_TIMEOUT_SCALE = float(os.environ.get("RUST_PAINTER_TEST_TIMEOUT_SCALE", "1"))


def _t(seconds: float) -> float:
    return seconds * _TIMEOUT_SCALE


def _wait_until(predicate, timeout: float = 2.0) -> bool:
    timeout = _t(timeout)
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.002)
    return bool(predicate())


def test_painter_completes_and_releases_mouse() -> None:
    input_controller = MockInputController()
    completed = threading.Event()
    states: list[PainterState] = []
    painter = Painter(
        input_controller,
        on_state_change=lambda state, _reason: states.append(state),
        on_complete=lambda _progress: completed.set(),
    )

    assert painter.start(_dot_plan(3), _profile(), _settings())
    assert painter.wait(_t(2.0))

    assert completed.is_set()
    assert painter.state is PainterState.COMPLETED
    assert painter.progress.completed_strokes == 3
    assert painter.progress.percent == 100.0
    assert not input_controller.held_buttons
    assert PainterState.RUNNING in states
    assert states[-1] is PainterState.COMPLETED


def _sized_profile(name: str, *, sign_rows: int = 320) -> CalibrationProfile:
    """A calibrated profile whose brush model describes a known sign.

    The canvas is 320px tall, so ``sign_rows=320`` makes one Size unit exactly
    one screen pixel and keeps the arithmetic in these tests readable.
    """

    profile = CalibrationProfile.new(
        name,
        canvas=ScreenRect(100, 100, 640, 320),
        color_box=ScreenRect(600, 500, 100, 100),
        hue_bar=ScreenRect(720, 500, 12, 100),
        brush_size_box=ScreenRect(800, 100, 60, 24),
    )
    profile.metadata["brush_size_model"] = fit_brush_size_model(
        [(size, size / sign_rows) for size in (60, 30, 12)]
    ).to_dict()
    return profile


def _typed_values(controller: MockInputController) -> list[str]:
    """Every value the painter committed into the Size field, in order."""

    values: list[str] = []
    text = ""
    for event in controller.events:
        if event.kind != "key_down":
            continue
        value = event.value
        if value == 0xBE:  # VK_OEM_PERIOD
            text += "."
        elif isinstance(value, str) and len(value) == 1 and value.isdigit():
            text += value
        elif value == "ENTER":
            if text:
                values.append(text)
            text = ""
        else:
            text = ""
    return values


def test_automatic_brush_size_types_the_number_for_one_logical_cell() -> None:
    controller = MockInputController()
    profile = _sized_profile("Auto brush")
    plan = PaintPlan(
        64,
        32,
        (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),),
    )
    painter = Painter(controller, screen_capture=_panel_capture)

    assert painter.start(plan, profile, _settings(apply_brush_size=True))
    assert painter.wait(_t(2.0))

    # A 640x320 canvas under a 64x32 grid has 10px cells; a one-cell brush
    # targets the full pitch plus half a texel so the sign's snapping can never
    # open a bare stripe. One Size unit is one screen pixel here, so 10.5.
    assert painter.state is PainterState.COMPLETED
    assert _typed_values(controller) == ["10.5"]
    assert not controller.held_buttons


def test_brush_size_is_committed_by_clearing_the_field_and_pressing_enter() -> None:
    controller = MockInputController()
    profile = _sized_profile("Commit brush")
    plan = PaintPlan(
        64,
        32,
        (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),),
    )
    painter = Painter(controller, screen_capture=_panel_capture)

    assert painter.start(plan, profile, _settings(apply_brush_size=True))
    assert painter.wait(_t(2.0))

    keys = [
        str(event.value)
        for event in controller.events
        if event.kind == "key_down"
    ]
    # Old contents go from both sides of the caret, wherever the click put it.
    # The period arrives as the raw OEM virtual-key code, stringified by the
    # event recorder.
    assert keys == ["BACKSPACE"] * 6 + ["DELETE"] * 6 + ["1", "0", str(0xBE), "5", "ENTER"]
    box = profile.brush_size_box
    assert any(
        event.kind == "move"
        and event.x is not None
        and event.y is not None
        and box.contains(event.x, event.y)
        for event in controller.events
    )


def test_brush_sizing_never_disturbs_the_selected_color() -> None:
    """Typing a number cannot repaint the picker, so the group color stands."""

    controller = MockInputController()
    profile = _sized_profile("Color safe")
    plan = PaintPlan(
        64,
        32,
        (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),),
    )
    painter = Painter(controller, screen_capture=_panel_capture)

    assert painter.start(plan, profile, _settings(apply_brush_size=True))
    assert painter.wait(_t(2.0))

    expected = map_rgb_to_picker(
        (40, 80, 160),
        profile.hue_bar,
        profile.color_box,
        hue_direction="bottom_to_top",
        saturation_direction="left_low",
        value_direction="top_bright",
    ).hue
    hue_moves = [
        (event.x, event.y)
        for event in controller.events
        if event.kind == "move"
        and event.x is not None
        and event.y is not None
        and profile.hue_bar.contains(event.x, event.y)
    ]
    # Exactly one hue selection: the old search picked a temporary magenta
    # first and had to re-select the real color afterwards.
    assert hue_moves == [tuple(round(value) for value in expected)]


def test_optimized_plan_switches_brush_size() -> None:
    controller = MockInputController()
    profile = _sized_profile("Size aware")
    plan = PaintPlan(
        64,
        32,
        (
            ColorGroup(
                (30, 60, 200),
                (Stroke(10, 10, 30, 10),),
                105,
                brush_diameter=5,
            ),
            ColorGroup((30, 60, 200), (Stroke(0, 0, 5, 0),), 6),
            ColorGroup(
                (200, 40, 20),
                (Stroke(8, 20, 20, 20),),
                65,
                brush_diameter=5,
            ),
        ),
    )
    painter = Painter(controller, screen_capture=_panel_capture)

    assert painter.start(plan, profile, _settings(apply_brush_size=True))
    assert painter.wait(_t(3.0))
    assert painter.state is PainterState.COMPLETED
    assert not controller.held_buttons

    # 5 cells of a 10px pitch is 50 units; one cell is the 10px pitch plus a
    # half-texel snap margin.  Every switch is one computed number, so
    # returning to 5 costs no search at all.
    assert _typed_values(controller) == ["50", "10.5", "50"]


def test_multi_cell_brush_beyond_the_size_field_is_refused_before_painting() -> None:
    controller = MockInputController()
    # A sign only 40 rows tall over a 320px canvas: 8 screen px per Size unit,
    # so the field's maximum of 100 still cannot reach a 5-cell, 50px band...
    # but a coarse sign reaches it easily, so squeeze the range instead.
    profile = _sized_profile("Weak field", sign_rows=6400)
    plan = PaintPlan(
        64,
        32,
        (
            ColorGroup(
                (30, 60, 200),
                (Stroke(10, 10, 30, 10),),
                105,
                brush_diameter=5,
            ),
        ),
    )
    painter = Painter(controller, screen_capture=_panel_capture)

    # Refusing at configure time means the sign is still blank when the user
    # finds out, instead of half painted with stripes.
    with pytest.raises(ValueError, match="reaches only"):
        painter.configure(plan, profile, _settings(apply_brush_size=True))


def test_resolution_finer_than_the_sign_warns_but_still_paints() -> None:
    """A calibrated sign must stay paintable; softened detail is the user's call.

    40 sign rows under a 320px canvas: Rust's smallest brush is 8px while a
    cell of this plan is only 2px, so strokes will overlap - the job warns and
    paints rather than refusing the canvas it was given.
    """

    controller = MockInputController()
    profile = _sized_profile("Coarse sign", sign_rows=40)
    plan = PaintPlan(
        320,
        160,
        (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),),
    )
    painter = Painter(controller, screen_capture=_panel_capture)

    assert painter.start(plan, profile, _settings(apply_brush_size=True))
    assert painter.wait(_t(2.0))

    assert painter.state is PainterState.COMPLETED
    # The brush floor is 1.00, so the impossible sub-cell request lands there.
    assert _typed_values(controller) == ["1"]


def test_multi_size_plan_requires_brush_sizing_calibration() -> None:
    plan = PaintPlan(
        8,
        8,
        (ColorGroup((10, 20, 30), (Stroke(0, 0, 0, 0),), 1, brush_diameter=3),),
    )
    controller = MockInputController()
    # The guard protects real input; mocks stand in for the system backend.
    controller.emits_real_input = True  # type: ignore[misc]
    painter = Painter(controller)
    with pytest.raises(ValueError, match="brush sizes"):
        painter.configure(plan, _profile(), _settings())


def test_profile_color_correction_changes_picker_command() -> None:
    controller = MockInputController()
    profile = _profile()
    model = ColorCorrectionModel(
        forward_matrix=(
            (0.5, 0.0, 0.0, 0.0),
            (0.0, 0.5, 0.0, 0.0),
            (0.0, 0.0, 0.5, 0.0),
        ),
        fit_rmse=0.0,
        sample_count=32,
        captured_at="2026-08-17T00:00:00+00:00",
    )
    profile.metadata["color_correction"] = model.to_dict()
    plan = PaintPlan(
        1,
        1,
        (ColorGroup((128, 128, 128), (Stroke(0, 0, 0, 0),), 1),),
    )
    painter = Painter(controller)

    assert painter.start(plan, profile, _settings())
    assert painter.wait(_t(2.0))

    color_box_moves = [
        event
        for event in controller.events
        if event.kind == "move"
        and event.x is not None
        and event.y is not None
        and 600 <= event.x < 700
        and 100 <= event.y < 200
    ]
    assert color_box_moves
    # The measured material halves brightness, so desired mid-gray requires a
    # full-value white picker command at the top of the S/V box - pulled 2%
    # inside the widget, because Rust ignores clicks on its outermost pixels.
    assert color_box_moves[0].y == 102


def test_pause_holds_progress_then_resume_completes() -> None:
    input_controller = MockInputController()
    painter = Painter(input_controller)
    painter.start(
        _dot_plan(20),
        _profile(),
        _settings(
            mouse_down_duration_seconds=0.004,
            delay_between_strokes_seconds=0.02,
        ),
    )
    assert _wait_until(lambda: painter.progress.completed_strokes >= 2)

    assert painter.pause()
    assert painter.state is PainterState.PAUSED
    assert not input_controller.held_buttons
    # Allow an already completed stroke's progress notification to settle, then
    # verify no queued actions continue while the worker waits.
    time.sleep(0.03)
    paused_at = painter.progress.completed_strokes
    event_count = len(input_controller.events)
    time.sleep(0.06)
    assert painter.progress.completed_strokes == paused_at
    assert len(input_controller.events) == event_count

    assert painter.resume()
    assert painter.wait(_t(3.0))
    assert painter.state is PainterState.COMPLETED
    assert painter.progress.completed_strokes == 20
    assert not input_controller.held_buttons


def test_abort_stops_queued_strokes_and_releases_mouse() -> None:
    input_controller = MockInputController()
    painter = Painter(input_controller)
    painter.start(
        _dot_plan(50),
        _profile(),
        _settings(
            mouse_down_duration_seconds=0.004,
            delay_between_strokes_seconds=0.015,
        ),
    )
    assert _wait_until(lambda: painter.progress.completed_strokes >= 2)

    assert painter.abort("test emergency stop")
    assert painter.wait(_t(2.0))
    settled_event_count = len(input_controller.events)
    time.sleep(0.04)

    assert painter.state is PainterState.ABORTED
    assert painter.progress.completed_strokes < 50
    assert len(input_controller.events) == settled_event_count
    assert not input_controller.held_buttons


def test_pause_during_drag_restarts_unfinished_stroke() -> None:
    input_controller = MockInputController()
    plan = PaintPlan(
        100,
        1,
        (ColorGroup((0, 150, 255), (Stroke(0, 0, 99, 0),), 100),),
    )
    painter = Painter(input_controller)
    painter.start(
        plan,
        _profile(canvas_width=1000),
        _settings(
            stroke_speed_pixels_per_second=4_000.0,
            stroke_interpolation_step_pixels=4.0,
        ),
    )
    # Two color-picker clicks precede the long canvas drag.
    assert _wait_until(
        lambda: sum(event.kind == "mouse_down" for event in input_controller.events) >= 3
    )
    assert painter.pause()
    assert not input_controller.held_buttons
    assert painter.resume()
    assert painter.wait(_t(3.0))

    mouse_down_count = sum(
        event.kind == "mouse_down" for event in input_controller.events
    )
    assert painter.state is PainterState.COMPLETED
    assert mouse_down_count >= 6  # color is reselected, then the stroke restarts
    assert not input_controller.held_buttons


def test_focus_guard_pauses_before_any_input_and_rechecks_on_resume() -> None:
    input_controller = MockInputController()
    input_controller.emits_real_input = True
    foreground = {"matches": False}
    painter = Painter(
        input_controller,
        foreground_checker=lambda _requirement: foreground["matches"],
    )
    painter.start(
        _dot_plan(1),
        _profile(),
        _settings(
            require_foreground=True,
            focus_check_interval_seconds=0.001,
        ),
    )
    assert _wait_until(lambda: painter.state is PainterState.PAUSED)
    assert input_controller.events == []

    # An immediate resume while the wrong window remains active must pause
    # again before any picker or canvas input is emitted.
    assert painter.resume()
    assert _wait_until(lambda: painter.state is PainterState.PAUSED)
    assert input_controller.events == []

    foreground["matches"] = True
    assert painter.resume()
    assert painter.wait(_t(2.0))
    assert painter.state is PainterState.COMPLETED
    assert input_controller.events


def test_corner_emergency_stop_aborts_before_next_click() -> None:
    class CornerInput(MockInputController):
        emits_real_input = True

        def get_cursor_position(self) -> tuple[int, int]:
            return (0, 0)

    input_controller = CornerInput()
    painter = Painter(
        input_controller,
        virtual_screen_provider=lambda: VirtualScreen(0, 0, 1200, 900),
    )
    painter.start(
        _dot_plan(3),
        _profile(),
        _settings(
            corner_abort_enabled=True,
            corner_abort_margin_pixels=2,
            corner_abort_minimum_distance_pixels=50,
        ),
    )
    assert painter.wait(_t(2.0))
    assert painter.state is PainterState.ABORTED
    assert not input_controller.held_buttons


def test_dry_run_skips_real_time_delays_and_focus_guard() -> None:
    input_controller = DryRunInputController()
    painter = Painter(
        input_controller,
        foreground_checker=lambda _requirement: (_ for _ in ()).throw(
            AssertionError("dry run should not inspect another application")
        ),
    )
    started = time.monotonic()
    painter.start(
        _dot_plan(3),
        _profile(),
        _settings(
            countdown_seconds=2.0,
            mouse_down_duration_seconds=2.0,
            delay_after_hue_seconds=2.0,
            delay_after_saturation_value_seconds=2.0,
            delay_between_strokes_seconds=2.0,
            delay_between_colors_seconds=2.0,
            require_foreground=True,
        ),
    )
    assert painter.wait(_t(1.0))
    assert time.monotonic() - started < 0.5
    assert painter.state is PainterState.COMPLETED
    assert input_controller.events == []


def test_canonical_foreground_setting_wins_over_legacy_alias() -> None:
    settings = PainterSettings.from_mapping(
        {
            "painting": {},
            "safety": {
                "require_rust_foreground": True,
                "require_foreground": False,
            },
        }
    )
    assert settings.require_foreground is True


def test_foreground_guard_requires_a_target_identity() -> None:
    with pytest.raises(ValueError, match="expected window title or process"):
        PainterSettings(
            require_foreground=True,
            expected_window_title_contains="",
            expected_process_name=None,
        )


def test_pause_at_countdown_boundary_remains_resumable() -> None:
    input_controller = MockInputController()
    painter = Painter(input_controller)
    boundary = threading.Event()
    continue_into_running = threading.Event()
    original_enter_running = painter._enter_running_after_countdown

    def gated_enter_running() -> None:
        boundary.set()
        assert continue_into_running.wait(_t(1.0))
        original_enter_running()

    painter._enter_running_after_countdown = gated_enter_running  # type: ignore[method-assign]
    assert painter.start(
        _dot_plan(1),
        _profile(),
        _settings(countdown_seconds=0.01),
    )
    assert boundary.wait(_t(1.0))
    assert painter.pause("boundary test")
    continue_into_running.set()
    time.sleep(0.03)

    assert painter.state is PainterState.PAUSED
    assert painter.progress.state is PainterState.PAUSED
    assert input_controller.events == []
    assert painter.resume()
    assert painter.wait(_t(2.0))
    assert painter.state is PainterState.COMPLETED


def test_abort_of_ready_job_prevents_late_start() -> None:
    input_controller = MockInputController()
    painter = Painter(input_controller)
    painter.configure(_dot_plan(1), _profile(), _settings())

    assert painter.state is PainterState.READY
    assert painter.abort("cancel pending start")
    assert painter.start() is False
    assert painter.state is PainterState.ABORTED
    assert input_controller.events == []


def test_brush_application_requires_the_size_value_box() -> None:
    painter = Painter(MockInputController())
    with pytest.raises(ValueError, match="Size value box"):
        painter.configure(
            _dot_plan(1),
            _profile(),
            _settings(apply_brush_size=True),
        )


def test_brush_application_requires_a_way_to_clear_the_sign() -> None:
    """The job measures the brush on the sign, so the probes must be erasable."""

    profile = _profile()
    profile.brush_size_box = ScreenRect(800, 100, 60, 24)
    controller = MockInputController()
    # The guard protects real input; mocks stand in for the system backend.
    controller.emits_real_input = True  # type: ignore[misc]
    painter = Painter(controller)
    with pytest.raises(ValueError, match="clear control"):
        painter.configure(
            _dot_plan(1),
            profile,
            _settings(apply_brush_size=True),
        )

    profile.clear_button = ScreenRect(880, 100, 24, 24)
    painter.configure(_dot_plan(1), profile, _settings(apply_brush_size=True))


def test_foreground_failure_reason_calls_out_an_impossible_windows_name() -> None:
    """The generic message left macOS users with no idea what went wrong."""

    from app.painter import _foreground_failure_reason

    windows_name = _settings(
        require_foreground=True, expected_process_name="RustClient.exe"
    )
    reason = _foreground_failure_reason(windows_name)
    if os.name == "nt":
        assert reason == "foreground window lost"
    else:
        assert "RustClient.exe" in reason
        assert "Settings" in reason

    posix_name = _settings(require_foreground=True, expected_process_name="RustClient")
    assert _foreground_failure_reason(posix_name) == "foreground window lost"


class _HandInput(MockInputController):
    """A real-input mock whose cursor can lag behind, or be moved by a hand."""

    emits_real_input = True

    def __init__(self) -> None:
        super().__init__()
        self.hand_offset = (0, 0)
        self.report_lag = 0
        self.absolute_cursor: tuple[int, int] | None = None
        self._history: list[tuple[int, int]] = []

    def move_mouse(self, x: float, y: float) -> None:
        super().move_mouse(x, y)
        self._history.append((int(round(x)), int(round(y))))

    def get_cursor_position(self) -> tuple[int, int]:
        if self.absolute_cursor is not None:
            return self.absolute_cursor
        if not self._history:
            return super().get_cursor_position()
        index = max(0, len(self._history) - 1 - self.report_lag)
        point = self._history[index]
        return (point[0] + self.hand_offset[0], point[1] + self.hand_offset[1])


def test_mouse_movement_pauses_and_resume_repeats_the_interrupted_stroke() -> None:
    input_controller = _HandInput()
    input_controller.hand_offset = (60, 40)
    painter = Painter(input_controller, screen_capture=_panel_capture)
    painter.start(
        _dot_plan(4),
        _profile(),
        _settings(
            mouse_down_duration_seconds=0.01,
            delay_between_strokes_seconds=0.01,
            pause_on_mouse_move=True,
        ),
    )

    assert _wait_until(lambda: painter.state is PainterState.PAUSED)
    assert painter.state_reason.startswith("mouse moved")
    # A pause mid-click must not leave the button down on the canvas.
    assert not input_controller.held_buttons
    before = len(input_controller.events)
    time.sleep(_t(0.05))
    assert len(input_controller.events) == before

    # Letting go of the mouse and resuming continues the same job.
    input_controller.hand_offset = (0, 0)
    assert painter.resume()
    assert painter.wait(_t(3.0))
    assert painter.state is PainterState.COMPLETED
    assert painter.progress.completed_strokes == 4
    assert not input_controller.held_buttons


def test_queued_input_lag_is_not_mistaken_for_mouse_movement() -> None:
    # SendInput is asynchronous, so a sample can still report a point commanded
    # several events ago.  That must never read as a hand on the mouse.
    input_controller = _HandInput()
    input_controller.report_lag = 4
    painter = Painter(input_controller, screen_capture=_panel_capture)
    painter.start(
        _dot_plan(12),
        _profile(),
        # Deliberate per-stroke time so the cursor is sampled many times over a
        # history that is already full, not just once at the very start.
        _settings(
            mouse_down_duration_seconds=0.005,
            delay_between_strokes_seconds=0.005,
            pause_on_mouse_move=True,
        ),
    )

    assert painter.wait(_t(3.0))
    assert painter.state is PainterState.COMPLETED
    assert painter.progress.completed_strokes == 12


def test_corner_emergency_stop_still_aborts_while_paused() -> None:
    input_controller = _HandInput()
    painter = Painter(
        input_controller,
        virtual_screen_provider=lambda: VirtualScreen(0, 0, 1200, 900),
        screen_capture=_panel_capture,
    )
    painter.start(
        _dot_plan(400),
        _profile(),
        _settings(
            delay_between_strokes_seconds=0.02,
            corner_abort_enabled=True,
            corner_abort_margin_pixels=2,
            corner_abort_minimum_distance_pixels=50,
            pause_on_mouse_move=False,
        ),
    )
    assert _wait_until(lambda: painter.state is PainterState.RUNNING)
    assert painter.pause("user")
    assert _wait_until(lambda: painter.state is PainterState.PAUSED)

    # The corner gesture is what a user reaches for once painting is already
    # interrupted, so a paused job must still honour it.
    input_controller.absolute_cursor = (0, 0)
    assert _wait_until(lambda: painter.state is PainterState.ABORTED)
    assert painter.wait(_t(2.0))
    assert not input_controller.held_buttons


def _sign_simulator(controller: MockInputController, canvas: ScreenRect, sign_rows: int):
    """A fake sign that paints bands as wide as the Size number typed into it.

    Replaying the recorded events on every capture keeps the simulator honest:
    it only ever knows what the painter actually did, so a probe that forgot to
    commit its number shows up as a band of the previous width.
    """

    palette = ((255, 0, 255), (0, 255, 0), (255, 200, 0), (0, 200, 255))
    scale = canvas.height / sign_rows

    def capture(rect) -> Image.Image:
        image = Image.new("RGB", (rect.width, rect.height), (96, 96, 96))
        if (rect.left, rect.top) != (canvas.left, canvas.top):
            # The picker measurement captures other regions; a flat panel there
            # leaves the calibration exactly as written.
            return Image.new("RGB", (rect.width, rect.height), (21, 21, 12))
        draw = ImageDraw.Draw(image)
        size = 0
        digits = ""
        position = (0, 0)
        painted = 0
        for event in controller.events:
            if event.kind == "move" and event.x is not None and event.y is not None:
                position = (event.x, event.y)
            elif event.kind == "key_down":
                value = event.value
                if value == 0xBE:  # VK_OEM_PERIOD
                    digits += "."
                elif isinstance(value, str) and len(value) == 1 and value.isdigit():
                    digits += value
                elif value == "ENTER":
                    size = float(digits) if digits else size
                    digits = ""
                else:
                    digits = ""
            elif event.kind == "mouse_down" and canvas.contains(*position):
                height = max(1, round(size * scale))
                top = round(rect.height / 2 - height / 2)
                draw.rectangle(
                    (10, top, rect.width - 11, top + height - 1),
                    fill=palette[painted % len(palette)],
                )
                painted += 1
        return image

    return capture


def _clearable_sign(
    controller: MockInputController,
    canvas: ScreenRect,
    clear_button: ScreenRect,
    sign_rows: int,
):
    """A fake sign that paints Size-wide bands and wipes on the clear click.

    Like :func:`_sign_simulator` it replays the recorded events on every
    capture, so it only ever knows what the painter really did - including
    whether it actually clicked the control that clears the sign.
    """

    palette = ((255, 0, 255), (0, 255, 0), (255, 200, 0), (0, 200, 255))
    scale = canvas.height / sign_rows

    def capture(rect) -> Image.Image:
        if (rect.left, rect.top) != (canvas.left, canvas.top):
            return Image.new("RGB", (rect.width, rect.height), (21, 21, 12))
        image = Image.new("RGB", (rect.width, rect.height), (96, 96, 96))
        draw = ImageDraw.Draw(image)
        size = 0.0
        digits = ""
        position = (0, 0)
        painted = 0
        for event in controller.events:
            if event.kind == "move" and event.x is not None and event.y is not None:
                position = (event.x, event.y)
            elif event.kind == "key_down":
                value = event.value
                if value == 0xBE:  # VK_OEM_PERIOD
                    digits += "."
                elif isinstance(value, str) and len(value) == 1 and value.isdigit():
                    digits += value
                elif value == "ENTER":
                    size = float(digits) if digits else size
                    digits = ""
                else:
                    digits = ""
            elif event.kind == "mouse_down":
                if clear_button.contains(*position):
                    image = Image.new("RGB", (rect.width, rect.height), (96, 96, 96))
                    draw = ImageDraw.Draw(image)
                    painted = 0
                elif canvas.contains(*position):
                    height = max(1, round(size * scale))
                    top = round(rect.height / 2 - height / 2)
                    draw.rectangle(
                        (10, top, rect.width - 11, top + height - 1),
                        fill=palette[painted % len(palette)],
                    )
                    painted += 1
        return image

    return capture


def _impatient(painter: Painter) -> Painter:
    """Strip the frame-rate waits a real Rust client needs but a fake does not.

    The waits exist because Rust redraws at about 15 FPS; a simulated sign
    repaints instantly, so leaving them in would spend ten seconds proving
    nothing about the ordering these tests are checking.
    """

    painter._CAPTURE_SETTLE_SECONDS = 0.0  # type: ignore[misc]
    painter._CLEAR_SETTLE_SECONDS = 0.0  # type: ignore[misc]
    painter._KEY_HOLD_SECONDS = 0.0  # type: ignore[misc]
    painter._KEY_GAP_SECONDS = 0.0  # type: ignore[misc]
    return painter


def _calibrating_profile(name: str) -> CalibrationProfile:
    return CalibrationProfile.new(
        name,
        canvas=ScreenRect(100, 100, 640, 320),
        color_box=ScreenRect(600, 500, 100, 100),
        hue_bar=ScreenRect(720, 500, 12, 100),
        brush_size_box=ScreenRect(800, 100, 60, 24),
        clear_button=ScreenRect(880, 100, 24, 24),
    )


def _position_at(controller: MockInputController, index: int) -> tuple[int, int]:
    """Where the cursor had been commanded to by event ``index``."""

    position = (0, 0)
    for event in controller.events[: index + 1]:
        if event.kind == "move" and event.x is not None and event.y is not None:
            position = (event.x, event.y)
    return position


def _one_cell_plan() -> PaintPlan:
    return PaintPlan(64, 32, (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),))


def test_a_paint_job_measures_the_brush_then_wipes_it_before_painting() -> None:
    """The whole point of dropping the manual step: it happens on every run.

    A stored measurement describes a sign the user may since have re-framed or
    walked away from, so the run measures the sign in front of it - and the
    probes it paints have to be gone before the artwork starts.
    """

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _calibrating_profile("Self calibrating")
    assert profile.canvas is not None and profile.clear_button is not None
    painter = _impatient(
        Painter(
            controller,
            screen_capture=_clearable_sign(
                controller, profile.canvas, profile.clear_button, sign_rows=320
            ),
        )
    )

    assert painter.start(
        _one_cell_plan(), profile, _settings(apply_brush_size=True, verify_passes=0)
    )
    assert painter.wait(_t(10.0))
    assert painter.state is PainterState.COMPLETED

    # The job fitted its own model rather than leaning on the profile's.
    model = painter.measured_brush_size_model
    assert model is not None
    assert model.sign_pixel_rows == pytest.approx(320.0, rel=0.05)

    typed = _typed_values(controller)
    # Probe sizes first, then the size the artwork actually wants: a 640x320
    # canvas under a 64x32 grid is a 10px cell, plus half a texel of overlap.
    assert len(typed) > 1
    assert typed[-1] == "10.5"

    # The clear click has to land after the last probe and before the artwork.
    clicks = [
        index
        for index, event in enumerate(controller.events)
        if event.kind == "mouse_down"
    ]
    cleared_at = next(
        index
        for index in clicks
        if profile.clear_button.contains(*_position_at(controller, index))
    )
    on_canvas = [
        index
        for index in clicks
        if profile.canvas.contains(*_position_at(controller, index))
    ]
    assert on_canvas, "the job never painted on the sign"
    assert on_canvas[-1] > cleared_at
    assert not controller.held_buttons


def test_a_clear_control_that_clears_nothing_warns_but_keeps_painting(
    caplog,
) -> None:
    """A clear click is trusted; a sign that looks unchanged only logs.

    Rust's redraw can lag past the capture, so a stopped job here failed real
    runs whose signs had genuinely cleared.  The suspicious capture is worth a
    log line for a truly misdragged trash box, never an abort.
    """

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _calibrating_profile("Deaf clear")
    assert profile.canvas is not None
    # The simulated sign wipes on a control nothing will ever click, so its
    # probe bands survive the painter's clear click.
    painter = _impatient(
        Painter(
            controller,
            screen_capture=_clearable_sign(
                controller,
                profile.canvas,
                ScreenRect(2000, 2000, 4, 4),
                sign_rows=320,
            ),
        )
    )
    errors: list[str] = []
    painter.set_callbacks(on_error=lambda exc: errors.append(str(exc)))

    with caplog.at_level(logging.WARNING, logger="rust_painter.painter"):
        assert painter.start(
            _one_cell_plan(), profile, _settings(apply_brush_size=True, verify_passes=0)
        )
        assert painter.wait(_t(10.0))

    assert painter.state is PainterState.COMPLETED
    assert not errors
    assert any(
        "unchanged after clicking Rust's clear control" in record.message
        for record in caplog.records
    )
    assert not controller.held_buttons


def test_a_pause_during_calibration_measures_again_instead_of_failing() -> None:
    """Pausing in the first seconds must not throw the whole job away.

    A pause hands the mouse back, so the probes on either side of it describe
    different signs; the run restarts the measurement rather than fitting a
    line through both halves.
    """

    from app.painter import _RetryAction

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _calibrating_profile("Interrupted")
    assert profile.canvas is not None and profile.clear_button is not None
    painter = _impatient(
        Painter(
            controller,
            screen_capture=_clearable_sign(
                controller, profile.canvas, profile.clear_button, sign_rows=320
            ),
        )
    )
    real_measure = painter._measure_brush_size_model
    attempts: list[int] = []

    def flaky(job):
        attempts.append(1)
        if len(attempts) == 1:
            raise _RetryAction
        return real_measure(job)

    painter._measure_brush_size_model = flaky  # type: ignore[method-assign]

    assert painter.start(
        _one_cell_plan(), profile, _settings(apply_brush_size=True, verify_passes=0)
    )
    assert painter.wait(_t(10.0))

    assert attempts == [1, 1]
    assert painter.state is PainterState.COMPLETED
    assert _typed_values(controller)[-1] == "10.5"


def test_calibration_that_is_never_left_alone_stops_the_job() -> None:
    from app.painter import _RetryAction

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _calibrating_profile("Always interrupted")
    painter = _impatient(Painter(controller, screen_capture=_panel_capture))

    def never(_job):
        raise _RetryAction

    painter._measure_brush_size_model = never  # type: ignore[method-assign]
    errors: list[str] = []
    painter.set_callbacks(on_error=lambda exc: errors.append(str(exc)))

    assert painter.start(
        _one_cell_plan(), profile, _settings(apply_brush_size=True, verify_passes=0)
    )
    assert painter.wait(_t(5.0))

    assert painter.state is PainterState.ERROR
    assert errors and "interrupted every time" in errors[0]
    assert not controller.held_buttons


def test_a_dry_run_never_paints_calibration_probes() -> None:
    """Nothing that does not emit input may put strokes on somebody's sign."""

    controller = MockInputController()  # emits_real_input is False
    profile = _calibrating_profile("Dry")
    profile.metadata["brush_size_model"] = fit_brush_size_model(
        [(size, size / 320.0) for size in (60, 30, 12)]
    ).to_dict()
    painter = Painter(controller, screen_capture=_panel_capture)

    assert painter.start(
        _one_cell_plan(), profile, _settings(apply_brush_size=True, verify_passes=0)
    )
    assert painter.wait(_t(2.0))

    assert painter.state is PainterState.COMPLETED
    assert painter.measured_brush_size_model is None
    # Only the artwork's own brush number was typed; nothing was probed.
    assert _typed_values(controller) == ["10.5"]


def test_brush_measurement_fits_the_sign_it_probes() -> None:
    controller = MockInputController()
    profile = CalibrationProfile.new(
        "Measure me",
        canvas=ScreenRect(100, 100, 640, 320),
        color_box=ScreenRect(600, 500, 100, 100),
        hue_bar=ScreenRect(720, 500, 12, 100),
        brush_size_box=ScreenRect(800, 100, 60, 24),
    )
    painter = Painter(
        controller,
        screen_capture=_sign_simulator(controller, profile.canvas, sign_rows=128),
    )

    painter.configure_brush_measurement(profile, _settings())
    assert painter.start()
    assert painter.wait(_t(5.0))

    assert painter.state is PainterState.COMPLETED
    model = painter.measured_brush_size_model
    assert model is not None
    # 128 rows over a 320px canvas: the fit has to recover the sign, not the
    # screen pixels it happened to be measured in.
    assert model.sign_pixel_rows == pytest.approx(128.0, rel=0.02)
    # With no resolution to aim at, the fallback ladder is what gets probed.
    assert _typed_values(controller) == ["32", "16", "8", "4"]
    assert not controller.held_buttons


def test_brush_measurement_probes_around_the_brush_the_plan_needs() -> None:
    """Reading a fitted line far below its own data is what broke sizing.

    A scout stroke turns the wanted cell into a rough Size number, and the real
    probes then straddle it, so the model is interpolated where it is used
    instead of extrapolated down from a brush ten times wider.
    """

    controller = MockInputController()
    profile = CalibrationProfile.new(
        "Bracketed",
        canvas=ScreenRect(100, 100, 640, 320),
        color_box=ScreenRect(600, 500, 100, 100),
        hue_bar=ScreenRect(720, 500, 12, 100),
        brush_size_box=ScreenRect(800, 100, 60, 24),
    )
    painter = Painter(
        controller,
        screen_capture=_sign_simulator(controller, profile.canvas, sign_rows=128),
    )

    # One cell wants five Size units on a 128-row sign.
    painter.configure_brush_measurement(profile, _settings(), cell_fraction=5 / 128)
    assert painter.start()
    assert painter.wait(_t(5.0))

    assert painter.state is PainterState.COMPLETED
    typed = _typed_values(controller)
    assert typed[0] == "24"  # the scout, which is never fitted
    # 8x through 0.5x of the wanted size, so multi-cell fill brushes are
    # measured rather than extrapolated to.
    assert typed[1:] == ["40", "20", "10", "5", "2.5"]
    model = painter.measured_brush_size_model
    assert model is not None
    # The simulator rounds bands to whole pixels, which feeds a small
    # quantization error into the fit; a twentieth of a size unit is well
    # under what one sign texel resolves.
    assert model.clamped_size_for_fraction(5 / 128) == pytest.approx(5.0, abs=0.15)


def test_brush_measurement_reports_a_size_field_that_ignores_typing() -> None:
    """Rust eating the digits as hotbar keys must not become a silent model."""

    controller = MockInputController()
    profile = CalibrationProfile.new(
        "Deaf field",
        canvas=ScreenRect(100, 100, 640, 320),
        color_box=ScreenRect(600, 500, 100, 100),
        hue_bar=ScreenRect(720, 500, 12, 100),
        brush_size_box=ScreenRect(800, 100, 60, 24),
    )
    canvas = profile.canvas

    def stuck_capture(rect) -> Image.Image:
        if (rect.left, rect.top) != (canvas.left, canvas.top):
            return Image.new("RGB", (rect.width, rect.height), (21, 21, 12))
        image = Image.new("RGB", (rect.width, rect.height), (96, 96, 96))
        # Every probe paints the same 20px band whatever was typed, and each in
        # its own color so the diff still sees a change.
        painted = sum(
            1
            for event in controller.events
            if event.kind == "mouse_down"
        )
        top = rect.height // 2 - 10
        ImageDraw.Draw(image).rectangle(
            (10, top, rect.width - 11, top + 19),
            fill=(255, (painted * 40) % 256, 255),
        )
        return image

    errors: list[str] = []
    painter = Painter(
        controller,
        screen_capture=stuck_capture,
        on_error=lambda exc: errors.append(str(exc)),
    )

    painter.configure_brush_measurement(profile, _settings())
    assert painter.start()
    assert painter.wait(_t(5.0))

    assert painter.state is PainterState.ERROR
    assert errors and "did not grow" in errors[0]
    assert painter.measured_brush_size_model is None


def test_one_cell_overlap_tapers_from_half_a_texel_to_a_quarter() -> None:
    """The stripe hedge shrinks as the cell grid approaches the texel grid.

    With cells two texels or coarser, cell boundaries land at arbitrary
    fractional texel positions and the in-game-proven half texel of overlap is
    what closes the worst case.  At native resolution the grids line up by
    construction, so only calibration slop remains: a quarter texel still
    bridges a snapped-away row while bleeding half as far into the
    neighbouring detail the native plan exists to keep.
    """

    # One Size unit is one screen pixel: a 320-row sign on a 320px canvas.
    model = fit_brush_size_model([(size, size / 320.0) for size in (60, 30, 12)])
    target = PaintingTarget(
        canvas=ScreenRect(100, 100, 640, 320),
        color_box=ScreenRect(600, 500, 100, 100),
        hue_bar=ScreenRect(720, 500, 12, 100),
    )

    def texels(plan_width: int, plan_height: int) -> float:
        plan = PaintPlan(plan_width, plan_height, ())
        return Painter._brush_target_fraction(target, plan, 1, 1.0, model) * 320.0

    # 10px cells are 10 texels: far above the taper, the full half texel.
    assert texels(64, 32) == pytest.approx(10.5)
    # 2px cells are exactly 2 texels: the taper's proven upper anchor.
    assert texels(320, 160) == pytest.approx(2.5)
    # 1.25 texels per cell sits inside the taper.
    assert texels(512, 256) == pytest.approx(1.5625)
    # Native resolution: one cell per texel, and only a quarter texel of hedge.
    assert texels(640, 320) == pytest.approx(1.25)


def test_native_resolution_types_a_size_barely_over_one() -> None:
    """At one cell per texel the typed Size number stays next to Rust's minimum.

    This is the whole quality budget of a Max-resolution plan: the earlier
    fixed half-texel hedge would have typed 1.5 and smeared every stroke into
    both neighbouring texel rows.
    """

    model = fit_brush_size_model([(size, size / 320.0) for size in (60, 30, 12)])
    target = PaintingTarget(
        canvas=ScreenRect(100, 100, 640, 320),
        color_box=ScreenRect(600, 500, 100, 100),
        hue_bar=ScreenRect(720, 500, 12, 100),
    )
    plan = PaintPlan(640, 320, ())

    fraction = Painter._brush_target_fraction(target, plan, 1, 1.0, model)

    assert model.clamped_size_for_fraction(fraction) == pytest.approx(1.25)
