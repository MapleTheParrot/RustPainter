from __future__ import annotations

import threading
import time

import pytest
from PIL import Image, ImageDraw

from app.color_calibration import ColorCorrectionModel
from app.color_mapping import map_rgb_to_picker
from app.input_controller import DryRunInputController, MockInputController
from app.models import ColorGroup, PaintPlan, ScreenRect, Stroke
from app.painter import Painter, PainterSettings, PainterState
from app.screen import VirtualScreen
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


def _dot_plan(count: int) -> PaintPlan:
    strokes = tuple(Stroke(x, 0, x, 0) for x in range(count))
    return PaintPlan(
        count,
        1,
        (ColorGroup((220, 40, 20), strokes, count),),
    )


def _wait_until(predicate, timeout: float = 2.0) -> bool:
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
    assert painter.wait(2.0)

    assert completed.is_set()
    assert painter.state is PainterState.COMPLETED
    assert painter.progress.completed_strokes == 3
    assert painter.progress.percent == 100.0
    assert not input_controller.held_buttons
    assert PainterState.RUNNING in states
    assert states[-1] is PainterState.COMPLETED


def test_automatic_brush_size_matches_preview_to_logical_cell() -> None:
    controller = MockInputController()
    slider = ScreenRect(800, 100, 101, 12)
    profile = CalibrationProfile.new(
        "Auto brush",
        canvas=ScreenRect(100, 100, 640, 320),
        color_box=ScreenRect(600, 500, 100, 100),
        hue_bar=ScreenRect(720, 500, 12, 100),
        brush_slider=slider,
        brush_preview=ScreenRect(800, 150, 100, 100),
    )
    # A profile correction that swaps red and green would normally turn the
    # magenta calibration request into cyan. Calibration must bypass it.
    profile.metadata["color_correction"] = ColorCorrectionModel(
        forward_matrix=(
            (0.0, 1.0, 0.0, 0.0),
            (1.0, 0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0, 0.0),
        ),
        fit_rmse=0.0,
        sample_count=32,
        captured_at="2026-08-17T00:00:00+00:00",
    ).to_dict()
    captured_diameters: list[int] = []

    def capture_preview(_rect) -> Image.Image:
        x, _y = controller.get_cursor_position()
        fraction = min(1.0, max(0.0, (x - slider.left) / (slider.width - 1)))
        diameter = round(2 + fraction * 30)
        captured_diameters.append(diameter)
        image = Image.new("RGB", (100, 100), (72, 72, 72))
        left = (100 - diameter) // 2
        top = (100 - diameter) // 2
        ImageDraw.Draw(image).rectangle(
            (left, top, left + diameter - 1, top + diameter - 1),
            fill=(255, 0, 255),
        )
        return image

    plan = PaintPlan(
        64,
        32,
        (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),),
    )
    painter = Painter(controller, screen_capture=capture_preview)

    assert painter.start(plan, profile, _settings(apply_brush_size=True))
    assert painter.wait(2.0)

    # A 640x320 canvas with a 64x32 grid has 10px cells. The automatic
    # target deliberately uses 90% coverage, so the selected preview is ~9px.
    assert painter.state is PainterState.COMPLETED
    assert captured_diameters
    assert min(abs(diameter - 9) for diameter in captured_diameters) <= 1
    assert not controller.held_buttons
    expected_hue = map_rgb_to_picker(
        (255, 0, 255),
        profile.hue_bar,
        profile.color_box,
        hue_direction="bottom_to_top",
        saturation_direction="left_low",
        value_direction="top_bright",
    ).hue
    first_hue_move = next(
        event
        for event in controller.events
        if event.kind == "move"
        and event.x is not None
        and event.y is not None
        and profile.hue_bar.contains(event.x, event.y)
    )
    assert (first_hue_move.x, first_hue_move.y) == tuple(round(v) for v in expected_hue)


def test_automatic_brush_size_retries_tiny_preview_with_center_crop() -> None:
    controller = MockInputController()
    slider = ScreenRect(800, 100, 101, 12)
    original_preview = ScreenRect(800, 150, 120, 120)
    profile = CalibrationProfile.new(
        "Tiny auto brush",
        canvas=ScreenRect(100, 100, 640, 320),
        color_box=ScreenRect(600, 500, 100, 100),
        hue_bar=ScreenRect(720, 500, 12, 100),
        brush_slider=slider,
        brush_preview=original_preview,
    )
    capture_sizes: list[tuple[int, int]] = []

    def capture_preview(rect) -> Image.Image:
        capture_sizes.append((rect.width, rect.height))
        image = Image.new("RGB", (rect.width, rect.height), (72, 72, 72))
        # A two-pixel brush is below the full 120x120 capture's area threshold,
        # but becomes viable in the smaller temporary center capture.
        left = (rect.width - 2) // 2
        top = (rect.height - 2) // 2
        ImageDraw.Draw(image).rectangle(
            (left, top, left + 1, top + 1), fill=(255, 0, 255)
        )
        return image

    plan = PaintPlan(
        320,
        160,
        (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),),
    )
    painter = Painter(controller, screen_capture=capture_preview)

    assert painter.start(plan, profile, _settings(apply_brush_size=True))
    assert painter.wait(2.0)

    assert painter.state is PainterState.COMPLETED
    assert (120, 120) in capture_sizes
    assert (60, 60) in capture_sizes
    assert profile.brush_preview == original_preview


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
    assert painter.wait(2.0)

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
    # full-value white picker command at the top of the S/V box.
    assert color_box_moves[0].y == 100


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
    assert painter.wait(3.0)
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
    assert painter.wait(2.0)
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
    assert painter.wait(3.0)

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
    assert painter.wait(2.0)
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
    assert painter.wait(2.0)
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
    assert painter.wait(1.0)
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
        assert continue_into_running.wait(1.0)
        original_enter_running()

    painter._enter_running_after_countdown = gated_enter_running  # type: ignore[method-assign]
    assert painter.start(
        _dot_plan(1),
        _profile(),
        _settings(countdown_seconds=0.01),
    )
    assert boundary.wait(1.0)
    assert painter.pause("boundary test")
    continue_into_running.set()
    time.sleep(0.03)

    assert painter.state is PainterState.PAUSED
    assert painter.progress.state is PainterState.PAUSED
    assert input_controller.events == []
    assert painter.resume()
    assert painter.wait(2.0)
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


def test_brush_application_requires_slider_calibration() -> None:
    painter = Painter(MockInputController())
    with pytest.raises(ValueError, match="no brush slider"):
        painter.configure(
            _dot_plan(1),
            _profile(),
            _settings(apply_brush_size=True),
        )
