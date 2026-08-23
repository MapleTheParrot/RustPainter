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
from app.paint_timing import (
    LONG_DRAG_MAX_TEXELS_PER_SECOND,
    SETTLE_FLOOR_SECONDS,
    STROKE_GAP_FLOOR_SECONDS,
)
from app.painter import Painter, PainterSettings, PainterState, PaintingTarget
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
        "progress_callback_interval_seconds": 0.0,
        "safety_poll_interval_seconds": 0.002,
        # Checking each color as it goes down, and reading each pick back
        # off the panel, capture the screen; the tests that exercise them
        # turn them on against a simulated sign.
        "confirm_strokes": False,
        "verify_color_picks": False,
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


def test_a_paused_job_takes_new_timing_and_keeps_its_shape() -> None:
    """A pause is when timing gets changed, and the resumed strokes run on it.

    Only the timing and the guards move; what shaped the job - the brush it
    measured, the spacing its strokes were laid out with - stays as it was
    configured, whatever the new settings say about it.
    """

    input_controller = MockInputController()
    painter = Painter(input_controller)
    started = _settings(
        mouse_down_duration_seconds=0.004,
        delay_between_strokes_seconds=0.0,
        logical_pixel_spacing=1.0,
    )
    retuned = _settings(
        mouse_down_duration_seconds=0.004,
        delay_between_strokes_seconds=0.05,
        delay_between_colors_seconds=0.2,
        logical_pixel_spacing=2.0,
        apply_brush_size=True,
    )
    # Nothing to retune before the job is paused.
    assert not painter.retune(retuned)
    painter.start(_dot_plan(8), _profile(), started)
    assert not painter.retune(retuned)
    assert _wait_until(lambda: painter.progress.completed_strokes >= 2)

    assert painter.pause()
    assert painter.retune(retuned)
    live = painter._job.settings
    assert live.delay_between_strokes_seconds == 0.05
    assert live.delay_between_colors_seconds == 0.2
    assert live.logical_pixel_spacing == 1.0
    assert live.apply_brush_size is False
    # The run's pace no longer says anything about the machine's overhead.
    assert painter.paint_phase_timing is None

    paused_at = painter.progress.completed_strokes
    resumed = time.monotonic()
    assert painter.resume()
    assert painter.wait(_t(5.0))
    assert painter.state is PainterState.COMPLETED
    # Every stroke after the pause waited the new gap, which the old one
    # never did.
    remaining = 8 - paused_at
    assert time.monotonic() - resumed >= remaining * 0.05


def test_anti_afk_break_saves_jumps_reopens_and_reselects_the_color() -> None:
    """Every interval the job leaves the sign, jumps, and comes back.

    The break is Save, Space, the interact key (E) to reopen the sign, and
    then the color selected again before the next stroke, since the painting
    UI was closed and reopened in between.  The mouse is not touched between
    Save and the reopen: the game owns the cursor while the UI is closed.
    """

    input_controller = MockInputController()
    painter = Painter(input_controller)
    profile = _profile()
    profile.save_button = ScreenRect(600, 300, 100, 30)
    # Four strokes 0.3 s apart with a 0.5 s interval: the break falls once,
    # before the third stroke, and the interval counts afresh from its end.
    painter.start(
        _dot_plan(4),
        profile,
        _settings(
            mouse_down_duration_seconds=0.004,
            delay_between_strokes_seconds=0.3,
            anti_afk_enabled=True,
            anti_afk_interval_seconds=0.5,
        ),
    )
    assert painter.wait(_t(15.0))
    assert painter.state is PainterState.COMPLETED
    assert painter.progress.completed_strokes == 4

    events = input_controller.events
    kinds = [(event.kind, event.x, event.y, event.value) for event in events]
    keys = [index for index, event in enumerate(events) if event.kind == "key_down"]
    assert [events[index].value for index in keys] == ["SPACE", "E"], kinds
    for index in keys[:1]:
        # Save was clicked, at its centre, before the jump ...
        before = events[:index]
        save_moves = [
            event
            for event in before
            if event.kind == "move" and (event.x, event.y) == (650, 314)
        ]
        assert save_moves, kinds
        # ... and after it the sign was reopened with E, with no mouse event
        # in between, then the color was picked again (a move into the hue
        # bar) before the next stroke.
        after = events[index + 1 :]
        assert after[0].kind == "key_up", kinds
        assert (after[1].kind, after[1].value) == ("key_down", "E"), kinds
        assert after[2].kind == "key_up", kinds
        reselect = next(event for event in after[3:] if event.kind == "move")
        assert 720 <= reselect.x < 732, kinds
    assert not input_controller.held_buttons


def test_anti_afk_save_click_does_not_trip_the_mouse_guard() -> None:
    """Rust recentres the cursor the moment Save is pressed.

    The painting UI closes on the press, the game takes the cursor back and
    puts it at the screen centre while the button is still held.  That jump
    is the game's, not a hand's, so the break must not read it as movement
    and pause itself.
    """

    class SignInput(_HandInput):
        def mouse_down(self, button) -> None:
            super().mouse_down(button)
            # Pressed on Save: the UI closes and the cursor is recentred.
            if self._history and self._history[-1] == (650, 314):
                self.absolute_cursor = (960, 540)

        def move_mouse(self, x: float, y: float) -> None:
            super().move_mouse(x, y)
            # The painter's next move (the reopened sign) takes the cursor back.
            self.absolute_cursor = None

    input_controller = SignInput()
    painter = Painter(input_controller, screen_capture=_panel_capture)
    profile = _profile()
    profile.save_button = ScreenRect(600, 300, 100, 30)
    painter.start(
        _dot_plan(4),
        profile,
        _settings(
            mouse_down_duration_seconds=0.004,
            delay_between_strokes_seconds=0.3,
            pause_on_mouse_move=True,
            mouse_move_pause_threshold_pixels=24,
            anti_afk_enabled=True,
            anti_afk_interval_seconds=1.0,
        ),
    )
    assert painter.wait(_t(30.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.progress.completed_strokes == 4
    keys = [e.value for e in input_controller.events if e.kind == "key_down"]
    assert keys and keys == ["SPACE", "E"] * (len(keys) // 2), keys


def test_anti_afk_needs_the_save_button_only_for_real_input() -> None:
    class RealInput(MockInputController):
        emits_real_input = True

    settings = _settings(anti_afk_enabled=True)
    with pytest.raises(ValueError, match="Save button"):
        Painter(RealInput()).configure(_dot_plan(1), _profile(), settings)

    # Mock input runs the plan without a Save button and without a break.
    input_controller = MockInputController()
    painter = Painter(input_controller)
    painter.start(_dot_plan(3), _profile(), _settings(anti_afk_enabled=True, anti_afk_interval_seconds=0.01))
    assert painter.wait(_t(2.0))
    assert painter.state is PainterState.COMPLETED
    assert not any(event.kind == "key_down" for event in input_controller.events)


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


def test_a_cursor_flung_to_a_screen_corner_only_pauses() -> None:
    """Nothing the mouse does aborts a job; only the user's Stop does.

    A cursor found in a screen corner used to be an emergency-stop gesture.
    A game recentring its cursor can land there too, and an abort throws
    away every unsaved stroke, so the corner is now just more movement: the
    job pauses and resumes from the same stroke.
    """

    class CornerInput(MockInputController):
        emits_real_input = True

        def get_cursor_position(self) -> tuple[int, int]:
            return (0, 0)

    input_controller = CornerInput()
    painter = Painter(input_controller)
    painter.start(
        _dot_plan(3),
        _profile(),
        _settings(pause_on_mouse_move=True, mouse_move_pause_threshold_pixels=10),
    )
    assert _wait_until(lambda: painter.state is PainterState.PAUSED)
    assert painter.state_reason.startswith("mouse moved")
    assert not input_controller.held_buttons
    assert painter.abort("user")
    assert painter.wait(_t(2.0))


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
    painter._CONFIRM_SETTLE_SECONDS = 0.0  # type: ignore[misc]
    painter._CLEAR_SETTLE_SECONDS = 0.0  # type: ignore[misc]
    painter._KEY_HOLD_SECONDS = 0.0  # type: ignore[misc]
    painter._KEY_GAP_SECONDS = 0.0  # type: ignore[misc]
    painter._KEY_GAP_SECONDS = 0.0  # type: ignore[misc]
    painter._SETTLE_FLOOR_SECONDS = 0.0  # type: ignore[misc]
    painter._STROKE_GAP_FLOOR_SECONDS = 0.0  # type: ignore[misc]
    painter._LONG_DRAG_MAX_TEXELS_PER_SECOND = float("inf")  # type: ignore[misc]
    painter._LONG_DRAG_MAX_STEP_TEXELS = float("inf")  # type: ignore[misc]
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
    # The conversion interpolates between the bracketing probes rather than
    # reading a global fitted line: the simulator quantizes bands to whole
    # pixels, so its size-5 probe painted 12px where the wanted cell is 12.5 -
    # and the interpolated answer sizes up until the measurements say the cell
    # is covered, instead of trusting a line that misses the nearby probes.
    size = model.clamped_size_for_fraction(5 / 128)
    assert 5.0 <= size <= 5.5
    assert model.fraction_for_size(size) >= 5 / 128


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


def test_non_square_cells_size_the_brush_to_the_wider_pitch() -> None:
    """The bug behind bare seams between rows on non-square logical cells.

    A 640x320 canvas under a 64x20 grid has cells 10px wide and 16px tall.
    Sizing to the narrower pitch (10px, the old ``min``) leaves a 6px bare
    stripe under every row boundary; sizing to the wider pitch overshoots the
    columns instead, which the later-painted color simply owns.
    """

    controller = MockInputController()
    profile = _sized_profile("Tall cells")
    plan = PaintPlan(
        64,
        20,
        (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),),
    )
    painter = Painter(controller, screen_capture=_panel_capture)

    assert painter.start(plan, profile, _settings(apply_brush_size=True))
    assert painter.wait(_t(2.0))

    assert painter.state is PainterState.COMPLETED
    # 16px pitch plus the half-texel hedge; the old code typed 10.5.
    assert _typed_values(controller) == ["16.5"]


def test_row_sized_brush_covers_cell_width_by_stroke_extension() -> None:
    """A brush narrower than the cell drags further instead of sizing up.

    One Size unit paints one vertical pixel but only half a horizontal one
    here - the shape a calibrated rectangle whose aspect does not match the
    sign texture's produces.  Sizing up to cover the columns would smear the
    rows; keeping the row-exact size and extending each stroke sideways
    covers the full cell width with no vertical overshoot at all.
    """

    model = fit_brush_size_model(
        [(size, size / 320.0) for size in (12, 30, 60)],
        samples_x=[(size, size * 0.5 / 640.0) for size in (12, 30, 60)],
    )
    target = PaintingTarget(
        canvas=ScreenRect(100, 100, 640, 320),
        color_box=ScreenRect(600, 500, 100, 100),
        hue_bar=ScreenRect(720, 500, 12, 100),
    )
    plan = PaintPlan(64, 32, ())

    size = Painter._brush_plan_size(target, plan, 1, 1.0, model)
    assert size == pytest.approx(10.5)  # rows decide the Size number alone
    # The 10.5-unit brush paints only 5.25px of a 10px-wide cell; each stroke
    # end reaches out the 2.5px that covers the rest (plus a half-texel hedge).
    extension = Painter._stroke_extension_pixels(target.canvas, plan, model, size)
    assert extension == pytest.approx(2.5)

    # A comfortably wide footprint needs no extension at all.
    wide = fit_brush_size_model(
        [(size_, size_ / 320.0) for size_ in (12, 30, 60)],
        samples_x=[(size_, size_ * 3.0 / 640.0) for size_ in (12, 30, 60)],
    )
    assert Painter._brush_plan_size(target, plan, 1, 1.0, wide) == pytest.approx(10.5)
    assert Painter._stroke_extension_pixels(target.canvas, plan, wide, 10.5) == 0.0


def test_stroke_grid_registers_to_the_canonical_texture_extent() -> None:
    """Cells must be whole texels wide or stamps drift one texel per few cells.

    The live sign measured 318.4 columns x 238.4 rows inside the hand-dragged
    rectangle - a 320x240 texture with its outermost texels under the frame.
    Laying 10 cells on the rectangle makes them 31.84 texels wide; stamps land
    on whole texels, so the rounding slips one texel every few cells and a
    later neighbour's stamp eats a texel column of the cell before it (seen
    directly in a sign texture downloaded from the game).  Registering the
    grid to 320 x the measured texel size makes the pitch exact.
    """

    model = fit_brush_size_model(
        [(10, 10 / 238.4), (50, 50 / 238.4)],
        samples_x=[(10, 10 / 318.4), (50, 50 / 318.4)],
    )
    canvas = ScreenRect(491, -1260, 1299, 1080)

    registered = Painter._registered_canvas(canvas, model)

    assert registered.left == canvas.left and registered.top == canvas.top
    assert registered.width == pytest.approx(320 * 1299 / 318.4)
    assert registered.height == pytest.approx(240 * 1080 / 238.4)

    # Without a horizontal measurement there is nothing to register against.
    vertical_only = fit_brush_size_model([(10, 10 / 238.4), (50, 50 / 238.4)])
    assert Painter._registered_canvas(canvas, vertical_only) is canvas
    assert Painter._registered_canvas(canvas, None) is canvas


def test_measured_rendering_bias_shifts_artwork_strokes_the_other_way() -> None:
    """Rust stamps about a texel off the cursor; painting must cancel it.

    The model carries the measured bias (paint landed 5px right and 2px down
    here), so the dab is commanded 5px left and 2px up of the cell center and
    the rendered result lands centered.
    """

    controller = MockInputController()
    profile = CalibrationProfile.new(
        "Biased sign",
        canvas=ScreenRect(100, 100, 640, 320),
        color_box=ScreenRect(600, 500, 100, 100),
        hue_bar=ScreenRect(720, 500, 12, 100),
        brush_size_box=ScreenRect(800, 100, 60, 24),
    )
    profile.metadata["brush_size_model"] = fit_brush_size_model(
        [(size, size / 320.0) for size in (60, 30, 12)],
        bias=(5 / 640.0, 2 / 320.0),
    ).to_dict()
    plan = PaintPlan(
        64,
        32,
        (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),),
    )
    painter = Painter(controller, screen_capture=_panel_capture)

    assert painter.start(plan, profile, _settings(apply_brush_size=True))
    assert painter.wait(_t(2.0))
    assert painter.state is PainterState.COMPLETED

    canvas = profile.canvas
    dabs = [
        _position_at(controller, index)
        for index, event in enumerate(controller.events)
        if event.kind == "mouse_down"
        and canvas.contains(*_position_at(controller, index))
    ]
    # Cell (0, 0) of a 64x32 plan on this canvas centers at (105, 105); the
    # measured bias pulls the command to (100, 103).
    assert dabs == [(100, 103)]


class _TimedInput(MockInputController):
    """A real-input mock that timestamps every press and release."""

    emits_real_input = True

    def __init__(self) -> None:
        super().__init__()
        # (pressed, last move while pressed, released) per press
        self.presses: list[tuple[float, float, float]] = []
        self._pressed_at: float | None = None
        self._moved_at: float | None = None

    def move_mouse(self, x: float, y: float) -> None:
        super().move_mouse(x, y)
        self._moved_at = time.monotonic()

    def mouse_down(self, button="left") -> None:  # type: ignore[override]
        super().mouse_down(button)
        self._pressed_at = self._moved_at = time.monotonic()

    def mouse_up(self, button="left") -> None:  # type: ignore[override]
        released_at = time.monotonic()
        super().mouse_up(button)
        if self._pressed_at is not None:
            self.presses.append((self._pressed_at, self._moved_at or released_at, released_at))
            self._pressed_at = None


def _canvas_presses(controller: _TimedInput) -> list[tuple[float, float]]:
    """(press length, hold after the last move) of each canvas press, in order."""

    canvas = ScreenRect(100, 100, 400, 80)
    timings: list[tuple[float, float]] = []
    press_index = 0
    for index, event in enumerate(controller.events):
        if event.kind != "mouse_down":
            continue
        position = _position_at(controller, index)
        pressed, moved, released = controller.presses[press_index]
        press_index += 1
        if canvas.contains(*position):
            timings.append((released - pressed, released - moved))
    return timings


def test_a_short_drag_is_held_as_long_as_a_dab() -> None:
    """A run of a few cells is a few screen pixels: over in milliseconds.

    Read back from a real sign, every hole was such a run, missing from the
    middle of a stroke, while dabs - which already had a hold - were fine.
    The press now lasts a frame whatever its length, so the game sees the
    cursor held at the far end before the button comes up.
    """

    controller = _TimedInput()
    painter = Painter(controller, screen_capture=_panel_capture)
    # One cell per screen pixel: a three-cell run is a two-pixel drag.
    plan = PaintPlan(
        400,
        1,
        (
            ColorGroup(
                (220, 40, 20),
                (Stroke(0, 0, 2, 0), Stroke(10, 0, 10, 0)),
                4,
            ),
        ),
    )
    assert painter.start(plan, _profile(), _settings())
    assert painter.wait(_t(5.0))
    assert painter.state is PainterState.COMPLETED

    (drag, drag_tail), (dab, _) = _canvas_presses(controller)
    assert drag >= Painter._MIN_PRESS_SECONDS * 0.9
    assert dab >= Painter._MIN_PRESS_SECONDS * 0.9
    # The wait is spent at the far end of the drag, not before it starts.
    assert drag_tail >= Painter._MIN_PRESS_SECONDS * 0.5


def test_a_long_drag_dwells_one_frame_at_its_end() -> None:
    """The far end of a long drag is held across a frame before release.

    The game paints between cursor samples taken at ~15 FPS, so the last
    inter-sample stretch of a drag - up to sixteen texels of it - exists
    only if the game gets a frame with the cursor resting at the far end.
    Short strokes have always been held for this reason; the murica XXL
    post-mortem extended the same dwell to long drags, one frame each.
    """

    controller = _TimedInput()
    painter = Painter(controller, screen_capture=_panel_capture)
    plan = PaintPlan(400, 1, (ColorGroup((220, 40, 20), (Stroke(0, 0, 399, 0),), 400),))
    # 399 px at 2000 px/s is a 200 ms drag, well past the minimum press.
    assert painter.start(plan, _profile(), _settings(stroke_speed_pixels_per_second=2000.0))
    assert painter.wait(_t(5.0))
    assert painter.state is PainterState.COMPLETED

    ((drag, tail),) = _canvas_presses(controller)
    assert drag >= 0.18
    # The button stays down for one press-floor at the far end, no longer.
    assert tail >= Painter._MIN_PRESS_SECONDS * 0.8
    assert tail < Painter._MIN_PRESS_SECONDS * 2.0 + 0.03 * _TIMEOUT_SCALE


def _turbo(**overrides: object) -> PainterSettings:
    """The fastest preset as the user can type it, floors and all."""

    values: dict[str, object] = {
        "stroke_speed_pixels_per_second": 2200.0,
        "mouse_down_duration_seconds": 0.012,
        "delay_after_hue_seconds": 0.0,
        "delay_after_saturation_value_seconds": 0.0,
        "delay_between_strokes_seconds": 0.0,
        "delay_between_colors_seconds": 0.0,
        "stroke_interpolation_step_pixels": 8.0,
    }
    values.update(overrides)
    return _settings(**values)


def _moves_during_press(controller: MockInputController, press: int) -> list[tuple[int, int]]:
    """The cursor positions commanded while the ``press``-th button was down."""

    positions: list[tuple[int, int]] = []
    seen = -1
    down = False
    for event in controller.events:
        if event.kind == "mouse_down":
            seen += 1
            down = seen == press
        elif event.kind == "mouse_up":
            if down:
                break
            down = False
        elif down and event.kind == "move" and event.x is not None and event.y is not None:
            positions.append((event.x, event.y))
    return positions


def test_a_short_run_at_top_speed_costs_only_the_frame_hold() -> None:
    """A run of a few texels is a dab that moved: the hold lands it, and the
    long-drag cap never touches it."""

    controller = _TimedInput()
    painter = Painter(controller, screen_capture=_panel_capture)
    # One cell per screen pixel on the 400-wide canvas: a two-pixel run.
    plan = PaintPlan(400, 1, (ColorGroup((220, 40, 20), (Stroke(0, 0, 2, 0),), 3),))
    assert painter.start(plan, _profile(), _turbo())
    assert painter.wait(_t(5.0))
    assert painter.state is PainterState.COMPLETED

    ((drag, _),) = _canvas_presses(controller)
    assert Painter._MIN_PRESS_SECONDS * 0.9 <= drag < Painter._MIN_PRESS_SECONDS + 0.05 * _TIMEOUT_SCALE


def test_a_long_drag_at_top_speed_is_paced_by_the_texel() -> None:
    """However fast the preset, a long drag crosses the sign no faster than
    the rate the game paints faithfully, with a cursor event on every texel.
    No measured grid here, so the logical cell - one pixel - is the texel."""

    controller = _TimedInput()
    painter = Painter(controller, screen_capture=_panel_capture)
    plan = PaintPlan(400, 1, (ColorGroup((220, 40, 20), (Stroke(0, 0, 399, 0),), 400),))
    assert painter.start(plan, _profile(), _turbo())
    assert painter.wait(_t(10.0))
    assert painter.state is PainterState.COMPLETED

    ((drag, _),) = _canvas_presses(controller)
    # 399 px at the capped 250 texels/s, not at 2200 px/s.
    assert drag >= 399 / LONG_DRAG_MAX_TEXELS_PER_SECOND * 0.9
    # The canvas press is the third (two picker clicks come first), and the
    # 8 px step was brought down to the texel.
    moves = _moves_during_press(controller, 2)
    assert len(moves) >= 399
    assert all(abs(b[0] - a[0]) <= 1 for a, b in zip(moves, moves[1:]))
    assert moves[-1][0] - moves[0][0] >= 398


def test_delays_under_the_frame_floor_are_run_at_the_floor() -> None:
    """Zero between the picker clicks or between strokes is run as a frame's
    worth - the game would not see the click otherwise - whatever the preset
    says."""

    controller = _TimedInput()
    painter = Painter(controller, screen_capture=_panel_capture)
    plan = PaintPlan(
        400, 1, (ColorGroup((220, 40, 20), (Stroke(0, 0, 0, 0), Stroke(10, 0, 10, 0)), 2),)
    )
    assert painter.start(plan, _profile(), _turbo())
    assert painter.wait(_t(5.0))
    assert painter.state is PainterState.COMPLETED

    hue, saturation_value, first_dab, second_dab = controller.presses
    assert saturation_value[0] - hue[2] >= SETTLE_FLOOR_SECONDS * 0.9
    assert first_dab[0] - saturation_value[2] >= SETTLE_FLOOR_SECONDS * 0.9
    assert second_dab[0] - first_dab[2] >= STROKE_GAP_FLOOR_SECONDS * 0.9


def test_dry_runs_do_not_wait_out_the_minimum_press() -> None:
    controller = _TimedInput()
    controller.emits_real_input = False  # type: ignore[misc]
    painter = Painter(controller)
    plan = PaintPlan(400, 1, (ColorGroup((220, 40, 20), (Stroke(0, 0, 2, 0),), 3),))
    assert painter.start(plan, _profile(), _settings())
    assert painter.wait(_t(5.0))
    assert painter.state is PainterState.COMPLETED

    ((drag, _),) = _canvas_presses(controller)
    assert drag < Painter._MIN_PRESS_SECONDS / 2


def test_the_touch_up_pass_uses_the_cleared_sign_to_see_holes() -> None:
    """End to end: the job clears the sign, keeps that capture, and the
    verification pass repaints a cell that stayed bare - which a one-color
    plan gives the old two-way comparison no second color to notice by.
    """

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _calibrating_profile("Holes")
    assert profile.canvas is not None and profile.clear_button is not None
    canvas, clear_button = profile.canvas, profile.clear_button
    probing_sign = _clearable_sign(controller, canvas, clear_button, sign_rows=320)
    dark, bare = (40, 40, 40), (96, 96, 96)
    hole = (30, 20)  # logical (x, y) on the 64x32 plan; 10 px cells

    def capture(rect) -> Image.Image:
        if (rect.left, rect.top) != (canvas.left, canvas.top):
            return probing_sign(rect)
        position = (0, 0)
        cleared = False
        artwork_strokes = 0
        for event in controller.events:
            if event.kind == "move" and event.x is not None and event.y is not None:
                position = (event.x, event.y)
            elif event.kind == "mouse_down":
                if clear_button.contains(*position):
                    cleared = True
                    artwork_strokes = 0
                elif cleared and canvas.contains(*position):
                    artwork_strokes += 1
        if not cleared:
            return probing_sign(rect)  # the probes, until the clear click
        image = Image.new("RGB", (rect.width, rect.height), bare)
        if artwork_strokes:
            image.paste(dark, (0, 0, rect.width, rect.height))
            x, y = hole
            image.paste(bare, (x * 10, y * 10, x * 10 + 10, y * 10 + 10))
        return image

    painter = _impatient(Painter(controller, screen_capture=capture))
    plan = PaintPlan(
        64,
        32,
        (ColorGroup(dark, tuple(Stroke(0, y, 63, y) for y in range(32)), 64 * 32),),
    )
    assert painter.start(
        plan, profile, _settings(apply_brush_size=True, verify_passes=1)
    )
    assert painter.wait(_t(30.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason

    cleared_at = next(
        index
        for index, event in enumerate(controller.events)
        if event.kind == "mouse_down"
        and clear_button.contains(*_position_at(controller, index))
    )
    presses = [
        _position_at(controller, index)
        for index, event in enumerate(controller.events)
        if index > cleared_at
        and event.kind == "mouse_down"
        and canvas.contains(*_position_at(controller, index))
    ]
    # 32 artwork rows, then exactly one touch-up press, on the hole.
    assert len(presses) == 33, presses
    center = (canvas.left + hole[0] * 10 + 5, canvas.top + hole[1] * 10 + 5)
    touch_up = presses[-1]
    assert abs(touch_up[0] - center[0]) <= 8 and abs(touch_up[1] - center[1]) <= 8, (
        touch_up,
        center,
    )
    # The pass ran to its end, so its clock is on record for the estimate.
    timing = painter.touch_up_timing
    assert timing is not None
    assert timing.passes == 1
    assert timing.seconds > 0.0
    # And the artwork's own clock does not count it.
    measured = painter.paint_phase_timing
    assert measured is not None and measured.strokes == 32


def test_time_left_is_predicted_before_the_first_stroke_and_measured_after() -> None:
    input_controller = MockInputController()
    painter = Painter(input_controller)
    painter.configure(_dot_plan(40), _profile(), _settings())
    job = painter._job
    assert job is not None
    # A schedule priced from the painter's own timing rules, not "unknown".
    initial = painter._initial_estimate(job)
    assert initial is not None and initial > 0
    # Mock input runs no checks and no touch-up, so neither is priced.
    assert painter._checks_estimate(job, initial) == 0.0
    assert painter._touch_up_estimate(job, initial) == 0.0

    assert painter.start()
    assert painter.wait(_t(2.0))
    assert painter.state is PainterState.COMPLETED
    measured = painter.paint_phase_timing
    assert measured is not None
    assert measured.strokes == 40
    assert measured.predicted_seconds > 0
    assert measured.actual_seconds >= 0
    assert painter.progress.percent == 100.0


def test_the_first_estimate_counts_the_checks_and_the_touch_up() -> None:
    """Under real input the time left before the first stroke is the whole
    job's: the artwork, a capture per color that paints plus the repaints
    the checks turn out to need, and the touch-up pass at the end, each
    priced from what earlier runs measured."""

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    painter = Painter(
        controller,
        screen_capture=_panel_capture,
        check_capture_seconds=2.0,
        check_repaint_fraction=0.5,
        touch_up_fraction=0.25,
    )
    plan = _confirmation_plan()  # two colors that paint
    painter.configure(
        plan,
        _profile(),
        _settings(
            confirm_strokes=True, verify_passes=1, require_foreground=False
        ),
    )
    job = painter._job
    assert job is not None
    painting = painter._work_schedule(job.plan, job.target, job.settings).total
    assert painter._checks_estimate(job, painting) == pytest.approx(
        2 * 2.0 + 0.5 * painting
    )
    assert painter._touch_up_estimate(job, painting) == pytest.approx(0.25 * painting)
    initial = painter._initial_estimate(job)
    assert initial == pytest.approx(painting * 1.75 + 4.0)

    # Turned off, neither is priced.
    painter.configure(
        plan,
        _profile(),
        _settings(confirm_strokes=False, verify_passes=0, require_foreground=False),
    )
    job = painter._job
    assert job is not None
    assert painter._checks_estimate(job, painting) == 0.0
    assert painter._touch_up_estimate(job, painting) == 0.0


def test_the_touch_up_to_come_is_in_the_time_left_but_not_the_percent() -> None:
    painter = Painter(MockInputController())
    painter.configure(_dot_plan(3), _profile(), _settings())
    painter._set_progress(
        color_index=1,
        total_colors=1,
        stroke_index_in_color=50,
        strokes_in_color=100,
        completed_strokes=50,
        total_strokes=100,
        completed_work=50.0,
        total_work=100.0,
        phase_elapsed=50.0,
        pending_seconds=30.0,
        message="Painting",
    )
    progress = painter.progress
    assert progress.percent == pytest.approx(50.0)
    assert progress.estimated_remaining_seconds == pytest.approx(80.0)


def test_a_later_pass_is_timed_against_its_own_clock() -> None:
    """A touch-up re-enters the plan loop after the artwork; measured against
    the whole run's elapsed, one repainted cell would claim hours left."""

    painter = Painter(MockInputController())
    painter.configure(_dot_plan(3), _profile(), _settings())
    painter._started_at = time.monotonic() - 3600.0  # an hour of artwork

    painter._set_progress(
        color_index=1,
        total_colors=1,
        stroke_index_in_color=1,
        strokes_in_color=100,
        completed_strokes=1,
        total_strokes=100,
        completed_work=1.0,
        total_work=100.0,
        phase_elapsed=1.0,
        message="Touching up",
    )
    remaining = painter.progress.estimated_remaining_seconds
    assert remaining is not None
    # Ninety-nine seconds of predicted work left, paced by one second done.
    assert 90.0 < remaining < 110.0
    assert painter.progress.elapsed_seconds >= 3600.0


def _sign_with_bare_rows(
    controller: MockInputController,
    canvas: ScreenRect,
    *,
    before: Image.Image,
    hole_rows: range,
):
    """A 64x32-cell sign that shows ``before`` until the first artwork stroke,
    then the artwork with ``hole_rows`` of the dark color never registered."""

    dark, near_bare, bare = (40, 40, 40), (104, 104, 104), (96, 96, 96)

    def capture(rect) -> Image.Image:
        if (rect.left, rect.top) != (canvas.left, canvas.top):
            return _panel_capture(rect)
        position = (0, 0)
        artwork_strokes = 0
        for event in controller.events:
            if event.kind == "move" and event.x is not None and event.y is not None:
                position = (event.x, event.y)
            elif event.kind == "mouse_down" and canvas.contains(*position):
                artwork_strokes += 1
        if not artwork_strokes:
            return before.copy()
        image = Image.new("RGB", (rect.width, rect.height), dark)
        image.paste(near_bare, (0, 280, rect.width, rect.height))
        for y in hole_rows:
            image.paste(bare, (0, y * 10, rect.width, y * 10 + 10))
        return image

    return capture, dark, near_bare


def _bare_rows_plan(dark, near_bare) -> PaintPlan:
    return PaintPlan(
        64,
        32,
        (
            ColorGroup(dark, tuple(Stroke(0, y, 63, y) for y in range(28)), 64 * 28),
            ColorGroup(near_bare, tuple(Stroke(0, y, 63, y) for y in range(28, 32)), 64 * 4),
        ),
    )


def _press_points_on(controller: MockInputController, canvas: ScreenRect) -> list[tuple[int, int]]:
    return [
        _position_at(controller, index)
        for index, event in enumerate(controller.events)
        if event.kind == "mouse_down" and canvas.contains(*_position_at(controller, index))
    ]


def test_with_sizing_off_the_sign_before_painting_is_the_bare_reference() -> None:
    """Live: sizing off, so nothing cleared the sign and nothing captured it
    bare; thirty rows the game never registered read as bare canvas, which
    without a bare reference is "some other color, scattered" - set aside.
    The sign as it stands before the first stroke is captured instead, and
    once it is known to be bare those rows are holes and are repainted."""

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _calibrating_profile("Bare before")
    assert profile.canvas is not None
    canvas = profile.canvas
    hole_rows = range(10, 18)  # 512 cells: past the scattered-verdict allowance
    capture, dark, near_bare = _sign_with_bare_rows(
        controller,
        canvas,
        before=Image.new("RGB", (canvas.width, canvas.height), (96, 96, 96)),
        hole_rows=hole_rows,
    )
    painter = _impatient(Painter(controller, screen_capture=capture))
    assert painter.start(
        _bare_rows_plan(dark, near_bare),
        profile,
        _settings(apply_brush_size=False, verify_passes=1),
    )
    assert painter.wait(_t(30.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason

    presses = _press_points_on(controller, canvas)
    touch_ups = presses[32:]
    assert len(presses) == 32 + len(hole_rows), presses
    rows_repainted = sorted((y - canvas.top) // 10 for _, y in touch_ups)
    assert rows_repainted == list(hole_rows)


def test_a_sign_that_is_not_bare_before_painting_gives_no_bare_reference() -> None:
    """The same holes over an earlier artwork: its capture is no bare sign,
    so it is not used as one, and the verdicts stay what they were."""

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _calibrating_profile("Painted before")
    assert profile.canvas is not None
    canvas = profile.canvas
    earlier = Image.new("RGB", (canvas.width, canvas.height), (96, 96, 96))
    for index, color in enumerate(((220, 30, 30), (30, 200, 60), (40, 60, 230), (250, 220, 20))):
        earlier.paste(color, (index * 160, 0, index * 160 + 160, canvas.height))
    capture, dark, near_bare = _sign_with_bare_rows(
        controller, canvas, before=earlier, hole_rows=range(10, 18)
    )
    painter = _impatient(Painter(controller, screen_capture=capture))
    assert painter.start(
        _bare_rows_plan(dark, near_bare),
        profile,
        _settings(apply_brush_size=False, verify_passes=1),
    )
    assert painter.wait(_t(30.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert len(_press_points_on(controller, canvas)) == 32


# ------------------------------------------------------------------ UI guard


def _ui_screen(profile):
    """A screen showing Rust's painting UI at the profile's rectangles."""

    from test_ui_guard import FakeScreen

    return FakeScreen(profile)


def _quick_ui_guard(painter: Painter) -> Painter:
    """Look for the UI every few milliseconds instead of every second."""

    painter._UI_GUARD_INTERVAL_SECONDS = 0.02  # type: ignore[misc]
    painter._UI_GUARD_RECHECK_SECONDS = 0.01  # type: ignore[misc]
    return painter


def test_the_ui_guard_pauses_when_the_painting_ui_disappears() -> None:
    """A server restart or a kick takes the sign off the screen mid-job.

    The Rust window is still in front and the cursor still goes where it is
    sent, so neither existing guard notices.  The calibrated widgets no
    longer being on the screen is what does, and the job pauses - it never
    aborts - and carries on from the same stroke once the sign is back.
    """

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    screen = _ui_screen(profile)
    painter = _quick_ui_guard(Painter(controller, screen_capture=screen))
    painter.start(
        _dot_plan(40),
        profile,
        _settings(
            mouse_down_duration_seconds=0.004,
            delay_between_strokes_seconds=0.02,
            ui_guard_enabled=True,
        ),
    )
    assert _wait_until(lambda: painter.progress.completed_strokes >= 3)
    assert painter.state is PainterState.RUNNING

    screen.open = False
    assert _wait_until(lambda: painter.state is PainterState.PAUSED)
    assert "painting UI not found" in painter.state_reason
    paused_at = painter.progress.completed_strokes
    assert paused_at < 40
    event_count = len(controller.events)
    time.sleep(0.06)
    assert len(controller.events) == event_count

    # Resumed with the sign still closed: the guard looks before the next
    # stroke and pauses again without any input going out.
    assert painter.resume()
    assert _wait_until(lambda: painter.state is PainterState.PAUSED)
    assert len(controller.events) == event_count

    screen.open = True
    assert painter.resume()
    assert painter.wait(_t(5.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.progress.completed_strokes == 40
    assert not controller.held_buttons


def test_a_job_started_without_the_painting_ui_pauses_before_any_input() -> None:
    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    screen = _ui_screen(profile)
    screen.open = False
    painter = _quick_ui_guard(Painter(controller, screen_capture=screen))
    painter.start(_dot_plan(3), profile, _settings(ui_guard_enabled=True))
    assert _wait_until(lambda: painter.state is PainterState.PAUSED)
    assert "painting UI not found" in painter.state_reason
    assert controller.events == []

    screen.open = True
    assert painter.resume()
    assert painter.wait(_t(3.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.progress.completed_strokes == 3


def test_turning_the_ui_guard_off_during_its_pause_lets_the_job_go_on() -> None:
    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile()
    screen = _ui_screen(profile)
    screen.open = False
    painter = _quick_ui_guard(Painter(controller, screen_capture=screen))
    painter.start(_dot_plan(3), profile, _settings(ui_guard_enabled=True))
    assert _wait_until(lambda: painter.state is PainterState.PAUSED)

    assert painter.retune(_settings(ui_guard_enabled=False))
    assert painter.resume()
    assert painter.wait(_t(3.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason


def test_the_ui_guard_is_off_for_dry_runs_and_when_disabled() -> None:
    profile = _profile()
    for controller in (DryRunInputController(), MockInputController()):
        screen = _ui_screen(profile)
        screen.open = False
        painter = _quick_ui_guard(Painter(controller, screen_capture=screen))
        painter.start(_dot_plan(2), profile, _settings(ui_guard_enabled=True))
        assert painter.wait(_t(3.0))
        assert painter.state is PainterState.COMPLETED, painter.state_reason

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    screen = _ui_screen(profile)
    screen.open = False
    painter = _quick_ui_guard(Painter(controller, screen_capture=screen))
    painter.start(_dot_plan(2), profile, _settings(ui_guard_enabled=False))
    assert painter.wait(_t(3.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason


class _SignInput(MockInputController):
    """Real input whose Save click closes the sign and whose E reopens it."""

    emits_real_input = True

    def __init__(self, screen, save_centre: tuple[int, int], *, reopens: bool = True) -> None:
        super().__init__()
        self.screen = screen
        self.save_centre = save_centre
        self.reopens = reopens
        self._last_move: tuple[int, int] | None = None

    def move_mouse(self, x: float, y: float) -> None:
        super().move_mouse(x, y)
        self._last_move = (int(round(x)), int(round(y)))

    def mouse_down(self, button) -> None:
        super().mouse_down(button)
        if self._last_move == self.save_centre:
            self.screen.open = False

    def press_key(self, key: str, *, hold_seconds: float = 0.0) -> None:
        super().press_key(key, hold_seconds=hold_seconds)
        if key == "E" and self.reopens:
            self.screen.open = True


def test_the_anti_afk_break_closing_the_sign_does_not_trip_the_ui_guard() -> None:
    """The break leaves the painting UI on purpose, and the guard lets it.

    Between Save and the reopen the widgets are off the screen exactly as
    they would be after a kick; the difference is that the job did it, and
    knows to look away until it has pressed E.
    """

    profile = _profile()
    profile.save_button = ScreenRect(600, 300, 100, 30)
    screen = _ui_screen(profile)
    controller = _SignInput(screen, (650, 314))
    states: list[PainterState] = []
    painter = _quick_ui_guard(
        Painter(
            controller,
            screen_capture=screen,
            on_state_change=lambda state, _reason: states.append(state),
        )
    )
    painter._AFK_REOPEN_GRACE_SECONDS = 0.3  # type: ignore[misc]
    painter._AFK_REOPEN_POLL_SECONDS = 0.02  # type: ignore[misc]
    painter.start(
        _dot_plan(4),
        profile,
        _settings(
            mouse_down_duration_seconds=0.004,
            delay_between_strokes_seconds=0.3,
            anti_afk_enabled=True,
            anti_afk_interval_seconds=0.5,
            ui_guard_enabled=True,
        ),
    )
    assert painter.wait(_t(30.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.progress.completed_strokes == 4
    assert PainterState.PAUSED not in states, states
    # Real input runs every wait at the game's frame floor, so a stroke and
    # its gap outrun the half-second interval and the break recurs; each one
    # closed the sign and the guard let every one of them.
    keys = [event.value for event in controller.events if event.kind == "key_down"]
    assert keys and keys == ["SPACE", "E"] * (len(keys) // 2), keys


def test_a_sign_that_does_not_reopen_after_the_break_pauses_the_job() -> None:
    """E pressed, and no painting UI came back: the job waits for a hand.

    Without the guard the next stroke would be a click in the game world.
    The break still counts as taken - the user reopening the sign is proof
    enough of a player who is not idle - so resuming paints straight away.
    """

    profile = _profile()
    profile.save_button = ScreenRect(600, 300, 100, 30)
    screen = _ui_screen(profile)
    controller = _SignInput(screen, (650, 314), reopens=False)
    painter = _quick_ui_guard(Painter(controller, screen_capture=screen))
    painter._AFK_REOPEN_GRACE_SECONDS = 0.3  # type: ignore[misc]
    painter._AFK_REOPEN_POLL_SECONDS = 0.02  # type: ignore[misc]
    painter.start(
        _dot_plan(4),
        profile,
        _settings(
            mouse_down_duration_seconds=0.004,
            delay_between_strokes_seconds=0.3,
            anti_afk_enabled=True,
            anti_afk_interval_seconds=0.5,
            ui_guard_enabled=True,
        ),
    )
    assert _wait_until(lambda: painter.state is PainterState.PAUSED, timeout=10.0)
    assert "did not reopen" in painter.state_reason
    keys = [event.value for event in controller.events if event.kind == "key_down"]
    assert keys == ["SPACE", "E"], keys
    events_at_pause = len(controller.events)

    # The player opens the sign and resumes: the job paints on, and does
    # not take another break before it has painted a stroke.
    screen.open = True
    controller.reopens = True
    assert painter.resume()
    assert painter.wait(_t(30.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.progress.completed_strokes == 4
    later = [
        event.kind for event in controller.events[events_at_pause:]
        if event.kind in {"mouse_down", "key_down"}
    ]
    assert later and later[0] == "mouse_down", later


# ------------------------------------------------------------ resume offset


def _two_color_plan() -> PaintPlan:
    """Six strokes in two colors on a 64x32 grid."""

    first = tuple(Stroke(x, 0, x, 0) for x in (0, 10, 20, 30))
    second = tuple(Stroke(x, 8, x, 8) for x in (5, 15))
    return PaintPlan(
        64,
        32,
        (ColorGroup((40, 80, 160), first, 4), ColorGroup((200, 40, 40), second, 2)),
    )


def _canvas_clicks(controller: MockInputController, canvas: ScreenRect) -> list[tuple[int, int]]:
    return [
        _position_at(controller, index)
        for index, event in enumerate(controller.events)
        if event.kind == "mouse_down" and canvas.contains(*_position_at(controller, index))
    ]


def test_a_resumed_job_skips_the_clear_and_probe_and_paints_from_its_offset() -> None:
    """Picking a half-painted sign up again must not wipe or probe it.

    Everything the interrupted job painted is on the sign, so the resumed
    one neither clears it nor stamps probes on it; it types the stored
    brush for the group it resumes into, selects that group's color, and
    paints from the offset stroke to the end.
    """

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _calibrating_profile("Resumable")
    assert profile.canvas is not None and profile.clear_button is not None
    profile.metadata["brush_size_model"] = fit_brush_size_model(
        [(size, size / 320.0) for size in (60, 30, 12)]
    ).to_dict()
    painter = _impatient(
        Painter(
            controller,
            screen_capture=_clearable_sign(
                controller, profile.canvas, profile.clear_button, sign_rows=320
            ),
        )
    )
    plan = _two_color_plan()
    progress: list[int] = []
    painter.set_callbacks(on_progress=lambda p: progress.append(p.completed_strokes))
    assert painter.start(
        plan,
        profile,
        _settings(apply_brush_size=True, verify_passes=0),
        start_stroke=5,
    )
    assert painter.wait(_t(10.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.progress.completed_strokes == 6
    assert painter.progress.total_strokes == 6
    # Progress never reads below the offset.
    assert progress and min(progress) >= 5

    # The clear control was never clicked and no probe went down: the only
    # canvas click is the one remaining stroke, at the last stroke's cell.
    clicks = [
        _position_at(controller, index)
        for index, event in enumerate(controller.events)
        if event.kind == "mouse_down"
    ]
    assert not any(profile.clear_button.contains(*click) for click in clicks)
    on_canvas = _canvas_clicks(controller, profile.canvas)
    assert len(on_canvas) == 1, on_canvas
    cell_width = profile.canvas.width / plan.width
    cell_height = profile.canvas.height / plan.height
    x, y = on_canvas[0]
    assert abs(x - (profile.canvas.left + 15.5 * cell_width)) < cell_width
    assert abs(y - (profile.canvas.top + 8.5 * cell_height)) < cell_height
    # The stored brush was typed once, for the group resumed into; the
    # probes a fresh job types first are absent.
    typed = _typed_values(controller)
    assert typed == ["10.5"], typed
    # The job measured nothing, so the profile's stored model stays as is.
    assert painter.measured_brush_size_model is None
    assert not controller.held_buttons


def test_resuming_at_the_end_paints_nothing_and_goes_straight_to_touch_up() -> None:
    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _calibrating_profile("Resumable")
    assert profile.canvas is not None and profile.clear_button is not None
    painter = _impatient(
        Painter(
            controller,
            screen_capture=_clearable_sign(
                controller, profile.canvas, profile.clear_button, sign_rows=320
            ),
        )
    )
    plan = _two_color_plan()
    assert painter.start(
        plan, profile, _settings(apply_brush_size=False, verify_passes=1), start_stroke=6
    )
    assert painter.wait(_t(10.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert painter.progress.completed_strokes == 6
    # No artwork stroke went down before the verification pass looked.
    assert _canvas_clicks(controller, profile.canvas) == []


def test_a_start_offset_past_the_plan_is_refused() -> None:
    painter = Painter(MockInputController())
    with pytest.raises(ValueError, match="start_stroke"):
        painter.configure(_two_color_plan(), _profile(), _settings(), start_stroke=7)
    with pytest.raises(ValueError, match="start_stroke"):
        painter.configure(_two_color_plan(), _profile(), _settings(), start_stroke=-1)
    painter.configure(_two_color_plan(), _profile(), _settings(), start_stroke=6)
    assert painter.progress.completed_strokes == 6


def test_a_resumed_job_without_stored_measurements_warns_and_paints_on(caplog) -> None:
    """No grid and no brush model on file: the job says so and carries on.

    The strokes are aimed by the calibration rectangle and the brush is
    whatever Rust has set - worse than a measured sign, far better than a
    probe through the picture.
    """

    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _calibrating_profile("Unmeasured")
    assert profile.canvas is not None and profile.clear_button is not None
    painter = _impatient(
        Painter(
            controller,
            screen_capture=_clearable_sign(
                controller, profile.canvas, profile.clear_button, sign_rows=320
            ),
        )
    )
    with caplog.at_level(logging.WARNING, logger="rust_painter.painter"):
        assert painter.start(
            _two_color_plan(),
            profile,
            _settings(apply_brush_size=True, verify_passes=0),
            start_stroke=2,
        )
        assert painter.wait(_t(10.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    warnings = [record.getMessage() for record in caplog.records]
    assert any("No texel grid is stored" in message for message in warnings), warnings
    assert any("No brush model is stored" in message for message in warnings), warnings
    assert _typed_values(controller) == []
    assert len(_canvas_clicks(controller, profile.canvas)) == 4


def test_a_resumed_job_estimates_only_the_work_ahead_of_it() -> None:
    """The first time-left is for the strokes left, not the whole plan.

    And the percent starts where the sign is: a job resumed two thirds in
    reports itself two thirds done before its first stroke, not at zero.
    """

    controller = MockInputController()
    profile = _profile()
    plan = _dot_plan(30)
    fresh = Painter(controller)
    fresh.configure(plan, profile, _settings(delay_between_strokes_seconds=0.1))
    resumed = Painter(MockInputController())
    resumed.configure(
        plan, profile, _settings(delay_between_strokes_seconds=0.1), start_stroke=20
    )
    whole = fresh._initial_estimate(fresh._job)
    part = resumed._initial_estimate(resumed._job)
    assert whole is not None and part is not None
    assert 0 < part < whole
    assert part == pytest.approx(whole / 3, rel=0.2)

    seen: list[tuple[int, float]] = []
    resumed.set_callbacks(
        on_progress=lambda p: seen.append((p.completed_strokes, p.percent))
    )
    assert resumed.start()
    assert resumed.wait(_t(10.0))
    assert resumed.state is PainterState.COMPLETED
    assert seen[0][0] == 20
    painting = [percent for strokes, percent in seen if 20 < strokes < 30]
    assert painting and min(painting) > 60.0


def test_seconds_until_anti_afk_counts_down_from_the_job_start_and_the_last_break() -> None:
    """The readout knows when the next break is, and when there is none."""

    input_controller = MockInputController()
    painter = Painter(input_controller)
    assert painter.seconds_until_anti_afk() is None

    profile = _profile()
    profile.save_button = ScreenRect(600, 300, 100, 30)
    painter.start(
        _dot_plan(30),
        profile,
        _settings(
            mouse_down_duration_seconds=0.004,
            delay_between_strokes_seconds=0.02,
            anti_afk_enabled=True,
            anti_afk_interval_seconds=60.0,
        ),
    )
    assert _wait_until(lambda: painter.progress.completed_strokes >= 1)
    until = painter.seconds_until_anti_afk()
    assert until is not None and 55.0 < until <= 60.0
    # The clock stops while the job is paused, and the time spent paused is
    # not owed to the break once the job resumes.
    assert painter.pause()
    paused = painter.seconds_until_anti_afk()
    time.sleep(0.2)
    assert painter.seconds_until_anti_afk() == paused
    assert painter.resume()
    resumed = painter.seconds_until_anti_afk()
    assert resumed is not None and paused is not None and resumed > paused - 0.1
    assert painter.wait(_t(5.0))
    assert painter.state is PainterState.COMPLETED
    assert painter.seconds_until_anti_afk() is None

    # With the option off there is nothing to count down to.
    painter.start(_dot_plan(3), profile, _settings(anti_afk_enabled=False))
    assert painter.seconds_until_anti_afk() is None
    assert painter.wait(_t(5.0))


# ------------------------------------------------ checking colors as they land


_BARE = (96, 96, 96)


def _forgetful_sign(
    controller: MockInputController,
    canvas: ScreenRect,
    hue_bar: ScreenRect,
    plan: PaintPlan,
    *,
    drop_every: int,
):
    """A fake sign that never registers every ``drop_every``-th press.

    The game, on a large sign, samples its input so slowly that a press can
    fall entirely between two samples and paint nothing; this sign does the
    same on a schedule.  A press that does register paints the cell under
    it and every cell the cursor crosses before the release.  Replaying the
    recorded events on each capture keeps it honest about what the painter
    actually did, repaints included; the color a press paints is the plan's
    next color each time the hue bar is clicked.  Returns the capture and a
    function giving the cells currently painted.
    """

    cell_w = canvas.width / plan.width
    cell_h = canvas.height / plan.height
    colors = [group.color for group in plan.color_groups]

    def cell_of(position: tuple[int, int]) -> tuple[int, int] | None:
        if not canvas.contains(*position):
            return None
        return (
            int((position[0] - canvas.left) // cell_w),
            int((position[1] - canvas.top) // cell_h),
        )

    def painted() -> dict[tuple[int, int], tuple[int, int, int]]:
        cells: dict[tuple[int, int], tuple[int, int, int]] = {}
        position = (0, 0)
        presses = 0
        pressed_on_canvas = False
        registered = False
        color_index = -1
        for event in controller.events:
            if event.kind == "move" and event.x is not None and event.y is not None:
                position = (event.x, event.y)
                if pressed_on_canvas and registered:
                    cell = cell_of(position)
                    if cell is not None:
                        cells[cell] = colors[color_index]
            elif event.kind == "mouse_down":
                cell = cell_of(position)
                if cell is None:
                    # A picker click; the hue bar is visited once per color.
                    pressed_on_canvas = False
                    if hue_bar.contains(*position):
                        color_index = min(color_index + 1, len(colors) - 1)
                    continue
                presses += 1
                pressed_on_canvas = True
                registered = presses % drop_every != 0
                if registered:
                    cells[cell] = colors[color_index]
            elif event.kind == "mouse_up":
                pressed_on_canvas = False
        return cells

    def capture(rect) -> Image.Image:
        if (rect.left, rect.top) != (canvas.left, canvas.top):
            return Image.new("RGB", (rect.width, rect.height), (21, 21, 12))
        image = Image.new("RGB", (rect.width, rect.height), _BARE)
        draw = ImageDraw.Draw(image)
        for (x, y), color in painted().items():
            draw.rectangle(
                (
                    round(x * cell_w),
                    round(y * cell_h),
                    round((x + 1) * cell_w) - 1,
                    round((y + 1) * cell_h) - 1,
                ),
                fill=color,
            )
        return image

    return capture, painted


def _confirmation_plan() -> PaintPlan:
    red, blue = (220, 40, 20), (20, 60, 220)
    # Two long rows of one color, then forty dabs of another: the dabs are
    # what the game drops most, and what a lost press costs a whole cell.
    return PaintPlan(
        20,
        4,
        (
            ColorGroup(red, (Stroke(0, 0, 19, 0), Stroke(0, 1, 19, 1)), 40),
            ColorGroup(
                blue, tuple(Stroke(x, y, x, y) for y in (2, 3) for x in range(20)), 40
            ),
        ),
    )


def _run_forgetful_sign(*, confirm: bool, drop_every: int = 3):
    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile(canvas_width=400)  # 400x80 canvas: 20x4 cells of 20 px
    canvas, hue_bar = profile.canvas, profile.hue_bar
    plan = _confirmation_plan()
    capture, painted = _forgetful_sign(
        controller, canvas, hue_bar, plan, drop_every=drop_every
    )
    painter = _impatient(Painter(controller, screen_capture=capture))
    painter._MIN_PRESS_SECONDS = 0.0  # type: ignore[misc]
    settings = _settings(
        confirm_strokes=confirm,
        confirm_max_rounds=4,
        verify_passes=0,
        require_foreground=False,
        pause_on_mouse_move=False,
    )
    assert painter.start(plan, profile, settings)
    assert painter.wait(_t(60.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    return painter, painted, plan


def test_a_color_that_came_out_with_misses_is_repainted_before_the_next() -> None:
    painter, painted, plan = _run_forgetful_sign(confirm=True)
    cells = painted()
    expected = {
        (x, y): group.color
        for group in plan.color_groups
        for stroke in group.strokes
        for y in (stroke.start_y,)
        for x in range(stroke.start_x, stroke.end_x + 1)
    }
    missing = {cell for cell in expected if cells.get(cell) != expected[cell]}
    assert not missing, sorted(missing)
    summary = painter.confirmation_summary
    assert summary.colors == 2
    assert summary.missed > 0
    assert summary.repainted_strokes > 0
    assert summary.unrepaired == 0
    assert summary.skipped_reason == ""


def test_the_checks_are_clocked_apart_from_the_painting() -> None:
    """What the run measured is what the next estimate learns from: how
    many colors had a capture, what the captures cost, and what the
    repaints on top of them did."""

    painter, _painted, _plan = _run_forgetful_sign(confirm=True)
    measured = painter.paint_phase_timing
    assert measured is not None
    assert measured.colors_checked == 2
    assert measured.check_capture_seconds > 0.0
    # Both colors missed something and were repainted, so the checking
    # clock holds more than the two captures.
    assert measured.checking_seconds > measured.check_capture_seconds
    assert measured.checking_seconds <= measured.actual_seconds
    # No touch-up was asked for, so nothing is claimed about one.
    assert painter.touch_up_timing is None


def test_without_the_check_the_dropped_presses_stay_holes() -> None:
    painter, painted, plan = _run_forgetful_sign(confirm=False)
    cells = painted()
    planned = sum(len(group.strokes) for group in plan.color_groups)
    assert len(cells) < 80  # the two rows and the dabs do not all land
    assert painter.confirmation_summary.skipped_reason == "turned off"
    assert painter.confirmation_summary.colors == 0
    # Exactly the planned presses went down: nothing was repainted.
    presses = sum(1 for event in painter.input.events if event.kind == "mouse_down")
    assert presses >= planned


# ------------------------------------------------- reading color picks back


def _picky_sign(
    controller: MockInputController,
    profile: CalibrationProfile,
    plan: PaintPlan,
    *,
    swallow_sv,
):
    """A fake sign and color panel whose picker can swallow clicks.

    The selected color is whatever the last hue-bar click and the last
    saturation / value click that *registered* say it is, exactly as
    Rust's is; ``swallow_sv(n)`` decides whether the n-th S/V click (from
    1) is ignored.  The panel draws the selected color as a flat block
    right of the hue bar, as Rust does, and every press on the sign paints
    its cell in the selected color.  Returns the capture and a function
    giving the cells painted.
    """

    from app.color_mapping import picker_points_to_rgb

    canvas, hue_bar, color_box = profile.canvas, profile.hue_bar, profile.color_box
    cell_w = canvas.width / plan.width
    cell_h = canvas.height / plan.height
    panel = (21, 21, 12)
    swatch = ScreenRect(
        hue_bar.left + hue_bar.width + 2,
        hue_bar.top,
        int(hue_bar.width * 1.75),
        int(hue_bar.height * 0.75),
    )

    def selected_color(hue_point, sv_point):
        return picker_points_to_rgb(
            hue_point,
            sv_point,
            hue_bar,
            color_box,
            hue_direction="bottom_to_top",
            saturation_direction="left_low",
            value_direction="top_bright",
        )

    def replay():
        position = (0, 0)
        hue_point = (hue_bar.left, hue_bar.top)
        sv_point = (color_box.left, color_box.top)
        sv_clicks = 0
        cells: dict[tuple[int, int], tuple[int, int, int]] = {}
        for event in controller.events:
            if event.kind == "move" and event.x is not None and event.y is not None:
                position = (event.x, event.y)
            elif event.kind == "mouse_down":
                if hue_bar.contains(*position):
                    hue_point = position
                elif color_box.contains(*position):
                    sv_clicks += 1
                    if not swallow_sv(sv_clicks):
                        sv_point = position
                elif canvas.contains(*position):
                    cell = (
                        int((position[0] - canvas.left) // cell_w),
                        int((position[1] - canvas.top) // cell_h),
                    )
                    cells[cell] = selected_color(hue_point, sv_point)
        return cells, selected_color(hue_point, sv_point)

    def painted() -> dict[tuple[int, int], tuple[int, int, int]]:
        return replay()[0]

    def capture(rect) -> Image.Image:
        cells, selected = replay()
        image = Image.new("RGB", (rect.width, rect.height), panel)
        draw = ImageDraw.Draw(image)

        def box(r: ScreenRect):
            return (
                r.left - rect.left,
                r.top - rect.top,
                r.left + r.width - 1 - rect.left,
                r.top + r.height - 1 - rect.top,
            )

        draw.rectangle(box(canvas), fill=_BARE)
        for (x, y), color in cells.items():
            draw.rectangle(
                box(
                    ScreenRect(
                        canvas.left + round(x * cell_w),
                        canvas.top + round(y * cell_h),
                        round(cell_w),
                        round(cell_h),
                    )
                ),
                fill=color,
            )
        draw.rectangle(box(swatch), fill=selected)
        return image

    return capture, painted


def _three_color_plan() -> PaintPlan:
    # Three colors that differ in saturation and value as well as hue, so a
    # swallowed S/V click changes what goes down.
    colors = ((220, 40, 20), (40, 100, 200), (30, 200, 60))
    return PaintPlan(
        20,
        3,
        tuple(
            ColorGroup(color, tuple(Stroke(x, y, x, y) for x in range(20)), 20)
            for y, color in enumerate(colors)
        ),
    )


def _run_picky_sign(*, swallow_sv):
    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile(canvas_width=400)
    plan = _three_color_plan()
    capture, painted = _picky_sign(controller, profile, plan, swallow_sv=swallow_sv)
    painter = _impatient(Painter(controller, screen_capture=capture))
    painter._MIN_PRESS_SECONDS = 0.0  # type: ignore[misc]
    painter._SWATCH_RECHECK_SECONDS = 0.0  # type: ignore[misc]
    settings = _settings(
        verify_color_picks=True,
        verify_passes=0,
        require_foreground=False,
        pause_on_mouse_move=False,
    )
    assert painter.start(plan, profile, settings)
    return painter, painted, plan


def _painted_as_planned(cells, plan: PaintPlan) -> bool:
    """Every planned cell painted, each within the test picker's quantisation
    of its own color - a 100 px box rounds a channel by a few units."""

    expected = {
        (stroke.start_x, stroke.start_y): group.color
        for group in plan.color_groups
        for stroke in group.strokes
    }
    if set(cells) != set(expected):
        return False
    return all(
        max(abs(a - b) for a, b in zip(cells[cell], color)) <= 8
        for cell, color in expected.items()
    )


def test_every_color_is_read_back_off_the_panel_when_the_clicks_land() -> None:
    painter, painted, plan = _run_picky_sign(swallow_sv=lambda n: False)
    assert painter.wait(_t(60.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert _painted_as_planned(painted(), plan), painted()
    summary = painter.color_pick_summary
    assert summary.skipped_reason == ""
    assert summary.picks == 3
    assert summary.retried == 0
    assert summary.failed == 0


def test_a_swallowed_picker_click_is_clicked_again_until_the_panel_agrees() -> None:
    """The locator color's S/V click is the first; the second color's is
    the third.  Swallowed, it would paint that whole row in the first
    color."""

    painter, painted, plan = _run_picky_sign(swallow_sv=lambda n: n == 3)
    assert painter.wait(_t(60.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert _painted_as_planned(painted(), plan), painted()
    summary = painter.color_pick_summary
    assert summary.picks == 3
    assert summary.retried == 1
    assert summary.failed == 0


def test_a_color_the_panel_never_shows_pauses_the_job_for_the_user() -> None:
    """When the clicks keep failing the job stops before a stroke goes
    down wrong, says which color, and picks it again once resumed."""

    stuck = {"on": True}

    def swallow(n: int) -> bool:
        # Every S/V click from the second color's first one on, until released.
        return stuck["on"] and n >= 3

    painter, painted, plan = _run_picky_sign(swallow_sv=swallow)
    deadline = time.monotonic() + _t(60.0)
    while painter.state is not PainterState.PAUSED and time.monotonic() < deadline:
        time.sleep(0.01)
    assert painter.state is PainterState.PAUSED, painter.state
    reason = painter.state_reason
    assert "did not take #" in reason, reason
    shown = reason.split("did not take #")[1][:6]
    asked = tuple(int(shown[i : i + 2], 16) for i in (0, 2, 4))
    assert max(abs(a - b) for a, b in zip(asked, (40, 100, 200))) <= 8, reason
    # Nothing of the second color went down in the first color's paint.
    assert all(cell[1] == 0 for cell in painted())
    assert painter.color_pick_summary.failed == 1
    stuck["on"] = False
    assert painter.resume()
    assert painter.wait(_t(60.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    assert _painted_as_planned(painted(), plan), painted()


def test_picks_are_not_read_back_when_the_panel_has_no_color_block() -> None:
    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    profile = _profile(canvas_width=400)
    plan = _three_color_plan()
    painter = _impatient(Painter(controller, screen_capture=_panel_capture))
    painter._MIN_PRESS_SECONDS = 0.0  # type: ignore[misc]
    painter._SWATCH_RECHECK_SECONDS = 0.0  # type: ignore[misc]
    settings = _settings(
        verify_color_picks=True,
        verify_passes=0,
        require_foreground=False,
        pause_on_mouse_move=False,
    )
    assert painter.start(plan, profile, settings)
    assert painter.wait(_t(60.0))
    assert painter.state is PainterState.COMPLETED, painter.state_reason
    summary = painter.color_pick_summary
    assert summary.skipped_reason == "no color block found beside the hue bar"
    assert summary.picks == 0


def test_picks_are_not_read_back_when_turned_off() -> None:
    controller = MockInputController()
    controller.emits_real_input = True  # type: ignore[misc]
    painter = _impatient(Painter(controller, screen_capture=_panel_capture))
    painter._MIN_PRESS_SECONDS = 0.0  # type: ignore[misc]
    settings = _settings(
        verify_color_picks=False,
        verify_passes=0,
        require_foreground=False,
        pause_on_mouse_move=False,
    )
    assert painter.start(_three_color_plan(), _profile(), settings)
    assert painter.wait(_t(60.0))
    assert painter.state is PainterState.COMPLETED
    assert painter.color_pick_summary.skipped_reason == "turned off"
