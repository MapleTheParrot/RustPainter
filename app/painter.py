"""Threaded, interruptible paint-plan execution with conservative safety checks."""

from __future__ import annotations

import contextlib
import ctypes
import logging
import math
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Callable, Iterator

from .brush_calibration import (
    BRUSH_SIZE_MAX,
    BRUSH_SIZE_MIN,
    BrushSizeModel,
    StrokeBand,
    fit_brush_size_model,
    measure_stroke_band,
)
from .color_calibration import ColorCorrectionModel
from .color_mapping import map_rgb_to_picker
from .coordinates import RectangleLike, clamp_to_rect, logical_stroke_to_screen, normalized_point
from .input_controller import InputController, MouseButton
from .models import PaintPlan, RGBColor, ScreenRect
from .picker_calibration import trim_to_widget
from .screen import (
    ForegroundRequirement,
    VirtualScreen,
    capture_region,
    foreground_window_matches,
    get_virtual_screen,
)


LOGGER = logging.getLogger("rust_painter.painter")

# ``SendInput`` only queues a move, so ``GetCursorPos`` can still report a point
# the painter commanded a few events ago. Comparing a sample against a short
# history of commanded points, rather than only the newest one, keeps that
# ordinary lag from reading as a person grabbing the mouse.
_COMMANDED_POINT_HISTORY = 8

# A hand on the mouse crosses the pause threshold in a few dozen milliseconds.
# Anything still accumulating after this long is rounding noise, and its total
# is discarded rather than allowed to creep into a pause.
_MOUSE_DRIFT_WINDOW_SECONDS = 0.6


@contextlib.contextmanager
def _high_resolution_timer() -> Iterator[None]:
    """Stop the OS from distorting paint timing, per platform.

    Windows: default timer granularity rounds short waits up to ~15.6 ms,
    which silently inflates every configured inter-stroke and interpolation
    delay, so request 1 ms resolution.

    macOS: App Nap throttles timers and background threads once an app stops
    being frontmost. Painting *requires* the game to be frontmost instead, so
    without opting out the job would be throttled for its entire duration.
    """

    acquired = False
    activity = None
    if os.name == "nt":
        try:
            acquired = ctypes.WinDLL("winmm").timeBeginPeriod(1) == 0
        except (AttributeError, OSError):
            acquired = False
    elif sys.platform == "darwin":
        try:
            import Foundation

            # Keeps timers running at full rate while the app is in the
            # background. It does not keep the display awake. The literal is
            # Apple's documented value, used only if the binding is absent.
            options = getattr(Foundation, "NSActivityUserInitiated", 0x00FFFFFF)
            activity = Foundation.NSProcessInfo.processInfo().beginActivityWithOptions_reason_(
                options, "Painting a sign"
            )
        except Exception:
            LOGGER.warning("Could not opt out of App Nap; painting may be throttled")
            activity = None
    try:
        yield
    finally:
        if acquired:
            try:
                ctypes.WinDLL("winmm").timeEndPeriod(1)
            except (AttributeError, OSError):
                LOGGER.warning("Could not restore the Windows timer resolution")
        if activity is not None:
            try:
                import Foundation

                Foundation.NSProcessInfo.processInfo().endActivity_(activity)
            except Exception:
                LOGGER.warning("Could not end the App Nap exemption")


def _foreground_failure_reason(settings: "PainterSettings") -> str:
    """Explain a foreground-guard failure the user can actually act on."""

    name = str(settings.expected_process_name or "").strip()
    if name and os.name != "nt" and name.casefold().endswith(".exe"):
        return (
            f"foreground guard can never match the Windows process name "
            f"{name!r} on this platform - clear it under Settings > Safety"
        )
    return "foreground window lost"


class PainterState(str, Enum):
    IDLE = "idle"
    READY = "ready"
    COUNTDOWN = "countdown"
    RUNNING = "running"
    PAUSED = "paused"
    ABORTED = "aborted"
    COMPLETED = "completed"
    ERROR = "error"


@dataclass(frozen=True, slots=True)
class PickerDirections:
    hue: str = "bottom_to_top"
    saturation: str = "left_low"
    value: str = "top_bright"


@dataclass(frozen=True, slots=True)
class PaintingTarget:
    canvas: RectangleLike
    color_box: RectangleLike
    hue_bar: RectangleLike
    brush_size_box: RectangleLike | None = None
    picker_directions: PickerDirections = PickerDirections()
    color_correction: ColorCorrectionModel | None = None
    brush_size_model: BrushSizeModel | None = None

    @classmethod
    def from_profile(cls, profile: object) -> "PaintingTarget":
        canvas = getattr(profile, "canvas", None)
        color_box = getattr(profile, "color_box", None)
        hue_bar = getattr(profile, "hue_bar", None)
        if canvas is None or color_box is None or hue_bar is None:
            raise ValueError("Profile needs canvas, color box, and hue bar calibration")

        metadata = getattr(profile, "metadata", {})
        correction_value = (
            metadata.get("color_correction") if isinstance(metadata, Mapping) else None
        )
        correction = (
            ColorCorrectionModel.from_dict(correction_value)
            if isinstance(correction_value, Mapping)
            else None
        )
        sizing_value = (
            metadata.get("brush_size_model") if isinstance(metadata, Mapping) else None
        )
        brush_size_model = (
            BrushSizeModel.from_dict(sizing_value)
            if isinstance(sizing_value, Mapping)
            else None
        )
        return cls(
            canvas=canvas,
            color_box=color_box,
            hue_bar=hue_bar,
            brush_size_box=getattr(profile, "brush_size_box", None),
            picker_directions=PickerDirections(
                hue="bottom_to_top",
                saturation="left_low",
                value="top_bright",
            ),
            color_correction=correction,
            brush_size_model=brush_size_model,
        )


@dataclass(frozen=True, slots=True)
class PainterSettings:
    """All timing and safety values that may need in-game tuning."""

    stroke_speed_pixels_per_second: float = 700.0
    mouse_down_duration_seconds: float = 0.025
    delay_after_hue_seconds: float = 0.06
    delay_after_saturation_value_seconds: float = 0.06
    delay_between_strokes_seconds: float = 0.025
    delay_between_colors_seconds: float = 0.10
    stroke_interpolation_step_pixels: float = 3.0
    logical_pixel_spacing: float = 1.0
    brush_size: float = 1.0
    apply_brush_size: bool = False
    # After painting, capture the canvas and repaint decisively wrong cells,
    # up to this many correction passes. Zero disables verification.
    verify_passes: int = 1
    brush_direction: str = "low_to_high"
    delay_after_brush_seconds: float = 0.06
    countdown_seconds: float = 3.0
    require_foreground: bool = False
    expected_window_title_contains: str | None = "Rust"
    expected_process_name: str | None = "RustClient.exe"
    focus_check_interval_seconds: float = 0.05
    corner_abort_enabled: bool = True
    corner_abort_margin_pixels: int = 3
    corner_abort_minimum_distance_pixels: float = 80.0
    pause_on_mouse_move: bool = True
    mouse_move_pause_threshold_pixels: float = 24.0
    mouse_move_tolerance_pixels: float = 3.0
    safety_poll_interval_seconds: float = 0.01
    progress_callback_interval_seconds: float = 0.04

    def __post_init__(self) -> None:
        positive = {
            "stroke_speed_pixels_per_second": self.stroke_speed_pixels_per_second,
            "stroke_interpolation_step_pixels": self.stroke_interpolation_step_pixels,
            "logical_pixel_spacing": self.logical_pixel_spacing,
            "focus_check_interval_seconds": self.focus_check_interval_seconds,
            "safety_poll_interval_seconds": self.safety_poll_interval_seconds,
            "mouse_move_pause_threshold_pixels": self.mouse_move_pause_threshold_pixels,
        }
        for name, value in positive.items():
            if value <= 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite positive number")
        nonnegative = {
            "mouse_down_duration_seconds": self.mouse_down_duration_seconds,
            "delay_after_hue_seconds": self.delay_after_hue_seconds,
            "delay_after_saturation_value_seconds": self.delay_after_saturation_value_seconds,
            "delay_between_strokes_seconds": self.delay_between_strokes_seconds,
            "delay_between_colors_seconds": self.delay_between_colors_seconds,
            "delay_after_brush_seconds": self.delay_after_brush_seconds,
            "countdown_seconds": self.countdown_seconds,
            "corner_abort_minimum_distance_pixels": self.corner_abort_minimum_distance_pixels,
            "mouse_move_tolerance_pixels": self.mouse_move_tolerance_pixels,
            "progress_callback_interval_seconds": self.progress_callback_interval_seconds,
        }
        for name, value in nonnegative.items():
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite non-negative number")
        if self.corner_abort_margin_pixels < 0:
            raise ValueError("corner_abort_margin_pixels cannot be negative")
        if isinstance(self.verify_passes, bool) or not isinstance(
            self.verify_passes, int
        ) or not 0 <= self.verify_passes <= 5:
            raise ValueError("verify_passes must be an integer between 0 and 5")
        if self.brush_direction not in {"low_to_high", "high_to_low"}:
            raise ValueError("brush_direction must be low_to_high or high_to_low")
        if not math.isfinite(self.brush_size) or not 0.0 <= self.brush_size <= 1.0:
            raise ValueError("brush_size must be a finite value between 0 and 1")
        if self.require_foreground and not (
            str(self.expected_window_title_contains or "").strip()
            or str(self.expected_process_name or "").strip()
        ):
            raise ValueError(
                "Foreground protection needs an expected window title or process name"
            )

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "PainterSettings":
        """Read either a flat mapping or the application's nested settings."""

        painting = values.get("painting", values)
        safety = values.get("safety", values)
        if not isinstance(painting, Mapping) or not isinstance(safety, Mapping):
            raise TypeError("Painter settings sections must be mappings")

        def pick(section: Mapping[str, Any], name: str, default: Any, *aliases: str) -> Any:
            for key in (name, *aliases):
                if key in section:
                    return section[key]
            return default

        return cls(
            stroke_speed_pixels_per_second=float(
                pick(painting, "stroke_speed_pixels_per_second", 700.0, "stroke_speed")
            ),
            mouse_down_duration_seconds=float(
                pick(painting, "mouse_down_duration_seconds", 0.025, "dot_duration_seconds")
            ),
            delay_after_hue_seconds=float(
                pick(painting, "delay_after_hue_seconds", 0.06)
            ),
            delay_after_saturation_value_seconds=float(
                pick(painting, "delay_after_saturation_value_seconds", 0.06, "delay_after_sv_seconds")
            ),
            delay_between_strokes_seconds=float(
                pick(painting, "delay_between_strokes_seconds", 0.025)
            ),
            delay_between_colors_seconds=float(
                pick(painting, "delay_between_colors_seconds", 0.10)
            ),
            stroke_interpolation_step_pixels=float(
                pick(painting, "stroke_interpolation_step_pixels", 3.0)
            ),
            logical_pixel_spacing=float(pick(painting, "logical_pixel_spacing", 1.0)),
            brush_size=float(pick(painting, "brush_size", 1.0)),
            apply_brush_size=bool(pick(painting, "apply_brush_size", False)),
            verify_passes=int(pick(painting, "verify_passes", 1)),
            brush_direction=str(pick(painting, "brush_direction", "low_to_high")),
            delay_after_brush_seconds=float(pick(painting, "delay_after_brush_seconds", 0.06)),
            countdown_seconds=float(pick(safety, "countdown_seconds", 3.0)),
            require_foreground=bool(
                pick(safety, "require_rust_foreground", False, "require_foreground")
            ),
            expected_window_title_contains=pick(
                safety, "expected_window_title_contains", "Rust"
            ),
            expected_process_name=pick(safety, "expected_process_name", "RustClient.exe"),
            focus_check_interval_seconds=float(
                pick(safety, "focus_check_interval_seconds", 0.05)
            ),
            corner_abort_enabled=bool(pick(safety, "corner_abort_enabled", True)),
            corner_abort_margin_pixels=int(
                pick(safety, "corner_abort_margin_pixels", 3)
            ),
            corner_abort_minimum_distance_pixels=float(
                pick(safety, "corner_abort_minimum_distance_pixels", 80.0)
            ),
            pause_on_mouse_move=bool(pick(safety, "pause_on_mouse_move", True)),
            mouse_move_pause_threshold_pixels=float(
                pick(safety, "mouse_move_pause_threshold_pixels", 24.0)
            ),
            mouse_move_tolerance_pixels=float(
                pick(safety, "mouse_move_tolerance_pixels", 3.0)
            ),
            safety_poll_interval_seconds=float(
                pick(safety, "safety_poll_interval_seconds", 0.01)
            ),
            progress_callback_interval_seconds=float(
                pick(values, "progress_callback_interval_seconds", 0.04)
            ),
        )


@dataclass(frozen=True, slots=True)
class PaintProgress:
    state: PainterState
    color_index: int
    total_colors: int
    stroke_index_in_color: int
    strokes_in_color: int
    completed_strokes: int
    total_strokes: int
    percent: float
    elapsed_seconds: float
    estimated_remaining_seconds: float | None
    message: str = ""

    @property
    def stroke_index(self) -> int:
        """Global completed-stroke index, useful for ``427 / 1840`` UI text."""

        return self.completed_strokes


ProgressCallback = Callable[[PaintProgress], None]
StateCallback = Callable[[PainterState, str], None]


@dataclass(slots=True)
class _Job:
    plan: PaintPlan
    target: PaintingTarget
    settings: PainterSettings
    # "paint" runs the plan; "measure_brush" paints probe strokes instead and
    # fits what Rust's Size numbers actually cover.
    mode: str = "paint"
    # Canvas-height fraction of one logical cell, so a measurement can place its
    # probes around the brush the plan will really ask for.
    cell_fraction: float | None = None


class _AbortRequested(Exception):
    pass


class _RetryAction(Exception):
    """A pause released the button; repeat the unfinished click/stroke."""


class Painter:
    """Execute one configured paint plan on a daemon worker thread.

    The class never calls Qt. Callback clients are responsible for bridging
    worker-thread notifications into their UI framework's main thread.
    """

    _ACTIVE_STATES = {PainterState.COUNTDOWN, PainterState.RUNNING, PainterState.PAUSED}

    def __init__(
        self,
        input_controller: InputController,
        *,
        on_progress: ProgressCallback | None = None,
        on_state_change: StateCallback | None = None,
        on_complete: Callable[[PaintProgress], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_countdown: Callable[[int], None] | None = None,
        foreground_checker: Callable[[ForegroundRequirement], bool] | None = None,
        virtual_screen_provider: Callable[[], VirtualScreen] | None = None,
        screen_capture: Callable[[RectangleLike], Any] | None = None,
    ) -> None:
        self.input = input_controller
        self._on_progress = on_progress
        self._on_state_change = on_state_change
        self._on_complete = on_complete
        self._on_error = on_error
        self._on_countdown = on_countdown
        self._foreground_checker = foreground_checker or (
            lambda requirement: foreground_window_matches(requirement)
        )
        self._virtual_screen_provider = virtual_screen_provider or get_virtual_screen
        self._screen_capture = screen_capture or capture_region

        self._condition = threading.Condition(threading.RLock())
        self._abort_event = threading.Event()
        self._pause_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._job: _Job | None = None
        self._state = PainterState.IDLE
        self._state_reason = ""
        self._state_before_pause = PainterState.RUNNING
        self._abort_requested = False
        self._pause_generation = 0
        self._started_at: float | None = None
        self._paused_at: float | None = None
        self._paused_seconds = 0.0
        self._last_focus_check = 0.0
        self._last_corner_check = 0.0
        self._last_commanded_point: tuple[int, int] | None = None
        self._commanded_history: deque[tuple[int, int]] = deque(
            maxlen=_COMMANDED_POINT_HISTORY
        )
        self._mouse_drift_pixels = 0.0
        self._mouse_drift_started = 0.0
        self._measured_brush_size_model: BrushSizeModel | None = None
        self._last_progress_emit = 0.0
        self._progress = PaintProgress(
            PainterState.IDLE, 0, 0, 0, 0, 0, 0, 0.0, 0.0, None
        )

    @property
    def state(self) -> PainterState:
        with self._condition:
            return self._state

    @property
    def state_reason(self) -> str:
        with self._condition:
            return self._state_reason

    @property
    def progress(self) -> PaintProgress:
        with self._condition:
            return self._progress

    @property
    def is_active(self) -> bool:
        return self.state in self._ACTIVE_STATES

    @property
    def is_alive(self) -> bool:
        with self._condition:
            return self._thread is not None and self._thread.is_alive()

    def set_callbacks(
        self,
        *,
        on_progress: ProgressCallback | None = None,
        on_state_change: StateCallback | None = None,
        on_complete: Callable[[PaintProgress], None] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
        on_countdown: Callable[[int], None] | None = None,
    ) -> None:
        with self._condition:
            self._on_progress = on_progress
            self._on_state_change = on_state_change
            self._on_complete = on_complete
            self._on_error = on_error
            self._on_countdown = on_countdown

    def configure(
        self,
        plan: PaintPlan,
        profile: object | None = None,
        settings: PainterSettings | Mapping[str, Any] | None = None,
        *,
        target: PaintingTarget | None = None,
        canvas: RectangleLike | None = None,
        color_box: RectangleLike | None = None,
        hue_bar: RectangleLike | None = None,
        brush_size_box: RectangleLike | None = None,
        brush_size_model: BrushSizeModel | None = None,
        picker_directions: PickerDirections | None = None,
    ) -> None:
        """Prepare a job without starting it, suitable for an F8 callback."""

        if target is None:
            if profile is not None:
                target = PaintingTarget.from_profile(profile)
            else:
                if canvas is None or color_box is None or hue_bar is None:
                    raise ValueError("Provide a calibrated profile or all required rectangles")
                target = PaintingTarget(
                    canvas=canvas,
                    color_box=color_box,
                    hue_bar=hue_bar,
                    brush_size_box=brush_size_box,
                    picker_directions=picker_directions or PickerDirections(),
                    brush_size_model=brush_size_model,
                )
        resolved_settings = (
            settings
            if isinstance(settings, PainterSettings)
            else PainterSettings.from_mapping(settings) if settings is not None else PainterSettings()
        )
        self._validate_job(plan, target, resolved_settings)
        total_strokes = sum(len(group.strokes) for group in plan.color_groups)
        with self._condition:
            if self._state in self._ACTIVE_STATES or (
                self._thread is not None and self._thread.is_alive()
            ):
                raise RuntimeError("Cannot replace a paint job while one is active")
            self._job = _Job(plan, target, resolved_settings)
            self._abort_requested = False
            self._abort_event.clear()
            self._pause_event.clear()
            self._pause_generation = 0
            self._state_before_pause = PainterState.RUNNING
            self._progress = PaintProgress(
                PainterState.READY,
                0,
                len(plan.color_groups),
                0,
                0,
                0,
                total_strokes,
                0.0,
                0.0,
                None,
                "Ready",
            )
        self._transition(PainterState.READY, "configured")
        self._emit_progress(force=True)

    prepare = configure

    def configure_brush_measurement(
        self,
        profile: object | None = None,
        settings: PainterSettings | Mapping[str, Any] | None = None,
        *,
        target: PaintingTarget | None = None,
        cell_fraction: float | None = None,
    ) -> None:
        """Prepare a job that measures Rust's Size numbers instead of painting.

        The job paints its own probe strokes on the calibrated sign, so it
        carries a placeholder plan purely to satisfy the shared machinery that
        every job runs through.
        """

        if target is None:
            if profile is None:
                raise ValueError("Provide a calibrated profile or a painting target")
            target = PaintingTarget.from_profile(profile)
        if target.brush_size_box is None:
            raise ValueError(
                "Calibrate Rust's Size value box before measuring what its numbers paint"
            )
        resolved_settings = (
            settings
            if isinstance(settings, PainterSettings)
            else PainterSettings.from_mapping(settings) if settings is not None else PainterSettings()
        )
        placeholder = PaintPlan(width=1, height=1, color_groups=())
        with self._condition:
            if self._state in self._ACTIVE_STATES or (
                self._thread is not None and self._thread.is_alive()
            ):
                raise RuntimeError("Cannot replace a paint job while one is active")
            self._job = _Job(
                placeholder,
                target,
                resolved_settings,
                mode="measure_brush",
                cell_fraction=cell_fraction,
            )
            self._measured_brush_size_model = None
            self._abort_requested = False
            self._abort_event.clear()
            self._pause_event.clear()
            self._pause_generation = 0
            self._state_before_pause = PainterState.RUNNING
            self._progress = PaintProgress(
                PainterState.READY, 0, 0, 0, 0, 0, 0, 0.0, 0.0, None, "Ready to measure"
            )
            self._transition(PainterState.READY, "configured")

    @property
    def measured_brush_size_model(self) -> BrushSizeModel | None:
        """The model fitted by the last completed measurement job, if any."""

        with self._condition:
            return self._measured_brush_size_model

    def start(
        self,
        plan: PaintPlan | None = None,
        profile: object | None = None,
        settings: PainterSettings | Mapping[str, Any] | None = None,
        **target_kwargs: Any,
    ) -> bool:
        """Start the configured job, or configure and start in one call."""

        if plan is not None:
            self.configure(plan, profile, settings, **target_kwargs)
        with self._condition:
            if self._job is None:
                raise RuntimeError("No paint job has been configured")
            if self._state == PainterState.PAUSED:
                return self.resume()
            # An emergency stop of a READY job is terminal until the caller
            # explicitly configures it again. This closes the publish/start
            # race used by GUI global-hotkey cancellation.
            if self._state == PainterState.ABORTED and self._abort_requested:
                return False
            if self._state in {PainterState.COUNTDOWN, PainterState.RUNNING}:
                return False
            if self._thread is not None and self._thread.is_alive():
                return False
            self._abort_requested = False
            self._abort_event.clear()
            self._pause_event.clear()
            self._started_at = time.monotonic()
            self._paused_at = None
            self._paused_seconds = 0.0
            self._last_focus_check = 0.0
            self._last_corner_check = 0.0
            self._reset_mouse_movement_baseline()
            self._last_progress_emit = 0.0
            total_strokes = sum(
                len(group.strokes) for group in self._job.plan.color_groups
            )
            self._progress = PaintProgress(
                self._state,
                0,
                len(self._job.plan.color_groups),
                0,
                0,
                0,
                total_strokes,
                0.0,
                0.0,
                None,
                "Starting",
            )
            self._thread = threading.Thread(
                target=self._run,
                name="RustPainterWorker",
                daemon=True,
            )
            thread = self._thread
        LOGGER.info("Painting worker starting")
        thread.start()
        return True

    def start_or_resume(self) -> bool:
        if self.state == PainterState.PAUSED:
            return self.resume()
        return self.start()

    def pause(self, reason: str = "user") -> bool:
        # Publish the request before waiting for the input gate.  This prevents
        # the worker from winning the lock again and emitting a new event after
        # the user has requested a pause.
        self._pause_event.set()
        with self._condition:
            if self._state not in {PainterState.COUNTDOWN, PainterState.RUNNING}:
                if self._state is not PainterState.PAUSED:
                    self._pause_event.clear()
                return False
            self._state_before_pause = self._state
            self._pause_generation += 1
            self._paused_at = time.monotonic()
            self._state = PainterState.PAUSED
            self._state_reason = reason
            self._condition.notify_all()
        self._safe_release_all()
        LOGGER.info("Painting paused: %s", reason)
        self._emit_state(PainterState.PAUSED, reason)
        self._update_progress_state(PainterState.PAUSED, f"Paused: {reason}")
        return True

    def resume(self) -> bool:
        with self._condition:
            if self._state != PainterState.PAUSED or self._abort_requested:
                return False
            now = time.monotonic()
            if self._paused_at is not None:
                self._paused_seconds += now - self._paused_at
            self._paused_at = None
            resumed_state = self._state_before_pause
            self._state = resumed_state
            self._state_reason = "resumed"
            # Force the worker to verify focus again before its very next input.
            self._last_focus_check = 0.0
            # The cursor is wherever the user left it, so there is no commanded
            # point it should currently match. Without re-baselining, the first
            # sample after resuming would read as movement and pause again.
            self._reset_mouse_movement_baseline()
            self._pause_event.clear()
            self._condition.notify_all()
        LOGGER.info("Painting resumed")
        self._emit_state(resumed_state, "resumed")
        message = "Countdown resumed" if resumed_state is PainterState.COUNTDOWN else "Painting"
        self._update_progress_state(resumed_state, message)
        return True

    def abort(self, reason: str = "user") -> bool:
        # Set the event before acquiring the shared input gate so no worker can
        # slip another move/down event in while this thread waits for the lock.
        self._abort_event.set()
        with self._condition:
            if self._state not in self._ACTIVE_STATES and self._state != PainterState.READY:
                self._abort_event.clear()
                return False
            self._abort_requested = True
            self._pause_event.clear()
            self._pause_generation += 1
            self._state = PainterState.ABORTED
            self._state_reason = reason
            self._condition.notify_all()
        # Releasing from the calling/hotkey thread makes emergency stop immediate,
        # even if the worker is between safety polls.
        self._safe_release_all()
        LOGGER.warning("Painting aborted: %s", reason)
        self._emit_state(PainterState.ABORTED, reason)
        self._update_progress_state(PainterState.ABORTED, f"Aborted: {reason}")
        return True

    def wait(self, timeout: float | None = None) -> bool:
        with self._condition:
            thread = self._thread
        if thread is None:
            return True
        if thread is threading.current_thread():
            return False
        thread.join(timeout)
        return not thread.is_alive()

    def shutdown(self, timeout: float = 2.0) -> None:
        self.abort("shutdown")
        self.wait(timeout)
        self._safe_release_all()

    close = shutdown

    def _validate_job(
        self,
        plan: PaintPlan,
        target: PaintingTarget,
        settings: PainterSettings,
    ) -> None:
        if plan.width <= 0 or plan.height <= 0:
            raise ValueError("Paint plan dimensions must be positive")
        for label, rect in (
            ("canvas", target.canvas),
            ("color box", target.color_box),
            ("hue bar", target.hue_bar),
        ):
            if rect.width <= 0 or rect.height <= 0:
                raise ValueError(f"{label} calibration must have positive dimensions")
        if settings.apply_brush_size:
            if target.brush_size_box is None:
                raise ValueError(
                    "Automatic brush sizing is enabled, but Rust's Size value box "
                    "is not calibrated"
                )
            if target.brush_size_box.width <= 0 or target.brush_size_box.height <= 0:
                raise ValueError("Size value box calibration must have positive dimensions")
            if target.brush_size_model is None:
                raise ValueError(
                    "Automatic brush sizing is enabled, but this profile has not "
                    "measured what Rust's Size numbers paint. Run Measure Brush Size."
                )
            self._validate_brush_reach(plan, target, settings, target.brush_size_model)
        # A dry run only visualizes the plan, so it may carry brush metadata
        # that real input could not honor with the current calibration.
        if getattr(self.input, "emits_real_input", True):
            if any(group.brush_diameter > 1 for group in plan.color_groups) and not (
                settings.apply_brush_size and target.brush_size_box
            ):
                raise ValueError(
                    "This plan uses multiple brush sizes, which needs automatic brush "
                    "sizing enabled with the Size value box calibrated"
                )
            if (
                any(group.brush_diameter > 1 for group in plan.color_groups)
                and settings.logical_pixel_spacing > 1.0
            ):
                # Spacing spreads stroke geometry while the brush target stays
                # capped at one unspaced cell, so multi-cell bands would leave
                # unpainted rows between their sweeps.
                raise ValueError(
                    "Multi-cell brush passes need Logical spacing at 1.0 or below"
                )

    # The plan treats a cell as final once its color's last stroke lands, so a
    # brush that spills half a cell each side starts destroying finished
    # neighbours.  Below this ratio the spill stays inside the seams the plan
    # already tolerates; above it the painted sign visibly diverges from the
    # preview, and refusing beats quietly painting the wrong image.
    _DETAIL_OVERSHOOT_LIMIT = 1.5

    def _validate_brush_reach(
        self,
        plan: PaintPlan,
        target: PaintingTarget,
        settings: PainterSettings,
        model: BrushSizeModel,
    ) -> None:
        """Refuse plans asking for footprints Rust's Size field cannot reach.

        Every number this job would type is known before a single stroke is
        painted, so an unreachable brush is reported while the sign is still
        blank rather than halfway through covering it.
        """

        canvas = target.canvas
        pitch = min(canvas.width / plan.width, canvas.height / plan.height)
        diameters = {max(1, int(group.brush_diameter)) for group in plan.color_groups}
        for diameter in sorted(diameters):
            wanted = self._brush_target_fraction(
                target, plan, diameter, settings.logical_pixel_spacing
            )
            size = model.clamped_size_for_fraction(wanted)
            painted = model.fraction_for_size(size) * canvas.height
            nominal = pitch * diameter
            smallest, largest = model.fitted_range
            if size * 2 < smallest or size > largest * 2:
                LOGGER.warning(
                    "Brush size %d for %d cell(s) sits outside the %d-%d range this "
                    "profile was measured over, so it is an extrapolation. Re-run "
                    "Measure Brush Size at this painting resolution.",
                    size,
                    diameter,
                    smallest,
                    largest,
                )
            if diameter == 1 and painted > nominal * self._DETAIL_OVERSHOOT_LIMIT:
                rows = max(1, int(model.sign_pixel_rows))
                raise ValueError(
                    f"Rust's smallest brush covers {painted / pitch:.1f} logical cells "
                    "here, so detail strokes would overwrite their neighbours. This "
                    f"sign is about {rows} rows tall - lower the painting resolution "
                    f"to {rows} rows or fewer, or calibrate a larger sign."
                )
            # Adjacent multi-cell bands overlap one row, which tolerates a brush
            # up to one cell under its nominal footprint; anything narrower
            # leaves stripes the plan already counts as covered.
            if diameter > 1 and painted < nominal * (diameter - 1) / diameter:
                raise ValueError(
                    f"A {diameter}-cell brush needs {nominal:.0f}px but the Size field "
                    f"reaches only {painted:.0f}px here. Choose a lower optimization "
                    "mode or a higher painting resolution."
                )

    def _run(self) -> None:
        if getattr(self.input, "emits_real_input", True):
            with _high_resolution_timer():
                self._run_job()
        else:
            self._run_job()

    def _run_job(self) -> None:
        final_progress: PaintProgress | None = None
        try:
            job = self._job
            if job is None:
                raise RuntimeError("Paint job disappeared before execution")
            if job.settings.countdown_seconds > 0:
                self._transition(PainterState.COUNTDOWN, "countdown")
                self._run_countdown(job.settings.countdown_seconds)
            self._enter_running_after_countdown()
            # RUNNING is set before the first guard so a zero-second countdown
            # can still enter the ordinary PAUSED state when focus is wrong.
            self._checkpoint(check_focus=True)
            job.target = self._measured_picker_target(job.target)
            if job.mode == "measure_brush":
                measured = self._measure_brush_size_model(job)
                with self._condition:
                    self._measured_brush_size_model = measured
            else:
                self._update_progress_state(PainterState.RUNNING, "Painting")
                self._execute_plan(job)
                self._verify_and_touch_up(job)
            self._checkpoint(check_focus=False)
            self._finish_completed()
            self._update_progress_state(PainterState.COMPLETED, "Completed")
            final_progress = self.progress
            if job.mode == "measure_brush":
                LOGGER.info("Brush measurement completed")
            else:
                LOGGER.info(
                    "Painting completed: %d strokes", final_progress.completed_strokes
                )
            self._safe_callback(self._on_complete, final_progress, label="completion")
        except _AbortRequested:
            if self.state != PainterState.ABORTED:
                self._transition(PainterState.ABORTED, "abort_requested")
                self._update_progress_state(PainterState.ABORTED, "Aborted")
        except BaseException as exc:
            if self.state is not PainterState.ABORTED:
                LOGGER.exception("Painting failed")
                self._transition(PainterState.ERROR, str(exc))
                self._update_progress_state(PainterState.ERROR, f"Error: {exc}")
                self._safe_callback(self._on_error, exc, label="error")
        finally:
            self._safe_release_all()
            with self._condition:
                self._paused_at = None
                self._condition.notify_all()

    def _run_countdown(self, seconds: float) -> None:
        remaining = seconds
        last_displayed: int | None = None
        while remaining > 0:
            displayed = max(1, int(math.ceil(remaining)))
            if displayed != last_displayed:
                last_displayed = displayed
                self._safe_callback(self._on_countdown, displayed, label="countdown")
                self._update_progress_state(PainterState.COUNTDOWN, f"Starting in {displayed}")
            slice_seconds = min(0.1, remaining)
            self._interruptible_sleep(slice_seconds, check_focus=False)
            # Paused time is spent inside the checkpoint and must not consume
            # the user's focus countdown.
            remaining -= slice_seconds

    @staticmethod
    def _brush_target_fraction(
        target: PaintingTarget,
        plan: PaintPlan,
        diameter_cells: int,
        spacing: float,
    ) -> float:
        """The canvas-height fraction a ``diameter_cells`` brush should paint.

        The plan is stretched across the whole canvas, so one row is
        ``1/height`` of it and one column ``1/width``.  A round or square brush
        spans the same distance both ways and therefore has to respect
        whichever pitch is tighter.  The result is expressed against the canvas
        height because that is the axis brush calibration measured.
        """

        canvas = target.canvas
        pitch = min(canvas.width / plan.width, canvas.height / plan.height)
        span = pitch * diameter_cells * min(spacing, 1.0)
        if diameter_cells <= 1:
            # Slightly underfilling a cell is safer than overwriting both
            # adjacent rows. Rust's sign texture visually hides the small seam.
            span *= 0.90
        return span / canvas.height

    def _apply_brush_size(self, job: _Job, diameter_cells: int, epoch: int) -> None:
        """Type the Size number that paints ``diameter_cells`` logical cells.

        Rust sizes the brush in the sign's own texture pixels, so the profile's
        measured model turns a wanted footprint straight into a number.  There
        is nothing to search for and nothing to capture: the value is computed,
        typed, and committed.  ``_validate_job`` has already confirmed the plan
        only asks for footprints the field can actually reach.
        """

        settings = job.settings
        box = job.target.brush_size_box
        model = job.target.brush_size_model
        if not settings.apply_brush_size or box is None or model is None:
            return
        fraction = self._brush_target_fraction(
            job.target, job.plan, diameter_cells, settings.logical_pixel_spacing
        )
        size = model.clamped_size_for_fraction(fraction)
        self._update_progress_state(
            PainterState.RUNNING,
            f"Brush size {size} for {diameter_cells} logical "
            f"cell{'s' if diameter_cells != 1 else ''}",
        )
        self._write_brush_size(box, size, settings, epoch)
        LOGGER.info(
            "Brush size %d typed for %d cell(s): wanted %.5f of the sign, paints %.5f",
            size,
            diameter_cells,
            fraction,
            model.fraction_for_size(size),
        )

    def _write_brush_size(
        self,
        box: RectangleLike,
        size: int,
        settings: PainterSettings,
        epoch: int,
    ) -> None:
        """Focus Rust's Size field, replace its contents, and commit with Enter."""

        settle = (
            max(settings.delay_after_brush_seconds, 0.05)
            if self.input.emits_real_input
            else 0.0
        )
        self._safe_click(normalized_point(box, 0.5, 0.5), epoch)
        self._interruptible_sleep(settle, epoch=epoch, check_focus=True)
        # The field holds at most three digits, and clearing from both sides of
        # the caret empties it wherever the click happened to place it.
        for key in ("BACKSPACE",) * 4 + ("DELETE",) * 4:
            self._checkpoint(epoch=epoch, check_focus=True)
            self.input.press_key(key)
        for digit in str(size):
            self._checkpoint(epoch=epoch, check_focus=True)
            self.input.press_key(digit)
        self._checkpoint(epoch=epoch, check_focus=True)
        self.input.press_key("ENTER")
        self._interruptible_sleep(settle, epoch=epoch, check_focus=True)

    # A first, throwaway stroke used only to learn the scale of the sign, so
    # the real probes can be placed around the brush the plan will ask for.
    _BRUSH_SCOUT_SIZE = 24

    # Probes are placed at these multiples of the brush one logical cell needs,
    # which brackets the working size from both sides.  Fitting a line and then
    # reading it far below its data is what made an earlier version answer 2
    # where 5 was wanted: a couple of pixels of error is nothing at size 60 and
    # is the entire answer at size 2.
    _BRUSH_PROBE_MULTIPLES = (4.0, 2.0, 1.0, 0.5)

    # Fallback ladder for a measurement with no resolution to aim at.
    _BRUSH_FALLBACK_SIZES = (32, 16, 8, 4)

    # One color per probe, so a stroke drawn inside a previous, wider band still
    # reads as a change against the capture taken just before it.
    _BRUSH_PROBE_COLORS = (
        (255, 0, 255),
        (0, 255, 0),
        (255, 200, 0),
        (0, 200, 255),
        (255, 80, 80),
    )

    def _measure_brush_size_model(self, job: _Job) -> BrushSizeModel:
        """Paint probe strokes and fit Size number to painted canvas fraction.

        Each probe types a number, drags one stroke through the middle of the
        sign, and measures the band it left behind.  Reading the sign itself is
        what makes the result independent of Rust's preview tile and of how
        close the camera happens to be standing.
        """

        target = job.target
        settings = job.settings
        box = target.brush_size_box
        if box is None:
            raise RuntimeError("Rust's Size value box is not calibrated")
        canvas = ScreenRect(
            target.canvas.left,
            target.canvas.top,
            target.canvas.width,
            target.canvas.height,
        )
        # Parked over the color box, the cursor cannot shadow the capture.
        park = (
            int(round(target.color_box.left + target.color_box.width / 2.0)),
            int(round(target.color_box.top + target.color_box.height / 2.0)),
        )
        # Drawn through the vertical middle and well short of both sides, so the
        # straight section of the stroke dominates the measurement even when the
        # widest probe rounds off its ends.
        stroke_y = int(round(canvas.top + canvas.height / 2.0))
        start = (int(round(canvas.left + canvas.width * 0.15)), stroke_y)
        end = (int(round(canvas.left + canvas.width * 0.85)), stroke_y)

        samples: list[tuple[int, float]] = []
        clipped: list[int] = []
        probe_index = 0

        def probe(size: int, label: str) -> "StrokeBand | None":
            nonlocal probe_index
            epoch = self._pause_generation_value()
            self._update_progress_state(
                PainterState.RUNNING, f"Measuring brush size {size} ({label})"
            )
            color = self._BRUSH_PROBE_COLORS[probe_index % len(self._BRUSH_PROBE_COLORS)]
            probe_index += 1
            self._select_color(color, target, settings, epoch, apply_correction=False)
            self._write_brush_size(box, size, settings, epoch)
            before = self._capture_parked(canvas, park, epoch)
            self._screen_stroke(start, end, settings, epoch)
            after = self._capture_parked(canvas, park, epoch)
            try:
                band = measure_stroke_band(before, after)
            except ValueError as exc:
                LOGGER.info("Brush probe %d could not be measured: %s", size, exc)
                return None
            LOGGER.info(
                "Brush probe %d covered %.1f px of %d, touched %.1f px (%s)",
                size,
                band.height,
                canvas.height,
                band.touched_height,
                "clipped" if band.clipped else "clear",
            )
            return band

        sizes = self._probe_sizes(job, canvas, probe)
        for index, size in enumerate(sizes):
            band = probe(size, f"{index + 1} of {len(sizes)}")
            if band is None:
                continue
            if band.clipped:
                # The band ran off the sign, so its height is a floor rather
                # than a measurement. A smaller probe still carries the fit.
                clipped.append(size)
                continue
            samples.append((size, band.height / canvas.height))

        if len(samples) < 2:
            detail = (
                "Sizes "
                + ", ".join(str(size) for size in clipped)
                + " covered the whole sign, so stand closer or calibrate a larger sign."
                if clipped
                else "Confirm the paint tool is selected and the sign fills the "
                "calibrated canvas."
            )
            raise RuntimeError(
                "Brush measurement needs two usable probe strokes but got "
                f"{len(samples)}. {detail}"
            )
        model = fit_brush_size_model(samples)
        LOGGER.info(
            "Brush size model: %.6f of the sign per unit (~%.0f sign rows), offset %.6f",
            model.slope,
            model.sign_pixel_rows,
            model.intercept,
        )
        return model

    def _probe_sizes(
        self,
        job: _Job,
        canvas: ScreenRect,
        probe: Callable[[int, str], Any],
    ) -> tuple[int, ...]:
        """Choose probe sizes that bracket the brush this profile will use.

        A scout stroke establishes roughly how much of the sign one Size unit
        covers, which turns the requested cell size into a number.  The probes
        then straddle it, so the fitted line is read inside its own data rather
        than extrapolated down to a brush several times narrower than anything
        measured.
        """

        cell_fraction = job.cell_fraction
        if not cell_fraction or cell_fraction <= 0.0:
            return self._BRUSH_FALLBACK_SIZES
        band = probe(self._BRUSH_SCOUT_SIZE, "finding the scale")
        if band is None or band.clipped or band.height <= 0.0:
            LOGGER.info("Brush scout was unusable; probing the fallback sizes")
            return self._BRUSH_FALLBACK_SIZES
        # The scout ignores any offset, so this is only ever a placement hint -
        # the probes it chooses are what actually get fitted.
        per_unit = (band.height / canvas.height) / self._BRUSH_SCOUT_SIZE
        wanted = cell_fraction / per_unit
        sizes: list[int] = []
        for multiple in self._BRUSH_PROBE_MULTIPLES:
            size = int(min(BRUSH_SIZE_MAX, max(BRUSH_SIZE_MIN, round(wanted * multiple))))
            if size not in sizes:
                sizes.append(size)
        LOGGER.info(
            "Brush scout: one cell needs about size %.1f, probing %s",
            wanted,
            ", ".join(str(size) for size in sizes),
        )
        return tuple(sizes)

    def _capture_parked(
        self, canvas: ScreenRect, park: tuple[int, int], epoch: int
    ) -> Any:
        """Move the cursor off the sign, let Rust settle, then capture it."""

        self._move(park, epoch)
        self._interruptible_sleep(0.35, epoch=epoch, check_focus=True)
        self._checkpoint(epoch=epoch, check_focus=True)
        return self._screen_capture(canvas)

    def _measured_picker_target(self, target: PaintingTarget) -> PaintingTarget:
        """Shrink the picker rectangles to the widgets Rust is really drawing.

        A rectangle dragged one pixel wide sends saturation 0 and hue 0 degrees
        onto the panel behind the widget, where the click does nothing at all
        and the color stays whatever the previous group selected.  Measuring
        here rather than at calibration time means profiles already on disk are
        corrected too, and a picker that moved slightly is re-measured on every
        run.  Any failure leaves the calibration exactly as the user drew it.
        """

        if not getattr(self.input, "emits_real_input", True):
            return target
        measured: dict[str, ScreenRect] = {}
        for name in ("color_box", "hue_bar"):
            rect = getattr(target, name)
            region = ScreenRect(rect.left, rect.top, rect.width, rect.height)
            try:
                trimmed = trim_to_widget(self._screen_capture(region), region)
            except Exception:
                LOGGER.warning(
                    "Could not measure the %s; using it as calibrated", name, exc_info=True
                )
                continue
            if trimmed == region:
                continue
            LOGGER.info(
                "Trimmed %s to the drawn widget: %d,%d %dx%d -> %d,%d %dx%d",
                name,
                region.left,
                region.top,
                region.width,
                region.height,
                trimmed.left,
                trimmed.top,
                trimmed.width,
                trimmed.height,
            )
            measured[name] = trimmed
        return replace(target, **measured) if measured else target

    def _execute_plan(self, job: _Job, plan: PaintPlan | None = None) -> None:
        plan = job.plan if plan is None else plan
        target, settings = job.target, job.settings
        completed = 0
        total = sum(len(group.strokes) for group in plan.color_groups)
        total_colors = len(plan.color_groups)
        # Weight progress by stroke length plus a fixed per-stroke overhead so
        # percent/ETA stay honest when merged strokes vary widely in length.
        per_stroke_overhead = 4
        total_work = sum(
            stroke.pixel_count + per_stroke_overhead
            for group in plan.color_groups
            for stroke in group.strokes
        )
        completed_work = 0
        if total == 0:
            self._set_progress(
                color_index=0,
                total_colors=total_colors,
                stroke_index_in_color=0,
                strokes_in_color=0,
                completed_strokes=0,
                total_strokes=0,
                message="Nothing to paint",
            )
            return
        sizing_enabled = (
            settings.apply_brush_size
            and target.brush_size_box is not None
            and target.brush_size_model is not None
        )
        # Physical brush facts and the pause epoch they were established under.
        # A pause hands the mouse back to the user, who may change the brush in
        # Rust, so an epoch bump re-applies the size before the next stroke -
        # mirroring the (color, epoch) guard on the picker selection.
        applied_diameter: int | None = None
        applied_epoch: int | None = None
        selected: tuple[RGBColor, int] | None = None
        for color_index, group in enumerate(plan.color_groups, start=1):
            diameter = max(1, int(group.brush_diameter))
            for index_in_group, stroke in enumerate(group.strokes, start=1):
                while True:
                    self._checkpoint(check_focus=True)
                    current_epoch = self._pause_generation_value()
                    try:
                        if applied_epoch != current_epoch:
                            applied_diameter = None
                            applied_epoch = current_epoch
                        if sizing_enabled and diameter != applied_diameter:
                            self._apply_brush_size(job, diameter, current_epoch)
                            applied_diameter = diameter
                        if selected != (group.color, current_epoch):
                            self._select_color(group.color, target, settings, current_epoch)
                            selected = (group.color, current_epoch)
                        self._execute_stroke(stroke, plan, target.canvas, settings, current_epoch)
                        break
                    except _RetryAction:
                        selected = None
                        continue
                completed += 1
                completed_work += stroke.pixel_count + per_stroke_overhead
                self._set_progress(
                    color_index=color_index,
                    total_colors=total_colors,
                    stroke_index_in_color=index_in_group,
                    strokes_in_color=len(group.strokes),
                    completed_strokes=completed,
                    total_strokes=total,
                    completed_work=completed_work,
                    total_work=total_work,
                    message="Painting",
                )
                self._interruptible_sleep(
                    settings.delay_between_strokes_seconds, check_focus=True
                )
            self._interruptible_sleep(settings.delay_between_colors_seconds, check_focus=True)

    def _verify_and_touch_up(self, job: _Job) -> None:
        """Read the sign back and repaint the cells that missed their color.

        The comparison is relative - a cell is wrong only when its captured
        color sits decisively closer to a *different* plan color than to its
        own - so lighting and the sign's material shift, which move every
        color together, never trigger a repaint.  Each pass captures, decides,
        and repaints; a clean capture or an implausible one ends the loop.
        """

        import numpy as np

        from .verification import (
            UNRELIABLE_CAPTURE_FRACTION,
            mismatched_cells,
            plan_expectations,
            sample_cell_colors,
            touch_up_plan,
        )

        plan, target, settings = job.plan, job.target, job.settings
        if settings.verify_passes <= 0 or not getattr(
            self.input, "emits_real_input", True
        ):
            return
        if not any(group.strokes for group in plan.color_groups):
            return
        indices, palette = plan_expectations(plan)
        covered = int((indices >= 0).sum())
        if covered == 0:
            return
        canvas = ScreenRect(
            target.canvas.left,
            target.canvas.top,
            target.canvas.width,
            target.canvas.height,
        )
        # Parked over the color box, the cursor cannot shadow the capture.
        park = (
            int(round(target.color_box.left + target.color_box.width / 2.0)),
            int(round(target.color_box.top + target.color_box.height / 2.0)),
        )
        pass_number = 1
        while pass_number <= settings.verify_passes:
            try:
                epoch = self._pause_generation_value()
                self._update_progress_state(
                    PainterState.RUNNING,
                    f"Verifying the painted sign (pass {pass_number})",
                )
                self._move(park, epoch)
                self._interruptible_sleep(0.35, epoch=epoch, check_focus=True)
                self._checkpoint(epoch=epoch, check_focus=True)
                capture = self._screen_capture(canvas)
                sampled = sample_cell_colors(
                    np.asarray(capture.convert("RGB"), dtype=np.float32),
                    plan.width,
                    plan.height,
                )
                mismatch = mismatched_cells(sampled, indices, palette)
                wrong = int(mismatch.sum())
                if wrong == 0:
                    LOGGER.info(
                        "Verification pass %d: the sign matches the plan", pass_number
                    )
                    self._update_progress_state(
                        PainterState.RUNNING, "Verified: the sign matches the plan"
                    )
                    return
                if wrong > covered * UNRELIABLE_CAPTURE_FRACTION:
                    LOGGER.warning(
                        "Verification read %d of %d cells as wrong; the capture "
                        "looks unreliable (occluded sign, open menu, moved view), "
                        "so no touch-up will be painted from it",
                        wrong,
                        covered,
                    )
                    return
                LOGGER.info(
                    "Verification pass %d: repainting %d of %d cells",
                    pass_number,
                    wrong,
                    covered,
                )
                self._update_progress_state(
                    PainterState.RUNNING,
                    f"Touching up {wrong} cells (pass {pass_number})",
                )
                self._execute_plan(job, plan=touch_up_plan(mismatch, indices, palette))
                pass_number += 1
            except _RetryAction:
                # A pause released the mouse mid-pass; redo this pass whole,
                # from a fresh capture, once painting resumes.
                continue

    def _select_color(
        self,
        color: RGBColor,
        target: PaintingTarget,
        settings: PainterSettings,
        epoch: int,
        *,
        apply_correction: bool = True,
    ) -> None:
        directions = target.picker_directions
        picker_color = (
            target.color_correction.correct(color)
            if apply_correction and target.color_correction is not None
            else color
        )
        coordinates = map_rgb_to_picker(
            picker_color,
            target.hue_bar,
            target.color_box,
            hue_direction=directions.hue,
            saturation_direction=directions.saturation,
            value_direction=directions.value,
        )
        self._safe_click(coordinates.hue, epoch)
        self._interruptible_sleep(
            settings.delay_after_hue_seconds, epoch=epoch, check_focus=True
        )
        self._safe_click(coordinates.saturation_value, epoch)
        self._interruptible_sleep(
            settings.delay_after_saturation_value_seconds,
            epoch=epoch,
            check_focus=True,
        )

    def _execute_stroke(
        self,
        stroke: object,
        plan: PaintPlan,
        canvas: RectangleLike,
        settings: PainterSettings,
        epoch: int,
    ) -> None:
        start, end = logical_stroke_to_screen(stroke, plan.width, plan.height, canvas)  # type: ignore[arg-type]
        start = self._space_and_clamp(start, canvas, settings.logical_pixel_spacing)
        end = self._space_and_clamp(end, canvas, settings.logical_pixel_spacing)
        # Cell centers such as 0.5 must use floor, not Python's ties-to-even
        # round(), or adjacent logical pixels can collapse onto one coordinate.
        start_int = math.floor(start[0]), math.floor(start[1])
        end_int = math.floor(end[0]), math.floor(end[1])
        self._screen_stroke(start_int, end_int, settings, epoch)

    def _screen_stroke(
        self,
        start_int: tuple[int, int],
        end_int: tuple[int, int],
        settings: PainterSettings,
        epoch: int,
    ) -> None:
        """Drag between two physical points, or dab when they are the same."""

        self._checkpoint(epoch=epoch, check_focus=True)
        self._move(start_int, epoch)
        self._checkpoint(epoch=epoch, check_focus=True)
        self._mouse_down(epoch)
        try:
            distance = math.hypot(end_int[0] - start_int[0], end_int[1] - start_int[1])
            if distance == 0:
                self._interruptible_sleep(
                    settings.mouse_down_duration_seconds,
                    epoch=epoch,
                    check_focus=True,
                )
                return
            steps = max(1, int(math.ceil(distance / settings.stroke_interpolation_step_pixels)))
            duration = distance / settings.stroke_speed_pixels_per_second
            delay = duration / steps
            for step in range(1, steps + 1):
                self._checkpoint(epoch=epoch, check_focus=True)
                ratio = step / steps
                point = (
                    math.floor(start_int[0] + (end_int[0] - start_int[0]) * ratio),
                    math.floor(start_int[1] + (end_int[1] - start_int[1]) * ratio),
                )
                self._move(point, epoch)
                self._interruptible_sleep(delay, epoch=epoch, check_focus=True)
        finally:
            self.input.mouse_up(MouseButton.LEFT)

    @staticmethod
    def _space_and_clamp(
        point: tuple[float, float], canvas: RectangleLike, spacing: float
    ) -> tuple[float, float]:
        if spacing != 1.0:
            center_x = canvas.left + canvas.width / 2.0
            center_y = canvas.top + canvas.height / 2.0
            point = (
                center_x + (point[0] - center_x) * spacing,
                center_y + (point[1] - center_y) * spacing,
            )
        return clamp_to_rect(point[0], point[1], canvas)

    def _safe_click(self, point: tuple[float, float], epoch: int) -> None:
        self._checkpoint(epoch=epoch, check_focus=True)
        # Picker normalization already targets inclusive physical endpoints, for
        # which conventional rounding is appropriate.
        target = int(round(point[0])), int(round(point[1]))
        self._move(target, epoch)
        self._checkpoint(epoch=epoch, check_focus=True)
        self._mouse_down(epoch)
        try:
            settings = self._job.settings if self._job is not None else PainterSettings()
            self._interruptible_sleep(
                settings.mouse_down_duration_seconds,
                epoch=epoch,
                check_focus=True,
            )
        finally:
            self.input.mouse_up(MouseButton.LEFT)

    def _guarded_input(self, epoch: int, action: Callable[[], None]) -> None:
        """Atomically validate job state and emit one non-release input event."""

        with self._condition:
            if self._abort_event.is_set() or self._abort_requested:
                raise _AbortRequested
            if (
                self._pause_event.is_set()
                or self._state is PainterState.PAUSED
                or self._pause_generation != epoch
            ):
                raise _RetryAction
            action()

    def _move(self, point: tuple[int, int], epoch: int) -> None:
        def emit() -> None:
            self.input.move_mouse(*point)
            self._last_commanded_point = point
            self._commanded_history.append(point)

        self._guarded_input(epoch, emit)

    def _mouse_down(self, epoch: int) -> None:
        self._guarded_input(epoch, lambda: self.input.mouse_down(MouseButton.LEFT))

    def _checkpoint(self, *, epoch: int | None = None, check_focus: bool) -> None:
        while True:
            with self._condition:
                if self._abort_event.is_set() or self._abort_requested:
                    raise _AbortRequested
                paused = (
                    self._pause_event.is_set() or self._state == PainterState.PAUSED
                )
                if paused:
                    self._condition.wait(timeout=0.05)
            if paused:
                # Keep watching the corner while paused. The corner gesture is
                # exactly what a user reaches for once the mouse has already
                # interrupted the job and they want it gone rather than held,
                # and a paused worker is the only thread still polling.
                job = self._job
                if job is not None:
                    self._check_cursor(job.settings, time.monotonic(), allow_pause=False)
                continue
            if self._check_safety(check_focus=check_focus):
                continue
            with self._condition:
                if self._abort_event.is_set() or self._abort_requested:
                    raise _AbortRequested
                if epoch is not None and self._pause_generation != epoch:
                    raise _RetryAction
            return

    def _check_safety(self, *, check_focus: bool) -> bool:
        job = self._job
        if job is None:
            return False
        settings = job.settings
        now = time.monotonic()
        if (
            check_focus
            and settings.require_foreground
            and getattr(self.input, "emits_real_input", True)
            and now - self._last_focus_check >= settings.focus_check_interval_seconds
        ):
            self._last_focus_check = now
            requirement = ForegroundRequirement(
                title_contains=settings.expected_window_title_contains or None,
                executable=settings.expected_process_name or None,
            )
            if not self._foreground_checker(requirement):
                self.pause(_foreground_failure_reason(settings))
                return True

        return self._check_cursor(settings, now, allow_pause=True)

    def _check_cursor(
        self, settings: PainterSettings, now: float, *, allow_pause: bool
    ) -> bool:
        """Sample the real cursor for the corner stop and for user movement.

        ``allow_pause`` is False when the job is already paused: the corner
        emergency stop still has to work there, but there is nothing left to
        pause. Returns True when the job just paused and the caller must
        re-evaluate its state.
        """

        watch_movement = allow_pause and settings.pause_on_mouse_move
        if not settings.corner_abort_enabled and not watch_movement:
            return False
        if not getattr(self.input, "emits_real_input", True):
            return False
        # resume() clears the baseline from another thread, so snapshot both
        # halves together rather than testing one and then reading the other.
        with self._condition:
            expected = self._last_commanded_point
            history = tuple(self._commanded_history)
        if expected is None or not history:
            return False
        if now - self._last_corner_check < settings.safety_poll_interval_seconds:
            return False
        self._last_corner_check = now
        try:
            cursor = self.input.get_cursor_position()
            if settings.corner_abort_enabled and self._is_manual_corner_stop(
                cursor, expected, settings
            ):
                self.abort("mouse moved to emergency corner")
                raise _AbortRequested
        except _AbortRequested:
            raise
        except Exception:
            LOGGER.warning("Could not read the cursor for safety checks", exc_info=True)
            return False
        if not watch_movement:
            return False
        return self._check_mouse_movement(cursor, history, settings, now)

    def _check_mouse_movement(
        self,
        cursor: tuple[int, int],
        history: tuple[tuple[int, int], ...],
        settings: PainterSettings,
        now: float,
    ) -> bool:
        """Pause once the cursor keeps leaving the path the painter commands.

        Detection is positional on purpose. A low-level mouse hook could tell
        injected input from physical input outright, but installing one next to
        an anti-cheat protected game is precisely the behaviour anti-cheat
        looks for, so movement is inferred from the gap between where the
        painter put the cursor and where the cursor actually is.

        The painter re-warps the cursor every few milliseconds, which erases
        most of a gap as fast as it appears, so single samples are small even
        during a deliberate grab. Accumulating the part of each gap that
        exceeds the tolerance is what makes a sustained hand movement separable
        from the queued-input lag of one sample.
        """

        distance = min(
            math.hypot(cursor[0] - point[0], cursor[1] - point[1])
            for point in history
        )
        tolerance = settings.mouse_move_tolerance_pixels
        if distance <= tolerance:
            self._reset_mouse_drift()
            return False
        if self._mouse_drift_started == 0.0:
            self._mouse_drift_started = now
        elif now - self._mouse_drift_started > _MOUSE_DRIFT_WINDOW_SECONDS:
            # A hand crosses the threshold in a few dozen milliseconds. A gap
            # that needs this long only ever creeps a hair past the tolerance,
            # which is rounding rather than movement, so drop what it added
            # instead of letting it reach a pause over many seconds.
            self._mouse_drift_pixels = 0.0
            self._mouse_drift_started = now
        self._mouse_drift_pixels += distance - tolerance
        if self._mouse_drift_pixels < settings.mouse_move_pause_threshold_pixels:
            return False
        self._reset_mouse_drift()
        self.pause("mouse moved - resume to continue from the same stroke")
        return True

    def _reset_mouse_drift(self) -> None:
        self._mouse_drift_pixels = 0.0
        self._mouse_drift_started = 0.0

    def _reset_mouse_movement_baseline(self) -> None:
        self._last_commanded_point = None
        self._commanded_history.clear()
        self._reset_mouse_drift()

    def _is_manual_corner_stop(
        self,
        cursor: tuple[int, int],
        expected: tuple[int, int],
        settings: PainterSettings,
    ) -> bool:
        screen = self._virtual_screen_provider()
        margin = settings.corner_abort_margin_pixels
        near_left = cursor[0] <= screen.left + margin
        near_right = cursor[0] >= screen.right - 1 - margin
        near_top = cursor[1] <= screen.top + margin
        near_bottom = cursor[1] >= screen.bottom - 1 - margin
        at_corner = (near_left or near_right) and (near_top or near_bottom)
        displacement = math.hypot(cursor[0] - expected[0], cursor[1] - expected[1])
        return at_corner and displacement >= settings.corner_abort_minimum_distance_pixels

    def _interruptible_sleep(
        self,
        duration: float,
        *,
        epoch: int | None = None,
        check_focus: bool,
    ) -> None:
        remaining = max(0.0, duration)
        if getattr(self.input, "skip_timing", False):
            self._checkpoint(epoch=epoch, check_focus=check_focus)
            return
        if remaining == 0:
            self._checkpoint(epoch=epoch, check_focus=check_focus)
            return
        poll = self._job.settings.safety_poll_interval_seconds if self._job else 0.01
        while remaining > 0:
            self._checkpoint(epoch=epoch, check_focus=check_focus)
            interval = min(poll, remaining)
            started = time.monotonic()
            # Windows condition/event waits can be rounded to the legacy timer
            # quantum on some Python/OS combinations.  Short interpolated drag
            # slices use Python's high-resolution sleep; the next checkpoint is
            # still at most a few milliseconds away. Longer waits use the event
            # so an emergency abort wakes them immediately.
            if interval < 0.01:
                time.sleep(interval)
                if self._abort_event.is_set():
                    raise _AbortRequested
            elif self._abort_event.wait(interval):
                raise _AbortRequested
            remaining -= max(0.0, time.monotonic() - started)

    def _pause_generation_value(self) -> int:
        with self._condition:
            return self._pause_generation

    def _active_elapsed(self) -> float:
        if self._started_at is None:
            return 0.0
        paused = self._paused_seconds
        if self._paused_at is not None:
            paused += time.monotonic() - self._paused_at
        return max(0.0, time.monotonic() - self._started_at - paused)

    def _set_progress(
        self,
        *,
        color_index: int,
        total_colors: int,
        stroke_index_in_color: int,
        strokes_in_color: int,
        completed_strokes: int,
        total_strokes: int,
        completed_work: int | None = None,
        total_work: int | None = None,
        message: str,
    ) -> None:
        elapsed = self._active_elapsed()
        if completed_work is None or not total_work:
            completed_work = completed_strokes
            total_work = total_strokes
        percent = 100.0 if total_work == 0 else completed_work * 100.0 / total_work
        remaining = None
        if 0 < completed_work < total_work:
            remaining = elapsed / completed_work * (total_work - completed_work)
        with self._condition:
            self._progress = PaintProgress(
                self._state,
                color_index,
                total_colors,
                stroke_index_in_color,
                strokes_in_color,
                completed_strokes,
                total_strokes,
                percent,
                elapsed,
                remaining,
                message,
            )
        self._emit_progress(force=completed_strokes == total_strokes)

    def _update_progress_state(self, state: PainterState, message: str) -> None:
        with self._condition:
            # A pause/abort can win between the worker's checkpoint and this
            # presentation update. Never make progress claim an older state
            # than the authoritative state machine.
            if self._state is not state:
                return
            old = self._progress
            self._progress = PaintProgress(
                state,
                old.color_index,
                old.total_colors,
                old.stroke_index_in_color,
                old.strokes_in_color,
                old.completed_strokes,
                old.total_strokes,
                old.percent,
                self._active_elapsed(),
                old.estimated_remaining_seconds,
                message,
            )
        self._emit_progress(force=True)

    def _emit_progress(self, *, force: bool) -> None:
        callback = self._on_progress
        if callback is None:
            return
        now = time.monotonic()
        job = self._job
        interval = job.settings.progress_callback_interval_seconds if job else 0.04
        if not force and now - self._last_progress_emit < interval:
            return
        self._last_progress_emit = now
        self._safe_callback(callback, self.progress, label="progress")

    def _transition(self, state: PainterState, reason: str) -> None:
        with self._condition:
            # Abort is terminal for the active worker; a late transition from a
            # just-unblocked action must never overwrite it.
            if self._state == PainterState.ABORTED and state != PainterState.READY:
                return
            self._state = state
            self._state_reason = reason
            self._condition.notify_all()
        self._emit_state(state, reason)

    def _enter_running_after_countdown(self) -> None:
        """Atomically honor a pause/abort at the countdown-to-run boundary."""

        while True:
            with self._condition:
                if self._abort_event.is_set() or self._abort_requested:
                    raise _AbortRequested
                if self._pause_event.is_set() or self._state is PainterState.PAUSED:
                    self._condition.wait(timeout=0.05)
                    continue
                self._state = PainterState.RUNNING
                self._state_reason = "started"
                self._condition.notify_all()
                break
        self._emit_state(PainterState.RUNNING, "started")

    def _finish_completed(self) -> None:
        """Commit completion without overwriting a simultaneous pause/abort."""

        while True:
            with self._condition:
                if self._abort_event.is_set() or self._abort_requested:
                    raise _AbortRequested
                if self._pause_event.is_set() or self._state is PainterState.PAUSED:
                    self._condition.wait(timeout=0.05)
                    continue
                self._state = PainterState.COMPLETED
                self._state_reason = "completed"
                self._condition.notify_all()
                break
        self._emit_state(PainterState.COMPLETED, "completed")

    def _emit_state(self, state: PainterState, reason: str) -> None:
        self._safe_callback(self._on_state_change, state, reason, label="state")

    @staticmethod
    def _safe_callback(callback: Callable[..., Any] | None, *args: Any, label: str) -> None:
        if callback is None:
            return
        try:
            callback(*args)
        except Exception:
            LOGGER.exception("Painter %s callback failed", label)

    def _safe_release_all(self) -> None:
        try:
            self.input.release_all()
        except Exception:
            LOGGER.exception("Could not release all held input")


# Backward-friendly names for GUI code.
PaintingProgress = PaintProgress
PainterController = Painter


__all__ = [
    "PaintProgress",
    "Painter",
    "PainterController",
    "PainterSettings",
    "PainterState",
    "PaintingProgress",
    "PaintingTarget",
    "PickerDirections",
]
