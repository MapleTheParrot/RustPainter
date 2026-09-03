"""Threaded, interruptible paint-plan execution with conservative safety checks."""

from __future__ import annotations

import contextlib
import ctypes
import logging
import math
import os
import threading
import time
from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass, replace
from enum import Enum
from types import SimpleNamespace
from typing import Any, Callable, ClassVar, Iterator, Sequence

from .brush_calibration import (
    BRUSH_SIZE_MAX,
    BRUSH_SIZE_MIN,
    SIGN_TEXTURE_SIZES,
    BrushSizeModel,
    StrokeBand,
    canonical_texture_rows,
    fit_brush_size_model,
    format_brush_size,
    measure_stroke_band,
)
from .color_calibration import ColorCorrectionModel
from .color_mapping import picker_click_plan, rgb_to_hsv
from .color_swatch import LOCATOR_COLOR, SwatchReading, locate_swatch, read_swatch
from .coordinates import RectangleLike, clamp_to_rect, logical_stroke_to_screen, normalized_point
from .cursor_map import (
    ANCHOR_LANES,
    DEFAULT_LANES,
    MAX_LANES,
    SWEEP_MARGIN_PIXELS,
    SweepError,
    attribute_sweep,
    check_lattice,
    fill_in_sweep,
    grid_from_tables,
    lane_line,
    lane_offsets,
    lattice_targets,
    sweep_positions,
    unread_positions,
)
from .native_plan import is_native, nativize_plan, stroke_index_map
from .input_controller import InputController, MouseButton
from .models import PaintPlan, RGBColor, ScreenRect, Stroke
from .paint_timing import (
    BRUSH_CALIBRATION_SECONDS,
    FAST_CALIBRATION_SECONDS,
    CHECK_CAPTURE_SECONDS_DEFAULT,
    CHECK_REPAINT_FRACTION_DEFAULT,
    CONFIRM_SETTLE_SECONDS,
    DAB_PROBE_DOTS,
    DAB_PROBE_MAX_MISSES,
    DAB_PROBE_MIN_DABS,
    DAB_PROBE_SIZES,
    DEFAULT_STROKE_OVERHEAD_SECONDS,
    DRAG_RATE_PROBE_MIN_RUN_TEXELS,
    DRAG_RATE_PROBE_TEXELS_PER_SECOND,
    STROKE_GAP_PROBE_CANDIDATES,
    KEY_GAP_SECONDS,
    KEY_HOLD_SECONDS,
    LONG_DRAG_MAX_STEP_TEXELS,
    LONG_DRAG_MAX_TEXELS_PER_SECOND,
    MIN_PRESS_SECONDS,
    PICKER_CLICK_HOLD_SECONDS,
    PRESS_HOLD_PROBE_CANDIDATES,
    PRESS_HOLD_PROBE_DOTS,
    PRESS_HOLD_PROBE_MIN_STROKES,
    SETTLE_FLOOR_SECONDS,
    SHIFT_LINE_MIN_TEXELS,
    SHIFT_LINE_MODIFIER_LEAD_SECONDS,
    STROKE_GAP_FLOOR_SECONDS,
    TOUCH_UP_FRACTION_DEFAULT,
    PhaseTiming,
    PlanWorkSchedule,
    TouchUpTiming,
    StrokeTiming,
    fields_below_floor,
    remaining_seconds,
    stroke_pace,
)
from .picker_calibration import trim_to_widget
from .texel_grid import (
    AIM_AUDIT_MAX_PITCH,
    GridProbePlan,
    TexelGridModel,
    audit_cursor_map,
    find_quad_edges,
    locate_stamps,
    measure_grid,
    stamp_diff,
)
from .screen import (
    ForegroundRequirement,
    capture_region,
    foreground_window_matches,
)
from .sign_export import ExportWatcher, SignExport
from .ui_guard import PaintingUiGuard


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


# A hand-drawn canvas rectangle can be a pixel out at a corner. A cursor map
# that proves every interior sample but cannot hit one or two corners is still
# useful: the map itself was measured from Rust's export, and the normal
# verification pass will still report any border cell that needs attention.
# Do not relax a whole edge -- that would hide a real mapping error.
_SWEEP_CHECK_CORNER_MISS_LIMIT = 4


def _only_isolated_corner_misses(
    misses: Sequence[tuple[int, int]], columns: int, rows: int
) -> bool:
    """Whether failed map checks are a few corners, not an edge/interior fault."""

    corners = {
        (0, 0),
        (columns - 1, 0),
        (0, rows - 1),
        (columns - 1, rows - 1),
    }
    return bool(misses) and len(misses) <= _SWEEP_CHECK_CORNER_MISS_LIMIT and all(
        target in corners for target in misses
    )


@contextlib.contextmanager
def _high_resolution_timer() -> Iterator[None]:
    """Request 1 ms timer resolution while painting.

    The default Windows timer granularity rounds short waits up to ~15.6 ms,
    which silently inflates every configured inter-stroke and interpolation
    delay.
    """

    acquired = False
    if os.name == "nt":
        try:
            acquired = ctypes.WinDLL("winmm").timeBeginPeriod(1) == 0
        except (AttributeError, OSError):
            acquired = False
    try:
        yield
    finally:
        if acquired:
            try:
                ctypes.WinDLL("winmm").timeEndPeriod(1)
            except (AttributeError, OSError):
                LOGGER.warning("Could not restore the Windows timer resolution")


ABOVE_NORMAL_PRIORITY_CLASS = 0x00008000


@contextlib.contextmanager
def _above_normal_priority() -> Iterator[None]:
    """Run the process a notch above normal while painting.

    A stroke is a chain of tightly timed sleeps and SendInput calls; when the
    game, a capture, and a background task all want the CPU, the scheduler
    can hold this process past a frame and turn a short press into a missed
    one.  One priority notch keeps the input thread on schedule without
    starving the game the way a realtime class would.  The previous class is
    restored on the way out, and any failure leaves the priority alone.
    """

    previous: int | None = None
    kernel32 = None
    handle = None
    if os.name == "nt":
        try:
            kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
            # The pseudo-handle is (HANDLE)-1; without prototypes ctypes
            # truncates it through a 32-bit int and every call fails quietly.
            kernel32.GetCurrentProcess.restype = ctypes.c_void_p
            kernel32.GetPriorityClass.argtypes = (ctypes.c_void_p,)
            kernel32.GetPriorityClass.restype = ctypes.c_uint32
            kernel32.SetPriorityClass.argtypes = (ctypes.c_void_p, ctypes.c_uint32)
            kernel32.SetPriorityClass.restype = ctypes.c_int
            handle = kernel32.GetCurrentProcess()
            previous = int(kernel32.GetPriorityClass(handle)) or None
            if previous is not None and previous != ABOVE_NORMAL_PRIORITY_CLASS:
                if kernel32.SetPriorityClass(handle, ABOVE_NORMAL_PRIORITY_CLASS) == 0:
                    previous = None
            else:
                previous = None
        except (AttributeError, OSError):
            previous = None
    try:
        yield
    finally:
        if previous is not None and kernel32 is not None:
            try:
                kernel32.SetPriorityClass(handle, previous)
            except (AttributeError, OSError):
                LOGGER.warning("Could not restore the process priority class")


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
    circle_brush_button: RectangleLike | None = None
    square_brush_button: RectangleLike | None = None
    clear_button: RectangleLike | None = None
    save_button: RectangleLike | None = None
    download_button: RectangleLike | None = None
    picker_directions: PickerDirections = PickerDirections()
    color_correction: ColorCorrectionModel | None = None
    brush_size_model: BrushSizeModel | None = None
    # The texel grid an earlier job measured on this profile's sign.  A paint
    # job that measures its own grid ignores it; one that cannot - automatic
    # sizing off, which is what types the probe's brush - paints on it when it
    # still sits on the calibrated rectangle.
    texel_grid: TexelGridModel | None = None
    # The bare sign's colour, measured when a job last cleared this sign.
    # A job that resumes onto an existing painting never sees the sign
    # cleared, and a plan that covers every cell leaves no unpainted cell to
    # read the wood from - so without this the touch-up pass cannot tell a
    # hole from a cell painted some other colour, and holes go unrepaired.
    bare_color: RGBColor | None = None
    cached_press_hold_seconds: float | None = None
    cached_stroke_gap_seconds: float | None = None
    cached_drag_rate: float | None = None
    cached_detail_size: float | None = None
    cached_dab_sweep: bool = False

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
        bare_value = (
            metadata.get("bare_sign_color") if isinstance(metadata, Mapping) else None
        )
        bare_color = None
        if isinstance(bare_value, (list, tuple)) and len(bare_value) == 3:
            try:
                bare_color = tuple(
                    min(255, max(0, int(round(float(channel))))) for channel in bare_value
                )
            except (TypeError, ValueError):
                LOGGER.warning("The profile's stored bare-sign colour is invalid")
        grid_value = metadata.get("texel_grid") if isinstance(metadata, Mapping) else None
        texel_grid = None
        if isinstance(grid_value, Mapping):
            try:
                texel_grid = TexelGridModel.from_dict(grid_value)
            except (KeyError, TypeError, ValueError):
                LOGGER.warning("The profile's stored texel grid is invalid", exc_info=True)
        performance = (
            metadata.get("calibration_performance")
            if isinstance(metadata, Mapping)
            else None
        )
        performance = performance if isinstance(performance, Mapping) else {}

        def cached_number(name: str, low: float, high: float) -> float | None:
            value = performance.get(name)
            if value is None or isinstance(value, bool):
                return None
            try:
                number = float(value)
            except (TypeError, ValueError):
                return None
            return number if math.isfinite(number) and low <= number <= high else None

        return cls(
            canvas=canvas,
            color_box=color_box,
            hue_bar=hue_bar,
            brush_size_box=getattr(profile, "brush_size_box", None),
            circle_brush_button=getattr(profile, "circle_brush_button", None),
            square_brush_button=getattr(profile, "square_brush_button", None),
            clear_button=getattr(profile, "clear_button", None),
            save_button=getattr(profile, "save_button", None),
            download_button=getattr(profile, "download_button", None),
            picker_directions=PickerDirections(
                hue="bottom_to_top",
                saturation="left_low",
                value="top_bright",
            ),
            color_correction=correction,
            brush_size_model=brush_size_model,
            texel_grid=texel_grid,
            bare_color=bare_color,
            cached_press_hold_seconds=cached_number("pressHoldSeconds", 0.001, 1.0),
            cached_stroke_gap_seconds=cached_number("strokeGapSeconds", 0.0, 1.0),
            cached_drag_rate=cached_number("dragTexelsPerSecond", 1.0, 10000.0),
            cached_detail_size=cached_number("detailSize", BRUSH_SIZE_MIN, BRUSH_SIZE_MAX),
            cached_dab_sweep=bool(performance.get("dabSweep", False)),
        )


@dataclass(frozen=True, slots=True)
class PainterSettings:
    """All timing and safety values that may need in-game tuning."""

    stroke_speed_pixels_per_second: float = 700.0
    mouse_down_duration_seconds: float = 0.07
    delay_after_hue_seconds: float = 0.09
    delay_after_saturation_value_seconds: float = 0.09
    delay_between_strokes_seconds: float = 0.02
    delay_between_colors_seconds: float = 0.12
    stroke_interpolation_step_pixels: float = 4.0
    logical_pixel_spacing: float = 1.0
    brush_size: float = 1.0
    apply_brush_size: bool = False
    # Prefer a passive sign-edge validation and the profile's saved model/grid.
    # A missing or mismatched measurement automatically runs the full probes.
    reuse_calibration: bool = True
    # Measure the sign's texel grid at the start of each job and lay the
    # strokes on it.  Needs the same calibration as brush sizing; a sign the
    # probe cannot read falls back to the brush-derived grid, so this is an
    # escape hatch rather than a feature switch.
    measure_texel_grid: bool = True
    # Draw straight runs of :data:`SHIFT_LINE_MIN_TEXELS` texels or more with
    # Rust's line tool: Shift held through a drag, and on release the game
    # itself fills the straight stroke between the press and the release.
    # Only ever used after a probe stroke proves the mechanic on this sign; a
    # sign that fails the probe paints with drags exactly as before.
    use_line_tool: bool = True
    # Measure this sign's timing floors instead of assuming them: the
    # shortest press hold that lands every dab, the shortest gap between
    # strokes the game keeps apart, and the fastest long drag it paints
    # whole (batches of dots and probe drags, read back from captures).  A
    # sign that fails a probe keeps that floor; drag dwells always keep it.
    measure_press_hold: bool = True
    # Prove the one-cell brush on this sign before painting: batches of lone
    # dabs at rising Size numbers until one lands them all, which the job's
    # single-cell strokes and touch-up then use.  Off, or unproven, the
    # one-cell brush stays the game's smallest.
    measure_dab_size: bool = True
    # After painting, audit the canvas and repaint decisively wrong cells,
    # up to this many times. Every repair is followed by another audit, so a
    # job never assumes its last touch-up landed. Zero disables verification.
    verify_passes: int = 2
    # Check each color as it goes down: once its strokes are painted the
    # sign is captured, the cells that did not take the color are repainted,
    # and the capture is repeated - up to this many rounds per color.  Off
    # by default: it was built against presses the game was thought to
    # drop, which measurement found it does not, and its first live outing
    # misread a sixth of a sign's cells as missing and spent four rounds
    # repainting them.  The picks are read back instead (below).
    confirm_strokes: bool = False
    confirm_max_rounds: int = 4
    # Between color groups, every this many seconds of the main plan, the
    # sign's exported texture is read and the holes in what is already final
    # are refilled while the paint is fresh - so a long run learns of the
    # game's dropped dabs in minutes instead of at the end.  Zero disables;
    # needs the download button calibrated.
    interim_audit_seconds: float = 600.0
    # Read the selected color back off the panel after every pick, and
    # pick again when the clicks did not take.
    verify_color_picks: bool = True
    brush_direction: str = "low_to_high"
    brush_shape: str = "auto"
    manage_ui_scale: bool = False
    painting_ui_scale: float = 0.5
    normal_ui_scale: float = 1.0
    console_key: str = "F1"
    delay_after_brush_seconds: float = 0.07
    countdown_seconds: float = 3.0
    require_foreground: bool = False
    expected_window_title_contains: str | None = "Rust"
    expected_process_name: str | None = "RustClient.exe"
    focus_check_interval_seconds: float = 0.05
    # A temporary focus loss is common during a long unattended paint.  Keep
    # the job safe while Rust is elsewhere, then retry when it returns unless
    # the user has taken the mouse back.
    auto_resume_on_focus_return: bool = True
    auto_resume_focus_retry_seconds: float = 10.0
    pause_on_mouse_move: bool = True
    mouse_move_pause_threshold_pixels: float = 24.0
    mouse_move_tolerance_pixels: float = 3.0
    safety_poll_interval_seconds: float = 0.01
    progress_callback_interval_seconds: float = 0.04
    # Every interval, save the sign, jump, and reopen the sign, so a server
    # that kicks idle players sees one moving.  Needs the Save button.
    anti_afk_enabled: bool = False
    anti_afk_interval_seconds: float = 1800.0
    # Pause when the painting UI's calibrated widgets are no longer on the
    # screen - a kick, a server restart, or a sign closed by hand.  The
    # anti-AFK break closes the UI on purpose and suspends the guard while
    # it does.
    ui_guard_enabled: bool = False

    # The fields a paused job may take new values for.  Everything else
    # shaped the job - which brush it measured, how its strokes were laid
    # out, whether it counted down - and changing those under a half-painted
    # sign would not produce the sign the plan promised.
    RETUNABLE_FIELDS: ClassVar[tuple[str, ...]] = (
        "stroke_speed_pixels_per_second",
        "mouse_down_duration_seconds",
        "delay_after_hue_seconds",
        "delay_after_saturation_value_seconds",
        "delay_between_strokes_seconds",
        "delay_between_colors_seconds",
        "stroke_interpolation_step_pixels",
        "delay_after_brush_seconds",
        "use_line_tool",
        "measure_press_hold",
        "measure_dab_size",
        "verify_passes",
        "confirm_strokes",
        "confirm_max_rounds",
        "verify_color_picks",
        "require_foreground",
        "expected_window_title_contains",
        "expected_process_name",
        "focus_check_interval_seconds",
        "auto_resume_on_focus_return",
        "auto_resume_focus_retry_seconds",
        "pause_on_mouse_move",
        "mouse_move_pause_threshold_pixels",
        "mouse_move_tolerance_pixels",
        "safety_poll_interval_seconds",
        "progress_callback_interval_seconds",
        "anti_afk_enabled",
        "anti_afk_interval_seconds",
        "ui_guard_enabled",
    )

    def retuned(self, other: "PainterSettings") -> "PainterSettings":
        """These settings with ``other``'s timing and safety values."""

        return replace(
            self, **{name: getattr(other, name) for name in self.RETUNABLE_FIELDS}
        )

    def __post_init__(self) -> None:
        positive = {
            "stroke_speed_pixels_per_second": self.stroke_speed_pixels_per_second,
            "stroke_interpolation_step_pixels": self.stroke_interpolation_step_pixels,
            "logical_pixel_spacing": self.logical_pixel_spacing,
            "focus_check_interval_seconds": self.focus_check_interval_seconds,
            "auto_resume_focus_retry_seconds": self.auto_resume_focus_retry_seconds,
            "safety_poll_interval_seconds": self.safety_poll_interval_seconds,
            "mouse_move_pause_threshold_pixels": self.mouse_move_pause_threshold_pixels,
            "anti_afk_interval_seconds": self.anti_afk_interval_seconds,
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
            "mouse_move_tolerance_pixels": self.mouse_move_tolerance_pixels,
            "progress_callback_interval_seconds": self.progress_callback_interval_seconds,
        }
        for name, value in nonnegative.items():
            if value < 0 or not math.isfinite(value):
                raise ValueError(f"{name} must be a finite non-negative number")
        if isinstance(self.verify_passes, bool) or not isinstance(
            self.verify_passes, int
        ) or not 0 <= self.verify_passes <= 5:
            raise ValueError("verify_passes must be an integer between 0 and 5")
        if isinstance(self.confirm_max_rounds, bool) or not isinstance(
            self.confirm_max_rounds, int
        ) or not 1 <= self.confirm_max_rounds <= 8:
            raise ValueError("confirm_max_rounds must be an integer between 1 and 8")
        if self.brush_direction not in {"low_to_high", "high_to_low"}:
            raise ValueError("brush_direction must be low_to_high or high_to_low")
        if self.brush_shape not in {"auto", "circle", "square"}:
            raise ValueError("brush_shape must be auto, circle, or square")
        from .game_console import validate_console_key, validate_ui_scale

        validate_ui_scale(self.painting_ui_scale)
        validate_ui_scale(self.normal_ui_scale)
        validate_console_key(self.console_key)
        if self.manage_ui_scale and not self.require_foreground:
            raise ValueError(
                "Temporary Rust UI scaling requires the foreground guard"
            )
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
        game = values.get("game", values)
        safety = values.get("safety", values)
        if not isinstance(painting, Mapping) or not isinstance(game, Mapping) or not isinstance(safety, Mapping):
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
                pick(painting, "mouse_down_duration_seconds", 0.07, "dot_duration_seconds")
            ),
            delay_after_hue_seconds=float(
                pick(painting, "delay_after_hue_seconds", 0.09)
            ),
            delay_after_saturation_value_seconds=float(
                pick(painting, "delay_after_saturation_value_seconds", 0.09, "delay_after_sv_seconds")
            ),
            delay_between_strokes_seconds=float(
                pick(painting, "delay_between_strokes_seconds", 0.02)
            ),
            delay_between_colors_seconds=float(
                pick(painting, "delay_between_colors_seconds", 0.12)
            ),
            stroke_interpolation_step_pixels=float(
                pick(painting, "stroke_interpolation_step_pixels", 4.0)
            ),
            logical_pixel_spacing=float(pick(painting, "logical_pixel_spacing", 1.0)),
            brush_size=float(pick(painting, "brush_size", 1.0)),
            apply_brush_size=bool(pick(painting, "apply_brush_size", False)),
            reuse_calibration=bool(pick(painting, "reuse_calibration", True)),
            measure_texel_grid=bool(pick(painting, "measure_texel_grid", True)),
            use_line_tool=bool(pick(painting, "use_line_tool", True)),
            measure_press_hold=bool(pick(painting, "measure_press_hold", True)),
            measure_dab_size=bool(pick(painting, "measure_dab_size", True)),
            verify_passes=int(pick(painting, "verify_passes", 2)),
            confirm_strokes=bool(pick(painting, "confirm_strokes", False)),
            confirm_max_rounds=int(pick(painting, "confirm_max_rounds", 4)),
            interim_audit_seconds=float(pick(painting, "interim_audit_seconds", 600.0)),
            verify_color_picks=bool(pick(painting, "verify_color_picks", True)),
            brush_direction=str(pick(painting, "brush_direction", "low_to_high")),
            brush_shape=str(pick(painting, "brush_shape", "auto")),
            manage_ui_scale=bool(pick(game, "manage_ui_scale", False)),
            painting_ui_scale=float(pick(game, "painting_ui_scale", 0.5)),
            normal_ui_scale=float(pick(game, "normal_ui_scale", 1.0)),
            console_key=str(pick(game, "console_key", "F1")),
            delay_after_brush_seconds=float(pick(painting, "delay_after_brush_seconds", 0.07)),
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
            auto_resume_on_focus_return=bool(
                pick(safety, "auto_resume_on_focus_return", True)
            ),
            auto_resume_focus_retry_seconds=float(
                pick(safety, "auto_resume_focus_retry_seconds", 10.0)
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
            anti_afk_enabled=bool(pick(safety, "anti_afk_enabled", False)),
            anti_afk_interval_seconds=(
                float(pick(safety, "anti_afk_interval_seconds", 0.0))
                or float(pick(safety, "anti_afk_interval_minutes", 30.0)) * 60.0
            ),
            ui_guard_enabled=bool(pick(safety, "ui_guard_enabled", True)),
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
    # "calibrate" while the job measures this sign's brush and wipes the probe
    # strokes; "paint" once the artwork itself is going down.  Clients that
    # document a run - a timelapse recorder above all - use this to start when
    # the picture starts rather than when the worker does.
    phase: str = "paint"
    # Reserved for concise run warnings supplied by future checks.
    alerts: tuple[str, ...] = ()

    @property
    def stroke_index(self) -> int:
        """Global completed-stroke index, useful for ``427 / 1840`` UI text."""

        return self.completed_strokes


ProgressCallback = Callable[[PaintProgress], None]
StateCallback = Callable[[PainterState, str], None]


@dataclass(frozen=True, slots=True)
class ConfirmationSummary:
    """What checking each color as it went down found and did, for the run report.

    ``judged`` cells are those a capture could decide; ``missed`` is how
    many of them had not taken their color when first checked, ``repainted``
    how many repaint strokes went down for them, ``unrepaired`` how many
    were still missing when the rounds ran out.
    """

    colors: int = 0
    judged: int = 0
    missed: int = 0
    repainted_strokes: int = 0
    unrepaired: int = 0
    rounds: int = 0
    skipped_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "colorsChecked": self.colors,
            "cellsJudged": self.judged,
            "cellsMissedFirstCheck": self.missed,
            "repaintStrokes": self.repainted_strokes,
            "cellsUnrepaired": self.unrepaired,
            "repaintRounds": self.rounds,
            "skippedReason": self.skipped_reason,
        }


@dataclass(frozen=True, slots=True)
class ColorPickSummary:
    """What reading each color pick back off the panel found, for the run report.

    ``picks`` is how many colors were read back, ``retried`` how many took
    a second (or later) round of clicks, ``failed`` how many never read
    right and paused the job.  ``skipped_reason`` says why the panel was
    not read at all, or from which pick it stopped being.
    """

    picks: int = 0
    retried: int = 0
    failed: int = 0
    skipped_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "colorsRead": self.picks,
            "picksRetried": self.retried,
            "picksFailed": self.failed,
            "skippedReason": self.skipped_reason,
        }


@dataclass(slots=True)
class _Confirmation:
    """The per-job state of checking colors as they go down."""

    canvas: ScreenRect
    park: tuple[int, int]
    indices: Any  # final color index per cell
    palette: Any
    centers: tuple[Any, Any] | None
    blend: Any
    # Per-cell RGB reading of the sign as it stood before the color being
    # painted now; replaced by the latest capture after each color.
    reference: Any
    colors: int = 0
    judged: int = 0
    missed: int = 0
    repainted_strokes: int = 0
    unrepaired: int = 0
    rounds: int = 0


@dataclass(slots=True)
class _Job:
    plan: PaintPlan
    target: PaintingTarget
    settings: PainterSettings
    # "paint" measures this sign's brush, wipes the probes off, and then runs
    # the plan; "measure_brush" stops after the measurement.
    mode: str = "paint"
    # Canvas-height fraction of one logical cell, so a measurement can place its
    # probes around the brush the plan will really ask for.  A paint job fills
    # this in from its own plan; a measurement-only job is told.
    cell_fraction: float | None = None
    # The sign as captured right after it was cleared, before any artwork
    # went down.  Verification reads the bare sign's color from it, so a
    # stroke the game dropped is recognised as a hole rather than only when
    # the wood happens to resemble some other palette entry.
    bare_canvas: Any = None
    # The texel grid this job paints on: the one it measured on this sign if
    # the probe could read one, or - when automatic sizing is off, so the
    # probe's brush is never typed - the profile's stored grid, which
    # describes where the sign sat on screen the day it was measured and is
    # only trusted while it still sits on the calibrated rectangle.
    texel_grid: TexelGridModel | None = None
    # The plan's first ``start_stroke`` strokes are taken as already on the
    # sign.  A resumed job paints on a sign it did not clear and measures
    # nothing on it, trusting the profile's stored grid and brush model.
    start_stroke: int = 0
    # Whether this sign proved the Shift-click line tool: a probe stroke drew
    # a line and the capture showed the texels between its endpoints painted.
    # False until proven, so a resumed job, a sizing-off job, or a sign whose
    # probe fails all keep painting long runs as drags.
    line_tool_ok: bool = False
    # When the plan was laid out on the sign's texels (see _adopt_native_plan):
    # for each native stroke, the index of the plan stroke it came from, and
    # how many strokes that plan had.  Progress, the resume record and the
    # start offset all speak in the plan's own strokes.
    stroke_origins: tuple[int, ...] | None = None
    original_total: int | None = None


@dataclass(frozen=True, slots=True)
class _Aiming:
    """How a plan's cells are placed on the screen (see ``Painter._aiming``)."""

    sizing: bool
    model: BrushSizeModel | None
    bias: tuple[float, float]
    paint_canvas: RectangleLike
    clamp_canvas: RectangleLike
    mapper: "Callable[[float, float], tuple[float, float]] | None"
    texel_pitch: float
    # Every cell is one texel and sits exactly on it: the stroke geometry
    # needs no sideways extension, and lone dabs are pure stationary presses.
    native: bool
    # The swept cursor map when the job has one and the plan is native on
    # it: strokes are then looked up texel by texel (see _execute_stroke).
    swept: "TexelGridModel | None" = None


def _describe_seconds(seconds: float) -> str:
    """``45s``, ``12 min``, ``1 h 05 min`` - for log lines and status text."""

    seconds = max(0.0, float(seconds))
    if seconds < 90:
        return f"{int(round(seconds))}s"
    minutes = int(round(seconds / 60.0))
    if minutes < 60:
        return f"{minutes} min"
    hours, minutes = divmod(minutes, 60)
    return f"{hours} h {minutes:02d} min"


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
        screen_capture: Callable[[RectangleLike], Any] | None = None,
        stroke_overhead_seconds: float = DEFAULT_STROKE_OVERHEAD_SECONDS,
        check_capture_seconds: float = CHECK_CAPTURE_SECONDS_DEFAULT,
        check_repaint_fraction: float = CHECK_REPAINT_FRACTION_DEFAULT,
        touch_up_fraction: float = TOUCH_UP_FRACTION_DEFAULT,
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
        self._auto_resume_focus_wait = False
        self._auto_resume_cursor_anchor: tuple[int, int] | None = None
        self._next_auto_focus_retry_at = 0.0
        self._last_cursor_check = 0.0
        self._last_commanded_point: tuple[int, int] | None = None
        self._commanded_history: deque[tuple[int, int]] = deque(
            maxlen=_COMMANDED_POINT_HISTORY
        )
        self._mouse_drift_pixels = 0.0
        self._mouse_drift_started = 0.0
        self._measured_brush_size_model: BrushSizeModel | None = None
        self._measured_texel_grid: TexelGridModel | None = None
        # The press hold this job's probe proved on its sign, or None while
        # unmeasured; applies to stationary presses only.
        self._measured_press_hold_seconds: float | None = None
        # The Size number this job's probe proved lands a lone dab on its
        # sign, or None while unmeasured (then the smallest brush is used).
        self._measured_detail_size: float | None = None
        # Whether a lone dab sweeps across its texel rather than pressing at
        # a point.  False until a probe says this sign needs the sweep.
        self._measured_dab_sweep: bool = False
        # The gap between strokes and the long-drag rate this job's probes
        # proved on its sign, or None while unmeasured (then the floors).
        self._measured_stroke_gap_seconds: float | None = None
        self._measured_drag_rate: float | None = None
        # The bare sign's colour, from a capture this job took of it cleared.
        self._measured_bare_color: RGBColor | None = None
        # How the sign's texture is read back after the download button is
        # clicked: the desktop watcher by default, a simulated sign in tests.
        self._export_watcher = ExportWatcher()
        self._export_reader: "Callable[[], SignExport | None] | None" = None
        self._native_plan: PaintPlan | None = None
        self._last_export: SignExport | None = None
        self._exports_taken = 0
        # Per-cell aim corrections in screen pixels, learned by the touch-up
        # from where a cell's dab actually landed.  Cleared per job.
        self._cell_nudges: dict[tuple[int, int], tuple[int, int]] = {}
        self._last_progress_emit = 0.0
        # Per-stroke overhead learned from earlier runs on this machine; the
        # work schedule prices every stroke with it, so the first time-left
        # shown is already this machine's, not a generic one.
        self._stroke_overhead_seconds = stroke_overhead_seconds
        # What earlier runs showed checking colors and touching up cost, so
        # the time left counts the work that comes after the artwork.
        self._check_capture_seconds = max(0.0, float(check_capture_seconds))
        self._check_repaint_fraction = max(0.0, float(check_repaint_fraction))
        self._touch_up_fraction = max(0.0, float(touch_up_fraction))
        self._paint_phase_timing: PhaseTiming | None = None
        self._touch_up_timing: TouchUpTiming | None = None
        # Set once a pause changes the stroke timing.  The run's predicted
        # seconds were priced on the timing it started with, so its measured
        # pace then says nothing about the machine's per-stroke overhead.
        self._timing_retuned = False
        # When the job last proved to the server it was not idle: the start
        # of the job, and then every anti-AFK break.
        self._last_anti_afk_at = 0.0
        # The painting UI's fingerprint, taken as the job reaches the sign.
        # Suspended through the anti-AFK break, which closes the UI itself.
        self._ui_guard: PaintingUiGuard | None = None
        self._ui_guard_suspended = False
        self._last_ui_check = 0.0
        self._ui_missing_checks = 0
        self._session_ui_scale_applied = False
        self._ui_scale_restore_error: str | None = None
        # Where the panel shows the selected color, found beside the hue bar
        # as the job starts; None while picks are not read back.
        self._swatch: ScreenRect | None = None
        self._color_pick_summary = ColorPickSummary()
        self._confirmation_summary = ConfirmationSummary()
        self._confirmation_seconds = 0.0
        self._check_capture_clock = 0.0
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
    def ui_scale_restore_error(self) -> str | None:
        """Why an emergency cleanup could not restore Rust's normal scale."""

        with self._condition:
            return self._ui_scale_restore_error

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

    def seconds_until_anti_afk(self) -> float | None:
        """How long until the running job's next anti-AFK break, or None.

        None when there is no job, or the job will not take breaks: the
        option is off, or the profile has no Save button to leave the sign
        by.  The clock stops while the job is paused and carries on from
        the same value when it resumes.
        """

        with self._condition:
            job = self._job
            if job is None or self._state not in self._ACTIVE_STATES:
                return None
            settings = job.settings
            if not settings.anti_afk_enabled or job.target.save_button is None:
                return None
            due_at = self._last_anti_afk_at + settings.anti_afk_interval_seconds
            now = time.monotonic()
            if self._state is PainterState.PAUSED and self._paused_at is not None:
                now = self._paused_at
            return max(0.0, due_at - now)

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
        clear_button: RectangleLike | None = None,
        brush_size_model: BrushSizeModel | None = None,
        picker_directions: PickerDirections | None = None,
        start_stroke: int = 0,
    ) -> None:
        """Prepare a job without starting it, suitable for an F8 callback.

        ``start_stroke`` resumes a plan partway: that many strokes, in plan
        order, are taken as already on the sign.  The sign is neither
        cleared nor probed - both would destroy what is there - and the
        brush and grid come from the profile's stored measurements.
        """

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
                    clear_button=clear_button,
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
        if (
            isinstance(start_stroke, bool)
            or not isinstance(start_stroke, int)
            or not 0 <= start_stroke <= total_strokes
        ):
            raise ValueError(
                f"start_stroke must be an integer from 0 to {total_strokes}, "
                f"the plan's stroke count"
            )
        with self._condition:
            if self._state in self._ACTIVE_STATES or (
                self._thread is not None and self._thread.is_alive()
            ):
                raise RuntimeError("Cannot replace a paint job while one is active")
            self._job = _Job(plan, target, resolved_settings, start_stroke=start_stroke)
            self._measured_press_hold_seconds = None
            self._measured_detail_size = None
            self._measured_dab_sweep = False
            self._native_plan = None
            self._measured_stroke_gap_seconds = None
            self._measured_drag_rate = None
            self._measured_bare_color = None
            self._cell_nudges = {}
            self._abort_requested = False
            self._session_ui_scale_applied = False
            self._ui_scale_restore_error = None
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
                start_stroke,
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
        """Prepare a job that measures Rust's Size numbers and stops there.

        Ordinary paint jobs measure the brush themselves, so this is the
        standalone path: seeding a model for the planner before a sign's first
        paint, and scoring a measurement on its own.  The job paints its own
        probe strokes on the calibrated sign and does *not* clear them, so it
        carries a placeholder plan purely to satisfy the shared machinery every
        job runs through.
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
            self._measured_texel_grid = None
            self._measured_press_hold_seconds = None
            self._measured_detail_size = None
            self._measured_dab_sweep = False
            self._measured_stroke_gap_seconds = None
            self._measured_drag_rate = None
            self._measured_bare_color = None
            self._cell_nudges = {}
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

    @property
    def confirmation_summary(self) -> ConfirmationSummary:
        """What checking each color as it went down found, so far or in all."""

        with self._condition:
            return self._confirmation_summary

    @property
    def color_pick_summary(self) -> ColorPickSummary:
        """What reading each color pick back found, so far or in all."""

        with self._condition:
            return self._color_pick_summary

    @property
    def measured_dab_sweep(self) -> bool:
        """Whether lone dabs sweep across their texel on this sign."""

        with self._condition:
            return self._measured_dab_sweep

    @property
    def last_export(self) -> SignExport | None:
        """The sign's texture as last read from Rust's export, if any."""

        with self._condition:
            return self._last_export

    def use_export_reader(self, reader: "Callable[[], SignExport | None] | None") -> None:
        """Read the sign's texture through ``reader`` after each download click.

        For tests driving a simulated sign; a real job reads the desktop.
        """

        self._export_reader = reader

    def _export_sign(self, job: _Job, epoch: int, *, why: str) -> SignExport | None:
        """Click Rust's download button and read the texture it writes.

        The cursor is parked off the sign first; the click is held a frame
        like every other control click; the file that appears is read and
        taken off the desktop.  Returns None when the button is not
        calibrated, the export never appears, or it is the wrong size for
        the plan - a stale file, or a sign other than the one calibrated.
        """

        export = self._export_texture(job, epoch, why=why)
        if export is None:
            return None
        if (export.columns, export.rows) != (job.plan.width, job.plan.height):
            LOGGER.warning(
                "The export is %dx%d but the plan is %dx%d; it is not this sign's "
                "texture at the plan's resolution, so the screen is used for %s",
                export.columns,
                export.rows,
                job.plan.width,
                job.plan.height,
                why,
            )
            return None
        with self._condition:
            self._last_export = export
            self._exports_taken += 1
        LOGGER.info(
            "Read the sign's texture from Rust's export for %s: %d of %d texels painted",
            why,
            int(export.painted.sum()),
            export.painted.size,
        )
        return export

    # A download click the game answers with no file is retried this many
    # times: live, one click in ten produced nothing while the next did.
    _EXPORT_ATTEMPTS = 3
    _EXPORT_HOVER_SECONDS = 0.08
    # The game uploads the sign to the server in the background, bottom rows
    # last, and the download button reads - and re-syncs the client from -
    # the server's copy: an export clicked 0.3 s after a burst of presses
    # discarded the last half-second of them for good (measured live: 19 of
    # 800 lost; none lost after 3 s or 5 s).  So the sign is left alone this
    # long before any export.
    _EXPORT_SETTLE_SECONDS = 3.0

    def _export_texture(self, job: _Job, epoch: int, *, why: str) -> SignExport | None:
        """Click Rust's download button and read the texture it writes, whole.

        The sign's texture at its own size, whatever the plan's resolution;
        :meth:`_export_sign` adds the plan-size check.  None when the button
        is not calibrated or no file appears after a few clicks.
        """

        button = job.target.download_button
        if button is None:
            return None
        target = job.target
        park = self._park_point(target)
        point = normalized_point(button, 0.5, 0.5)
        self._move(park, epoch)
        if self._CAPTURE_SETTLE_SECONDS > 0:  # a simulated sign has no server to sync
            self._interruptible_sleep(
                self._EXPORT_SETTLE_SECONDS, epoch=epoch, check_focus=True
            )
        for attempt in range(self._EXPORT_ATTEMPTS):
            self._checkpoint(epoch=epoch, check_focus=True)
            if self._export_reader is None:
                self._export_watcher.snapshot()
            self._move((int(round(point[0])), int(round(point[1]))), epoch)
            # The button wants a frame with the cursor over it before the
            # press: clicked the instant the cursor arrived, one download
            # click in ten did nothing (live).
            self._interruptible_sleep(
                self._settle(self._EXPORT_HOVER_SECONDS), epoch=epoch, check_focus=True
            )
            self._checkpoint(epoch=epoch, check_focus=True)
            self._mouse_down(epoch)
            try:
                self._interruptible_sleep(
                    self._PICKER_CLICK_HOLD_SECONDS, epoch=epoch, check_focus=True
                )
            finally:
                self.input.mouse_up(MouseButton.LEFT)
            self._move(park, epoch)
            if self._export_reader is not None:
                export = self._export_reader()
            else:
                export = self._export_watcher.collect(
                    sleep=lambda s: self._interruptible_sleep(s, epoch=epoch, check_focus=True)
                )
            if export is not None:
                return export
            LOGGER.warning(
                "No export appeared for %s (click %d of %d)",
                why,
                attempt + 1,
                self._EXPORT_ATTEMPTS,
            )
        LOGGER.warning("No export could be read for %s; falling back to the screen", why)
        return None

    @property
    def measured_bare_color(self) -> RGBColor | None:
        """The bare sign's colour, if this job saw the sign cleared.

        Worth storing on the profile: a later job that resumes onto this
        sign, or touches it up as it stands, has no way to see the wood for
        itself and cannot recognise a hole without it.
        """

        with self._condition:
            return self._measured_bare_color

    @property
    def measured_stroke_gap_seconds(self) -> float | None:
        """The between-strokes gap this job's probe proved, if any."""

        with self._condition:
            return self._measured_stroke_gap_seconds

    @property
    def measured_drag_rate(self) -> float | None:
        """The long-drag rate, in texels per second, this job's probe proved."""

        with self._condition:
            return self._measured_drag_rate

    @property
    def measured_detail_size(self) -> float | None:
        """The Size number this job's probe proved for lone dabs, if any."""

        with self._condition:
            return self._measured_detail_size

    @property
    def measured_press_hold_seconds(self) -> float | None:
        """The press hold this job's probe proved on its sign, if any."""

        with self._condition:
            return self._measured_press_hold_seconds

    @property
    def measured_texel_grid(self) -> TexelGridModel | None:
        """The texel grid the last job painted on, if it had one.

        Measured on the sign by the job itself, or - with automatic sizing
        off - the profile's stored grid it adopted.
        """

        with self._condition:
            return self._measured_texel_grid

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
            self._last_cursor_check = 0.0
            self._reset_mouse_movement_baseline()
            self._last_progress_emit = 0.0
            self._paint_phase_timing = None
            self._timing_retuned = False
            self._last_anti_afk_at = self._started_at
            self._ui_guard = None
            self._ui_guard_suspended = False
            self._last_ui_check = 0.0
            self._ui_missing_checks = 0
            self._swatch = None
            self._color_pick_summary = ColorPickSummary()
            self._confirmation_summary = ConfirmationSummary()
            self._confirmation_seconds = 0.0
            self._check_capture_clock = 0.0
            self._touch_up_timing = None
            total_strokes = sum(
                len(group.strokes) for group in self._job.plan.color_groups
            )
            self._progress = PaintProgress(
                self._state,
                0,
                len(self._job.plan.color_groups),
                0,
                0,
                self._job.start_stroke,
                total_strokes,
                0.0,
                0.0,
                self._initial_estimate(self._job),
                "Starting",
            )
            self._thread = threading.Thread(
                target=self._run,
                name="RustPainterWorker",
                daemon=True,
            )
            thread = self._thread
            job_settings = self._job.settings
        LOGGER.info("Painting worker starting")
        if bool(getattr(self.input, "emits_real_input", True)):
            lifted = fields_below_floor(job_settings)
            if lifted:
                # Once per job, not per stroke: the floors are the game's
                # frame rate, and a setting under one is run at the floor.
                LOGGER.info(
                    "Timing settings below the game's frame floor are run at "
                    "the floor: %s",
                    ", ".join(lifted),
                )
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
            # A successful manual or automatic recovery consumes this
            # one-shot permission. Any later pause must arm itself explicitly.
            self._auto_resume_focus_wait = False
            self._auto_resume_cursor_anchor = None
            now = time.monotonic()
            if self._paused_at is not None:
                paused_for = now - self._paused_at
                self._paused_seconds += paused_for
                # The anti-AFK clock stops with the job: a break is not owed
                # for time spent paused.
                self._last_anti_afk_at += paused_for
            self._paused_at = None
            resumed_state = self._state_before_pause
            self._state = resumed_state
            self._state_reason = "resumed"
            # Force the worker to verify focus again before its very next input.
            self._last_focus_check = 0.0
            # Likewise the painting UI: a pause is when the sign gets closed
            # and reopened, so its next look must not wait out the interval.
            self._last_ui_check = 0.0
            self._ui_missing_checks = 0
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

    def retune(self, settings: PainterSettings | Mapping[str, Any]) -> bool:
        """Give a paused job new timing and safety values for when it resumes.

        A pause is the one moment the user can see the sign and the painter
        at the same time, so it is when a hold that looked too short, or a
        guard that keeps tripping, gets changed.  Only the retunable fields
        are taken from ``settings``; the rest of the job stays as it was
        configured.  Returns False when there is no paused job to retune.
        """

        resolved = (
            settings
            if isinstance(settings, PainterSettings)
            else PainterSettings.from_mapping(settings)
        )
        with self._condition:
            job = self._job
            if job is None or self._state != PainterState.PAUSED:
                return False
            before = job.settings
            job.settings = before.retuned(resolved)
            if StrokeTiming.from_settings(
                job.settings, overhead_seconds=0.0, real_input=True
            ) != StrokeTiming.from_settings(
                before, overhead_seconds=0.0, real_input=True
            ):
                self._timing_retuned = True
        LOGGER.info("Painting settings retuned while paused")
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
            # Every real paint job measures the brush itself, and the probe
            # strokes it paints have to be wiped before the artwork goes down.
            if getattr(self.input, "emits_real_input", True):
                if target.clear_button is None:
                    raise ValueError(
                        "Automatic brush sizing is enabled, but Rust's clear "
                        "control is not calibrated, so the brush calibration "
                        "strokes could not be wiped before painting"
                    )
                if target.clear_button.width <= 0 or target.clear_button.height <= 0:
                    raise ValueError(
                        "Clear control calibration must have positive dimensions"
                    )
            if target.brush_size_model is not None:
                # A stored model is only ever a preview of what the run will
                # measure, but checking it here turns an unpaintable resolution
                # into an error before the countdown instead of after it.
                self._validate_brush_reach(plan, target, settings, target.brush_size_model)
        if settings.anti_afk_enabled and getattr(self.input, "emits_real_input", True):
            if target.save_button is None:
                raise ValueError(
                    "The anti-AFK break is enabled, but Rust's Save button is not "
                    "calibrated, so the painting UI could not be closed to jump"
                )
            if target.save_button.width <= 0 or target.save_button.height <= 0:
                raise ValueError("Save button calibration must have positive dimensions")
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

        diameters = {max(1, int(group.brush_diameter)) for group in plan.color_groups}
        for diameter in sorted(diameters):
            size = self._brush_plan_size(
                target, plan, diameter, settings.logical_pixel_spacing, model
            )
            checks = self._brush_footprint_checks(target, plan, diameter, size, model)
            smallest, largest = model.fitted_range
            if size * 2 < smallest or size > largest * 2:
                LOGGER.warning(
                    "Brush size %s for %d cell(s) sits outside the %s-%s range the "
                    "probes covered, so it is an extrapolation. The next run "
                    "re-measures around this painting resolution.",
                    format_brush_size(size),
                    diameter,
                    format_brush_size(smallest),
                    format_brush_size(largest),
                )
            worst_overshoot = max(
                painted / nominal for _, painted, nominal in checks
            )
            if diameter == 1 and worst_overshoot > self._DETAIL_OVERSHOOT_LIMIT:
                # A calibrated sign must stay paintable: the plan's cells are
                # finer than the game's minimum brush, so detail will soften
                # as neighbouring strokes overlap - a degraded image is the
                # user's call to make, refusing to paint is not.
                rows = max(1, int(model.sign_pixel_rows))
                LOGGER.warning(
                    "Rust's smallest brush covers %.1f logical cells at this "
                    "resolution, so fine detail will blur together. This sign "
                    "resolves about %d rows; at or below that the plan is "
                    "pixel-accurate.",
                    worst_overshoot,
                    rows,
                )
            # Adjacent multi-cell bands overlap one row, which tolerates a brush
            # up to one cell under its nominal footprint; anything narrower
            # leaves stripes the plan already counts as covered.
            for axis, painted, nominal in checks:
                if axis == "columns" and model.has_horizontal_model:
                    # Row-sized brushes cover the columns by stroke extension,
                    # so a narrow footprint here is by design, not a stripe.
                    continue
                if diameter > 1 and painted < nominal * (diameter - 1) / diameter:
                    raise ValueError(
                        f"A {diameter}-cell brush needs {nominal:.0f}px across its "
                        f"{axis} but the Size field reaches only {painted:.0f}px "
                        "here. Choose a lower optimization mode or a higher "
                        "painting resolution."
                    )

    def _run(self) -> None:
        if getattr(self.input, "emits_real_input", True):
            with _high_resolution_timer(), _above_normal_priority():
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
            self._apply_session_ui_scale(job)
            self._confirm_painting_ui(job)
            job.target = self._measured_picker_target(job.target)
            self._select_brush(job)
            self._locate_color_swatch(job)
            if job.mode == "measure_brush":
                measured = self._measure_brush_size_model(job)
                with self._condition:
                    self._measured_brush_size_model = measured
                job.target = replace(job.target, brush_size_model=measured)
                grid = self._measure_texel_grid_safely(job)
                with self._condition:
                    self._measured_texel_grid = grid
            else:
                if job.start_stroke > 0:
                    self._prepare_resumed_sign(job)
                else:
                    self._calibrate_brush_for_plan(job)
                self._update_progress_state(
                    PainterState.RUNNING, "Painting", phase="paint"
                )
                self._execute_plan(job)
                self._verify_and_touch_up(job)
            self._checkpoint(check_focus=True)
            self._restore_session_ui_scale(job)
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
            job = self._job
            if job is not None and self._session_ui_scale_applied:
                self._restore_session_ui_scale(job, best_effort=True)
            self._safe_release_all()
            with self._condition:
                self._paused_at = None
                self._condition.notify_all()

    def _apply_session_ui_scale(self, job: _Job) -> None:
        settings = job.settings
        if not settings.manage_ui_scale or not getattr(self.input, "emits_real_input", True):
            return
        from .game_console import set_ui_scale

        self._update_progress_state(PainterState.RUNNING, "Setting Rust UI scale")
        # Mark restoration necessary before the first console keystroke. If
        # Stop lands after Rust applies the command but before this method
        # returns, the finally block must still put the user's scale back.
        with self._condition:
            self._session_ui_scale_applied = True
            self._ui_scale_restore_error = None
        requirement = ForegroundRequirement(
            title_contains=settings.expected_window_title_contains or None,
            executable=settings.expected_process_name or None,
        )

        def foreground_checkpoint() -> None:
            if not self._foreground_checker(requirement):
                raise RuntimeError("Rust stopped being the foreground window")

        self._ui_guard_suspended = True
        try:
            set_ui_scale(
                self.input,
                settings.painting_ui_scale,
                console_key=settings.console_key,
                checkpoint=lambda: self._checkpoint(check_focus=True),
                close_checkpoint=foreground_checkpoint,
                sleep=lambda seconds: self._interruptible_sleep(seconds, check_focus=True),
            )
        finally:
            self._ui_guard_suspended = False
        LOGGER.info("Rust UI scale set to %.2f for this paint session", settings.painting_ui_scale)

    def _restore_session_ui_scale(self, job: _Job, *, best_effort: bool = False) -> None:
        if not self._session_ui_scale_applied:
            return
        settings = job.settings
        from .game_console import set_ui_scale

        requirement = ForegroundRequirement(
            title_contains=settings.expected_window_title_contains or None,
            executable=settings.expected_process_name or None,
        )

        def foreground_checkpoint() -> None:
            if not self._foreground_checker(requirement):
                raise RuntimeError("Rust was not foreground when the session stopped")

        try:
            if best_effort:
                foreground_checkpoint()
                checkpoint = foreground_checkpoint
                sleeper = time.sleep
            else:
                checkpoint = lambda: self._checkpoint(check_focus=True)
                sleeper = lambda seconds: self._interruptible_sleep(seconds, check_focus=True)
            self._ui_guard_suspended = True
            try:
                set_ui_scale(
                    self.input,
                    settings.normal_ui_scale,
                    console_key=settings.console_key,
                    checkpoint=checkpoint,
                    close_checkpoint=foreground_checkpoint,
                    sleep=sleeper,
                )
            finally:
                self._ui_guard_suspended = False
        except BaseException as exc:
            if not best_effort:
                raise
            message = str(exc) or type(exc).__name__
            with self._condition:
                self._ui_scale_restore_error = message
            LOGGER.error("Could not restore Rust UI scale: %s", message)
            return
        with self._condition:
            self._session_ui_scale_applied = False
            self._ui_scale_restore_error = None
        LOGGER.info("Rust UI scale restored to %.2f", settings.normal_ui_scale)

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

    def _adopt_stored_texel_grid(self, job: _Job) -> None:
        """Paint on the grid an earlier job measured, if it still fits.

        With automatic sizing off the probe cannot run - it types the
        smallest brush and needs the sign wiped afterwards - so the choice is
        between the stored grid and the calibrated rectangle.  The rectangle
        is hand-dragged: live it missed the texture's cursor lattice by a
        couple of texels across and walked a fraction of a texel down the
        sign, which at native resolution put every other row on a texel
        boundary and left it bare.  The stored grid was counted on the sign,
        and as long as it still sits on the rectangle it is the better aim.
        A sign re-framed since would need the rectangle re-dragged, which
        moves the grid off it and back onto the rectangle's own layout.
        """

        settings = job.settings
        grid = job.target.texel_grid
        if not settings.measure_texel_grid or grid is None:
            if self._plan_is_finer_than_rectangle_aim(job):
                LOGGER.warning(
                    "Automatic brush sizing is off and no texel grid has been "
                    "measured on this sign, so strokes are laid out on the "
                    "calibration rectangle - good to about half a texel, which "
                    "at this resolution can leave whole rows bare. Turn "
                    "automatic brush sizing on to measure the sign's grid."
                )
            return
        if not grid.agrees_with(job.target.canvas):
            LOGGER.warning(
                "The stored texel grid (%dx%d texels from %.0f, %.0f, measured %s) "
                "no longer sits on the calibrated rectangle, so strokes are laid "
                "out on the rectangle instead; turn automatic brush sizing on to "
                "measure the sign again",
                grid.columns,
                grid.rows,
                grid.origin_x,
                grid.origin_y,
                grid.captured_at or "earlier",
            )
            return
        job.texel_grid = grid
        with self._condition:
            self._measured_texel_grid = grid
        LOGGER.info(
            "This job stamps no probe of its own; painting on the texel grid "
            "measured %s: %dx%d texels, %.4f x %.4f px each from %.2f, %.2f",
            grid.captured_at or "earlier",
            grid.columns,
            grid.rows,
            grid.pitch_x,
            grid.pitch_y,
            grid.origin_x,
            grid.origin_y,
        )

    @staticmethod
    def _plan_is_finer_than_rectangle_aim(job: _Job) -> bool:
        """Whether half-texel aim would visibly cost this plan rows or columns.

        The rectangle's layout errs by a fraction of a texel; a cell several
        texels wide absorbs that, a cell one or two texels wide does not.
        Without a measurement the texel count is taken from the brush model
        when there is one, else the plan is assumed fine enough to matter.
        """

        model = job.target.brush_size_model
        plan = job.plan
        if model is None:
            return True
        rows = model.sign_pixel_rows
        if not math.isfinite(rows) or rows <= 0:
            return True
        return plan.height * 2.0 >= rows

    def _reference_bare_sign(self, job: _Job) -> None:
        """Capture the sign before painting as the touch-up pass's bare reference.

        The calibration step wipes the sign and keeps that capture; without it
        a dropped stroke reads as "some other color" rather than as bare sign,
        and when the capture cannot resolve single cells those verdicts are
        set aside - live, thirty bare rows went unrepaired that way.  The sign
        as it stands before the first stroke is nearly always the cleared sign
        the user is painting over, so it serves, after a check that it really
        is one surface and not an earlier artwork.
        """

        import numpy as np

        from .verification import capture_looks_bare, sample_cell_colors

        if job.bare_canvas is not None or job.settings.verify_passes <= 0:
            return
        target = job.target
        canvas = ScreenRect(
            target.canvas.left,
            target.canvas.top,
            target.canvas.width,
            target.canvas.height,
        )
        park = self._park_point(target)
        for _attempt in range(self._CALIBRATION_ATTEMPTS):
            # A pause hands the mouse back partway through; the capture is
            # simply taken again once painting resumes.
            epoch = self._pause_generation_value()
            try:
                capture = self._capture_parked(canvas, park, epoch)
                sampled = sample_cell_colors(
                    np.asarray(capture.convert("RGB"), dtype=np.float32),
                    job.plan.width,
                    job.plan.height,
                    centers=self._grid_cell_centers(job, job.plan, canvas),
                )
                bare = capture_looks_bare(sampled)
                break
            except _RetryAction:
                continue
            except _AbortRequested:
                raise
            except Exception:
                LOGGER.exception("The sign could not be captured before painting")
                return
        else:
            LOGGER.info(
                "The sign was not captured before painting (paused every time "
                "it was tried), so the touch-up pass has no bare reference"
            )
            return
        if bare:
            job.bare_canvas = capture
            self._remember_bare_color(capture)
            LOGGER.info(
                "The sign looks bare before painting; the touch-up pass will "
                "read holes against this capture"
            )
        else:
            LOGGER.info(
                "The sign is not bare before painting, so the touch-up pass has "
                "no bare reference and can only repaint cells that match no color"
            )

    @staticmethod
    def _brush_target_fraction(
        target: PaintingTarget,
        plan: PaintPlan,
        diameter_cells: int,
        spacing: float,
        model: BrushSizeModel | None = None,
    ) -> float:
        """The canvas-height fraction a ``diameter_cells`` brush should paint.

        The plan is stretched across the whole canvas, so one row is
        ``1/height`` of it and one column ``1/width``.  A round or square brush
        spans the same distance both ways and therefore has to cover whichever
        pitch is *wider*: cells are only square when the calibrated rectangle's
        aspect divides evenly into the plan's, and sizing to the narrow axis
        leaves a bare stripe along every seam of the wide one.  Overshooting
        the narrow axis instead is invisible - the later-painted color simply
        owns the shared texels.  The result is expressed against the canvas
        height because that is the axis brush calibration measured.  (When the
        model carries a horizontal measurement, :meth:`_brush_plan_size` sizes
        each axis against its own data instead of using this square-footprint
        assumption.)

        A one-cell brush targets the full pitch plus a fraction of a sign
        texel.  The sign renders every stroke snapped to its own texture rows,
        so a brush sized exactly to the pitch still comes out narrow on the
        rows where snapping lands low - and those rows show as bare stripes
        across the painting.  (An earlier version undershot to 90% instead,
        and the stripes were plainly visible in game.)

        How much overlap the hedge needs depends on how far the plan's cell
        grid can drift from the texel grid.  With cells two texels or coarser,
        cell boundaries land at arbitrary fractional texel positions, and half
        a texel - proven in game - is what closes the worst case.  At native
        resolution one cell *is* one texel, the grids line up by construction,
        and the only residual error is canvas-calibration slop; a quarter
        texel still bridges a snapped-away row there, while bleeding half as
        far into neighbouring cells whose detail is the whole point of
        painting at native resolution.  The taper between those anchors is
        linear.  Either way the overlap costs nothing visible at its own
        scale: boundaries are texel-quantized regardless, and the
        later-painted color simply owns the shared texel.
        """

        canvas = target.canvas
        pitch = max(canvas.width / plan.width, canvas.height / plan.height)
        span = pitch * diameter_cells * min(spacing, 1.0)
        fraction = span / canvas.height
        if diameter_cells <= 1 and model is not None and model.slope > 0:
            texels_per_cell = fraction / model.slope
            overlap = min(0.5, max(0.25, 0.25 * texels_per_cell))
            fraction += overlap * model.slope
        return fraction

    @staticmethod
    def _axis_brush_fraction(
        cells: int, diameter_cells: int, spacing: float, slope: float
    ) -> float:
        """The fraction of one canvas axis a ``diameter_cells`` brush should paint.

        One cell is ``1/cells`` of the axis, so the wanted span is dimensionless
        before the canvas size ever enters.  The overlap hedge is the same
        half-to-quarter texel taper :meth:`_brush_target_fraction` documents,
        expressed in this axis's own texels.
        """

        fraction = (diameter_cells * min(spacing, 1.0)) / cells
        if diameter_cells <= 1 and slope > 0:
            texels_per_cell = fraction / slope
            overlap = min(0.5, max(0.25, 0.25 * texels_per_cell))
            fraction += overlap * slope
        return fraction

    @classmethod
    def _brush_plan_size(
        cls,
        target: PaintingTarget,
        plan: PaintPlan,
        diameter_cells: int,
        spacing: float,
        model: BrushSizeModel,
    ) -> float:
        """The Size number to type for a ``diameter_cells`` pass of this plan.

        With a horizontal measurement the brush is sized to the *rows* alone:
        Rust's brush is square in the sign's own texels, so on cells wider
        than tall a row-exact brush undercovers the columns - and that gap is
        closed by geometry instead, extending each stroke sideways by the
        shortfall (:meth:`_stroke_extension_pixels`).  Rows can never be
        stretched that way (strokes are horizontal), so the vertical pitch is
        the one the Size number must honor exactly.  Without a horizontal
        measurement, the vertical model is read under the older
        square-in-screen-pixels assumption and sized to the wider pitch.
        """

        if model.has_horizontal_model:
            return model.clamped_size_for_fraction(
                cls._axis_brush_fraction(
                    plan.height, diameter_cells, spacing, model.slope
                )
            )
        return model.clamped_size_for_fraction(
            cls._brush_target_fraction(target, plan, diameter_cells, spacing, model)
        )

    @staticmethod
    def _stroke_extension_pixels(
        canvas: RectangleLike,
        plan: PaintPlan,
        model: BrushSizeModel | None,
        size: float,
    ) -> float:
        """How far each stroke end reaches out so cells are covered edge-to-edge.

        The brush is sized to the rows, so on cells wider than tall it paints
        a band narrower than the cell at each stroke end.  Dragging that much
        further out (a dab becomes a tiny horizontal drag) covers the full
        cell width with no vertical overshoot at all - the trick a human sign
        painter uses when the roller is narrower than the board.  ``canvas``
        is whatever rectangle the strokes are being laid out on, so the pitch
        matches the grid the endpoints came from.
        """

        if model is None or not model.has_horizontal_model:
            return 0.0
        pitch_x = canvas.width / plan.width
        texel_x = model.slope_x * canvas.width
        brush_width = model.fraction_x_for_size(size) * canvas.width
        # The half-texel hedge mirrors the vertical one: stamps snap to whole
        # texels, and a boundary reached exactly can still round away.
        return max(0.0, (pitch_x + 0.5 * texel_x - brush_width) / 2.0)

    @staticmethod
    def _registered_canvas(
        canvas: RectangleLike, model: BrushSizeModel | None
    ) -> RectangleLike:
        """Stretch the stroke grid to the sign texture's canonical extent.

        The calibrated rectangle covers the sign only to hand-drag precision -
        318.4 of a 320-column texture in live measurement - so cells laid out
        on the rectangle are a fraction of a texel narrower than the texture's
        own grid.  Stamps land on whole texels, and that fraction accumulates:
        every dozen cells the rounding slips one texel and a later neighbour's
        stamp eats a texel of the cell before it, which a sign texture
        downloaded from the game showed as painted cells of visibly uneven
        width.  Anchoring at the rectangle's origin and stretching the grid to
        ``canonical texels x measured texel size`` makes the cell pitch exact,
        so stamps tile uniformly however many cells the plan has.
        """

        if model is None or not model.has_horizontal_model:
            return canvas
        rows = canonical_texture_rows(model.sign_pixel_rows)
        columns = canonical_texture_rows(model.sign_pixel_columns)
        if rows < 8 or columns < 8:
            return canvas
        width = columns * model.slope_x * canvas.width
        height = rows * model.slope * canvas.height
        # A canonical guess far from the rectangle would stretch the artwork
        # off the sign; hand-drag slop is a couple of texels, not a tenth.
        if not (0.9 <= width / canvas.width <= 1.1):
            return canvas
        if not (0.9 <= height / canvas.height <= 1.1):
            return canvas
        return SimpleNamespace(
            left=canvas.left, top=canvas.top, width=width, height=height
        )

    def _select_brush(self, job: _Job) -> None:
        """Click the profile's calibrated solid brush before any probes or paint."""

        shape = job.settings.brush_shape
        if shape == "auto":
            # A small logical grid is intentional pixel art; its square cells
            # need the square stamp.  Larger grids benefit from the smoother
            # circular edge.  The GUI normally resolves auto before start.
            shape = "square" if min(job.plan.width, job.plan.height) <= 64 else "circle"
        button = (
            job.target.square_brush_button
            if shape == "square"
            else job.target.circle_brush_button
        )
        if button is None:
            raise ValueError(
                f"Calibrate Rust's {shape} brush button before painting."
            )
        epoch = self._pause_generation
        self._safe_click(button.center, epoch)
        self._interruptible_sleep(
            self._settle(job.settings.delay_after_brush_seconds),
            epoch=epoch,
            check_focus=True,
        )
        LOGGER.info("Selected the calibrated %s brush", shape)

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
        grid = job.texel_grid
        if (
            diameter_cells <= 1
            and grid is not None
            and abs(job.plan.width - grid.columns) <= 2
            and abs(job.plan.height - grid.rows) <= 2
        ):
            # Native resolution on a measured grid: every cell is one texel
            # and sits exactly on it, so the brush is the smallest the game
            # has - one texel.  The half-texel overlap the model would add
            # exists to bridge a grid that is off by a fraction of a texel,
            # and on an exact grid it does the opposite: a 1.1-texel stamp
            # spills into the neighbour whenever the cursor sits in the
            # outer part of its texel (live: a tenth of the dots of a test
            # lattice landed a texel over for exactly that reason).  A plan
            # a texel or two off the grid still counts as native: requiring
            # exact equality let a one-column probe miscount silently swap
            # this one-texel dot for a 1.24-texel one, which chewed fine
            # detail across a whole 5.6-hour run.  When this sign's dab probe
            # found the smallest brush missing lone dabs and proved a larger
            # Size that lands them, that Size is the one-cell brush instead.
            size = BRUSH_SIZE_MIN
            measured = self._measured_detail_size
            if measured is not None and settings.measure_dab_size and not grid.swept:
                size = max(size, measured)
        else:
            size = self._brush_plan_size(
                job.target, job.plan, diameter_cells, settings.logical_pixel_spacing, model
            )
        self._update_progress_state(
            PainterState.RUNNING,
            f"Brush size {format_brush_size(size)} for {diameter_cells} logical "
            f"cell{'s' if diameter_cells != 1 else ''}",
        )
        self._write_brush_size(box, size, settings, epoch)
        LOGGER.info(
            "Brush size %s typed for %d cell(s): paints %.5f of the sign height%s",
            format_brush_size(size),
            diameter_cells,
            model.fraction_for_size(size),
            (
                f", {model.fraction_x_for_size(size):.5f} of its width"
                if model.has_horizontal_model
                else ""
            ),
        )

    # The OEM period key, for the decimal point in sizes such as "1.35".
    _VK_PERIOD = 0xBE

    # Rust has been observed running its painting UI at 15 FPS, where a press
    # and release inside one 67 ms frame can be sampled as nothing at all.  A
    # dropped digit with the field unfocused is a hotbar key, so every
    # keystroke is held across a frame boundary and separated from the next.
    _KEY_HOLD_SECONDS = KEY_HOLD_SECONDS
    _KEY_GAP_SECONDS = KEY_GAP_SECONDS

    def _press_field_key(self, key: int | str, epoch: int) -> None:
        self._checkpoint(epoch=epoch, check_focus=True)
        if self.input.emits_real_input:
            self.input.press_key(key, hold_seconds=self._KEY_HOLD_SECONDS)
            self._interruptible_sleep(
                self._KEY_GAP_SECONDS, epoch=epoch, check_focus=True
            )
        else:
            self.input.press_key(key)

    def _write_brush_size(
        self,
        box: RectangleLike,
        size: float,
        settings: PainterSettings,
        epoch: int,
    ) -> None:
        """Focus Rust's Size field, replace its contents, and commit with Enter.

        The field holds hundredths from 1.00 to 100.00 (verified by typing and
        photographing it), so sizes are written as decimals: at the detail end
        the gap between 1 and 2 is the difference between a correct brush and
        one twice as wide as its cell.
        """

        settle = (
            self._settle(settings.delay_after_brush_seconds)
            if self.input.emits_real_input
            else 0.0
        )
        self._safe_click(
            normalized_point(box, 0.5, 0.5),
            epoch,
            hold_floor=self._PICKER_CLICK_HOLD_SECONDS,
        )
        self._interruptible_sleep(settle, epoch=epoch, check_focus=True)
        # The field holds at most six characters, and clearing from both sides
        # of the caret empties it wherever the click happened to place it.
        for key in ("BACKSPACE",) * 6 + ("DELETE",) * 6:
            self._press_field_key(key, epoch)
        for char in format_brush_size(size):
            self._press_field_key(self._VK_PERIOD if char == "." else char, epoch)
        self._press_field_key("ENTER", epoch)
        self._interruptible_sleep(settle, epoch=epoch, check_focus=True)

    # A first, throwaway stroke used only to learn the scale of the sign, so
    # the real probes can be placed around the brush the plan will ask for.
    _BRUSH_SCOUT_SIZE = 24.0

    # Probes are placed at these multiples of the brush one logical cell needs,
    # which brackets both the detail brush and the optimizer's multi-cell fill
    # passes (up to 7 cells).  Fitting a line and then reading it far outside
    # its data is what made an earlier version answer 2 where 5 was wanted: a
    # couple of pixels of error is nothing at size 60 and is the entire answer
    # at size 2.  A probe that runs off the sign is skipped as clipped, so the
    # wide end costs nothing on a close-up sign.
    _BRUSH_PROBE_MULTIPLES = (8.0, 4.0, 2.0, 1.0, 0.5)

    # Fallback ladder for a measurement with no resolution to aim at.
    _BRUSH_FALLBACK_SIZES = (32.0, 16.0, 8.0, 4.0)

    # One color per probe, so a stroke drawn inside a previous, wider band still
    # reads as a change against the capture taken just before it.
    _BRUSH_PROBE_COLORS = (
        (255, 0, 255),
        (0, 255, 0),
        (255, 200, 0),
        (0, 200, 255),
        (255, 80, 80),
    )

    @classmethod
    def _probe_color_against(
        cls, capture: Any, points: "Sequence[tuple[float, float]]", skip: RGBColor | None = None
    ) -> RGBColor:
        """The probe colour that will show best against what is on the sign.

        Probes are read as the difference between two captures, so a probe in
        the colour already sitting where it lands reads as nothing: live, a
        second measurement on an uncleared sign found its scout stroke and
        its scout stamp both "did not change the sign", and the run fell
        back to guesswork.  Each probe therefore picks, from the table, the
        colour farthest from the pixels under its points; ``skip`` keeps two
        consecutive probes at the same place from choosing alike.
        """

        import numpy as np

        pixels = np.asarray(capture.convert("RGB"), dtype=np.float32)
        height, width = pixels.shape[:2]
        samples = []
        for x, y in points:
            column = min(max(int(round(x)), 0), width - 1)
            row = min(max(int(round(y)), 0), height - 1)
            samples.append(pixels[row, column])
        if not samples:
            return cls._BRUSH_PROBE_COLORS[0]
        under = np.array(samples, dtype=np.float32)
        best = None
        for color in cls._BRUSH_PROBE_COLORS:
            if skip is not None and color == skip:
                continue
            distance = float(np.linalg.norm(under - np.array(color, dtype=np.float32), axis=1).min())
            if best is None or distance > best[0]:
                best = (distance, color)
        return best[1] if best is not None else cls._BRUSH_PROBE_COLORS[0]

    def _calibrate_brush_for_plan(self, job: _Job) -> None:
        """Validate or measure this sign's brush, then start from a clear sign.

        A saved model is reused only when a passive edge check shows its texel
        grid still fits the visible sign. Any missing or uncertain evidence
        falls through to fresh probe strokes, which are erased before artwork.
        """

        settings = job.settings
        if not self.input.emits_real_input:
            return
        if not settings.apply_brush_size:
            # Nothing is typed into Rust's Size field and nothing is stamped:
            # the brush is whatever the user set by hand, and so is the sign.
            # The aim still needs a grid, and the sign needs a bare reference
            # for the touch-up pass to tell a dropped stroke from a wrong
            # color - both come from what is already at hand.
            self._adopt_stored_texel_grid(job)
            self._reference_bare_sign(job)
            return
        if job.target.brush_size_box is None or job.target.clear_button is None:
            # ``_validate_job`` refuses this combination up front; a job that
            # reaches here without the rectangles came from a caller that built
            # its own target, and painting on a guessed brush is worse than not.
            raise RuntimeError(
                "Automatic brush sizing needs Rust's Size field and clear "
                "control calibrated"
            )
        job.cell_fraction = self._brush_target_fraction(
            job.target, job.plan, 1, settings.logical_pixel_spacing
        )
        if self._reuse_saved_calibration(job):
            return
        for attempt in range(self._CALIBRATION_ATTEMPTS):
            try:
                model = self._measure_brush_size_model(job)
                job.target = replace(job.target, brush_size_model=model)
                with self._condition:
                    self._measured_brush_size_model = model
                job.texel_grid = self._sweep_cursor_map_safely(job)
                if job.texel_grid is None:
                    job.texel_grid = self._measure_texel_grid_safely(job)
                with self._condition:
                    self._measured_texel_grid = job.texel_grid
                if job.texel_grid is not None and job.texel_grid.swept:
                    # The one exact stamp and a measured map: the plan goes
                    # down texel by texel, and the only timing left to prove
                    # is how fast the game takes a press.
                    self._adopt_native_plan(job)
                    self._probe_press_timing_by_export(job)
                    self._clear_canvas(job)
                    break
                if job.texel_grid is None and job.target.texel_grid is not None:
                    # The probe is allowed to fail on a hard sign (live: the
                    # fine-pitch XXL probe succeeds most runs, not all); a
                    # grid an earlier run measured on this same sign is a far
                    # better aim than the brush-model fallback, which at
                    # native resolution scatters detail by half a texel.
                    stored = job.target.texel_grid
                    if stored.agrees_with(job.target.canvas):
                        job.texel_grid = stored
                        with self._condition:
                            self._measured_texel_grid = stored
                        LOGGER.warning(
                            "Painting on the texel grid measured %s (%dx%d "
                            "texels): this run's own probe failed and the "
                            "stored grid still sits on the rectangle",
                            stored.captured_at or "earlier",
                            stored.columns,
                            stored.rows,
                        )
                self._probe_press_hold(job)
                self._probe_stroke_gap(job)
                self._probe_dab_size(job)
                self._probe_drag_rate(job)
                job.line_tool_ok = self._probe_line_tool(job)
                self._clear_canvas(job)
                break
            except _RetryAction:
                # A pause handed the mouse back partway through, so the probes
                # after it describe a sign somebody may have been drawing on
                # themselves. Measure the whole thing again once painting
                # resumes rather than fit a line through both halves.
                LOGGER.info(
                    "Brush calibration was interrupted (attempt %d); measuring again",
                    attempt + 1,
                )
        else:
            raise RuntimeError(
                "The brush measurement was interrupted every time it was tried. "
                "Let the job run without pausing it during the first few strokes."
            )
        # Checked against the model that will actually be typed, so a sign the
        # plan cannot be painted on is refused before any artwork goes down.
        self._validate_brush_reach(job.plan, job.target, settings, model)

    def _reuse_saved_calibration(self, job: _Job) -> bool:
        """Use saved measurements only while visible sign edges still match."""

        import numpy as np

        settings = job.settings
        target = job.target
        model = target.brush_size_model
        grid = target.texel_grid
        if (
            not settings.reuse_calibration
            or model is None
            or grid is None
            or not grid.agrees_with(target.canvas)
            or (
                settings.measure_dab_size
                and self._plan_needs_dab_probe(job.plan)
                and target.cached_detail_size is None
            )
        ):
            return False
        canvas = ScreenRect(
            target.canvas.left,
            target.canvas.top,
            target.canvas.width,
            target.canvas.height,
        )
        margin = max(
            self._EDGE_MARGIN_MIN_PIXELS,
            int(round(self._EDGE_MARGIN_FRACTION * max(canvas.width, canvas.height))),
        )
        wide = ScreenRect(
            canvas.left - margin,
            canvas.top - margin,
            canvas.width + 2 * margin,
            canvas.height + 2 * margin,
        )
        park = self._park_point(target)
        epoch = self._pause_generation_value()
        try:
            self._update_progress_state(
                PainterState.RUNNING,
                "Checking the saved sign measurement",
                phase="calibrate",
            )
            self._move(park, epoch)
            if self.input.emits_real_input:
                self._interruptible_sleep(
                    self._CAPTURE_SETTLE_SECONDS, epoch=epoch, check_focus=True
                )
            capture = np.asarray(
                self._screen_capture(wide).convert("RGB"), dtype=np.float32
            )
            registered = grid.registered_rect()
            expected = (
                registered.left - wide.left,
                registered.top - wide.top,
                registered.left + registered.width - wide.left,
                registered.top + registered.height - wide.top,
            )
            found = find_quad_edges(capture, expected, max(4, margin - 2))
        except (_RetryAction, _AbortRequested):
            raise
        except Exception as exc:
            LOGGER.info("Saved calibration could not be checked (%s); measuring fully", exc)
            return False
        tolerance = max(4.0, 2.5 * max(grid.pitch_x, grid.pitch_y))
        comparisons = [
            abs(actual - wanted)
            for actual, wanted in zip(found, expected)
            if actual is not None
        ]
        if len(comparisons) < 3 or max(comparisons, default=float("inf")) > tolerance:
            LOGGER.info(
                "Saved calibration did not match the visible sign (%d edges, worst %s px); "
                "measuring fully",
                len(comparisons),
                "?" if not comparisons else f"{max(comparisons):.1f}",
            )
            return False

        if grid.swept:
            # A swept map is reused only after it proves itself the way a
            # fresh one does: a lattice of dabs through it, read back from
            # the export.  A sign that has moved by a pixel fails it.
            if target.download_button is None or not self._check_stored_cursor_map(job, grid):
                LOGGER.info("The stored cursor map did not prove out; measuring fully")
                return False
        job.texel_grid = grid
        self._measured_brush_size_model = model
        self._measured_texel_grid = grid
        self._measured_press_hold_seconds = target.cached_press_hold_seconds
        self._measured_stroke_gap_seconds = target.cached_stroke_gap_seconds
        self._measured_drag_rate = target.cached_drag_rate
        self._measured_detail_size = target.cached_detail_size
        self._measured_dab_sweep = target.cached_dab_sweep
        self._validate_brush_reach(job.plan, target, settings, model)
        LOGGER.info(
            "Saved brush and %dx%d texel grid match the visible sign; skipping full probes",
            grid.columns,
            grid.rows,
        )
        self._clear_canvas(job)
        return True

    def _check_stored_cursor_map(self, job: _Job, grid: TexelGridModel) -> bool:
        """Dab a lattice through a stored map and read the export: all exact?"""

        import numpy as np

        target = job.target
        settings = job.settings
        box = target.brush_size_box
        if box is None:
            return False
        try:
            epoch = self._pause_generation_value()
            self._update_progress_state(
                PainterState.RUNNING, "Proving the stored cursor map", phase="calibrate"
            )
            self._write_brush_size(box, BRUSH_SIZE_MIN, settings, epoch)
            self._clear_canvas(job, quiet=True)
            epoch = self._pause_generation_value()
            self._select_color(self._SWEEP_COLOR, target, settings, epoch, apply_correction=False)
            targets = lattice_targets(grid.columns, grid.rows, *self._SWEEP_CHECK_LATTICE)
            hold = min(self._SWEEP_HOLD_SECONDS, self._MIN_PRESS_SECONDS)
            gap = min(self._SWEEP_GAP_SECONDS, self._STROKE_GAP_FLOOR_SECONDS)
            for u, v in targets:
                point = grid.aim_pixel(u, v)
                self._screen_stroke(point, point, settings, epoch, hold_seconds=hold)
                self._interruptible_sleep(gap, epoch=epoch, check_focus=True)
            export = self._export_texture(job, epoch, why="the stored cursor map check")
            if export is None or (export.columns, export.rows) != (grid.columns, grid.rows):
                return False
            exact, wrong = check_lattice(np.where(export.painted, 255, 0), targets)
            if wrong:
                epoch = self._pause_generation_value()
                for u, v in wrong:
                    point = grid.aim_pixel(u, v)
                    self._screen_stroke(point, point, settings, epoch, hold_seconds=hold)
                    self._interruptible_sleep(gap, epoch=epoch, check_focus=True)
                export = self._export_texture(job, epoch, why="the stored cursor map re-check")
                if export is None:
                    return False
                exact, wrong = check_lattice(np.where(export.painted, 255, 0), targets)
            LOGGER.info(
                "Stored cursor map check: %d of %d lattice dabs on their texel%s",
                exact,
                len(targets),
                "" if not wrong else f"; missed {wrong[:6]}",
            )
            return not wrong
        except (_RetryAction, _AbortRequested):
            raise
        except Exception as exc:
            LOGGER.info("The stored cursor map could not be checked (%s)", exc)
            return False

    def _prepare_resumed_sign(self, job: _Job) -> None:
        """Take a half-painted sign as it is: no clear, no probe.

        A resumed job's sign already carries hours of strokes.  The probe
        would paint over them and the clear would wipe them, so neither
        runs; the job paints on the texel grid and with the brush model the
        profile stored from an earlier job on this sign - the same trust the
        sizing-off path extends - and says so when either is missing, since
        then the strokes are aimed by the hand-dragged rectangle and the
        brush is whatever Rust has set.  The touch-up pass at the end reads
        the sign without a bare reference, which is how it reads any sign
        it did not see cleared.
        """

        settings = job.settings
        if not self.input.emits_real_input:
            return
        self._update_progress_state(
            PainterState.RUNNING,
            f"Resuming from stroke {job.start_stroke:,} on the sign as it is",
            phase="calibrate",
        )
        LOGGER.info(
            "Resuming at stroke %d of %d: the sign is neither cleared nor "
            "probed, so the stored grid and brush model are used as they are",
            job.start_stroke,
            sum(len(group.strokes) for group in job.plan.color_groups),
        )
        if job.target.texel_grid is not None:
            self._adopt_stored_texel_grid(job)
            self._adopt_native_plan(job)
        else:
            LOGGER.warning(
                "No texel grid is stored for this profile, so the resumed "
                "strokes are laid out on the calibration rectangle - good to "
                "about half a texel, which can leave whole rows bare. A fresh "
                "job with automatic brush sizing on measures and stores one."
            )
        target = job.target
        if settings.measure_press_hold:
            # The press hold, stroke gap and drag rate an earlier job proved
            # are the sign's and the machine's, not this painting's - the
            # same trust the stored grid and brush model above are extended -
            # and without them every resumed stroke pays the full frame-floor
            # timings, which on a dab-heavy touch-up more than doubles it.
            adopted = []
            if target.cached_press_hold_seconds is not None:
                self._measured_press_hold_seconds = target.cached_press_hold_seconds
                adopted.append(f"{target.cached_press_hold_seconds * 1000.0:.0f} ms press hold")
            if target.cached_stroke_gap_seconds is not None:
                self._measured_stroke_gap_seconds = target.cached_stroke_gap_seconds
                adopted.append(f"{target.cached_stroke_gap_seconds * 1000.0:.0f} ms stroke gap")
            if target.cached_drag_rate is not None:
                self._measured_drag_rate = target.cached_drag_rate
                adopted.append(f"{target.cached_drag_rate:.0f} texels/s drags")
            if adopted:
                LOGGER.info(
                    "Resuming with the timings an earlier job proved on this "
                    "sign: %s",
                    ", ".join(adopted),
                )
        if settings.measure_dab_size and target.cached_detail_size is not None:
            # Likewise the one-cell brush: the dab probe cannot run on a sign
            # that must not be wiped, and starting at Size 1 where an earlier
            # job already proved it misses costs a whole touch-up pass to
            # rediscover.
            self._measured_detail_size = target.cached_detail_size
            self._measured_dab_sweep = target.cached_dab_sweep
            LOGGER.info(
                "Resuming with the one-cell brush an earlier job proved on this "
                "sign: Size %s, %s",
                format_brush_size(target.cached_detail_size),
                "swept across each texel"
                if target.cached_dab_sweep
                else "pressed at a point",
            )
        if not settings.apply_brush_size:
            return
        model = job.target.brush_size_model
        if model is None:
            LOGGER.warning(
                "No brush model is stored for this profile, so no Size value "
                "is typed on resume: the brush is whatever Rust has set now. "
                "A fresh job with automatic brush sizing on measures and "
                "stores one."
            )
            return
        LOGGER.info(
            "Resuming with the brush model stored %s", model.captured_at or "earlier"
        )
        self._validate_brush_reach(job.plan, job.target, settings, model)

    @staticmethod
    def _brush_footprint_checks(
        target: PaintingTarget,
        plan: PaintPlan,
        diameter: int,
        size: float,
        model: BrushSizeModel,
    ) -> list[tuple[str, float, float]]:
        """Painted and required extent per axis, in that axis's screen pixels.

        Without a horizontal measurement the vertical footprint stands in for
        both axes - the square-in-screen-pixels assumption sizing itself
        falls back to.
        """

        canvas = target.canvas
        painted_y = model.fraction_for_size(size) * canvas.height
        checks = [("rows", painted_y, canvas.height / plan.height * diameter)]
        if model.has_horizontal_model:
            checks.append(
                (
                    "columns",
                    model.fraction_x_for_size(size) * canvas.width,
                    canvas.width / plan.width * diameter,
                )
            )
        else:
            checks.append(("columns", painted_y, canvas.width / plan.width * diameter))
        return checks

    def _detail_brush_overshoot(self, job: _Job) -> float:
        """How many logical cells the one-cell brush really covers, at worst.

        1.0 is a brush that fits its cell; anything above spills onto the
        neighbours.  Without a measured model the brush is whatever the user
        set in Rust, and is assumed to fit.
        """

        model = job.target.brush_size_model
        if model is None or not job.settings.apply_brush_size:
            return 1.0
        size = self._brush_plan_size(
            job.target, job.plan, 1, job.settings.logical_pixel_spacing, model
        )
        measured = self._measured_detail_size
        if measured is not None and job.settings.measure_dab_size:
            size = max(size, measured)
        checks = self._brush_footprint_checks(job.target, job.plan, 1, size, model)
        return max(painted / nominal for _, painted, nominal in checks)

    # How many times a paused-out measurement is restarted before the job gives
    # up. Pausing during the opening strokes is easy to do by accident once;
    # doing it three times running is somebody who wants the job stopped.
    _CALIBRATION_ATTEMPTS = 3

    # A cleared sign has to differ from the probed one by more than capture
    # noise on a lit, textured surface, or the click missed the control.
    _CLEAR_CONTRAST = 24.0

    # Rust wipes the sign asynchronously and can take several frames to show
    # it, so the sign gets a generous beat to go blank before painting starts.
    _CLEAR_SETTLE_SECONDS = 0.5

    def _clear_canvas(self, job: _Job, *, quiet: bool = False) -> None:
        """Click Rust's clear control and give the sign time to go blank.

        The clear click is trusted.  A capture comparison used to stop the job
        here, but Rust's redraw sometimes lags past the capture and the check
        then failed signs that had genuinely cleared - and a job stopped by a
        false alarm costs far more than probe strokes under the artwork ever
        would.  The comparison survives only as a log line so a truly
        miscalibrated clear button can still be diagnosed afterwards.
        """

        button = job.target.clear_button
        if button is None:
            return
        epoch = self._pause_generation_value()
        self._update_progress_state(
            PainterState.RUNNING, "Clearing the sign", phase="calibrate"
        )
        canvas = ScreenRect(
            job.target.canvas.left,
            job.target.canvas.top,
            job.target.canvas.width,
            job.target.canvas.height,
        )
        park = self._park_point(job.target)
        before = self._capture_parked(canvas, park, epoch)
        self._safe_click(
            normalized_point(button, 0.5, 0.5),
            epoch,
            hold_floor=self._PICKER_CLICK_HOLD_SECONDS,
        )
        self._interruptible_sleep(
            max(
                self._settle(job.settings.delay_between_colors_seconds),
                self._CLEAR_SETTLE_SECONDS,
            ),
            epoch=epoch,
            check_focus=True,
        )
        after = self._capture_parked(canvas, park, epoch)
        if not self._canvas_changed(before, after):
            LOGGER.warning(
                "The sign looked unchanged after clicking Rust's clear control. "
                "If probe strokes show through the artwork, recalibrate the "
                "clear control over the button that wipes the sign."
            )
        else:
            if not quiet:
                LOGGER.info("Cleared the sign after measuring the brush")
            self._remember_bare_color(after)
        job.bare_canvas = after

    def _remember_bare_color(self, capture: Any) -> None:
        """Keep the median colour of a capture of the cleared sign."""

        import numpy as np

        try:
            pixels = np.asarray(capture.convert("RGB"), dtype=np.float32).reshape(-1, 3)
            median = np.median(pixels, axis=0)
        except Exception:
            LOGGER.debug("The bare sign's colour could not be read", exc_info=True)
            return
        color = tuple(int(round(float(channel))) for channel in median)
        with self._condition:
            self._measured_bare_color = color  # type: ignore[assignment]
        LOGGER.info("The bare sign reads #%02X%02X%02X", *color)

    def _canvas_changed(self, before: Any, after: Any) -> bool:
        """Whether two captures of the sign differ by more than capture noise."""

        import numpy as np

        first = np.asarray(before.convert("RGB"), dtype=np.float32)
        second = np.asarray(after.convert("RGB"), dtype=np.float32)
        if first.shape != second.shape:
            return True
        return bool(
            float(np.linalg.norm(second - first, axis=2).max()) >= self._CLEAR_CONTRAST
        )

    def _measure_brush_size_model(self, job: _Job) -> BrushSizeModel:
        """Paint probe strokes and fit Size number to painted canvas fraction.

        Each probe types a number, drags one stroke through the middle of the
        sign, and measures the band it left behind.  Reading the sign itself is
        what makes the result independent of Rust's preview tile and of how
        close the camera happens to be standing.
        """

        import numpy as np

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
        park = self._park_point(target)
        # Drawn through the vertical middle and well short of both sides.  The
        # drag is deliberately short: the band extends past each end by half
        # the brush's horizontal footprint, and a size-100 probe on a close-up
        # sign is wider than the margin a longer drag would leave.  A clipped
        # band loses its horizontal sample, and losing the widest probe is
        # what left the horizontal fit leaning on a handful of narrow ones -
        # measured live as the column count swinging six percent between runs.
        stroke_y = int(round(canvas.top + canvas.height / 2.0))
        start = (int(round(canvas.left + canvas.width * 0.32)), stroke_y)
        end = (int(round(canvas.left + canvas.width * 0.68)), stroke_y)
        drag_pixels = float(end[0] - start[0])

        samples: list[tuple[float, float]] = []
        samples_x: list[tuple[float, float]] = []
        # Rendering bias per probe: where the solid band's center landed minus
        # where the stroke was commanded, in capture pixels.  A live probe
        # showed Rust stamping about a texel left and a fraction of one down of
        # the cursor; measuring it here lets painting cancel it out.
        biases_x: list[float] = []
        biases_y: list[float] = []
        drag_center_x = (start[0] + end[0]) / 2.0 - canvas.left
        stroke_y_local = float(stroke_y - canvas.top)
        clipped: list[float] = []
        last_color: RGBColor | None = None

        def probe(size: float, label: str) -> "StrokeBand | None":
            nonlocal last_color
            epoch = self._pause_generation_value()
            self._update_progress_state(
                PainterState.RUNNING,
                f"Measuring brush size {format_brush_size(size)} ({label})",
                phase="calibrate",
            )
            self._write_brush_size(box, size, settings, epoch)
            before = self._capture_parked(canvas, park, epoch)
            # Chosen against what the band will cover, so a probe over an
            # earlier probe of the same colour cannot read as no change.
            along = [
                (start[0] + (end[0] - start[0]) * t - canvas.left, stroke_y_local)
                for t in (0.0, 0.25, 0.5, 0.75, 1.0)
            ]
            color = self._probe_color_against(before, along, skip=last_color)
            last_color = color
            self._select_color(color, target, settings, epoch, apply_correction=False)
            self._screen_stroke(start, end, settings, epoch)
            after = self._capture_parked(canvas, park, epoch)
            try:
                band = measure_stroke_band(before, after)
            except ValueError as exc:
                LOGGER.info(
                    "Brush probe %s could not be measured: %s",
                    format_brush_size(size),
                    exc,
                )
                return None
            LOGGER.info(
                "Brush probe %s covered %.1f px of %d, touched %.1f px (%s)",
                format_brush_size(size),
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
            biases_y.append(band.center_y - stroke_y_local)
            # The band's ends stick out half the brush's *horizontal* footprint
            # past each drag endpoint, so the same capture also measures the
            # axis the vertical band cannot: subtract the drag and what is
            # left is how wide this Size number really paints on this sign.
            if not band.x_clipped and band.width > drag_pixels:
                samples_x.append((size, (band.width - drag_pixels) / canvas.width))
                biases_x.append(band.center_x - drag_center_x)

        if len(samples) < 2:
            detail = (
                "Sizes "
                + ", ".join(format_brush_size(size) for size in clipped)
                + " covered the whole sign, so stand closer or calibrate a larger sign."
                if clipped
                else "Confirm the paint tool is selected and the sign fills the "
                "calibrated canvas."
            )
            raise RuntimeError(
                "Brush measurement needs two usable probe strokes but got "
                f"{len(samples)}. {detail}"
            )
        # The median shrugs off one probe whose band was misread; positive
        # means the paint landed right/down of the command.
        bias = (
            float(np.median(biases_x)) / canvas.width if biases_x else 0.0,
            float(np.median(biases_y)) / canvas.height if biases_y else 0.0,
        )
        model = fit_brush_size_model(samples, samples_x=samples_x, bias=bias)
        LOGGER.info(
            "Brush size model: %.6f of the sign per unit (~%.0f sign rows), "
            "offset %.6f%s; rendering bias %+.1fpx, %+.1fpx",
            model.slope,
            model.sign_pixel_rows,
            model.intercept,
            (
                f"; horizontal {model.slope_x:.6f} per unit "
                f"(~{model.sign_pixel_columns:.0f} sign columns)"
                if model.has_horizontal_model
                else "; horizontal footprint not measurable"
            ),
            model.bias_x * canvas.width,
            model.bias_y * canvas.height,
        )
        return model

    @staticmethod
    def _grid_cell_centers(
        job: _Job, plan: PaintPlan, canvas: ScreenRect
    ) -> tuple[Any, Any] | None:
        """Where the plan's cells sit in a capture of ``canvas``, per the grid.

        ``None`` when no grid was measured, which leaves the sampler on the
        rectangle's own even spacing.
        """

        import numpy as np

        grid = job.texel_grid
        if grid is None or not grid.agrees_with(canvas):
            return None
        rect = grid.registered_rect()
        columns = np.arange(plan.width, dtype=np.float64)
        rows = np.arange(plan.height, dtype=np.float64)
        centers_x = rect.left - canvas.left + (columns + 0.5) * rect.width / plan.width
        centers_y = rect.top - canvas.top + (rows + 0.5) * rect.height / plan.height
        return centers_x, centers_y

    def _measure_texel_grid_safely(self, job: _Job) -> TexelGridModel | None:
        """Probe the texel grid, and paint on the old inference if it fails.

        A sign whose stamps do not snap, a capture the probe cannot read, or a
        result that does not sit on the calibrated rectangle all end here as
        ``None`` with the reason logged; only a pause or an abort is allowed
        through, because those belong to the job, not the measurement.
        """

        settings = job.settings
        if (
            not settings.measure_texel_grid
            or not settings.apply_brush_size
            or not self.input.emits_real_input
            or job.target.brush_size_box is None
        ):
            return None
        try:
            return self._measure_texel_grid(job)
        except (_RetryAction, _AbortRequested):
            raise
        except Exception as exc:  # the fallback is the previous behaviour
            LOGGER.warning(
                "The texel grid could not be measured (%s); strokes are laid "
                "out from the brush measurement instead",
                exc,
            )
            return None

    # Margin captured around the calibrated rectangle so the sign quad's own
    # edges are in shot, as a fraction of the longer side and a floor in pixels.
    _EDGE_MARGIN_FRACTION = 0.02
    _EDGE_MARGIN_MIN_PIXELS = 8

    # The sweep's own press timing, before this sign's timing has been probed:
    # a fraction of the frame floor, well above the shortest press the game
    # was measured to take (100 of 100 at 6 ms, gap 0).
    _SWEEP_HOLD_SECONDS = 0.02
    _SWEEP_GAP_SECONDS = 0.005
    # The probe colour for the sweeps: saturated, far from any bare sign.
    _SWEEP_COLOR: RGBColor = (220, 40, 40)
    # A measured map is proved before it is trusted: this lattice of dabs
    # through it must all land on their texels.
    _SWEEP_CHECK_LATTICE = (24, 12)

    def _sweep_cursor_map_safely(self, job: _Job) -> TexelGridModel | None:
        """Measure the cursor map from the export, or None to fall back.

        Needs the download button and the Size field; a sweep that cannot
        be read (no export, lanes that collided, a moved sign) is logged and
        leaves the staircase probe to try.  A pause or abort passes through.
        """

        settings = job.settings
        target = job.target
        if (
            not settings.measure_texel_grid
            or not settings.apply_brush_size
            or not self.input.emits_real_input
            or target.brush_size_box is None
            or target.download_button is None
            or target.clear_button is None
        ):
            return None
        try:
            return self._sweep_cursor_map(job)
        except (_RetryAction, _AbortRequested):
            raise
        except Exception as exc:
            LOGGER.warning(
                "The cursor map could not be swept from the export (%s); "
                "falling back to the staircase probe",
                exc,
            )
            return None

    def _sweep_cursor_map(self, job: _Job) -> TexelGridModel:
        """Press at every pixel across the sign and read the texels it stamped.

        One press per screen pixel along x on a few rows, an export, a wipe,
        then the same along y; :func:`app.cursor_map.attribute_sweep` turns
        each export into a table of which texel every pixel stamps.  The map
        is then proved with a lattice of dabs before anything trusts it.
        """

        import numpy as np

        target = job.target
        settings = job.settings
        canvas = ScreenRect(
            target.canvas.left,
            target.canvas.top,
            target.canvas.width,
            target.canvas.height,
        )
        box = target.brush_size_box
        assert box is not None
        epoch = self._pause_generation_value()
        self._update_progress_state(
            PainterState.RUNNING,
            "Measuring where the cursor stamps every texel (export sweep)",
            phase="calibrate",
        )
        self._write_brush_size(box, BRUSH_SIZE_MIN, settings, epoch)
        margin = SWEEP_MARGIN_PIXELS
        # Never slower than the job's own floors (a simulated sign runs them
        # at zero), never faster than the sweep's proven timing.
        hold = min(self._SWEEP_HOLD_SECONDS, self._MIN_PRESS_SECONDS)
        gap = min(self._SWEEP_GAP_SECONDS, self._STROKE_GAP_FLOOR_SECONDS)

        def press(x: int, y: int, batch_epoch: int) -> None:
            self._screen_stroke((x, y), (x, y), settings, batch_epoch, hold_seconds=hold)
            self._interruptible_sleep(gap, epoch=batch_epoch, check_focus=True)

        def sweep(along_x: bool, lanes: int):
            self._clear_canvas(job, quiet=True)
            batch_epoch = self._pause_generation_value()
            self._select_color(self._SWEEP_COLOR, target, settings, batch_epoch, apply_correction=False)
            # A few lanes more than the sweep uses each carry one anchor
            # press at the middle of the rectangle, which pins the lattice
            # the lanes only fix up to a stride (see attribute_sweep).  Three
            # of them, because the game now and then drops a press.
            anchors = ANCHOR_LANES
            if along_x:
                positions = sweep_positions(canvas.left, canvas.width, margin)
                offsets = lane_offsets(canvas.height, lanes + anchors)
                across = [canvas.top + offset for offset in offsets[:lanes]]
                anchor_lanes = [canvas.top + offset for offset in offsets[lanes:]]
                anchor = canvas.left + canvas.width // 2
                interior = (canvas.left + 40, canvas.left + canvas.width - 41)
            else:
                positions = sweep_positions(canvas.top, canvas.height, margin)
                offsets = lane_offsets(canvas.width, lanes + anchors)
                across = [canvas.left + offset for offset in offsets[:lanes]]
                anchor_lanes = [canvas.left + offset for offset in offsets[lanes:]]
                anchor = canvas.top + canvas.height // 2
                interior = (canvas.top + 40, canvas.top + canvas.height - 41)
            self._update_progress_state(
                PainterState.RUNNING,
                f"Sweeping the cursor along {'x' if along_x else 'y'}: "
                f"{len(positions)} presses in {lanes} lanes",
                phase="calibrate",
            )
            for i, position in enumerate(positions):
                lane = across[i % lanes]
                if along_x:
                    press(position, lane, batch_epoch)
                else:
                    press(lane, position, batch_epoch)
            for anchor_lane in anchor_lanes:
                if along_x:
                    press(anchor, anchor_lane, batch_epoch)
                else:
                    press(anchor_lane, anchor, batch_epoch)
            export = self._export_texture(
                job, batch_epoch, why=f"the {'x' if along_x else 'y'} cursor sweep"
            )
            if export is None:
                raise SweepError("no export could be read after the sweep")
            alpha = np.where(export.painted, 255, 0)
            table = attribute_sweep(
                alpha, positions, lanes, along_x=along_x, interior=interior, anchor=anchor
            )
            # A press the game dropped leaves its pixel unread.  Press those
            # pixels again - on the first lane, whose row is known - and
            # read the export once more; a pixel still unread after that is
            # one the sign really does not answer to.
            for _pass in range(2):
                gaps = unread_positions(table)
                if not gaps:
                    break
                LOGGER.info(
                    "%s sweep: %d pixel(s) went unread; pressing them again",
                    "x" if along_x else "y",
                    len(gaps),
                )
                batch_epoch = self._pause_generation_value()
                for position in gaps:
                    if along_x:
                        press(position, across[0], batch_epoch)
                    else:
                        press(across[0], position, batch_epoch)
                export = self._export_texture(job, batch_epoch, why="the sweep's fill-in")
                if export is None:
                    break
                table = fill_in_sweep(table, np.where(export.painted, 255, 0), lane_line(alpha, along_x, 0), along_x=along_x)
            return table

        tables = []
        for along_x in (True, False):
            lanes = DEFAULT_LANES
            while True:
                try:
                    tables.append(sweep(along_x, lanes))
                    break
                except SweepError as exc:
                    if "share a texel" in str(exc) and lanes * 2 <= MAX_LANES:
                        LOGGER.info("%s; sweeping again with %d lanes", exc, lanes * 2)
                        lanes *= 2
                        continue
                    raise
        x_table, y_table = tables
        for name, table in (("column", x_table), ("row", y_table)):
            if table.unreachable:
                # A structural limit of the screen geometry: some texels
                # have no whole pixel that stamps them.  Refusing beats
                # painting a sign with holes nothing can fill.
                raise RuntimeError(
                    f"{len(table.unreachable)} texel {name}(s) of this sign cannot be "
                    f"reached by the cursor at this screen size ({table.pitch:.2f} px per "
                    f"texel): {list(table.unreachable[:10])}. Make the sign larger on "
                    "screen (stand closer, raise the resolution or the UI scale) and "
                    "calibrate again."
                )
        grid = grid_from_tables(x_table, y_table)
        LOGGER.info(
            "Cursor map swept from the export: %dx%d texels; columns answer to x %d-%d "
            "(%.4f px each from %.2f), rows to y %d-%d (%.4f px each from %.2f)",
            grid.columns,
            grid.rows,
            grid.aim_columns[0][0],
            grid.aim_columns[-1][1],
            grid.pitch_x,
            grid.origin_x,
            grid.aim_rows[0][0],
            grid.aim_rows[-1][1],
            grid.pitch_y,
            grid.origin_y,
        )
        # Prove it: a lattice of dabs aimed through the map, read back exactly.
        self._clear_canvas(job, quiet=True)
        batch_epoch = self._pause_generation_value()
        self._select_color(self._SWEEP_COLOR, target, settings, batch_epoch, apply_correction=False)
        targets = lattice_targets(grid.columns, grid.rows, *self._SWEEP_CHECK_LATTICE)
        self._update_progress_state(
            PainterState.RUNNING,
            f"Proving the cursor map with {len(targets)} dabs",
            phase="calibrate",
        )
        for u, v in targets:
            press(*grid.aim_pixel(u, v), batch_epoch)
        export = self._export_texture(job, batch_epoch, why="the cursor map check")
        if export is None:
            raise SweepError("no export could be read after the cursor map check")
        exact, wrong = check_lattice(np.where(export.painted, 255, 0), targets)
        if wrong:
            # The game drops the odd press; a miss is a miss only if the
            # same aim misses twice.
            LOGGER.info(
                "Cursor map check: %d of %d dabs missed on the first try; pressing them again",
                len(wrong),
                len(targets),
            )
            batch_epoch = self._pause_generation_value()
            for u, v in wrong:
                press(*grid.aim_pixel(u, v), batch_epoch)
            export = self._export_texture(job, batch_epoch, why="the cursor map re-check")
            if export is None:
                raise SweepError("no export could be read after the cursor map re-check")
            exact, wrong = check_lattice(np.where(export.painted, 255, 0), targets)
        if wrong and _only_isolated_corner_misses(wrong, grid.columns, grid.rows):
            LOGGER.warning(
                "Cursor map proved everywhere except corner test dabs %s; accepting the "
                "export-measured map and leaving those border cells to verification",
                wrong,
            )
        elif wrong:
            raise SweepError(
                f"the cursor map check landed {exact} of {len(targets)} dabs; "
                f"missed {wrong[:8]}"
            )
        LOGGER.info(
            "Cursor map proved: %d of %d lattice dabs landed on their texel",
            exact,
            len(targets),
        )
        return grid

    def _adopt_native_plan(self, job: _Job) -> None:
        """Execute the job's plan as one Size-1 stroke per texel row.

        A fresh job only: a resumed job's place is counted in its own
        plan's strokes, so its plan is left as it was.
        """

        grid = job.texel_grid
        if grid is None or not grid.swept:
            return
        if is_native(job.plan, grid.columns, grid.rows):
            return
        before = job.plan
        job.plan = nativize_plan(before, grid.columns, grid.rows)
        origins = stroke_index_map(before, job.plan)
        job.stroke_origins = tuple(origins)
        job.original_total = before.stroke_count
        if job.start_stroke > 0:
            # A resumed job's place is counted in the plan's own strokes.
            job.start_stroke = next(
                (i for i, origin in enumerate(origins) if origin >= job.start_stroke),
                len(origins),
            )
        with self._condition:
            self._native_plan = job.plan
        LOGGER.info(
            "Plan of %dx%d cells in %d strokes laid out on the sign's %dx%d texels "
            "as %d single-row strokes with the one-texel brush",
            before.width,
            before.height,
            before.stroke_count,
            grid.columns,
            grid.rows,
            job.plan.stroke_count,
        )

    @property
    def executed_plan(self) -> PaintPlan | None:
        """The plan the job really paints, when it differs from the one given."""

        with self._condition:
            return self._native_plan

    # Press timing candidates, fastest first, as (hold, gap) in seconds.  The
    # probe adopts the pair one step slower than the fastest clean one.
    _EXPORT_TIMING_CANDIDATES = ((0.006, 0.0), (0.012, 0.005), (0.024, 0.01), (0.04, 0.02))
    _EXPORT_TIMING_DOTS = 100

    def _probe_press_timing_by_export(self, job: _Job) -> None:
        """Prove the press hold and stroke gap on this sign, texel-exactly.

        Batches of dabs through the swept map at each candidate timing, one
        export per batch: a dab that left its texel bare was a press the
        game never saw.  The hold and gap adopted are one candidate slower
        than the fastest clean pair, so the timing in use always has a
        proven step of margin.  Never slower than the floors.
        """

        import numpy as np

        settings = job.settings
        grid = job.texel_grid
        if not settings.measure_press_hold or grid is None or not grid.swept:
            return
        target = job.target
        self._measured_press_hold_seconds = None
        self._measured_stroke_gap_seconds = None
        dots = self._EXPORT_TIMING_DOTS
        if grid.columns < dots // 2 or grid.rows < 16:
            return
        self._update_progress_state(
            PainterState.RUNNING,
            "Proving the press timing this sign takes (export)",
            phase="calibrate",
        )
        clean: list[tuple[float, float]] = []
        try:
            for batch, (hold, gap) in enumerate(self._EXPORT_TIMING_CANDIDATES):
                epoch = self._pause_generation_value()
                self._clear_canvas(job, quiet=True)
                epoch = self._pause_generation_value()
                self._select_color(self._SWEEP_COLOR, target, settings, epoch, apply_correction=False)
                texels = [
                    (
                        4 + (k * (grid.columns - 8)) // dots,
                        4 + ((k * 37 + batch * 17) % max(1, grid.rows - 8)),
                    )
                    for k in range(dots)
                ]
                for u, v in texels:
                    point = grid.aim_pixel(u, v)
                    self._screen_stroke(point, point, settings, epoch, hold_seconds=hold)
                    self._interruptible_sleep(gap, epoch=epoch, check_focus=True)
                export = self._export_texture(job, epoch, why=f"press timing batch {batch + 1}")
                if export is None:
                    LOGGER.info("Press timing probe: no export; keeping the floors")
                    return
                landed = sum(1 for u, v in texels if export.painted[v, u])
                LOGGER.info(
                    "Press timing probe: %d of %d dabs landed at a %d ms hold, %d ms gap",
                    landed,
                    dots,
                    int(round(hold * 1000)),
                    int(round(gap * 1000)),
                )
                if landed == dots:
                    clean.append((hold, gap))
                    break
        except (_RetryAction, _AbortRequested):
            raise
        except Exception as exc:
            LOGGER.warning("The press timing probe could not run (%s); keeping the floors", exc)
            return
        if not clean:
            LOGGER.info("Press timing probe: no candidate landed every dab; keeping the floors")
            return
        fastest = clean[-1]
        index = self._EXPORT_TIMING_CANDIDATES.index(fastest)
        adopted = self._EXPORT_TIMING_CANDIDATES[min(index + 1, len(self._EXPORT_TIMING_CANDIDATES) - 1)]
        self._measured_press_hold_seconds = float(adopted[0])
        self._measured_stroke_gap_seconds = float(adopted[1])
        LOGGER.info(
            "Adopting a %d ms press hold and %d ms stroke gap for this sign "
            "(%d ms / %d ms proved clean below them)",
            int(round(adopted[0] * 1000)),
            int(round(adopted[1] * 1000)),
            int(round(fastest[0] * 1000)),
            int(round(fastest[1] * 1000)),
        )

    def _measure_texel_grid(self, job: _Job) -> TexelGridModel:
        """Stamp the smallest brush around the sign and read its texel grid.

        The probe's logic lives in :mod:`app.texel_grid`; this method lends it
        the mouse and the captures.  A scout stamp, a staircase and a ladder
        per axis is under fifty dabs, each a single press, and every one of
        them is wiped with the brush probes before the artwork starts.
        """

        import numpy as np

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
        park = self._park_point(target)
        epoch = self._pause_generation_value()
        self._update_progress_state(
            PainterState.RUNNING, "Measuring the sign's texel grid", phase="calibrate"
        )
        self._write_brush_size(box, BRUSH_SIZE_MIN, settings, epoch)

        # The quad's edges, from a capture a little wider than the rectangle.
        margin = max(
            self._EDGE_MARGIN_MIN_PIXELS,
            int(round(self._EDGE_MARGIN_FRACTION * max(canvas.width, canvas.height))),
        )
        edges = None
        try:
            wide = ScreenRect(
                canvas.left - margin,
                canvas.top - margin,
                canvas.width + 2 * margin,
                canvas.height + 2 * margin,
            )
            self._move(park, epoch)
            if self.input.emits_real_input:
                self._interruptible_sleep(
                    self._CAPTURE_SETTLE_SECONDS, epoch=epoch, check_focus=True
                )
            found = find_quad_edges(
                np.asarray(self._screen_capture(wide).convert("RGB"), dtype=np.float32),
                (margin, margin, margin + canvas.width, margin + canvas.height),
                margin - 2,
            )
            edges = (
                None if found[0] is None else found[0] + wide.left,
                None if found[1] is None else found[1] + wide.top,
                None if found[2] is None else found[2] + wide.left,
                None if found[3] is None else found[3] + wide.top,
            )
            LOGGER.info(
                "Sign quad edges: left %s, top %s, right %s, bottom %s (rectangle "
                "%d, %d, %d, %d)",
                *("?" if e is None else f"{e:.1f}" for e in edges),
                canvas.left,
                canvas.top,
                canvas.left + canvas.width,
                canvas.top + canvas.height,
            )
        except (_RetryAction, _AbortRequested):
            raise
        except Exception as exc:
            LOGGER.info("The sign's edges could not be captured (%s)", exc)

        last_batch_color: RGBColor | None = None

        def stamp_batch(plan: GridProbePlan) -> np.ndarray:
            nonlocal last_batch_color
            batch_epoch = self._pause_generation_value()
            before = self._capture_parked(canvas, park, batch_epoch)
            color = self._probe_color_against(
                before,
                [(x - canvas.left, y - canvas.top) for x, y in plan.points],
                skip=last_batch_color,
            )
            last_batch_color = color
            self._select_color(color, target, settings, batch_epoch, apply_correction=False)
            for x, y in plan.points:
                # Round half-up, the same convention painting uses, so the
                # cursor map is measured through the same rounding it will be
                # used through - flooring here while painting rounds costs a
                # systematic half pixel, a quarter texel on a fine sign.
                point = (math.floor(x + 0.5), math.floor(y + 0.5))
                point = (
                    min(max(point[0], canvas.left), canvas.left + canvas.width - 1),
                    min(max(point[1], canvas.top), canvas.top + canvas.height - 1),
                )
                self._screen_stroke(point, point, settings, batch_epoch)
            after = self._capture_parked(canvas, park, batch_epoch)
            return stamp_diff(before, after)

        model = target.brush_size_model
        stamp_hint = (
            model.fraction_for_size(BRUSH_SIZE_MIN) * canvas.height
            if model is not None
            else 6.0
        )
        grid = measure_grid(
            canvas,
            stamp_batch,
            pitch_hint=stamp_hint,
            stamp_hint=stamp_hint,
            edges=edges,
            texture_sizes=SIGN_TEXTURE_SIZES,
        )
        if not grid.agrees_with(canvas):
            raise ValueError(
                f"the measured grid ({grid.columns}x{grid.rows} texels at "
                f"{grid.origin_x:.0f}, {grid.origin_y:.0f}) does not sit on the "
                "calibrated rectangle"
            )
        LOGGER.info(
            "Texel grid: %dx%d texels, %.4f x %.4f px each, origin %.2f, %.2f; "
            "cursor lattice %.4f x %.4f px from %.2f, %.2f; worst rung %.2f texel (%s)",
            grid.columns,
            grid.rows,
            grid.pitch_x,
            grid.pitch_y,
            grid.origin_x,
            grid.origin_y,
            grid.aim_pitch_x,
            grid.aim_pitch_y,
            grid.aim_origin_x,
            grid.aim_origin_y,
            grid.residual,
            "counted from the sign's edges" if grid.from_edges else "counted from the rectangle",
        )
        if min(grid.pitch_x, grid.pitch_y) < AIM_AUDIT_MAX_PITCH:
            # On a fine sign the smooth cursor map is not the whole truth: a
            # DPI-scaled display quantizes the game's cursor reads, and phase
            # bands of columns land their dabs a texel over.  One dot per
            # column and per row finds them; whole-pixel nudges fix them.
            self._update_progress_state(
                PainterState.RUNNING,
                "Auditing the cursor aim, one dot per row and column",
                phase="calibrate",
            )
            grid = audit_cursor_map(grid, stamp_batch, canvas)
        return grid

    # A capture-diff value below this is noise; at or above it the texel
    # changed.  The same floor the grid probe reads its stamps with.
    _LINE_PROBE_DIFF_FLOOR = 24.0

    # Probe knobs as class attributes so a test driving a simulated sign can
    # scale them to its size and clock.
    _PRESS_HOLD_PROBE_CANDIDATES = PRESS_HOLD_PROBE_CANDIDATES
    _PRESS_HOLD_PROBE_DOTS = PRESS_HOLD_PROBE_DOTS
    _PRESS_HOLD_PROBE_MIN_STROKES = PRESS_HOLD_PROBE_MIN_STROKES

    _STROKE_GAP_PROBE_CANDIDATES = STROKE_GAP_PROBE_CANDIDATES

    def _dot_probe_setup(self, job: _Job) -> "tuple[TexelGridModel, ScreenRect, tuple[int, int], int] | None":
        """What the dot probes share, or None when this job should not stamp any."""

        settings = job.settings
        grid = job.texel_grid
        if not settings.measure_press_hold or grid is None or job.mode != "paint":
            return None
        total = sum(len(group.strokes) for group in job.plan.color_groups)
        if total < self._PRESS_HOLD_PROBE_MIN_STROKES:
            LOGGER.info(
                "The plan's %d strokes are too few for the timing probes to pay "
                "for their captures; keeping the frame floors",
                total,
            )
            return None
        dots = max(4, int(self._PRESS_HOLD_PROBE_DOTS))
        if grid.columns < dots or grid.rows < 16:
            return None
        target = job.target
        canvas = ScreenRect(
            target.canvas.left,
            target.canvas.top,
            target.canvas.width,
            target.canvas.height,
        )
        park = self._park_point(target)
        return grid, canvas, park, dots

    def _stamp_dot_batch(
        self,
        job: _Job,
        grid: TexelGridModel,
        canvas: ScreenRect,
        park: tuple[int, int],
        before: Any,
        batch: int,
        dots: int,
        *,
        hold_seconds: float | None,
        gap_seconds: float | None,
        last_color: RGBColor | None,
    ) -> "tuple[Any, int, RGBColor]":
        """Stamp one batch of probe dots and count the ones that never landed.

        Dots spread across the columns on rows staggered per batch, so no
        batch stamps where an earlier one did; each is a stationary press
        held ``hold_seconds`` (the job's hold when None) with
        ``gap_seconds`` before the next (the job's gap when None).  Returns
        the capture after the batch, the number of dots missing from it, and
        the colour used.
        """

        settings = job.settings
        target = job.target
        epoch = self._pause_generation_value()
        span = grid.columns - 8
        texels = [
            (
                4 + round(index * (span - 1) / (dots - 1)),
                4 + ((index * 37 + batch * 17) % max(1, grid.rows - 8)),
            )
            for index in range(dots)
        ]
        expected = [
            (
                grid.origin_x + (u + 0.5) * grid.pitch_x - canvas.left,
                grid.origin_y + (v + 0.5) * grid.pitch_y - canvas.top,
            )
            for u, v in texels
        ]
        color = self._probe_color_against(before, expected, skip=last_color)
        # A dot stamped in the colour already under it cannot be told from
        # a dot that never landed: an earlier probe's mark, or the artwork's
        # own colour on a sign painted before.  Those are left out of the
        # count rather than read as drops.
        import numpy as np

        pixels = np.asarray(before.convert("RGB"), dtype=np.float32)
        judgeable = []
        for x, y in expected:
            column = min(max(int(round(x)), 0), pixels.shape[1] - 1)
            row = min(max(int(round(y)), 0), pixels.shape[0] - 1)
            under = pixels[row, column]
            judgeable.append(
                float(np.linalg.norm(under - np.array(color, dtype=np.float32)))
                >= 2.0 * self._LINE_PROBE_DIFF_FLOOR
            )
        self._select_color(color, target, settings, epoch, apply_correction=False)
        for u, v in texels:
            x, y = grid.cursor_point(u + 0.5, v + 0.5)
            point = (math.floor(x + 0.5), math.floor(y + 0.5))
            point = (
                min(max(point[0], canvas.left), canvas.left + canvas.width - 1),
                min(max(point[1], canvas.top), canvas.top + canvas.height - 1),
            )
            self._screen_stroke(point, point, settings, epoch, hold_seconds=hold_seconds)
            self._interruptible_sleep(
                gap_seconds
                if gap_seconds is not None
                else self._stroke_gap(settings.delay_between_strokes_seconds),
                epoch=epoch,
                check_focus=True,
            )
        after = self._capture_parked(canvas, park, epoch)
        window = max(4.0, 2.0 * max(grid.pitch_x, grid.pitch_y))
        landed = locate_stamps(stamp_diff(before, after), expected, window)
        missed = sum(
            1 for centre, judge in zip(landed, judgeable) if judge and centre is None
        )
        return after, missed, color

    def _probe_press_hold(self, job: _Job) -> None:
        """Measure the shortest press hold that lands every dab on this sign.

        One batch of dots per candidate hold, longest hold first, each batch
        captured and counted; the first batch that drops a dot ends the
        descent.  The hold adopted for this job's stationary presses is the
        shortest clean one whose next-shorter neighbour also landed
        everything - one step of demonstrated margin - and never longer than
        the configured hold.  A sign that gives no two clean steps keeps the
        frame floor, exactly as before; the per-color checks and the
        touch-up pass stay underneath either way.  The dots are wiped with
        the other calibration marks before the artwork starts.
        """

        setup = self._dot_probe_setup(job)
        if setup is None:
            return
        grid, canvas, park, dots = setup
        settings = job.settings
        self._update_progress_state(
            PainterState.RUNNING,
            "Measuring the press hold this sign needs",
            phase="calibrate",
        )
        self._measured_press_hold_seconds = None
        clean: list[float] = []
        last_color: RGBColor | None = None
        try:
            before = self._capture_parked(canvas, park, self._pause_generation_value())
            for batch, hold in enumerate(self._PRESS_HOLD_PROBE_CANDIDATES):
                before, missed, last_color = self._stamp_dot_batch(
                    job, grid, canvas, park, before, batch, dots,
                    hold_seconds=hold, gap_seconds=None, last_color=last_color,
                )
                LOGGER.info(
                    "Press-hold probe: %d of %d dots landed at %d ms",
                    dots - missed,
                    dots,
                    int(round(hold * 1000)),
                )
                if missed:
                    break
                clean.append(hold)
        except (_RetryAction, _AbortRequested):
            raise
        except Exception as exc:
            LOGGER.warning(
                "The press-hold probe could not run (%s); keeping the frame floor",
                exc,
            )
            return
        floor_ms = int(round(self._drag_dwell_seconds(settings) * 1000))
        if len(clean) < 2:
            LOGGER.info(
                "Press-hold probe kept the %d ms hold: %s",
                floor_ms,
                "no candidate landed cleanly"
                if not clean
                else "only one candidate landed cleanly, which is no margin",
            )
            return
        adopted = clean[-2]
        self._measured_press_hold_seconds = float(adopted)
        LOGGER.info(
            "Adopting a %d ms press hold for this sign's dabs and line presses "
            "(%d ms proved clean below it; drags keep their %d ms dwell)",
            int(round(adopted * 1000)),
            int(round(clean[-1] * 1000)),
            floor_ms,
        )

    def _probe_stroke_gap(self, job: _Job) -> None:
        """Measure the shortest gap between strokes the game keeps apart.

        Same batches as the hold probe, at the hold this sign proved, with
        the gap between dots shrinking batch by batch; a dot that never
        lands is a press the game merged into the one before it.  The gap
        adopted is the shortest clean one with a clean step below it, never
        longer than the floor; anything less keeps the floor.
        """

        setup = self._dot_probe_setup(job)
        if setup is None:
            return
        grid, canvas, park, dots = setup
        settings = job.settings
        self._update_progress_state(
            PainterState.RUNNING,
            "Measuring the gap between strokes this sign needs",
            phase="calibrate",
        )
        self._measured_stroke_gap_seconds = None
        floor = self._stroke_gap(settings.delay_between_strokes_seconds)
        candidates = [gap for gap in self._STROKE_GAP_PROBE_CANDIDATES if gap < floor]
        if len(candidates) < 2:
            return
        clean: list[float] = []
        last_color: RGBColor | None = None
        try:
            before = self._capture_parked(canvas, park, self._pause_generation_value())
            for batch, gap in enumerate(candidates):
                before, missed, last_color = self._stamp_dot_batch(
                    job, grid, canvas, park, before, 8 + batch, dots,
                    hold_seconds=None, gap_seconds=gap, last_color=last_color,
                )
                LOGGER.info(
                    "Stroke-gap probe: %d of %d dots landed with %d ms between them",
                    dots - missed,
                    dots,
                    int(round(gap * 1000)),
                )
                if missed:
                    break
                clean.append(gap)
        except (_RetryAction, _AbortRequested):
            raise
        except Exception as exc:
            LOGGER.warning(
                "The stroke-gap probe could not run (%s); keeping the %d ms gap",
                exc,
                int(round(floor * 1000)),
            )
            return
        if len(clean) < 2:
            LOGGER.info(
                "Stroke-gap probe kept the %d ms gap: %s",
                int(round(floor * 1000)),
                "no candidate landed cleanly"
                if not clean
                else "only one candidate landed cleanly, which is no margin",
            )
            return
        adopted = clean[-2]
        self._measured_stroke_gap_seconds = float(adopted)
        LOGGER.info(
            "Adopting a %d ms gap between this sign's strokes (%d ms proved clean below it)",
            int(round(adopted * 1000)),
            int(round(clean[-1] * 1000)),
        )

    _DRAG_RATE_PROBE_TEXELS_PER_SECOND = DRAG_RATE_PROBE_TEXELS_PER_SECOND
    _DRAG_RATE_PROBE_MIN_RUN_TEXELS = DRAG_RATE_PROBE_MIN_RUN_TEXELS

    def _probe_drag_rate(self, job: _Job) -> None:
        """Measure how fast a long drag may run and still paint every texel.

        One drag across half a row per candidate rate, slowest first, each
        captured and read for coverage along the run; the first rate that
        leaves holes ends the climb.  The rate adopted is the second-fastest
        clean one - the fastest always has a proven step above it - and
        never below the floor the painter would use anyway.
        """

        settings = job.settings
        grid = job.texel_grid
        if not settings.measure_press_hold or grid is None or job.mode != "paint":
            return
        if not math.isfinite(self._LONG_DRAG_MAX_TEXELS_PER_SECOND):
            return  # drags are not capped at all: nothing to raise
        longest = 0
        for group in job.plan.color_groups:
            for stroke in group.strokes:
                if stroke.start_y == stroke.end_y:
                    longest = max(longest, stroke.pixel_count)
        if longest < self._DRAG_RATE_PROBE_MIN_RUN_TEXELS:
            return
        if grid.columns < 64 or grid.rows < 32:
            return
        aiming = self._aiming(job, job.plan)
        if aiming.mapper is None or not aiming.native:
            return
        target = job.target
        canvas = ScreenRect(
            target.canvas.left,
            target.canvas.top,
            target.canvas.width,
            target.canvas.height,
        )
        park = self._park_point(target)
        self._update_progress_state(
            PainterState.RUNNING,
            "Measuring how fast this sign takes a long drag",
            phase="calibrate",
        )
        self._measured_drag_rate = None
        first = grid.columns // 8
        last = grid.columns // 2

        def centre(u_texel: int, v_texel: int) -> tuple[int, int]:
            return (
                int(round(grid.origin_x + (u_texel + 0.5) * grid.pitch_x - canvas.left)),
                int(round(grid.origin_y + (v_texel + 0.5) * grid.pitch_y - canvas.top)),
            )

        clean: list[float] = []
        last_color: RGBColor | None = None
        try:
            before = self._capture_parked(canvas, park, self._pause_generation_value())
            for batch, rate in enumerate(self._DRAG_RATE_PROBE_TEXELS_PER_SECOND):
                epoch = self._pause_generation_value()
                row = grid.rows // 3 + 4 * batch
                if row >= grid.rows - 4:
                    break
                expected = [centre(k, row) for k in range(first, last + 1)]
                color = self._probe_color_against(before, expected, skip=last_color)
                last_color = color
                self._select_color(color, target, settings, epoch, apply_correction=False)
                self._execute_stroke(
                    Stroke(first, row, last, row),
                    job.plan,
                    aiming.paint_canvas,
                    settings,
                    epoch,
                    aiming.bias,
                    0.0,
                    clamp_rect=aiming.clamp_canvas,
                    mapper=aiming.mapper,
                    texel_pitch=aiming.texel_pitch,
                    drag_rate=rate,
                )
                after = self._capture_parked(canvas, park, epoch)
                diff = stamp_diff(before, after)
                before = after
                height, width = diff.shape[:2]

                def changed(u_texel: int, v_texel: int) -> bool:
                    x, y = centre(u_texel, v_texel)
                    window = diff[
                        max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)
                    ]
                    return bool(window.size) and float(window.max()) >= self._LINE_PROBE_DIFF_FLOOR

                interior = range(first + 1, last)
                covered = sum(
                    1
                    for k in interior
                    if any(changed(k, r) for r in (row - 1, row, row + 1) if 0 <= r < grid.rows)
                )
                coverage = covered / max(1, len(interior))
                LOGGER.info(
                    "Drag-rate probe: %d of %d texels painted at %d texels/s",
                    covered,
                    len(interior),
                    int(rate),
                )
                if coverage < 0.98:
                    break
                clean.append(rate)
        except (_RetryAction, _AbortRequested):
            raise
        except Exception as exc:
            LOGGER.warning(
                "The drag-rate probe could not run (%s); keeping %d texels/s",
                exc,
                int(self._LONG_DRAG_MAX_TEXELS_PER_SECOND),
            )
            return
        if len(clean) < 2:
            LOGGER.info(
                "Drag-rate probe kept %d texels/s: %s",
                int(self._LONG_DRAG_MAX_TEXELS_PER_SECOND),
                "no rate painted its run cleanly"
                if not clean
                else "only one rate painted cleanly, which is no margin",
            )
            return
        adopted = clean[-2]
        if adopted <= self._LONG_DRAG_MAX_TEXELS_PER_SECOND:
            LOGGER.info("Drag-rate probe: the %d texels/s cap already has its margin", int(adopted))
            return
        self._measured_drag_rate = float(adopted)
        LOGGER.info(
            "Adopting %d texels/s for this sign's long drags (%d proved clean above it)",
            int(adopted),
            int(clean[-1]),
        )

    _DAB_PROBE_SIZES = DAB_PROBE_SIZES
    _DAB_PROBE_DOTS = DAB_PROBE_DOTS
    _DAB_PROBE_MAX_MISSES = DAB_PROBE_MAX_MISSES
    _DAB_PROBE_MIN_DABS = DAB_PROBE_MIN_DABS

    def _plan_needs_dab_probe(self, plan: PaintPlan) -> bool:
        """Whether proving the detail brush can save more work than it costs."""

        lone = sum(
            1
            for group in plan.color_groups
            for stroke in group.strokes
            if stroke.pixel_count == 1
        )
        return lone >= self._DAB_PROBE_MIN_DABS

    def _probe_dab_size(self, job: _Job) -> None:
        """Prove the one-cell brush on this sign: which Size lands a lone dab.

        Batches of lone-dab strokes - the same ``Stroke`` a plan would hold,
        aimed, held and extended exactly as the artwork's - at each Size in
        :data:`DAB_PROBE_SIZES`, smallest first.  Every batch is captured and
        each dot scored: a hit is a stamp on its own texel, a spill a stamp
        that also reached a neighbour's.  The first Size whose batch misses
        no more than :data:`DAB_PROBE_MAX_MISSES` dots is the job's one-cell
        brush. If none succeeds the job stops before artwork: continuing with
        a brush the probe just proved can miss would make required texels bare.
        """

        settings = job.settings
        target = job.target
        grid = job.texel_grid
        box = target.brush_size_box
        model = target.brush_size_model
        if (
            not settings.measure_dab_size
            or not settings.apply_brush_size
            or grid is None
            or box is None
            or model is None
            or job.mode != "paint"
        ):
            return
        plan = job.plan
        aiming = self._aiming(job, plan)
        if not aiming.native or aiming.mapper is None:
            return
        lone = sum(
            1
            for group in plan.color_groups
            for stroke in group.strokes
            if stroke.pixel_count == 1
        )
        if not self._plan_needs_dab_probe(plan):
            LOGGER.info(
                "The plan's %d lone dabs are too few for the dab probe to pay "
                "for its captures; the one-cell brush stays the smallest",
                lone,
            )
            return
        dots = max(8, int(self._DAB_PROBE_DOTS))
        if grid.columns < 24 or grid.rows < 16:
            return
        canvas = ScreenRect(
            target.canvas.left,
            target.canvas.top,
            target.canvas.width,
            target.canvas.height,
        )
        park = self._park_point(target)
        self._measured_detail_size = None
        self._update_progress_state(
            PainterState.RUNNING,
            "Proving the one-cell brush with batches of lone dabs",
            phase="calibrate",
        )

        def batch_texels(batch: int) -> list[tuple[int, int]]:
            # Spread across the columns, rows staggered per batch so no
            # batch stamps where an earlier one did, and never on the
            # texel next to another dot of the same batch.
            span = grid.columns - 8
            return [
                (
                    4 + round(index * (span - 1) / (dots - 1)),
                    4 + ((index * 37 + batch * 23) % max(1, grid.rows - 8)),
                )
                for index in range(dots)
            ]

        def centre(u_texel: int, v_texel: int) -> tuple[int, int]:
            return (
                int(round(grid.origin_x + (u_texel + 0.5) * grid.pitch_x - canvas.left)),
                int(round(grid.origin_y + (v_texel + 0.5) * grid.pitch_y - canvas.top)),
            )

        def changed(diff: Any, u_texel: int, v_texel: int) -> float:
            if not (0 <= u_texel < grid.columns and 0 <= v_texel < grid.rows):
                return 0.0
            x, y = centre(u_texel, v_texel)
            window = diff[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
            return float(window.max()) if window.size else 0.0

        last_color: RGBColor | None = None
        chosen: tuple[float, bool] | None = None
        # Point before sweep at each Size: the least invasive stamp that
        # lands every dot wins.  A point touches nothing but its texel; a
        # sweep reaches sideways, and a wider brush reaches all round. A wider
        # native-resolution stamp can touch neighbours, but a Size-1 stamp
        # narrower than a texel can leave the requested texel completely bare.
        # Coverage wins that trade only after the smaller trials have failed,
        # and the final export still catches collateral wrong colors.
        trials = [
            (size, sweep)
            for size in self._DAB_PROBE_SIZES
            for sweep in (False, True)
        ]
        try:
            epoch = self._pause_generation_value()
            before = self._capture_parked(canvas, park, epoch)
            for batch, (size, sweep) in enumerate(trials):
                epoch = self._pause_generation_value()
                texels = batch_texels(batch)
                expected = [
                    (
                        grid.origin_x + (u + 0.5) * grid.pitch_x - canvas.left,
                        grid.origin_y + (v + 0.5) * grid.pitch_y - canvas.top,
                    )
                    for u, v in texels
                ]
                self._write_brush_size(box, size, settings, epoch)
                color = self._probe_color_against(before, expected, skip=last_color)
                last_color = color
                self._select_color(color, target, settings, epoch, apply_correction=False)
                extension = (
                    self._stroke_extension_pixels(
                        aiming.paint_canvas, plan, model, size
                    )
                    if sweep
                    else 0.0
                )
                for u, v in texels:
                    self._execute_stroke(
                        Stroke(u, v, u, v),
                        plan,
                        aiming.paint_canvas,
                        settings,
                        epoch,
                        aiming.bias,
                        extension,
                        clamp_rect=aiming.clamp_canvas,
                        mapper=aiming.mapper,
                        texel_pitch=aiming.texel_pitch,
                    )
                    self._interruptible_sleep(
                        self._stroke_gap(settings.delay_between_strokes_seconds),
                        epoch=epoch,
                        check_focus=True,
                    )
                after = self._capture_parked(canvas, park, epoch)
                diff = stamp_diff(before, after)
                before = after
                hits = 0
                spills = 0
                for u, v in texels:
                    own = changed(diff, u, v)
                    around = [changed(diff, u - 1, v), changed(diff, u + 1, v), changed(diff, u, v - 1), changed(diff, u, v + 1)]
                    peak = max([own] + around)
                    if own >= self._LINE_PROBE_DIFF_FLOOR and own >= 0.5 * peak:
                        hits += 1
                        spills += sum(
                            1 for value in around if value >= self._LINE_PROBE_DIFF_FLOOR and value >= 0.6 * own
                        )
                misses = dots - hits
                LOGGER.info(
                    "Dab probe: %d of %d lone dabs landed at Size %s %s, spilling "
                    "into %.2f neighbours each",
                    hits,
                    dots,
                    format_brush_size(size),
                    "swept across the texel" if sweep else "pressed at a point",
                    spills / max(1, hits),
                )
                if misses <= self._DAB_PROBE_MAX_MISSES:
                    chosen = (float(size), sweep)
                    break
        except (_RetryAction, _AbortRequested):
            raise
        except Exception as exc:
            raise RuntimeError(
                "The one-cell brush could not be proved on this sign; no artwork "
                f"was painted ({exc})"
            )
        if chosen is None:
            tried = ", ".join(
                format_brush_size(size) for size in self._DAB_PROBE_SIZES
            )
            raise RuntimeError(
                "No one-cell brush reliably covered its requested texels. "
                f"Tried Size {tried}; no artwork was painted."
            )
        size, sweep = chosen
        self._measured_detail_size = size
        self._measured_dab_sweep = sweep
        LOGGER.info(
            "Adopting Size %s %s for this sign's one-cell strokes",
            format_brush_size(size),
            "swept across each texel" if sweep else "pressed at a point",
        )

    def _probe_line_tool(self, job: _Job) -> bool:
        """Prove the Shift line tool on this sign with one stroke.

        A press a quarter of the way along the middle row and a release
        three quarters along it, Shift held throughout with one cursor jump
        between, then a capture: if the game filled the straight stroke
        between them, the texels along the span read as changed and every
        long straight run in the plan may go down the same way - a press, a
        jump and a release instead of a rate-capped drag, with the game
        itself filling the texels between, beyond the reach of the per-dab
        cursor quantization a DPI-scaled display adds.  Anything less - the
        mechanic missing, the modifier not seen, the line landing somewhere
        unexpected - leaves the tool unproven and every stroke a drag, which
        is the path measured over thousands of live strokes.  The probe line
        is wiped with the other calibration marks before the artwork starts.
        """

        settings = job.settings
        grid = job.texel_grid
        if not settings.use_line_tool or grid is None or job.mode != "paint":
            return False
        longest = 0
        for group in job.plan.color_groups:
            # A Shift line is only proved at the one-cell detail size below.
            # Rust has been observed drawing the same thin line when a wider
            # Size was selected, leaving a bare scanline between the
            # optimizer's multi-row bands.  Broad groups therefore stay on
            # the dense-drag path whose full brush footprint was measured.
            if max(1, int(group.brush_diameter)) != 1:
                continue
            for stroke in group.strokes:
                if stroke.start_x == stroke.end_x or stroke.start_y == stroke.end_y:
                    longest = max(longest, stroke.pixel_count)
        if longest < self._SHIFT_LINE_MIN_TEXELS:
            LOGGER.info(
                "No straight run reaches %d texels; the line tool is not probed",
                int(self._SHIFT_LINE_MIN_TEXELS),
            )
            return False
        row = grid.rows // 2
        first = grid.columns // 4
        last = grid.columns - 1 - grid.columns // 4
        if last - first < 8:
            return False
        target = job.target
        canvas = ScreenRect(
            target.canvas.left,
            target.canvas.top,
            target.canvas.width,
            target.canvas.height,
        )
        park = self._park_point(target)
        epoch = self._pause_generation_value()
        self._update_progress_state(
            PainterState.RUNNING,
            "Proving the Shift-click line tool with one stroke",
            phase="calibrate",
        )

        def center(u_texel: int, v_texel: int) -> tuple[float, float]:
            return (
                grid.origin_x + (u_texel + 0.5) * grid.pitch_x - canvas.left,
                grid.origin_y + (v_texel + 0.5) * grid.pitch_y - canvas.top,
            )

        try:
            before = self._capture_parked(canvas, park, epoch)
            color = self._probe_color_against(
                before, [center(k, row) for k in range(first, last + 1)]
            )
            self._select_color(color, target, settings, epoch, apply_correction=False)
            x0, y0 = grid.cursor_point(first + 0.5, row + 0.5)
            x1, y1 = grid.cursor_point(last + 0.5, row + 0.5)
            # One shared y, exactly as the executor commands a same-row run.
            shared_y = math.floor((y0 + y1) / 2.0 + 0.5)
            start = (math.floor(x0 + 0.5), shared_y)
            end = (math.floor(x1 + 0.5), shared_y)
            self._line_stroke(start, end, settings, epoch)
            after = self._capture_parked(canvas, park, epoch)
        except (_RetryAction, _AbortRequested):
            raise
        except Exception as exc:
            LOGGER.warning(
                "The line-tool probe could not run (%s); long runs stay drags", exc
            )
            return False
        diff = stamp_diff(before, after)
        height, width = diff.shape[:2]

        def changed(u_texel: int, v_texel: int) -> bool:
            x, y = center(u_texel, v_texel)
            column, line_row = int(round(x)), int(round(y))
            window = diff[
                max(0, line_row - 1) : min(height, line_row + 2),
                max(0, column - 1) : min(width, column + 2),
            ]
            return bool(window.size) and float(window.max()) >= self._LINE_PROBE_DIFF_FLOOR

        interior = range(first + 1, last)
        # The endpoints are ordinary presses; the interior is what only the
        # game's own line can have painted.  The stroke may ride a texel row
        # boundary, so each column accepts the row or either neighbour.
        covered = sum(
            1
            for k in interior
            if any(
                changed(k, r) for r in (row - 1, row, row + 1) if 0 <= r < grid.rows
            )
        )
        stray_rows = [r for r in (row - 3, row + 3) if 0 <= r < grid.rows]
        strayed = sum(1 for k in interior for r in stray_rows if changed(k, r))
        coverage = covered / max(1, len(interior))
        stray = strayed / max(1, len(interior) * max(1, len(stray_rows)))
        ok = coverage >= 0.9 and stray <= 0.3
        LOGGER.info(
            "Line-tool probe: %d of %d texels painted along the stroke, %.0f%% "
            "spill three rows out - %s",
            covered,
            len(interior),
            100.0 * stray,
            "long straight runs will use Shift-click lines"
            if ok
            else "long runs stay drags",
        )
        return ok

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
        sizes: list[float] = []
        for multiple in self._BRUSH_PROBE_MULTIPLES:
            size = float(
                min(BRUSH_SIZE_MAX, max(BRUSH_SIZE_MIN, round(wanted * multiple * 4) / 4))
            )
            if all(abs(size - existing) >= 0.25 for existing in sizes):
                sizes.append(size)
        LOGGER.info(
            "Brush scout: one cell needs about size %.2f, probing %s",
            wanted,
            ", ".join(format_brush_size(size) for size in sizes),
        )
        return tuple(sizes)

    # Rust has been observed running its painting UI at 15 FPS, so a capture
    # taken the instant the cursor moves away can still hold the frame that had
    # it in shot.  Five frames is comfortably past that.
    _CAPTURE_SETTLE_SECONDS = 0.35

    # Where the cursor waits between strokes and while the sign is read.
    # Not over the colour picker: hovering it between bursts of presses made
    # the game discard most of every third burst (measured live, 7 of 20
    # bursts lost from their fourth press on; none lost when parked on the
    # dark panel beside the sign).  Nothing but the sign lives to its left.
    _PARK_MARGIN_PIXELS = 40

    @classmethod
    def _park_point(cls, target: PaintingTarget) -> tuple[int, int]:
        canvas = target.canvas
        x = int(round(canvas.left - cls._PARK_MARGIN_PIXELS))
        y = int(round(canvas.top + canvas.height / 2.0))
        return x, y

    def _capture_parked(
        self, canvas: ScreenRect, park: tuple[int, int], epoch: int
    ) -> Any:
        """Move the cursor off the sign, let Rust settle, then capture it."""

        self._move(park, epoch)
        if self.input.emits_real_input:
            self._interruptible_sleep(
                self._CAPTURE_SETTLE_SECONDS, epoch=epoch, check_focus=True
            )
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

    def _stroke_timing(self, settings: PainterSettings) -> StrokeTiming:
        probes = bool(settings.measure_press_hold)
        return StrokeTiming.from_settings(
            settings,
            overhead_seconds=self._stroke_overhead_seconds,
            real_input=bool(getattr(self.input, "emits_real_input", True)),
            dab_press_seconds=self._measured_press_hold_seconds if probes else None,
            gap_seconds=self._measured_stroke_gap_seconds if probes else None,
            drag_texels_per_second=self._measured_drag_rate if probes else None,
        )

    def _work_schedule(
        self,
        plan: PaintPlan,
        target: PaintingTarget,
        settings: PainterSettings,
        grid: TexelGridModel | None = None,
        *,
        line_tool: bool = False,
    ) -> PlanWorkSchedule:
        cell_width = target.canvas.width / max(1, plan.width)
        sizing = bool(
            settings.apply_brush_size
            and target.brush_size_box is not None
            and target.brush_size_model is not None
        )
        pitch = self._texel_pitch_pixels(
            plan,
            target.canvas,
            target.brush_size_model if sizing else None,
            grid,
        )
        return PlanWorkSchedule(
            plan,
            self._stroke_timing(settings),
            cell_width,
            sizing=sizing,
            texel_pitch_pixels=pitch,
            line_min_pixels=(
                (self._SHIFT_LINE_MIN_TEXELS - 1.0) * pitch if line_tool else None
            ),
        )

    def _initial_estimate(self, job: _Job) -> float | None:
        """Time left before a single stroke has gone down: the model's word.

        Includes the brush measurement a sizing job runs first, so the figure
        shown while the probes are painted is already the whole job's.
        """

        if job.mode != "paint":
            return None
        try:
            schedule = self._work_schedule(job.plan, job.target, job.settings)
            total = schedule.total - self._skipped_work(
                schedule, job.plan, job.start_stroke
            )
        except Exception:  # an estimate must never stop a job from starting
            LOGGER.debug("Initial time estimate failed", exc_info=True)
            return None
        # The checks and the touch-up scale with the painting, not with the
        # brush measurement before it.
        total += self._checks_estimate(job, total) + self._touch_up_estimate(job, total)
        if (
            job.start_stroke == 0
            and job.settings.apply_brush_size
            and getattr(self.input, "emits_real_input", True)
            and job.target.brush_size_box is not None
        ):
            reusable = bool(
                job.settings.reuse_calibration
                and job.target.brush_size_model is not None
                and job.target.texel_grid is not None
                and job.target.texel_grid.agrees_with(job.target.canvas)
            )
            total += FAST_CALIBRATION_SECONDS if reusable else BRUSH_CALIBRATION_SECONDS
        return total

    def _checks_estimate(self, job: _Job, paint_seconds: float) -> float:
        """What checking each color as it goes down should add to ``paint_seconds``.

        Priced from what earlier runs measured: a capture per color that
        paints, and a share of the painting time in repaints.  Only before
        the first stroke - once the job is under way the checks are on its
        clock and the measured pace carries them.
        """

        if not job.settings.confirm_strokes or not getattr(
            self.input, "emits_real_input", True
        ):
            return 0.0
        colors = sum(1 for group in job.plan.color_groups if group.strokes)
        return colors * self._check_capture_seconds + max(
            0.0, paint_seconds
        ) * self._check_repaint_fraction

    def _touch_up_estimate(self, job: _Job, paint_seconds: float) -> float:
        """What the touch-up pass after ``paint_seconds`` of painting should take."""

        if job.settings.verify_passes <= 0 or not getattr(
            self.input, "emits_real_input", True
        ):
            return 0.0
        return max(0.0, paint_seconds) * self._touch_up_fraction

    @staticmethod
    def _skipped_work(
        schedule: PlanWorkSchedule, plan: PaintPlan, start_stroke: int
    ) -> float:
        """Predicted seconds of the plan's first ``start_stroke`` strokes."""

        skipped = 0.0
        remaining = start_stroke
        for group_index, group in enumerate(plan.color_groups):
            if remaining <= 0:
                break
            skipped += schedule.group_cost(group_index)
            taken = min(remaining, len(group.strokes))
            skipped += sum(
                schedule.stroke_cost(group_index, index) for index in range(taken)
            )
            remaining -= taken
        return skipped

    @property
    def paint_phase_timing(self) -> PhaseTiming | None:
        """Predicted versus measured seconds for the artwork's own strokes.

        Updated after every stroke of the main plan (touch-up passes are not
        counted), so an aborted run still reports what it measured.  Callers
        fold it into the learned per-stroke overhead for the next estimate.
        """

        with self._condition:
            if self._timing_retuned:
                return None
            return self._paint_phase_timing

    @property
    def touch_up_timing(self) -> TouchUpTiming | None:
        """How long the touch-up pass took, once it has run to its end.

        ``None`` while the pass is still to come, was not asked for, or was
        interrupted - a fraction of a pass says nothing about a whole one.
        """

        with self._condition:
            return self._touch_up_timing

    def _aiming(self, job: _Job, plan: PaintPlan) -> "_Aiming":
        """Where this plan's cells are on the screen, and how they are hit.

        The measured texel grid's cursor map when the job has one (with the
        plan's cells laid one-to-one on the texels, or stretched when the
        plan disagrees with the grid), else the brush-derived canvas and
        rendering bias.  Shared by the artwork, the touch-up and the dab
        probe, so a probe stroke lands exactly as an artwork stroke would.
        """

        target, settings = job.target, job.settings
        scale_u = scale_v = 1.0
        sizing_enabled = (
            settings.apply_brush_size
            and target.brush_size_box is not None
            and target.brush_size_model is not None
        )
        # The sign renders strokes slightly off from where they are commanded
        # (about a texel left and a fraction of one down, measured live).
        # Aiming every artwork coordinate the same distance the other way
        # cancels it.  Probe strokes never come through here, so they keep
        # measuring the raw response.  Only a freshly measured model is
        # trusted: with sizing off the model on file may describe another
        # sign, and a stale shift is worse than none.
        model = target.brush_size_model if sizing_enabled else None
        if model is not None:
            bias = (
                model.bias_x * target.canvas.width,
                model.bias_y * target.canvas.height,
            )
        else:
            bias = (0.0, 0.0)
        # Strokes are laid out on the texture's canonical extent rather than
        # the hand-dragged rectangle, so the cell pitch is texel-exact; the
        # physical rectangle still bounds every actual mouse coordinate.
        paint_canvas = self._registered_canvas(target.canvas, model)
        clamp_canvas: RectangleLike = target.canvas
        mapper: Callable[[float, float], tuple[float, float]] | None = None
        grid = job.texel_grid
        if grid is not None and grid.agrees_with(target.canvas):
            # Measured, not inferred: the cursor map, which says for every
            # texel exactly where the cursor stamps it - including the shear
            # a cursor ray-cast onto the sign in the world has against the
            # flat canvas.  The brush-derived bias describes the same thing
            # less precisely and for one spot only, so the grid replaces it.
            # The mouse stays on the texture, where the game takes clicks.
            bias = (0.0, 0.0)
            clamp_canvas = grid.clamp_rect(target.canvas)
            # The plan and the grid should agree exactly; when a stale plan
            # is a texel or two off the freshly measured grid, cells map onto
            # texels one to one rather than being stretched - a uniform
            # rescale turns a one-column disagreement into a half-texel
            # misplacement by mid-sign (measured live on the murica run:
            # scale 1026/1025 displaced fine detail across the whole sign).
            if (plan.width, plan.height) != (grid.columns, grid.rows):
                if (
                    abs(plan.width - grid.columns) <= 2
                    and abs(plan.height - grid.rows) <= 2
                ):
                    scale_u = 1.0
                    scale_v = 1.0
                    LOGGER.warning(
                        "Plan is %dx%d but the measured grid is %dx%d: aiming "
                        "cells one-to-one on texels; re-plan to use the full "
                        "sign",
                        plan.width,
                        plan.height,
                        grid.columns,
                        grid.rows,
                    )
                else:
                    scale_u = grid.columns / plan.width
                    scale_v = grid.rows / plan.height
                    LOGGER.warning(
                        "Plan is %dx%d but the measured grid is %dx%d: "
                        "stretching to fit - detail will land off its texels; "
                        "re-plan at the measured size",
                        plan.width,
                        plan.height,
                        grid.columns,
                        grid.rows,
                    )
            else:
                scale_u = 1.0
                scale_v = 1.0

            def mapper(cell_x: float, cell_y: float) -> tuple[float, float]:
                # Returned unrounded: the stroke applies its extension in
                # float and rounds once at the end, so a sub-pixel hedge can
                # never knock an endpoint a whole pixel sideways.
                x, y = grid.cursor_point((cell_x + 0.5) * scale_u, (cell_y + 0.5) * scale_v)
                return x, y
        texel_pitch = self._texel_pitch_pixels(plan, paint_canvas, model, grid)
        native = mapper is not None and scale_u == 1.0 and scale_v == 1.0
        swept = grid if (native and grid is not None and grid.swept) else None
        return _Aiming(
            sizing=sizing_enabled,
            model=model,
            bias=bias,
            paint_canvas=paint_canvas,
            clamp_canvas=clamp_canvas,
            mapper=mapper,
            texel_pitch=texel_pitch,
            native=native,
            swept=swept,
        )

    def _execute_plan(
        self,
        job: _Job,
        plan: PaintPlan | None = None,
        *,
        reference: Any = None,
    ) -> None:
        """Paint ``plan`` (the job's own by default), color by color.

        ``reference`` is a per-cell reading of the sign from just before this
        plan goes down, for checking each color against; the job's plan
        reads its own from the cleared sign, or captures one.
        """

        main_plan = plan is None
        plan = job.plan if plan is None else plan
        target, settings = job.target, job.settings
        # Strokes before the offset are already on the sign.  They count in
        # the progress shown - the sign really is that far along - but not
        # in the pace, which is measured on what this run paints.
        skip = job.start_stroke if main_plan else 0
        completed = 0
        painted = 0
        total = sum(len(group.strokes) for group in plan.color_groups)
        total_colors = len(plan.color_groups)
        # Progress is shown in the strokes of the plan the job was given,
        # which a native layout expands into more.
        origins = job.stroke_origins if main_plan else None
        shown_total = job.original_total if (main_plan and origins) else total

        def shown(completed_native: int) -> int:
            if not origins or completed_native <= 0:
                return completed_native
            return origins[min(completed_native, len(origins)) - 1] + 1
        # Progress advances in predicted seconds, priced from the same timing
        # rules the strokes below execute with, so percent and time left move
        # at the pace of the clock instead of racing through the big,
        # long-stroke colors and crawling through the small ones.
        schedule = self._work_schedule(
            plan,
            target,
            settings,
            job.texel_grid,
            line_tool=job.line_tool_ok and settings.use_line_tool,
        )
        total_work = schedule.total
        completed_work = 0.0
        skipped_work = 0.0
        # Each plan gets its own clock.  A touch-up pass re-enters here after
        # the artwork is done; timed against the whole run's elapsed it would
        # claim hours left for a few minutes of repainting.
        phase_started = self._active_elapsed()

        def record_phase_timing(completed_work: float, painted: int) -> None:
            """Keep the artwork's clock against its prediction up to date."""

            if not main_plan:
                return
            with self._condition:
                self._paint_phase_timing = PhaseTiming(
                    predicted_seconds=completed_work,
                    actual_seconds=self._active_elapsed() - phase_started,
                    strokes=painted,
                    checking_seconds=self._confirmation_seconds,
                    colors_checked=self._confirmation_summary.colors,
                    check_capture_seconds=self._check_capture_clock,
                )

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
        aiming = self._aiming(job, plan)
        sizing_enabled = aiming.sizing
        model = aiming.model
        bias = aiming.bias
        paint_canvas = aiming.paint_canvas
        clamp_canvas = aiming.clamp_canvas
        mapper = aiming.mapper
        texel_pitch = aiming.texel_pitch
        # Physical brush facts and the pause epoch they were established under.
        # A pause hands the mouse back to the user, who may change the brush in
        # Rust, so an epoch bump re-applies the size before the next stroke -
        # mirroring the (color, epoch) guard on the picker selection.
        applied_diameter: int | None = None
        applied_epoch: int | None = None
        selected: tuple[RGBColor, int] | None = None
        confirmation = self._confirmation_for(job, plan, reference=reference)
        # Interim export audits: every settings.interim_audit_seconds of the
        # main plan, between color groups, the sign's texture is read and
        # the holes in what is already final are refilled while the paint
        # is fresh - a long run learns of the game's dropped dabs in
        # minutes, instead of after hours at the final verification.
        audit_enabled = (
            main_plan
            and settings.interim_audit_seconds > 0
            and target.download_button is not None
            and getattr(self.input, "emits_real_input", True)
        )
        audit_layers: Any = None
        last_audit = phase_started
        painted_at_audit = 0
        for color_index, group in enumerate(plan.color_groups, start=1):
            if (
                audit_enabled
                and color_index > 1
                and completed >= skip
                and painted - painted_at_audit >= self._INTERIM_AUDIT_MIN_STROKES
                and self._active_elapsed() - last_audit >= settings.interim_audit_seconds
            ):
                if audit_layers is None:
                    from .verification import plan_layers

                    audit_indices, _underpaint, audit_palette = plan_layers(plan)
                    audit_layers = (
                        audit_indices,
                        audit_palette,
                        self._final_group_map(plan),
                    )
                if self._interim_audit(job, color_index - 1, audit_layers):
                    # The repaint selected its own colors and Size; what the
                    # loop believed applied no longer is.
                    selected = None
                    applied_diameter = None
                last_audit = self._active_elapsed()
                painted_at_audit = painted
            if skip > completed and skip - completed >= len(group.strokes):
                # The whole group is behind the offset.
                skipped_work += schedule.group_cost(color_index - 1) + sum(
                    schedule.stroke_cost(color_index - 1, index)
                    for index in range(len(group.strokes))
                )
                completed += len(group.strokes)
                continue
            first_index = skip - completed
            if first_index > 0:
                skipped_work += sum(
                    schedule.stroke_cost(color_index - 1, index)
                    for index in range(first_index)
                )
                completed += first_index
            completed_work += schedule.group_cost(color_index - 1)
            diameter = max(1, int(group.brush_diameter))
            # The sideways reach that covers a cell wider than the brush.
            # On a native grid a cell is one texel, and whether a lone dab
            # should sweep across it or press at a point is measured
            # (:meth:`_probe_dab_size`), not assumed.  Unmeasured, it
            # presses: the sign's own exported texture showed a third of its
            # lone texels painted a neighbour's colour, twice as often
            # sideways as up or down - the sweep's endpoints reaching past
            # the texel's edge - while the point presses of its touch-up
            # passes landed 83% of the hardest cells at the first attempt.
            swept = self._measured_dab_sweep
            extension = (
                self._stroke_extension_pixels(
                    paint_canvas,
                    plan,
                    model,
                    self._brush_plan_size(
                        target, plan, diameter, settings.logical_pixel_spacing, model
                    ),
                )
                if model is not None and (swept or not aiming.native)
                else 0.0
            )

            def paint_stroke(
                stroke: object,
                *,
                group=group,
                diameter: int = diameter,
                extension: float = extension,
            ) -> None:
                nonlocal applied_diameter, applied_epoch, selected
                while True:
                    self._checkpoint(check_focus=True)
                    # A pause may have retuned the job, so every stroke is
                    # held and paced by the settings the job has now, not the
                    # ones it started with.  The schedule keeps its original
                    # pricing: percent still climbs monotonically, and the
                    # time left is corrected by the measured pace anyway.
                    current = job.settings
                    try:
                        # Leaving the sign and coming back starts a new
                        # epoch, so the color and brush are set again before
                        # the stroke; a pause during the break retries it.
                        self._anti_afk_break_if_due(job)
                        current_epoch = self._pause_generation_value()
                        if applied_epoch != current_epoch:
                            applied_diameter = None
                            applied_epoch = current_epoch
                        if sizing_enabled and diameter != applied_diameter:
                            self._apply_brush_size(job, diameter, current_epoch)
                            applied_diameter = diameter
                        if selected != (group.color, current_epoch):
                            self._select_color(group.color, target, current, current_epoch)
                            selected = (group.color, current_epoch)
                        self._execute_stroke(
                            stroke,
                            plan,
                            paint_canvas,
                            current,
                            current_epoch,
                            bias,
                            extension,
                            clamp_rect=clamp_canvas,
                            mapper=mapper,
                            texel_pitch=texel_pitch,
                            # The startup probe proves a Shift line only at
                            # the one-cell detail size.  On the XXL sign Rust
                            # accepted a wider Size value but rendered broad
                            # Shift lines as thin scanlines, so three-row
                            # groups left regular horizontal gaps.  Dense
                            # drags honor the measured footprint and spacing.
                            line_tool=(
                                job.line_tool_ok
                                and current.use_line_tool
                                and diameter == 1
                            ),
                            drag_rate=self._drag_rate_cap(),
                            swept=aiming.swept,
                        )
                        return
                    except _RetryAction:
                        selected = None
                        continue

            painted_in_group = 0
            for index_in_group, stroke in enumerate(group.strokes, start=1):
                if index_in_group <= first_index:
                    continue
                paint_stroke(stroke)
                painted_in_group += 1
                completed += 1
                painted += 1
                completed_work += schedule.stroke_cost(color_index - 1, index_in_group - 1)
                phase_elapsed = self._active_elapsed() - phase_started
                record_phase_timing(completed_work, painted)
                self._set_progress(
                    color_index=color_index,
                    total_colors=total_colors,
                    stroke_index_in_color=index_in_group,
                    strokes_in_color=len(group.strokes),
                    completed_strokes=shown(completed),
                    total_strokes=shown_total,
                    completed_work=completed_work,
                    total_work=total_work,
                    skipped_work=skipped_work,
                    phase_elapsed=phase_elapsed,
                    pending_seconds=(
                        self._touch_up_estimate(job, total_work) if main_plan else 0.0
                    ),
                    message="Painting",
                )
                self._interruptible_sleep(
                    self._stroke_gap(job.settings.delay_between_strokes_seconds),
                    check_focus=True,
                )
            if confirmation is not None and painted_in_group:
                self._confirm_group(
                    confirmation,
                    job,
                    plan,
                    color_index,
                    group,
                    paint_stroke,
                )
                # The check comes after the color's last stroke, so the
                # record kept per stroke would otherwise miss the last one.
                record_phase_timing(completed_work, painted)
            self._interruptible_sleep(
                self._settle(job.settings.delay_between_colors_seconds),
                check_focus=True,
            )

    # ------------------------------------------- checking colors as they land

    def _press_hold_seconds(self, settings: PainterSettings) -> float:
        """How long a stationary press is held.

        The set hold lifted to the frame floor - or, when this sign's probe
        proved a shorter one lands every dot, that measured hold.  Only
        stationary presses (dabs, the line tool's clicks) take the measured
        value; drag dwells use :meth:`_drag_dwell_seconds`, because the
        measurement covers presses that never moved while dropped short
        drags were a real live failure the dwell exists to prevent.
        """

        hold = settings.mouse_down_duration_seconds
        if self.input.emits_real_input:
            hold = max(hold, self._MIN_PRESS_SECONDS)
        measured = self._measured_press_hold_seconds
        if measured is not None and settings.measure_press_hold:
            hold = min(hold, measured)
        return hold

    def _drag_dwell_seconds(self, settings: PainterSettings) -> float:
        """The frame a drag dwells at its far end: never below the floor."""

        hold = settings.mouse_down_duration_seconds
        if self.input.emits_real_input:
            hold = max(hold, self._MIN_PRESS_SECONDS)
        return hold

    def _cell_sampling(
        self, job: _Job, plan: PaintPlan, canvas: ScreenRect
    ) -> tuple[tuple[Any, Any] | None, Any]:
        """Where the plan's cells are read in a capture, and how they blend.

        The centres follow the measured grid when there is one (else the
        rectangle's even spacing, which is what the sampler uses on its
        own); the blend describes the bilinear mix a capture under three
        pixels per cell reads at each centre, and is ``None`` where the
        sampler takes a 3x3 median instead.
        """

        import numpy as np

        from .verification import MEDIAN_SAMPLING_MIN_CELL_PIXELS, CellBlend

        centers = self._grid_cell_centers(job, plan, canvas)
        if centers is not None:
            rect = job.texel_grid.registered_rect()  # type: ignore[union-attr]
            pitch_x = rect.width / plan.width
            pitch_y = rect.height / plan.height
            centers_x, centers_y = centers
        else:
            pitch_x = canvas.width / max(1, plan.width)
            pitch_y = canvas.height / max(1, plan.height)
            centers_x = (np.arange(plan.width) + 0.5) * pitch_x
            centers_y = (np.arange(plan.height) + 0.5) * pitch_y
        # The sampler switches to the centre pixel below three pixels per
        # cell, measured on the rectangle as it does (see sample_cell_colors).
        fine = (
            min(
                canvas.height / max(1, plan.height),
                canvas.width / max(1, plan.width),
            )
            < MEDIAN_SAMPLING_MIN_CELL_PIXELS
        )
        blend = (
            CellBlend.from_centers(centers_x, centers_y, pitch_x, pitch_y)
            if fine
            else None
        )
        return centers, blend

    def _confirmation_for(
        self, job: _Job, plan: PaintPlan, *, reference: Any = None
    ) -> _Confirmation | None:
        """Set up checking ``plan``'s colors as they go down, or say why not.

        The reference reading of the sign comes from the cleared sign the
        calibration kept, from ``reference`` (a touch-up pass hands over the
        capture it was planned from), or from a capture taken now - a
        resumed job, or one painting with sizing off, has nothing else.
        """

        import numpy as np

        from .verification import plan_layers, sample_cell_colors

        settings = job.settings
        summary_lock = self._condition

        def skip(reason: str) -> None:
            with summary_lock:
                self._confirmation_summary = replace(
                    self._confirmation_summary, skipped_reason=reason
                )

        if not settings.confirm_strokes:
            skip("turned off")
            return None
        if not getattr(self.input, "emits_real_input", True):
            skip("no real input")
            return None
        if not any(group.strokes for group in plan.color_groups):
            return None
        target = job.target
        canvas = ScreenRect(
            target.canvas.left,
            target.canvas.top,
            target.canvas.width,
            target.canvas.height,
        )
        park = self._park_point(target)
        indices, _underpaint, palette = plan_layers(plan)
        centers, blend = self._cell_sampling(job, plan, canvas)

        def sampled(capture: Any) -> Any:
            return sample_cell_colors(
                np.asarray(capture.convert("RGB"), dtype=np.float32),
                plan.width,
                plan.height,
                centers=centers,
            )

        if reference is None and plan is job.plan and job.bare_canvas is not None:
            try:
                reference = sampled(job.bare_canvas)
            except Exception:
                LOGGER.exception("The cleared sign's capture could not be sampled")
                reference = None
        if reference is None:
            for _attempt in range(self._CALIBRATION_ATTEMPTS):
                epoch = self._pause_generation_value()
                try:
                    reference = sampled(self._capture_parked(canvas, park, epoch))
                    break
                except _RetryAction:
                    continue
                except _AbortRequested:
                    raise
                except Exception:
                    LOGGER.exception("The sign could not be captured before painting")
                    skip("the sign could not be captured before painting")
                    return None
            else:
                skip("paused every time the sign was captured before painting")
                return None
        reference = np.asarray(reference, dtype=np.float32)
        if reference.shape[:2] != (plan.height, plan.width):
            skip("the reference capture does not match the plan")
            return None
        LOGGER.info(
            "Each color is checked as it goes down (up to %d repaint rounds); "
            "cells are read %s",
            settings.confirm_max_rounds,
            "through the bilinear blend of a fine sign"
            if blend is not None
            else "one per cell",
        )
        return _Confirmation(
            canvas=canvas,
            park=park,
            indices=indices,
            palette=palette,
            centers=centers,
            blend=blend,
            reference=reference,
        )

    def _publish_confirmation(self, state: _Confirmation) -> None:
        with self._condition:
            self._confirmation_summary = ConfirmationSummary(
                colors=state.colors,
                judged=state.judged,
                missed=state.missed,
                repainted_strokes=state.repainted_strokes,
                unrepaired=state.unrepaired,
                rounds=state.rounds,
                skipped_reason=self._confirmation_summary.skipped_reason,
            )

    def _confirm_group(
        self,
        state: _Confirmation,
        job: _Job,
        plan: PaintPlan,
        color_index: int,
        group: Any,
        paint_stroke: Callable[[object], None],
    ) -> None:
        """Capture the sign after a color and repaint the cells that missed it.

        Read against the sign as it stood before the color, so no palette
        or lighting model has to be right: a cell either moved to where the
        color's stamp would put it or it stayed where it was.  Misses are
        repainted in runs and the capture repeated, up to the configured
        rounds, and the last capture becomes the next color's reference.
        The press hold is adapted from the first round's miss rate.
        """

        import numpy as np

        from .verification import (
            apply_capture_lighting,
            confirm_cells,
            fit_sign_rendering,
            repaint_runs,
            sample_cell_colors,
            stroke_coverage,
        )

        settings = job.settings
        group_number = color_index
        total_colors = len(plan.color_groups)
        color = np.asarray(group.color, dtype=np.uint8)
        palette_index = int(np.flatnonzero((state.palette == color).all(axis=1))[0])
        diameter = max(1, int(getattr(group, "brush_diameter", 1)))
        radius = (diameter - 1) // 2
        # Only the cells this group's own strokes cover and that keep its
        # color: an optimized plan paints one color in several passes, and
        # the later passes' cells are not missing yet.
        covered = np.zeros(state.indices.shape, dtype=np.bool_)
        for stroke in group.strokes:
            covered[stroke_coverage(stroke, radius, covered.shape)] = True
        judge = covered & (state.indices == palette_index)
        if not judge.any():
            return
        # What this color should look like on this sign, from the colors
        # already on it; its nominal value until enough of them are down.
        earlier = np.where(state.indices < palette_index, state.indices, -1)
        coefficients = fit_sign_rendering(state.reference, earlier, state.palette)
        expected = apply_capture_lighting(
            np.asarray([group.color], dtype=np.float32), coefficients
        )[0]
        missed_before: int | None = None
        missed_count = 0
        stalled = 0
        latest = state.reference
        checking_started = self._active_elapsed()
        # The first round's capture is the fixed cost of checking a color;
        # the repaints and the captures that follow them are what this
        # sign's dropped presses cost, and the two are learned apart.
        for round_number in range(1, settings.confirm_max_rounds + 1):
            while True:
                epoch = self._pause_generation_value()
                try:
                    self._update_progress_state(
                        PainterState.RUNNING,
                        f"Checking color {group_number} of {total_colors}",
                    )
                    self._move(state.park, epoch)
                    self._interruptible_sleep(
                        self._CONFIRM_SETTLE_SECONDS, epoch=epoch, check_focus=True
                    )
                    self._checkpoint(epoch=epoch, check_focus=True)
                    capture = self._screen_capture(state.canvas)
                    break
                except _RetryAction:
                    continue
            try:
                latest = sample_cell_colors(
                    np.asarray(capture.convert("RGB"), dtype=np.float32),
                    plan.width,
                    plan.height,
                    centers=state.centers,
                )
            except Exception:
                LOGGER.exception("The sign could not be read after color %d", group_number)
                return
            hit, judged = confirm_cells(
                state.reference,
                latest,
                expected,
                judge,
                blend=state.blend,
                # Cells of earlier colors are never crossed by this one's
                # strokes, so any shift common to them is the capture's.
                stable=(state.indices >= 0) & (state.indices < palette_index),
            )
            missed = judged & ~hit
            judged_count = int(judged.sum())
            missed_count = int(missed.sum())
            if round_number == 1:
                self._check_capture_clock += max(
                    0.0, self._active_elapsed() - checking_started
                )
                state.colors += 1
                state.judged += judged_count
                state.missed += missed_count
                rate = missed_count / judged_count if judged_count else 0.0
                if missed_count:
                    LOGGER.info(
                        "Color %d of %d: %d of %d cells did not take (%.1f%%); repainting",
                        group_number,
                        total_colors,
                        missed_count,
                        judged_count,
                        rate * 100.0,
                    )
            if missed_count == 0:
                break
            if missed_before is not None and missed_count >= missed_before:
                # Repainting gained nothing: a burst of dropped presses, or
                # cells the capture cannot settle.  Once is bad luck; twice
                # running and the rest is left to the touch-up pass rather
                # than spent here.
                stalled += 1
                if stalled >= 2:
                    LOGGER.warning(
                        "Color %d of %d: %d cells still missing after repainting "
                        "twice without gain; leaving them to the touch-up pass",
                        group_number,
                        total_colors,
                        missed_count,
                    )
                    break
            else:
                stalled = 0
            missed_before = missed_count
            strokes = repaint_runs(
                missed,
                state.indices,
                palette_index,
                strokes=group.strokes,
                radius=radius,
            )
            self._update_progress_state(
                PainterState.RUNNING,
                f"Repainting {missed_count:,} cells of color {group_number} "
                f"(round {round_number})",
            )
            state.rounds += 1
            for stroke in strokes:
                paint_stroke(stroke)
                state.repainted_strokes += 1
                self._interruptible_sleep(
                    self._stroke_gap(job.settings.delay_between_strokes_seconds),
                    check_focus=True,
                )
            self._publish_confirmation(state)
        # The last round's repaint is not captured again; what it did not
        # mend is counted as unrepaired, which errs on the honest side.
        if missed_count:
            state.unrepaired += missed_count
        state.reference = latest
        self._confirmation_seconds += max(0.0, self._active_elapsed() - checking_started)
        self._publish_confirmation(state)

    _CONFIRM_SETTLE_SECONDS = CONFIRM_SETTLE_SECONDS

    # The break's own waits: for the painting UI to close after Save, for the
    # jump to land (and the server to have seen it), and for the UI to be
    # open again after the interact key reopens the sign.
    _AFK_SAVE_SETTLE_SECONDS = 0.5
    _AFK_JUMP_SETTLE_SECONDS = 2.0
    _AFK_REOPEN_SETTLE_SECONDS = 1.0
    _AFK_JUMP_HOLD_SECONDS = 0.1
    _AFK_INTERACT_KEY = "E"

    def _anti_afk_due(self, job: _Job) -> bool:
        settings = job.settings
        return bool(
            settings.anti_afk_enabled
            and job.target.save_button is not None
            and time.monotonic() - self._last_anti_afk_at
            >= settings.anti_afk_interval_seconds
        )

    def _anti_afk_break_if_due(self, job: _Job) -> None:
        """Every interval: save the sign, jump, and open the sign again.

        A server that kicks idle players watches for movement, and a player
        stood at a sign for an hour has made none.  The break leaves the
        painting UI through its Save button (the server keeps the work so
        far), jumps, and presses the interact key to reopen the sign - which
        works because the player is still looking at it: they were when the
        job started, and an idle camera does not turn.  The painter then
        carries on from the stroke it was about to make.

        Nothing touches the mouse between Save and the reopen: with the
        painting UI closed the game owns the cursor, so a click would swing
        the held item and a move would turn the camera.

        The cursor is the game's, not the painter's, from the moment the UI
        closes - and Rust closes it on the press of Save, not the release,
        recentring the cursor while the button is still held.  The mouse
        guard's baseline is therefore dropped before the press, not after the
        click, and stays dropped until the next stroke lays a new one.
        """

        if not self._anti_afk_due(job):
            return
        button = job.target.save_button
        assert button is not None
        epoch = self._pause_generation_value()
        LOGGER.info("Anti-AFK break: saving the sign, jumping, and reopening it with E")
        self._update_progress_state(
            PainterState.RUNNING,
            "Keeping the player awake: saving and jumping",
            phase="anti_afk",
        )
        target = normalized_point(button, 0.5, 0.5)
        self._checkpoint(epoch=epoch, check_focus=True)
        self._move((int(round(target[0])), int(round(target[1]))), epoch)
        self._checkpoint(epoch=epoch, check_focus=True)
        with self._condition:
            self._reset_mouse_movement_baseline()
        # The UI guard would read the break as the UI being lost, which is
        # exactly what the break does on purpose; it is told to look away
        # until the sign is open again, and then to look first thing.
        self._ui_guard_suspended = True
        try:
            self._mouse_down(epoch)
            try:
                self._interruptible_sleep(
                    self._PICKER_CLICK_HOLD_SECONDS, epoch=epoch, check_focus=True
                )
            finally:
                self.input.mouse_up(MouseButton.LEFT)
            self._interruptible_sleep(
                self._AFK_SAVE_SETTLE_SECONDS, epoch=epoch, check_focus=True
            )
            self._checkpoint(epoch=epoch, check_focus=True)
            self.input.press_key("SPACE", hold_seconds=self._AFK_JUMP_HOLD_SECONDS)
            self._interruptible_sleep(
                self._AFK_JUMP_SETTLE_SECONDS, epoch=epoch, check_focus=True
            )
            self._checkpoint(epoch=epoch, check_focus=True)
            self.input.press_key(
                self._AFK_INTERACT_KEY, hold_seconds=self._AFK_JUMP_HOLD_SECONDS
            )
            self._interruptible_sleep(
                self._AFK_REOPEN_SETTLE_SECONDS, epoch=epoch, check_focus=True
            )
            reopened = self._await_painting_ui(job, epoch)
        finally:
            self._ui_guard_suspended = False
        with self._condition:
            self._reset_mouse_movement_baseline()
            self._last_anti_afk_at = time.monotonic()
            # A new epoch: the sign was left and re-entered, so the next
            # stroke re-selects its color and re-applies its brush size.
            self._pause_generation += 1
        if not reopened:
            # The player is awake - they will be the one reopening the sign
            # - so the break counts as taken, and the job waits for them
            # rather than painting strokes into the game world.
            self.pause(self._UI_NOT_REOPENED_REASON)
            self._checkpoint(check_focus=True)
        self._update_progress_state(PainterState.RUNNING, "Painting", phase="paint")

    # How long after the interact key the painting UI may take to be drawn
    # again before the break gives up waiting for it, and how often it looks.
    _AFK_REOPEN_GRACE_SECONDS = 4.0
    _AFK_REOPEN_POLL_SECONDS = 0.5

    def _await_painting_ui(self, job: _Job, epoch: int) -> bool:
        """Wait for the reopened sign's UI to be drawn again after the break.

        True when it is there, or when nothing is watching for it; False
        when the grace period runs out without it.  An unarmed guard - the
        job started with it off and a pause turned it on - is armed here
        from the reopened UI, which is as good a first look as any.
        """

        guard = self._ui_guard
        if guard is None or not self._ui_guard_wanted(job.settings):
            return True
        deadline = time.monotonic() + self._AFK_REOPEN_GRACE_SECONDS
        while True:
            try:
                if guard.armed:
                    present = guard.check(self._screen_capture).present
                else:
                    present = guard.arm(self._screen_capture)
            except Exception:
                LOGGER.warning(
                    "Could not look for the painting UI after the anti-AFK break",
                    exc_info=True,
                )
                return True
            if present:
                self._ui_missing_checks = 0
                self._last_ui_check = time.monotonic()
                return True
            if time.monotonic() >= deadline:
                LOGGER.warning(
                    "The painting UI did not come back within %.0fs of pressing %s",
                    self._AFK_REOPEN_GRACE_SECONDS,
                    self._AFK_INTERACT_KEY,
                )
                guard.disarm()
                return False
            self._interruptible_sleep(
                self._AFK_REOPEN_POLL_SECONDS, epoch=epoch, check_focus=True
            )

    # ------------------------------------------------------------ UI guard

    _UI_NOT_FOUND_REASON = "painting UI not found - open the sign again and resume"
    _UI_NOT_REOPENED_REASON = (
        "the sign did not reopen after the anti-AFK break - open it and resume"
    )
    # The pause reasons that mean the sign itself went away - the ones a
    # "resume from here" record is stamped with, since the strokes up to
    # the pause are on a sign the game had been saving all along.
    UI_LOSS_REASONS: ClassVar[frozenset[str]] = frozenset(
        {_UI_NOT_FOUND_REASON, _UI_NOT_REOPENED_REASON}
    )
    # Looks at the screen this often while painting, and this much sooner
    # again after a look that found nothing: one failed look can be a frame
    # of something drawn over the UI, two half a second apart are the UI gone.
    _UI_GUARD_INTERVAL_SECONDS = 1.0
    _UI_GUARD_RECHECK_SECONDS = 0.5
    _UI_GUARD_MISSING_CHECKS_TO_PAUSE = 2

    def _ui_guard_wanted(self, settings: PainterSettings) -> bool:
        return bool(
            settings.ui_guard_enabled
            and getattr(self.input, "emits_real_input", True)
        )

    def _confirm_painting_ui(self, job: _Job) -> None:
        """Fingerprint the painting UI before the job's first input.

        The fingerprint has to come from the painting UI, and the hue bar is
        what proves it did.  A screen without one - the countdown ran out
        before the sign was opened, say - pauses the job until the user
        opens the sign and resumes, when it is fingerprinted afresh.  That
        is the same mistake the guard exists to catch, caught a stroke
        earlier.  Turning the guard off during the pause lets the job go
        on without it.
        """

        self._ui_guard = None
        if not getattr(self.input, "emits_real_input", True):
            return
        guard = PaintingUiGuard.for_target(job.target)
        if guard is None:
            return
        self._ui_guard = guard
        while self._ui_guard_wanted(job.settings) and not guard.armed:
            if self._arm_ui_guard(guard):
                return
            self.pause(self._UI_NOT_FOUND_REASON)
            # Waits out the pause.  The safety check the wait ends on arms
            # the guard itself when the UI is back, or pauses again.
            self._checkpoint(check_focus=True)

    def _arm_ui_guard(self, guard: PaintingUiGuard) -> bool:
        """Fingerprint the UI now; True when the screen showed it.

        A capture that fails outright switches the guard off for the job:
        a screen that cannot be read is not one the guard can watch, and
        the job should not stall on it.
        """

        try:
            armed = guard.arm(self._screen_capture)
        except Exception:
            LOGGER.warning(
                "Could not fingerprint the painting UI; the UI guard is off "
                "for this job",
                exc_info=True,
            )
            self._ui_guard = None
            return True
        if armed:
            self._ui_missing_checks = 0
            self._last_ui_check = time.monotonic()
            LOGGER.info(
                "UI guard armed on %s",
                ", ".join(region.name for region in guard.regions),
            )
            return True
        LOGGER.warning(
            "The painting UI was not recognised on the screen: the hue bar is "
            "not where it was calibrated"
        )
        return False

    def _check_painting_ui(self, settings: PainterSettings, now: float) -> bool:
        """Pause when the painting UI's widgets have gone from the screen.

        Returns True when the job just paused and the caller must re-evaluate
        its state.  A guard a pause turned on, or one a pause for a missing
        UI disarmed, is armed from the screen here - when the hue bar is on
        it, and with a pause when it is not.
        """

        guard = self._ui_guard
        if (
            guard is None
            or self._ui_guard_suspended
            or not self._ui_guard_wanted(settings)
        ):
            return False
        interval = (
            self._UI_GUARD_RECHECK_SECONDS
            if self._ui_missing_checks
            else self._UI_GUARD_INTERVAL_SECONDS
        )
        if now - self._last_ui_check < interval:
            return False
        self._last_ui_check = now
        if not guard.armed:
            if self._arm_ui_guard(guard):
                return False
            self.pause(self._UI_NOT_FOUND_REASON)
            return True
        try:
            verdict = guard.check(self._screen_capture)
        except Exception:
            LOGGER.warning("Could not look for the painting UI", exc_info=True)
            return False
        if verdict.present:
            self._ui_missing_checks = 0
            return False
        self._ui_missing_checks += 1
        if self._ui_missing_checks < self._UI_GUARD_MISSING_CHECKS_TO_PAUSE:
            return False
        self._ui_missing_checks = 0
        LOGGER.warning(
            "The painting UI is no longer on the screen (%s); pausing",
            verdict.describe(),
        )
        # Fingerprinted again once the user has the sign open and resumes.
        guard.disarm()
        self.pause(self._UI_NOT_FOUND_REASON)
        return True

    # An interim audit is worth its export only after this many fresh
    # strokes: fewer can have missed at most a few cells, and the final
    # verification covers them.
    _INTERIM_AUDIT_MIN_STROKES = 500

    @staticmethod
    def _final_group_map(plan: PaintPlan) -> Any:
        """Which color group, by painting order, gives each cell its final color.

        The same replay :func:`plan_layers` runs, keeping the group ordinal
        instead of the palette row: a cell is only judgeable mid-paint once
        the group that finishes it has been painted.  ``-1`` where no stroke
        reaches.
        """

        import numpy as np

        final = np.full((plan.height, plan.width), -1, dtype=np.int32)
        for number, group in enumerate(plan.color_groups):
            radius = (max(1, group.brush_diameter) - 1) // 2
            for stroke in group.strokes:
                x0 = min(stroke.start_x, stroke.end_x)
                x1 = max(stroke.start_x, stroke.end_x)
                y0 = min(stroke.start_y, stroke.end_y)
                y1 = max(stroke.start_y, stroke.end_y)
                final[
                    max(0, y0 - radius) : min(plan.height, y1 + radius + 1),
                    max(0, x0) : min(plan.width, x1 + 1),
                ] = number
        return final

    def _interim_audit(self, job: _Job, groups_done: int, layers: Any) -> bool:
        """Read the export mid-paint and refill the holes in what is final.

        Only cells whose final color group is already painted are judged,
        and only holes are refilled - a hole is a dab the game dropped, the
        one mistake that compounds silently for hours.  Wrong-color
        verdicts wait for the end, where the recolor gate decides whether
        touching them helps at all.  Runs between color groups, so the
        repaint's own color and Size changes cannot desynchronise a group
        mid-paint; the caller re-applies both after a repaint.  A pause
        during the audit abandons it - the next due audit sees the same
        holes - but still reports the brush state unknown.

        Returns whether the loop's applied color and Size may have changed.
        """

        import numpy as np

        from .verification import touch_up_plan

        indices, palette, final_group = layers
        started = self._active_elapsed()
        try:
            self._update_progress_state(
                PainterState.RUNNING,
                f"Auditing the painting so far ({groups_done} colors down)",
                phase="audit",
            )
            epoch = self._pause_generation_value()
            export = self._export_sign(
                job, epoch, why=f"interim audit after {groups_done} colors"
            )
            if export is None:
                return False
            judgeable = (final_group >= 0) & (final_group < groups_done)
            miss = judgeable & ~np.asarray(export.painted, dtype=np.bool_)
            count = int(miss.sum())
            if count == 0:
                LOGGER.info(
                    "Interim audit after %d colors: everything painted so far "
                    "is on the sign",
                    groups_done,
                )
                return False
            repaint = touch_up_plan(miss, indices, palette)
            LOGGER.info(
                "Interim audit after %d colors: refilling %d missed cells in "
                "%d strokes while the paint is fresh",
                groups_done,
                count,
                repaint.stroke_count,
            )
            self._update_progress_state(
                PainterState.RUNNING,
                f"Refilling {count:,} missed cells the interim audit found",
                phase="audit",
            )
            self._execute_plan(job, plan=repaint, reference=export.rgb)
            return True
        except _RetryAction:
            LOGGER.info(
                "The interim audit was interrupted by a pause; painting goes "
                "on and the next due audit will see the same holes"
            )
            return True
        finally:
            # Audit time is checking, not stroking: kept out of the learned
            # per-stroke overhead the way the color checks are.
            self._confirmation_seconds += max(0.0, self._active_elapsed() - started)
            self._update_progress_state(PainterState.RUNNING, "Painting", phase="paint")

    def _verify_and_touch_up(self, job: _Job) -> None:
        """Read the sign back and repaint the cells that missed their color.

        The comparison is relative - a cell is wrong only when its captured
        color sits decisively closer to a *different* plan color than to its
        own - so lighting and the sign's material shift, which move every
        color together, never trigger a repaint.  Each pass captures, decides,
        and repaints; a repaint that is mostly lone dabs opens with probing
        batches (:meth:`_touch_up_probe_batches`) so a stamp that cannot land
        is corrected within a batch rather than after the whole pass.  Every
        repaint is audited, including the final allowed
        one; a clean capture or an implausible one ends the loop. With a Rust
        export, covered texels that remain untouched after the repair budget
        are an error. Plan-uncovered (for example transparent) texels are never
        part of that verdict.
        """

        import numpy as np

        from .verification import (
            RECOLOR_MIN_CELL_PIXELS,
            UNRELIABLE_CAPTURE_FRACTION,
            classify_cells,
            classify_export,
            plan_layers,
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
        indices, underpaint, palette = plan_layers(plan)
        covered = int((indices >= 0).sum())
        if covered == 0:
            return
        canvas = ScreenRect(
            target.canvas.left,
            target.canvas.top,
            target.canvas.width,
            target.canvas.height,
        )
        # Recoloring a single cell needs a brush that fits the cell and a
        # capture that can tell the cell from its neighbours.  A plan finer
        # than either still gets its holes filled - a hole is a stroke's
        # worth of bare sign, which both can see - but a cell read as the
        # wrong color there is as likely a neighbour's paint as a mistake,
        # and "correcting" it with a brush wider than the cell would smear
        # the neighbours it was read from.
        cell_pixels = min(canvas.width / plan.width, canvas.height / plan.height)
        overshoot = self._detail_brush_overshoot(job)
        exact = job.texel_grid is not None and job.texel_grid.swept and is_native(
            plan, job.texel_grid.columns, job.texel_grid.rows
        )
        recolor = exact or (
            cell_pixels >= RECOLOR_MIN_CELL_PIXELS
            and overshoot <= self._DETAIL_OVERSHOOT_LIMIT
        )
        if not recolor:
            LOGGER.info(
                "Verification will fill holes only: cells are %.2f px across and "
                "the smallest brush covers %.1f cells, too fine to recolor "
                "single cells without smearing their neighbours",
                cell_pixels,
                overshoot,
            )
        bare_sampled = None
        if job.bare_canvas is None and target.bare_color is not None:
            # This job never saw the sign cleared - it resumed onto an
            # existing painting, or is touching one up as it stands - so the
            # wood comes from what an earlier job on this sign measured.  A
            # single colour is all the classifier takes from a bare capture
            # (it medians it), and the capture's lighting is normalized
            # before the comparison, so a colour read under other light
            # still places the wood.
            bare_sampled = np.full(
                (plan.height, plan.width, 3), np.array(target.bare_color, dtype=np.float32)
            )
            LOGGER.info(
                "Reading holes against the bare sign colour #%02X%02X%02X stored "
                "for this profile; this job did not see the sign cleared",
                *target.bare_color,
            )
        if job.bare_canvas is not None:
            try:
                bare_sampled = sample_cell_colors(
                    np.asarray(job.bare_canvas.convert("RGB"), dtype=np.float32),
                    plan.width,
                    plan.height,
                    centers=self._grid_cell_centers(job, plan, canvas),
                )
            except Exception:
                LOGGER.exception("The bare-sign capture could not be sampled")
        # Parked over the color box, the cursor cannot shadow the capture.
        park = self._park_point(target)
        audit_number = 1
        repairs = 0
        touch_up_started = self._active_elapsed()
        # The cells the previous pass repainted: a cell still wrong after a
        # stationary press at its audited aim is one this stamp cannot
        # reach, and the next pass reaches for it with a bigger one.
        previous_repaint: Any = None

        def record_touch_up() -> None:
            with self._condition:
                self._touch_up_timing = TouchUpTiming(
                    seconds=max(0.0, self._active_elapsed() - touch_up_started),
                    passes=audit_number,
                )

        native = self._aiming(job, plan).native
        previous_targets: Any = None
        previous_wrong: int | None = None

        def read_sign(why: str) -> tuple[Any, Any, SignExport | None]:
            """One reading of the sign: the exact export, or the screen."""

            epoch = self._pause_generation_value()
            export = self._export_sign(job, epoch, why=why)
            if export is not None:
                # Texel-exact: what was never painted, and what is the
                # wrong colour, straight from the game's own texture.  The
                # recolor gate applies here as on the screen path: a
                # one-cell brush wider than a cell must not "correct"
                # wrong-colour cells, it would smear the neighbours.
                return (
                    classify_export(
                        export.rgb, export.painted, indices, palette, recolor=recolor
                    ),
                    export.rgb,
                    export,
                )
            self._move(park, epoch)
            self._interruptible_sleep(0.35, epoch=epoch, check_focus=True)
            self._checkpoint(epoch=epoch, check_focus=True)
            capture = self._screen_capture(canvas)
            sampled = sample_cell_colors(
                np.asarray(capture.convert("RGB"), dtype=np.float32),
                plan.width,
                plan.height,
                centers=self._grid_cell_centers(job, plan, canvas),
            )
            verdict = classify_cells(
                sampled,
                indices,
                palette,
                bare_sampled=bare_sampled,
                underpaint=underpaint,
                recolor=recolor,
            )
            return verdict, sampled, None

        while True:
            try:
                self._update_progress_state(
                    PainterState.RUNNING,
                    f"Verifying the painted sign (audit {audit_number})",
                    phase="verify",
                )
                verdict, sampled, export = read_sign(
                    f"verification audit {audit_number}"
                )
                if export is not None and previous_targets is not None:
                    self._learn_cell_nudges(
                        export,
                        indices,
                        palette,
                        previous_targets,
                        verdict.cells,
                        only_displaced=exact,
                    )
                mismatch = verdict.cells
                wrong = verdict.count
                LOGGER.info(
                    "Verification audit %d read %d blank, %d unexplained and %d "
                    "wrong-color cells of %d%s",
                    audit_number,
                    verdict.blank,
                    verdict.unexplained,
                    verdict.wrong_color,
                    covered,
                    "" if recolor else " (wrong-color cells are left alone)",
                )
                if verdict.discarded:
                    LOGGER.warning(
                        "Verification audit %d also read %d cells as the wrong "
                        "color, scattered through colors that are otherwise "
                        "right; the capture is not resolving cells at this "
                        "size, so they are left alone and only holes are filled",
                        audit_number,
                        verdict.discarded,
                    )
                if wrong == 0:
                    if verdict.wrong_color:
                        message = "Verified: no holes left on the sign"
                    else:
                        message = "Verified: the sign matches the plan"
                    LOGGER.info("Verification audit %d: %s", audit_number, message)
                    self._update_progress_state(PainterState.RUNNING, message)
                    record_touch_up()
                    return
                allowed_passes = settings.verify_passes
                if exact:
                    # A refill of exact one-texel presses can only leave a
                    # hole when the game dropped a press; give it the room
                    # to refill those too, as long as each pass gains.
                    allowed_passes = max(settings.verify_passes, self._EXACT_REPAIR_PASSES)
                    if (
                        repairs >= settings.verify_passes
                        and previous_wrong is not None
                        and wrong >= previous_wrong
                    ):
                        allowed_passes = repairs  # the extra passes must gain
                previous_wrong = wrong
                if repairs >= allowed_passes:
                    record_touch_up()
                    if export is not None and verdict.blank:
                        noun = "texel" if verdict.blank == 1 else "texels"
                        raise RuntimeError(
                            f"Final verification found {verdict.blank:,} required "
                            f"{noun} still untouched after "
                            f"{repairs} repair{'s' if repairs != 1 else ''}. "
                            "Transparent and intentionally unpainted texels were ignored."
                        )
                    if export is not None:
                        LOGGER.warning(
                            "Final export has no untouched required texels, but %d "
                            "wrong-color cells remain after %d repairs",
                            wrong,
                            repairs,
                        )
                        return
                    LOGGER.warning(
                        "Final verification still found %d cells wrong after %d "
                        "repairs; no exact untouched-texel verdict was available",
                        wrong,
                        repairs,
                    )
                    return
                if export is None and wrong > covered * UNRELIABLE_CAPTURE_FRACTION:
                    LOGGER.warning(
                        "Verification read %d of %d cells as wrong; the capture "
                        "looks unreliable (occluded sign, open menu, moved view), "
                        "so no touch-up will be painted from it",
                        wrong,
                        covered,
                    )
                    return
                if previous_repaint is not None and not exact and (
                    not native or export is None or audit_number >= 3
                ):
                    # A bigger brush covers a logical cell of several texels
                    # that the smallest missed.  On a native plan a cell is
                    # one texel and Size 1.25 paints two to four of them
                    # (measured on the sign's export), so an exact export
                    # first answers a native miss by re-aiming - but only
                    # for one pass: a cell still missed at its corrected aim
                    # is one the stamp cannot reach (a sign measured live
                    # spent hours of re-aimed Size-1 passes landing 3% of
                    # such cells), and from the third audit the wider stamp
                    # is allowed; the audit repaints any neighbour it
                    # discolours.  Without an export, repeated
                    # screen-visible holes need the wider stamp at once.
                    self._escalate_touch_up_brush(
                        job, audit_number, mismatch, previous_repaint
                    )
                previous_repaint = mismatch.copy()
                previous_targets = mismatch.copy()
                if not exact:
                    # The batches, re-aims and Size ladder exist for stamps
                    # that cannot be trusted to land; a swept map with the
                    # one-texel brush lands them, so the pass paints whole.
                    mismatch, sampled = self._touch_up_probe_batches(
                        job,
                        mismatch,
                        indices,
                        palette,
                        sampled,
                        covered=covered,
                        native=native,
                        audit_number=audit_number,
                        read_sign=read_sign,
                    )
                wrong = int(mismatch.sum())
                if wrong == 0:
                    # The batches alone repaired everything they saw; the
                    # next audit confirms the sign whole and ends the loop.
                    repairs += 1
                    audit_number += 1
                    continue
                repaint = touch_up_plan(mismatch, indices, palette)
                predicted = self._work_schedule(repaint, target, settings).total
                LOGGER.info(
                    "Verification audit %d: repainting %d of %d cells in %d "
                    "strokes, about %s",
                    audit_number,
                    wrong,
                    covered,
                    repaint.stroke_count,
                    _describe_seconds(predicted),
                )
                self._update_progress_state(
                    PainterState.RUNNING,
                    f"Touching up {wrong:,} cells (repair {repairs + 1}, "
                    f"about {_describe_seconds(predicted)})",
                    phase="verify",
                )
                # The capture the touch-up was planned from is the reading
                # of the sign before it, so its colors are checked too.
                self._execute_plan(job, plan=repaint, reference=sampled)
                repairs += 1
                audit_number += 1
            except _RetryAction:
                # A pause released the mouse mid-pass; redo this pass whole,
                # from a fresh capture, once painting resumes.
                continue

    # How many rounds of picker clicks a color gets before the panel is
    # looked for again, and how many after that before the job pauses.
    # A pass whose repaints mostly did not take is using a stamp that cannot
    # reach those cells: at least this many, and this share, of the cells
    # repainted last pass still wrong raises the one-cell brush a step.
    _TOUCH_UP_STUBBORN_MIN = 10
    _TOUCH_UP_STUBBORN_FRACTION = 0.25
    # Repair passes a texel-exact job may take while each pass still gains.
    _EXACT_REPAIR_PASSES = 8

    # A big touch-up is tested before it is trusted: batches of this many of
    # its lone dabs go down first, each read back, so a stamp that cannot
    # land is caught within a batch instead of after the whole pass.  Below
    # the minimum the pass itself is cheaper than the extra readings.
    _TOUCH_UP_PROBE_BATCH_DABS = 96
    _TOUCH_UP_PROBE_MIN_DABS = 200

    def _touch_up_probe_batches(
        self,
        job: _Job,
        mismatch: Any,
        indices: Any,
        palette: Any,
        sampled: Any,
        *,
        covered: int,
        native: bool,
        audit_number: int,
        read_sign: Any,
    ) -> tuple[Any, Any]:
        """Paint a large touch-up's opening dabs in probing batches.

        A touch-up pass used to paint every missing cell and only then read
        the sign back, so a one-cell stamp that cannot land - a resumed sign
        whose Size was never probed - cost a whole pass to discover, and at
        the flat frame-hold every dab pays such a pass runs to hours.  When
        the repaint is mostly lone dabs, a spread batch of them goes down
        first and is read back: a batch that mostly landed lets the rest of
        the pass proceed as it is; one that mostly missed raises the
        one-cell brush a ladder step (the reaction a whole failed pass used
        to buy) and tests another batch at the new Size.  On a native plan
        read from an exact export a miss is first treated as an aiming
        error: the batch's misses are re-aimed from the export and pressed
        once more.  But only once - a sign was measured live spending hours
        of re-aimed Size-1 passes to land 3% of its leftover holes, the
        cells no whole-pixel aim reaches - so re-aimed dabs that still miss
        escalate the Size like any others.  The wider stamp can touch a
        neighbour, but the neighbours are ordinary reachable cells and the
        audit that follows repaints any it discolours.

        Returns the mismatch and the reference reading as the batches leave
        them, for the rest of the pass to paint from.
        """

        import numpy as np

        from .verification import UNRELIABLE_CAPTURE_FRACTION, touch_up_plan

        settings = job.settings
        if not settings.measure_dab_size or not settings.apply_brush_size:
            return mismatch, sampled
        lone = [
            (int(stroke.start_x), int(stroke.start_y))
            for group in touch_up_plan(mismatch, indices, palette).color_groups
            for stroke in group.strokes
            if int(stroke.pixel_count) <= 1
        ]
        if len(lone) < self._TOUCH_UP_PROBE_MIN_DABS:
            return mismatch, sampled
        # The cells arrive in row-major order; striding through them spreads
        # every batch across the sign, so one odd region cannot pass or fail
        # a stamp the rest of the sign would judge the other way.
        stride = max(1, len(lone) // self._TOUCH_UP_PROBE_BATCH_DABS)
        ordered = [cell for start in range(stride) for cell in lone[start::stride]]
        batch_number = 0
        offset = 0
        aim_retried = False
        while offset < len(ordered):
            batch = ordered[offset : offset + self._TOUCH_UP_PROBE_BATCH_DABS]
            offset += len(batch)
            if len(batch) < max(12, self._TOUCH_UP_PROBE_BATCH_DABS // 4):
                break
            batch_number += 1
            batch_mask = np.zeros_like(mismatch)
            batch_mask[
                np.array([cell[1] for cell in batch]),
                np.array([cell[0] for cell in batch]),
            ] = True
            self._update_progress_state(
                PainterState.RUNNING,
                f"Testing the touch-up stamp on {len(batch)} cells "
                f"(audit {audit_number}, batch {batch_number})",
                phase="verify",
            )
            self._execute_plan(
                job,
                plan=touch_up_plan(batch_mask, indices, palette),
                reference=sampled,
            )
            verdict, read, export = read_sign(
                f"touch-up batch {batch_number} of audit {audit_number}"
            )
            if export is None and verdict.count > covered * UNRELIABLE_CAPTURE_FRACTION:
                LOGGER.warning(
                    "The reading after touch-up batch %d looks unreliable (%d of "
                    "%d cells wrong); the rest of the pass is painted without "
                    "batching",
                    batch_number,
                    verdict.count,
                    covered,
                )
                return mismatch & ~batch_mask, sampled
            mismatch, sampled = verdict.cells, read
            stubborn = int((mismatch & batch_mask).sum())
            if export is not None:
                self._learn_cell_nudges(export, indices, palette, batch_mask, mismatch)
            if stubborn < max(
                6, math.ceil(self._TOUCH_UP_STUBBORN_FRACTION * len(batch))
            ):
                LOGGER.info(
                    "Touch-up batch %d: %d of %d dabs landed; the stamp works "
                    "and the rest of the pass proceeds",
                    batch_number,
                    len(batch) - stubborn,
                    len(batch),
                )
                break
            if native and export is not None and not aim_retried:
                # One chance for the re-aims just learned: press the same
                # still-wrong cells again at their corrected aims.  Mostly
                # landing proves the misses were aim and the pass can trust
                # Size as it is; mostly missing again proves the stamp
                # itself cannot reach these cells and the ladder is next.
                aim_retried = True
                retry_mask = mismatch & batch_mask
                retried = int(retry_mask.sum())
                if retried == 0:
                    break
                LOGGER.info(
                    "Touch-up batch %d: %d of %d dabs missed on the native "
                    "plan; pressing the %d still-wrong cells once more at "
                    "their corrected aims",
                    batch_number,
                    stubborn,
                    len(batch),
                    retried,
                )
                self._update_progress_state(
                    PainterState.RUNNING,
                    f"Retrying {retried} re-aimed cells "
                    f"(audit {audit_number}, batch {batch_number})",
                    phase="verify",
                )
                self._execute_plan(
                    job,
                    plan=touch_up_plan(retry_mask, indices, palette),
                    reference=sampled,
                )
                verdict, read, export = read_sign(
                    f"re-aimed retry of touch-up batch {batch_number}"
                )
                mismatch, sampled = verdict.cells, read
                still = int((mismatch & retry_mask).sum())
                if export is not None:
                    self._learn_cell_nudges(
                        export, indices, palette, retry_mask, mismatch
                    )
                if still < max(
                    6, math.ceil(self._TOUCH_UP_STUBBORN_FRACTION * retried)
                ):
                    LOGGER.info(
                        "Touch-up batch %d: %d of %d re-aimed dabs landed; "
                        "the misses were aim, not Size, and the pass proceeds",
                        batch_number,
                        retried - still,
                        retried,
                    )
                    break
                LOGGER.info(
                    "Touch-up batch %d: %d of %d dabs still miss at their "
                    "corrected aims; the stamp cannot reach these cells at "
                    "this Size",
                    batch_number,
                    still,
                    retried,
                )
            current, adopted = self._raise_one_cell_brush()
            if adopted is None:
                LOGGER.info(
                    "Touch-up batch %d: %d of %d dabs missed, but the one-cell "
                    "brush is already at Size %s, the largest the painter will "
                    "try",
                    batch_number,
                    stubborn,
                    len(batch),
                    format_brush_size(current),
                )
                break
            LOGGER.info(
                "Touch-up batch %d: %d of %d dabs missed at Size %s; raising "
                "the one-cell brush to Size %s and testing another batch",
                batch_number,
                stubborn,
                len(batch),
                format_brush_size(current),
                format_brush_size(adopted),
            )
        return mismatch, sampled

    # A dab that missed its cell is re-aimed this many pixels the other way
    # on the next pass; a miss the export cannot place is searched upward
    # first (the direction every miss took in the live measurements), then
    # the other ways.
    _CELL_NUDGE_STEPS_PIXELS = (1, 2, 3)
    _CELL_NUDGE_SEARCH = ((0, -1), (0, 1), (-1, 0), (1, 0))

    def _learn_cell_nudges(
        self,
        export: SignExport,
        indices: Any,
        palette: Any,
        targets: Any,
        still_wrong: Any,
        *,
        only_displaced: bool = False,
    ) -> None:
        """Re-aim the cells the last pass repainted and the export says it missed.

        Where a dab lands is repeatable (measured live: the same aim lands
        the same texel every time), so a cell that came out bare while a
        neighbour took its colour was aimed a pixel or so towards that
        neighbour - and is aimed that much the other way next pass.  A miss
        the export cannot place is stepped through the four directions.
        """

        import numpy as np

        rows, cols = indices.shape
        missed = np.argwhere(targets & still_wrong)
        if len(missed) == 0:
            return
        learned = 0
        for v, u in missed:
            own = int(indices[v, u])
            if own < 0:
                continue
            own_color = np.asarray(palette[own], dtype=np.float32)
            found: tuple[int, int] | None = None
            for du, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nu, nv = u + du, v + dv
                if not (0 <= nu < cols and 0 <= nv < rows):
                    continue
                if indices[nv, nu] == own:
                    continue  # its own colour belongs there anyway
                if export.painted[nv, nu] and float(
                    np.linalg.norm(export.rgb[nv, nu] - own_color)
                ) < 40.0:
                    found = (du, dv)
                    break
            key = (int(u), int(v))
            current = self._cell_nudges.get(key, (0, 0))
            if found is None and only_displaced:
                # On a swept map a bare cell with no displaced neighbour is
                # a press the game dropped, not an aim to correct: the same
                # pixel is pressed again.
                continue
            if found is not None:
                # Landed one texel over: aim a step the other way, growing
                # by a pixel each time the same miss repeats.
                step = 1
                for candidate in self._CELL_NUDGE_STEPS_PIXELS:
                    if abs(current[0]) >= candidate or abs(current[1]) >= candidate:
                        step = candidate + 1
                nudge = (-found[0] * step, -found[1] * step)
            else:
                tried = sum(1 for _ in [current]) if current != (0, 0) else 0
                order = self._CELL_NUDGE_SEARCH
                position = 0
                if current != (0, 0):
                    for i, (dx, dy) in enumerate(order):
                        if (dx, dy) == (int(np.sign(current[0])), int(np.sign(current[1]))):
                            position = i + 1
                            break
                dx, dy = order[position % len(order)]
                nudge = (dx, dy)
            self._cell_nudges[key] = (int(nudge[0]), int(nudge[1]))
            learned += 1
        LOGGER.info(
            "Re-aiming %d cells the last pass missed, from where the export shows "
            "their dabs landed",
            learned,
        )

    def _escalate_touch_up_brush(
        self, job: _Job, pass_number: int, mismatch: Any, previous: Any
    ) -> None:
        """Raise the one-cell brush when last pass's repaints did not take.

        A touch-up dab is a stationary press at the cell's audited aim.  A
        cell that is still bare after one is a cell this stamp cannot reach
        from any whole pixel the cursor can stand on - the lone-dab holes a
        finished XXL sign shows - and repeating the same press would only
        repeat the miss.  The next Size up (the same ladder the dab probe
        climbs) is typed for this pass's single-cell strokes instead.  On a
        sign that was cleared and probed this rarely triggers, since the
        probe already found the Size that lands; on a sign resumed or
        touched up as it is, where nothing may be stamped to probe, it is
        how the touch-up still gets the holes filled.
        """

        settings = job.settings
        if not settings.measure_dab_size or not settings.apply_brush_size:
            return
        repainted = int(previous.sum())
        stubborn = int((mismatch & previous).sum())
        if stubborn < max(
            self._TOUCH_UP_STUBBORN_MIN, self._TOUCH_UP_STUBBORN_FRACTION * repainted
        ):
            return
        current, adopted = self._raise_one_cell_brush()
        if adopted is None:
            LOGGER.info(
                "Verification pass %d: %d of the %d cells repainted last pass are "
                "still wrong, and the one-cell brush is already at Size %s, the "
                "largest the painter will try",
                pass_number,
                stubborn,
                repainted,
                format_brush_size(current),
            )
            return
        LOGGER.info(
            "Verification pass %d: %d of the %d cells repainted last pass are "
            "still wrong; raising the one-cell brush from Size %s to %s for "
            "this pass",
            pass_number,
            stubborn,
            repainted,
            format_brush_size(current),
            format_brush_size(adopted),
        )

    def _raise_one_cell_brush(self) -> tuple[float, float | None]:
        """Raise the one-cell brush one ladder step; the old Size and the new.

        The ladder is the dab probe's (:data:`DAB_PROBE_SIZES`).  Returns
        ``(current, adopted)``; ``adopted`` is None, and nothing changes,
        when the brush already stands at the top.
        """

        current = self._measured_detail_size or BRUSH_SIZE_MIN
        larger = [size for size in self._DAB_PROBE_SIZES if size > current + 1e-9]
        if not larger:
            return current, None
        self._measured_detail_size = float(larger[0])
        return current, float(larger[0])

    _PICK_ATTEMPTS = 3
    _PICK_ATTEMPTS_AFTER_RELOCATE = 2
    # At a small Rust UI scale a click can occasionally arrive while the
    # picker is still redrawing.  The normal rounds below are intentionally
    # quick, but do not make a user resume a job just to repeat the exact
    # same safe operation.  These are full, delayed retries before the
    # fail-safe pause; no paint stroke can happen until the swatch agrees.
    _PICK_RECOVERY_ROUNDS = 2
    _PICK_RECOVERY_DELAY_SECONDS = 0.75
    # When the panel does not yet show the color, it is read once more after
    # this long: the frame carrying the click may not have been presented.
    _SWATCH_RECHECK_SECONDS = 0.15

    def _select_color(
        self,
        color: RGBColor,
        target: PaintingTarget,
        settings: PainterSettings,
        epoch: int,
        *,
        apply_correction: bool = True,
    ) -> None:
        """Pick ``color`` on the panel, and make sure the panel agrees.

        Two clicks select a color, one on the hue bar and one on the
        saturation / value box, and a click the game swallows leaves the
        previous color selected - the whole group then goes down in it.  So
        when the panel's selected-color block was found at the start of the
        job, it is read after the clicks and they are repeated, held
        longer, until it shows the color.  A color the panel will not show
        after repeated rounds pauses the job for the user to look.
        """

        picker_color = (
            target.color_correction.correct(color)
            if apply_correction and target.color_correction is not None
            else color
        )
        if self._swatch is None:
            # No read-back to catch a swallowed edge click, so stay a couple
            # of pixels inside the widgets rather than on their exact edge.
            hue_point, sv_point, _expected = self._picker_plan(
                picker_color, target, self._PICKER_BLIND_MARGIN_PIXELS
            )
            self._click_picker(hue_point, sv_point, settings, epoch)
            return
        reading = self._pick_until_shown(
            picker_color, target, settings, epoch, self._PICK_ATTEMPTS
        )
        if reading is None or self._swatch is None:
            return
        # The block may have moved, or be covered: look for it again, then
        # give the color a couple more rounds.
        LOGGER.warning(
            "The panel still shows %s after selecting #%02X%02X%02X; looking "
            "for its color block again",
            reading.hex,
            *picker_color,
        )
        if not self._find_swatch(target, settings, epoch):
            self._stop_reading_picks(
                "the color block was lost at color %d" % (self._color_pick_summary.picks + 1)
            )
            hue_point, sv_point, _expected = self._picker_plan(
                picker_color, target, self._PICKER_BLIND_MARGIN_PIXELS
            )
            self._click_picker(hue_point, sv_point, settings, epoch)
            return
        reading = self._pick_until_shown(
            picker_color,
            target,
            settings,
            epoch,
            self._PICK_ATTEMPTS_AFTER_RELOCATE,
            first_attempt=self._PICK_ATTEMPTS + 1,
        )
        if reading is None or self._swatch is None:
            return
        reading = self._rescue_nearby_picker_hue(
            picker_color, target, settings, epoch
        )
        if reading is None or self._swatch is None:
            return
        reading = self._correct_picker_geometry_from_swatch(
            picker_color, target, settings, epoch, reading
        )
        if reading is None or self._swatch is None:
            return
        reading = self._recover_delayed_color_pick(
            picker_color, target, settings, epoch, reading
        )
        if reading is None or self._swatch is None:
            return
        with self._condition:
            self._color_pick_summary = replace(
                self._color_pick_summary, failed=self._color_pick_summary.failed + 1
            )
        LOGGER.warning(
            "The color picker did not take #%02X%02X%02X after %d rounds of "
            "clicks (the panel shows %s); pausing",
            *picker_color,
            self._PICK_ATTEMPTS + self._PICK_ATTEMPTS_AFTER_RELOCATE,
            reading.hex,
        )
        self.pause(
            "the color picker did not take #%02X%02X%02X (the panel shows %s) - "
            "check the sign's color panel and resume" % (*picker_color, reading.hex)
        )
        # Waits out the pause; resuming starts a new epoch, and the caller
        # picks the color again under it.
        self._checkpoint(epoch=epoch, check_focus=True)

    def _correct_picker_geometry_from_swatch(
        self,
        picker_color: RGBColor,
        target: PaintingTarget,
        settings: PainterSettings,
        epoch: int,
        reading: SwatchReading,
    ) -> SwatchReading | None:
        """Correct small picker-rectangle offsets from its displayed colour.

        At 0.5 UI scale, a calibrated widget can be a few physical pixels
        away from Rust's live raster.  The displayed swatch gives the actual
        hue, saturation, and value, so it is enough to move each click by the
        corresponding pixel error instead of asking the player to type an
        exact colour into Rust's unreliable text field.
        """

        hue_point, sv_point, _expected = self._picker_plan(picker_color, target, 0.0)
        wanted = rgb_to_hsv(picker_color)
        shown = rgb_to_hsv(reading.color)
        hue_delta = (wanted.hue - shown.hue + 180.0) % 360.0 - 180.0
        hue_sign = -1.0 if target.picker_directions.hue == "bottom_to_top" else 1.0
        saturation_sign = (
            -1.0 if target.picker_directions.saturation == "left_high" else 1.0
        )
        value_sign = -1.0 if target.picker_directions.value == "top_bright" else 1.0
        corrected_hue = clamp_to_rect(
            hue_point[0],
            hue_point[1] + hue_sign * hue_delta * (target.hue_bar.height - 1) / 360.0,
            target.hue_bar,
        )
        corrected_sv = clamp_to_rect(
            sv_point[0]
            + saturation_sign
            * (wanted.saturation - shown.saturation)
            * (target.color_box.width - 1),
            sv_point[1]
            + value_sign * (wanted.value - shown.value) * (target.color_box.height - 1),
            target.color_box,
        )
        corrected_hue = int(round(corrected_hue[0])), int(round(corrected_hue[1]))
        corrected_sv = int(round(corrected_sv[0])), int(round(corrected_sv[1]))
        if corrected_hue == hue_point and corrected_sv == sv_point:
            return reading
        LOGGER.warning(
            "The picker raster appears offset; correcting its hue and S/V clicks "
            "from %s toward #%02X%02X%02X",
            reading.hex,
            *picker_color,
        )
        self._click_picker(corrected_hue, corrected_sv, settings, epoch, retry=True)
        if self._swatch is None:
            return None
        corrected_reading = self._read_selected_color(picker_color, epoch)
        if corrected_reading is None:
            return None
        if corrected_reading.matches(picker_color):
            with self._condition:
                summary = self._color_pick_summary
                self._color_pick_summary = replace(
                    summary, picks=summary.picks + 1, retried=summary.retried + 1
                )
            LOGGER.info("Color #%02X%02X%02X took after picker-raster correction", *picker_color)
            return None
        return corrected_reading

    def _recover_delayed_color_pick(
        self,
        picker_color: RGBColor,
        target: PaintingTarget,
        settings: PainterSettings,
        epoch: int,
        reading: SwatchReading,
    ) -> SwatchReading | None:
        """Retry a verified pick after Rust has had a full redraw interval.

        This covers the intermittent small-scale case where every immediate
        click is processed against an old picker frame.  Keep the existing
        swatch rather than locating it again: a failed locator pick would
        itself be another transient UI failure and must not disable the
        safety check.
        """

        for round_number in range(1, self._PICK_RECOVERY_ROUNDS + 1):
            LOGGER.warning(
                "The panel still shows %s after selecting #%02X%02X%02X; "
                "waiting for the picker and retrying (%d of %d)",
                reading.hex,
                *picker_color,
                round_number,
                self._PICK_RECOVERY_ROUNDS,
            )
            self._interruptible_sleep(
                self._PICK_RECOVERY_DELAY_SECONDS, epoch=epoch, check_focus=True
            )
            reading = self._pick_until_shown(
                picker_color, target, settings, epoch, self._PICK_ATTEMPTS
            )
            if reading is None or self._swatch is None:
                return None
            reading = self._rescue_nearby_picker_hue(
                picker_color, target, settings, epoch
            )
            if reading is None or self._swatch is None:
                return None
        return reading

    def _picker_plan(
        self, picker_color: RGBColor, target: PaintingTarget, margin_pixels: float
    ) -> tuple[tuple[int, int], tuple[int, int], RGBColor]:
        """Click points and the color they select, at one edge margin."""

        directions = target.picker_directions
        return picker_click_plan(
            picker_color,
            target.hue_bar,
            target.color_box,
            hue_direction=directions.hue,
            saturation_direction=directions.saturation,
            value_direction=directions.value,
            margin_pixels=margin_pixels,
        )

    def _pick_margin_for_attempt(self, attempt: int) -> float:
        """The edge margin an attempt clicks with.

        The first attempts click the exact computed point - measured live,
        the widgets take clicks at their very edges almost everywhere, and
        the exact point is what colors at the gamut's rim (pure reds, pure
        white) need.  Later attempts assume the click died in a dead edge
        pixel (the hue bar's bottom two, live) and pull inward, trading a
        shade of accuracy for a click that lands.
        """

        schedule = self._PICK_MARGIN_SCHEDULE_PIXELS
        return schedule[min(attempt - 1, len(schedule) - 1)]

    def _pick_until_shown(
        self,
        picker_color: RGBColor,
        target: PaintingTarget,
        settings: PainterSettings,
        epoch: int,
        attempts: int,
        *,
        first_attempt: int = 1,
    ) -> SwatchReading | None:
        """Click the picker until the panel shows the selected color.

        Each attempt recomputes its click points at that attempt's edge
        margin - the exact point first, then progressively inside the
        widgets - and verifies the panel against the color THOSE points
        select.  Returns None once the panel agrees (or once it stopped
        being read), else the last reading.
        """

        reading: SwatchReading | None = None
        for attempt in range(first_attempt, first_attempt + attempts):
            retry = attempt > 1
            hue_point, sv_point, expected = self._picker_plan(
                picker_color, target, self._pick_margin_for_attempt(attempt)
            )
            self._click_picker(hue_point, sv_point, settings, epoch, retry=retry)
            if self._swatch is None:
                return None
            reading = self._read_selected_color(expected, epoch)
            if reading is None:
                return None
            if reading.matches(expected):
                with self._condition:
                    summary = self._color_pick_summary
                    self._color_pick_summary = replace(
                        summary,
                        picks=summary.picks + 1,
                        retried=summary.retried + (1 if retry else 0),
                    )
                if retry:
                    LOGGER.info(
                        "Color #%02X%02X%02X took on round %d of clicks", *expected, attempt
                    )
                return None
            LOGGER.warning(
                "The panel shows %s after selecting #%02X%02X%02X (round %d); clicking again",
                reading.hex,
                *expected,
                attempt,
            )
        return reading

    @classmethod
    def _nearby_hue_points(
        cls, hue_point: tuple[int, int], target: PaintingTarget
    ) -> tuple[tuple[int, int], ...]:
        """Whole-pixel hue neighbours, ordered closest-first and in-bounds.

        Rust's picker can be one or two physical pixels away from the
        calibrated gradient after a UI-scale change.  Changing a fractional
        inset does not help in that situation: it rounds straight back to the
        same pixel.  Keep this deliberately narrow; the swatch read-back is
        still the authority and no stroke is painted while searching.
        """

        points: list[tuple[int, int]] = []
        for offset in cls._PICKER_HUE_RESCUE_OFFSETS:
            point = (hue_point[0], hue_point[1] + offset)
            if target.hue_bar.contains(*point) and point not in points:
                points.append(point)
        return tuple(points)

    def _rescue_nearby_picker_hue(
        self,
        picker_color: RGBColor,
        target: PaintingTarget,
        settings: PainterSettings,
        epoch: int,
    ) -> SwatchReading | None:
        """Search adjacent hue pixels when normal verified picks disagree.

        This is a last, safe recovery for picker geometry that is slightly
        stale or rounded differently by Rust.  It deliberately does not
        accept a merely-near colour: the selected swatch must match the
        intended picker command before the caller may paint with it.
        """

        hue_point, sv_point, _expected = self._picker_plan(picker_color, target, 0.0)
        reading: SwatchReading | None = None
        for hue_candidate in self._nearby_hue_points(hue_point, target):
            self._click_picker(hue_candidate, sv_point, settings, epoch, retry=True)
            if self._swatch is None:
                return None
            reading = self._read_selected_color(picker_color, epoch)
            if reading is None:
                return None
            if reading.matches(picker_color):
                with self._condition:
                    summary = self._color_pick_summary
                    self._color_pick_summary = replace(
                        summary,
                        picks=summary.picks + 1,
                        retried=summary.retried + 1,
                    )
                LOGGER.info(
                    "Color #%02X%02X%02X took after searching a nearby hue pixel",
                    *picker_color,
                )
                return None
        return reading

    def _click_picker(
        self,
        hue_point: tuple[float, float],
        sv_point: tuple[float, float],
        settings: PainterSettings,
        epoch: int,
        *,
        retry: bool = False,
    ) -> None:
        """The two picker clicks; a retry holds them and waits twice as long."""

        scale = 2.0 if retry else 1.0
        self._safe_click(
            hue_point, epoch, hold_floor=self._PICKER_CLICK_HOLD_SECONDS * scale
        )
        self._interruptible_sleep(
            self._settle(settings.delay_after_hue_seconds) * scale,
            epoch=epoch,
            check_focus=True,
        )
        self._safe_click(
            sv_point, epoch, hold_floor=self._PICKER_CLICK_HOLD_SECONDS * scale
        )
        self._interruptible_sleep(
            self._settle(settings.delay_after_saturation_value_seconds) * scale,
            epoch=epoch,
            check_focus=True,
        )

    def _read_selected_color(self, expected: RGBColor, epoch: int) -> SwatchReading | None:
        """Read the panel's color block; once more after a moment if it disagrees.

        None when the block cannot be captured, which stops it being read
        for the rest of the job rather than failing every color.
        """

        swatch = self._swatch
        if swatch is None:
            return None
        try:
            reading = read_swatch(self._screen_capture, swatch)
            if reading.matches(expected):
                return reading
            self._interruptible_sleep(
                self._SWATCH_RECHECK_SECONDS, epoch=epoch, check_focus=True
            )
            return read_swatch(self._screen_capture, swatch)
        except (_RetryAction, _AbortRequested):
            raise
        except Exception:
            LOGGER.warning("Could not read the panel's color block", exc_info=True)
            self._stop_reading_picks("the color block could not be captured")
            return None

    def _locate_color_swatch(self, job: _Job) -> None:
        """Find the panel's selected-color block, so every pick can be read back."""

        settings = job.settings
        if not settings.verify_color_picks:
            self._stop_reading_picks("turned off")
            return
        if not getattr(self.input, "emits_real_input", True):
            self._stop_reading_picks("no real input")
            return
        epoch = self._pause_generation_value()
        if self._find_swatch(job.target, settings, epoch):
            LOGGER.info(
                "Color picks are read back from the panel's color block at %d,%d %dx%d",
                self._swatch.left,  # type: ignore[union-attr]
                self._swatch.top,  # type: ignore[union-attr]
                self._swatch.width,  # type: ignore[union-attr]
                self._swatch.height,  # type: ignore[union-attr]
            )
        else:
            self._stop_reading_picks("no color block found beside the hue bar")
            LOGGER.warning(
                "No selected-color block was found beside the hue bar; color "
                "picks are not read back on this job"
            )

    def _find_swatch(self, target: PaintingTarget, settings: PainterSettings, epoch: int) -> bool:
        """Select the locator color and look for the block showing it."""

        hue_point, sv_point, shown = self._picker_plan(
            LOCATOR_COLOR, target, self._PICKER_BLIND_MARGIN_PIXELS
        )
        hue_bar = ScreenRect(
            target.hue_bar.left, target.hue_bar.top, target.hue_bar.width, target.hue_bar.height
        )
        for attempt in range(2):
            self._click_picker(hue_point, sv_point, settings, epoch, retry=attempt > 0)
            self._interruptible_sleep(
                self._SWATCH_RECHECK_SECONDS, epoch=epoch, check_focus=True
            )
            try:
                found = locate_swatch(self._screen_capture, hue_bar, shown)
            except (_RetryAction, _AbortRequested):
                raise
            except Exception:
                LOGGER.warning("Could not look for the panel's color block", exc_info=True)
                found = None
            if found is not None:
                self._swatch = found
                return True
        self._swatch = None
        return False

    def _stop_reading_picks(self, reason: str) -> None:
        self._swatch = None
        with self._condition:
            self._color_pick_summary = replace(self._color_pick_summary, skipped_reason=reason)

    def _execute_stroke(
        self,
        stroke: object,
        plan: PaintPlan,
        canvas: RectangleLike,
        settings: PainterSettings,
        epoch: int,
        bias: tuple[float, float] = (0.0, 0.0),
        extension: float = 0.0,
        clamp_rect: RectangleLike | None = None,
        mapper: "Callable[[float, float], tuple[float, float]] | None" = None,
        texel_pitch: float | None = None,
        line_tool: bool = False,
        drag_rate: float | None = None,
        swept: TexelGridModel | None = None,
    ) -> None:
        # The line tool draws straight between its endpoints, so only a run
        # the plan itself laid straight may use it; a diagonal keeps gliding.
        line_tool = line_tool and (
            stroke.start_x == stroke.end_x or stroke.start_y == stroke.end_y  # type: ignore[attr-defined]
        )
        if swept is not None and (
            stroke.start_y == stroke.end_y or stroke.start_x == stroke.end_x  # type: ignore[attr-defined]
        ):
            # Texel-exact: the press goes to the pixel measured to stamp the
            # first texel, and a run ends at the pixel measured to paint the
            # last texel and no further.  No rounding, clamp or extension:
            # every pixel here was seen accepted in the sweep.
            u0, v0 = int(stroke.start_x), int(stroke.start_y)  # type: ignore[attr-defined]
            u1, v1 = int(stroke.end_x), int(stroke.end_y)  # type: ignore[attr-defined]
            start_int = swept.aim_pixel(u0, v0)
            if (u0, v0) == (u1, v1):
                # A cell whose dab the export saw land on a neighbour is
                # aimed the measured pixel the other way.
                nudge = self._cell_nudges.get((u0, v0))
                if nudge is not None:
                    start_int = (start_int[0] + nudge[0], start_int[1] + nudge[1])
                end_int = start_int
            elif v0 == v1:
                end_int = (swept.drag_end_x(u1, 1 if u1 > u0 else -1), start_int[1])
            else:
                end_int = (start_int[0], swept.drag_end_y(v1, 1 if v1 > v0 else -1))
            self._screen_stroke(
                start_int,
                end_int,
                settings,
                epoch,
                texel_pitch=texel_pitch,
                line_tool=line_tool,
                # The normal one-jump drag is the fast, measured path for a
                # swept map. A proved Shift line is faster still for long
                # axis-aligned runs, and _screen_stroke selects it first.
                jump=not line_tool,
            )
            return
        if mapper is not None:
            # A measured cursor map places each cell itself; the rectangle
            # only bounds the mouse.
            start = mapper(stroke.start_x, stroke.start_y)  # type: ignore[attr-defined]
            end = mapper(stroke.end_x, stroke.end_y)  # type: ignore[attr-defined]
            # A stroke the plan lays along one row (or one column) must be
            # commanded along one row on screen too.  The cursor map's shear
            # makes the two endpoints' ideal ys differ by a fraction of a
            # pixel, and rounding them separately can split them across a
            # pixel boundary - the whole drag then rides half a texel off
            # its row (the murica run's dominant fine-detail error).  One
            # shared coordinate, from the stroke's middle, keeps the drag on
            # its row; the sub-pixel shear across one stroke's length is far
            # smaller than the half-texel the split costs.
            if stroke.start_y == stroke.end_y:  # type: ignore[attr-defined]
                shared_y = (start[1] + end[1]) / 2.0
                start = (start[0], shared_y)
                end = (end[0], shared_y)
            if stroke.start_x == stroke.end_x:  # type: ignore[attr-defined]
                shared_x = (start[0] + end[0]) / 2.0
                start = (shared_x, start[1])
                end = (shared_x, end[1])
            if start == end:
                # A cell the touch-up saw land in a neighbour is aimed the
                # measured pixel or two the other way.
                nudge = self._cell_nudges.get(
                    (int(stroke.start_x), int(stroke.start_y))  # type: ignore[attr-defined]
                )
                if nudge is not None:
                    start = (start[0] + nudge[0], start[1] + nudge[1])
                    end = start
        else:
            start, end = logical_stroke_to_screen(stroke, plan.width, plan.height, canvas)  # type: ignore[arg-type]
        if extension > 0.0:
            # The brush is row-sized, so each stroke reaches out sideways to
            # cover the cell width; a dab becomes a tiny horizontal drag.
            direction = 1.0 if end[0] >= start[0] else -1.0
            start = (start[0] - direction * extension, start[1])
            end = (end[0] + direction * extension, end[1])
        if bias != (0.0, 0.0):
            # The sign paints where the cursor is plus its measured rendering
            # bias, so the cursor aims the same distance the other way.
            start = (start[0] - bias[0], start[1] - bias[1])
            end = (end[0] - bias[0], end[1] - bias[1])
        # The mouse itself must stay inside the *calibrated* rectangle even
        # when the strokes were laid out on the registered texture extent.
        bounds = clamp_rect if clamp_rect is not None else canvas
        start = self._space_and_clamp(start, bounds, settings.logical_pixel_spacing)
        end = self._space_and_clamp(end, bounds, settings.logical_pixel_spacing)
        if mapper is not None:
            # Round half-up, ONCE, after every sub-pixel adjustment: the old
            # order rounded inside the cursor map and floored here, so a
            # 0.14 px extension pulled every dab's left end a whole pixel
            # (0.57 texel) left of its aim.  The map's outputs are
            # continuous, so half-up is the unbiased choice.
            start_int = math.floor(start[0] + 0.5), math.floor(start[1] + 0.5)
            end_int = math.floor(end[0] + 0.5), math.floor(end[1] + 0.5)
        else:
            # Cell centers such as 0.5 must use floor, not round: on a
            # one-pixel-per-cell canvas half-up would push every center into
            # the next cell over.
            start_int = math.floor(start[0]), math.floor(start[1])
            end_int = math.floor(end[0]), math.floor(end[1])
        self._screen_stroke(
            start_int,
            end_int,
            settings,
            epoch,
            texel_pitch=texel_pitch,
            line_tool=line_tool,
            drag_rate=drag_rate,
        )

    @staticmethod
    def _texel_pitch_pixels(
        plan: PaintPlan,
        canvas: RectangleLike,
        model: BrushSizeModel | None,
        grid: TexelGridModel | None,
    ) -> float:
        """Screen pixels per sign texel, from the best measurement at hand.

        The grid probe measures it directly.  Failing that, the brush model's
        slope is one Size unit - one texel - as a fraction of the canvas.
        Failing both, a logical cell, which is never smaller than a texel, so
        the cap it sets on long drags errs on the quick side.
        """

        if grid is not None:
            return min(grid.pitch_x, grid.pitch_y)
        if model is not None:
            candidates = [model.slope * canvas.height]
            if model.has_horizontal_model:
                candidates.append(model.slope_x * canvas.width)
            pitch = min(candidates)
            if math.isfinite(pitch) and pitch > 0.0:
                return pitch
        return min(canvas.width / max(1, plan.width), canvas.height / max(1, plan.height))

    def _settle(self, seconds: float) -> float:
        """A picker, brush or between-colors delay, lifted to a frame."""

        if self.input.emits_real_input:
            return max(seconds, self._SETTLE_FLOOR_SECONDS)
        return seconds

    def _stroke_gap(self, seconds: float) -> float:
        if not self.input.emits_real_input:
            return seconds
        gap = max(seconds, self._STROKE_GAP_FLOOR_SECONDS)
        measured = self._measured_stroke_gap_seconds
        if measured is not None and self._timing_probes_apply():
            gap = min(gap, measured)
        return gap

    def _timing_probes_apply(self) -> bool:
        """Whether the job's probed floors are in force (its switch is on)."""

        job = self._job
        return job is None or bool(job.settings.measure_press_hold)

    def _drag_rate_cap(self) -> float:
        """Texels per second a long drag is capped at: the floor, or the probed rate."""

        measured = self._measured_drag_rate
        if measured is not None and self._timing_probes_apply():
            return max(self._LONG_DRAG_MAX_TEXELS_PER_SECOND, measured)
        return self._LONG_DRAG_MAX_TEXELS_PER_SECOND

    def _screen_stroke(
        self,
        start_int: tuple[int, int],
        end_int: tuple[int, int],
        settings: PainterSettings,
        epoch: int,
        *,
        texel_pitch: float | None = None,
        line_tool: bool = False,
        hold_seconds: float | None = None,
        drag_rate: float | None = None,
        jump: bool = False,
    ) -> None:
        """Drag between two physical points, or dab when they are the same.

        With ``jump`` the drag is one cursor move from start to end: the
        game fills every texel centre the segment crosses however far it
        is (measured 1016 of 1016 at 5,000 texels a second), so pacing
        buys nothing.  The dwell before the release stays: a release under
        30 ms after the press lost a third of such drags live.

        ``texel_pitch`` paces the drag: a run of a few texels goes at the
        set speed and is caught by the frame hold below, a longer drag is
        capped to a rate and an event spacing the game paints faithfully
        (:func:`app.paint_timing.stroke_pace`).  With ``line_tool`` (probed
        on this sign, straight stroke) a run of
        :data:`SHIFT_LINE_MIN_TEXELS` texels or more is not dragged along:
        the game fills it between a press and a release made with Shift held.
        """

        if (
            line_tool
            and texel_pitch is not None
            and math.isfinite(texel_pitch)
            and texel_pitch > 0.0
            and math.hypot(end_int[0] - start_int[0], end_int[1] - start_int[1])
            # A 32-texel inclusive run has 31 texel intervals between its
            # endpoint centres. Compare intervals so the documented
            # threshold is actually 32 texels, not 33.
            >= (self._SHIFT_LINE_MIN_TEXELS - 1.0) * texel_pitch
        ):
            self._line_stroke(start_int, end_int, settings, epoch)
            return
        self._checkpoint(epoch=epoch, check_focus=True)
        self._move(start_int, epoch)
        self._checkpoint(epoch=epoch, check_focus=True)
        self._mouse_down(epoch)
        pressed_at = time.monotonic()
        try:
            distance = math.hypot(end_int[0] - start_int[0], end_int[1] - start_int[1])
            if distance == 0:
                # A dab is one press and release - inside a single 67 ms frame
                # of Rust's 15 FPS paint UI it can be sampled as nothing at
                # all, exactly like the picker clicks held for the same reason.
                # A silently dropped dab is a missing cell the verification
                # pass then has to buy back with a whole extra
                # capture-and-repaint round.
                self._interruptible_sleep(
                    hold_seconds
                    if hold_seconds is not None
                    else self._press_hold_seconds(settings),
                    epoch=epoch,
                    check_focus=True,
                )
                return
            if jump:
                self._checkpoint(epoch=epoch, check_focus=True)
                self._move(end_int, epoch)
                self._interruptible_sleep(
                    self._drag_dwell_seconds(settings), epoch=epoch, check_focus=True
                )
                return
            pace = stroke_pace(
                distance,
                speed_pixels_per_second=settings.stroke_speed_pixels_per_second,
                step_pixels=settings.stroke_interpolation_step_pixels,
                texel_pitch_pixels=texel_pitch if texel_pitch is not None else float("nan"),
                real_input=bool(self.input.emits_real_input),
                max_texels_per_second=(
                    drag_rate if drag_rate is not None else self._drag_rate_cap()
                ),
                max_step_texels=self._LONG_DRAG_MAX_STEP_TEXELS,
            )
            steps = max(1, int(math.ceil(distance / pace.step_pixels)))
            delay = pace.move_seconds / steps
            for step in range(1, steps + 1):
                self._checkpoint(epoch=epoch, check_focus=True)
                ratio = step / steps
                # Waypoints round half-up: flooring biased every intermediate
                # point down-left, so a drag whose endpoints straddled a pixel
                # boundary in y rode the shifted row for its whole length
                # instead of splitting near its middle.
                point = (
                    math.floor(start_int[0] + (end_int[0] - start_int[0]) * ratio + 0.5),
                    math.floor(start_int[1] + (end_int[1] - start_int[1]) * ratio + 0.5),
                )
                self._move(point, epoch)
                self._interruptible_sleep(delay, epoch=epoch, check_focus=True)
            # A short drag is a dab that moved.  At a fine painting resolution
            # a run of a few cells is only a few screen pixels, so the whole
            # press - down, one or two moves, up - is over in under 10 ms and
            # can fall inside one frame just as a dab can.  The game then
            # keeps at most the press position and the rest of the run stays
            # bare; read back from a real sign, every hole was a run of 2-8
            # cells missing from the middle of a stroke while dabs, protected
            # by their hold, were fine.  Keeping the button down at the end
            # until the press has lasted a frame gives the game a frame in
            # which it sees the cursor held at the far end of the run, and
            # costs nothing on the long strokes that already spend frames
            # moving.  The hold is measured from the press rather than summed
            # from the nominal delays so the scheduler's own slack counts.
            # Long drags dwell a frame at the far end too: their travel time
            # exceeds the press floor, but the last inter-frame segment of a
            # capped drag - up to ~16 texels at 15 FPS - otherwise exists
            # only between two samples the game may never take together.
            if self.input.emits_real_input:
                elapsed = time.monotonic() - pressed_at
                hold = self._drag_dwell_seconds(settings)
                if pace.move_seconds >= hold:
                    remaining = hold
                else:
                    remaining = hold - elapsed
                if remaining > 0:
                    self._interruptible_sleep(remaining, epoch=epoch, check_focus=True)
        finally:
            self.input.mouse_up(MouseButton.LEFT)

    def _line_stroke(
        self,
        start_int: tuple[int, int],
        end_int: tuple[int, int],
        settings: PainterSettings,
        epoch: int,
    ) -> None:
        """Draw a straight stroke with Rust's line tool: Shift held through a drag.

        With Shift down, the game does not paint the cursor's path; it takes
        the press as one end and the release as the other and fills the
        straight run between them itself, in its own texture space.  So the
        cursor goes to the start, Shift goes down, the button goes down and
        is held a frame so the game registers where the stroke begins, the
        cursor jumps straight to the end and is held there a frame so the
        game sees it pressed at the far end, and the release draws the line.
        Shift stays down a moment past the release, since the fill happens
        on release and a frame that saw the release without the modifier
        would paint nothing but two dabs.  Every exit - a pause, an abort,
        an input failure - releases the button and Shift, because a
        modifier left down would turn the next stroke's drag into a line.
        """

        hold = self._press_hold_seconds(settings)
        lead = (
            self._SHIFT_LINE_MODIFIER_LEAD_SECONDS
            if self.input.emits_real_input
            else 0.0
        )
        self._checkpoint(epoch=epoch, check_focus=True)
        self._move(start_int, epoch)
        self._checkpoint(epoch=epoch, check_focus=True)
        self.input.key_down(self._LINE_TOOL_KEY)
        try:
            if lead:
                self._interruptible_sleep(lead, epoch=epoch, check_focus=True)
            self._mouse_down(epoch)
            try:
                self._interruptible_sleep(hold, epoch=epoch, check_focus=True)
                self._checkpoint(epoch=epoch, check_focus=True)
                self._move(end_int, epoch)
                self._interruptible_sleep(hold, epoch=epoch, check_focus=True)
            finally:
                self.input.mouse_up(MouseButton.LEFT)
            if lead:
                self._interruptible_sleep(lead, epoch=epoch, check_focus=True)
        finally:
            self.input.key_up(self._LINE_TOOL_KEY)

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

    # Some picker-widget edge pixels ignore clicks (measured live: the hue
    # bar's bottom two; its top edge and the S/V corner take clicks fine).
    # The old answer was to pull every click 2% of the widget inward, which
    # on a wrapping 360-degree hue bar silently cost 7.2 degrees of red at
    # each end - the murica sign's flag reds all painted at 351 degrees.
    # Now the first attempts click the exact computed point and the panel
    # read-back decides: a swallowed click is retried progressively deeper.
    _PICK_MARGIN_SCHEDULE_PIXELS: tuple[float, ...] = (0.0, 0.0, 2.0, 2.0, 4.0)
    # The picker is normally exact.  If its live raster has shifted by a
    # handful of pixels, scan only this small hue neighbourhood and require
    # swatch read-back before allowing the next stroke.
    _PICKER_HUE_RESCUE_OFFSETS: tuple[int, ...] = (
        -1,
        1,
        -2,
        2,
        -3,
        3,
        -4,
        4,
        -5,
        5,
        -6,
        6,
        -7,
        7,
        -8,
        8,
    )
    # Without a readable panel there is no way to see a swallowed click, so
    # blind picks stay this far inside the widgets - past every dead edge
    # pixel seen live, at a color cost of ~2 degrees of hue at the ends.
    _PICKER_BLIND_MARGIN_PIXELS = 2.0

    # Rust has been observed running its paint UI at 15 FPS, where a click held
    # shorter than one 67 ms frame can be sampled as nothing.  Picker clicks are
    # rare (a handful per color change), so holding them across a frame costs
    # nothing next to a silently unchanged color.
    _PICKER_CLICK_HOLD_SECONDS = PICKER_CLICK_HOLD_SECONDS

    # Every stroke's press lasts at least this long, for the same 15 FPS
    # reason: a dab, or a drag so short it would otherwise be over in a few
    # milliseconds, is held until the press has straddled a frame boundary in
    # all but the unluckiest alignment.  Slightly under a frame rather than
    # the picker's 90 ms because these strokes can number in the thousands.
    # Long drags spend more than this moving and are not held at all.
    _MIN_PRESS_SECONDS = MIN_PRESS_SECONDS

    # The floors under the settle and between-stroke delays, and the rate
    # cap on long drags (:mod:`app.paint_timing`).  Class attributes so a
    # test driving a simulated sign, which has no frame rate, can switch
    # them off the way it does the other frame waits.
    _SETTLE_FLOOR_SECONDS = SETTLE_FLOOR_SECONDS
    _STROKE_GAP_FLOOR_SECONDS = STROKE_GAP_FLOOR_SECONDS
    _LONG_DRAG_MAX_TEXELS_PER_SECOND = LONG_DRAG_MAX_TEXELS_PER_SECOND
    _LONG_DRAG_MAX_STEP_TEXELS = LONG_DRAG_MAX_STEP_TEXELS

    # The Shift line tool: the key held through the line's drag, how long a
    # straight run must be before a press-jump-release beats a capped drag,
    # and the lead/trail that keeps the modifier around the press and release.
    _LINE_TOOL_KEY = "SHIFT"
    _SHIFT_LINE_MIN_TEXELS = SHIFT_LINE_MIN_TEXELS
    _SHIFT_LINE_MODIFIER_LEAD_SECONDS = SHIFT_LINE_MODIFIER_LEAD_SECONDS

    def _safe_click(
        self,
        point: tuple[float, float],
        epoch: int,
        *,
        hold_floor: float = 0.0,
    ) -> None:
        self._checkpoint(epoch=epoch, check_focus=True)
        # Picker normalization already targets inclusive physical endpoints, for
        # which conventional rounding is appropriate.
        target = int(round(point[0])), int(round(point[1]))
        self._move(target, epoch)
        self._checkpoint(epoch=epoch, check_focus=True)
        self._mouse_down(epoch)
        try:
            settings = self._job.settings if self._job is not None else PainterSettings()
            hold = max(settings.mouse_down_duration_seconds, hold_floor)
            self._interruptible_sleep(hold, epoch=epoch, check_focus=True)
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
                self._try_auto_resume_after_focus_loss()
                with self._condition:
                    if self._pause_event.is_set() or self._state == PainterState.PAUSED:
                        self._condition.wait(timeout=0.05)
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
                self._pause_for_foreground_loss(settings)
                return True
        if check_focus and self._check_painting_ui(settings, now):
            return True

        return self._check_cursor(settings, now)

    def _pause_for_foreground_loss(self, settings: PainterSettings) -> None:
        """Pause for focus loss and arm hands-off recovery when configured."""

        anchor: tuple[int, int] | None = None
        if settings.auto_resume_on_focus_return:
            try:
                anchor = self.input.get_cursor_position()
            except Exception:
                LOGGER.warning("Could not read the cursor for focus recovery", exc_info=True)
        self.pause("foreground window lost")
        with self._condition:
            # The pause may have lost a race to Stop; never revive such a job.
            if self._state is not PainterState.PAUSED or self._abort_requested:
                return
            self._auto_resume_focus_wait = settings.auto_resume_on_focus_return
            self._auto_resume_cursor_anchor = anchor
            self._next_auto_focus_retry_at = (
                time.monotonic() + settings.auto_resume_focus_retry_seconds
            )
        if settings.auto_resume_on_focus_return:
            LOGGER.info(
                "Rust focus lost; retrying in %.0f seconds unless the mouse moves",
                settings.auto_resume_focus_retry_seconds,
            )

    def _try_auto_resume_after_focus_loss(self) -> None:
        """Resume a focus-loss pause only after Rust returns and the mouse rests."""

        with self._condition:
            job = self._job
            if (
                job is None
                or self._state is not PainterState.PAUSED
                or not self._auto_resume_focus_wait
                or self._abort_requested
            ):
                return
            now = time.monotonic()
            if now < self._next_auto_focus_retry_at:
                return
            settings = job.settings
            self._next_auto_focus_retry_at = now + settings.auto_resume_focus_retry_seconds
            anchor = self._auto_resume_cursor_anchor

        if anchor is not None:
            try:
                cursor = self.input.get_cursor_position()
            except Exception:
                LOGGER.warning("Could not read the cursor for focus recovery", exc_info=True)
                return
            moved = math.hypot(cursor[0] - anchor[0], cursor[1] - anchor[1])
            if moved >= settings.mouse_move_pause_threshold_pixels:
                reason = "foreground window lost; automatic retry cancelled because mouse moved"
                with self._condition:
                    if self._state is not PainterState.PAUSED:
                        return
                    self._auto_resume_focus_wait = False
                    self._auto_resume_cursor_anchor = None
                    self._state_reason = reason
                LOGGER.info(reason)
                self._emit_state(PainterState.PAUSED, reason)
                self._update_progress_state(PainterState.PAUSED, f"Paused: {reason}")
                return

        requirement = ForegroundRequirement(
            title_contains=settings.expected_window_title_contains or None,
            executable=settings.expected_process_name or None,
        )
        if not self._foreground_checker(requirement):
            LOGGER.info("Rust is still not foreground; retrying focus recovery later")
            return
        LOGGER.info("Rust focus returned; automatically resuming painting")
        self.resume()

    def _check_cursor(self, settings: PainterSettings, now: float) -> bool:
        """Sample the real cursor for user movement.

        Movement only ever pauses; nothing but the user's Stop button or
        abort hotkey aborts a job, so a stray cursor can never throw away the
        work.  Returns True when the job just paused and the caller must
        re-evaluate its state.
        """

        if not settings.pause_on_mouse_move:
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
        if now - self._last_cursor_check < settings.safety_poll_interval_seconds:
            return False
        self._last_cursor_check = now
        try:
            cursor = self.input.get_cursor_position()
        except Exception:
            LOGGER.warning("Could not read the cursor for safety checks", exc_info=True)
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
        completed_work: float | None = None,
        total_work: float | None = None,
        skipped_work: float = 0.0,
        phase_elapsed: float | None = None,
        pending_seconds: float = 0.0,
        message: str,
    ) -> None:
        """Publish progress.

        ``skipped_work`` is the predicted cost of strokes a resumed job took
        as already painted: it counts toward the percent, since the sign is
        that far along, and not toward the pace, which this run's own
        strokes set.  ``pending_seconds`` is work that follows this plan -
        the touch-up pass after the artwork - and is added to the time left
        without moving the percent, which is the artwork's.
        """

        elapsed = self._active_elapsed()
        if completed_work is None or not total_work:
            completed_work = float(completed_strokes)
            total_work = float(total_strokes)
            skipped_work = 0.0
        percent = (
            100.0
            if total_work <= 0 or completed_strokes >= total_strokes
            else min(100.0, (skipped_work + completed_work) * 100.0 / total_work)
        )
        remaining = None
        if completed_strokes < total_strokes:
            remaining = remaining_seconds(
                elapsed if phase_elapsed is None else phase_elapsed,
                completed_work,
                max(0.0, total_work - skipped_work),
            )
        if remaining is not None and pending_seconds > 0.0:
            remaining += pending_seconds
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
                self._progress.phase,
                self._progress.alerts,
            )
        self._emit_progress(force=completed_strokes == total_strokes)

    def _update_progress_state(
        self, state: PainterState, message: str, *, phase: str | None = None
    ) -> None:
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
                old.phase if phase is None else phase,
                old.alerts,
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
