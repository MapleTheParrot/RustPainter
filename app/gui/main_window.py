"""Main PySide6 window for RustPainter."""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass, fields, replace
from enum import Enum
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from PIL import Image, ImageOps, ImageQt
from PySide6.QtCore import (
    QByteArray,
    QObject,
    QPointF,
    QRunnable,
    QRectF,
    QSize,
    Qt,
    QThreadPool,
    QTimer,
    Signal,
    Slot,
)
from PySide6.QtGui import (
    QAction,
    QColor,
    QCloseEvent,
    QDragEnterEvent,
    QDragMoveEvent,
    QDropEvent,
    QFont,
    QImage,
    QKeySequence,
    QPainter,
    QPixmap,
    QShortcut,
)
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.color_calibration import ColorCorrectionModel
from app.image_processing import (
    calculate_fit_size,
    crop_centering,
    fill_crop_box,
    process_image,
    quantize_image,
)
from app.calibration import (
    CalibrationPreviewOverlay,
    capture_display_metadata,
    select_screen_rect,
)
from app.models import (
    BackgroundRemovalScope,
    CropAlignment,
    ImageProcessOptions,
    PaintMode,
    PaintPlan,
    ProcessedImage,
    ScaleMode,
    SharpenMode,
    ScreenRect,
    TransparencyMode,
)
from app.paint_plan import count_unmerged_strokes, generate_paint_plan
from app.paint_timing import (
    BRUSH_CALIBRATION_SECONDS,
    MIN_PRESS_SECONDS,
    SETTLE_FLOOR_SECONDS,
    STROKE_GAP_FLOOR_SECONDS,
    LearnedTiming,
    PlanProfile,
    RunEstimate,
    StrokeTiming,
)
from app.paint_optimizer import (
    BrushCapabilities,
    OptimizationStatistics,
    mode_options,
    optimize_paint_plan,
)
from app.hotkeys import SUPPORTED_HOTKEY_CHOICES
from app.profiles import Profile, ProfileStore
from app.resume_record import (
    ResumeRecord,
    ResumeRecordStore,
    advanced as advanced_record,
    plan_fingerprint,
    plan_prefix_labels,
    record_for_job,
)
from app.settings import DEFAULT_COLOR_COUNT, SettingsStore, default_settings
from app.timelapse_export import (
    DEFAULT_FRAME_RATE,
    MAX_FRAME_RATE,
    MIN_FRAME_RATE,
    ExportCancelled,
    available_formats,
    export_session,
    format_for,
    session_frames,
)

from .assets import icon as art_icon, pixmap as art_pixmap, tinted_pixmap
from .styles import DANGER, ON_ACCENT, TEXT, badge_foreground, state_badge_style
from .text_render import (
    GRADIENT_DIRECTIONS,
    MAX_OUTLINE_WIDTH,
    TextStyle,
    draw_text,
    layer_font,
    text_size,
)
from .widgets import (
    ScreenshotViewer,
    BusyOverlay,
    CalibrationStatus,
    ColorButton,
    CountdownDialog,
    InlineNotice,
    NoWheelComboBox,
    NoWheelDoubleSpinBox,
    NoWheelFontComboBox,
    NoWheelSpinBox,
    PreviewLabel,
    QtLogHandler,
    Spinner,
    TextEditorPreview,
    dropped_image_path,
)


LOGGER = logging.getLogger("rust_painter")


# Header badge artwork per painter state; anything unlisted falls back to the
# neutral status dial.
_BADGE_ICONS: dict[str, str] = {
    "idle": "check",
    "ready": "check",
    "completed": "check",
    "running": "play",
    "paused": "pause",
    "countdown": "clock",
    "error": "abort",
    "aborted": "abort",
}


QUALITY_LONG_EDGE: dict[str, int] = {
    "Very Fast": 64,
    "Fast": 128,
    "Balanced": 256,
    "High": 384,
    "Very High": 512,
}

# The quality entry with no fixed long edge: it asks the measured brush model
# how many texture rows this sign actually holds and plans one logical cell
# per texel.  Before any job has measured the sign it plans one cell per
# screen pixel of the calibrated canvas instead - the finest grid the mouse
# can address - so the preset works from the first paint, with brush size 1
# merely repeating itself wherever screen pixels outnumber texels.
MAX_QUALITY_PRESET = "Max"

# How long a change waits before the plan is recalculated for it.  A control
# that moves once - a combo box, a checkbox - has said everything it is going
# to say, so it only needs long enough to coalesce with whatever moves with
# it.  Typing is different: every character is a change, and recalculating
# between keystrokes throws work away and keeps the busy overlay on screen for
# as long as the sentence takes to write.  So text waits for an actual pause
# instead - long enough that ordinary typing never starts a plan, short enough
# that stopping to look at the preview does not feel like waiting for it.
PLAN_SETTLE_MS = 180
TYPING_SETTLE_MS = 600

# How many finished plans are kept so that going back to a combination of
# settings already computed is instant.  A count alone is the wrong bound:
# six thumbnail-sized plans cost nothing, while six plans of a sign painted at
# native texel resolution are hundreds of megabytes, so the cache is held to a
# rough memory budget as well and gives up its oldest entries to stay inside
# it.  One entry is always kept, however large, because the plan on screen is
# in the cache too.
PLAN_CACHE_ENTRIES = 6
PLAN_CACHE_BYTES = 96 * 1024 * 1024

# Roughly what one entry costs: a Stroke plus its slot in the group's tuple,
# and per logical cell an RGBA quantized image, a boolean paint mask, and the
# RGBA simulation the Rust preview shows.
_STROKE_BYTES = 72
_CELL_BYTES = 4 + 1 + 4


# The crop-alignment entry a dragged crop selects.  Its centering is stored
# per image rather than as one of the five named anchors, which cannot express
# a crop the user framed by hand.
CUSTOM_CROP_VALUE = "custom"
CUSTOM_CROP_LABEL = "Custom - dragged"

# What the preview heading offers to teach, chosen by the tab in front and by
# whether the sign's window can actually be dragged over the source right now.
PREVIEW_HINTS: dict[str, str] = {
    "source": "Click the preview or drop an image anywhere to open it",
    "crop": "Drag the image to choose what the sign shows",
    "rust": "Read-only - edit the artwork on the Source tab",
}

# The floor under each timing spinbox, in milliseconds: Rust samples its
# paint UI at ~15 FPS, and an event shorter than a frame is not seen at
# all.  The painter runs anything under a floor at the floor, so the
# spinboxes stop there too - a value they cannot reach is not a speed.
SPEED_FLOORS_MS: dict[str, int] = {
    "dot_ms": int(round(MIN_PRESS_SECONDS * 1000)),
    "hue_ms": int(round(SETTLE_FLOOR_SECONDS * 1000)),
    "sv_ms": int(round(SETTLE_FLOOR_SECONDS * 1000)),
    "brush_ms": int(round(SETTLE_FLOOR_SECONDS * 1000)),
    "stroke_ms": int(round(STROKE_GAP_FLOOR_SECONDS * 1000)),
    "color_ms": int(round(SETTLE_FLOOR_SECONDS * 1000)),
}

# All timing values one speed preset controls, in the spinbox units used
# below.  The presets differ only where a difference can be painted: the
# holds and settles sit at their floors in every preset but Relaxed, which
# adds margin above them, and the stroke speed is what separates the rest -
# and even that only on long drags, since the painter caps those at a rate
# the game paints faithfully and runs short runs flat out regardless.
# Turbo is the floors everywhere.
SPEED_PRESETS: dict[str, dict[str, float]] = {
    "Relaxed": {
        "stroke_speed": 400.0,
        "dot_ms": 80,
        "hue_ms": 120,
        "sv_ms": 120,
        "stroke_ms": 35,
        "color_ms": 180,
        "interp_px": 3.0,
    },
    "Standard": {
        "stroke_speed": 700.0,
        "dot_ms": 70,
        "hue_ms": 90,
        "sv_ms": 90,
        "stroke_ms": 20,
        "color_ms": 120,
        "interp_px": 4.0,
    },
    "Fast": {
        "stroke_speed": 1300.0,
        "dot_ms": 70,
        "hue_ms": 70,
        "sv_ms": 70,
        "stroke_ms": 20,
        "color_ms": 80,
        "interp_px": 6.0,
    },
    "Turbo": {
        "stroke_speed": 2200.0,
        "dot_ms": 70,
        "hue_ms": 70,
        "sv_ms": 70,
        "stroke_ms": 20,
        "color_ms": 70,
        "interp_px": 8.0,
    },
}


def _floored_speed_values(values: dict[str, float]) -> dict[str, float]:
    """Preset values as the painter would run them."""

    return {
        key: max(float(value), float(SPEED_FLOORS_MS.get(key, 0)))
        for key, value in values.items()
    }

# Logical-pixel gap each stroke-merging mode may paint across.
MERGE_MODE_GAPS: dict[str, int | None] = {"off": 0, "balanced": 6, "maximum": None}
# Shown in the (disabled) merge box while a paint mode other than Exact is
# chosen: those modes merge through the optimizer, and a greyed-out "Off"
# read as merging being switched off when it was the opposite.
MERGE_MODE_OPTIMIZER = "optimizer"

# Bounds on the pixel font size of a text layer. Layers keep their size as a
# fraction of the logical canvas height and derive pixels within these bounds,
# so a caption keeps its proportions when the resolution changes.
MIN_TEXT_SIZE = 4
MAX_TEXT_SIZE = 256

# Matches the limit the settings schema validates, so a sign that fills up
# says so instead of failing the next save.
MAX_TEXT_LAYERS = 20

# Text edits of the same kind arriving inside this window fold into one
# undoable step, so holding an arrow key down leaves one thing to undo.
TEXT_HISTORY_COALESCE_SECONDS = 0.8

# How many steps back the text canvas can walk before the oldest are dropped.
MAX_TEXT_HISTORY = 100


# The layers, the layer the side panel names, and the canvas selection - one
# point the text canvas can be walked back to.
_TextSnapshot = tuple[tuple["_TextOverlayOptions", ...], int, tuple[int, ...]]


@dataclass(slots=True)
class _ProcessResult:
    serial: int
    processed: ProcessedImage
    plan: PaintPlan
    simulation: Image.Image
    # Stroke lengths and group changes reduced to what the time estimate
    # needs, built here off the GUI thread because it walks every stroke.
    timing_profile: PlanProfile
    unmerged_stroke_count: int
    optimization: OptimizationStatistics | None = None


@dataclass(slots=True)
class _LoadResult:
    serial: int
    path: Path
    image: Image.Image
    preview: Image.Image


@dataclass(slots=True)
class _PendingPaint:
    plan: PaintPlan
    profile: Any
    settings: dict[str, Any]
    dry_run: bool
    display_snapshot: Any = None
    # Strokes already on the sign, taken as painted: a job picked up where
    # an interrupted one left off starts here instead of at zero.
    start_stroke: int = 0


@dataclass(frozen=True, slots=True)
class _TextOverlayOptions:
    """A small, worker-safe snapshot of the text controls.

    ``font_size`` is in logical canvas pixels, which is what both renderers
    need, but it is derived from ``size_ratio`` — the height of the text as a
    fraction of the canvas. The ratio is what survives a change of painting
    resolution, so text keeps the same size on the finished sign whether it was
    placed under the Very Fast or the Very High preset.

    ``outline_width`` is in the same logical pixels, and a gradient runs from
    ``color`` to ``gradient_color`` across the text's own line box.
    """

    text: str
    font_family: str
    font_size: int
    color: tuple[int, int, int]
    x: float = 0.5
    y: float = 0.5
    bold: bool = False
    italic: bool = False
    size_ratio: float = 0.0
    gradient: bool = False
    gradient_color: tuple[int, int, int] = (255, 255, 255)
    gradient_direction: str = "vertical"
    outline_width: int = 0
    outline_color: tuple[int, int, int] = (0, 0, 0)


class _WorkerSignals(QObject):
    completed = Signal(object)
    failed = Signal(int, str)


def _hashable(value: Any) -> Any:
    """A stable, hashable stand-in for anything a plan was computed from.

    Dataclasses are walked field by field rather than listed by name, so an
    option added later is part of the cache key without anyone remembering to
    put it there - the failure mode of the alternative is a stale plan shown
    for settings that never produced it.
    """

    if hasattr(value, "__dataclass_fields__"):
        return tuple(
            (field.name, _hashable(getattr(value, field.name)))
            for field in fields(value)
        )
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, (list, tuple)):
        return tuple(_hashable(item) for item in value)
    if isinstance(value, Mapping):
        return tuple(sorted((str(key), _hashable(item)) for key, item in value.items()))
    return value


def _glyph_label(icon_name: str, edge: int) -> QLabel:
    """A baked icon as a plain label, for headings and inline captions."""

    glyph = QLabel()
    glyph.setFixedSize(edge, edge)
    glyph.setScaledContents(True)
    glyph.setPixmap(art_pixmap(icon_name, edge * 2))
    return glyph


def _rgb(color: QColor) -> tuple[int, int, int]:
    """The plain channel triple the layer model and the renderer both use."""

    return (color.red(), color.green(), color.blue())


@dataclass(frozen=True)
class _PickerGeometry:
    """The color panel's widgets, for quantizing plans to selectable colors."""

    hue_bar: Any
    color_box: Any
    hue_direction: str
    saturation_direction: str
    value_direction: str


def _predicted_sign_colors(
    rgb: np.ndarray, mask: np.ndarray, correction: ColorCorrectionModel
) -> np.ndarray:
    """Render painted colors as the measured sign response will return them.

    Painting sends ``correct(color)`` to the picker, so anything the material
    can reach comes back unchanged and the preview is left alone.  What this
    does surface is the colors that clip: outside the measured gamut the
    inverse is unreachable, and the preview stops promising a color the sign
    will never produce.
    """

    painted = rgb[mask]
    if painted.size == 0:
        return rgb
    # At most a palette's worth of distinct colors, so the per-color model calls
    # stay cheap however large the canvas is.
    unique, inverse = np.unique(painted.reshape(-1, 3), axis=0, return_inverse=True)
    predicted = np.array(
        [
            correction.predict(correction.correct(tuple(int(v) for v in color)))
            for color in unique
        ],
        dtype=np.uint8,
    )
    result = rgb.copy()
    result[mask] = predicted[inverse.reshape(-1)]
    return result


def _snap_processed_to_picker(
    processed: ProcessedImage, picker: "_PickerGeometry"
) -> ProcessedImage:
    """Replace every palette color with the one the picker can actually select.

    The color picker is clicked on whole pixels of finite widgets, so not
    every RGB is selectable: hues quantize to the bar's pixel rows and the
    extreme reds beyond its top row do not exist on it at all.  Quantizing
    the plan to the selectable palette makes plan, preview, and painted sign
    agree - the murica run's flag reds were promised at hue 359.7 and painted
    at 351.4 because nothing in the plan knew the difference.
    """

    from app.color_mapping import reachable_color

    rgb = np.asarray(processed.image.convert("RGB"), dtype=np.uint8)
    mask = np.asarray(processed.paint_mask, dtype=np.bool_)
    painted = rgb[mask]
    if painted.size == 0:
        return processed
    unique, inverse = np.unique(painted.reshape(-1, 3), axis=0, return_inverse=True)
    snapped = np.array(
        [
            reachable_color(
                tuple(int(v) for v in color),
                picker.hue_bar,
                picker.color_box,
                hue_direction=picker.hue_direction,
                saturation_direction=picker.saturation_direction,
                value_direction=picker.value_direction,
            )
            for color in unique
        ],
        dtype=np.uint8,
    )
    if np.array_equal(snapped, unique):
        return processed
    result = rgb.copy()
    result[mask] = snapped[inverse.reshape(-1)]
    return ProcessedImage(
        Image.fromarray(result, mode="RGB"),
        processed.paint_mask,
        processed.requested_colors,
    )


def _compose_checker_backdrop(
    source: np.ndarray,
    mask: np.ndarray,
    correction: ColorCorrectionModel | None = None,
) -> Image.Image:
    """Put painted pixels over the unpainted-checker, without Qt objects."""

    height, width = mask.shape
    scale = max(1, min(8, 800 // max(1, max(width, height))))
    tile = max(1, 8 // scale)
    row_tiles = (np.arange(height, dtype=np.int32) // tile)[:, None]
    column_tiles = (np.arange(width, dtype=np.int32) // tile)[None, :]
    dark_tiles = ((row_tiles + column_tiles) & 1).astype(np.bool_)
    checker = np.empty((height, width, 3), dtype=np.uint8)
    checker[:] = (38, 40, 37)
    checker[dark_tiles] = (58, 61, 56)
    if correction is not None:
        source = _predicted_sign_colors(source, mask, correction)
    checker[mask] = source[mask]
    return Image.fromarray(checker, mode="RGB")


def _build_simulation_image(
    processed: ProcessedImage, correction: ColorCorrectionModel | None = None
) -> Image.Image:
    """Build the checker-backed preview of the plan's logical target."""

    return _compose_checker_backdrop(
        np.asarray(processed.image.convert("RGB"), dtype=np.uint8),
        np.asarray(processed.paint_mask, dtype=np.bool_),
        correction,
    )


def _apply_text_overlays(
    processed: ProcessedImage,
    overlay_options: tuple[_TextOverlayOptions, ...],
) -> ProcessedImage:
    """Render text layers in logical-pixel space before the final palette pass."""

    visible_layers = tuple(layer for layer in overlay_options if layer.text.strip())
    if not visible_layers:
        return processed

    width, height = processed.size
    overlay_qt = QImage(width, height, QImage.Format.Format_RGBA8888)
    overlay_qt.fill(Qt.GlobalColor.transparent)
    painter = QPainter(overlay_qt)
    try:
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing, True)
        for layer in visible_layers:
            draw_text(
                painter,
                layer.text,
                layer_font(layer),
                QPointF(layer.x * width, layer.y * height),
                TextStyle.from_layer(layer),
            )
    finally:
        painter.end()

    overlay = ImageQt.fromqimage(overlay_qt).convert("RGBA")
    overlay_array = np.asarray(overlay, dtype=np.uint8).copy()
    # The quantizer snaps antialiased fringe texels to full palette colors,
    # which fattens every stroke and closes small counters like a P's bowl.
    # Thresholding coverage first keeps a texel either fully lettered or
    # untouched, so letters stay the width the font drew.
    overlay_mask = overlay_array[:, :, 3] >= 128
    if not np.any(overlay_mask):
        return processed
    overlay_array[:, :, 3] = np.where(overlay_mask, 255, 0).astype(np.uint8)
    overlay = Image.fromarray(overlay_array, mode="RGBA")

    combined_mask = np.asarray(processed.paint_mask, dtype=np.bool_).copy()
    combined_mask |= overlay_mask
    composited = Image.alpha_composite(processed.image.convert("RGBA"), overlay)
    quantized = quantize_image(
        composited,
        processed.requested_colors,
        paint_mask=combined_mask,
    )
    return ProcessedImage(quantized, combined_mask, processed.requested_colors)


class _ImageWorker(QRunnable):
    """Run resampling, quantization, and plan generation off the GUI thread."""

    def __init__(
        self,
        serial: int,
        image: Image.Image,
        options: ImageProcessOptions,
        overpaint_gap: int | None = 0,
        text_overlays: tuple[_TextOverlayOptions, ...] = (),
        color_correction: ColorCorrectionModel | None = None,
        paint_mode: str = PaintMode.EXACT.value,
        capabilities: BrushCapabilities | None = None,
        picker: "_PickerGeometry | None" = None,
    ) -> None:
        super().__init__()
        self.serial = serial
        self.image = image
        self.options = options
        self.overpaint_gap = overpaint_gap
        self.text_overlays = text_overlays
        self.color_correction = color_correction
        self.paint_mode = paint_mode
        self.capabilities = capabilities
        self.picker = picker
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            base_processed = process_image(self.image, self.options)
            processed = _apply_text_overlays(base_processed, self.text_overlays)
            if self.picker is not None:
                # Quantize the palette to colors the picker can select, so
                # the plan asks for - and the preview promises - only colors
                # the sign can actually receive.
                processed = _snap_processed_to_picker(processed, self.picker)
            mode = PaintMode(self.paint_mode)
            optimization = None
            if mode is PaintMode.EXACT:
                plan = generate_paint_plan(processed, overpaint_gap=self.overpaint_gap)
                unmerged_stroke_count = (
                    plan.stroke_count
                    if self.overpaint_gap == 0
                    else count_unmerged_strokes(processed)
                )
                simulation_processed = processed
            else:
                optimizer_options = mode_options(
                    mode, preserve_dither=self.options.dither
                )
                optimized = optimize_paint_plan(
                    processed,
                    mode,
                    capabilities=self.capabilities,
                    options=optimizer_options,
                )
                plan = optimized.plan
                optimization = optimized.statistics
                # The savings headline compares against the exact plan the
                # same processed image would have produced.
                unmerged_stroke_count = count_unmerged_strokes(processed)
                optimized_processed = ProcessedImage(
                    optimized.image, optimized.paint_mask, processed.requested_colors
                )
                simulation_processed = optimized_processed
                processed = optimized_processed
            # The simulation is the plan's own target, text baked in and
            # palette-limited, so the Rust preview promises exactly what the
            # painter will put on the sign.  Text stays editable as vector
            # items over the source image instead.
            simulation = _build_simulation_image(
                simulation_processed, self.color_correction
            )
            self.signals.completed.emit(
                _ProcessResult(
                    self.serial,
                    processed,
                    plan,
                    simulation,
                    PlanProfile.from_plan(plan),
                    unmerged_stroke_count,
                    optimization,
                )
            )
        except Exception as exc:  # surfaced to the GUI and log
            LOGGER.exception("Image processing failed")
            self.signals.failed.emit(self.serial, str(exc))


class _ImageLoadWorker(QRunnable):
    """Decode and orient imported images without blocking the Qt event loop."""

    def __init__(self, serial: int, path: Path) -> None:
        super().__init__()
        self.serial = serial
        self.path = path
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            with Image.open(self.path) as opened:
                opened.seek(0)
                image = ImageOps.exif_transpose(opened).convert("RGBA")
                image.load()
            preview = ImageOps.contain(
                image,
                (1600, 1200),
                method=Image.Resampling.LANCZOS,
            )
            self.signals.completed.emit(
                _LoadResult(self.serial, self.path, image, preview)
            )
        except Exception as exc:
            LOGGER.exception("Could not load image %s", self.path)
            self.signals.failed.emit(self.serial, str(exc))


class _ExportSignals(QObject):
    progress = Signal(int, int)
    completed = Signal(str)
    failed = Signal(str)
    cancelled = Signal()


class _TimelapseExportWorker(QRunnable):
    """Encode one recorded session into a video without blocking the GUI."""

    def __init__(
        self,
        frames: list[Path],
        destination: Path,
        frame_rate: int,
        format_key: str,
    ) -> None:
        super().__init__()
        self._frames = frames
        self._destination = destination
        self._frame_rate = frame_rate
        self._format_key = format_key
        self._cancelled = threading.Event()
        self.signals = _ExportSignals()

    def cancel(self) -> None:
        self._cancelled.set()

    @Slot()
    def run(self) -> None:
        try:
            export_session(
                self._frames,
                self._destination,
                frame_rate=self._frame_rate,
                video_format=format_for(self._format_key),
                on_progress=lambda done, total: self.signals.progress.emit(done, total),
                should_cancel=self._cancelled.is_set,
            )
        except ExportCancelled:
            LOGGER.info("Timelapse export cancelled: %s", self._destination)
            self.signals.cancelled.emit()
        except Exception as exc:
            LOGGER.exception("Could not export the timelapse")
            self.signals.failed.emit(str(exc))
        else:
            self.signals.completed.emit(str(self._destination))


class _DebugCancelled(RuntimeError):
    """Internal control-flow marker for a safely interrupted debug action."""


class _NameDialog(QDialog):
    def __init__(self, title: str, initial: str = "", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        layout = QVBoxLayout(self)
        self.edit = QLineEdit(initial)
        self.edit.selectAll()
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(QLabel("Profile name"))
        layout.addWidget(self.edit)
        layout.addWidget(buttons)

    def _accept_if_valid(self) -> None:
        if self.edit.text().strip():
            self.accept()

    @property
    def name(self) -> str:
        return self.edit.text().strip()


class _PainterBridge(QObject):
    """Converts painter-thread callbacks into queued Qt signals."""

    progress = Signal(int, object)
    state = Signal(int, object, str)
    completed = Signal(int)
    error = Signal(int, str)
    pause_screenshot = Signal(int, str, str, str)
    start_requested = Signal()
    pause_requested = Signal()
    abort_requested = Signal()
    hotkey_error = Signal(str)
    debug_finished = Signal(str, str)


class MainWindow(QMainWindow):
    """The complete desktop workflow in one intentionally straightforward window."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("RustPainter")
        self.resize(1380, 860)
        self.setMinimumSize(1040, 680)
        self.setAcceptDrops(True)

        self._original_image: Image.Image | None = None
        self._image_path: Path | None = None
        self._processed: ProcessedImage | None = None
        self._plan: PaintPlan | None = None
        self._plan_metric_source: PaintPlan | None = None
        self._plan_timing_profile: PlanProfile | None = None
        # What a stroke costs on this machine beyond its scripted holds,
        # learned from every run and kept between sessions.
        self._learned_timing = LearnedTiming.load(self._timing_path())
        # Finished plans, newest last, keyed by everything that shaped them.
        # Stepping back to a preset already tried is then instant instead of
        # another full recalculation of a plan that has not changed.
        self._plan_cache: OrderedDict[tuple, _ProcessResult] = OrderedDict()
        # The key each dispatched worker is planning for, by serial.  Keeping
        # it per worker rather than as one pending slot is what lets a result
        # that arrived too late to be shown still be filed correctly, instead
        # of being filed under whatever the newest request happened to be.
        self._plan_keys: dict[int, tuple] = {}
        self._load_serial = 0
        self._load_pool = QThreadPool(self)
        self._load_pool.setMaxThreadCount(1)
        self._process_serial = 0
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)
        self._process_timer = QTimer(self)
        self._process_timer.setSingleShot(True)
        self._process_timer.setInterval(PLAN_SETTLE_MS)
        self._process_timer.timeout.connect(self._start_processing)
        # A control that moves once wants its plan promptly; a keyboard wants
        # to be left alone until the sentence is finished.  The flag says a
        # recalculation is owed but has not been handed to a worker yet, which
        # is what keeps the statistics reading as pending rather than absent.
        self._plan_pending = False
        # A recalculation that came due while the Source tab was in front is
        # held here instead of run, so the busy overlay never lands on top of
        # an edit in progress; switching to the Rust preview runs it.
        self._plan_deferred = False
        self._settings_timer = QTimer(self)
        self._settings_timer.setSingleShot(True)
        self._settings_timer.setInterval(350)
        self._settings_timer.timeout.connect(self._save_settings)
        # A calibrated profile follows the game: this poll notices Rust's
        # window sitting on a different monitor than the calibrated boxes and
        # offers a one-click move.
        self._rust_monitor_timer = QTimer(self)
        self._rust_monitor_timer.setInterval(5000)
        self._rust_monitor_timer.timeout.connect(self._check_rust_monitor)
        self._timelapse_recorder: Any = None
        self._run_report: Any = None
        # Where the running job has got to, on disk, so the job can be
        # picked up again after a server restart takes the sign away.
        self._resume_store: Any = None
        self._resume_record: ResumeRecord | None = None
        self._resume_record_written_at = 0.0
        self._resume_position: tuple[int, int] = (0, 0)
        # The plan on screen, fingerprinted once, and the simulation it
        # came with: the resume slider paints the first N strokes of it.
        self._plan_fingerprint: str | None = None
        self._plan_simulation: Image.Image | None = None
        self._resume_preview_timer = QTimer(self)
        self._resume_preview_timer.setSingleShot(True)
        self._resume_preview_timer.setInterval(60)
        self._resume_preview_timer.timeout.connect(self._refresh_resume_preview)
        # The screen as it was when a guard last paused the job: path,
        # reason, and time; and the viewers open on such screenshots.
        self._pause_screenshot: tuple[Path, str, str] | None = None
        self._offered_record: ResumeRecord | None = None
        # Which offer (plan, record) the tick and slider were last set from,
        # so re-planning the same picture leaves the user's choice alone.
        self._offered_key: tuple[str, str, int] | None = None
        self._screenshot_viewers: list[Any] = []
        self._paint_job_snapshot: Any = None
        self._timelapse_timer = QTimer(self)
        self._timelapse_timer.timeout.connect(self._capture_timelapse_frame)
        # The readout under the progress bar counts down to the next
        # anti-AFK break and the elapsed time, so it ticks on its own clock
        # rather than waiting for a progress update.
        self._active_detail = ""
        self._status_overlay_linger = QTimer(self)
        self._status_overlay_linger.setSingleShot(True)
        self._status_overlay_linger.setInterval(self._STATUS_OVERLAY_LINGER_MS)
        self._status_overlay_linger.timeout.connect(self._update_calibration_overlay)
        # While a job is paused the resume slider is lent out as a
        # viewfinder; what the notice said and where the slider stood are
        # kept so the resume offer comes back as it was.
        self._paused_viewfinder_notice: str | None = None
        self._resume_offer_value = 0
        self._active_detail_timer = QTimer(self)
        self._active_detail_timer.setInterval(1000)
        self._active_detail_timer.timeout.connect(self._refresh_active_detail)
        self._timelapse_export_pool = QThreadPool(self)
        self._timelapse_export_pool.setMaxThreadCount(1)
        self._timelapse_export: _TimelapseExportWorker | None = None
        # Players are modeless so a recording can be watched while the next
        # image is set up; the window keeps them alive and closes them with it.
        self._timelapse_players: list[Any] = []

        self._profile_store: Any = None
        self._settings_store: Any = None
        self._settings: dict[str, Any] = default_settings()
        self._current_profile: Any = None
        self._preview_correction: Any = None
        self._painter: Any = None
        self._paint_generation = 0
        self._pending_paint: _PendingPaint | None = None
        self._pending_start_cancelled = False
        self._color_chart_profile_id: str | None = None
        self._color_chart_path: Path | None = None
        self._hotkeys: Any = None
        self._hotkeys_ready = False
        self._last_hotkey_bindings: tuple[str, str, str] | None = None
        self._countdown: CountdownDialog | None = None
        self._countdown_callback_running = False
        self._debug_running = False
        self._debug_abort_event = threading.Event()
        self._debug_input_gate = threading.RLock()
        self._debug_controller: Any = None
        self._debug_thread: threading.Thread | None = None
        self._calibration_overlay: Any = None
        self._calibration_overlay_decision: tuple[Any, ...] | None = None
        self._calibration_preview: CalibrationPreviewOverlay | None = None
        self._applying_speed_preset = False
        # Seeded before the resolution controls exist, so this ratio is spelled
        # out against the default 256x128 canvas.
        self._text_layers = [
            _TextOverlayOptions("", "", 24, (255, 255, 255), size_ratio=24 / 128)
        ]
        self._selected_text_layer = 0
        # Which layers the side panel writes to; a canvas selection fills it
        # in, and an empty list falls back to the layer the combo box names.
        self._selected_text_indices: list[int] = [0]
        self._syncing_text_controls = False
        self._text_history: list[_TextSnapshot] = []
        self._text_history_index = 0
        self._text_history_kind = ""
        self._text_history_stamp = 0.0
        self._restoring_text_history = False
        # The decoded source at preview resolution, plus the size it is shown
        # at, which Stretch pulls away from the decoded one.
        self._source_pixmap = QPixmap()
        self._source_preview_size: tuple[int, int] | None = None
        # Where Fill anchors the region it keeps when the user has framed it
        # by hand; None follows the named crop alignment instead.
        self._crop_focus: tuple[float, float] | None = None
        self._last_named_crop = CropAlignment.CENTER.value
        # The Rust preview is only fronted once per imported image; afterwards
        # the user's tab choice is respected so text editing on the Source tab
        # is not interrupted by every reprocess.
        self._show_preview_after_processing = False
        self._plan_processing = False
        self._resolution_cap_note = ""
        self._closing = False
        self._painter_bridge = _PainterBridge()
        self._painter_bridge.progress.connect(self._on_paint_progress)
        self._painter_bridge.state.connect(self._on_paint_state)
        self._painter_bridge.completed.connect(self._on_paint_complete)
        self._painter_bridge.error.connect(self._on_paint_error)
        self._painter_bridge.start_requested.connect(self._start_or_resume)
        self._painter_bridge.pause_screenshot.connect(self._on_pause_screenshot)
        self._painter_bridge.pause_requested.connect(self._pause_painting)
        self._painter_bridge.abort_requested.connect(self._abort_painting)
        self._painter_bridge.hotkey_error.connect(self._on_hotkey_error)
        self._painter_bridge.debug_finished.connect(self._on_debug_finished)

        self._build_ui()
        self._connect_processing_controls()
        self._install_text_history_shortcuts()
        self._install_logging_handler()
        self._initialize_services()
        self._update_quality_dimensions()
        self._update_start_availability()
        self._reset_text_history()

        LOGGER.info("RustPainter started")

    # ------------------------------------------------------------------ UI

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 14, 18, 8)
        root_layout.setSpacing(12)
        self.page_stack = QStackedWidget()
        root_layout.addWidget(self._build_header())

        workspace = QWidget()
        workspace_layout = QHBoxLayout(workspace)
        workspace_layout.setContentsMargins(0, 0, 0, 0)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._wrap_scroll(self._build_profile_and_run_panel(), 490))
        splitter.addWidget(self._build_preview_area())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([500, 860])
        workspace_layout.addWidget(splitter)

        self.page_stack.addWidget(workspace)
        self.page_stack.addWidget(self._wrap_scroll(self._build_timelapse_page(), 0))
        self.page_stack.addWidget(self._build_settings_page())
        root_layout.addWidget(self.page_stack, 1)
        self.setCentralWidget(root)
        self.statusBar().showMessage("Ready")

        close_action = QAction("Close", self)
        close_action.setShortcut(QKeySequence.StandardKey.Quit)
        close_action.triggered.connect(self.close)
        self.addAction(close_action)

    def _build_header(self) -> QWidget:
        frame = QFrame()
        frame.setObjectName("appHeader")
        layout = QHBoxLayout(frame)
        layout.setContentsMargins(14, 9, 14, 9)
        layout.setSpacing(10)

        mark = QLabel()
        mark.setObjectName("appMark")
        icon = self.windowIcon()
        if not icon.isNull():
            mark.setPixmap(icon.pixmap(32, 32))
        else:
            mark.setText("RB")
        mark.setFixedSize(38, 38)
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title = QLabel("RustPainter")
        title.setObjectName("appTitle")

        self.workspace_nav_button = QPushButton("Workspace")
        self.workspace_nav_button.setObjectName("navButton")
        self.workspace_nav_button.setCheckable(True)
        self.workspace_nav_button.setAutoExclusive(True)
        self.workspace_nav_button.setChecked(True)
        self._set_icon(self.workspace_nav_button, "workspace", size=17)
        self.timelapse_nav_button = QPushButton("Timelapse")
        self.timelapse_nav_button.setObjectName("navButton")
        self.timelapse_nav_button.setCheckable(True)
        self.timelapse_nav_button.setAutoExclusive(True)
        self._set_icon(self.timelapse_nav_button, "clock", size=17)
        self.settings_nav_button = QPushButton("Settings")
        self.settings_nav_button.setObjectName("navButton")
        self.settings_nav_button.setCheckable(True)
        self.settings_nav_button.setAutoExclusive(True)
        self._set_icon(self.settings_nav_button, "settings", size=17)
        self.workspace_nav_button.clicked.connect(lambda: self.page_stack.setCurrentIndex(0))
        self.timelapse_nav_button.clicked.connect(self._show_timelapse_page)
        self.settings_nav_button.clicked.connect(lambda: self.page_stack.setCurrentIndex(2))

        self.state_badge_frame = QFrame()
        self.state_badge_frame.setObjectName("stateBadge")
        badge_layout = QHBoxLayout(self.state_badge_frame)
        badge_layout.setContentsMargins(11, 4, 13, 4)
        badge_layout.setSpacing(7)
        self.state_badge_icon = QLabel()
        self.state_badge_icon.setFixedSize(16, 16)
        self.state_badge_icon.setScaledContents(True)
        self.state_badge = QLabel()
        badge_layout.addWidget(self.state_badge_icon)
        badge_layout.addWidget(self.state_badge)
        self._set_state_badge("idle", "IDLE")

        layout.addWidget(mark)
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self.workspace_nav_button)
        layout.addWidget(self.timelapse_nav_button)
        layout.addWidget(self.settings_nav_button)
        layout.addSpacing(6)
        layout.addWidget(self.state_badge_frame)
        return frame

    def _set_state_badge(self, state: str, text: str) -> None:
        """Recolour the header badge and match its icon to the painter state."""

        self.state_badge.setText(text)
        self.state_badge_frame.setStyleSheet(state_badge_style(state))
        self.state_badge_icon.setPixmap(
            tinted_pixmap(_BADGE_ICONS.get(state, "status"), badge_foreground(state), 32)
        )

    @staticmethod
    def _set_icon(
        button: QPushButton, name: str, color: str | None = None, size: int = 18
    ) -> None:
        """Give a button one of the baked rust icons.

        The artwork already carries its own colour, so ``color`` is only used
        where an icon has to read as a different state (a green ready check,
        say) rather than as the default rust orange.
        """

        button.setIcon(art_icon(name, 64, color))
        button.setIconSize(QSize(size, size))

    @staticmethod
    def _step_panel(number: int, title: str) -> tuple[QFrame, QVBoxLayout]:
        """A numbered workflow card: badge, heading, then caller-owned content.

        Returns the frame to place in a layout together with the inner layout
        the step's controls go into.
        """

        frame = QFrame()
        frame.setObjectName("panel")
        outer = QVBoxLayout(frame)
        outer.setContentsMargins(13, 11, 13, 13)
        outer.setSpacing(9)

        heading = QHBoxLayout()
        heading.setSpacing(9)
        badge = QLabel(str(number))
        badge.setObjectName("stepBadge")
        badge.setFixedSize(20, 20)
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        label = QLabel(title.upper())
        label.setObjectName("panelTitle")
        heading.addWidget(badge)
        heading.addWidget(label)
        heading.addStretch(1)
        outer.addLayout(heading)

        body = QVBoxLayout()
        body.setSpacing(8)
        outer.addLayout(body)
        return frame, body

    @staticmethod
    def _wrap_scroll(content: QWidget, minimum_width: int) -> QScrollArea:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(minimum_width)
        content.setMinimumWidth(0)
        content.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        scroll.setWidget(content)
        return scroll

    def _build_image_settings(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Artwork")
        title.setObjectName("pageTitle")
        note = QLabel("Control how the source image is fitted and simplified for Rust.")
        note.setObjectName("muted")
        layout.addWidget(title)
        layout.addWidget(note)

        composition = QGroupBox("Background and transparency")
        form = QFormLayout(composition)
        self.background_combo = NoWheelComboBox()
        self.background_combo.addItem("Leave unpainted", "unpainted")
        self.background_combo.addItem("White", "white")
        self.background_combo.addItem("Black", "black")
        self.background_combo.addItem("Custom color", "custom")
        self.background_color_button = ColorButton("#ffffff")
        self.background_color_button.setEnabled(False)
        self.transparency_combo = NoWheelComboBox()
        self.transparency_combo.addItem("Leave unpainted", TransparencyMode.LEAVE_UNPAINTED.value)
        self.transparency_combo.addItem("Use background color", TransparencyMode.USE_BACKGROUND.value)
        self.alpha_fill_check = QCheckBox("Blend soft edges into the background")
        self.alpha_fill_check.setToolTip(
            "Painting has no transparency, so a half-transparent pixel has\n"
            "to become one solid color or none at all.  On, it is mixed\n"
            "into the background color, which suits artwork the sign really\n"
            "will carry that background behind.  Off, only the mostly\n"
            "opaque pixels are painted, in their own colors, so a cut-out\n"
            "subject is not ringed with a halo of background it never had."
        )
        form.addRow("Background fill", self.background_combo)
        form.addRow("Custom", self.background_color_button)
        form.addRow("Transparent pixels", self.transparency_combo)
        form.addRow("Alpha fill", self.alpha_fill_check)
        layout.addWidget(composition)

        strokes = QGroupBox("Strokes")
        form = QFormLayout(strokes)
        self.merge_combo = NoWheelComboBox()
        self.merge_combo.addItem("Off — exact strokes (slower, same picture)", "off")
        self.merge_combo.addItem("Balanced — small gaps", "balanced")
        self.merge_combo.addItem("Maximum — longest strokes", "maximum")
        self.merge_combo.addItem(
            "Automatic — handled by the optimizer", MERGE_MODE_OPTIMIZER
        )
        # The optimizer entry is a caption, not a choice: it is shown while
        # the box is disabled and hidden from the drop-down list.
        self.merge_combo.view().setRowHidden(
            self.merge_combo.findData(MERGE_MODE_OPTIMIZER), True
        )
        self._merge_mode_choice = "balanced"
        self._set_combo_data(self.merge_combo, "balanced")
        self.merge_combo.setToolTip(
            "Lets early colors paint straight through pixels that later colors\n"
            "repaint anyway. The finished image is identical, but fragmented\n"
            "areas need fewer strokes, so painting is faster.\n\n"
            "Only Exact paint mode uses this setting. Quality, Balanced and\n"
            "Fast merge automatically through the optimizer."
        )
        form.addRow("Stroke merging", self.merge_combo)
        layout.addWidget(strokes)

        layout.addStretch(1)
        return content

    def _build_paint_settings(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Painting")
        title.setObjectName("pageTitle")
        note = QLabel(
            "Tune brush behavior and input timing when a preset needs adjustment. "
            "Timing can also be changed while a job is paused, and takes effect "
            "when it resumes."
        )
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(note)

        speed_group = QGroupBox("Speed preset")
        speed_form = QFormLayout(speed_group)
        self.speed_preset_combo = NoWheelComboBox()
        self.speed_preset_combo.addItems([*SPEED_PRESETS.keys(), "Custom"])
        self.speed_preset_combo.setCurrentText("Standard")
        self.speed_preset_combo.setToolTip(
            "One-click timing profiles.  Every hold and settle has a floor of\n"
            "one game frame that no preset goes under, and long drags are\n"
            "capped at a rate the sign paints faithfully, so the presets differ\n"
            "in how much margin they add - not in whether the paint lands.\n"
            "Editing any value under Advanced Timing switches this to Custom."
        )
        speed_form.addRow("Preset", self.speed_preset_combo)
        speed_note = QLabel(
            "Turbo is the floors everywhere: as fast as Rust's 15 FPS paint UI "
            "can take input.  Every stroke still costs about a frame, so fewer "
            "strokes - the paint mode and merging - save more time than any "
            "preset."
        )
        speed_note.setWordWrap(True)
        speed_note.setObjectName("muted")
        speed_form.addRow("", speed_note)
        layout.addWidget(speed_group)

        advanced = QGroupBox("Advanced timing")
        advanced_layout = QFormLayout(advanced)
        self.pixel_spacing_spin = self._double_spin(0.25, 3.0, 1.0, 0.05, " ×")
        self.stroke_speed_spin = self._double_spin(10.0, 5000.0, 700.0, 10.0, " px/s")
        self.dot_duration_spin = self._int_spin(SPEED_FLOORS_MS["dot_ms"], 1000, 70, " ms")
        self.hue_delay_spin = self._int_spin(SPEED_FLOORS_MS["hue_ms"], 3000, 90, " ms")
        self.sv_delay_spin = self._int_spin(SPEED_FLOORS_MS["sv_ms"], 3000, 90, " ms")
        self.brush_delay_spin = self._int_spin(SPEED_FLOORS_MS["brush_ms"], 3000, 70, " ms")
        self.stroke_delay_spin = self._int_spin(
            SPEED_FLOORS_MS["stroke_ms"], 3000, 20, " ms"
        )
        self.color_delay_spin = self._int_spin(
            SPEED_FLOORS_MS["color_ms"], 5000, 120, " ms"
        )
        self.interpolation_spin = self._double_spin(1.0, 100.0, 4.0, 1.0, " px")
        self.stroke_speed_spin.setToolTip(
            "Cursor speed on a drag.  Runs of a few texels go at this speed\n"
            "whatever it is - the frame hold at their end is what lands them -\n"
            "and longer drags are capped at a rate the sign paints exactly,\n"
            "so past that cap this number changes nothing."
        )
        self.interpolation_spin.setToolTip(
            "Screen pixels between cursor events on a drag.  On a long drag\n"
            "the step is never wider than one sign texel, whatever is set here."
        )
        advanced_layout.addRow("Logical spacing", self.pixel_spacing_spin)
        advanced_layout.addRow("Stroke speed", self.stroke_speed_spin)
        advanced_layout.addRow("Dot hold", self.dot_duration_spin)
        advanced_layout.addRow("After hue", self.hue_delay_spin)
        advanced_layout.addRow("After S/V", self.sv_delay_spin)
        advanced_layout.addRow("After brush", self.brush_delay_spin)
        advanced_layout.addRow("Between strokes", self.stroke_delay_spin)
        self.line_tool_check = QCheckBox("Draw long straight runs with Shift-click lines")
        self.line_tool_check.setChecked(True)
        self.line_tool_check.setToolTip(
            "Rust's line tool: an anchor press, then a click with Shift held,\n"
            "and the game draws the straight stroke between them - a full row\n"
            "in two presses instead of a rate-capped drag.  Proven on each\n"
            "sign with one probe stroke before painting; a sign that fails\n"
            "the probe paints with drags exactly as before."
        )
        advanced_layout.addRow("Between colors", self.color_delay_spin)
        advanced_layout.addRow("Interpolation step", self.interpolation_spin)
        self.press_hold_check = QCheckBox("Measure this sign's timing floors")
        self.press_hold_check.setChecked(True)
        self.press_hold_check.setToolTip(
            "Before painting, probe dots and probe drags measure the floors the\n"
            "presets can never go under: the shortest press hold that lands\n"
            "every dab, the shortest gap between strokes the game keeps apart,\n"
            "and the fastest long drag it paints whole.  The job then runs at\n"
            "what this sign proved - the only speed past Turbo there is.  A\n"
            "sign that fails a probe keeps that floor; drags keep their dwell."
        )
        self.dab_size_check = QCheckBox("Prove the one-cell brush with lone dabs")
        self.dab_size_check.setChecked(True)
        self.dab_size_check.setToolTip(
            "Before painting, batches of lone dabs at rising Size numbers find\n"
            "the smallest brush that lands every one of them on this sign, and\n"
            "the job's single-cell strokes use it.  On large signs Rust's\n"
            "smallest brush is narrower than a texel and a lone dab can miss\n"
            "its texel entirely - the specks a finished XXL sign shows."
        )
        advanced_layout.addRow("", self.line_tool_check)
        advanced_layout.addRow("", self.press_hold_check)
        advanced_layout.addRow("", self.dab_size_check)
        layout.addWidget(advanced)

        touch_up = QGroupBox("Touch-up")
        touch_up_form = QFormLayout(touch_up)
        self.confirm_strokes_check = QCheckBox("Check each color as it goes down")
        self.confirm_strokes_check.setChecked(False)
        self.confirm_strokes_check.setToolTip(
            "After a color's strokes are painted, the sign is captured and the\n"
            "cells that did not take the color are repainted while it is still\n"
            "selected.  Off by default: the game does not drop presses, and on\n"
            "a fine sign the check can misread painted cells as missing and\n"
            "spend its rounds repainting them.  The touch-up pass at the end\n"
            "is what puts right whatever did go wrong."
        )
        self.verify_picks_check = QCheckBox("Read each color back before painting it")
        self.verify_picks_check.setChecked(True)
        self.verify_picks_check.setToolTip(
            "After the two picker clicks that select a color, the block beside\n"
            "the hue bar that shows the selected color is captured and compared\n"
            "with the color asked for; the clicks are repeated, held longer,\n"
            "until it agrees.  A click the game swallows otherwise paints the\n"
            "whole color group in the previous group's color - 43 of 240 groups\n"
            "on one 1024x512 sign.  Costs a few hundredths of a second a color."
        )
        self.confirm_rounds_spin = self._int_spin(1, 8, 4, "")
        self.confirm_rounds_spin.setToolTip(
            "How many times one color may be captured and its misses\n"
            "repainted before the job moves on.  Each round costs under a\n"
            "second plus the repaint; what is left goes to the touch-up pass."
        )
        touch_up_form.addRow("", self.verify_picks_check)
        touch_up_form.addRow("", self.confirm_strokes_check)
        touch_up_form.addRow("Rounds per color", self.confirm_rounds_spin)
        self.confirm_strokes_check.toggled.connect(self.confirm_rounds_spin.setEnabled)
        self.verify_passes_spin = self._int_spin(0, 5, 2, "")
        self.verify_passes_spin.setToolTip(
            "After the artwork is down, the sign is captured and compared with\n"
            "the plan, and cells that stayed bare or took the wrong color are\n"
            "repainted.  Each pass is one capture and repaint; the next pass\n"
            "checks the previous one, since the game can drop a touch-up\n"
            "stroke exactly as it dropped the original.  0 turns it off."
        )
        touch_up_form.addRow("Passes at the end", self.verify_passes_spin)
        touch_up_note = QLabel(
            "Checking each color catches the presses the game never saw, while "
            "the color is still selected.  The passes at the end read the whole "
            "sign back once more; on a plan finer than Rust's smallest brush "
            "they fill holes only."
        )
        touch_up_note.setWordWrap(True)
        touch_up_note.setObjectName("muted")
        touch_up_form.addRow("", touch_up_note)
        layout.addWidget(touch_up)

        layout.addStretch(1)
        return content

    def _build_timelapse_page(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        heading = QHBoxLayout()
        heading.setSpacing(10)
        title = QLabel("Timelapse")
        title.setObjectName("pageTitle")
        self.timelapse_status_badge = QLabel("Not recording")
        self.timelapse_status_badge.setObjectName("muted")
        heading.addWidget(_glyph_label("clock", 22))
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(self.timelapse_status_badge)
        layout.addLayout(heading)

        # One compact row of controls beats three rows of form labels: what
        # each one does is short enough to be the control's own label, and the
        # detail that used to sit in paragraphs is a tooltip away.
        capture_group = QGroupBox("Capture")
        capture_layout = QVBoxLayout(capture_group)
        capture_layout.setSpacing(9)
        capture_row = QHBoxLayout()
        capture_row.setSpacing(10)
        self.timelapse_check = QCheckBox("Record while painting")
        self.timelapse_check.setToolTip(
            "Saves a screenshot of the calibrated canvas at a regular interval\n"
            "while a job paints, so the finished frames can be assembled into\n"
            "a timelapse video.  Recording follows the job: it starts when\n"
            "painting starts, skips paused time, and stops when the job ends."
        )
        self.timelapse_interval_spin = self._int_spin(1, 600, 10, " s")
        self.timelapse_interval_spin.setToolTip(
            "How often a frame is captured. Painting a large sign can take an\n"
            "hour, so a frame every 10 seconds is usually plenty."
        )
        self.timelapse_final_check = QCheckBox("Frame at the finish")
        self.timelapse_final_check.setChecked(True)
        self.timelapse_final_check.setToolTip(
            "The interval rarely lands on the last stroke, so the finished sign\n"
            "gets one extra frame of its own."
        )
        capture_row.addWidget(self.timelapse_check)
        capture_row.addStretch(1)
        capture_row.addWidget(_glyph_label("clock", 16))
        capture_row.addWidget(QLabel("Every"))
        capture_row.addWidget(self.timelapse_interval_spin)
        capture_row.addWidget(self.timelapse_final_check)
        capture_layout.addLayout(capture_row)
        capture_note = QLabel(
            "Frames cover the calibrated canvas, so the sign must be calibrated "
            "in Rust setup."
        )
        capture_note.setWordWrap(True)
        capture_note.setObjectName("muted")
        capture_layout.addWidget(capture_note)
        layout.addWidget(capture_group)

        sessions_group = QGroupBox("Recordings")
        sessions_layout = QVBoxLayout(sessions_group)
        sessions_layout.setSpacing(9)
        self.timelapse_sessions = QListWidget()
        self.timelapse_sessions.setAlternatingRowColors(True)
        self.timelapse_sessions.setMinimumHeight(180)
        # Housekeeping is the common reason to come here, and housekeeping is
        # done in batches: shift-click takes a run of old recordings,
        # ctrl-click picks them out one by one, and Delete clears the lot.
        self.timelapse_sessions.setSelectionMode(
            QAbstractItemView.SelectionMode.ExtendedSelection
        )
        self.timelapse_sessions.setToolTip(
            "Every job keeps its own folder of numbered PNG frames.  Saving a "
            "video leaves that folder untouched.\n"
            "Shift-click for a run, ctrl-click to pick, Delete to remove them."
        )
        sessions_layout.addWidget(self.timelapse_sessions)
        self.timelapse_selection_label = QLabel("")
        self.timelapse_selection_label.setObjectName("muted")
        sessions_layout.addWidget(self.timelapse_selection_label)

        playback_row = QHBoxLayout()
        playback_row.setSpacing(8)
        self.play_session_button = QPushButton("Watch")
        self.play_session_button.setObjectName("accent")
        self._set_icon(self.play_session_button, "play", ON_ACCENT, size=16)
        self.play_session_button.setToolTip(
            "Play the selected recording back inside RustPainter."
        )
        self.export_session_button = QPushButton("Save video")
        self._set_icon(self.export_session_button, "drag-drop", size=16)
        self.export_session_button.setToolTip(
            "Write the selected recording to a single video file you can keep, "
            "upload, or share."
        )
        self.timelapse_format_combo = NoWheelComboBox()
        for video_format in available_formats():
            self.timelapse_format_combo.addItem(video_format.label, video_format.key)
        self.timelapse_format_combo.setToolTip(
            "Container for the exported video. AVI and GIF are written by "
            "RustPainter itself; MP4 is offered when ffmpeg is installed."
        )
        playback_row.addWidget(self.play_session_button)
        playback_row.addWidget(self.export_session_button)
        playback_row.addWidget(self.timelapse_format_combo)
        playback_row.addStretch(1)
        # Icon-only buttons for the housekeeping actions: they are used rarely,
        # and their labels were most of the text on the page.
        self.open_timelapse_button = self._icon_button(
            "drag-drop", "Open the timelapse folder"
        )
        self.open_session_button = self._icon_button(
            "workspace", "Open the selected recording's folder"
        )
        self.refresh_sessions_button = self._icon_button(
            "status", "Look for recordings again"
        )
        self.delete_session_button = self._icon_button(
            "trash",
            "Delete the selected recording and every frame in it",
            color=DANGER,
        )
        for button in (
            self.open_timelapse_button,
            self.open_session_button,
            self.refresh_sessions_button,
            self.delete_session_button,
        ):
            playback_row.addWidget(button)
        sessions_layout.addLayout(playback_row)

        # A slider says "how fast" without asking anyone to think in frames
        # per second first; the readout still names the number, because that
        # is what the exported file is written at.
        speed_row = QHBoxLayout()
        speed_row.setSpacing(9)
        speed_row.addWidget(_glyph_label("sliders", 16))
        speed_row.addWidget(QLabel("Speed"))
        self.timelapse_speed_slider = QSlider(Qt.Orientation.Horizontal)
        self.timelapse_speed_slider.setRange(MIN_FRAME_RATE, MAX_FRAME_RATE)
        self.timelapse_speed_slider.setValue(DEFAULT_FRAME_RATE)
        self.timelapse_speed_slider.setPageStep(5)
        self.timelapse_speed_slider.setToolTip(
            "How fast the recording plays, and the frame rate the saved video "
            "is written at.  A sign painted over an hour is worth watching "
            "faster than it happened."
        )
        self.timelapse_speed_label = QLabel("")
        self.timelapse_speed_label.setObjectName("muted")
        self.timelapse_speed_label.setMinimumWidth(150)
        self.timelapse_speed_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        speed_row.addWidget(QLabel("Slow"))
        speed_row.addWidget(self.timelapse_speed_slider, 1)
        speed_row.addWidget(QLabel("Fast"))
        speed_row.addWidget(self.timelapse_speed_label)
        sessions_layout.addLayout(speed_row)

        self.timelapse_export_progress = QProgressBar()
        self.timelapse_export_progress.setVisible(False)
        self.timelapse_export_progress.setTextVisible(True)
        sessions_layout.addWidget(self.timelapse_export_progress)
        layout.addWidget(sessions_group, 1)
        self._refresh_timelapse_speed_label()
        return content

    def _icon_button(
        self, icon_name: str, tooltip: str, color: str | None = None
    ) -> QPushButton:
        """A square, label-free button; its tooltip carries the whole meaning.

        The accessible name repeats the tooltip because it is the only text
        the button has, and a screen reader has nothing else to announce.
        """

        button = QPushButton()
        button.setObjectName("iconButton")
        button.setToolTip(tooltip)
        button.setAccessibleName(tooltip)
        button.setFixedSize(30, 30)
        self._set_icon(button, icon_name, color, size=16)
        return button

    def _refresh_timelapse_speed_label(self, *_args: Any) -> None:
        """Say what the slider means in both of the units that matter."""

        rate = self.timelapse_speed_slider.value()
        interval = max(1, self.timelapse_interval_spin.value())
        # One second of video covers this much of the paint job.
        covered = self._format_duration(rate * interval)
        self.timelapse_speed_label.setText(f"{rate} fps  •  {covered} per second")

    def _build_preview_area(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 4, 4, 4)
        layout.setSpacing(12)

        heading = QHBoxLayout()
        title = QLabel("PREVIEW")
        title.setObjectName("pageTitle")
        # The hint follows the tab and the scaling mode, because the one
        # gesture worth advertising is different on each of them.
        self.preview_hint_label = QLabel(PREVIEW_HINTS["source"])
        self.preview_hint_label.setObjectName("muted")
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(self.preview_hint_label)
        # The plan has one color per cell, and the tab has to blow those cells
        # up to fill itself.  Rust filters the sign's texture when it draws
        # it, so filtered scaling is the honest guess at the finished sign;
        # hard-edged cells exaggerate the grid but show exactly what each
        # stroke will paint.  Both views have a use, so it is a switch.
        self.smooth_preview_check = QCheckBox("Smooth")
        self.smooth_preview_check.setChecked(True)
        self.smooth_preview_check.setToolTip(
            "Scale the Rust preview the way the game draws the sign's texture, "
            "blending between cells.  Off shows every planned cell as a hard "
            "square, closer to the painter's own view of the plan."
        )
        heading.addWidget(self.smooth_preview_check)
        # What the painter promises to put on the sign, as a file: the only
        # way to hold it next to a screenshot of what it actually put there.
        self.export_preview_button = self._icon_button(
            "drag-drop", "Save the Rust preview as an image file"
        )
        self.export_preview_button.setEnabled(False)
        heading.addWidget(self.export_preview_button)
        layout.addLayout(heading)

        tabs = QTabWidget()
        browse_hint = "Click here or drop an image"
        self.original_preview = TextEditorPreview(
            "Browse an image to begin", smooth=True, hint=browse_hint
        )
        self.original_preview.setToolTip(
            "Drag text to move it, drag its handles to resize it, and "
            "double-click to edit it. Shift+drag a box across bare canvas, "
            "Ctrl+click or Ctrl+A to take several layers at once; the arrow "
            "keys nudge them, Ctrl+D or Ctrl+C copies them, and Delete "
            "removes them. Dragging snaps to the sign and to the other "
            "layers unless Alt is held. The bracketed border is the part of "
            "the image the sign will show - under Fill, drag bare canvas to "
            "move it onto what you want the sign to keep."
        )
        self.paint_preview = PreviewLabel(
            "Paint simulation will appear here",
            smooth=self.smooth_preview_check.isChecked(),
            hint=browse_hint,
            read_only_chip="Preview only",
        )
        self.smooth_preview_check.toggled.connect(self.paint_preview.set_smooth)
        self.smooth_preview_check.toggled.connect(self._schedule_settings_save)
        self.paint_preview.setToolTip(
            "Exactly what the painter will put on the sign - text is baked in "
            "and every color is palette-limited.  Editing happens on the "
            "Source tab; this one only shows the result."
        )
        # Trying to edit here is a reasonable mistake - the two tabs show the
        # same artwork - so it is answered in place, with the tab that does
        # take the edit one click away rather than a dialog to dismiss.
        self.preview_notice = InlineNotice(self.paint_preview, icon_name="pencil")
        self.preview_notice.actionTriggered.connect(self._show_source_tab)
        self.paint_preview.editAttempted.connect(self._on_read_only_edit_attempt)
        self.paint_preview.editElsewhereRequested.connect(self._show_source_tab)
        for preview in (self.original_preview, self.paint_preview):
            preview.browseRequested.connect(self._browse_image)
            preview.imageDropped.connect(
                lambda dropped: self.load_image(Path(dropped))
            )
        tabs.addTab(self.original_preview, "Source")
        tabs.addTab(self.paint_preview, "Rust preview")
        tabs.setTabToolTip(0, "The image as imported - move and edit text here")
        tabs.setTabToolTip(1, "The finished sign, read-only")
        tabs.currentChanged.connect(self._on_preview_tab_changed)
        self.preview_tabs = tabs
        # Recalculating covers the artwork it is recalculating, so the answer
        # to "is anything happening?" is where the user is already looking.
        # The tab bar stays uncovered so either tab is still reachable.
        self.plan_busy = BusyOverlay(tabs)
        layout.addWidget(tabs, 1)

        analysis = QFrame()
        analysis.setObjectName("panel")
        grid = QGridLayout(analysis)
        grid.setContentsMargins(13, 11, 13, 13)
        grid.setHorizontalSpacing(10)
        grid.setVerticalSpacing(9)
        analysis_head = QHBoxLayout()
        analysis_head.setSpacing(8)
        title = QLabel("PAINT PLAN")
        title.setObjectName("pageTitle")
        self.processing_spinner = Spinner(16)
        self.processing_spinner.setToolTip("Recalculating the paint plan…")
        analysis_head.addWidget(title)
        analysis_head.addWidget(self.processing_spinner)
        analysis_head.addStretch(1)
        grid.addLayout(analysis_head, 0, 0, 1, 4)
        self.analysis_resolution = self._metric("Resolution", "—", "resolution")
        self.analysis_colors = self._metric("Colors", "—", "palette")
        self.analysis_strokes = self._metric("Strokes", "—", "brush")
        self.analysis_time = self._metric("Est. time", "—", "clock")
        for column, widget in enumerate(
            (
                self.analysis_resolution,
                self.analysis_colors,
                self.analysis_strokes,
                self.analysis_time,
            )
        ):
            grid.addWidget(widget, 1, column)
        self.processing_label = QLabel("")
        self.processing_label.setObjectName("muted")
        grid.addWidget(self.processing_label, 2, 0, 1, 4)

        # While a job runs, the plan panel makes way for an enlarged progress
        # readout in the same spot; the small always-on strip below is hidden
        # so the numbers are not shown twice.
        active = QFrame()
        active.setObjectName("panel")
        active_layout = QVBoxLayout(active)
        active_layout.setContentsMargins(13, 11, 13, 13)
        active_layout.setSpacing(8)
        active_head = QHBoxLayout()
        active_head.setSpacing(8)
        self.active_progress_title = QLabel("PAINTING")
        self.active_progress_title.setObjectName("pageTitle")
        self.active_progress_state = QLabel("Starting…")
        self.active_progress_state.setObjectName("muted")
        active_head.addWidget(self.active_progress_title)
        active_head.addStretch(1)
        active_head.addWidget(self.active_progress_state)
        active_layout.addLayout(active_head)
        active_numbers = QHBoxLayout()
        active_numbers.setSpacing(14)
        self.active_percent_label = QLabel("0%")
        self.active_percent_label.setStyleSheet(
            "font-size: 34pt; font-weight: 800;"
        )
        self.active_remaining_label = QLabel("Estimating time left…")
        self.active_remaining_label.setStyleSheet(
            "font-size: 14pt; font-weight: 600;"
        )
        active_numbers.addWidget(self.active_percent_label)
        active_numbers.addStretch(1)
        active_numbers.addWidget(
            self.active_remaining_label,
            alignment=Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
        )
        active_layout.addLayout(active_numbers)
        self.active_paint_progress = QProgressBar()
        self.active_paint_progress.setRange(0, 1000)
        self.active_paint_progress.setValue(0)
        self.active_paint_progress.setMinimumHeight(22)
        self.active_paint_progress.setTextVisible(False)
        active_layout.addWidget(self.active_paint_progress)
        self.active_detail_label = QLabel("")
        self.active_detail_label.setObjectName("muted")
        active_layout.addWidget(self.active_detail_label)
        # A pause the painter called on its own comes with a screenshot of
        # the screen at that moment; this opens it.
        self.pause_screenshot_button = QPushButton("See what stopped it")
        self.pause_screenshot_button.setObjectName("compactButton")
        self.pause_screenshot_button.setToolTip(
            "Open the screenshot the app took of the whole screen the moment\n"
            "a guard paused the job, with the painter's reason above it."
        )
        self.pause_screenshot_button.setVisible(False)
        self.pause_screenshot_button.clicked.connect(self._show_pause_screenshot)
        active_layout.addWidget(
            self.pause_screenshot_button, 0, Qt.AlignmentFlag.AlignLeft
        )

        self.plan_progress_stack = QStackedWidget()
        self.plan_progress_stack.addWidget(analysis)
        self.plan_progress_stack.addWidget(active)
        layout.addWidget(self.plan_progress_stack)

        progress_frame = QFrame()
        progress_frame.setObjectName("panel")
        progress_layout = QVBoxLayout(progress_frame)
        progress_layout.setContentsMargins(13, 11, 13, 12)
        progress_head = QHBoxLayout()
        progress_head.setSpacing(8)
        progress_head.addWidget(_glyph_label("status", 20))
        self.progress_state_label = QLabel("Idle")
        self.progress_detail_label = QLabel("No active paint job")
        self.progress_detail_label.setObjectName("muted")
        progress_head.addWidget(self.progress_state_label)
        progress_head.addStretch(1)
        progress_head.addWidget(self.progress_detail_label)
        self.paint_progress = QProgressBar()
        self.paint_progress.setRange(0, 1000)
        self.paint_progress.setValue(0)
        progress_layout.addLayout(progress_head)
        progress_layout.addWidget(self.paint_progress)
        self.progress_frame = progress_frame
        layout.addWidget(progress_frame)

        return content

    @Slot()
    def _show_source_tab(self) -> None:
        self.preview_tabs.setCurrentIndex(0)
        self.original_preview.setFocus(Qt.FocusReason.OtherFocusReason)

    @Slot()
    def _on_read_only_edit_attempt(self) -> None:
        """Answer an edit aimed at the Rust preview without stopping the user."""

        self.preview_notice.show_message(
            "Read-only preview - the artwork is edited on the Source tab.",
            "Go to Source",
        )
        self.statusBar().showMessage(
            "The Rust preview is read-only; text and crop are edited on the "
            "Source tab",
            5000,
        )

    @Slot()
    def _on_preview_tab_changed(self, *_args: Any) -> None:
        self.preview_notice.hide()
        self._refresh_preview_hint()
        self._refresh_text_section_visibility()
        if self.preview_tabs.currentIndex() == 1 and self._plan_deferred:
            # A recalculation held back while the Source tab was being edited
            # is owed now that its result is the tab in front.
            self._plan_deferred = False
            self._schedule_processing()

    def _refresh_text_section_visibility(self) -> None:
        """Show the text controls with the tab they edit, the Source tab."""

        self.text_section.setVisible(self.preview_tabs.currentIndex() == 0)

    def _refresh_preview_hint(self) -> None:
        """Name the one gesture that matters on whichever tab is in front."""

        if self.preview_tabs.currentIndex() == 1:
            key = "rust"
        elif self.original_preview.can_pan_crop():
            key = "crop"
        else:
            key = "source"
        self.preview_hint_label.setText(PREVIEW_HINTS[key])

    def _set_active_progress_visible(self, active: bool) -> None:
        """Swap the plan panel for the enlarged progress readout while painting."""

        self.plan_progress_stack.setCurrentIndex(1 if active else 0)
        self.progress_frame.setVisible(not active)
        if active:
            self._active_detail_timer.start()
        else:
            self._active_detail_timer.stop()

    def _build_profile_and_run_panel(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 8, 4)
        layout.setSpacing(12)

        image_group, image_layout = self._step_panel(1, "Image")
        self.browse_button = QPushButton("Choose image")
        self.browse_button.setObjectName("accent")
        self.browse_button.setMinimumHeight(36)
        self._set_icon(self.browse_button, "choose-image", ON_ACCENT, size=20)
        self.browse_button.clicked.connect(self._browse_image)
        image_info = QHBoxLayout()
        self.image_name_label = QLabel("No image selected")
        self.image_name_label.setWordWrap(True)
        self.image_dimensions_label = QLabel("—")
        self.image_dimensions_label.setObjectName("muted")
        image_info.addWidget(self.image_name_label, 1)
        image_info.addWidget(self.image_dimensions_label)
        image_layout.addWidget(self.browse_button)
        image_layout.addLayout(image_info)

        quick_title = QLabel("Quick settings")
        quick_title.setObjectName("sectionTitle")
        image_layout.addWidget(quick_title)
        quick_grid = QGridLayout()
        quick_grid.setContentsMargins(0, 0, 0, 0)
        quick_grid.setHorizontalSpacing(10)
        quick_grid.setVerticalSpacing(5)
        self.scale_mode_combo = NoWheelComboBox()
        self.scale_mode_combo.addItem("Fit — show entire image", ScaleMode.FIT.value)
        self.scale_mode_combo.addItem("Fill / Crop", ScaleMode.FILL.value)
        self.scale_mode_combo.addItem("Stretch", ScaleMode.STRETCH.value)
        self.crop_alignment_combo = NoWheelComboBox()
        for label, value in (
            ("Center", CropAlignment.CENTER),
            ("Top", CropAlignment.TOP),
            ("Bottom", CropAlignment.BOTTOM),
            ("Left", CropAlignment.LEFT),
            ("Right", CropAlignment.RIGHT),
        ):
            self.crop_alignment_combo.addItem(label, value.value)
        # A dragged crop lands between the five named anchors, so it gets an
        # entry of its own rather than being rounded to the nearest one.
        self.crop_alignment_combo.addItem(CUSTOM_CROP_LABEL, CUSTOM_CROP_VALUE)
        self.crop_alignment_combo.setToolTip(
            "Which part of the image Fill keeps.  Dragging the image on the "
            "Source tab reframes it freely and switches this to Custom; "
            "picking a named anchor again puts the crop back on it."
        )
        self.quality_combo = NoWheelComboBox()
        self.quality_combo.addItems(
            [*QUALITY_LONG_EDGE.keys(), MAX_QUALITY_PRESET, "Custom"]
        )
        self.quality_combo.setCurrentText("Balanced")
        self.paint_mode_combo = NoWheelComboBox()
        self.paint_mode_combo.addItem("Exact — raw pixels", PaintMode.EXACT.value)
        self.paint_mode_combo.addItem("Quality — subtle cleanup", PaintMode.QUALITY.value)
        self.paint_mode_combo.addItem("Balanced — recommended", PaintMode.BALANCED.value)
        self.paint_mode_combo.addItem("Fast — biggest savings", PaintMode.FAST.value)
        self._set_combo_data(self.paint_mode_combo, PaintMode.BALANCED.value)
        self.paint_mode_combo.setToolTip(
            "How boldly planning may simplify the image to paint faster.\n"
            "Exact reproduces every quantized pixel with the classic plan.\n"
            "The optimized modes merge indistinguishable colors, absorb\n"
            "insignificant specks, and — with automatic brush sizing\n"
            "calibrated — fill large areas with a big brush first and\n"
            "paint the details over them."
        )
        # Everything that changes the picture lives here, beside the preview
        # that shows the change; the settings pages hold what changes only
        # how it is painted.
        self.color_count_combo = NoWheelComboBox()
        for value in (8, 16, 24, 32, 48, 64, 96, 128, 256):
            self.color_count_combo.addItem(str(value), value)
        self._set_combo_data(self.color_count_combo, DEFAULT_COLOR_COUNT)
        self.color_count_combo.setToolTip(
            "The most colors the plan may use.  Fewer colors mean fewer\n"
            "picker trips and fewer strokes; more keep gradients and shading."
        )
        self.sharpen_combo = NoWheelComboBox()
        self.sharpen_combo.addItem("Off", SharpenMode.OFF.value)
        self.sharpen_combo.addItem("Light — recommended", SharpenMode.LIGHT.value)
        self.sharpen_combo.addItem("Strong — for line art", SharpenMode.STRONG.value)
        self._set_combo_data(self.sharpen_combo, SharpenMode.LIGHT.value)
        self.sharpen_combo.setToolTip(
            "Rust draws the sign's texture with a filter that softens every\n"
            "edge, so an image shrunk to the sign looks blurrier in game than\n"
            "it did on screen.  Sharpening before painting puts back about\n"
            "the contrast that filter takes away.  Light suits nearly\n"
            "everything; Strong makes line art bite at the price of a faint\n"
            "halo beside dark lines.  Images that are enlarged rather than\n"
            "shrunk are never sharpened.  The change is a few levels per\n"
            "cell - compare with Smooth off on the Rust preview, or watch\n"
            "the stroke count."
        )
        for combo in (
            self.scale_mode_combo,
            self.quality_combo,
            self.crop_alignment_combo,
            self.paint_mode_combo,
            self.color_count_combo,
            self.sharpen_combo,
        ):
            combo.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            combo.setMinimumContentsLength(12)
            combo.setSizePolicy(
                QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
            )
        self.dither_check = QCheckBox("Dithering")
        self.dither_check.setToolTip(
            "Dithering improves gradients but usually creates many more strokes."
        )
        self.remove_background_check = QCheckBox("Remove background")
        self.remove_background_check.setToolTip(
            "Leave the backdrop unpainted so Rust only paints the subject.\n"
            "An even backdrop — a white product shot, a flat logo field — is\n"
            "usually most of the strokes, so skipping it saves a lot of time."
        )
        for column, (label, control) in enumerate(
            (("Scaling", self.scale_mode_combo), ("Quality", self.quality_combo))
        ):
            quick_grid.addWidget(QLabel(label), 0, column)
            quick_grid.addWidget(control, 1, column)
        for column, (label, control) in enumerate(
            (
                ("Crop alignment", self.crop_alignment_combo),
                ("Optimization", self.paint_mode_combo),
            )
        ):
            quick_grid.addWidget(QLabel(label), 2, column)
            quick_grid.addWidget(control, 3, column)
        for column, (label, control) in enumerate(
            (("Colors", self.color_count_combo), ("Sharpen", self.sharpen_combo))
        ):
            quick_grid.addWidget(QLabel(label), 4, column)
            quick_grid.addWidget(control, 5, column)
        for column, checkbox in enumerate(
            (self.remove_background_check, self.dither_check)
        ):
            quick_grid.addWidget(
                checkbox,
                6,
                column,
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )

        # Why the quality presets stop making a difference.  Three of them
        # ask for more cells than a small sign has texels, and the plan is
        # held at the sign's own resolution - which without a word of
        # explanation looks like the setting doing nothing at all.
        self.resolution_cap_panel = QFrame()
        self.resolution_cap_panel.setObjectName("inlinePanel")
        cap_layout = QHBoxLayout(self.resolution_cap_panel)
        cap_layout.setContentsMargins(10, 7, 10, 7)
        cap_layout.setSpacing(8)
        self.resolution_cap_icon = _glyph_label("resolution", 16)
        self.resolution_cap_label = QLabel()
        self.resolution_cap_label.setObjectName("muted")
        self.resolution_cap_label.setWordWrap(True)
        cap_layout.addWidget(self.resolution_cap_icon, 0, Qt.AlignmentFlag.AlignTop)
        cap_layout.addWidget(self.resolution_cap_label, 1)
        self.resolution_cap_panel.setVisible(False)
        quick_grid.addWidget(self.resolution_cap_panel, 7, 0, 1, 2)

        self.custom_resolution_panel = QFrame()
        self.custom_resolution_panel.setObjectName("inlinePanel")
        custom_layout = QHBoxLayout(self.custom_resolution_panel)
        custom_layout.setContentsMargins(10, 7, 10, 7)
        custom_layout.setSpacing(8)
        custom_layout.addWidget(QLabel("Custom resolution"))
        self.logical_width_spin = NoWheelSpinBox()
        self.logical_width_spin.setRange(8, 2048)
        self.logical_width_spin.setValue(256)
        # Applied on Enter or focus-out, not per keystroke: the other axis is
        # re-derived from whatever is applied and written back into both
        # boxes, and doing that after the second digit of "1024" rewrote the
        # box under the user's fingers.
        self.logical_width_spin.setKeyboardTracking(False)
        self.logical_width_spin.setToolTip(
            "Logical width. Height follows the calibrated canvas aspect ratio."
        )
        self.logical_height_spin = NoWheelSpinBox()
        self.logical_height_spin.setRange(8, 2048)
        self.logical_height_spin.setValue(128)
        self.logical_height_spin.setKeyboardTracking(False)
        self.logical_height_spin.setToolTip(
            "Logical height. Width follows the calibrated canvas aspect ratio."
        )
        custom_layout.addStretch(1)
        custom_layout.addWidget(self.logical_width_spin)
        custom_layout.addWidget(QLabel("×"))
        custom_layout.addWidget(self.logical_height_spin)
        quick_grid.addWidget(self.custom_resolution_panel, 8, 0, 1, 2)

        self.background_removal_panel = QFrame()
        self.background_removal_panel.setObjectName("inlinePanel")
        removal_grid = QGridLayout(self.background_removal_panel)
        removal_grid.setContentsMargins(10, 8, 10, 8)
        removal_grid.setHorizontalSpacing(8)
        removal_grid.setVerticalSpacing(6)
        self.removal_source_combo = NoWheelComboBox()
        self.removal_source_combo.addItem("Detect from the edges", "auto")
        self.removal_source_combo.addItem("Pick a color", "custom")
        self.removal_source_combo.setToolTip(
            "Detection votes on the colors ringing the artwork, which suits a\n"
            "plain backdrop. Pick a color when the subject reaches the edges."
        )
        self.removal_color_button = ColorButton(
            "#ffffff", dialog_title="Choose the background color to skip"
        )
        self.removal_color_button.setEnabled(False)
        self.removal_tolerance_spin = NoWheelSpinBox()
        self.removal_tolerance_spin.setRange(0, 100)
        self.removal_tolerance_spin.setValue(12)
        self.removal_tolerance_spin.setSuffix(" %")
        self.removal_tolerance_spin.setToolTip(
            "How far a pixel may drift from the background color and still be\n"
            "skipped. Raise it for photos and gradients; lower it when part of\n"
            "the subject starts disappearing."
        )
        self.removal_scope_combo = NoWheelComboBox()
        self.removal_scope_combo.addItem(
            "Smart - find the subject", BackgroundRemovalScope.SUBJECT.value
        )
        self.removal_scope_combo.addItem(
            "Touching the edges", BackgroundRemovalScope.CONNECTED.value
        )
        self.removal_scope_combo.addItem(
            "Anywhere in the image", BackgroundRemovalScope.EVERYWHERE.value
        )
        self.removal_scope_combo.setToolTip(
            "Smart reads several colors off a band around the artwork and\n"
            "grows a strict match outwards through a looser one, so a\n"
            "gradient, a vignette or a photographic backdrop comes away in\n"
            "one piece and no halo is left around the subject.  It reaches\n"
            "further than the tolerance alone says, so lower the tolerance\n"
            "rather than raising it if the subject starts going too.\n\n"
            "Touching the edges matches one flat color and keeps enclosed\n"
            "areas - the hole in an O, a white eye - painted.  Anywhere\n"
            "also skips every matching inner pocket."
        )
        for control in (self.removal_source_combo, self.removal_scope_combo):
            control.setSizeAdjustPolicy(
                QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon
            )
            control.setMinimumContentsLength(10)
            control.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        removal_grid.addWidget(QLabel("Background"), 0, 0)
        removal_grid.addWidget(self.removal_source_combo, 0, 1)
        removal_grid.addWidget(self.removal_color_button, 0, 2)
        removal_grid.addWidget(QLabel("Tolerance"), 1, 0)
        removal_grid.addWidget(self.removal_tolerance_spin, 1, 1)
        removal_grid.addWidget(self.removal_scope_combo, 1, 2)
        removal_grid.setColumnStretch(1, 1)
        removal_grid.setColumnStretch(2, 1)
        self.background_removal_panel.setVisible(False)
        quick_grid.addWidget(self.background_removal_panel, 9, 0, 1, 2)
        quick_grid.setColumnStretch(0, 1)
        quick_grid.setColumnStretch(1, 1)
        image_layout.addLayout(quick_grid)

        # Text is edited on the Source tab and only there, so its controls
        # come and go with that tab: on the Rust preview they would describe
        # a layer the user cannot reach, and the space is better spent on
        # the picture controls above.
        self.text_section = QWidget()
        text_section_layout = QVBoxLayout(self.text_section)
        text_section_layout.setContentsMargins(0, 0, 0, 0)
        text_section_layout.setSpacing(image_layout.spacing())
        text_heading = QHBoxLayout()
        text_title = QLabel("Text overlay")
        text_title.setObjectName("sectionTitle")
        self.add_text_button = QPushButton("Add text")
        self.add_text_button.setObjectName("compactButton")
        self.undo_text_button = QPushButton("Undo")
        self.undo_text_button.setObjectName("compactButton")
        self.undo_text_button.setToolTip(
            "Step back through the text layers only, not the rest of the\n"
            "settings. Ctrl+Z does the same from the Source tab."
        )
        self.redo_text_button = QPushButton("Redo")
        self.redo_text_button.setObjectName("compactButton")
        self.redo_text_button.setToolTip(
            "Step forward again through the text layers. Ctrl+Y or\n"
            "Ctrl+Shift+Z does the same from the Source tab."
        )
        self.duplicate_text_button = QPushButton("Duplicate")
        self.duplicate_text_button.setObjectName("compactButton")
        self.duplicate_text_button.setToolTip(
            "Copy every selected text layer. Ctrl+D or Ctrl+C does the\n"
            "same from the Source tab."
        )
        self.remove_text_button = QPushButton("Remove")
        self.remove_text_button.setObjectName("compactButton")
        self.remove_text_button.setToolTip(
            "Delete every selected text layer. The last one is emptied\n"
            "rather than removed, so there is always one to type into."
        )
        text_heading.addWidget(text_title)
        text_heading.addStretch(1)
        text_heading.addWidget(self.undo_text_button)
        text_heading.addWidget(self.redo_text_button)
        text_heading.addWidget(self.add_text_button)
        text_heading.addWidget(self.duplicate_text_button)
        text_heading.addWidget(self.remove_text_button)
        text_section_layout.addLayout(text_heading)

        self.text_options_panel = QFrame()
        self.text_options_panel.setObjectName("inlinePanel")
        text_grid = QGridLayout(self.text_options_panel)
        text_grid.setContentsMargins(10, 9, 10, 9)
        text_grid.setHorizontalSpacing(8)
        text_grid.setVerticalSpacing(7)
        self.text_layer_combo = NoWheelComboBox()
        self.text_layer_combo.addItem("Text 1")
        self.text_edit = QLineEdit()
        self.text_edit.setPlaceholderText("Type text for the image")
        self.text_edit.setMaxLength(500)
        self.text_font_combo = NoWheelFontComboBox()
        self.text_font_combo.setEditable(False)
        self.text_font_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self.text_size_spin = NoWheelSpinBox()
        self.text_size_spin.setRange(4, 256)
        self.text_size_spin.setValue(24)
        self.text_size_spin.setSuffix(" px")
        self.text_color_button = ColorButton(
            "#ffffff", dialog_title="Choose text color"
        )
        self.text_bold_check = QCheckBox("Bold")
        self.text_italic_check = QCheckBox("Italic")
        text_style_layout = QHBoxLayout()
        text_style_layout.setContentsMargins(0, 0, 0, 0)
        text_style_layout.setSpacing(12)
        text_style_layout.addWidget(self.text_bold_check)
        text_style_layout.addWidget(self.text_italic_check)
        text_style_layout.addStretch(1)

        self.text_gradient_check = QCheckBox("Gradient")
        self.text_gradient_check.setToolTip(
            "Fade the letters from the text color into a second one. The fade\n"
            "is quantized with the rest of the artwork, so a narrow palette\n"
            "will show it as bands rather than as a smooth ramp."
        )
        self.text_gradient_direction_combo = NoWheelComboBox()
        for label, value in (
            ("Top to bottom", "vertical"),
            ("Left to right", "horizontal"),
            ("Diagonal", "diagonal"),
        ):
            self.text_gradient_direction_combo.addItem(label, value)
        self.text_gradient_color_button = ColorButton(
            "#ff9336", dialog_title="Choose the color the text fades into"
        )
        self.text_outline_spin = NoWheelSpinBox()
        self.text_outline_spin.setRange(0, MAX_OUTLINE_WIDTH)
        self.text_outline_spin.setValue(0)
        self.text_outline_spin.setSuffix(" px")
        self.text_outline_spin.setSpecialValueText("None")
        self.text_outline_spin.setToolTip(
            "Ring every letter, in logical canvas pixels, so a caption stays\n"
            "readable over artwork it happens to share a color with."
        )
        self.text_outline_color_button = ColorButton(
            "#000000", dialog_title="Choose the outline color"
        )

        self.text_align_buttons: dict[str, QPushButton] = {}
        align_layout = QHBoxLayout()
        align_layout.setContentsMargins(0, 0, 0, 0)
        align_layout.setSpacing(4)
        for name, label, tip in (
            ("left", "Left", "the left edge of the sign"),
            ("center", "Center", "the middle of the sign, side to side"),
            ("right", "Right", "the right edge of the sign"),
            ("top", "Top", "the top edge of the sign"),
            ("middle", "Middle", "the middle of the sign, top to bottom"),
            ("bottom", "Bottom", "the bottom edge of the sign"),
        ):
            button = QPushButton(label)
            button.setObjectName("compactButton")
            button.setToolTip(f"Move the selected text to {tip}")
            button.clicked.connect(
                lambda _checked=False, key=name: self._align_text_layers(key)
            )
            align_layout.addWidget(button)
            self.text_align_buttons[name] = button
        align_layout.addStretch(1)

        self.text_spread_buttons: dict[str, QPushButton] = {}
        spread_layout = QHBoxLayout()
        spread_layout.setContentsMargins(0, 0, 0, 0)
        spread_layout.setSpacing(4)
        for name, label, tip in (
            ("across", "Across", "side to side"),
            ("down", "Down", "top to bottom"),
        ):
            button = QPushButton(label)
            button.setObjectName("compactButton")
            button.setToolTip(
                f"Leave an equal gap between three or more selected layers, {tip}"
            )
            button.clicked.connect(
                lambda _checked=False, key=name: self._distribute_text_layers(key)
            )
            spread_layout.addWidget(button)
            self.text_spread_buttons[name] = button
        spread_layout.addStretch(1)

        self.text_selection_label = QLabel("")
        self.text_selection_label.setObjectName("muted")
        self.text_selection_label.setWordWrap(True)

        text_grid.addWidget(QLabel("Layer"), 0, 0)
        text_grid.addWidget(self.text_layer_combo, 0, 1, 1, 3)
        text_grid.addWidget(QLabel("Text"), 1, 0)
        text_grid.addWidget(self.text_edit, 1, 1, 1, 3)
        text_grid.addWidget(QLabel("Font"), 2, 0)
        text_grid.addWidget(self.text_font_combo, 2, 1, 1, 3)
        text_grid.addWidget(QLabel("Size"), 3, 0)
        text_grid.addWidget(self.text_size_spin, 3, 1)
        text_grid.addWidget(QLabel("Color"), 3, 2)
        text_grid.addWidget(self.text_color_button, 3, 3)
        text_grid.addWidget(QLabel("Style"), 4, 0)
        text_grid.addLayout(text_style_layout, 4, 1, 1, 3)
        text_grid.addWidget(self.text_gradient_check, 5, 0)
        text_grid.addWidget(self.text_gradient_direction_combo, 5, 1)
        text_grid.addWidget(QLabel("Into"), 5, 2)
        text_grid.addWidget(self.text_gradient_color_button, 5, 3)
        text_grid.addWidget(QLabel("Outline"), 6, 0)
        text_grid.addWidget(self.text_outline_spin, 6, 1)
        text_grid.addWidget(QLabel("Color"), 6, 2)
        text_grid.addWidget(self.text_outline_color_button, 6, 3)
        text_grid.addWidget(QLabel("Align"), 7, 0)
        text_grid.addLayout(align_layout, 7, 1, 1, 3)
        text_grid.addWidget(QLabel("Spread"), 8, 0)
        text_grid.addLayout(spread_layout, 8, 1, 1, 3)
        text_grid.addWidget(self.text_selection_label, 9, 0, 1, 4)
        text_grid.setColumnStretch(1, 1)
        text_grid.setColumnStretch(3, 1)
        text_section_layout.addWidget(self.text_options_panel)
        image_layout.addWidget(self.text_section)
        layout.addWidget(image_group)

        profile_group, profile_layout = self._step_panel(2, "Rust setup")
        self.profile_combo = NoWheelComboBox()
        self.profile_combo.setToolTip(
            "Each profile stores one sign/UI layout's calibration rectangles."
        )
        profile_row = QHBoxLayout()
        profile_row.addWidget(self.profile_combo, 1)
        self.new_profile_button = QPushButton("")
        self.new_profile_button.setToolTip(
            "New profile (inherits the current calibration)"
        )
        self.rename_profile_button = QPushButton("")
        self.rename_profile_button.setToolTip("Rename profile")
        self.delete_profile_button = QPushButton("")
        self.delete_profile_button.setToolTip("Delete profile")
        for button, icon in (
            (self.new_profile_button, "drag-drop"),
            (self.rename_profile_button, "pencil"),
            (self.delete_profile_button, "trash"),
        ):
            button.setObjectName("iconButton")
            button.setFixedWidth(38)
            self._set_icon(button, icon, size=21)
            profile_row.addWidget(button)
        self.new_profile_button.setAccessibleName("New profile")
        self.rename_profile_button.setAccessibleName("Rename profile")
        self.delete_profile_button.setAccessibleName("Delete profile")
        profile_layout.addLayout(profile_row)

        calibration_title = QLabel("Calibration")
        calibration_title.setObjectName("sectionTitle")
        profile_layout.addWidget(calibration_title)

        calibration_grid = QGridLayout()
        calibration_grid.setVerticalSpacing(5)
        calibration_grid.setHorizontalSpacing(8)
        calibration_grid.setColumnStretch(0, 1)
        self.canvas_status = CalibrationStatus("Canvas")
        self.color_box_status = CalibrationStatus("Color box")
        self.hue_bar_status = CalibrationStatus("Hue bar")
        self.brush_size_box_status = CalibrationStatus("Size value box", optional=True)
        self.clear_button_status = CalibrationStatus("Clear button", optional=True)
        self.save_button_status = CalibrationStatus("Save button", optional=True)
        self.calibrate_canvas_button = QPushButton("Set")
        self.calibrate_color_box_button = QPushButton("Set")
        self.calibrate_hue_bar_button = QPushButton("Set")
        self.calibrate_brush_button = QPushButton("Set")
        self.calibrate_clear_button = QPushButton("Set")
        self.calibrate_save_button = QPushButton("Set")
        entries = (
            (self.canvas_status, self.calibrate_canvas_button, "Calibrate canvas"),
            (self.color_box_status, self.calibrate_color_box_button, "Calibrate color box"),
            (self.hue_bar_status, self.calibrate_hue_bar_button, "Calibrate hue bar"),
            (
                self.brush_size_box_status,
                self.calibrate_brush_button,
                "Calibrate the numeric Size field beside Rust's size slider",
            ),
            (
                self.clear_button_status,
                self.calibrate_clear_button,
                "Calibrate Rust's trash/clear icon, which wipes the sign between "
                "the brush measurement and the painting",
            ),
            (
                self.save_button_status,
                self.calibrate_save_button,
                "Calibrate Rust's Save changes button, which the anti-AFK break "
                "clicks to leave the painting UI before it jumps",
            ),
        )
        for row, (status, button, tooltip) in enumerate(entries):
            button.setObjectName("compactButton")
            button.setToolTip(tooltip)
            button.setFixedWidth(68)
            self._set_icon(button, "target", size=15)
            calibration_grid.addWidget(status, row, 0)
            calibration_grid.addWidget(button, row, 1)
        profile_layout.addLayout(calibration_grid)

        self.apply_brush_check = QCheckBox("Automatic brush sizing")
        self.apply_brush_check.setToolTip(
            "Types the Size number that paints exactly one logical image cell.\n"
            "Every paint job measures this sign's brush first and wipes the "
            "measurement off again, so there is nothing to run by hand - it "
            "only needs the Size value box and the clear button calibrated."
        )
        self.show_calibration_check = QCheckBox("Show boxes on screen")
        self.show_calibration_check.setToolTip(
            "Draws labeled red outlines over every calibrated region so you can\n"
            "confirm they still line up with Rust's painting UI. The outlines\n"
            "are click-through and hide automatically while painting."
        )
        profile_layout.addWidget(self.apply_brush_check)
        self.brush_model_status = QLabel("Brush size measured at the start of each job")
        self.brush_model_status.setObjectName("muted")
        self.brush_model_status.setWordWrap(True)
        profile_layout.addWidget(self.brush_model_status)
        profile_layout.addWidget(self.show_calibration_check)
        self.show_status_check = QCheckBox("Show status on screen")
        self.show_status_check.setToolTip(
            "Writes the job's state - PAINTING, PAUSED, ABORTED - in big\n"
            "letters across the sign while a job is on, so you can see at a\n"
            "glance that it is the app moving the mouse.  The text is seen\n"
            "only by you: the app's own screen captures look straight\n"
            "through it, so it never reaches a verification or a timelapse."
        )
        profile_layout.addWidget(self.show_status_check)
        self.canvas_geometry_label = QLabel("Canvas: not calibrated  •  Aspect: —")
        self.canvas_geometry_label.setObjectName("muted")
        profile_layout.addWidget(self.canvas_geometry_label)
        self.display_warning_label = QLabel("")
        self.display_warning_label.setWordWrap(True)
        self.display_warning_label.setStyleSheet("color: #e0a34b;")
        profile_layout.addWidget(self.display_warning_label)
        self.rust_monitor_label = QLabel("")
        self.rust_monitor_label.setWordWrap(True)
        self.rust_monitor_label.setStyleSheet("color: #e0a34b;")
        self.rust_monitor_label.setVisible(False)
        self.move_to_rust_button = QPushButton("Move boxes to Rust's monitor")
        self.move_to_rust_button.setToolTip(
            "Reprojects every calibrated rectangle onto the monitor the Rust\n"
            "window is currently on, scaling for a resolution difference."
        )
        self.move_to_rust_button.setVisible(False)
        profile_layout.addWidget(self.rust_monitor_label)
        profile_layout.addWidget(self.move_to_rust_button)
        layout.addWidget(profile_group)

        run_group, run_layout = self._step_panel(3, "Paint")
        run_layout.addWidget(self._build_resume_panel())
        sessions_row = QHBoxLayout()
        self.sessions_button = QPushButton("Sessions…")
        self.sessions_button.setObjectName("compactButton")
        self.sessions_button.setToolTip(
            "Every real painting run keeps its place as a session.  Open one\n"
            "to switch back to its image and settings - pause a long sign for\n"
            "a smaller one and come back to it - or delete sessions that are\n"
            "done with."
        )
        self.sessions_button.clicked.connect(self._show_sessions_dialog)
        sessions_row.addStretch(1)
        sessions_row.addWidget(self.sessions_button)
        run_layout.addLayout(sessions_row)
        self.start_button = QPushButton("START PAINTING  •  F8")
        self.start_button.setObjectName("accent")
        self.start_button.setMinimumHeight(44)
        self._set_icon(self.start_button, "play", ON_ACCENT, size=22)
        run_buttons = QHBoxLayout()
        self.pause_button = QPushButton("Pause  •  F9")
        self.abort_button = QPushButton("Stop  •  F10")
        self.abort_button.setObjectName("danger")
        self._set_icon(self.pause_button, "pause", TEXT, size=16)
        self._set_icon(self.abort_button, "abort", ON_ACCENT, size=16)
        run_buttons.addWidget(self.pause_button)
        run_buttons.addWidget(self.abort_button)
        run_layout.addWidget(self.start_button)
        run_layout.addLayout(run_buttons)
        layout.addWidget(run_group)
        layout.addStretch(1)
        return content

    # A tooltip shared by the resume controls: the one thing to know is that
    # the slider does not need to be exact.
    _RESUME_TOOLTIP = (
        "Carry on painting a sign that already has most of the picture on it\n"
        "- after a server restart, a kick, or a stop.  Rust saves the sign\n"
        "while the painting UI is open, so what was painted is still there;\n"
        "only the painter's place is lost, and this is where it is set.\n\n"
        "The Rust preview shows the picture as far as the slider: match it\n"
        "against the sign.  Precision is not needed - resuming early only\n"
        "repaints strokes that are already there, and the touch-up pass at\n"
        "the end repairs anything missed.  A resumed job paints on the sign\n"
        "as it is: no clear, no brush probe."
    )

    def _build_resume_panel(self) -> QWidget:
        """The "resume from here" controls: a tick, a slider, and a status line."""

        self.resume_panel = QFrame()
        self.resume_panel.setObjectName("inlinePanel")
        self.resume_panel.setToolTip(self._RESUME_TOOLTIP)
        layout = QVBoxLayout(self.resume_panel)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(6)
        heading = QHBoxLayout()
        heading.setSpacing(8)
        self.resume_check = QCheckBox("Resume from stroke")
        self.resume_check.setToolTip(self._RESUME_TOOLTIP)
        self.resume_position_label = QLabel("0 of 0")
        self.resume_position_label.setObjectName("muted")
        self.resume_position_label.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        )
        heading.addWidget(self.resume_check)
        heading.addStretch(1)
        self.resume_screenshot_button = QPushButton("See what stopped it")
        self.resume_screenshot_button.setObjectName("compactButton")
        self.resume_screenshot_button.setToolTip(
            "Open the screenshot the app took of the whole screen when the\n"
            "recorded job was paused."
        )
        self.resume_screenshot_button.setVisible(False)
        self.resume_screenshot_button.clicked.connect(self._show_offered_screenshot)
        heading.addWidget(self.resume_screenshot_button)
        heading.addWidget(self.resume_position_label)
        layout.addLayout(heading)
        self.resume_slider = QSlider(Qt.Orientation.Horizontal)
        self.resume_slider.setRange(0, 0)
        self.resume_slider.setToolTip(self._RESUME_TOOLTIP)
        layout.addWidget(self.resume_slider)
        self.resume_notice = QLabel("Load an image to see its plan")
        self.resume_notice.setObjectName("muted")
        self.resume_notice.setWordWrap(True)
        layout.addWidget(self.resume_notice)
        self.resume_check.toggled.connect(self._on_resume_controls_changed)
        self.resume_slider.valueChanged.connect(self._on_resume_controls_changed)
        self.resume_panel.setEnabled(False)
        return self.resume_panel

    # ---------------------------------------------------------- resume offer

    def _resume_start_stroke(self) -> int:
        """The stroke the next job starts at: 0 unless resuming is ticked."""

        plan = self._plan
        if plan is None or not self.resume_check.isChecked():
            return 0
        total = int(getattr(plan, "stroke_count", 0) or 0)
        return max(0, min(int(self.resume_slider.value()), total))

    def _refresh_resume_offer(self) -> None:
        """Offer the plan on screen its record, if one was written for it.

        A record is offered only to the plan it was written against - the
        stroke numbers index that plan's order and no other - so a plan
        with no record starts the slider at zero, and says so when the last
        interrupted job was planned differently.
        """

        plan = self._plan
        store = self._resume_store
        slider, check = self.resume_slider, self.resume_check
        slider.blockSignals(True)
        check.blockSignals(True)
        try:
            if not getattr(plan, "stroke_count", 0):
                self._plan_fingerprint = None
                self._offered_record = None
                self._offered_key = None
                self.resume_screenshot_button.setVisible(False)
                slider.setRange(0, 0)
                slider.setValue(0)
                check.setChecked(False)
                self.resume_notice.setText("Load an image to see its plan")
                self.resume_panel.setEnabled(False)
                return
            fingerprint = plan_fingerprint(plan)
            self._plan_fingerprint = fingerprint
            record = store.load(fingerprint) if store is not None else None
            slider.setRange(0, plan.stroke_count)
            self._offered_record = record if record is not None and record.resumable else None
            self.resume_screenshot_button.setVisible(
                self._offered_record is not None
                and bool(self._offered_record.screenshot_path)
                and Path(self._offered_record.screenshot_path).exists()
            )
            if record is not None and record.resumable:
                # The tick belongs to the user once they have seen this
                # offer: re-planning the same picture (a brush or picture
                # setting changed) must not re-tick a box they unticked, or
                # drag the slider back from where they put it.
                offer_key = (fingerprint, record.updated_at, record.completed_strokes)
                if offer_key != self._offered_key:
                    self._offered_key = offer_key
                    slider.setValue(min(record.completed_strokes, plan.stroke_count))
                    check.setChecked(True)
                how = (
                    "the sign went away while painting" if record.interrupted_by_ui_loss
                    else record.state
                )
                self.resume_notice.setText(
                    f"This plan was interrupted - {how} - at stroke "
                    f"{record.completed_strokes:,} of {record.total_strokes:,} "
                    f"({record.percent:.0f}%) on {record.updated_at}.  Tick to "
                    "carry on from there, or slide to where the sign really is."
                )
            else:
                self._offered_key = None
                slider.setValue(0)
                check.setChecked(False)
                notice = (
                    "No record for this plan.  To carry on a sign painted "
                    "earlier, slide to roughly where the picture stops on it "
                    "and tick Resume."
                )
                other = (
                    store.latest_resumable(excluding=(fingerprint,))
                    if store is not None
                    else None
                )
                if other is not None:
                    image = Path(other.image_path).name if other.image_path else "an image"
                    notice += (
                        f"  The last interrupted job ({image}, {other.describe()}) "
                        "was planned differently - its stroke numbers do not "
                        "apply to this plan."
                    )
                self.resume_notice.setText(notice)
            self.resume_panel.setEnabled(
                not self._painter_is_active() and not self._countdown_callback_running
            )
        finally:
            slider.blockSignals(False)
            check.blockSignals(False)
        self._on_resume_controls_changed()

    @Slot()
    def _on_resume_controls_changed(self, *_args: Any) -> None:
        plan = self._plan
        total = int(getattr(plan, "stroke_count", 0) or 0)
        value = min(int(self.resume_slider.value()), total)
        if self._paused_viewfinder_notice is None:
            self._resume_offer_value = value
        percent = value * 100.0 / total if total else 0.0
        self.resume_position_label.setText(f"{value:,} of {total:,}  ({percent:.0f}%)")
        self._resume_preview_timer.start()
        self._update_start_availability()

    @Slot()
    def _refresh_resume_preview(self) -> None:
        """Show the picture as far as the slider, or all of it when not resuming."""

        plan = self._plan
        simulation = self._plan_simulation
        if plan is None or simulation is None:
            return
        paused = self._painter_is_paused()
        if not paused and not self.resume_check.isChecked():
            self.paint_preview.set_source(self._pil_to_pixmap(simulation))
            return
        # A paused job's slider is a viewfinder, not an instruction: it
        # shows the picture as far as any stroke so the sign can be checked
        # against it, and the job carries on from where it stopped.
        count = (
            max(0, min(int(self.resume_slider.value()), plan.stroke_count))
            if paused
            else self._resume_start_stroke()
        )
        try:
            target = np.asarray(simulation.convert("RGB"), dtype=np.uint8)
            if target.shape[0] != plan.height or target.shape[1] != plan.width:
                self.paint_preview.set_source(self._pil_to_pixmap(simulation))
                return
            painted = plan_prefix_labels(plan, count) > 0
            backdrop = _compose_checker_backdrop(
                target, np.zeros(painted.shape, dtype=np.bool_)
            )
            partial = np.asarray(backdrop, dtype=np.uint8).copy()
            partial[painted] = target[painted]
            self.paint_preview.set_source(
                self._pil_to_pixmap(Image.fromarray(partial, mode="RGB"))
            )
        except Exception:
            LOGGER.exception("Could not render the resume preview")
            self.paint_preview.set_source(self._pil_to_pixmap(simulation))

    # ------------------------------------------------------ pause screenshots

    # Pauses the user asked for, which need no explaining.  Anything else -
    # the UI guard, the focus guard, the mouse guard, a sign that did not
    # reopen - is the painter stopping on its own, and comes with a picture.
    _USER_PAUSE_REASONS = frozenset({"user hotkey/button", "global hotkey", "user"})

    def _pause_screenshot_directory(self) -> Path:
        return self._local_data_directory() / "runs" / "pauses"

    def _capture_pause_screenshot(self, generation: int, reason: str) -> None:
        """Keep the whole screen as it is now, off the GUI thread.

        The job has just paused on its own, and whatever tripped it - a
        disconnect screen, a menu, the desktop - is on the screen at this
        moment and may be gone by the time anyone looks.  The capture and
        the PNG encode take a few hundred milliseconds for two monitors, so
        they run on a worker and the result arrives as a signal.
        """

        if self._closing:
            return
        directory = self._pause_screenshot_directory()
        taken_at = time.strftime("%Y-%m-%d %H:%M:%S")
        stamp = time.strftime("%Y%m%d-%H%M%S")
        slug = "".join(
            character if character.isalnum() else "-" for character in reason.lower()
        ).strip("-")[:40] or "paused"
        path = directory / f"{stamp}-{slug}.png"
        bridge = self._painter_bridge

        def capture() -> None:
            try:
                from app.screen import capture_region, get_virtual_screen_rect

                directory.mkdir(parents=True, exist_ok=True)
                image = capture_region(get_virtual_screen_rect())
                image.save(path, format="PNG", compress_level=1)
            except Exception:
                LOGGER.exception("Could not screenshot the screen at the pause")
                return
            LOGGER.info("Screenshot of the screen at the pause saved to %s", path)
            bridge.pause_screenshot.emit(generation, str(path), reason, taken_at)

        threading.Thread(
            target=capture, name="RustPainterPauseScreenshot", daemon=True
        ).start()

    @Slot(int, str, str, str)
    def _on_pause_screenshot(
        self, generation: int, path: str, reason: str, taken_at: str
    ) -> None:
        if generation != self._paint_generation or self._closing:
            return
        screenshot = Path(path)
        self._pause_screenshot = (screenshot, reason, taken_at)
        painter = self._painter
        state = getattr(getattr(painter, "state", None), "value", None)
        self.pause_screenshot_button.setVisible(state == "paused")
        self.statusBar().showMessage(
            f"Screenshot of what stopped the painting saved to {screenshot.name}", 8000
        )
        # The record carries it too, so a restart of the app can still show
        # what the sign looked like when the job stopped.
        record = self._resume_record
        store = self._resume_store
        if record is not None and store is not None:
            record = replace(record, screenshot_path=str(screenshot))
            self._resume_record = record
            try:
                store.save(record)
            except OSError:
                LOGGER.warning("Could not attach the screenshot to the resume record", exc_info=True)

    @Slot()
    def _show_pause_screenshot(self) -> None:
        if self._pause_screenshot is None:
            return
        path, reason, taken_at = self._pause_screenshot
        self._show_screenshot(path, reason=reason, taken_at=taken_at)

    @Slot()
    def _show_offered_screenshot(self) -> None:
        record = self._offered_record
        if record is None or not record.screenshot_path:
            return
        self._show_screenshot(
            Path(record.screenshot_path),
            reason=record.reason or record.state,
            taken_at=record.updated_at,
        )

    def _show_screenshot(self, path: Path, *, reason: str, taken_at: str) -> None:
        try:
            viewer = ScreenshotViewer(path, reason=reason, taken_at=taken_at, parent=self)
        except Exception:
            LOGGER.exception("Could not open the pause screenshot")
            QMessageBox.warning(self, "Screenshot", f"Could not open {path}")
            return
        viewer.setWindowModality(Qt.WindowModality.NonModal)
        viewer.destroyed.connect(
            lambda _obj=None, ref=viewer: self._screenshot_viewers.remove(ref)
            if ref in self._screenshot_viewers
            else None
        )
        self._screenshot_viewers.append(viewer)
        viewer.show()

    # --------------------------------------------------------- resume record

    def _open_resume_record(self, pending: _PendingPaint) -> None:
        """Start this job's record; written once the artwork is going down."""

        self._resume_record = None
        self._resume_record_written_at = 0.0
        self._resume_position = (pending.start_stroke, 0)
        if pending.dry_run or self._resume_store is None:
            return
        try:
            self._resume_record = record_for_job(
                pending.plan,
                profile=pending.profile,
                image_path=self._image_path,
                settings=pending.settings,
                completed_strokes=pending.start_stroke,
            )
        except Exception:
            LOGGER.exception("Could not prepare the resume record")

    # The record is refreshed this often while the artwork goes down: close
    # enough that a crash loses a few strokes, far enough that the disk is
    # not written on every stroke.
    _RESUME_RECORD_INTERVAL_SECONDS = 3.0

    def _note_resume_progress(self, progress: Any) -> None:
        """Keep the record at the job's current stroke while it paints."""

        record = self._resume_record
        if record is None:
            return
        phase = getattr(progress, "phase", "paint")
        state = getattr(getattr(progress, "state", None), "value", "")
        if phase == "paint":
            if state != "running":
                return
            self._resume_position = (
                int(progress.completed_strokes),
                int(progress.color_index),
            )
            now = time.monotonic()
            if now - self._resume_record_written_at < self._RESUME_RECORD_INTERVAL_SECONDS:
                return
        elif phase == "verify":
            # The artwork is all down; only the touch-up is left.  A job
            # interrupted now resumes at the end, which is the touch-up.
            if self._resume_position[0] >= record.total_strokes:
                return
            self._resume_position = (record.total_strokes, record.total_colors)
        else:
            return
        self._write_resume_record(state="running")

    def _write_resume_record(
        self, *, state: str, reason: str = "", finished: bool = False
    ) -> None:
        record = self._resume_record
        store = self._resume_store
        if record is None or store is None:
            return
        completed, color_index = self._resume_position
        from app.painter import Painter

        ui_loss_reasons = Painter.UI_LOSS_REASONS
        record = advanced_record(
            record,
            completed_strokes=completed,
            color_index=color_index,
            state=state,
            reason=reason,
            interrupted_by_ui_loss=reason in ui_loss_reasons,
            finished=finished,
        )
        self._resume_record = record
        self._resume_record_written_at = time.monotonic()
        try:
            store.save(record)
        except OSError:
            LOGGER.warning("Could not write the resume record", exc_info=True)

    def _close_resume_record(self, outcome: str, reason: str) -> None:
        """Stamp the record with how the job ended and re-offer it."""

        if self._resume_record is not None:
            if outcome == "completed":
                self._resume_position = (
                    self._resume_record.total_strokes,
                    self._resume_record.total_colors,
                )
            self._write_resume_record(
                state=outcome, reason=reason, finished=outcome == "completed"
            )
            self._resume_record = None
        self._refresh_resume_offer()

    # -------------------------------------------------------- paint sessions

    @Slot()
    def _show_sessions_dialog(self) -> None:
        """The saved paint sessions: open one to switch to it, or delete some."""

        store = self._resume_store
        if store is None:
            return
        from .sessions import SessionListDialog

        active = (
            self._resume_record.fingerprint
            if self._resume_record is not None
            and (self._painter_is_active() or self._painter_is_paused())
            else None
        )
        dialog = SessionListDialog(
            store,
            store.records(),
            current_fingerprint=self._plan_fingerprint,
            active_fingerprint=active,
            parent=self,
        )
        dialog.exec()
        # A deletion in the dialog may have removed the record the resume
        # slider is offering right now; the offer must not outlive it.
        self._refresh_resume_offer()
        if dialog.chosen is not None:
            self._open_paint_session(dialog.chosen)

    def _open_paint_session(self, record: ResumeRecord) -> None:
        """Switch the app to a saved session: its profile, settings, and image.

        The resume offer arms itself once the rebuilt plan matches the
        record's fingerprint; a plan that no longer comes out the same (the
        measured brush model has moved on, say) gets the existing "planned
        differently" notice instead of a wrong stroke number.
        """

        if self._countdown_callback_running or self._debug_running:
            self.statusBar().showMessage(
                "Wait for the current start to finish before switching sessions.", 5000
            )
            return
        image_path = Path(record.image_path) if record.image_path else None
        if image_path is None or not image_path.exists():
            QMessageBox.warning(
                self,
                "Image not found",
                "This session's image file is gone:\n"
                f"{record.image_path or '(never recorded)'}\n\n"
                "Move it back and open the session again, or load the image "
                "by hand and set the resume slider to where the sign is.",
            )
            return
        countdown_open = self._countdown is not None and self._countdown.isVisible()
        if self._painter_is_active() or self._painter_is_paused() or countdown_open:
            if (
                QMessageBox.question(
                    self,
                    "Switch paint sessions",
                    "A job is under way.  Stop it and switch?  Its place is "
                    "saved as it stops, and it stays in Sessions to come "
                    "back to.",
                    QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                    QMessageBox.StandardButton.No,
                )
                != QMessageBox.StandardButton.Yes
            ):
                return
            self._pending_start_cancelled = True
            self._pending_paint = None
            if countdown_open:
                self._countdown.reject()
                self._set_idle_ui("Start cancelled")
            if self._painter is not None:
                try:
                    self._painter.abort("switched paint sessions")
                except Exception:
                    LOGGER.exception("Could not stop the current job to switch sessions")
                    return
        # The profile first: selecting one re-derives the quality dimensions,
        # which must not overwrite the session's stored resolution below.
        if record.profile_id:
            index = self.profile_combo.findData(record.profile_id)
            if index >= 0:
                if index != self.profile_combo.currentIndex():
                    self.profile_combo.setCurrentIndex(index)
            else:
                self.statusBar().showMessage(
                    "This session's sign profile no longer exists; keeping "
                    "the current one.",
                    8000,
                )
        # The plan is a function of the image and of the picture and painting
        # settings it was made with; the rest of the settings document -
        # hotkeys, safety, timing retuned since - stays as the user has it now.
        stored = record.settings or {}
        if isinstance(stored.get("image"), dict) or isinstance(stored.get("painting"), dict):
            merged = self._settings_document()
            for section in ("image", "painting"):
                if isinstance(stored.get(section), dict):
                    merged[section] = dict(stored[section])
            try:
                self._apply_settings(merged)
                self._schedule_settings_save()
            except Exception:
                LOGGER.exception("Could not apply the session's stored settings")
        self.load_image(image_path)
        self.statusBar().showMessage(
            f"Opened session {image_path.name} — the resume offer returns "
            "when its plan is rebuilt.",
            8000,
        )

    def _build_settings_page(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(10)

        heading = QHBoxLayout()
        title = QLabel("Settings")
        title.setObjectName("pageTitle")
        note = QLabel("Saved automatically")
        note.setObjectName("muted")
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(note)
        layout.addLayout(heading)

        tabs = QTabWidget()
        tabs.setObjectName("settingsTabs")
        self.settings_tabs = tabs
        tabs.addTab(self._wrap_scroll(self._build_image_settings(), 0), "Artwork")
        tabs.addTab(self._wrap_scroll(self._build_paint_settings(), 0), "Painting")
        tabs.addTab(self._wrap_scroll(self._build_safety_settings(), 0), "Safety")
        tabs.addTab(self._wrap_scroll(self._build_color_settings(), 0), "Color")
        tabs.addTab(self._build_diagnostics_settings(), "Diagnostics")
        layout.addWidget(tabs, 1)
        return content

    def _build_safety_settings(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        title = QLabel("Safety and hotkeys")
        title.setObjectName("pageTitle")
        note = QLabel("These defaults keep real mouse input deliberate and easy to stop.")
        note.setObjectName("muted")
        layout.addWidget(title)
        layout.addWidget(note)

        safety_group = QGroupBox("Input safeguards")
        safety_form = QFormLayout(safety_group)
        self.countdown_spin = self._int_spin(1, 10, 3, " s")
        self.countdown_spin.setToolTip("Time to switch focus to Rust after pressing Start.")
        self.focus_guard_check = QCheckBox("Pause unless expected window is foreground")
        self.focus_guard_check.setChecked(True)
        self.expected_window_edit = QLineEdit("Rust")
        self.expected_window_edit.setPlaceholderText("Window title contains, e.g. Rust")
        self.expected_process_edit = QLineEdit("RustClient.exe")
        self.expected_process_edit.setPlaceholderText("Executable name (optional)")
        self.mouse_pause_check = QCheckBox("Moving the mouse pauses instead of painting over it")
        self.mouse_pause_check.setChecked(True)
        self.mouse_pause_check.setToolTip(
            "Painting stops the moment you take the mouse back and releases the "
            "button, then continues from the same stroke when you resume, so an "
            "accidental nudge never costs the whole run."
        )
        self.verify_ui_check = QCheckBox("Compare calibration reference before start")
        self.ui_guard_check = QCheckBox("Pause when the painting UI disappears")
        self.ui_guard_check.setChecked(True)
        self.ui_guard_check.setToolTip(
            "Once a second the painter looks at the calibrated colour box, hue\n"
            "bar, and Clear and Save buttons.  When they are no longer on the\n"
            "screen - a server restart, a kick, or the sign closed by hand -\n"
            "it pauses instead of painting into the game world, and carries\n"
            "on from the same stroke once you open the sign and resume.\n"
            "The anti-AFK break closes the sign on purpose and is exempt."
        )
        self.start_hotkey_combo = self._hotkey_combo("F8")
        self.pause_hotkey_combo = self._hotkey_combo("F9")
        self.abort_hotkey_combo = self._hotkey_combo("F10")
        safety_form.addRow("Countdown", self.countdown_spin)
        safety_form.addRow("Focus guard", self.focus_guard_check)
        safety_form.addRow("Expected window", self.expected_window_edit)
        safety_form.addRow("Expected process", self.expected_process_edit)
        safety_form.addRow("Mouse guard", self.mouse_pause_check)
        safety_form.addRow("UI check", self.verify_ui_check)
        safety_form.addRow("UI guard", self.ui_guard_check)
        safety_form.addRow("Start / resume", self.start_hotkey_combo)
        safety_form.addRow("Pause", self.pause_hotkey_combo)
        safety_form.addRow("Stop", self.abort_hotkey_combo)
        layout.addWidget(safety_group)

        # A server that kicks idle players watches for movement, and a player
        # stood at a sign for an hour has made none.  Every interval the job
        # saves the sign, jumps, clicks to reopen the sign, and carries on.
        afk_group = QGroupBox("Anti-AFK")
        afk_form = QFormLayout(afk_group)
        self.anti_afk_check = QCheckBox("Jump every so often so the server does not kick an idle player")
        self.anti_afk_check.setToolTip(
            "Every interval the painter clicks Rust's Save button to leave the\n"
            "painting UI, presses Space to jump, waits a second, clicks to open\n"
            "the sign again, and carries on from the same stroke.  You have to\n"
            "still be looking at the sign - you were when the job started, and\n"
            "an idle camera does not turn.  Needs the Save button calibrated."
        )
        self.anti_afk_interval_spin = self._int_spin(1, 180, 30, " min")
        self.anti_afk_interval_spin.setToolTip(
            "How long the job paints between jumps.  Set it under the server's\n"
            "idle kick time."
        )
        afk_form.addRow("Anti-AFK", self.anti_afk_check)
        afk_form.addRow("Every", self.anti_afk_interval_spin)
        afk_note = QLabel(
            "Closing the painting UI with Save keeps what has been painted so "
            "far; the click that reopens the sign lands wherever the game has "
            "the crosshair, which is the sign as long as nobody has turned."
        )
        afk_note.setWordWrap(True)
        afk_note.setObjectName("muted")
        afk_form.addRow("", afk_note)
        layout.addWidget(afk_group)
        layout.addStretch(1)
        return content

    def _build_color_settings(self) -> QWidget:
        content = QWidget()
        color_accuracy_layout = QVBoxLayout(content)
        color_accuracy_layout.setContentsMargins(18, 18, 18, 18)
        color_accuracy_layout.setSpacing(10)
        title = QLabel("Color correction")
        title.setObjectName("pageTitle")
        note = QLabel("Optional compensation for Rust's sign material and lighting.")
        note.setObjectName("muted")
        color_accuracy_layout.addWidget(title)
        color_accuracy_layout.addWidget(note)

        color_group = QGroupBox("Profile measurement")
        color_group_layout = QVBoxLayout(color_group)
        self.color_correction_status = QLabel("Not measured")
        self.color_correction_status.setObjectName("muted")
        self.color_correction_status.setWordWrap(True)
        self.prepare_color_chart_button = QPushButton("Prepare Calibration Chart")
        self.measure_color_chart_button = QPushButton("Measure Painted Chart")
        self.clear_color_correction_button = QPushButton("Clear Color Correction")
        color_group_layout.addWidget(self.color_correction_status)
        color_group_layout.addWidget(self.prepare_color_chart_button)
        color_group_layout.addWidget(self.measure_color_chart_button)
        color_group_layout.addWidget(self.clear_color_correction_button)
        picker_note = QLabel(
            "Picker layout is fixed for the current Rust UI: hue runs bottom → "
            "top, saturation left → right, brightness top → bottom."
        )
        picker_note.setWordWrap(True)
        picker_note.setObjectName("muted")
        color_group_layout.addWidget(picker_note)
        color_accuracy_layout.addWidget(color_group)
        color_accuracy_layout.addStretch(1)
        return content

    def _build_diagnostics_settings(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)
        title = QLabel("Diagnostics")
        title.setObjectName("pageTitle")
        note = QLabel("Verify individual coordinates before committing to a long paint.")
        note.setObjectName("muted")
        layout.addWidget(title)
        layout.addWidget(note)

        simulation_group = QGroupBox("Plan simulation")
        simulation_layout = QVBoxLayout(simulation_group)
        self.dry_run_check = QCheckBox("Run without mouse input")
        self.dry_run_check.setChecked(False)
        self.dry_run_check.setToolTip(
            "Diagnostic mode that evaluates the paint plan without controlling Rust."
        )
        simulation_note = QLabel(
            "Useful for timing or troubleshooting a plan. Leave this off for normal painting."
        )
        simulation_note.setWordWrap(True)
        simulation_note.setObjectName("muted")
        simulation_layout.addWidget(self.dry_run_check)
        simulation_layout.addWidget(simulation_note)
        layout.addWidget(simulation_group)

        debug_group = QGroupBox("Test actions")
        debug_layout = QGridLayout(debug_group)
        self.debug_buttons: dict[str, QPushButton] = {}
        debug_actions = (
            ("canvas_tl", "Canvas top-left"),
            ("canvas_center", "Canvas center"),
            ("canvas_br", "Canvas bottom-right"),
            ("test_hue", "Select test hue"),
            ("test_sv", "Select test S/V"),
            ("test_dot", "Paint one dot"),
            ("test_stroke", "Paint short stroke"),
        )
        for index, (key, text) in enumerate(debug_actions):
            button = QPushButton(text)
            self.debug_buttons[key] = button
            debug_layout.addWidget(button, index // 2, index % 2)
        self.capture_reference_button = QPushButton("Capture UI reference")
        debug_layout.addWidget(self.capture_reference_button, 4, 0, 1, 2)
        debug_note = QLabel(
            "Use these to verify a profile with single clicks and short strokes "
            "before starting a long paint."
        )
        debug_note.setWordWrap(True)
        debug_note.setObjectName("muted")
        debug_layout.addWidget(debug_note, 5, 0, 1, 2)
        layout.addWidget(debug_group)

        log_group = QGroupBox("Activity log")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(600)
        log_layout.addWidget(self.log_view)
        layout.addWidget(log_group, 1)
        return content

    @staticmethod
    def _metric(label: str, value: str, icon_name: str = "status") -> QWidget:
        box = QFrame()
        box.setObjectName("metricCard")
        layout = QVBoxLayout(box)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(4)

        heading = QHBoxLayout()
        heading.setSpacing(8)
        glyph = _glyph_label(icon_name, 22)
        name = QLabel(label.upper())
        name.setObjectName("muted")
        heading.addWidget(glyph)
        heading.addWidget(name)
        heading.addStretch(1)

        number = QLabel(value)
        number.setObjectName("metricValue")
        layout.addLayout(heading)
        layout.addWidget(number)
        box.value_label = number  # type: ignore[attr-defined]
        return box

    @staticmethod
    def _int_spin(minimum: int, maximum: int, value: int, suffix: str) -> QSpinBox:
        spin = NoWheelSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSuffix(suffix)
        return spin

    @staticmethod
    def _double_spin(
        minimum: float, maximum: float, value: float, step: float, suffix: str
    ) -> QDoubleSpinBox:
        spin = NoWheelDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setValue(value)
        spin.setSingleStep(step)
        spin.setDecimals(2)
        spin.setSuffix(suffix)
        return spin

    @staticmethod
    def _hotkey_combo(default: str) -> QComboBox:
        combo = NoWheelComboBox()
        combo.addItems(list(SUPPORTED_HOTKEY_CHOICES))
        combo.setToolTip(
            "Laptop keyboards that need Fn for F5-F12 never deliver those presses "
            "to the app. Pick a Ctrl+Alt combo instead."
        )
        combo.setCurrentText(default)
        return combo

    @staticmethod
    def _set_form_rows_visible(layout: QFormLayout, visible: bool) -> None:
        for row in range(layout.rowCount()):
            for role in (QFormLayout.ItemRole.LabelRole, QFormLayout.ItemRole.FieldRole):
                item = layout.itemAt(row, role)
                if item and item.widget():
                    item.widget().setVisible(visible)

    # -------------------------------------------------------- image processing

    def _connect_processing_controls(self) -> None:
        self.scale_mode_combo.currentIndexChanged.connect(self._on_scale_mode_changed)
        self.crop_alignment_combo.currentIndexChanged.connect(
            self._on_crop_alignment_changed
        )
        self.original_preview.cropFocusChanged.connect(self._on_crop_focus_dragged)
        self.original_preview.cropDragFinished.connect(self._schedule_settings_save)
        self.background_combo.currentIndexChanged.connect(self._on_background_changed)
        self.background_color_button.colorChanged.connect(self._schedule_processing)
        self.alpha_fill_check.toggled.connect(self._schedule_processing)
        self.remove_background_check.toggled.connect(self._on_background_removal_changed)
        self.removal_source_combo.currentIndexChanged.connect(
            self._on_background_removal_changed
        )
        self.removal_color_button.colorChanged.connect(self._schedule_processing)
        self.removal_tolerance_spin.valueChanged.connect(self._schedule_processing)
        self.removal_scope_combo.currentIndexChanged.connect(self._schedule_processing)
        self.transparency_combo.currentIndexChanged.connect(self._on_transparency_changed)
        self.quality_combo.currentIndexChanged.connect(self._update_quality_dimensions)
        self.logical_width_spin.valueChanged.connect(
            lambda _value: self._on_logical_dimension_changed("width")
        )
        self.logical_height_spin.valueChanged.connect(
            lambda _value: self._on_logical_dimension_changed("height")
        )
        self.color_count_combo.currentIndexChanged.connect(self._schedule_processing)
        self.dither_check.toggled.connect(self._schedule_processing)
        self.sharpen_combo.currentIndexChanged.connect(self._schedule_processing)
        self.merge_combo.currentIndexChanged.connect(self._on_merge_mode_changed)
        self.paint_mode_combo.currentIndexChanged.connect(self._on_paint_mode_changed)
        self.add_text_button.clicked.connect(self._add_text_layer)
        self.undo_text_button.clicked.connect(self._undo_text_edit)
        self.redo_text_button.clicked.connect(self._redo_text_edit)
        self.duplicate_text_button.clicked.connect(self._duplicate_selected_text_layer)
        self.remove_text_button.clicked.connect(self._remove_text_layer)
        self.text_layer_combo.currentIndexChanged.connect(self._select_text_layer)
        # Only the text itself belongs to one layer; every other control
        # rewrites the whole selection, so each reports what it changed.
        self.text_edit.textChanged.connect(self._on_text_edited)
        self.text_font_combo.currentFontChanged.connect(
            lambda font: self._apply_to_selected_text(font_family=font.family())
        )
        self.text_size_spin.valueChanged.connect(self._on_text_size_changed)
        self.text_color_button.colorChanged.connect(
            lambda color: self._apply_to_selected_text(color=_rgb(color))
        )
        self.text_bold_check.toggled.connect(
            lambda checked: self._apply_to_selected_text(bold=checked)
        )
        self.text_italic_check.toggled.connect(
            lambda checked: self._apply_to_selected_text(italic=checked)
        )
        self.text_gradient_check.toggled.connect(self._on_text_gradient_toggled)
        self.text_gradient_direction_combo.currentIndexChanged.connect(
            lambda _index: self._apply_to_selected_text(
                gradient_direction=self.text_gradient_direction_combo.currentData()
            )
        )
        self.text_gradient_color_button.colorChanged.connect(
            lambda color: self._apply_to_selected_text(gradient_color=_rgb(color))
        )
        self.text_outline_spin.valueChanged.connect(
            lambda value: self._apply_to_selected_text(outline_width=int(value))
        )
        self.text_outline_color_button.colorChanged.connect(
            lambda color: self._apply_to_selected_text(outline_color=_rgb(color))
        )
        self.original_preview.layerMoved.connect(self._on_text_layer_moved)
        self.original_preview.layerSelectionChanged.connect(
            self._on_canvas_selection_changed
        )
        self.original_preview.layerTextEdited.connect(self._on_canvas_text_edited)
        self.original_preview.layerResized.connect(self._on_canvas_text_resized)
        self.original_preview.layersDeleteRequested.connect(self._delete_text_layers)
        self.original_preview.layersDuplicateRequested.connect(
            self._duplicate_text_layers
        )
        self.original_preview.interactionFinished.connect(
            self._on_text_interaction_finished
        )
        self.original_preview.undoRequested.connect(self._undo_text_edit)
        self.original_preview.redoRequested.connect(self._redo_text_edit)
        self.speed_preset_combo.currentIndexChanged.connect(self._apply_speed_preset)
        for timing in (
            self.stroke_speed_spin,
            self.dot_duration_spin,
            self.hue_delay_spin,
            self.sv_delay_spin,
            self.brush_delay_spin,
            self.stroke_delay_spin,
            self.color_delay_spin,
        ):
            timing.valueChanged.connect(self._refresh_statistics)
        for preset_spin in (
            self.stroke_speed_spin,
            self.dot_duration_spin,
            self.hue_delay_spin,
            self.sv_delay_spin,
            self.stroke_delay_spin,
            self.color_delay_spin,
            self.interpolation_spin,
        ):
            preset_spin.valueChanged.connect(self._sync_speed_preset_combo)
        self.apply_brush_check.toggled.connect(self._refresh_statistics)
        self.apply_brush_check.toggled.connect(
            lambda _checked: self._refresh_profile_ui()
        )
        # Brush sizing decides whether optimized plans may use larger brushes,
        # and logical spacing above 1.0 rules multi-cell passes out entirely.
        self.apply_brush_check.toggled.connect(self._schedule_processing)
        self.pixel_spacing_spin.valueChanged.connect(self._schedule_processing)

    def _merge_mode(self) -> str:
        """The user's stroke-merging choice, whatever the box is showing."""

        data = str(self.merge_combo.currentData() or "")
        if data in MERGE_MODE_GAPS:
            self._merge_mode_choice = data
        return self._merge_mode_choice

    def _current_overpaint_gap(self) -> int | None:
        return MERGE_MODE_GAPS.get(self._merge_mode(), 6)

    @Slot()
    def _on_merge_mode_changed(self, *_args: Any) -> None:
        self._merge_mode()
        self._schedule_processing()

    def _current_paint_mode(self) -> str:
        # A calibration chart measures the raw material response, so it is
        # always planned exactly, whatever the user's normal mode is.
        if self._painting_calibration_chart():
            return PaintMode.EXACT.value
        return str(self.paint_mode_combo.currentData() or PaintMode.BALANCED.value)

    def _brush_capabilities(self) -> BrushCapabilities:
        """What the optimizer may plan with, given the current calibration."""

        size_box = self._profile_rect("brush_size_box")
        model = self._brush_size_model()
        canvas = self._profile_rect("canvas")
        cell_pixels = 0.0
        max_brush_pixels = 0.0
        if canvas is not None:
            cell_pixels = min(
                canvas.width / max(1, self.logical_width_spin.value()),
                canvas.height / max(1, self.logical_height_spin.value()),
            )
            if model is not None:
                # The widest band the Size field can reach on this sign, in the
                # same physical pixels the cell size is measured in, so the
                # planner never offers a brush the painter would have to clamp.
                max_brush_pixels = model.largest_fraction * canvas.height
        return BrushCapabilities(
            sizing=bool(
                self.apply_brush_check.isChecked()
                and size_box is not None
                and model is not None
                # Spacing above 1.0 spreads stroke geometry while the brush
                # stays capped at one unspaced cell, so multi-cell bands would
                # leave unpainted rows; keep those plans single-cell.
                and self.pixel_spacing_spin.value() <= 1.0
            ),
            cell_pixels=cell_pixels,
            max_brush_pixels=max_brush_pixels,
        )

    def _texel_grid(self) -> Any:
        """The texel grid the last job measured on this profile's sign, if any."""

        profile = self._current_profile
        if profile is None:
            return None
        stored = profile.metadata.get("texel_grid")
        if not isinstance(stored, dict):
            return None
        try:
            from app.texel_grid import TexelGridModel

            return TexelGridModel.from_dict(stored)
        except (KeyError, TypeError, ValueError):
            LOGGER.warning("Stored texel grid is invalid", exc_info=True)
            return None

    def _brush_size_model(self) -> Any:
        """The profile's measured Size-number model, or ``None`` if unmeasured."""

        profile = self._current_profile
        stored = (
            profile.metadata.get("brush_size_model")
            if profile is not None and isinstance(profile.metadata, dict)
            else None
        )
        if not isinstance(stored, dict):
            return None
        from app.brush_calibration import BrushSizeModel

        try:
            return BrushSizeModel.from_dict(stored)
        except (KeyError, TypeError, ValueError):
            LOGGER.warning("Stored brush size model is invalid", exc_info=True)
            return None

    @Slot()
    def _on_paint_mode_changed(self, *_args: Any) -> None:
        self._sync_paint_mode_dependent_controls()
        self._schedule_processing()

    def _sync_paint_mode_dependent_controls(self) -> None:
        """Stroke merging is superseded by the optimizer outside Exact mode.

        The box then shows that it is automatic rather than the user's Exact
        mode choice, which it remembers for when Exact is chosen again.
        """

        exact = self.paint_mode_combo.currentData() == PaintMode.EXACT.value
        combo = self.merge_combo
        wanted = self._merge_mode() if exact else MERGE_MODE_OPTIMIZER
        if str(combo.currentData() or "") != wanted:
            combo.blockSignals(True)
            try:
                combo.setCurrentIndex(combo.findData(wanted))
            finally:
                combo.blockSignals(False)
        combo.setEnabled(exact)

    @staticmethod
    def _text_layer_label(index: int, layer: _TextOverlayOptions) -> str:
        summary = " ".join(layer.text.strip().split())
        if len(summary) > 22:
            summary = summary[:21] + "…"
        return f"Text {index + 1}" + (f" — {summary}" if summary else "")

    def _rebuild_text_layer_combo(self) -> None:
        self.text_layer_combo.blockSignals(True)
        self.text_layer_combo.clear()
        for index, layer in enumerate(self._text_layers):
            self.text_layer_combo.addItem(self._text_layer_label(index, layer), index)
        self.text_layer_combo.setCurrentIndex(self._selected_text_layer)
        self.text_layer_combo.blockSignals(False)
        self.remove_text_button.setEnabled(bool(self._text_layers))

    def _primary_text_index(self) -> int:
        """The layer the combo box names, which owns the text field."""

        return min(max(self._selected_text_layer, 0), len(self._text_layers) - 1)

    def _edit_target_indices(self) -> list[int]:
        """Which layers the side panel writes to.

        A canvas selection wins when there is one, so a font or a color change
        reaches every layer the user swept up.  Otherwise only the layer the
        combo box names is edited, which is the case before an image is opened
        and after a click on bare canvas has cleared the selection.
        """

        if not self._text_layers:
            return []
        selected = [
            index
            for index in self._selected_text_indices
            if 0 <= index < len(self._text_layers)
        ]
        return selected or [self._primary_text_index()]

    def _sync_text_controls(self) -> None:
        if not self._text_layers:
            return
        layer = self._text_layers[self._primary_text_index()]
        self._syncing_text_controls = True
        try:
            self.text_edit.setText(layer.text)
            if layer.font_family:
                self.text_font_combo.setCurrentFont(QFont(layer.font_family))
            self.text_size_spin.setValue(layer.font_size)
            self.text_color_button.set_color(QColor(*layer.color))
            self.text_bold_check.setChecked(layer.bold)
            self.text_italic_check.setChecked(layer.italic)
            self.text_gradient_check.setChecked(layer.gradient)
            self._set_combo_data(
                self.text_gradient_direction_combo, layer.gradient_direction
            )
            self.text_gradient_color_button.set_color(QColor(*layer.gradient_color))
            self.text_outline_spin.setValue(layer.outline_width)
            self.text_outline_color_button.set_color(QColor(*layer.outline_color))
        finally:
            self._syncing_text_controls = False
        self._refresh_text_selection_state()

    def _refresh_text_selection_state(self) -> None:
        """Show what the panel is about to edit, and gate what needs a group."""

        gradient = self.text_gradient_check.isChecked()
        self.text_gradient_direction_combo.setEnabled(gradient)
        self.text_gradient_color_button.setEnabled(gradient)
        self.text_outline_color_button.setEnabled(self.text_outline_spin.value() > 0)
        selected = len(self._edit_target_indices())
        # Spreading layers out only means something once there is a layer in
        # the middle to move.
        for button in self.text_spread_buttons.values():
            button.setEnabled(selected >= 3)
        self.text_selection_label.setText(
            f"{selected} layers selected — everything but the text itself "
            "applies to all of them."
            if selected > 1
            else "Drag a box across the Source tab, or Ctrl+click, to edit "
            "several layers at once."
        )

    @Slot()
    def _add_text_layer(self) -> None:
        if len(self._text_layers) >= MAX_TEXT_LAYERS:
            self.statusBar().showMessage(
                f"A sign can hold at most {MAX_TEXT_LAYERS} text layers", 4000
            )
            return
        offset = min(0.24, len(self._text_layers) * 0.06)
        font_size = self.text_size_spin.value()
        self._text_layers.append(
            _TextOverlayOptions(
                "",
                self.text_font_combo.currentFont().family(),
                font_size,
                _rgb(self.text_color_button.color()),
                x=min(0.85, 0.5 + offset),
                y=min(0.85, 0.5 + offset),
                bold=self.text_bold_check.isChecked(),
                italic=self.text_italic_check.isChecked(),
                size_ratio=self._text_size_ratio(font_size),
                gradient=self.text_gradient_check.isChecked(),
                gradient_color=_rgb(self.text_gradient_color_button.color()),
                gradient_direction=self.text_gradient_direction_combo.currentData(),
                outline_width=self.text_outline_spin.value(),
                outline_color=_rgb(self.text_outline_color_button.color()),
            )
        )
        self._selected_text_layer = len(self._text_layers) - 1
        self._selected_text_indices = [self._selected_text_layer]
        self._rebuild_text_layer_combo()
        self._sync_text_controls()
        self._refresh_text_editor_layers()
        self.text_edit.setFocus()
        self._record_text_history("add")
        self._schedule_settings_save()

    @Slot()
    def _duplicate_selected_text_layer(self) -> None:
        self._duplicate_text_layers(self._edit_target_indices())

    @Slot(object)
    def _duplicate_text_layers(self, indices: Any) -> None:
        """Insert a copy of each named layer, nudged clear of its original."""

        wanted = self._valid_text_indices(indices)
        if not wanted:
            return
        room = MAX_TEXT_LAYERS - len(self._text_layers)
        if room <= 0:
            self.statusBar().showMessage(
                f"A sign can hold at most {MAX_TEXT_LAYERS} text layers", 4000
            )
            return
        if len(wanted) > room:
            wanted = wanted[:room]
            self.statusBar().showMessage(
                f"Room for {room} more layer{'s' if room != 1 else ''}, "
                "so only that many were copied",
                4000,
            )
        layers: list[_TextOverlayOptions] = []
        copies: list[int] = []
        for index, layer in enumerate(self._text_layers):
            layers.append(layer)
            if index not in wanted:
                continue
            nudge = self._text_size_ratio(layer.font_size) * 0.5
            copies.append(len(layers))
            layers.append(
                replace(
                    layer,
                    x=min(max(layer.x + nudge * 0.5, 0.0), 1.0),
                    y=min(max(layer.y + nudge, 0.0), 1.0),
                )
            )
        self._text_layers = layers
        self._selected_text_indices = copies
        self._selected_text_layer = copies[-1]
        self._rebuild_text_layer_combo()
        self._sync_text_controls()
        self._refresh_text_editor_layers()
        self._record_text_history("duplicate")
        self._schedule_processing()
        self._schedule_settings_save()

    @Slot()
    def _remove_text_layer(self) -> None:
        self._delete_text_layers(self._edit_target_indices())

    @Slot(object)
    def _delete_text_layers(self, indices: Any) -> None:
        """Drop layers, keeping a single empty one to type into."""

        wanted = self._valid_text_indices(indices)
        if not wanted:
            return
        kept = [
            layer
            for index, layer in enumerate(self._text_layers)
            if index not in set(wanted)
        ]
        if kept:
            self._text_layers = kept
            # Land on the layer before the first gap, which is where the eye
            # already is however many were removed.
            self._selected_text_layer = min(max(wanted[0] - 1, 0), len(kept) - 1)
        else:
            self._text_layers = [replace(self._text_layers[0], text="")]
            self._selected_text_layer = 0
        self._selected_text_indices = []
        self._rebuild_text_layer_combo()
        self._sync_text_controls()
        self._refresh_text_editor_layers()
        self._record_text_history("delete")
        self._schedule_processing()
        self._schedule_settings_save()

    def _valid_text_indices(self, indices: Any) -> list[int]:
        return sorted(
            {
                int(index)
                for index in indices
                if 0 <= int(index) < len(self._text_layers)
            }
        )

    @Slot(int)
    def _select_text_layer(self, index: int) -> None:
        if not 0 <= index < len(self._text_layers):
            return
        self._selected_text_layer = index
        self._selected_text_indices = [index]
        if self.text_layer_combo.currentIndex() != index:
            self.text_layer_combo.blockSignals(True)
            self.text_layer_combo.setCurrentIndex(index)
            self.text_layer_combo.blockSignals(False)
        self._sync_text_controls()
        self.original_preview.select_layer(index)

    @Slot()
    def _on_canvas_selection_changed(self) -> None:
        """Let the canvas decide what the side panel is pointed at."""

        self._selected_text_indices = self._valid_text_indices(
            self.original_preview.selected_indices()
        )
        primary = self.original_preview.primary_index()
        if primary is not None and 0 <= primary < len(self._text_layers):
            self._selected_text_layer = primary
            if self.text_layer_combo.currentIndex() != primary:
                self.text_layer_combo.blockSignals(True)
                self.text_layer_combo.setCurrentIndex(primary)
                self.text_layer_combo.blockSignals(False)
        self._sync_text_controls()

    def _apply_to_selected_text(self, **fields: Any) -> None:
        """Write one styling change to every layer the panel is pointed at."""

        if self._syncing_text_controls:
            return
        changed = False
        for index in self._edit_target_indices():
            updated = replace(self._text_layers[index], **fields)
            if updated != self._text_layers[index]:
                self._text_layers[index] = updated
                changed = True
        self._refresh_text_selection_state()
        if not changed:
            return
        self._rebuild_text_layer_combo()
        self._refresh_text_editor_layers()
        # Keyed by which controls moved, so a run of size changes folds into
        # one step while a size change and then a color change do not.
        self._record_text_history("style:" + ",".join(sorted(fields)))
        self._schedule_processing()
        self._schedule_settings_save()

    @Slot(str)
    def _on_text_edited(self, text: str) -> None:
        """Only the named layer takes the typed text, however many are picked."""

        if self._syncing_text_controls or not self._text_layers:
            return
        index = self._primary_text_index()
        if self._text_layers[index].text == text:
            return
        self._text_layers[index] = replace(self._text_layers[index], text=text)
        self._rebuild_text_layer_combo()
        self._refresh_text_editor_layers()
        self._record_text_history("text")
        # The editor draws the new characters over the source immediately; the
        # plan waits for the typing to stop, because a plan for half a word is
        # thrown away by the next keystroke anyway.
        self._schedule_processing(typing=True)
        self._schedule_settings_save()

    @Slot(int)
    def _on_text_size_changed(self, font_size: int) -> None:
        self._apply_to_selected_text(
            font_size=int(font_size), size_ratio=self._text_size_ratio(int(font_size))
        )

    @Slot(bool)
    def _on_text_gradient_toggled(self, enabled: bool) -> None:
        # The second color and the direction only mean something while the
        # gradient is on, and _apply_to_selected_text gates them on the way out.
        self._apply_to_selected_text(gradient=bool(enabled))

    @Slot(str)
    def _align_text_layers(self, edge: str) -> None:
        """Park the selected layers against one edge or midline of the sign."""

        across = {"left", "center", "right"}
        if edge not in across | {"top", "middle", "bottom"}:
            return
        targets = self._edit_target_indices()
        if not targets:
            return
        axis = "x" if edge in across else "y"
        for index in targets:
            layer = self._text_layers[index]
            extent = self._text_layer_extent(layer)
            # Parked against an edge means touching it, so half the layer's
            # own width or height stands between its centre and that edge.
            half = extent[0 if axis == "x" else 1] / 2.0
            position = {
                "left": half,
                "center": 0.5,
                "right": 1.0 - half,
                "top": half,
                "middle": 0.5,
                "bottom": 1.0 - half,
            }[edge]
            self._text_layers[index] = replace(
                layer, **{axis: min(max(position, 0.0), 1.0)}
            )
        self._refresh_text_editor_layers()
        self._record_text_history("align")
        self._schedule_processing()
        self._schedule_settings_save()

    @Slot(str)
    def _distribute_text_layers(self, axis: str) -> None:
        """Even out the gaps between three or more selected layers."""

        targets = self._edit_target_indices()
        if len(targets) < 3:
            self.statusBar().showMessage(
                "Select three or more text layers to spread them out", 4000
            )
            return
        field = "x" if axis == "across" else "y"
        ordered = sorted(targets, key=lambda index: getattr(self._text_layers[index], field))
        first = getattr(self._text_layers[ordered[0]], field)
        last = getattr(self._text_layers[ordered[-1]], field)
        step = (last - first) / (len(ordered) - 1)
        for position, index in enumerate(ordered[1:-1], start=1):
            self._text_layers[index] = replace(
                self._text_layers[index], **{field: first + step * position}
            )
        self._refresh_text_editor_layers()
        self._record_text_history("spread")
        self._schedule_processing()
        self._schedule_settings_save()

    # ------------------------------------------------------------ text history

    def _text_snapshot(self) -> _TextSnapshot:
        return (
            tuple(self._text_layers),
            self._selected_text_layer,
            tuple(self._selected_text_indices),
        )

    def _reset_text_history(self) -> None:
        """Start the history over from the layers as they now stand.

        Loading a settings document replaces every layer at once, which is not
        something an undo should be able to walk back into.
        """

        self._text_history = [self._text_snapshot()]
        self._text_history_index = 0
        self._text_history_kind = ""
        self._text_history_stamp = 0.0
        self._refresh_text_history_buttons()

    def _record_text_history(self, kind: str) -> None:
        """Remember the layers as they now stand, as one undoable step.

        Successive edits of the same kind fold into one step while they keep
        arriving, so holding an arrow key down or typing a word leaves a single
        thing to undo rather than one per keystroke.
        """

        if self._restoring_text_history or not self._text_history:
            return
        snapshot = self._text_snapshot()
        if snapshot[0] == self._text_history[self._text_history_index][0]:
            # A change of selection alone is not worth a step of its own, but
            # the step it belongs to should still remember it.
            self._text_history[self._text_history_index] = snapshot
            return
        now = time.monotonic()
        continues = (
            kind == self._text_history_kind
            and now - self._text_history_stamp < TEXT_HISTORY_COALESCE_SECONDS
            # Never fold into the entry the history opened with; undoing back
            # to the layers as they were loaded has to stay possible.
            and self._text_history_index > 0
        )
        del self._text_history[self._text_history_index + 1 :]
        if continues:
            self._text_history[self._text_history_index] = snapshot
        else:
            self._text_history.append(snapshot)
            self._text_history_index = len(self._text_history) - 1
        dropped = len(self._text_history) - MAX_TEXT_HISTORY
        if dropped > 0:
            del self._text_history[:dropped]
            self._text_history_index -= dropped
        self._text_history_kind = kind
        self._text_history_stamp = now
        self._refresh_text_history_buttons()

    def _install_text_history_shortcuts(self) -> None:
        """Give the text history the window's own undo and redo keys.

        The canvas answers Ctrl+Z itself, but text is just as often typed into
        the side panel, where those keys used to reach no further than one
        line edit's private stack.  Binding them window-wide means the same
        keys walk the same history wherever the focus happens to sit.
        """

        for sequences, handler in (
            (("Ctrl+Z",), self._undo_text_edit),
            (("Ctrl+Y", "Ctrl+Shift+Z"), self._redo_text_edit),
        ):
            for sequence in sequences:
                shortcut = QShortcut(QKeySequence(sequence), self)
                shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
                shortcut.activated.connect(handler)

    def _refresh_text_history_buttons(self) -> None:
        self.undo_text_button.setEnabled(self._text_history_index > 0)
        self.redo_text_button.setEnabled(
            self._text_history_index < len(self._text_history) - 1
        )

    @Slot()
    def _undo_text_edit(self) -> None:
        self._step_text_history(-1)

    @Slot()
    def _redo_text_edit(self) -> None:
        self._step_text_history(1)

    def _step_text_history(self, direction: int) -> None:
        target = self._text_history_index + direction
        if not 0 <= target < len(self._text_history):
            self.statusBar().showMessage(
                "No text edit left to undo" if direction < 0 else "Nothing to redo",
                2500,
            )
            return
        self._text_history_index = target
        layers, primary, selected = self._text_history[target]
        self._restoring_text_history = True
        try:
            # Pixel sizes are re-derived from the stored ratio, so a step taken
            # under another quality preset still comes back the right size.
            self._text_layers = [
                replace(
                    layer,
                    font_size=self._text_font_size(
                        layer.size_ratio or self._text_size_ratio(layer.font_size)
                    ),
                )
                for layer in layers
            ]
            self._selected_text_layer = min(
                max(primary, 0), len(self._text_layers) - 1
            )
            self._selected_text_indices = [
                index for index in selected if 0 <= index < len(self._text_layers)
            ]
            self._rebuild_text_layer_combo()
            self._sync_text_controls()
            self._refresh_text_editor_layers()
        finally:
            self._restoring_text_history = False
        # The step just landed on is finished; the next edit opens a new one.
        self._text_history_kind = ""
        self._refresh_text_history_buttons()
        self._schedule_processing()
        self._schedule_settings_save()

    def _text_layer_extent(self, layer: _TextOverlayOptions) -> tuple[float, float]:
        """How much of the canvas a layer covers, as width and height fractions."""

        width, height = text_size(layer.text, layer_font(layer))
        outline = 2 * TextStyle.from_layer(layer).outline
        return (
            (width + outline) / max(1, self.logical_width_spin.value()),
            (height + outline) / self._logical_height(),
        )

    @Slot(int, float, float)
    def _on_text_layer_moved(self, index: int, x: float, y: float) -> None:
        if not 0 <= index < len(self._text_layers):
            return
        self._text_layers[index] = replace(
            self._text_layers[index],
            x=min(max(x, 0.0), 1.0),
            y=min(max(y, 0.0), 1.0),
        )
        self._schedule_processing()
        self._schedule_settings_save()

    @Slot(int, str)
    def _on_canvas_text_edited(self, index: int, text: str) -> None:
        if not 0 <= index < len(self._text_layers):
            return
        self._text_layers[index] = replace(self._text_layers[index], text=text[:500])
        self._selected_text_layer = index
        self._syncing_text_controls = True
        try:
            self.text_edit.setText(self._text_layers[index].text)
        finally:
            self._syncing_text_controls = False
        self._rebuild_text_layer_combo()
        self._record_text_history("text")
        self._schedule_processing()
        self._schedule_settings_save()

    @Slot(int, int)
    def _on_canvas_text_resized(self, index: int, font_size: int) -> None:
        if not 0 <= index < len(self._text_layers):
            return
        font_size = min(max(int(font_size), MIN_TEXT_SIZE), MAX_TEXT_SIZE)
        self._text_layers[index] = replace(
            self._text_layers[index],
            font_size=font_size,
            size_ratio=self._text_size_ratio(font_size),
        )
        self._selected_text_layer = index
        self._syncing_text_controls = True
        try:
            self.text_size_spin.setValue(font_size)
        finally:
            self._syncing_text_controls = False
        self._record_text_history("resize")
        self._schedule_processing()
        self._schedule_settings_save()

    @Slot()
    def _on_text_interaction_finished(self) -> None:
        # A drag reports every step it takes; the whole drag is one step back.
        self._record_text_history("move")
        QTimer.singleShot(0, self._refresh_text_editor_layers)

    def _source_backdrop_size(self) -> tuple[int, int] | None:
        """The size the source preview is shown at while text is placed.

        Fit and Fill keep the source's own shape, so the decoded preview is
        shown as it is.  Stretch does not: it squeezes the whole image onto a
        canvas of another shape, and a caption dropped onto a face in a square
        photo lands somewhere else entirely once that photo is pulled wide.
        The backdrop is therefore pre-distorted to the canvas aspect, which is
        both what the sign will show and what makes the editor's one uniform
        text scale the true one.
        """

        if self._source_pixmap.isNull():
            return None
        preview = (self._source_pixmap.width(), self._source_pixmap.height())
        if min(preview) <= 0:
            return None
        if self.scale_mode_combo.currentData() != ScaleMode.STRETCH.value:
            return preview
        return calculate_fit_size(
            (max(1, self.logical_width_spin.value()), self._logical_height()),
            preview,
        )

    def _refresh_source_backdrop(self) -> None:
        """Re-display the source whenever the canvas would reshape it."""

        size = self._source_backdrop_size()
        if size is None or size == self._source_preview_size:
            return
        self._source_preview_size = size
        pixmap = self._source_pixmap
        if size != (pixmap.width(), pixmap.height()):
            pixmap = pixmap.scaled(
                size[0],
                size[1],
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        self.original_preview.set_source(pixmap)

    def _source_canvas_geometry(self) -> tuple[QRectF, float] | None:
        """Where the sign canvas lies on the source preview pixmap.

        Text layers live in canvas fractions and logical pixels, but they are
        edited over the raw source image, whose aspect ratio the sign rarely
        shares.  The returned rectangle is the canvas in preview-pixmap pixels
        - larger than the pixmap under Fit letterboxing, smaller under Fill
        cropping - and the float is how many preview pixels one logical canvas
        pixel spans, mirroring the mapping ``scale_image`` applies when the
        text is baked.  Stretch needs no rectangle of its own - the backdrop
        is already displayed at the canvas's shape, so the whole pixmap is the
        canvas and one scale serves both axes.
        """

        image = self._original_image
        preview_size = self._source_backdrop_size()
        if image is None or preview_size is None:
            return None
        preview_width, preview_height = preview_size
        if min(image.width, image.height, preview_width, preview_height) <= 0:
            return None
        logical_width = max(1, self.logical_width_spin.value())
        logical_height = max(1, self.logical_height_spin.value())
        mode = self.scale_mode_combo.currentData()
        if mode == ScaleMode.FILL.value:
            centering = self._crop_centering()
            left, top, right, bottom = fill_crop_box(
                (image.width, image.height),
                (logical_width, logical_height),
                centering,
            )
            scale = preview_width / image.width
            rect = QRectF(
                left * scale,
                top * scale,
                (right - left) * scale,
                (bottom - top) * scale,
            )
        elif mode == ScaleMode.FIT.value:
            fitted_width, fitted_height = calculate_fit_size(
                (image.width, image.height), (logical_width, logical_height)
            )
            # The integer paste offsets are the ones scale_image really uses.
            paste_x = (logical_width - fitted_width) // 2
            paste_y = (logical_height - fitted_height) // 2
            scale = preview_width / fitted_width
            rect = QRectF(
                -paste_x * scale,
                -paste_y * scale,
                logical_width * scale,
                logical_height * scale,
            )
        else:
            rect = QRectF(0.0, 0.0, float(preview_width), float(preview_height))
        return rect, rect.height() / logical_height

    def _refresh_text_editor_layers(self) -> None:
        self._refresh_source_backdrop()
        self.original_preview.set_crop_pannable(
            self.scale_mode_combo.currentData() == ScaleMode.FILL.value
        )
        geometry = self._source_canvas_geometry()
        if geometry is not None:
            self.original_preview.set_canvas_geometry(*geometry)
        self.original_preview.set_layers(
            self._text_layers,
            self._edit_target_indices(),
            self._selected_text_layer,
        )
        self._refresh_preview_hint()

    def _logical_height(self) -> int:
        return max(1, self.logical_height_spin.value())

    def _text_size_ratio(self, font_size: int) -> float:
        """Express a pixel font size as a fraction of the logical canvas."""

        return float(font_size) / self._logical_height()

    def _text_font_size(self, size_ratio: float) -> int:
        """The pixel font size that reproduces a ratio at the current canvas."""

        return min(
            max(round(size_ratio * self._logical_height()), MIN_TEXT_SIZE),
            MAX_TEXT_SIZE,
        )

    def _rescale_text_layers(self) -> None:
        """Re-derive every layer's pixel size for the current canvas height.

        The stored ratio is never rewritten here, so clamping a layer at a tiny
        resolution does not lose its real size: raising the quality preset again
        restores it exactly.
        """

        changed = False
        for index, layer in enumerate(self._text_layers):
            ratio = layer.size_ratio or self._text_size_ratio(layer.font_size)
            font_size = self._text_font_size(ratio)
            if (font_size, ratio) != (layer.font_size, layer.size_ratio):
                self._text_layers[index] = replace(
                    layer, font_size=font_size, size_ratio=ratio
                )
                changed = True
        if not changed:
            return
        self._sync_text_controls()
        self._refresh_text_editor_layers()
        self._schedule_settings_save()

    def _speed_preset_values(self) -> dict[str, float]:
        return {
            "stroke_speed": self.stroke_speed_spin.value(),
            "dot_ms": self.dot_duration_spin.value(),
            "hue_ms": self.hue_delay_spin.value(),
            "sv_ms": self.sv_delay_spin.value(),
            "stroke_ms": self.stroke_delay_spin.value(),
            "color_ms": self.color_delay_spin.value(),
            "interp_px": self.interpolation_spin.value(),
        }

    def _detect_speed_preset(self) -> str:
        # Compared as the painter runs them, so a profile saved before the
        # floors existed - with a 28 ms hold that was always run at 70 - still
        # reads as the preset it was.
        current = _floored_speed_values(self._speed_preset_values())
        for name, values in SPEED_PRESETS.items():
            expected_values = _floored_speed_values(values)
            if all(
                math.isclose(float(current[key]), float(expected), abs_tol=0.01)
                for key, expected in expected_values.items()
            ):
                return name
        return "Custom"

    @Slot()
    def _apply_speed_preset(self, *_args: Any) -> None:
        values = SPEED_PRESETS.get(self.speed_preset_combo.currentText())
        if values is None:
            return
        self._applying_speed_preset = True
        try:
            self.stroke_speed_spin.setValue(float(values["stroke_speed"]))
            self.dot_duration_spin.setValue(int(values["dot_ms"]))
            self.hue_delay_spin.setValue(int(values["hue_ms"]))
            self.sv_delay_spin.setValue(int(values["sv_ms"]))
            self.stroke_delay_spin.setValue(int(values["stroke_ms"]))
            self.color_delay_spin.setValue(int(values["color_ms"]))
            self.interpolation_spin.setValue(float(values["interp_px"]))
        finally:
            self._applying_speed_preset = False
        self._refresh_statistics()
        self._schedule_settings_save()

    @Slot()
    def _sync_speed_preset_combo(self, *_args: Any) -> None:
        if self._applying_speed_preset:
            return
        detected = self._detect_speed_preset()
        if detected != self.speed_preset_combo.currentText():
            self.speed_preset_combo.blockSignals(True)
            self.speed_preset_combo.setCurrentText(detected)
            self.speed_preset_combo.blockSignals(False)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:  # noqa: N802 - Qt API
        if dropped_image_path(event) is not None:
            event.acceptProposedAction()

    def dragMoveEvent(self, event: QDragMoveEvent) -> None:  # noqa: N802 - Qt API
        if dropped_image_path(event) is not None:
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:  # noqa: N802 - Qt API
        path = dropped_image_path(event)
        if path is None:
            return
        event.acceptProposedAction()
        self.load_image(path)

    @Slot()
    def _browse_image(self) -> None:
        selected, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an image",
            str(self._image_path.parent if self._image_path else Path.home()),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff);;All files (*.*)",
        )
        if selected:
            self.load_image(Path(selected))

    def load_image(self, path: Path) -> None:
        path = Path(path)
        self._load_serial += 1
        serial = self._load_serial
        self._process_serial += 1
        self._process_timer.stop()
        self._plan_pending = False
        self._plan_deferred = False
        self._thread_pool.clear()
        # Every cached plan belongs to the image it was made from.
        self._plan_cache.clear()
        self._plan_keys.clear()
        self._original_image = None
        self._image_path = None
        self._processed = None
        self._plan = None
        self._plan_metric_source = None
        self._plan_timing_profile = None
        self._source_pixmap = QPixmap()
        self._source_preview_size = None
        self._show_preview_after_processing = True
        self.original_preview.clear_source("Decoding image…")
        self.paint_preview.clear_source("Waiting for the new image")
        self.image_name_label.setText(path.name)
        self.image_dimensions_label.setText("Loading…")
        self.processing_label.setText("Decoding image…")
        self._set_plan_processing(True, "Opening the image")
        self._refresh_statistics()
        self._update_start_availability()

        # Coalesce queued imports. A decode already running finishes safely,
        # while its serial prevents it from replacing the newest selection.
        self._load_pool.clear()
        worker = _ImageLoadWorker(serial, path)
        worker.signals.completed.connect(self._on_image_loaded)
        worker.signals.failed.connect(self._on_image_load_failed)
        self._load_pool.start(worker)

    @Slot(object)
    def _on_image_loaded(self, result: _LoadResult) -> None:
        if result.serial != self._load_serial or self._closing:
            return
        self._image_path = result.path
        self._original_image = result.image
        self.image_name_label.setText(result.path.name)
        self.image_dimensions_label.setText(
            f"{result.image.width:,} × {result.image.height:,} px  •  {result.image.mode}"
        )
        self._source_pixmap = self._pil_to_pixmap(result.preview)
        self._source_preview_size = None
        self._refresh_text_editor_layers()
        LOGGER.info(
            "Loaded image: %s (%dx%d)",
            result.path,
            result.image.width,
            result.image.height,
        )
        self._schedule_processing()

    @Slot(int, str)
    def _on_image_load_failed(self, serial: int, message: str) -> None:
        if serial != self._load_serial or self._closing:
            return
        self.image_dimensions_label.setText("Could not load image")
        self.processing_label.setText(f"Could not load image: {message}")
        self._plan_pending = False
        self._set_plan_processing(False)
        self._refresh_statistics()
        self._update_start_availability()
        QMessageBox.critical(self, "Could not load image", message)

    def _crop_centering(self) -> tuple[float, float]:
        """The centering Fill keeps right now, dragged or named."""

        return crop_centering(self._named_crop_alignment(), self._crop_focus)

    def _named_crop_alignment(self) -> str:
        """The last named anchor, which "Custom" leaves standing behind it."""

        value = str(self.crop_alignment_combo.currentData() or "")
        if value and value != CUSTOM_CROP_VALUE:
            return value
        return self._last_named_crop

    @Slot()
    def _on_crop_alignment_changed(self, *_args: Any) -> None:
        """Follow a named anchor, or keep the crop the user dragged."""

        value = str(self.crop_alignment_combo.currentData() or "")
        if value == CUSTOM_CROP_VALUE:
            # Selecting Custom without ever having dragged means "leave it
            # where it is", so it starts from the anchor being left behind.
            if self._crop_focus is None:
                self._crop_focus = crop_centering(self._last_named_crop)
        else:
            self._last_named_crop = value or self._last_named_crop
            self._crop_focus = None
        self._refresh_text_editor_layers()
        self._schedule_processing()

    @Slot(float, float)
    def _on_crop_focus_dragged(self, x: float, y: float) -> None:
        """Reframe the sign onto the part of the source the drag picked."""

        focus = (min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0))
        if focus == self._crop_focus:
            return
        self._crop_focus = focus
        if self.crop_alignment_combo.currentData() != CUSTOM_CROP_VALUE:
            self.crop_alignment_combo.blockSignals(True)
            self._set_combo_data(self.crop_alignment_combo, CUSTOM_CROP_VALUE)
            self.crop_alignment_combo.blockSignals(False)
        # The dashed frame and every text layer anchored to it follow the drag
        # immediately; the plan itself catches up on the usual debounce.
        self._refresh_text_editor_layers()
        self._schedule_processing()

    @Slot()
    def _on_scale_mode_changed(self) -> None:
        fill = self.scale_mode_combo.currentData() == ScaleMode.FILL.value
        fit = self.scale_mode_combo.currentData() == ScaleMode.FIT.value
        alpha_fill = (
            self.transparency_combo.currentData()
            == TransparencyMode.USE_BACKGROUND.value
        )
        self.crop_alignment_combo.setEnabled(fill)
        self.background_combo.setEnabled(fit or alpha_fill)
        self.background_color_button.setEnabled(
            (fit or alpha_fill) and self.background_combo.currentData() == "custom"
        )
        # Stretch reshapes the backdrop text is placed over, so the editor is
        # brought up to date now rather than when the reprocess lands.
        self._refresh_text_editor_layers()
        self._schedule_processing()

    @Slot()
    def _on_background_changed(self) -> None:
        fit = self.scale_mode_combo.currentData() == ScaleMode.FIT.value
        alpha_fill = (
            self.transparency_combo.currentData()
            == TransparencyMode.USE_BACKGROUND.value
        )
        if alpha_fill and self.background_combo.currentData() == "unpainted":
            # "Use background" needs an actual paint color. Keep the visible
            # selection consistent with the white fallback used by processing.
            self._set_combo_data(self.background_combo, "white")
            return
        self.background_color_button.setEnabled(
            (fit or alpha_fill) and self.background_combo.currentData() == "custom"
        )
        self._schedule_processing()

    @Slot()
    def _on_background_removal_changed(self, *_args: Any) -> None:
        enabled = self.remove_background_check.isChecked()
        self.background_removal_panel.setVisible(enabled)
        self.removal_color_button.setEnabled(
            enabled and self.removal_source_combo.currentData() == "custom"
        )
        self._schedule_processing()

    @Slot()
    def _on_transparency_changed(self) -> None:
        alpha_fill = (
            self.transparency_combo.currentData()
            == TransparencyMode.USE_BACKGROUND.value
        )
        if alpha_fill and self.background_combo.currentData() == "unpainted":
            self._set_combo_data(self.background_combo, "white")
        self._on_scale_mode_changed()

    def _on_logical_dimension_changed(self, axis: str) -> None:
        if not self.logical_width_spin.signalsBlocked() and self.quality_combo.currentText() != "Custom":
            self.quality_combo.blockSignals(True)
            self.quality_combo.setCurrentText("Custom")
            self.quality_combo.blockSignals(False)
        self._sync_custom_resolution(axis)
        self._refresh_resolution_cap_notice()
        self._rescale_text_layers()
        # A canvas of a new shape restretches the backdrop under it, which
        # _rescale_text_layers only redraws when a layer's size moved too.
        self._refresh_text_editor_layers()
        self._schedule_processing()

    def _sync_custom_resolution(self, source_axis: str = "width") -> None:
        """Keep logical cells square on the calibrated physical canvas."""

        if self.quality_combo.currentText() != "Custom":
            return
        aspect = max(0.001, self._canvas_aspect_ratio())
        if source_axis == "height":
            height = self.logical_height_spin.value()
            width = round(height * aspect)
            if width < 8 or width > 2048:
                width = max(8, min(2048, width))
                height = round(width / aspect)
        else:
            width = self.logical_width_spin.value()
            height = round(width / aspect)
            if height < 8 or height > 2048:
                height = max(8, min(2048, height))
                width = round(height * aspect)
        width, height = self._snap_to_sign_texture(width, height, source_axis)
        width, height = self._cap_to_sign_resolution(width, height)
        width = max(8, min(2048, width))
        height = max(8, min(2048, height))
        self.logical_width_spin.blockSignals(True)
        self.logical_height_spin.blockSignals(True)
        self.logical_width_spin.setValue(width)
        self.logical_height_spin.setValue(height)
        self.logical_width_spin.blockSignals(False)
        self.logical_height_spin.blockSignals(False)

    @staticmethod
    def _snap_to_sign_texture(
        width: int, height: int, source_axis: str
    ) -> tuple[int, int]:
        """Let a typed texture size land on the texture, not a pixel off it.

        The derived axis comes from the rectangle's shape, which is hand
        dragged: 1024 typed on a 1.997 rectangle derives 513 rows, and a
        513-row plan on a 512-row sign puts two rows on one texel somewhere.
        When the typed number is an edge of a real sign texture and the
        derived one is within a hair of that texture's other edge, the plan
        is for that texture.
        """

        from app.brush_calibration import SIGN_TEXTURE_SIZES

        typed, derived = (width, height) if source_axis == "width" else (height, width)
        candidates = [
            size[1] if source_axis == "width" else size[0]
            for size in SIGN_TEXTURE_SIZES
            if (size[0] if source_axis == "width" else size[1]) == typed
        ]
        if not candidates:
            return width, height
        nearest = min(candidates, key=lambda edge: abs(edge - derived))
        if abs(nearest - derived) > max(1, round(0.02 * nearest)):
            return width, height
        return (typed, nearest) if source_axis == "width" else (nearest, typed)

    def _sign_resolution_cap_source(self) -> str:
        """Where the cap comes from: "grid", "table", "brush" or "" for none."""

        if self._texel_grid() is not None:
            return "grid"
        model = self._brush_size_model()
        if model is None:
            return ""
        from app.brush_calibration import sign_texture_size

        if sign_texture_size(model.sign_pixel_rows, self._canvas_aspect_ratio()):
            return "table"
        return "brush"

    def _sign_resolution_cap(self) -> tuple[int, int] | None:
        """The largest logical size this sign's texture actually resolves.

        The brush measurement pins down how many texture rows the sign holds.
        Planning more rows than that cannot add detail - Rust's smallest brush
        already covers a full texel, so finer cells only make neighbouring
        strokes overpaint each other.  ``None`` until a job has measured this
        sign.
        """

        grid = self._texel_grid()
        if grid is not None:
            # Counted on the sign itself, texel by texel - but a probe on a
            # fine sign can still miss a frame-covered edge texel or, on a
            # stored grid from an older build, miscount outright (a 1024x512
            # XXL was stored as 1025x515, and Max planned cells that fought
            # over texels all run long).  A count within a few texels of the
            # game's own texture table IS that entry.
            from app.brush_calibration import SIGN_TEXTURE_SIZES

            columns, rows = grid.columns, grid.rows
            for width, height in SIGN_TEXTURE_SIZES:
                if abs(columns - width) <= 5 and abs(rows - height) <= 5:
                    columns, rows = width, height
                    break
            return (
                max(8, min(2048, columns)),
                max(8, min(2048, rows)),
            )
        model = self._brush_size_model()
        if model is None:
            return None
        from app.brush_calibration import canonical_texture_rows, sign_texture_size

        # The brush's row count is rough - a Size unit is about 0.8 of a
        # texel, and at two screen pixels per texel the smallest probe is a
        # one-pixel reading - but with the rectangle's shape it picks out
        # the sign's entry in Rust's own size table, which is exact.  Live,
        # the count alone said 640 rows of a 512-row XXL canvas, and Max
        # planned a quarter more cells than the sign could show.
        from_table = sign_texture_size(model.sign_pixel_rows, self._canvas_aspect_ratio())
        if from_table is not None:
            return (
                max(8, min(2048, from_table[0])),
                max(8, min(2048, from_table[1])),
            )
        # Rust's sign textures come in canonical sizes (powers of two and the
        # 4:3 family measured in game), and the measured count carries a little
        # noise: 527 measured rows on a 512-row sign is normal.  Snapping to
        # the canonical size is what lets a native-resolution plan line every
        # cell up with its texel instead of scattering collisions wherever the
        # noise said an extra row existed.
        rows = canonical_texture_rows(model.sign_pixel_rows)
        if rows < 8:
            return None
        height = min(rows, 2048)
        # Columns come from the measured horizontal footprint when the model
        # has one; deriving them from the calibrated rectangle's aspect assumes
        # the texture and the rectangle have the same shape, which a live probe
        # showed they need not (a 320x240 texture under a 1.20 rectangle).
        columns = model.sign_pixel_columns
        if columns > 0:
            width = canonical_texture_rows(columns)
        else:
            aspect = max(0.001, self._canvas_aspect_ratio())
            width = canonical_texture_rows(height * aspect)
        width = max(8, min(2048, width))
        return width, height

    def _screen_resolution_cap(self) -> tuple[int, int] | None:
        """The finest grid an unmeasured sign can usefully be painted on.

        Strokes land on whole screen pixels, so the calibrated canvas's own
        pixel grid already holds every cell a plan could address.  Where the
        screen is finer than the sign's texture, brush size 1 paints some
        texels more than once - the cost is repeated strokes, never lost
        detail - and a measured job later snaps the grid to the true texel
        count.  ``None`` until the canvas is calibrated.
        """

        rect = self._profile_rect("canvas")
        if rect is None:
            return None
        scale = min(1.0, 2048 / max(rect.width, rect.height))
        width = max(8, round(rect.width * scale))
        height = max(8, round(rect.height * scale))
        return width, height

    def _preset_dimensions(self, preset: str) -> tuple[int, int]:
        """The logical size a named quality preset asks for, before capping."""

        aspect = self._canvas_aspect_ratio()
        longest = QUALITY_LONG_EDGE[preset]
        if aspect >= 1.0:
            return longest, max(8, round(longest / aspect))
        return max(8, round(longest * aspect)), longest

    def _presets_at_the_cap(self, cap: tuple[int, int]) -> list[str]:
        """Every preset that lands on the sign's own resolution.

        Max is always one of them - that is what Max means - so the list is
        only worth showing when a fixed-size preset joins it, which is
        exactly the case where turning the quality up changes nothing.
        """

        cap_width, cap_height = cap
        capped = [
            preset
            for preset in QUALITY_LONG_EDGE
            if (lambda size: size[0] > cap_width or size[1] > cap_height)(
                self._preset_dimensions(preset)
            )
        ]
        if not capped:
            return []
        return [*capped, MAX_QUALITY_PRESET]

    @staticmethod
    def _join_names(names: list[str]) -> str:
        if len(names) <= 1:
            return "".join(names)
        return f"{', '.join(names[:-1])} and {names[-1]}"

    def _refresh_quality_preset_availability(self) -> bool:
        """Grey out the presets this sign has no texels for.

        A preset that asks for more cells than the texture holds is held at
        the sign's resolution, so offering it is a false affordance: it
        looks like a finer setting and paints exactly what the entry above
        it paints.  Max stays, being the honest name for the ceiling, and so
        does Custom, which can still ask for less.  Returns whether the
        selection had to move.
        """

        cap = self._sign_resolution_cap()
        moved = False
        for index in range(self.quality_combo.count()):
            preset = self.quality_combo.itemText(index)
            if preset not in QUALITY_LONG_EDGE:
                continue
            size = self._preset_dimensions(preset)
            unavailable = cap is not None and (size[0] > cap[0] or size[1] > cap[1])
            # Flags via the model, which is what greys the row out; the
            # entry stays in the list so the ceiling is visible rather than
            # mysteriously absent.
            flags = (
                Qt.ItemFlag.NoItemFlags
                if unavailable
                else Qt.ItemFlag.ItemIsSelectable | Qt.ItemFlag.ItemIsEnabled
            )
            self.quality_combo.setItemData(index, flags, Qt.ItemDataRole.UserRole - 1)
            if unavailable:
                self.quality_combo.setItemData(
                    index,
                    f"This sign holds only {cap[0]}×{cap[1]} texels, fewer than "
                    f"{preset}'s {size[0]}×{size[1]}.  Max paints the sign's own "
                    "resolution.",
                    Qt.ItemDataRole.ToolTipRole,
                )
                if index == self.quality_combo.currentIndex():
                    # Selected but unpaintable - from a saved setting, or a
                    # measurement that has just lowered the ceiling.  Max is
                    # the same size and says what it is.
                    self.quality_combo.blockSignals(True)
                    self.quality_combo.setCurrentText(MAX_QUALITY_PRESET)
                    self.quality_combo.blockSignals(False)
                    moved = True
            else:
                self.quality_combo.setItemData(index, None, Qt.ItemDataRole.ToolTipRole)
        return moved

    def _refresh_resolution_cap_notice(self) -> None:
        """Say, next to the setting itself, when the sign is the limit.

        The plan summary already notes a capped resolution once a plan
        exists, but the question this answers - "why does turning the
        quality up do nothing?" - is asked at the combo box, often before
        any image is loaded, so the answer belongs there too.
        """

        cap = self._sign_resolution_cap()
        if cap is None:
            self.resolution_cap_panel.setVisible(False)
            return
        cap_width, cap_height = cap
        source = self._sign_resolution_cap_source()
        if source == "grid":
            self.resolution_cap_panel.setToolTip(
                f"The last paint job measured this sign's texture at "
                f"{cap_width}×{cap_height} texels by stamping its grid, so this "
                "is the sign's own resolution rather than an estimate.  A plan "
                "with more cells than that cannot add detail: neighbouring "
                "cells would land on the same texel and overpaint each other."
            )
        elif source == "table":
            self.resolution_cap_panel.setToolTip(
                f"{cap_width}×{cap_height} is the texture size Rust's own sign "
                "data declares for a sign of this shape, picked by the brush "
                "measurement.  The next paint job measures the sign's grid "
                "directly and confirms it."
            )
        else:
            self.resolution_cap_panel.setToolTip(
                f"Estimated from the brush measurement at about "
                f"{cap_width}×{cap_height} texels; no sign in Rust's size data "
                "matches, so this is rough.  The next paint job measures the "
                "sign's grid directly and may refine this."
            )
        if self.quality_combo.currentText() == "Custom":
            requested = (
                self.logical_width_spin.value(),
                self.logical_height_spin.value(),
            )
            if requested[0] < cap_width or requested[1] < cap_height:
                self.resolution_cap_panel.setVisible(False)
                return
            basis = {
                "grid": "",
                "table": " (Rust's own sign data)",
                "brush": " (estimated from the brush)",
            }[source]
            about = "about " if source == "brush" else ""
            self.resolution_cap_label.setText(
                f"This sign holds {about}{cap_width}×{cap_height} texels{basis} — "
                "as fine as a custom resolution can go here."
            )
            self.resolution_cap_panel.setVisible(True)
            return
        # Only while the ceiling is actually in the way - on a coarser
        # choice it is not, and a notice that is always up is one nobody
        # reads.  With the unpaintable presets greyed out, that means while
        # Max is selected.
        shared = self._presets_at_the_cap(cap)
        if self.quality_combo.currentText() not in shared:
            self.resolution_cap_panel.setVisible(False)
            return
        unavailable = [preset for preset in shared if preset != MAX_QUALITY_PRESET]
        self.resolution_cap_label.setText(
            f"Max is painting this sign's full resolution, {cap_width}×"
            f"{cap_height} texels. {self._join_names(unavailable)} "
            f"{'ask' if len(unavailable) > 1 else 'asks'} for more than the "
            "sign holds, so they are greyed out."
        )
        self.resolution_cap_panel.setVisible(True)

    def _cap_to_sign_resolution(self, width: int, height: int) -> tuple[int, int]:
        """Hold a requested logical size at what the sign can actually show.

        Records the note the plan summary appends, so a capped resolution is
        announced next to the stroke counts rather than silently swapped in.
        """

        self._resolution_cap_note = ""
        cap = self._sign_resolution_cap()
        if cap is None:
            return width, height
        cap_width, cap_height = cap
        if width <= cap_width and height <= cap_height:
            return width, height
        self._resolution_cap_note = (
            f"  •  capped at {cap_width}×{cap_height}, all this sign's "
            "texture resolves"
        )
        return cap_width, cap_height

    @Slot()
    def _update_quality_dimensions(self) -> None:
        # Availability first: a preset the sign cannot hold is greyed out
        # here, and the selection may move off it before the size is read.
        self._refresh_quality_preset_availability()
        preset = self.quality_combo.currentText()
        custom = preset == "Custom"
        self.custom_resolution_panel.setVisible(custom)
        self.logical_width_spin.setEnabled(custom)
        self.logical_height_spin.setEnabled(custom)
        if not custom:
            aspect = self._canvas_aspect_ratio()
            if preset == MAX_QUALITY_PRESET:
                # One logical cell per sign texel - the resolution ceiling
                # itself, so there is nothing further to cap.  An unmeasured
                # sign uses its screen-pixel grid instead: strokes land on
                # whole screen pixels, so that grid already holds every cell
                # a plan could express, and a later measurement only trims
                # the strokes that would repeat inside one texel.
                self._resolution_cap_note = ""
                measured = self._sign_resolution_cap()
                cap = measured or self._screen_resolution_cap()
                if cap is not None:
                    width, height = cap
                    if measured is not None:
                        # Not a cap - it is what Max asked for - but the
                        # number is still the run's headline fact, and the
                        # summary is where a finished plan is read.
                        self._resolution_cap_note = (
                            f"  •  {width}×{height}, this sign's full resolution"
                        )
                else:
                    # Neither a measurement nor a calibrated canvas: nothing
                    # to derive a grid from until the sign is framed.
                    longest = QUALITY_LONG_EDGE["Very High"]
                    if aspect >= 1.0:
                        width, height = longest, max(8, round(longest / aspect))
                    else:
                        width, height = max(8, round(longest * aspect)), longest
            else:
                longest = QUALITY_LONG_EDGE[preset]
                if aspect >= 1.0:
                    width, height = longest, max(8, round(longest / aspect))
                else:
                    width, height = max(8, round(longest * aspect)), longest
                width, height = self._cap_to_sign_resolution(width, height)
            self.logical_width_spin.blockSignals(True)
            self.logical_height_spin.blockSignals(True)
            self.logical_width_spin.setValue(width)
            self.logical_height_spin.setValue(height)
            self.logical_width_spin.blockSignals(False)
            self.logical_height_spin.blockSignals(False)
        else:
            self._sync_custom_resolution("width")
        self._refresh_resolution_cap_notice()
        self._rescale_text_layers()
        self._schedule_processing()

    def _plan_cache_key(self) -> tuple:
        """Everything a finished plan depends on, as one hashable value.

        Built by walking the option dataclasses rather than by listing fields
        here, so a new option joins the key the day it is added instead of
        silently letting a stale plan be reused.
        """

        return (
            _hashable(self._processing_options()),
            self._current_overpaint_gap(),
            self._text_overlay_options(),
            _hashable(self._color_correction_model()),
            self._current_paint_mode(),
            _hashable(self._brush_capabilities()),
        )

    @staticmethod
    def _plan_cache_cost(result: _ProcessResult) -> int:
        """Roughly how much memory one cached plan is holding on to."""

        plan = result.plan
        return (
            plan.stroke_count * _STROKE_BYTES
            + plan.width * plan.height * _CELL_BYTES
        )

    def _remember_plan(self, key: tuple, result: _ProcessResult) -> None:
        """Keep a finished plan, dropping the oldest ones to stay in budget."""

        self._plan_cache.pop(key, None)
        self._plan_cache[key] = result
        held = sum(
            self._plan_cache_cost(entry) for entry in self._plan_cache.values()
        )
        while len(self._plan_cache) > 1 and (
            len(self._plan_cache) > PLAN_CACHE_ENTRIES or held > PLAN_CACHE_BYTES
        ):
            _, dropped = self._plan_cache.popitem(last=False)
            held -= self._plan_cache_cost(dropped)

    @Slot()
    def _schedule_processing(self, *_args: Any, typing: bool = False) -> None:
        """Queue a recalculation, waiting out the settle delay it deserves.

        Nothing about the recalculation is announced here.  Saying "working"
        the instant a control moves means saying it again on the next
        keystroke, and the announcement outlives every one of them: the busy
        overlay goes up on the first character and only comes down once typing
        stops.  The waiting is instead announced by whichever recalculation
        actually survives the settle delay, in :meth:`_start_processing`.
        """

        if self._original_image is None:
            return
        self._process_serial += 1
        key = self._plan_cache_key()
        cached = self._plan_cache.get(key)
        if cached is not None:
            # Nothing about this combination has changed since it was planned,
            # so there is nothing to recompute and nothing to wait for.
            self._plan_cache.move_to_end(key)
            self._process_timer.stop()
            self._plan_pending = False
            self._plan_deferred = False
            self._thread_pool.clear()
            self._set_plan_processing(False)
            self._on_processing_complete(replace(cached, serial=self._process_serial))
            return
        self._plan = None
        self._processed = None
        self._plan_metric_source = None
        self._plan_timing_profile = None
        self._plan_pending = True
        self._process_timer.start(TYPING_SETTLE_MS if typing else PLAN_SETTLE_MS)
        self._refresh_statistics()
        self._update_start_availability()

    def _set_plan_processing(
        self, processing: bool, title: str = "Recalculating the paint plan"
    ) -> None:
        """Show or hide the visible signs that the plan is being recalculated.

        The small spinner beside the PAINT PLAN heading is easy to miss when
        the control that started the work is in another panel, so the busy
        overlay covers the preview as well and names the settings being
        planned for - which is the question a slow recalculation raises.
        """

        self._plan_processing = processing
        if processing:
            self._plan_pending = False
            self.processing_spinner.start()
            self.plan_busy.set_top_inset(self.preview_tabs.tabBar().height())
            self.plan_busy.begin(title, self._plan_summary())
        else:
            self.processing_spinner.stop()
            self.plan_busy.end()

    def _plan_summary(self) -> str:
        """The settings a running recalculation is planning against."""

        mode = self.paint_mode_combo.currentText().split("—")[0].strip()
        return (
            f"{self.quality_combo.currentText()} quality  •  {mode} "
            f"optimization  •  {self.logical_width_spin.value()} × "
            f"{self._logical_height()}"
        )

    def _background_color(self) -> tuple[int, int, int] | None:
        mode = self.background_combo.currentData()
        if mode == "white":
            return (255, 255, 255)
        if mode == "black":
            return (0, 0, 0)
        if mode == "custom":
            color = self.background_color_button.color()
            return color.red(), color.green(), color.blue()
        return None

    def _processing_options(self) -> ImageProcessOptions:
        background = self._background_color()
        transparency = TransparencyMode(self.transparency_combo.currentData())
        transparent_fill = background or (255, 255, 255)
        removal_color = self.removal_color_button.color()
        return ImageProcessOptions(
            logical_width=self.logical_width_spin.value(),
            logical_height=self.logical_height_spin.value(),
            scale_mode=ScaleMode(self.scale_mode_combo.currentData()),
            crop_alignment=CropAlignment(self._named_crop_alignment()),
            crop_focus=self._crop_focus,
            color_count=int(self.color_count_combo.currentData()),
            dither=self.dither_check.isChecked(),
            sharpen=SharpenMode(self.sharpen_combo.currentData()),
            background_color=background,
            transparency_mode=transparency,
            transparent_fill_color=transparent_fill,
            alpha_threshold=0,
            alpha_fill=self.alpha_fill_check.isChecked(),
            remove_background=self.remove_background_check.isChecked(),
            background_removal_color=(
                (removal_color.red(), removal_color.green(), removal_color.blue())
                if self.removal_source_combo.currentData() == "custom"
                else None
            ),
            background_removal_tolerance=float(self.removal_tolerance_spin.value()),
            background_removal_scope=BackgroundRemovalScope(
                self.removal_scope_combo.currentData()
            ),
        )

    # ------------------------------------------------- exporting the preview

    def _preview_export_image(self, transparent: bool) -> Image.Image | None:
        """The Rust preview as a file, which is not quite what is on screen.

        On screen, unpainted cells are drawn over a checkerboard so they read
        as bare sign rather than as black paint.  A file is going to be
        measured against a capture instead of looked at, so where the format
        can say "nothing here" it says it, and only where it cannot does the
        checker stand in.
        """

        processed = self._processed
        if processed is None:
            return None
        correction = self._color_correction_model()
        if not transparent:
            return _build_simulation_image(processed, correction)
        mask = np.asarray(processed.paint_mask, dtype=np.bool_)
        rgb = np.asarray(processed.image.convert("RGB"), dtype=np.uint8)
        if correction is not None:
            rgb = _predicted_sign_colors(rgb, mask, correction)
        alpha = np.where(mask, 255, 0).astype(np.uint8)
        return Image.fromarray(np.dstack((rgb, alpha)), mode="RGBA")

    def _last_preview_directory(self) -> Path:
        stored = self._settings.get("ui", {}).get("last_preview_export_directory")
        if isinstance(stored, str) and stored:
            candidate = Path(stored)
            if candidate.is_dir():
                return candidate
        return self._image_path.parent if self._image_path else Path.home()

    @Slot()
    def _export_rust_preview(self) -> None:
        """Write the Rust preview out at one pixel per logical cell.

        That is the grid the plan is expressed in, so a capture of the finished
        sign scaled down to the same size lines up cell for cell and the
        difference between promise and result is a straight subtraction.
        """

        processed = self._processed
        if processed is None:
            QMessageBox.information(
                self,
                "Nothing to save",
                "Load an image and let the paint plan finish first.",
            )
            return
        width, height = processed.image.size
        stem = self._image_path.stem if self._image_path else "sign"
        suggested = self._last_preview_directory() / f"{stem}-rust-{width}x{height}.png"
        chosen, _filter = QFileDialog.getSaveFileName(
            self,
            "Save the Rust preview",
            str(suggested),
            "PNG image (*.png);;WebP image (*.webp);;"
            "JPEG image (*.jpg *.jpeg);;Bitmap image (*.bmp);;All files (*)",
        )
        if not chosen:
            return
        destination = Path(chosen)
        if not destination.suffix:
            destination = destination.with_suffix(".png")
        transparent = destination.suffix.lower() in {
            ".png",
            ".webp",
            ".tif",
            ".tiff",
        }
        image = self._preview_export_image(transparent)
        if image is None:
            return
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            image.save(destination)
        except (OSError, ValueError) as exc:
            LOGGER.exception("Could not save the Rust preview")
            QMessageBox.warning(self, "Could not save the preview", str(exc))
            return
        self._settings.setdefault("ui", {})["last_preview_export_directory"] = str(
            destination.parent
        )
        self._schedule_settings_save()
        LOGGER.info("Saved the Rust preview to %s (%dx%d)", destination, width, height)
        self.statusBar().showMessage(
            f"Saved the Rust preview to {destination}"
            f"  •  {width} × {height}, one pixel per logical cell",
            10000,
        )

    def _text_overlay_options(self) -> tuple[_TextOverlayOptions, ...]:
        return tuple(self._text_layers)

    def _painting_calibration_chart(self, profile: Any = None) -> bool:
        """Whether the loaded image is the chart prepared for ``profile``.

        The chart exists to measure the raw Rust material response, so it is
        both painted and previewed with any earlier correction out of the way.
        """

        target = self._current_profile if profile is None else profile
        return bool(
            isinstance(target, Profile)
            and self._color_chart_profile_id == target.id
            and self._color_chart_path is not None
            and self._image_path == self._color_chart_path
        )

    def _picker_geometry(self) -> _PickerGeometry | None:
        """The active profile's color panel widgets, for palette snapping.

        Skipped while painting the calibration chart: the chart measures the
        raw picker response, so its colors must not be pre-snapped.
        """

        if self._painting_calibration_chart():
            return None
        profile = self._current_profile
        if profile is None or profile.hue_bar is None or profile.color_box is None:
            return None
        return _PickerGeometry(
            hue_bar=profile.hue_bar,
            color_box=profile.color_box,
            hue_direction=profile.hue_direction,
            saturation_direction=profile.saturation_direction,
            value_direction=profile.value_direction,
        )

    def _color_correction_model(self) -> ColorCorrectionModel | None:
        """The active profile's measured sign response, when it is usable."""

        if self._painting_calibration_chart():
            return None
        profile = self._current_profile
        stored = profile.metadata.get("color_correction") if profile else None
        if not isinstance(stored, Mapping):
            return None
        try:
            return ColorCorrectionModel.from_dict(stored)
        except (TypeError, ValueError):
            LOGGER.warning("Ignoring an unreadable stored color correction")
            return None

    @Slot()
    def _start_processing(self) -> None:
        if self._original_image is None:
            return
        if (
            self.preview_tabs.currentIndex() == 0
            and not self._show_preview_after_processing
        ):
            # The Source tab is where text is edited, and the recalculation's
            # busy overlay would cover the editing in progress.  Hold the
            # recalculation instead; flipping to the Rust preview - the only
            # place its result is visible - runs it then.  A freshly imported
            # image is exempt: its first plan is what fronts the preview.
            self._plan_deferred = True
            return
        self._plan_deferred = False
        serial = self._process_serial
        # The settle delay has passed, so this recalculation is the one that
        # is really going to run and is the one worth announcing.  The overlay
        # waits out a delay of its own on top, so short work still never
        # flashes anything on screen.
        self._plan_pending = False
        self.processing_label.setText("Updating paint simulation…")
        self._set_plan_processing(True)
        # Drop queued stale previews; an already running worker is allowed to
        # finish, but its serial prevents it from replacing newer settings.
        self._thread_pool.clear()
        # The worker-side processor makes its own detached copy; avoid copying
        # a potentially huge source image on the GUI thread here.
        worker = _ImageWorker(
            serial,
            self._original_image,
            self._processing_options(),
            self._current_overpaint_gap(),
            self._text_overlay_options(),
            self._color_correction_model(),
            self._current_paint_mode(),
            self._brush_capabilities(),
            self._picker_geometry(),
        )
        worker.signals.completed.connect(self._on_processing_complete)
        worker.signals.failed.connect(self._on_processing_failed)
        # Recorded rather than recomputed on arrival, so the result is always
        # filed under the settings it was actually planned from.  A worker
        # dropped from the queue above never reports back, so the oldest
        # entries are let go rather than waiting for a result that will not
        # come; only one worker runs at a time.
        self._plan_keys[serial] = self._plan_cache_key()
        for stale in sorted(self._plan_keys)[:-2]:
            del self._plan_keys[stale]
        self._thread_pool.start(worker)

    @Slot(object)
    def _on_processing_complete(self, result: _ProcessResult) -> None:
        # Filed before the staleness check: a plan whose settings have already
        # been left behind is still a correct plan for those settings, and
        # keeping it is exactly what makes flicking back to them instant.
        key = self._plan_keys.pop(result.serial, None)
        if key is not None and not self._closing:
            self._remember_plan(key, result)
        if result.serial != self._process_serial or self._closing:
            return
        self._set_plan_processing(False)
        self._processed = result.processed
        self._plan = result.plan
        self._plan_metric_source = result.plan
        self._plan_timing_profile = result.timing_profile
        self._plan_simulation = result.simulation
        self.paint_preview.set_source(self._pil_to_pixmap(result.simulation))
        # Re-offered per plan: the record, if any, belongs to this plan's
        # stroke order and no other; and the preview follows the slider.
        self._refresh_resume_offer()
        if not (
            self.original_preview.is_interacting
            or self.original_preview.is_panning_crop
        ):
            # Scale-mode or resolution changes land here, so the editor's
            # canvas mapping is refreshed alongside the simulation.
            self._refresh_text_editor_layers()
        if self._show_preview_after_processing:
            self._show_preview_after_processing = False
            self.preview_tabs.setCurrentIndex(1)
        optimization = result.optimization
        merged_away = result.unmerged_stroke_count - result.plan.stroke_count
        if optimization is not None:
            saved_note = ""
            if merged_away > 0 and result.unmerged_stroke_count > 0:
                saved_percent = merged_away * 100.0 / result.unmerged_stroke_count
                saved_note = (
                    f", {result.unmerged_stroke_count:,}→"
                    f"{result.plan.stroke_count:,} strokes (−{saved_percent:.0f}%)"
                )
            merge_note = (
                f"  •  {optimization.mode} optimization: "
                f"{optimization.input_colors}→{optimization.output_colors} colors"
                f"{saved_note}  •  ~{optimization.similarity_percent:.0f}% similar"
            )
        elif merged_away > 0 and result.unmerged_stroke_count > 0:
            saved_percent = merged_away * 100.0 / result.unmerged_stroke_count
            merge_note = (
                f"  •  stroke merging removed {merged_away:,} strokes "
                f"(−{saved_percent:.0f}%)"
            )
        else:
            merge_note = ""
        if (
            result.processed.painted_pixel_count == 0
            and self.remove_background_check.isChecked()
        ):
            # An over-wide tolerance swallows the subject too, and an empty
            # plan otherwise looks like a plain "0" in the statistics.
            self.processing_label.setText(
                "Background removal skipped the whole image — lower the "
                "tolerance or choose a different background color"
            )
        else:
            self.processing_label.setText(
                f"{result.processed.painted_pixel_count:,} logical pixels will "
                "be painted" + merge_note + self._resolution_cap_note
            )
        LOGGER.info(
            "Generated %dx%d plan: %d colors, %d strokes",
            result.plan.width,
            result.plan.height,
            len(result.plan.color_groups),
            result.plan.stroke_count,
        )
        self._refresh_statistics()
        self._update_start_availability()

    @Slot(int, str)
    def _on_processing_failed(self, serial: int, message: str) -> None:
        self._plan_keys.pop(serial, None)
        if serial != self._process_serial or self._closing:
            return
        self._set_plan_processing(False)
        self.processing_label.setText(f"Could not process image: {message}")
        self._plan = None
        self._processed = None
        self._plan_metric_source = None
        self._plan_simulation = None
        self._refresh_resume_offer()
        self._refresh_statistics()
        self._update_start_availability()

    def _refresh_statistics(self, *_args: Any) -> None:
        plan = self._plan
        # The preview can be saved exactly when there is one to save.
        self.export_preview_button.setEnabled(self._processed is not None)
        if plan is None:
            # While a recalculation is in flight the metrics read as pending
            # rather than absent, so the numbers do not just vanish.
            placeholder = (
                "…" if self._plan_processing or self._plan_pending else "—"
            )
            for widget in (
                self.analysis_resolution,
                self.analysis_colors,
                self.analysis_strokes,
                self.analysis_time,
            ):
                widget.value_label.setText(placeholder)  # type: ignore[attr-defined]
            return
        self.analysis_resolution.value_label.setText(  # type: ignore[attr-defined]
            f"{plan.width} × {plan.height}"
        )
        # Optimized plans emit several passes per color, so count colors.
        self.analysis_colors.value_label.setText(  # type: ignore[attr-defined]
            str(len({group.color for group in plan.color_groups}))
        )
        self.analysis_strokes.value_label.setText(  # type: ignore[attr-defined]
            f"{plan.stroke_count:,}"
        )
        estimate = self._estimate(plan)
        # One figure for the whole job, checks and touch-up included; what
        # each part costs, and how well it is known, is a hover away.
        self.analysis_time.value_label.setText(  # type: ignore[attr-defined]
            self._format_duration(estimate.total)
        )
        self.analysis_time.setToolTip(self._describe_estimate(estimate))

    def _describe_estimate(self, estimate: RunEstimate) -> str:
        learned = self._learned_timing
        lines = [f"Painting: {self._format_duration(estimate.paint)}"]
        if estimate.calibration > 0:
            lines.append(f"Brush measurement: {self._format_duration(estimate.calibration)}")
        if estimate.countdown > 0:
            lines.append(f"Countdown: {self._format_duration(estimate.countdown)}")
        if self.confirm_strokes_check.isChecked():
            known = (
                f"from {learned.check_samples} run{'s' if learned.check_samples != 1 else ''}"
                if learned.check_samples
                else "a guess until a run has measured it"
            )
            lines.append(
                f"Color checks: {self._format_duration(estimate.checks)} ({known})"
            )
        if self.verify_passes_spin.value() > 0:
            known = (
                f"from {learned.touch_up_samples} run{'s' if learned.touch_up_samples != 1 else ''}"
                if learned.touch_up_samples
                else "a guess until a run has finished one"
            )
            lines.append(
                f"Touch-up: {self._format_duration(estimate.touch_up)} ({known})"
            )
        lines.append(
            "The checks and the touch-up repaint whatever the game dropped, which "
            "no estimate can know before the sign exists; each finished run "
            "refines these figures."
        )
        return "\n".join(lines)

    def _estimate_seconds(self, plan: PaintPlan) -> float:
        return self._estimate(plan).total

    def _estimate(self, plan: PaintPlan) -> RunEstimate:
        """Predict the run from the painter's own timing rules.

        Strokes are priced as the painter executes them - a held press per
        stroke, held picker clicks per color change, a retyped Size field per
        brush change - plus the countdown and the brush measurement that
        precede the first stroke.  Mouse speed barely matters: at any usable
        setting nearly every stroke is shorter than the frame it is held for.
        Checking colors as they go down and the touch-up pass at the end are
        priced from what earlier runs measured them at.
        """

        canvas = self._profile_rect("canvas")
        cell_width = canvas.width / plan.width if canvas else 1.0
        if plan is self._plan_metric_source and self._plan_timing_profile is not None:
            profile = self._plan_timing_profile
        else:
            profile = PlanProfile.from_plan(plan)
        document = self._settings_document()
        from app.painter import Painter, PainterSettings

        try:
            settings = PainterSettings.from_mapping(document)
        except (TypeError, ValueError):
            settings = PainterSettings()
        timing = StrokeTiming.from_settings(
            settings, overhead_seconds=self._learned_timing.overhead_seconds
        )
        # The painter measures the brush before every sizing run, which needs
        # the Size field and clear control calibrated.
        calibrates = bool(
            self.apply_brush_check.isChecked()
            and self._profile_rect("brush_size_box") is not None
            and self._profile_rect("clear_button") is not None
        )
        # Long drags are paced by the sign's texel pitch, which the painter
        # takes from the grid or brush model measured on this sign; priced
        # without it a plan of long sweeps is promised at a speed the drags
        # are never driven at.
        pitch = (
            Painter._texel_pitch_pixels(
                plan,
                canvas,
                self._brush_size_model() if calibrates else None,
                self._texel_grid(),
            )
            if canvas is not None
            else None
        )
        paint = profile.seconds(
            timing, cell_width, sizing=calibrates, texel_pitch_pixels=pitch
        )
        checks = (
            self._learned_timing.check_seconds(
                sum(1 for group in profile.groups if group.stroke_count), paint
            )
            if self.confirm_strokes_check.isChecked()
            else 0.0
        )
        touch_up = (
            self._learned_timing.touch_up_seconds(paint)
            if self.verify_passes_spin.value() > 0
            else 0.0
        )
        return RunEstimate(
            paint=paint,
            checks=checks,
            touch_up=touch_up,
            calibration=BRUSH_CALIBRATION_SECONDS if calibrates else 0.0,
            countdown=max(0.0, float(self.countdown_spin.value())),
        )

    @classmethod
    def _timing_path(cls) -> Path:
        return cls._local_data_directory() / "timing.json"

    def _learn_timing(self) -> None:
        """Fold the finished (or stopped) run's pace into the estimate."""

        painter = self._painter
        measured = getattr(painter, "paint_phase_timing", None) if painter else None
        if measured is None:
            return
        if getattr(getattr(painter, "input", None), "emits_real_input", True) is False:
            return  # a dry run skips the holds the estimate is about
        before = self._learned_timing.overhead_seconds
        # Time spent checking colors and repainting the game's dropped
        # presses is this sign's, not this machine's per-stroke cost; it is
        # learned on its own, as is the touch-up pass when one ran to its end.
        checking = float(getattr(measured, "checking_seconds", 0.0) or 0.0)
        capture = float(getattr(measured, "check_capture_seconds", 0.0) or 0.0)
        colors_checked = int(getattr(measured, "colors_checked", 0) or 0)
        touch_up = getattr(painter, "touch_up_timing", None)
        learned = self._learned_timing.observe(
            predicted_seconds=measured.predicted_seconds,
            actual_seconds=max(0.0, measured.actual_seconds - checking),
            strokes=measured.strokes,
            colors_checked=colors_checked,
            check_capture_seconds=capture,
            check_repaint_seconds=max(0.0, checking - capture),
            touch_up_seconds=(
                float(touch_up.seconds) if touch_up is not None else None
            ),
        )
        if not learned:
            return
        LOGGER.info(
            "Run timing: predicted %.0fs, took %.0fs over %d strokes (%.0fs of it "
            "checking %d colors and repainting); per-stroke overhead %.1f ms -> "
            "%.1f ms, a check %.2fs a color plus %.0f%% of the painting in "
            "repaints, the touch-up %s%.0f%% of the painting",
            measured.predicted_seconds,
            measured.actual_seconds,
            measured.strokes,
            checking,
            colors_checked,
            before * 1000.0,
            self._learned_timing.overhead_seconds * 1000.0,
            self._learned_timing.check_capture_seconds,
            self._learned_timing.check_repaint_fraction * 100.0,
            (
                f"took {touch_up.seconds:.0f}s in {touch_up.passes} pass"
                f"{'es' if touch_up.passes != 1 else ''}, now "
                if touch_up is not None
                else ""
            ),
            self._learned_timing.touch_up_fraction * 100.0,
        )
        try:
            self._learned_timing.save(self._timing_path())
        except OSError:
            LOGGER.warning("Could not save the learned timing", exc_info=True)
        self._refresh_statistics()

    @staticmethod
    def _format_duration(seconds: float) -> str:
        seconds = max(0, int(round(seconds)))
        hours, remainder = divmod(seconds, 3600)
        minutes, secs = divmod(remainder, 60)
        if hours:
            return f"{hours}h {minutes:02d}m"
        if minutes:
            return f"{minutes}m {secs:02d}s"
        return f"{secs}s"

    @staticmethod
    def _pil_to_pixmap(image: Image.Image) -> QPixmap:
        qt_image = ImageQt.ImageQt(image.convert("RGBA"))
        return QPixmap.fromImage(QImage(qt_image).copy())

    def _canvas_aspect_ratio(self) -> float:
        rect = self._profile_rect("canvas")
        return rect.aspect_ratio if rect else 2.0

    # Service integration methods are kept below the visual/image code so that
    # the platform-specific pieces remain easy to audit.

    # -------------------------------------------------------- profiles/settings

    def _install_logging_handler(self) -> None:
        handler = QtLogHandler(self.log_view.appendPlainText)
        handler.setLevel(logging.INFO)
        handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)s  %(message)s", "%H:%M:%S"))
        logging.getLogger("rust_painter").addHandler(handler)
        self._qt_log_handler = handler

    @staticmethod
    def _local_data_directory() -> Path:
        override = os.environ.get("RUST_PAINTER_DATA_DIR")
        if override:
            root = Path(override).expanduser()
            root.mkdir(parents=True, exist_ok=True)
            return root
        local = os.environ.get("LOCALAPPDATA")
        root = Path(local) / "RustPainter" if local else Path.cwd() / "data"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _initialize_services(self) -> None:
        data = self._local_data_directory()
        self._profile_store = ProfileStore(data / "profiles")
        self._settings_store = SettingsStore(data / "settings.json")
        self._resume_store = ResumeRecordStore(data / "runs" / "resume")
        try:
            self._settings = self._settings_store.load()
        except Exception as exc:
            LOGGER.exception("Could not load settings; using defaults")
            self._settings = default_settings()
            QMessageBox.warning(
                self,
                "Settings could not be loaded",
                f"Defaults will be used for this session.\n\n{exc}",
            )
        self._apply_settings(self._settings)
        stored_geometry = self._settings.get("ui", {}).get("window_geometry")
        if isinstance(stored_geometry, str) and stored_geometry:
            try:
                self.restoreGeometry(
                    QByteArray.fromBase64(stored_geometry.encode("ascii"))
                )
            except Exception:
                LOGGER.warning("Could not restore the saved window geometry")
        self._connect_service_controls()
        self._reload_profiles(self._settings.get("ui", {}).get("selected_profile_id"))
        self._register_hotkeys()
        self._refresh_timelapse_sessions()
        if sys.platform == "win32":
            self._rust_monitor_timer.start()

    def _connect_service_controls(self) -> None:
        self.profile_combo.currentIndexChanged.connect(self._on_profile_selected)
        self.new_profile_button.clicked.connect(self._new_profile)
        self.rename_profile_button.clicked.connect(self._rename_profile)
        self.delete_profile_button.clicked.connect(self._delete_profile)
        self.calibrate_canvas_button.clicked.connect(
            lambda: self._begin_calibration("canvas", "sign paintable canvas")
        )
        self.calibrate_color_box_button.clicked.connect(
            lambda: self._begin_calibration("color_box", "saturation / value box")
        )
        self.calibrate_hue_bar_button.clicked.connect(
            lambda: self._begin_calibration("hue_bar", "vertical hue bar")
        )
        self.calibrate_brush_button.clicked.connect(
            lambda: self._begin_calibration(
                "brush_size_box", "numeric Size field beside the size slider"
            )
        )
        self.calibrate_clear_button.clicked.connect(
            lambda: self._begin_calibration(
                "clear_button", "trash / clear icon that wipes the sign"
            )
        )
        self.calibrate_save_button.clicked.connect(
            lambda: self._begin_calibration(
                "save_button", "Save changes button that closes the painting UI"
            )
        )
        # Whether the Save button is needed follows the switch.
        self.anti_afk_check.toggled.connect(
            lambda _checked: self._refresh_profile_ui()
        )
        self.prepare_color_chart_button.clicked.connect(self._prepare_color_chart)
        self.measure_color_chart_button.clicked.connect(self._measure_color_chart)
        self.clear_color_correction_button.clicked.connect(self._clear_color_correction)
        self.start_button.clicked.connect(self._start_or_resume)
        self.pause_button.clicked.connect(self._pause_painting)
        self.abort_button.clicked.connect(self._abort_painting)
        self.capture_reference_button.clicked.connect(self._capture_reference)
        for name, button in self.debug_buttons.items():
            button.clicked.connect(lambda _checked=False, action=name: self._run_debug_action(action))

        self.show_calibration_check.toggled.connect(self._on_show_calibration_toggled)
        self.show_status_check.toggled.connect(self._on_show_calibration_toggled)
        self.move_to_rust_button.clicked.connect(self._move_calibration_to_rust_monitor)
        self.timelapse_speed_slider.valueChanged.connect(
            self._refresh_timelapse_speed_label
        )
        self.timelapse_interval_spin.valueChanged.connect(
            self._refresh_timelapse_speed_label
        )
        self.export_preview_button.clicked.connect(self._export_rust_preview)
        self.open_timelapse_button.clicked.connect(self._open_timelapse_folder)
        self.open_session_button.clicked.connect(self._open_selected_session)
        self.delete_session_button.clicked.connect(self._delete_selected_sessions)
        self.refresh_sessions_button.clicked.connect(self._refresh_timelapse_sessions)
        self.play_session_button.clicked.connect(self._play_selected_session)
        self.export_session_button.clicked.connect(self._export_selected_session)
        self.timelapse_sessions.itemSelectionChanged.connect(
            self._sync_session_buttons
        )
        self.timelapse_sessions.itemDoubleClicked.connect(
            lambda _item: self._play_selected_session()
        )
        # Delete is what the key is for everywhere else a list of files is
        # shown, and it is scoped to the list so it cannot fire from elsewhere
        # on the page.
        self.delete_session_shortcut = QShortcut(
            QKeySequence.StandardKey.Delete, self.timelapse_sessions
        )
        self.delete_session_shortcut.setContext(
            Qt.ShortcutContext.WidgetShortcut
        )
        self.delete_session_shortcut.activated.connect(self._delete_selected_sessions)

        settings_controls = (
            self.scale_mode_combo,
            self.crop_alignment_combo,
            self.background_combo,
            self.transparency_combo,
            self.alpha_fill_check,
            self.quality_combo,
            self.paint_mode_combo,
            self.logical_width_spin,
            self.logical_height_spin,
            self.color_count_combo,
            self.dither_check,
            self.sharpen_combo,
            self.merge_combo,
            self.show_calibration_check,
            self.show_status_check,
            self.background_color_button,
            self.remove_background_check,
            self.removal_source_combo,
            self.removal_color_button,
            self.removal_tolerance_spin,
            self.removal_scope_combo,
            self.pixel_spacing_spin,
            self.stroke_speed_spin,
            self.dot_duration_spin,
            self.hue_delay_spin,
            self.sv_delay_spin,
            self.brush_delay_spin,
            self.stroke_delay_spin,
            self.color_delay_spin,
            self.interpolation_spin,
            self.line_tool_check,
            self.press_hold_check,
            self.dab_size_check,
            self.apply_brush_check,
            self.verify_passes_spin,
            self.verify_picks_check,
            self.confirm_strokes_check,
            self.confirm_rounds_spin,
            self.timelapse_check,
            self.timelapse_interval_spin,
            self.timelapse_final_check,
            self.timelapse_speed_slider,
            self.timelapse_format_combo,
            self.countdown_spin,
            self.dry_run_check,
            self.focus_guard_check,
            self.expected_window_edit,
            self.expected_process_edit,
            self.mouse_pause_check,
            self.verify_ui_check,
            self.ui_guard_check,
            self.anti_afk_check,
            self.anti_afk_interval_spin,
            self.start_hotkey_combo,
            self.pause_hotkey_combo,
            self.abort_hotkey_combo,
        )
        for control in settings_controls:
            if isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self._schedule_settings_save)
            elif isinstance(control, (QSpinBox, QDoubleSpinBox, QSlider)):
                control.valueChanged.connect(self._schedule_settings_save)
            elif isinstance(control, QCheckBox):
                control.toggled.connect(self._schedule_settings_save)
            elif isinstance(control, QLineEdit):
                control.textChanged.connect(self._schedule_settings_save)
            elif isinstance(control, ColorButton):
                control.colorChanged.connect(self._schedule_settings_save)

        self.start_hotkey_combo.currentIndexChanged.connect(self._register_hotkeys)
        self.pause_hotkey_combo.currentIndexChanged.connect(self._register_hotkeys)
        self.abort_hotkey_combo.currentIndexChanged.connect(self._register_hotkeys)
        self.dry_run_check.toggled.connect(self._update_start_availability)

    @staticmethod
    def _set_combo_data(combo: QComboBox, value: Any) -> None:
        index = combo.findData(value)
        if index >= 0:
            combo.setCurrentIndex(index)

    def _apply_settings(self, settings: dict[str, Any]) -> None:
        image = settings.get("image", {})
        painting = settings.get("painting", {})
        hotkeys = settings.get("hotkeys", {})
        safety = settings.get("safety", {})
        execution = settings.get("execution", {})
        controls = self.findChildren(QWidget)
        for control in controls:
            control.blockSignals(True)
        try:
            self._set_combo_data(self.scale_mode_combo, image.get("scale_mode", "fit"))
            stored_focus = image.get("crop_focus")
            self._crop_focus = (
                (float(stored_focus[0]), float(stored_focus[1]))
                if isinstance(stored_focus, (list, tuple)) and len(stored_focus) == 2
                else None
            )
            self._last_named_crop = str(image.get("crop_alignment", "center"))
            self._set_combo_data(
                self.crop_alignment_combo,
                CUSTOM_CROP_VALUE if self._crop_focus else self._last_named_crop,
            )
            preset = str(image.get("quality_preset", "balanced")).replace("_", " ").title()
            if self.quality_combo.findText(preset) >= 0:
                self.quality_combo.setCurrentText(preset)
            self._set_combo_data(
                self.paint_mode_combo, str(image.get("paint_mode", "balanced"))
            )
            self.logical_width_spin.setValue(int(image.get("logical_width", 256)))
            self.logical_height_spin.setValue(int(image.get("logical_height", 128)))
            self._set_combo_data(
                self.color_count_combo,
                int(image.get("color_count", DEFAULT_COLOR_COUNT)),
            )
            self.dither_check.setChecked(bool(image.get("dithering", False)))
            self._set_combo_data(
                self.sharpen_combo, str(image.get("sharpen", SharpenMode.LIGHT.value))
            )
            self._set_combo_data(
                self.background_combo, image.get("background_mode", "unpainted")
            )
            self.background_color_button.set_color(image.get("background_color", "#ffffff"))
            self._set_combo_data(
                self.transparency_combo,
                image.get("transparent_pixels", "leave_unpainted"),
            )
            self.alpha_fill_check.setChecked(bool(image.get("alpha_fill", False)))
            self.remove_background_check.setChecked(
                bool(image.get("remove_background", False))
            )
            self._set_combo_data(
                self.removal_source_combo,
                image.get("background_removal_source", "auto"),
            )
            self.removal_color_button.set_color(
                image.get("background_removal_color", "#FFFFFF")
            )
            self.removal_tolerance_spin.setValue(
                int(image.get("background_removal_tolerance", 12))
            )
            self._set_combo_data(
                self.removal_scope_combo,
                image.get("background_removal_scope", "subject"),
            )
            text_overlay = image.get("text_overlay", {})
            layer_values = text_overlay.get("layers", [])
            if (
                bool(text_overlay.get("enabled", False))
                and str(text_overlay.get("text", "")).strip()
                and (
                    not layer_values
                    or not any(str(layer.get("text", "")).strip() for layer in layer_values)
                )
            ):
                position_y = {"top": 0.12, "center": 0.5, "bottom": 0.88}.get(
                    str(text_overlay.get("position", "center")), 0.5
                )
                layer_values = [
                    {
                        "text": text_overlay.get("text", ""),
                        "font_family": text_overlay.get("font_family", ""),
                        "font_size": text_overlay.get("font_size", 24),
                        "color": text_overlay.get("color", "#FFFFFF"),
                        "x": 0.5,
                        "y": position_y,
                        "bold": text_overlay.get("bold", False),
                        "italic": text_overlay.get("italic", False),
                    }
                ]
            self._text_layers = []
            for layer_value in layer_values:
                color = QColor(str(layer_value.get("color", "#FFFFFF")))
                font_size = int(layer_value.get("font_size", 24))
                # Documents written before sizes were stored as a ratio only
                # have pixels, which belong to the resolution saved with them.
                gradient_color = QColor(
                    str(layer_value.get("gradient_color", "#FFFFFF"))
                )
                outline_color = QColor(str(layer_value.get("outline_color", "#000000")))
                direction = str(layer_value.get("gradient_direction", "vertical"))
                self._text_layers.append(
                    _TextOverlayOptions(
                        text=str(layer_value.get("text", "")),
                        font_family=str(layer_value.get("font_family", "")),
                        font_size=font_size,
                        color=_rgb(color),
                        x=float(layer_value.get("x", 0.5)),
                        y=float(layer_value.get("y", 0.5)),
                        bold=bool(layer_value.get("bold", False)),
                        italic=bool(layer_value.get("italic", False)),
                        size_ratio=float(layer_value.get("size_ratio", 0.0))
                        or self._text_size_ratio(font_size),
                        gradient=bool(layer_value.get("gradient", False)),
                        gradient_color=_rgb(gradient_color),
                        gradient_direction=direction
                        if direction in GRADIENT_DIRECTIONS
                        else "vertical",
                        outline_width=min(
                            max(int(layer_value.get("outline_width", 0)), 0),
                            MAX_OUTLINE_WIDTH,
                        ),
                        outline_color=_rgb(outline_color),
                    )
                )
            if not self._text_layers:
                self._text_layers = [
                    _TextOverlayOptions(
                        "",
                        "",
                        24,
                        (255, 255, 255),
                        size_ratio=self._text_size_ratio(24),
                    )
                ]
            self._selected_text_layer = 0
            self._selected_text_indices = [0]
            self._rebuild_text_layer_combo()
            self._sync_text_controls()
            # Loading a document replaces every layer at once, which is not
            # something the undo history should be able to walk back into.
            self._reset_text_history()

            self.pixel_spacing_spin.setValue(
                float(painting.get("logical_pixel_spacing", 1.0))
            )
            self.stroke_speed_spin.setValue(
                float(painting.get("stroke_speed_pixels_per_second", 700.0))
            )
            self.dot_duration_spin.setValue(
                round(float(painting.get("mouse_down_duration_seconds", 0.07)) * 1000)
            )
            self.hue_delay_spin.setValue(
                round(float(painting.get("delay_after_hue_seconds", 0.09)) * 1000)
            )
            self.sv_delay_spin.setValue(
                round(
                    float(painting.get("delay_after_saturation_value_seconds", 0.09))
                    * 1000
                )
            )
            self.brush_delay_spin.setValue(
                round(float(painting.get("delay_after_brush_seconds", 0.07)) * 1000)
            )
            self.stroke_delay_spin.setValue(
                round(float(painting.get("delay_between_strokes_seconds", 0.02)) * 1000)
            )
            self.color_delay_spin.setValue(
                round(float(painting.get("delay_between_colors_seconds", 0.12)) * 1000)
            )
            self.interpolation_spin.setValue(
                float(painting.get("stroke_interpolation_step_pixels", 4.0))
            )
            self.apply_brush_check.setChecked(bool(painting.get("apply_brush_size", False)))
            self.line_tool_check.setChecked(bool(painting.get("use_line_tool", True)))
            self.press_hold_check.setChecked(
                bool(painting.get("measure_press_hold", True))
            )
            self.dab_size_check.setChecked(bool(painting.get("measure_dab_size", True)))
            self.verify_passes_spin.setValue(int(painting.get("verify_passes", 2)))
            self.confirm_strokes_check.setChecked(
                bool(painting.get("confirm_strokes", False))
            )
            self.confirm_rounds_spin.setValue(int(painting.get("confirm_max_rounds", 4)))
            self.confirm_rounds_spin.setEnabled(self.confirm_strokes_check.isChecked())
            self.verify_picks_check.setChecked(bool(painting.get("verify_color_picks", True)))
            timelapse = settings.get("timelapse", {})
            self.timelapse_check.setChecked(bool(timelapse.get("enabled", False)))
            self.timelapse_interval_spin.setValue(
                int(timelapse.get("interval_seconds", 10))
            )
            self.timelapse_final_check.setChecked(
                bool(timelapse.get("capture_final_frame", True))
            )
            self.timelapse_speed_slider.setValue(
                int(timelapse.get("playback_frame_rate", DEFAULT_FRAME_RATE))
            )
            self._set_combo_data(
                self.timelapse_format_combo, str(timelapse.get("export_format", "avi"))
            )
            merge_mode = str(painting.get("stroke_merge_mode", "balanced"))
            if merge_mode not in MERGE_MODE_GAPS:
                merge_mode = "balanced"
            self._merge_mode_choice = merge_mode
            self.merge_combo.setCurrentIndex(self.merge_combo.findData(merge_mode))
            self.speed_preset_combo.setCurrentText(self._detect_speed_preset())
            ui = settings.get("ui", {})
            self.show_status_check.setChecked(
                bool(ui.get("show_status_overlay", True))
            )
            self.show_calibration_check.setChecked(
                bool(ui.get("show_calibration_overlay", True))
            )
            self.smooth_preview_check.setChecked(
                bool(ui.get("smooth_rust_preview", True))
            )
            # Signals are held while settings load, so the label is told
            # directly rather than through the checkbox's toggle.
            self.paint_preview.set_smooth(self.smooth_preview_check.isChecked())

            self.countdown_spin.setValue(int(safety.get("countdown_seconds", 3)))
            self.mouse_pause_check.setChecked(bool(safety.get("pause_on_mouse_move", True)))
            self.focus_guard_check.setChecked(
                bool(safety.get("require_rust_foreground", True))
            )
            self.expected_window_edit.setText(
                str(safety.get("expected_window_title_contains", "Rust") or "")
            )
            self.expected_process_edit.setText(
                str(safety.get("expected_process_name", "") or "")
            )
            self.verify_ui_check.setChecked(bool(safety.get("verify_calibrated_ui", False)))
            self.ui_guard_check.setChecked(bool(safety.get("ui_guard_enabled", True)))
            self.anti_afk_check.setChecked(bool(safety.get("anti_afk_enabled", False)))
            self.anti_afk_interval_spin.setValue(
                int(safety.get("anti_afk_interval_minutes", 30))
            )
            self.dry_run_check.setChecked(bool(execution.get("dry_run", False)))
            self.start_hotkey_combo.setCurrentText(str(hotkeys.get("start_resume", "F8")))
            self.pause_hotkey_combo.setCurrentText(str(hotkeys.get("pause", "F9")))
            self.abort_hotkey_combo.setCurrentText(str(hotkeys.get("abort", "F10")))
        finally:
            for control in controls:
                control.blockSignals(False)
        logging.getLogger("rust_painter.input").setLevel(
            logging.DEBUG
            if bool(execution.get("debug_mouse_logging", False))
            else logging.INFO
        )
        self._on_transparency_changed()
        self._on_background_removal_changed()
        self._sync_paint_mode_dependent_controls()
        self._refresh_timelapse_speed_label()
        self._refresh_text_editor_layers()

    def _settings_document(self) -> dict[str, Any]:
        current = self._settings.copy()
        current["image"] = {
            **current.get("image", {}),
            "scale_mode": self.scale_mode_combo.currentData(),
            "crop_alignment": self._named_crop_alignment(),
            "crop_focus": list(self._crop_focus) if self._crop_focus else None,
            "quality_preset": self.quality_combo.currentText().lower().replace(" ", "_"),
            "paint_mode": str(self.paint_mode_combo.currentData() or "balanced"),
            "logical_width": self.logical_width_spin.value(),
            "logical_height": self.logical_height_spin.value(),
            "color_count": int(self.color_count_combo.currentData()),
            "dithering": self.dither_check.isChecked(),
            "sharpen": str(self.sharpen_combo.currentData() or SharpenMode.LIGHT.value),
            "background_mode": self.background_combo.currentData(),
            "background_color": self.background_color_button.color().name().upper(),
            "transparent_pixels": self.transparency_combo.currentData(),
            "alpha_fill": self.alpha_fill_check.isChecked(),
            "remove_background": self.remove_background_check.isChecked(),
            "background_removal_source": self.removal_source_combo.currentData(),
            "background_removal_color": self.removal_color_button.color()
            .name()
            .upper(),
            "background_removal_tolerance": self.removal_tolerance_spin.value(),
            "background_removal_scope": self.removal_scope_combo.currentData(),
            "text_overlay": {
                "layers": [
                    {
                        "text": layer.text,
                        "font_family": layer.font_family,
                        "font_size": layer.font_size,
                        "size_ratio": layer.size_ratio
                        or self._text_size_ratio(layer.font_size),
                        "color": "#{:02X}{:02X}{:02X}".format(*layer.color),
                        "x": layer.x,
                        "y": layer.y,
                        "bold": layer.bold,
                        "italic": layer.italic,
                        "gradient": layer.gradient,
                        "gradient_color": "#{:02X}{:02X}{:02X}".format(
                            *layer.gradient_color
                        ),
                        "gradient_direction": layer.gradient_direction,
                        "outline_width": layer.outline_width,
                        "outline_color": "#{:02X}{:02X}{:02X}".format(
                            *layer.outline_color
                        ),
                    }
                    for layer in self._text_layers
                ]
            },
        }
        current["painting"] = {
            **current.get("painting", {}),
            # Retain legacy keys so old settings documents remain readable,
            # while the UI now derives size from the calibrated preview.
            "brush_size": 0.0,
            "logical_pixel_spacing": self.pixel_spacing_spin.value(),
            "stroke_speed_pixels_per_second": self.stroke_speed_spin.value(),
            "mouse_down_duration_seconds": self.dot_duration_spin.value() / 1000.0,
            "delay_after_hue_seconds": self.hue_delay_spin.value() / 1000.0,
            "delay_after_saturation_value_seconds": self.sv_delay_spin.value() / 1000.0,
            "delay_after_brush_seconds": self.brush_delay_spin.value() / 1000.0,
            "delay_between_strokes_seconds": self.stroke_delay_spin.value() / 1000.0,
            "delay_between_colors_seconds": self.color_delay_spin.value() / 1000.0,
            "stroke_interpolation_step_pixels": self.interpolation_spin.value(),
            "apply_brush_size": self.apply_brush_check.isChecked(),
            "use_line_tool": self.line_tool_check.isChecked(),
            "measure_press_hold": self.press_hold_check.isChecked(),
            "measure_dab_size": self.dab_size_check.isChecked(),
            "brush_direction": "low_to_high",
            "stroke_merge_mode": self._merge_mode(),
            "verify_passes": int(self.verify_passes_spin.value()),
            "confirm_strokes": self.confirm_strokes_check.isChecked(),
            "confirm_max_rounds": int(self.confirm_rounds_spin.value()),
            "verify_color_picks": self.verify_picks_check.isChecked(),
        }
        current["timelapse"] = {
            **current.get("timelapse", {}),
            "enabled": self.timelapse_check.isChecked(),
            "interval_seconds": self.timelapse_interval_spin.value(),
            "capture_final_frame": self.timelapse_final_check.isChecked(),
            "playback_frame_rate": self.timelapse_speed_slider.value(),
            "export_format": str(self.timelapse_format_combo.currentData() or "avi"),
        }
        current["hotkeys"] = {
            **current.get("hotkeys", {}),
            "start_resume": self.start_hotkey_combo.currentText(),
            "pause": self.pause_hotkey_combo.currentText(),
            "abort": self.abort_hotkey_combo.currentText(),
        }
        current["safety"] = {
            **current.get("safety", {}),
            "countdown_seconds": self.countdown_spin.value(),
            "pause_on_mouse_move": self.mouse_pause_check.isChecked(),
            "require_rust_foreground": self.focus_guard_check.isChecked(),
            "expected_window_title_contains": self.expected_window_edit.text().strip(),
            "expected_process_name": self.expected_process_edit.text().strip(),
            "verify_calibrated_ui": self.verify_ui_check.isChecked(),
            "ui_guard_enabled": self.ui_guard_check.isChecked(),
            "anti_afk_enabled": self.anti_afk_check.isChecked(),
            "anti_afk_interval_minutes": self.anti_afk_interval_spin.value(),
        }
        current["execution"] = {
            **current.get("execution", {}),
            "dry_run": self.dry_run_check.isChecked(),
        }
        current["ui"] = {
            **current.get("ui", {}),
            "selected_profile_id": self._current_profile.id if self._current_profile else None,
            "last_image_path": str(self._image_path) if self._image_path else None,
            "show_calibration_overlay": self.show_calibration_check.isChecked(),
            "show_status_overlay": self.show_status_check.isChecked(),
            "smooth_rust_preview": self.smooth_preview_check.isChecked(),
            "window_geometry": bytes(self.saveGeometry().toBase64()).decode("ascii"),
        }
        return current

    @Slot()
    def _schedule_settings_save(self, *_args: Any) -> None:
        self._settings_timer.start()

    @Slot()
    def _save_settings(self) -> None:
        if self._settings_store is None:
            return
        try:
            self._settings = self._settings_store.save(self._settings_document())
        except Exception as exc:
            LOGGER.exception("Could not save settings")
            self.statusBar().showMessage(f"Could not save settings: {exc}", 8000)

    def _reload_profiles(self, preferred_id: str | None = None) -> None:
        try:
            profiles = self._profile_store.list_profiles()
            if not profiles:
                profiles = [self._profile_store.ensure_default_profile("Large Wooden Sign")]
        except Exception as exc:
            LOGGER.exception("Could not load profiles")
            QMessageBox.critical(self, "Could not load profiles", str(exc))
            return
        preferred_id = preferred_id or (
            self._current_profile.id if self._current_profile else None
        )
        self.profile_combo.blockSignals(True)
        self.profile_combo.clear()
        for profile in profiles:
            self.profile_combo.addItem(profile.name, profile.id)
        index = self.profile_combo.findData(preferred_id)
        if index < 0:
            default = self._profile_store.get_default()
            index = self.profile_combo.findData(default.id if default else profiles[0].id)
        self.profile_combo.setCurrentIndex(max(0, index))
        self.profile_combo.blockSignals(False)
        self._on_profile_selected()

    @Slot()
    def _on_profile_selected(self, *_args: Any) -> None:
        profile_id = self.profile_combo.currentData()
        if not profile_id:
            self._current_profile = None
            self._refresh_profile_ui()
            return
        try:
            self._current_profile = self._profile_store.require(str(profile_id))
            self._profile_store.set_default(self._current_profile.id)
        except Exception as exc:
            LOGGER.exception("Could not select profile")
            QMessageBox.warning(self, "Profile error", str(exc))
            return
        LOGGER.info("Loaded profile: %s", self._current_profile.name)
        self._refresh_profile_ui()
        self._update_quality_dimensions()
        self._schedule_settings_save()

    def _refresh_profile_ui(self) -> None:
        profile = self._current_profile
        status = profile.calibration_status if profile else {}
        self.canvas_status.set_calibrated(bool(status.get("canvas")))
        self.color_box_status.set_calibrated(bool(status.get("color_box")))
        self.hue_bar_status.set_calibrated(bool(status.get("hue_bar")))
        # Automatic sizing measures the brush on every run and wipes the
        # probes afterwards, so it needs both the Size field and the control
        # that clears the sign; with it off, neither is used.
        brush_optional = not self.apply_brush_check.isChecked()
        self.brush_size_box_status.set_calibrated(
            bool(status.get("brush_size_box")), brush_optional
        )
        self.clear_button_status.set_calibrated(
            bool(status.get("clear_button")), brush_optional
        )
        # The anti-AFK break leaves the painting UI through Save; with the
        # break off, the button is never clicked.
        self.save_button_status.set_calibrated(
            bool(status.get("save_button")), not self.anti_afk_check.isChecked()
        )
        self._refresh_brush_model_status()
        if self._refresh_quality_preset_availability():
            self._update_quality_dimensions()
        self._refresh_resolution_cap_notice()
        correction = (
            profile.metadata.get("color_correction")
            if profile and isinstance(profile.metadata, dict)
            else None
        )
        # The preview renders artwork through this model, so a measured, cleared,
        # or switched-to correction has to rebuild it.
        if correction != self._preview_correction:
            self._preview_correction = deepcopy(correction)
            self._schedule_processing()
        if isinstance(correction, dict):
            try:
                error = float(correction.get("fitRmse", 0.0)) * 255.0
                if not math.isfinite(error) or error < 0:
                    raise ValueError
                self.color_correction_status.setText(
                    f"Measured for this profile • fit error {error:.1f} RGB levels"
                )
            except (TypeError, ValueError):
                self.color_correction_status.setText(
                    "Stored correction is invalid • clear it and measure again"
                )
        elif self._painting_calibration_chart(profile):
            self.color_correction_status.setText(
                "Chart prepared • paint it, then click Measure Painted Chart"
            )
        else:
            self.color_correction_status.setText("Not measured")
        self.clear_color_correction_button.setEnabled(isinstance(correction, dict))
        if profile and profile.canvas:
            rect = profile.canvas
            self.canvas_geometry_label.setText(
                f"Canvas: {rect.width:,} × {rect.height:,} px  •  "
                f"Aspect: {rect.aspect_ratio:.4f}"
            )
        else:
            self.canvas_geometry_label.setText("Canvas: not calibrated  •  Aspect: —")
        self._refresh_max_quality_hint()
        self._refresh_display_warning()
        self._update_start_availability()

    def _refresh_max_quality_hint(self) -> None:
        """Say what grid Max quality will plan against right now.

        Max means "one logical cell per sign texel" once a job has measured
        the sign; before that it plans on the calibrated canvas's screen-pixel
        grid, which brush size 1 paints without losing detail.  The tooltip
        carries the distinction, so the entry never has to be greyed out and
        the selection survives a measurement coming or going - the grid it
        resolves to is re-derived wherever the model or canvas changes.
        """

        index = self.quality_combo.findText(MAX_QUALITY_PRESET)
        if index < 0:
            return
        item = self.quality_combo.model().item(index)
        if item is None:
            return
        item.setToolTip(
            "One logical cell per sign texture pixel"
            if self._sign_resolution_cap() is not None
            else "One logical cell per screen pixel until a paint job with "
            "automatic brush sizing measures this sign's texture"
        )

    @Slot()
    def _new_profile(self) -> None:
        dialog = _NameDialog("New sign profile", "Custom Sign", self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        source = self._current_profile
        try:
            profile = self._profile_store.create(dialog.name, make_default=True)
            if source is not None:
                # Assume the physical setup is unchanged: start the new profile
                # from the current calibration instead of empty rectangles. Any
                # region can still be recalibrated individually afterwards.
                candidate = Profile.from_dict(profile.to_dict())
                for field in (
                    "canvas",
                    "color_box",
                    "hue_bar",
                    "brush_size_box",
                    "clear_button",
                    "save_button",
                ):
                    setattr(candidate, field, getattr(source, field, None))
                candidate.display = source.display
                if isinstance(source.metadata, dict):
                    for key in ("color_correction", "brush_size_model", "texel_grid"):
                        if key in source.metadata:
                            candidate.metadata[key] = deepcopy(source.metadata[key])
                profile = self._profile_store.save(candidate)
                LOGGER.info(
                    "New profile %s inherited calibration from %s",
                    profile.name,
                    source.name,
                )
        except Exception as exc:
            QMessageBox.warning(self, "Could not create profile", str(exc))
            return
        self._reload_profiles(profile.id)

    @Slot()
    def _rename_profile(self) -> None:
        if self._current_profile is None:
            return
        dialog = _NameDialog("Rename sign profile", self._current_profile.name, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        try:
            profile = self._profile_store.rename(self._current_profile.id, dialog.name)
        except Exception as exc:
            QMessageBox.warning(self, "Could not rename profile", str(exc))
            return
        self._reload_profiles(profile.id)

    @Slot()
    def _delete_profile(self) -> None:
        profile = self._current_profile
        if profile is None:
            return
        answer = QMessageBox.question(
            self,
            "Delete profile",
            f"Delete “{profile.name}” and its calibration rectangles?",
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            self._profile_store.delete(profile.id)
            self._current_profile = None
            self._reload_profiles()
        except Exception as exc:
            QMessageBox.warning(self, "Could not delete profile", str(exc))

    def _profile_rect(self, name: str) -> ScreenRect | None:
        if self._current_profile is None:
            return None
        value = getattr(self._current_profile, name, None)
        return value if isinstance(value, ScreenRect) else None

    def _begin_calibration(self, field: str, description: str) -> None:
        if self._current_profile is None:
            QMessageBox.information(self, "Create a profile", "Create a sign profile first.")
            return
        operation_active = (
            self._painter_is_active()
            or self._debug_running
            or self._countdown_callback_running
            or bool(self._countdown and self._countdown.isVisible())
        )
        if operation_active:
            QMessageBox.warning(
                self,
                "Operation is active",
                "Stop or wait for the current operation before changing calibration.",
            )
            return
        instruction = f"Drag just inside the {description}"
        if self._calibration_preview is not None:
            self._calibration_preview.hide()
        try:
            rectangle = select_screen_rect(self, instruction, minimum_size=3)
        except Exception as exc:
            LOGGER.exception("Calibration overlay failed")
            QMessageBox.critical(self, "Calibration failed", str(exc))
            self._update_calibration_overlay()
            return
        if rectangle is None:
            self.statusBar().showMessage("Calibration cancelled", 4000)
            self._update_calibration_overlay()
            return
        try:
            current_display = capture_display_metadata()
            display_changes = (
                self._current_profile.display.differences(current_display)
                if self._current_profile.display is not None
                else []
            )
        except Exception as exc:
            LOGGER.exception("Could not capture display layout for calibration")
            QMessageBox.critical(self, "Calibration failed", str(exc))
            self._update_calibration_overlay()
            return
        # Work on a detached candidate so a persistence failure cannot leave
        # the live profile half-cleared while the stored profile stays intact.
        candidate = Profile.from_dict(self._current_profile.to_dict())
        if display_changes:
            # Rectangle coordinates are one coherent set. Mixing old and new
            # display layouts would make the profile look current while some
            # targets still point at the previous monitor arrangement.
            for other in (
                "canvas",
                "color_box",
                "hue_bar",
                "brush_size_box",
                "clear_button",
                "save_button",
            ):
                if other != field:
                    setattr(candidate, other, None)
            candidate.metadata.pop("ui_reference", None)
        setattr(candidate, field, rectangle)
        removed_reference: Any = None
        if field in {"color_box", "hue_bar"}:
            removed_reference = candidate.metadata.pop("ui_reference", None)
        if field in {"canvas", "color_box", "hue_bar"}:
            candidate.metadata.pop("color_correction", None)
        if field == "canvas" and self._canvas_shape_changed(rectangle):
            # The brush model is a fraction of the sign, so re-framing the same
            # sign keeps it valid however the zoom changed.  A different aspect
            # ratio means a different sign, whose texture resolution the old
            # measurement says nothing about - and neither does its texel
            # count.
            candidate.metadata.pop("brush_size_model", None)
            candidate.metadata.pop("texel_grid", None)
        try:
            candidate.display = current_display
            self._current_profile = self._profile_store.save(candidate)
        except Exception as exc:
            LOGGER.exception("Could not save calibration")
            QMessageBox.critical(self, "Could not save calibration", str(exc))
            self._update_calibration_overlay()
            return
        if display_changes:
            LOGGER.warning(
                "Display changed during profile recalibration; invalidated other rectangles: %s",
                "; ".join(display_changes),
            )
            QMessageBox.information(
                self,
                "Complete recalibration required",
                "The display layout changed, so the other stored rectangles were cleared. "
                "Recalibrate the remaining canvas and picker regions before painting.",
            )
        elif removed_reference is not None:
            LOGGER.info("Picker calibration changed; invalidated the saved UI reference")
        LOGGER.info(
            "Calibrated %s: left=%d top=%d width=%d height=%d",
            field,
            rectangle.left,
            rectangle.top,
            rectangle.width,
            rectangle.height,
        )
        self._refresh_profile_ui()
        if field == "canvas":
            self._update_quality_dimensions()
        elif field == "brush_size_box":
            # This calibration changes what the optimizer may plan with.
            self._schedule_processing()

    def _refresh_display_warning(self) -> None:
        profile = self._current_profile
        if profile is None or profile.display is None:
            self.display_warning_label.setText("")
            return
        try:
            current = capture_display_metadata()
            differences = profile.display.differences(current)
        except Exception as exc:
            LOGGER.warning("Could not compare display layout: %s", exc)
            self.display_warning_label.setText("Display layout could not be verified.")
            return
        if differences:
            self.display_warning_label.setText(
                "⚠ Display changed: " + "; ".join(differences) + ". Recalibrate before painting."
            )
        else:
            self.display_warning_label.setText("")

    def _rust_monitor_mismatch(self) -> tuple[ScreenRect, ScreenRect] | None:
        """(calibrated monitor, Rust's monitor) when they differ, else None."""

        canvas = self._profile_rect("canvas")
        if canvas is None:
            return None
        title = self.expected_window_edit.text().strip() or None
        process = self.expected_process_edit.text().strip() or None
        if not title and not process:
            return None
        from app.screen import (
            find_window_matching,
            monitor_rect_at,
            window_monitor_rect,
        )

        try:
            window = find_window_matching(title_contains=title, executable=process)
            if window is None:
                return None
            rust_monitor = window_monitor_rect(window.hwnd)
            center_x, center_y = canvas.center
            calibrated_monitor = monitor_rect_at(int(center_x), int(center_y))
        except Exception:
            LOGGER.warning("Could not resolve Rust's monitor", exc_info=True)
            return None
        if rust_monitor is None or calibrated_monitor is None:
            return None
        if rust_monitor == calibrated_monitor:
            return None
        return calibrated_monitor, rust_monitor

    @Slot()
    def _check_rust_monitor(self) -> None:
        """Offer to follow the game when it sits on a different monitor."""

        if self._closing:
            return
        busy = (
            self._painter_is_active()
            or self._debug_running
            or self._countdown_callback_running
            or bool(self._countdown and self._countdown.isVisible())
        )
        mismatch = None if busy else self._rust_monitor_mismatch()
        if mismatch is None:
            self.rust_monitor_label.setVisible(False)
            self.move_to_rust_button.setVisible(False)
            return
        _source, target = mismatch
        self.rust_monitor_label.setText(
            f"⚠ Rust is on the {target.width}×{target.height} monitor at "
            f"({target.left}, {target.top}), but the calibrated boxes are on "
            "another monitor."
        )
        self.rust_monitor_label.setVisible(True)
        self.move_to_rust_button.setVisible(True)

    @Slot()
    def _move_calibration_to_rust_monitor(self) -> None:
        """Reproject every calibrated rectangle onto Rust's current monitor."""

        mismatch = self._rust_monitor_mismatch()
        profile = self._current_profile
        if mismatch is None or profile is None:
            self._check_rust_monitor()
            return
        from app.screen import map_rect_between_monitors

        source, target = mismatch
        candidate = Profile.from_dict(profile.to_dict())
        moved: list[str] = []
        for name in (
            "canvas",
            "color_box",
            "hue_bar",
            "brush_size_box",
            "clear_button",
            "save_button",
        ):
            rect = getattr(candidate, name, None)
            if rect is None:
                continue
            setattr(candidate, name, map_rect_between_monitors(rect, source, target))
            moved.append(name.replace("_", " "))
        if not moved:
            return
        # The saved picker screenshot was captured at the old coordinates.
        candidate.metadata.pop("ui_reference", None)
        try:
            self._current_profile = self._profile_store.save(candidate)
        except Exception as exc:
            LOGGER.exception("Could not move the calibration to Rust's monitor")
            QMessageBox.warning(self, "Could not move the calibration", str(exc))
            return
        LOGGER.info(
            "Moved the calibration to Rust's monitor at (%d, %d): %s",
            target.left,
            target.top,
            ", ".join(moved),
        )
        self.statusBar().showMessage(
            "Calibration boxes moved to Rust's monitor — verify them with "
            "Show boxes on screen",
            8000,
        )
        self.rust_monitor_label.setVisible(False)
        self.move_to_rust_button.setVisible(False)
        self._refresh_profile_ui()
        self._update_quality_dimensions()
        self._update_calibration_overlay()

    @Slot()
    def _on_show_calibration_toggled(self, *_args: Any) -> None:
        self._update_calibration_overlay()

    def _update_calibration_overlay(self) -> None:
        """Show, refresh, or hide the labeled on-screen calibration outlines."""

        profile = self._current_profile
        entries: list[tuple[str, Any]] = []
        if profile is not None:
            entries = [
                (label, rect)
                for label, rect in (
                    ("Canvas", getattr(profile, "canvas", None)),
                    ("Color box", getattr(profile, "color_box", None)),
                    ("Hue bar", getattr(profile, "hue_bar", None)),
                    ("Size value box", getattr(profile, "brush_size_box", None)),
                    ("Clear button", getattr(profile, "clear_button", None)),
                    ("Save button", getattr(profile, "save_button", None)),
                )
                if rect is not None
            ]
        # A paused job is the moment the outlines are most wanted: nothing is
        # being clicked, and what the user is checking is whether the boxes
        # still line up with Rust before letting the job carry on.  The
        # overlay takes no input, and a recording skips paused frames, so
        # nothing downstream sees it either.
        busy = (
            (self._painter_is_active() and not self._painter_is_paused())
            or self._debug_running
            or self._countdown_callback_running
            or bool(self._countdown and self._countdown.isVisible())
        )
        show_boxes = (
            self.show_calibration_check.isChecked() and bool(entries) and not busy
        )
        status = self._status_overlay_text()
        canvas = getattr(profile, "canvas", None) if profile is not None else None
        show_status = (
            self.show_status_check.isChecked()
            and status is not None
            and canvas is not None
        )
        # An overlay that quietly decides not to appear is indistinguishable
        # from a broken one, so every change of mind is logged with what
        # decided it.  Only changes: this runs on every progress update.
        decision = (show_boxes, show_status, status)
        if decision != self._calibration_overlay_decision:
            self._calibration_overlay_decision = decision
            LOGGER.info(
                "Calibration overlay: boxes=%s (switch %s, %d rectangles, busy %s), "
                "status=%s (switch %s, word %s, canvas %s)",
                show_boxes,
                self.show_calibration_check.isChecked(),
                len(entries),
                busy,
                show_status,
                self.show_status_check.isChecked(),
                status,
                canvas is not None,
            )
        if self._closing or not (show_boxes or show_status):
            if self._calibration_preview is not None and self._calibration_preview.isVisible():
                self._calibration_preview.hide()
            return
        try:
            if self._calibration_preview is None:
                self._calibration_preview = CalibrationPreviewOverlay()
            self._calibration_preview.set_rectangles(entries if show_boxes else [])
            self._calibration_preview.set_status(
                (status, canvas) if show_status else None
            )
            if not self._calibration_preview.isVisible():
                self._calibration_preview.show_overlay()
        except Exception:
            LOGGER.exception("Could not display the calibration overlay")

    # What the corner label on the sign's monitor says for each state the job
    # can be in.  A job's last word - stopped, done, failed - stays up a
    # moment after the job, long enough to be read, and then the label falls
    # back to IDLE: with the switch on it is up for as long as the app is,
    # so seeing it is how the user knows the app is running and watching.
    _STATUS_OVERLAY_WORDS = {
        "countdown": "GET READY",
        "running": "PAINTING",
        "paused": "PAUSED",
        "aborted": "ABORTED",
        "completed": "COMPLETE",
        "error": "ERROR",
    }
    _STATUS_OVERLAY_IDLE = "IDLE"
    _STATUS_OVERLAY_LINGER_MS = 4000

    def _status_overlay_text(self) -> str | None:
        painter = self._painter
        if self._countdown is not None and self._countdown.isVisible():
            return self._STATUS_OVERLAY_WORDS["countdown"]
        if painter is None:
            return self._STATUS_OVERLAY_IDLE
        value = getattr(getattr(painter, "state", None), "value", "")
        if value in {"countdown", "running", "paused"}:
            return self._STATUS_OVERLAY_WORDS[value]
        if value in self._STATUS_OVERLAY_WORDS and self._status_overlay_linger.isActive():
            return self._STATUS_OVERLAY_WORDS[value]
        return self._STATUS_OVERLAY_IDLE

    @staticmethod
    def _union_rect(*rectangles: ScreenRect) -> ScreenRect:
        left = min(rect.left for rect in rectangles)
        top = min(rect.top for rect in rectangles)
        right = max(rect.right for rect in rectangles)
        bottom = max(rect.bottom for rect in rectangles)
        return ScreenRect(left, top, right - left, bottom - top)

    @Slot()
    def _capture_reference(self) -> None:
        if self._painter_is_active() or self._debug_running:
            QMessageBox.warning(
                self,
                "Operation is active",
                "Pause or stop the active operation before capturing a reference.",
            )
            return
        profile = self._current_profile
        if profile is None or profile.color_box is None or profile.hue_bar is None:
            QMessageBox.information(
                self,
                "Calibrate the picker",
                "Calibrate the color box and hue bar before capturing a UI reference.",
            )
            return
        QMessageBox.information(
            self,
            "Capture UI reference",
            "After you close this message, focus Rust and leave the painting interface unchanged. "
            "The calibrated picker region will be captured after a 3-second countdown.",
        )
        # Keep the countdown unowned so closing it does not reactivate this
        # window over Rust immediately before the screen capture.
        self._pending_start_cancelled = False
        self._launch_countdown(
            3,
            self._do_capture_reference,
            hint=f"{self.abort_hotkey_combo.currentText()} cancels",
        )

    def _do_capture_reference(self) -> None:
        if self._pending_start_cancelled or self._closing:
            return
        profile = self._current_profile
        if profile is None or profile.color_box is None or profile.hue_bar is None:
            return
        try:
            from app.screen import save_reference

            rectangle = self._union_rect(profile.color_box, profile.hue_bar)
            target = self._local_data_directory() / "references" / f"{profile.id}_picker.png"
            save_reference(rectangle, target)
            candidate = Profile.from_dict(profile.to_dict())
            candidate.metadata["ui_reference"] = {
                "path": str(target),
                "rect": rectangle.to_dict(),
                "minimum_similarity": 0.82,
            }
            self._current_profile = self._profile_store.save(candidate)
            LOGGER.info("Captured UI reference for %s", profile.name)
            self.statusBar().showMessage("UI reference captured", 5000)
        except Exception as exc:
            LOGGER.exception("Could not capture UI reference")
            QMessageBox.warning(self, "Reference capture failed", str(exc))

    def _verify_reference(self, profile: Any | None = None) -> bool:
        profile = profile or self._current_profile
        if not self.verify_ui_check.isChecked() or profile is None:
            return True
        reference = profile.metadata.get("ui_reference")
        if not isinstance(reference, dict):
            QMessageBox.warning(
                self,
                "No UI reference",
                "UI verification is enabled, but this profile has no captured reference.",
            )
            return False
        try:
            from app.screen import compare_region_to_reference

            rectangle = ScreenRect.from_dict(reference["rect"])
            comparison = compare_region_to_reference(
                rectangle,
                reference["path"],
                minimum_similarity=float(reference.get("minimum_similarity", 0.82)),
            )
        except Exception as exc:
            LOGGER.exception("UI reference comparison failed")
            QMessageBox.warning(self, "UI verification failed", str(exc))
            return False
        if comparison.passed:
            LOGGER.info("UI reference check passed (%.1f%%)", comparison.similarity * 100)
            return True
        LOGGER.warning("UI reference check failed (%.1f%%)", comparison.similarity * 100)
        answer = QMessageBox.warning(
            self,
            "Rust UI may not match",
            f"The calibrated picker region is only {comparison.similarity * 100:.1f}% similar "
            "to its reference. The window may have moved or the wrong UI may be open.\n\n"
            "Continue anyway?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer == QMessageBox.StandardButton.Yes

    @Slot()
    def _prepare_color_chart(self) -> None:
        if self._painter_is_active() or self._debug_running or self._countdown_callback_running:
            QMessageBox.warning(
                self,
                "Operation is active",
                "Finish or stop the current operation before preparing a color chart.",
            )
            return
        profile = self._current_profile
        if profile is None or not profile.is_ready:
            QMessageBox.information(
                self,
                "Calibration required",
                "Calibrate the canvas, color box, and hue bar before measuring sign colors.",
            )
            return
        answer = QMessageBox.warning(
            self,
            "Paint a disposable calibration chart",
            "This prepares a 32-color chart that must be painted across a blank/reset "
            "sign. It replaces the currently loaded image and will consume paint on "
            "that test sign.\n\nAfter it finishes, leave the Rust painting UI unchanged "
            "and click Measure Painted Chart. Continue?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return
        try:
            from app.color_calibration import build_calibration_chart

            directory = self._local_data_directory() / "color-calibration"
            directory.mkdir(parents=True, exist_ok=True)
            path = directory / f"{profile.id}-chart.png"
            build_calibration_chart().save(path, format="PNG")
            self._color_chart_profile_id = profile.id
            self._color_chart_path = path
            self._set_combo_data(self.scale_mode_combo, ScaleMode.STRETCH.value)
            self.quality_combo.setCurrentText("Very Fast")
            self.color_count_combo.setCurrentText("32")
            self.dither_check.setChecked(False)
            self._set_combo_data(self.sharpen_combo, SharpenMode.OFF.value)
            # The Optimization combo is deliberately left alone: while the
            # chart is the loaded image, _current_paint_mode() forces Exact,
            # and the user's saved mode survives for their next artwork.
            self.load_image(path)
            self.color_correction_status.setText(
                "Chart loading • paint it, then click Measure Painted Chart"
            )
            LOGGER.info("Prepared color calibration chart for %s", profile.name)
            self.statusBar().showMessage(
                "Calibration chart prepared. Paint it on a blank sign, then measure it.",
                10000,
            )
        except Exception as exc:
            LOGGER.exception("Could not prepare color calibration chart")
            QMessageBox.critical(self, "Could not prepare chart", str(exc))

    @Slot()
    def _measure_color_chart(self) -> None:
        profile = self._current_profile
        if (
            profile is None
            or self._color_chart_profile_id != profile.id
            or self._color_chart_path is None
            or self._image_path != self._color_chart_path
            or self._processed is None
        ):
            QMessageBox.information(
                self,
                "Prepare the chart first",
                "Prepare and completely paint the calibration chart for this profile "
                "before measuring it.",
            )
            return
        if self._painter_is_active() or self._debug_running:
            QMessageBox.warning(
                self,
                "Painting is still active",
                "Wait for the chart painting to complete before measuring it.",
            )
            return
        try:
            self._validate_profile_on_virtual_screen(profile, apply_brush_size=False)
        except ValueError as exc:
            QMessageBox.critical(self, "Canvas calibration is invalid", str(exc))
            return
        QMessageBox.information(
            self,
            "Measure painted chart",
            "Focus Rust and show the completed painted chart. Do not move the window. "
            "The calibrated sign canvas will be captured after the countdown.",
        )
        self._pending_start_cancelled = False
        self._launch_countdown(
            3,
            self._do_measure_color_chart,
            hint=f"{self.abort_hotkey_combo.currentText()} cancels",
        )

    def _do_measure_color_chart(self) -> None:
        if self._pending_start_cancelled or self._closing:
            return
        profile = self._current_profile
        processed = self._processed
        if (
            profile is None
            or profile.canvas is None
            or processed is None
            or self._color_chart_profile_id != profile.id
        ):
            return
        try:
            from app.color_calibration import fit_color_correction, sample_painted_chart
            from app.screen import capture_region, foreground_window_matches

            if self.focus_guard_check.isChecked() and not foreground_window_matches(
                title_contains=self.expected_window_edit.text().strip() or None,
                executable=self.expected_process_edit.text().strip() or None,
            ):
                raise RuntimeError(
                    "The expected Rust window is not foreground. No chart was captured."
                )

            capture = capture_region(profile.canvas)
            commanded, observed = sample_painted_chart(capture, processed.image)
            model = fit_color_correction(commanded, observed)
            command_array = np.asarray(commanded, dtype=np.float64) / 255.0
            observed_array = np.asarray(observed, dtype=np.float64) / 255.0
            uncorrected_error = float(
                np.sqrt(np.mean((command_array - observed_array) ** 2))
            )
            document = model.to_dict()
            document["uncorrectedRmse"] = uncorrected_error
            candidate = Profile.from_dict(profile.to_dict())
            candidate.metadata["color_correction"] = document
            self._current_profile = self._profile_store.save(candidate)
            capture_path = (
                self._local_data_directory()
                / "color-calibration"
                / f"{profile.id}-painted.png"
            )
            capture.save(capture_path, format="PNG")
            LOGGER.info(
                "Measured color correction for %s: raw %.1f, fit %.1f RGB levels",
                profile.name,
                uncorrected_error * 255.0,
                model.fit_rmse * 255.0,
            )
            self._refresh_profile_ui()
            QMessageBox.information(
                self,
                "Color correction saved",
                f"Measured {model.sample_count} swatches. Raw material difference: "
                f"{uncorrected_error * 255.0:.1f} RGB levels; model fit error: "
                f"{model.fit_rmse * 255.0:.1f}.\n\nReset or use a fresh sign, reload "
                "your artwork, and the correction will be applied automatically.",
            )
        except Exception as exc:
            LOGGER.exception("Could not measure painted color chart")
            QMessageBox.critical(
                self,
                "Color measurement failed",
                f"{exc}\n\nConfirm the chart finished painting and the entire calibrated "
                "canvas is visible.",
            )

    @Slot()
    def _clear_color_correction(self) -> None:
        profile = self._current_profile
        if profile is None or "color_correction" not in profile.metadata:
            return
        try:
            candidate = Profile.from_dict(profile.to_dict())
            candidate.metadata.pop("color_correction", None)
            self._current_profile = self._profile_store.save(candidate)
            self._refresh_profile_ui()
            self.statusBar().showMessage("Color correction cleared", 5000)
        except Exception as exc:
            LOGGER.exception("Could not clear color correction")
            QMessageBox.warning(self, "Could not clear correction", str(exc))

    def _refresh_brush_model_status(self) -> None:
        """Say what automatic sizing will do on the next run, or what it needs."""

        if not self.apply_brush_check.isChecked():
            grid = self._texel_grid()
            canvas = self._profile_rect("canvas")
            if grid is not None and canvas is not None and grid.agrees_with(canvas):
                aim = (
                    f"strokes are aimed by the {grid.columns}×{grid.rows}-texel "
                    "grid the last measurement counted on this sign"
                )
            else:
                aim = (
                    "without a measured grid, strokes are laid out on the "
                    "calibration rectangle, good to about half a texel - turn "
                    "sizing on to measure the sign"
                )
            self.brush_model_status.setText(
                "Automatic brush sizing is off - Rust keeps whatever brush size "
                f"you set by hand; {aim}"
            )
            return
        missing = self._missing_sizing_rectangles()
        if missing:
            self.brush_model_status.setText(
                "Automatic brush sizing needs the "
                + " and ".join(missing)
                + " calibrated"
            )
            return
        model = self._brush_size_model()
        if model is None:
            self.brush_model_status.setText(
                "Ready - the next paint job measures this sign's brush and wipes "
                "the measurement before painting"
            )
            return
        from app.brush_calibration import canonical_texture_rows

        rows = canonical_texture_rows(model.sign_pixel_rows)
        grid = self._texel_grid()
        if grid is not None:
            texture = (
                f"a {grid.columns}×{grid.rows}-texel texture, counted on the sign"
            )
        elif (cap := self._sign_resolution_cap()) and self._sign_resolution_cap_source() == "table":
            texture = f"a {cap[0]}×{cap[1]}-texel texture, by Rust's sign data"
        else:
            texture = f"about a {rows}-row texture, inferred from the brush"
        self.brush_model_status.setText(
            f"Last measured: size 1 covers {model.smallest_fraction * 100:.2f}% of "
            f"the sign ({texture}). Every job measures again before it paints."
        )

    def _canvas_shape_changed(self, rectangle: Any) -> bool:
        """Whether a new canvas rectangle describes a differently shaped sign.

        Standing closer scales both sides together and leaves the aspect ratio
        alone, which is exactly the case the brush model is built to survive.
        A changed ratio means a different sign, so its measurement cannot carry
        over.
        """

        previous = self._profile_rect("canvas")
        if previous is None or previous.height <= 0 or rectangle.height <= 0:
            return False
        before = previous.width / previous.height
        after = rectangle.width / rectangle.height
        return abs(before - after) > before * 0.05

    def _store_measured_brush_model(self) -> None:
        """Keep the model the finished job measured on its way in.

        The run itself no longer needs it - it measured its own - but the
        planner does: knowing what the sign's Size numbers reach is what lets
        the optimizer offer multi-cell brush passes on the *next* image, and
        what the preview needs to stop promising detail the brush cannot hold.
        A failure to save is only logged: the paint job succeeded, and saying
        so with an error dialog would be a lie about what just happened.
        """

        painter = self._painter
        profile = self._current_profile
        model = painter.measured_brush_size_model if painter is not None else None
        grid = getattr(painter, "measured_texel_grid", None) if painter is not None else None
        if model is None or profile is None:
            return
        stored = profile.metadata.get("brush_size_model")
        stored_grid = profile.metadata.get("texel_grid")
        grid_value = grid.to_dict() if grid is not None else None
        if (
            isinstance(stored, dict)
            and stored.get("slope") == model.slope
            and (grid is None or stored_grid == grid_value)
        ):
            return
        try:
            candidate = Profile.from_dict(profile.to_dict())
            candidate.metadata["brush_size_model"] = model.to_dict()
            if grid_value is not None:
                candidate.metadata["texel_grid"] = grid_value
            self._current_profile = self._profile_store.save(candidate)
        except Exception:
            LOGGER.exception("Could not save the measured brush size model")
            return
        LOGGER.info(
            "Brush size measured: %.6f of the sign per unit (~%.0f rows)%s",
            model.slope,
            model.sign_pixel_rows,
            (
                f"; texel grid {grid.columns}x{grid.rows} counted on the sign"
                if grid is not None
                else ""
            ),
        )
        self._refresh_profile_ui()
        # A fresh measurement can move the sign's resolution ceiling, and the
        # capped dimensions feed text scaling, so the logical size is
        # re-derived rather than only reprocessed.
        self._update_quality_dimensions()

    @Slot()
    def _register_hotkeys(self, *_args: Any) -> None:
        self._refresh_hotkey_labels()
        if os.environ.get("RUST_PAINTER_DISABLE_HOTKEYS") == "1":
            self._hotkeys_ready = False
            self._update_start_availability()
            return
        requested = (
            self.start_hotkey_combo.currentText(),
            self.pause_hotkey_combo.currentText(),
            self.abort_hotkey_combo.currentText(),
        )
        if len({value.upper() for value in requested}) != len(requested):
            self._restore_last_hotkey_selection()
            self._hotkeys_ready = bool(
                self._hotkeys is not None and getattr(self._hotkeys, "running", False)
            )
            self._on_hotkey_error("Start, pause, and stop hotkeys must be different.")
            self._update_start_availability()
            return
        previous = self._hotkeys
        try:
            from app.hotkeys import GlobalHotkeyManager, HotkeyBindings

            bindings = HotkeyBindings(
                start_resume=requested[0],
                pause=requested[1],
                abort=requested[2],
            )
            candidate = GlobalHotkeyManager(
                on_start_resume=self._painter_bridge.start_requested.emit,
                on_pause=self._hotkey_pause_immediate,
                on_abort=self._hotkey_abort_immediate,
                bindings=bindings,
                on_error=self._hotkey_failure_immediate,
            )
            if previous is not None:
                previous.stop()
            if candidate.start():
                self._hotkeys = candidate
                self._hotkeys_ready = True
                self._last_hotkey_bindings = requested
                LOGGER.info("Global hotkeys are active")
            else:
                candidate.stop()
                restored = bool(previous is not None and previous.start())
                self._hotkeys = previous if restored else None
                self._hotkeys_ready = restored
                self._restore_last_hotkey_selection()
                detail = candidate.startup_error or "one or more bindings are unavailable"
                raise RuntimeError(f"Could not activate global hotkeys: {detail}")
        except Exception as exc:
            if previous is not None and not getattr(previous, "running", False):
                try:
                    if previous.start():
                        self._hotkeys = previous
                        self._hotkeys_ready = True
                        self._restore_last_hotkey_selection()
                except Exception:
                    LOGGER.exception("Could not restore the previous global hotkeys")
            LOGGER.exception("Could not configure global hotkeys")
            self._on_hotkey_error(str(exc))
        finally:
            self._update_start_availability()

    def _restore_last_hotkey_selection(self) -> None:
        if self._last_hotkey_bindings is None:
            return
        combos = (
            self.start_hotkey_combo,
            self.pause_hotkey_combo,
            self.abort_hotkey_combo,
        )
        for combo, value in zip(combos, self._last_hotkey_bindings, strict=True):
            combo.blockSignals(True)
            combo.setCurrentText(value)
            combo.blockSignals(False)
        self._refresh_hotkey_labels()

    def _hotkey_pause_immediate(self) -> None:
        painter = self._painter
        if painter is not None:
            painter.pause("global hotkey")
        self._painter_bridge.pause_requested.emit()

    def _hotkey_abort_immediate(self) -> None:
        # This runs on the Win32 hotkey thread. Painter.abort is thread-safe and
        # releases held input before the GUI event loop gets a chance to update.
        self._pending_start_cancelled = True
        painter = self._painter
        if painter is not None:
            try:
                painter.abort("global emergency hotkey")
            except Exception:
                LOGGER.exception("Could not abort painter from emergency hotkey")
        self._debug_abort_event.set()
        with self._debug_input_gate:
            controller = self._debug_controller
            if controller is not None:
                try:
                    controller.release_all()
                except Exception:
                    LOGGER.exception("Could not release debug input from emergency hotkey")
        try:
            self._painter_bridge.abort_requested.emit()
        except Exception:
            LOGGER.exception("Could not queue emergency-stop UI update")

    def _hotkey_failure_immediate(self, error: BaseException) -> None:
        """Fail closed on the hotkey thread before its Qt warning is handled."""

        self._hotkeys_ready = False
        self._hotkey_abort_immediate()
        try:
            self._painter_bridge.hotkey_error.emit(str(error))
        except Exception:
            LOGGER.exception("Could not queue global-hotkey failure warning")

    def _emergency_hotkey_available(self) -> bool:
        manager = self._hotkeys
        return bool(
            self._hotkeys_ready
            and manager is not None
            and getattr(manager, "running", False)
        )

    def _refresh_hotkey_labels(self) -> None:
        self.pause_button.setText(
            f"Pause  •  {self.pause_hotkey_combo.currentText()}"
        )
        self.abort_button.setText(
            f"Stop  •  {self.abort_hotkey_combo.currentText()}"
        )
        self._update_start_availability()

    @Slot(str)
    def _on_hotkey_error(self, message: str) -> None:
        # A live global abort binding is a prerequisite for every real-input
        # operation.  Registration errors may arrive asynchronously after a
        # previously healthy message loop exits, so derive readiness again
        # instead of trusting the last successful registration.
        self._hotkeys_ready = bool(
            self._hotkeys is not None and getattr(self._hotkeys, "running", False)
        )
        self.statusBar().showMessage(f"Global hotkey warning: {message}", 10000)
        LOGGER.warning("Global hotkey warning: %s", message)
        self._update_start_availability()

    # -------------------------------------------------------------- execution

    def _painter_state_value(self) -> str | None:
        """The painter's state as its plain string, or None when there is none."""

        painter = self._painter
        if painter is None:
            return None
        return getattr(getattr(painter, "state", None), "value", None)

    def _painter_is_paused(self) -> bool:
        return self._painter_state_value() == "paused"

    def _painter_is_active(self) -> bool:
        if self._painter is None:
            return False
        try:
            from app.painter import PainterState

            return bool(
                self._painter.is_active
                or (
                    self._painter.state is PainterState.READY
                    and self._painter.is_alive
                )
            )
        except Exception:
            return bool(getattr(self._painter, "is_active", False))

    def _update_start_availability(self) -> None:
        active = self._painter_is_active()
        countdown_active = bool(self._countdown and self._countdown.isVisible())
        job_locked = (
            active
            or countdown_active
            or self._countdown_callback_running
            or self._debug_running
        )
        paused = self._painter_is_paused()
        profile_ready = bool(
            self._current_profile
            and self._current_profile.is_ready
            and not self._missing_sizing_rectangles()
            and not self._missing_anti_afk_rectangles()
        )
        can_dry_run = self.dry_run_check.isChecked() and self._plan is not None
        can_start = (self._plan is not None and profile_ready) or can_dry_run or paused
        if (
            not self.dry_run_check.isChecked()
            and not self._emergency_hotkey_available()
            and not paused
        ):
            can_start = False
        enabled = can_start and not countdown_active and (not active or paused)
        self.start_button.setEnabled(enabled)
        # A disabled button with no explanation reads as a broken app.  Name the
        # blocker instead, because the two common ones -- unfinished calibration
        # and a hotkey another program already owns -- are both user-fixable.
        self.start_button.setToolTip(
            "" if enabled else self._start_blocked_reason(profile_ready, paused)
        )
        resume_at = 0 if active or paused else self._resume_start_stroke()
        self.start_button.setText(
            f"RESUME PAINTING  •  {self.start_hotkey_combo.currentText()}"
            if paused
            else f"RESUME FROM STROKE {resume_at:,}  •  {self.start_hotkey_combo.currentText()}"
            if resume_at
            else f"START PAINTING  •  {self.start_hotkey_combo.currentText()}"
        )
        self.pause_button.setEnabled(active and not paused)
        self.abort_button.setEnabled(
            active
            or paused
            or countdown_active
            or self._countdown_callback_running
            or self._debug_running
        )
        self._set_job_controls_locked(job_locked, retunable=paused)
        self._update_calibration_overlay()

    def _missing_sizing_rectangles(self) -> list[str]:
        """Rectangles automatic brush sizing needs but this profile lacks.

        Both only matter while sizing is on: the run types a Size number it
        measured on the sign, and it can only measure on the sign if it can
        wipe the probe strokes off again afterwards.
        """

        if not self.apply_brush_check.isChecked():
            return []
        return [
            label
            for label, name in (
                ("Size value box", "brush_size_box"),
                ("clear button", "clear_button"),
            )
            if self._profile_rect(name) is None
        ]

    def _missing_anti_afk_rectangles(self) -> list[str]:
        """The rectangle the anti-AFK break needs but this profile lacks.

        The break leaves the painting UI through Rust's Save button, so with
        the break on, the button has to be calibrated.
        """

        if not self.anti_afk_check.isChecked():
            return []
        return ["Save button"] if self._profile_rect("save_button") is None else []

    def _start_blocked_reason(self, profile_ready: bool, paused: bool) -> str:
        if self._painter_is_active() and not paused:
            return "A paint job is already running."
        if self._countdown and self._countdown.isVisible():
            return "Waiting for the countdown to finish."
        if self._plan is None:
            if self._plan_deferred:
                return (
                    "The paint plan is waiting on your edits - switch to the "
                    "Rust preview tab to recalculate it."
                )
            return "Load an image and wait for its paint plan to finish."
        if not profile_ready and not self.dry_run_check.isChecked():
            missing = [
                label
                for label, done in (
                    ("canvas", self._current_profile.canvas is not None),
                    ("color box", self._current_profile.color_box is not None),
                    ("hue bar", self._current_profile.hue_bar is not None),
                )
                if self._current_profile is not None and not done
            ]
            if missing:
                return "Calibrate the " + ", ".join(missing) + " before painting."
            sizing = self._missing_sizing_rectangles()
            if sizing:
                return (
                    "Automatic brush sizing measures this sign's brush before "
                    "every job and wipes the measurement off again, so it needs "
                    "the " + " and ".join(sizing) + " calibrated. Turn automatic "
                    "brush sizing off to paint with whatever brush Rust has set."
                )
            if self._missing_anti_afk_rectangles():
                return (
                    "The anti-AFK break leaves the painting UI through Rust's "
                    "Save button, so it needs the Save button calibrated. Turn "
                    "Anti-AFK off under Settings to paint without it."
                )
            return "Finish calibrating this profile before painting."
        if not self.dry_run_check.isChecked() and not self._emergency_hotkey_available():
            return (
                "The global stop hotkey is not active, so real painting is blocked. "
                "Another program may already own "
                f"{self.abort_hotkey_combo.currentText()}; pick different hotkeys above."
            )
        return "Painting is unavailable right now."

    # The controls a paused job may still take new values from: the holds
    # and speeds every remaining stroke is run with, the guards that decide
    # when to stop, and whether a timelapse is being recorded.  The painter
    # applies its share on resume; the recording follows the controls then
    # too.  Everything that shaped the plan or the job stays locked until
    # the job is over.
    def _retunable_controls(self) -> tuple[QWidget, ...]:
        return (
            self.speed_preset_combo,
            self.stroke_speed_spin,
            self.dot_duration_spin,
            self.hue_delay_spin,
            self.sv_delay_spin,
            self.brush_delay_spin,
            self.stroke_delay_spin,
            self.color_delay_spin,
            self.interpolation_spin,
            self.line_tool_check,
            self.press_hold_check,
            self.dab_size_check,
            self.verify_passes_spin,
            self.verify_picks_check,
            self.confirm_strokes_check,
            self.confirm_rounds_spin,
            self.focus_guard_check,
            self.expected_window_edit,
            self.expected_process_edit,
            self.mouse_pause_check,
            self.ui_guard_check,
            self.anti_afk_check,
            self.anti_afk_interval_spin,
            self.timelapse_check,
            self.timelapse_interval_spin,
            self.timelapse_final_check,
        )

    def _set_job_controls_locked(self, locked: bool, *, retunable: bool = False) -> None:
        controls = (
            self.browse_button,
            self.scale_mode_combo,
            self.crop_alignment_combo,
            self.background_combo,
            self.background_color_button,
            self.transparency_combo,
            self.alpha_fill_check,
            self.remove_background_check,
            self.removal_source_combo,
            self.removal_color_button,
            self.removal_tolerance_spin,
            self.removal_scope_combo,
            self.quality_combo,
            self.paint_mode_combo,
            self.logical_width_spin,
            self.logical_height_spin,
            self.color_count_combo,
            self.dither_check,
            self.sharpen_combo,
            self.merge_combo,
            self.speed_preset_combo,
            self.pixel_spacing_spin,
            self.stroke_speed_spin,
            self.dot_duration_spin,
            self.hue_delay_spin,
            self.sv_delay_spin,
            self.brush_delay_spin,
            self.stroke_delay_spin,
            self.color_delay_spin,
            self.interpolation_spin,
            self.line_tool_check,
            self.press_hold_check,
            self.dab_size_check,
            self.apply_brush_check,
            self.verify_passes_spin,
            self.verify_picks_check,
            self.confirm_strokes_check,
            self.confirm_rounds_spin,
            self.timelapse_check,
            self.timelapse_interval_spin,
            self.timelapse_final_check,
            self.profile_combo,
            self.new_profile_button,
            self.rename_profile_button,
            self.delete_profile_button,
            self.calibrate_canvas_button,
            self.calibrate_color_box_button,
            self.calibrate_hue_bar_button,
            self.calibrate_brush_button,
            self.calibrate_clear_button,
            self.calibrate_save_button,
            self.countdown_spin,
            self.dry_run_check,
            self.focus_guard_check,
            self.expected_window_edit,
            self.expected_process_edit,
            self.mouse_pause_check,
            self.verify_ui_check,
            self.ui_guard_check,
            self.anti_afk_check,
            self.anti_afk_interval_spin,
            self.start_hotkey_combo,
            self.pause_hotkey_combo,
            self.abort_hotkey_combo,
            self.capture_reference_button,
            self.prepare_color_chart_button,
            self.measure_color_chart_button,
            self.clear_color_correction_button,
            self.resume_panel,
            *self.debug_buttons.values(),
        )
        live = set(self._retunable_controls()) if retunable else set()
        for control in controls:
            control.setEnabled(not locked or control in live)
        # A paused job lends out its slider alone - to compare the sign
        # with the picture as far as a stroke - never the tick that would
        # change where a job starts.
        paused_viewfinder = locked and retunable and bool(
            getattr(self._plan, "stroke_count", 0)
        )
        self.resume_check.setEnabled(not locked)
        self.resume_screenshot_button.setEnabled(not locked)
        self.resume_slider.setEnabled(not locked or paused_viewfinder)
        if paused_viewfinder:
            self.resume_panel.setEnabled(True)
        if not locked:
            is_fit = self.scale_mode_combo.currentData() == ScaleMode.FIT.value
            is_fill = self.scale_mode_combo.currentData() == ScaleMode.FILL.value
            alpha_fill = (
                self.transparency_combo.currentData()
                == TransparencyMode.USE_BACKGROUND.value
            )
            self.crop_alignment_combo.setEnabled(is_fill)
            self.background_combo.setEnabled(is_fit or alpha_fill)
            self.background_color_button.setEnabled(
                (is_fit or alpha_fill)
                and self.background_combo.currentData() == "custom"
            )
            self.removal_color_button.setEnabled(
                self.remove_background_check.isChecked()
                and self.removal_source_combo.currentData() == "custom"
            )
            custom = self.quality_combo.currentText() == "Custom"
            self.logical_width_spin.setEnabled(custom)
            self.logical_height_spin.setEnabled(custom)
            self._sync_paint_mode_dependent_controls()
            self.resume_panel.setEnabled(
                bool(getattr(self._plan, "stroke_count", 0))
            )
            has_profile = self._current_profile is not None
            self.rename_profile_button.setEnabled(has_profile)
            self.delete_profile_button.setEnabled(has_profile)
            self.clear_color_correction_button.setEnabled(
                bool(
                    self._current_profile
                    and isinstance(self._current_profile.metadata, dict)
                    and "color_correction" in self._current_profile.metadata
                )
            )

    def _launch_countdown(
        self,
        seconds: int,
        callback: Any,
        *,
        hint: str,
    ) -> bool:
        if (
            self._countdown_callback_running
            or self._debug_running
            or (self._countdown is not None and self._countdown.isVisible())
        ):
            self.statusBar().showMessage("Another countdown is already active.", 4000)
            return False

        def run_callback() -> None:
            self._countdown_callback_running = True
            try:
                callback()
            finally:
                self._countdown_callback_running = False
                self._update_start_availability()

        dialog = CountdownDialog(seconds, run_callback, None, hint=hint)
        self._countdown = dialog

        def finished(result: int, finished_dialog: CountdownDialog = dialog) -> None:
            if self._countdown is finished_dialog:
                self._countdown = None
            finished_dialog.deleteLater()
            if result != QDialog.DialogCode.Accepted:
                self._pending_paint = None
                self._pending_start_cancelled = True
                self._update_start_availability()

        dialog.finished.connect(finished)
        dialog.show()
        self._update_start_availability()
        return True

    @Slot()
    def _start_or_resume(self) -> None:
        if self._closing:
            return
        if self._countdown_callback_running or self._debug_running:
            return
        if self._painter is not None:
            try:
                from app.painter import PainterState

                if self._painter.state == PainterState.PAUSED:
                    self._retune_paused_painter()
                    self._sync_timelapse_with_controls()
                    self._painter.resume()
                    return
                if self._painter_is_active():
                    return
            except Exception:
                pass
        if self._countdown is not None and self._countdown.isVisible():
            return
        if self._plan is None:
            self.statusBar().showMessage("Load an image and wait for its paint plan.", 5000)
            return
        if (
            not self.dry_run_check.isChecked()
            and not self._emergency_hotkey_available()
        ):
            QMessageBox.critical(
                self,
                "Emergency hotkey unavailable",
                "Real painting is disabled because the global stop hotkey is not active. "
                "Choose three distinct, available hotkeys and try again.",
            )
            return
        if not self.dry_run_check.isChecked() and not (
            self._current_profile and self._current_profile.is_ready
        ):
            QMessageBox.warning(
                self,
                "Calibration incomplete",
                "Real painting requires a calibrated canvas, color box, and hue bar.",
            )
            return
        if not self.dry_run_check.isChecked():
            self._refresh_display_warning()
            try:
                self._validate_profile_on_virtual_screen(
                    apply_brush_size=self.apply_brush_check.isChecked(),
                    anti_afk=self.anti_afk_check.isChecked(),
                )
            except ValueError as exc:
                QMessageBox.critical(self, "Calibration is outside the desktop", str(exc))
                return
        if (
            not self.dry_run_check.isChecked()
            and self.display_warning_label.text()
            and QMessageBox.warning(
                self,
                "Display configuration changed",
                self.display_warning_label.text()
                + "\n\nContinue with the stored coordinates anyway?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        try:
            dry_run = self.dry_run_check.isChecked()
            target_profile = self._execution_profile(dry_run, self._plan)
            profile_snapshot = (
                Profile.from_dict(target_profile.to_dict())
                if isinstance(target_profile, Profile)
                else target_profile
            )
            if self._painting_calibration_chart(profile_snapshot):
                # Calibration charts measure the raw Rust material response;
                # never feed an earlier correction back into its own chart.
                profile_snapshot.metadata.pop("color_correction", None)
            self._pending_paint = _PendingPaint(
                plan=self._plan,
                profile=profile_snapshot,
                settings=self._settings_document(),
                dry_run=dry_run,
                display_snapshot=(capture_display_metadata() if not dry_run else None),
                start_stroke=self._resume_start_stroke(),
            )
            self._pending_start_cancelled = False
        except Exception as exc:
            LOGGER.exception("Could not snapshot the pending paint job")
            QMessageBox.critical(self, "Could not prepare paint job", str(exc))
            return
        seconds = self.countdown_spin.value()
        # An unowned countdown is less likely to return foreground focus to the
        # main window when it closes; the painter still verifies Rust before input.
        launched = self._launch_countdown(
            seconds,
            self._begin_paint_after_countdown,
            hint=f"{self.abort_hotkey_combo.currentText()} cancels before input begins",
        )
        if not launched:
            self._pending_paint = None

    def _validate_profile_on_virtual_screen(
        self,
        profile: Any | None = None,
        *,
        apply_brush_size: bool = False,
        anti_afk: bool = False,
    ) -> None:
        profile = profile or self._current_profile
        if profile is None:
            raise ValueError("No sign profile is selected")
        from app.screen import get_virtual_screen

        desktop = get_virtual_screen()
        names = ["canvas", "color_box", "hue_bar"]
        if apply_brush_size:
            names.extend(("brush_size_box", "clear_button"))
        if anti_afk:
            names.append("save_button")
        required = {
            "save_button": (
                "The anti-AFK break leaves the painting UI through Rust's Save "
                "button, so it requires the Save button to be calibrated."
            ),
            "brush_size_box": (
                "Automatic brush sizing requires Rust's numeric Size field to be "
                "calibrated."
            ),
            "clear_button": (
                "Automatic brush sizing measures the brush on the sign before "
                "painting, so it requires Rust's clear button to be calibrated - "
                "that is what wipes the measurement off again."
            ),
        }
        for name in names:
            rectangle = getattr(profile, name, None)
            if rectangle is None:
                if name in required:
                    raise ValueError(required[name])
                continue
            if not (
                desktop.left <= rectangle.left
                and desktop.top <= rectangle.top
                and rectangle.right <= desktop.right
                and rectangle.bottom <= desktop.bottom
            ):
                raise ValueError(
                    f"The profile's {name.replace('_', ' ')} rectangle "
                    f"({rectangle.left}, {rectangle.top}, {rectangle.width}, {rectangle.height}) "
                    "is not wholly inside the current virtual desktop. Recalibrate it."
                )

    def _begin_paint_after_countdown(self) -> None:
        pending = self._pending_paint
        if pending is None or self._pending_start_cancelled:
            return
        self._pending_paint = None
        # The pending job is cleared before painting starts, but the run
        # report needs the settings and profile exactly as they were at the
        # countdown - reading them back later describes some edit made
        # afterwards and blames this run for it.
        self._paint_job_snapshot = pending
        dry_run = pending.dry_run
        if not dry_run and not self._emergency_hotkey_available():
            QMessageBox.critical(
                self,
                "Emergency hotkey unavailable",
                "Painting was cancelled because the global stop hotkey stopped "
                "during the countdown.",
            )
            self._set_idle_ui("Start cancelled: emergency hotkey unavailable")
            return
        if not dry_run:
            try:
                self._validate_profile_on_virtual_screen(
                    pending.profile,
                    apply_brush_size=bool(
                        pending.settings.get("painting", {}).get(
                            "apply_brush_size", False
                        )
                    ),
                    anti_afk=bool(
                        pending.settings.get("safety", {}).get(
                            "anti_afk_enabled", False
                        )
                    ),
                )
                if pending.display_snapshot is not None:
                    display_changes = pending.display_snapshot.differences(
                        capture_display_metadata()
                    )
                    if display_changes:
                        raise ValueError(
                            "Display configuration changed during the countdown: "
                            + "; ".join(display_changes)
                        )
            except Exception as exc:
                LOGGER.error("Painting start validation failed: %s", exc)
                QMessageBox.critical(self, "Painting start cancelled", str(exc))
                self._set_idle_ui("Display changed during countdown")
                return
        if not dry_run and not self._verify_reference(pending.profile):
            self._set_idle_ui("Start cancelled by UI verification")
            return
        if self._pending_start_cancelled:
            self._set_idle_ui("Start cancelled")
            return
        try:
            from app.input_controller import (
                DryRunInputController,
                create_system_input_controller,
            )
            from app.painter import Painter

            input_controller = (
                DryRunInputController(
                    detailed_logging=bool(
                        pending.settings.get("execution", {}).get(
                            "debug_mouse_logging", False
                        )
                    )
                )
                if dry_run
                else create_system_input_controller()
            )
            settings = self._painter_settings(pending.settings, dry_run)
            if self._pending_start_cancelled:
                self._set_idle_ui("Start cancelled")
                return
            if self._painter is not None:
                old_painter = self._painter
                old_painter.shutdown(timeout=0.5)
                if getattr(old_painter.input, "held_buttons", frozenset()):
                    # shutdown() logs and suppresses release failures so that
                    # every shutdown step can run. Before discarding the only
                    # controller that tracks a held button, retry explicitly
                    # and fail closed if Windows still rejects the release.
                    old_painter.input.release_all()
                if getattr(old_painter.input, "held_buttons", frozenset()):
                    raise RuntimeError(
                        "The previous input controller still reports a held mouse button. "
                        "Painting will not restart until it can be released."
                    )
                if bool(getattr(old_painter, "is_alive", False)):
                    raise RuntimeError(
                        "The previous paint worker did not stop in time. Painting will "
                        "not restart while its final input cleanup may still be running."
                    )
            generation = self._paint_generation + 1
            painter = Painter(
                input_controller,
                stroke_overhead_seconds=self._learned_timing.overhead_seconds,
                check_capture_seconds=self._learned_timing.check_capture_seconds,
                check_repaint_fraction=self._learned_timing.check_repaint_fraction,
                touch_up_fraction=self._learned_timing.touch_up_fraction,
                on_progress=lambda progress: self._painter_bridge.progress.emit(
                    generation, progress
                ),
                on_state_change=lambda state, reason: self._painter_bridge.state.emit(
                    generation, state, reason
                ),
                on_complete=lambda _progress: self._painter_bridge.completed.emit(
                    generation
                ),
                on_error=lambda exc: self._painter_bridge.error.emit(
                    generation, str(exc)
                ),
            )
            # Publish only a configured READY painter. The hotkey thread can
            # then abort it atomically even before its worker thread starts.
            painter.configure(
                pending.plan,
                pending.profile,
                settings,
                start_stroke=pending.start_stroke,
            )
            if self._pending_start_cancelled:
                painter.shutdown(timeout=0.5)
                self._set_idle_ui("Start cancelled")
                return
            self._paint_generation = generation
            self._painter = painter
            self._open_resume_record(pending)
            if self._pending_start_cancelled:
                painter.shutdown(timeout=0.5)
                self._set_idle_ui("Start cancelled")
                return
            if not painter.start():
                if self._pending_start_cancelled or painter.state.value == "aborted":
                    self._set_idle_ui("Start cancelled")
                    return
                raise RuntimeError("The configured paint worker could not be started")
            LOGGER.info(
                "%s started: %d colors, %d strokes%s",
                "Dry run" if dry_run else "Painting",
                len(pending.plan.color_groups),
                pending.plan.stroke_count,
                (
                    f", resuming from stroke {pending.start_stroke:,}"
                    if pending.start_stroke
                    else ""
                ),
            )
            self._update_start_availability()
        except Exception as exc:
            LOGGER.exception("Could not start painting")
            self._on_paint_error(self._paint_generation, str(exc))

    def _painter_settings(self, settings_document: dict[str, Any], dry_run: bool) -> Any:
        """The painter's settings for a job, with a dry run's waits taken out."""

        from app.painter import PainterSettings

        # The visible Qt countdown has already completed.  Keeping the
        # worker countdown at zero avoids a confusing second countdown.
        settings_document["safety"]["countdown_seconds"] = 0
        if dry_run:
            # A dry run is a plan visualizer, not a wall-clock simulation.
            # Preserve ordering/progress while omitting deliberate waits.
            settings_document["painting"].update(
                stroke_speed_pixels_per_second=1_000_000_000.0,
                mouse_down_duration_seconds=0.0,
                delay_after_hue_seconds=0.0,
                delay_after_saturation_value_seconds=0.0,
                delay_between_strokes_seconds=0.0,
                delay_between_colors_seconds=0.0,
                stroke_interpolation_step_pixels=100_000.0,
                apply_brush_size=False,
            )
            settings_document["safety"]["require_rust_foreground"] = False
            settings_document["safety"]["anti_afk_enabled"] = False
        return PainterSettings.from_mapping(settings_document)

    def _retune_paused_painter(self) -> None:
        """Give the paused job the timing and guards the controls show now.

        The timing and safety controls stay live through a pause precisely
        so a hold that looked too short on the sign can be lengthened before
        the next stroke.  A value the painter rejects is reported and left
        out; the job resumes on the timing it had rather than not at all.
        """

        painter = self._painter
        if painter is None:
            return
        # A dry run's input controller is the one thing that says for sure
        # the job is a dry run, whatever the controls show now.
        dry_run = not bool(
            getattr(getattr(painter, "input", None), "emits_real_input", True)
        )
        try:
            settings = self._painter_settings(self._settings_document(), dry_run)
            painter.retune(settings)
        except Exception as exc:
            LOGGER.warning("Could not apply the changed settings to the paused job: %s", exc)
            self.statusBar().showMessage(
                f"Resumed on the previous timing - the changed settings were not "
                f"accepted: {exc}",
                8000,
            )

    def _execution_profile(self, dry_run: bool, plan: PaintPlan) -> Any:
        profile = self._current_profile
        if profile is not None and profile.is_ready:
            return profile
        if not dry_run:
            raise ValueError("The selected profile is not fully calibrated")
        # Dry runs never emit input, so placeholder rectangles safely allow the
        # complete plan/control flow to be exercised before calibration.
        from types import SimpleNamespace

        return SimpleNamespace(
            canvas=ScreenRect(0, 0, max(1, plan.width), max(1, plan.height)),
            color_box=ScreenRect(0, 0, 101, 101),
            hue_bar=ScreenRect(102, 0, 7, 360),
            brush_size_box=None,
            metadata={},
            hue_direction="bottom_to_top",
            saturation_direction="left_low",
            value_direction="top_bright",
        )

    @Slot()
    def _pause_painting(self) -> None:
        if self._painter is not None and self._painter.pause("user hotkey/button"):
            self._update_start_availability()

    @Slot()
    def _abort_painting(self) -> None:
        self._pending_start_cancelled = True
        self._pending_paint = None
        self._debug_abort_event.set()
        with self._debug_input_gate:
            controller = self._debug_controller
            if controller is not None:
                try:
                    controller.release_all()
                    debug_thread = self._debug_thread
                    if debug_thread is None or not debug_thread.is_alive():
                        self._debug_controller = None
                        self._debug_running = False
                except Exception:
                    LOGGER.exception("Could not release debug input during abort")
        if self._countdown is not None and self._countdown.isVisible():
            self._countdown.reject()
            self._set_idle_ui("Start cancelled")
        if self._painter is not None:
            self._painter.abort("emergency stop")
        self._update_start_availability()

    @Slot(int, object)
    def _on_paint_progress(self, generation: int, progress: Any) -> None:
        if generation != self._paint_generation:
            return
        painter = self._painter
        if painter is not None:
            current = getattr(getattr(painter, "state", None), "value", None)
            reported = getattr(getattr(progress, "state", None), "value", None)
            if current is not None and reported is not None and current != reported:
                return
        percent = min(max(float(progress.percent), 0.0), 100.0)
        self.paint_progress.setValue(round(percent * 10))
        self.progress_state_label.setText(str(progress.message or progress.state.value))
        remaining = (
            f" • {self._format_duration(progress.estimated_remaining_seconds)} remaining"
            if progress.estimated_remaining_seconds is not None
            else ""
        )
        detail = (
            f"Color {progress.color_index:,} / {progress.total_colors:,}  •  "
            f"Stroke {progress.completed_strokes:,} / {progress.total_strokes:,}"
        )
        self.progress_detail_label.setText(f"{detail}  •  {percent:.1f}%{remaining}")
        if getattr(progress, "phase", "paint") == "verify":
            # The touch-up pass restarts the bar for its own, smaller plan;
            # titled anything less specific, it reads as the job starting over.
            self.active_progress_title.setText("TOUCHING UP")
        if (
            getattr(progress, "phase", "paint") == "paint"
            and getattr(progress.state, "value", "") == "running"
        ):
            # Recording starts here rather than when the worker enters RUNNING,
            # so the finished video opens on a blank sign instead of on the
            # calibration strokes the job wipes off before it paints.  The
            # state check matters: the job's final progress update is also a
            # painting one, and without it a finished job would open a new
            # recording the moment it closed the old one.
            self._maybe_start_timelapse()
            self._maybe_start_run_report()
        self.active_paint_progress.setValue(round(percent * 10))
        self.active_progress_state.setText(
            str(progress.message or progress.state.value)
        )
        self.active_percent_label.setText(f"{percent:.0f}%")
        self.active_remaining_label.setText(
            f"{self._format_duration(progress.estimated_remaining_seconds)} remaining"
            if progress.estimated_remaining_seconds is not None
            else "Estimating time left…"
        )
        self._active_detail = detail
        self._refresh_active_detail(progress)
        # Traced last, so the update that opens the report is itself the
        # trace's first row - which is what dates the start of the artwork.
        # Pauses are traced too: an unexplained gap in the timeline is the
        # first thing looked for when a run took longer than it should have.
        report = self._run_report
        if report is not None:
            recorder = self._timelapse_recorder
            report.sample_progress(
                progress,
                timelapse_frame=int(getattr(recorder, "frame_count", 0) or 0),
            )
        self._note_resume_progress(progress)

    _PAUSED_VIEWFINDER_NOTICE = (
        "Paused.  Slide to see the picture as far as any stroke and check "
        "the sign against it; the job carries on from where it stopped "
        "whatever the slider says."
    )

    def _offer_paused_viewfinder(self) -> None:
        """Put the paused job's stroke on the slider and show the picture to there."""

        painter = self._painter
        plan = self._plan
        if painter is None or not getattr(plan, "stroke_count", 0):
            return
        if self._paused_viewfinder_notice is None:
            self._paused_viewfinder_notice = self.resume_notice.text()
        try:
            completed = int(painter.progress.completed_strokes)
        except Exception:
            completed = 0
        slider = self.resume_slider
        slider.blockSignals(True)
        try:
            slider.setRange(0, plan.stroke_count)
            slider.setValue(max(0, min(completed, plan.stroke_count)))
        finally:
            slider.blockSignals(False)
        self.resume_notice.setText(self._PAUSED_VIEWFINDER_NOTICE)
        self._on_resume_controls_changed()

    def _withdraw_paused_viewfinder(self) -> None:
        """Give the slider and the preview back to the next job's resume offer."""

        notice = self._paused_viewfinder_notice
        if notice is None:
            return
        self._paused_viewfinder_notice = None
        self.resume_notice.setText(notice)
        plan = self._plan
        slider = self.resume_slider
        slider.blockSignals(True)
        try:
            slider.setValue(self._resume_offer_value)
        finally:
            slider.blockSignals(False)
        if plan is not None:
            self._on_resume_controls_changed()

    @Slot()
    def _refresh_active_detail(self, progress: Any = None) -> None:
        """Write the line under the progress bar: stroke, elapsed, next break."""

        painter = self._painter
        if progress is None:
            if painter is None:
                return
            progress = painter.progress
        parts = [self._active_detail, f"{self._format_duration(progress.elapsed_seconds)} elapsed"]
        until_break = None
        if painter is not None:
            try:
                until_break = painter.seconds_until_anti_afk()
            except Exception:
                until_break = None
        if until_break is not None:
            parts.append(
                "anti-AFK break due now"
                if until_break <= 0
                else f"anti-AFK in {self._format_duration(until_break)}"
            )
        self.active_detail_label.setText("  •  ".join(part for part in parts if part))

    @Slot(int, object, str)
    def _on_paint_state(self, generation: int, state: Any, reason: str) -> None:
        if generation != self._paint_generation:
            return
        value = getattr(state, "value", str(state))
        painter = self._painter
        if painter is not None:
            current = getattr(getattr(painter, "state", None), "value", None)
            if current is not None and current != value:
                # State callbacks can originate on both the worker and global
                # hotkey threads.  Ignore a callback whose transition has
                # already been superseded by a same-generation emergency stop.
                return
        badge_text = {
            "running": "PAINTING",
            "paused": "PAUSED",
            "error": "ERROR",
            "aborted": "STOPPED",
        }.get(value, value.upper())
        self._set_state_badge(value, badge_text)
        active = value in {"countdown", "running", "paused"}
        self._set_active_progress_visible(active)
        if active:
            self.active_progress_title.setText(
                {"countdown": "GET READY", "paused": "PAUSED"}.get(value, "PAINTING")
            )
        if value == "paused":
            # Where the job stopped, and why, written the moment it does:
            # a pause the UI guard called is the one a resume is for, and
            # the app may well be closed before the job is resumed.
            self._write_resume_record(state="paused", reason=reason)
            if reason not in self._USER_PAUSE_REASONS:
                self._capture_pause_screenshot(generation, reason)
            self._offer_paused_viewfinder()
        elif value == "running":
            self.pause_screenshot_button.setVisible(False)
            self._withdraw_paused_viewfinder()
        if value in {"completed", "aborted", "error"}:
            self._withdraw_paused_viewfinder()
            self._status_overlay_linger.start()
        if value in {"completed", "aborted", "error"}:
            self._finish_timelapse(final=value == "completed")
            self._finish_run_report(value, reason)
            self._close_resume_record(value, reason)
            self._learn_timing()
            # The run measured this sign whatever its outcome, and that
            # measurement is what stops the next plan from asking for a
            # resolution the brush cannot paint.  Losing it because the
            # user stopped the job is how the same mistake gets repeated.
            self._store_measured_brush_model()
        if reason:
            self.statusBar().showMessage(f"{value.title()}: {reason}", 5000)
        LOGGER.info("Painter state: %s (%s)", value, reason)
        self._update_start_availability()

    @Slot(int)
    def _on_paint_complete(self, generation: int) -> None:
        if generation != self._paint_generation:
            return
        self.paint_progress.setValue(1000)
        self.progress_state_label.setText("Completed")
        self._set_active_progress_visible(False)
        self._set_state_badge("completed", "COMPLETE")
        self._store_measured_brush_model()
        LOGGER.info("Paint plan completed")
        if self._painter is not None and getattr(self._painter.input, "is_dry_run", False):
            LOGGER.info(
                "Dry run evaluated %d strokes without emitting input",
                self._painter.progress.completed_strokes,
            )
        self._update_start_availability()

    @Slot(int, str)
    def _on_paint_error(self, generation: int, message: str) -> None:
        if generation != self._paint_generation:
            return
        self.progress_state_label.setText("Error")
        self.progress_detail_label.setText(message)
        self._set_active_progress_visible(False)
        self._set_state_badge("error", "ERROR")
        LOGGER.error("Painting error: %s", message)
        QMessageBox.critical(self, "Painting stopped", message)
        self._update_start_availability()

    # -------------------------------------------------------------- timelapse

    def _maybe_start_run_report(self) -> None:
        """Open this run's diagnostic folder as the artwork starts.

        Started from the same painting-phase progress update as the timelapse,
        so the report describes the picture rather than the probe strokes the
        job wipes off first - and so the brush the job measured for itself is
        already known and can be written down beside the plan that assumed it.
        """

        if self._run_report is not None or self._closing:
            return
        painter = self._painter
        plan = self._plan
        if painter is None or plan is None:
            return
        try:
            from app.run_report import RunReport

            report = RunReport(self._local_data_directory() / "runs")
        except Exception:
            LOGGER.exception("Could not open a run report")
            return
        self._run_report = report
        snapshot = self._paint_job_snapshot
        recorder = self._timelapse_recorder
        # The countdown snapshot is the truthful record; reading the live
        # controls is only a fallback for a job that reached painting without
        # one, and is still far better than a report with no settings at all.
        settings = getattr(snapshot, "settings", None) or self._settings_document()
        report.record_context(
            settings=settings,
            profile=getattr(snapshot, "profile", None) or self._current_profile,
            timelapse_directory=(
                recorder.directory if recorder is not None else None
            ),
            image_path=self._image_path,
            dry_run=bool(getattr(snapshot, "dry_run", False)),
        )
        report.record_plan(plan, self._processed)
        canvas = self._profile_rect("canvas")
        if canvas is not None:
            report.record_brush(
                painter.measured_brush_size_model,
                canvas_height=float(canvas.height),
                canvas_width=float(canvas.width),
                plan_width=plan.width,
                plan_height=plan.height,
                texel_grid=getattr(painter, "measured_texel_grid", None),
            )
        # A full-screen PNG takes long enough to encode that capturing it on
        # the GUI thread would visibly stall the first strokes' progress.
        threading.Thread(
            target=report.record_screen,
            name="RustPainterRunReportScreen",
            daemon=True,
        ).start()
        LOGGER.info("Run report recording to %s", report.directory)

    def _finish_run_report(self, outcome: str, reason: str) -> None:
        """Close the report out, capturing the sign as the run left it.

        An aborted or failed run is the one worth keeping: nobody needs a
        finished sign explained.
        """

        report = self._run_report
        if report is None:
            return
        self._run_report = None
        canvas = self._profile_rect("canvas")
        painter = self._painter
        plan = self._plan
        if canvas is not None and plan is not None and painter is not None:
            try:
                report.record_brush(
                    painter.measured_brush_size_model,
                    canvas_height=float(canvas.height),
                    canvas_width=float(canvas.width),
                    plan_width=plan.width,
                    plan_height=plan.height,
                    texel_grid=getattr(painter, "measured_texel_grid", None),
                )
            except Exception:
                LOGGER.exception("Could not record the brush in the run report")
        if painter is not None:
            try:
                report.record_confirmation(painter.confirmation_summary)
                report.record_color_picks(painter.color_pick_summary)
            except Exception:
                LOGGER.exception("Could not record the color checks in the run report")

        def wrap_up() -> None:
            if canvas is not None:
                report.record_canvas(canvas, "canvas_final")
            report.finish(outcome, reason)

        threading.Thread(
            target=wrap_up, name="RustPainterRunReportFinish", daemon=True
        ).start()

    def _maybe_start_timelapse(self) -> None:
        """Begin recording frames once a real paint job starts on the artwork."""

        if self._timelapse_recorder is not None or self._closing:
            return
        if not self.timelapse_check.isChecked():
            return
        painter = self._painter
        if painter is None or not getattr(painter.input, "emits_real_input", True):
            return
        canvas = self._profile_rect("canvas")
        if canvas is None:
            return
        from app.timelapse import TimelapseRecorder

        recorder = TimelapseRecorder(
            self._local_data_directory() / "timelapse", canvas
        )
        self._timelapse_recorder = recorder
        interval = max(1, self.timelapse_interval_spin.value())
        self._timelapse_timer.setInterval(interval * 1000)
        self._timelapse_timer.start()
        # The first frame shows the sign as painting begins.
        self._schedule_timelapse_frame(recorder)
        self._update_timelapse_status()
        LOGGER.info(
            "Timelapse recording to %s (a frame every %ds)",
            recorder.directory,
            interval,
        )

    def _sync_timelapse_with_controls(self) -> None:
        """Make the recording match the timelapse controls as a pause ends.

        The controls stay live through a pause so a job that started without
        a recording can get one from here on - the wish to keep a run tends
        to arrive once it is plainly going well - and so one that has a
        recording can change its pace or stop it.  A job still measuring its
        brush is left to the painting-phase progress update that starts
        every recording, so the video opens on the artwork and not on the
        probe strokes.
        """

        recorder = self._timelapse_recorder
        if not self.timelapse_check.isChecked():
            if recorder is not None:
                LOGGER.info("Timelapse recording stopped from the paused job")
                self._finish_timelapse(final=False)
            return
        interval = max(1, self.timelapse_interval_spin.value()) * 1000
        if recorder is None:
            progress = getattr(self._painter, "progress", None)
            if getattr(progress, "phase", "paint") != "calibrate":
                self._maybe_start_timelapse()
            return
        if self._timelapse_timer.interval() != interval:
            # Setting the interval restarts a running timer at the new pace.
            self._timelapse_timer.setInterval(interval)
            LOGGER.info("Timelapse interval changed to %ds", interval // 1000)
            self._update_timelapse_status()

    @Slot()
    def _capture_timelapse_frame(self) -> None:
        recorder = self._timelapse_recorder
        if recorder is None or self._closing:
            return
        painter = self._painter
        state = (
            getattr(getattr(painter, "state", None), "value", None)
            if painter is not None
            else None
        )
        # A paused job is not making visible progress; skip those frames.
        if state != "running":
            self._update_timelapse_status()
            return
        self._schedule_timelapse_frame(recorder)
        self._update_timelapse_status()

    def _update_timelapse_status(self) -> None:
        """Keep the page's badge honest about what recording is doing."""

        recorder = self._timelapse_recorder
        if recorder is None:
            self.timelapse_status_badge.setText("Not recording")
            return
        painter = self._painter
        state = (
            getattr(getattr(painter, "state", None), "value", None)
            if painter is not None
            else None
        )
        frames = recorder.frame_count
        label = "Paused" if state != "running" else "Recording"
        self.timelapse_status_badge.setText(
            f"{label} • {frames} frame{'s' if frames != 1 else ''} • "
            f"{recorder.directory.name}"
        )

    @staticmethod
    def _schedule_timelapse_frame(recorder: Any) -> None:
        # Captured off the GUI thread so encoding a large PNG never stalls
        # progress updates; the recorder skips a frame if one is in flight.
        threading.Thread(
            target=recorder.capture_frame,
            name="RustPainterTimelapse",
            daemon=True,
        ).start()

    def _finish_timelapse(self, *, final: bool) -> None:
        recorder = self._timelapse_recorder
        if recorder is None:
            return
        self._timelapse_timer.stop()
        self._timelapse_recorder = None
        capture_final = final and self.timelapse_final_check.isChecked()

        def wrap_up() -> None:
            if capture_final:
                recorder.capture_frame()
            LOGGER.info(
                "Timelapse saved %d frames to %s",
                recorder.frame_count,
                recorder.directory,
            )

        threading.Thread(
            target=wrap_up, name="RustPainterTimelapseFinish", daemon=True
        ).start()
        self._update_timelapse_status()
        self._refresh_timelapse_sessions()
        self.statusBar().showMessage(
            f"Timelapse frames saved to {recorder.directory}", 8000
        )

    def _timelapse_root(self) -> Path:
        return self._local_data_directory() / "timelapse"

    @Slot()
    def _show_timelapse_page(self) -> None:
        self.page_stack.setCurrentIndex(1)
        self._refresh_timelapse_sessions()
        self._update_timelapse_status()

    @Slot()
    def _refresh_timelapse_sessions(self) -> None:
        """List every recorded session, newest first, with its frame count."""

        selected = {path.name for path in self._selected_session_paths()}
        self.timelapse_sessions.clear()
        root = self._timelapse_root()
        try:
            sessions = sorted(
                (path for path in root.iterdir() if path.is_dir()),
                key=lambda path: path.name,
                reverse=True,
            )
        except OSError:
            sessions = []
        for session in sessions:
            frames = session_frames(session)
            megabytes = sum(frame.stat().st_size for frame in frames) / (1024 * 1024)
            item = QListWidgetItem(
                f"{session.name}  •  {len(frames)} frame"
                f"{'s' if len(frames) != 1 else ''}  •  {megabytes:.1f} MB"
            )
            item.setData(Qt.ItemDataRole.UserRole, str(session))
            self.timelapse_sessions.addItem(item)
            if session.name in selected:
                # setCurrentItem would clear the rest of the selection, so the
                # first survivor sets the current row and the others only join
                # the selection.
                if self.timelapse_sessions.currentItem() is None:
                    self.timelapse_sessions.setCurrentItem(item)
                item.setSelected(True)
        if not sessions:
            placeholder = QListWidgetItem("No recordings yet")
            placeholder.setFlags(Qt.ItemFlag.NoItemFlags)
            self.timelapse_sessions.addItem(placeholder)
        self._sync_session_buttons()

    def _selected_session_paths(self) -> list[Path]:
        """Every picked recording, in the order the list shows them."""

        paths = []
        for item in self.timelapse_sessions.selectedItems():
            value = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(value, str):
                paths.append(Path(value))
        return paths

    def _selected_session_path(self) -> Path | None:
        """The picked recording, when exactly one is picked.

        The actions that show a recording work on one at a time, and a
        multiple selection is not a vague version of a single one: it is a
        different request, so those actions decline it rather than guessing
        which member of it was meant.
        """

        paths = self._selected_session_paths()
        return paths[0] if len(paths) == 1 else None

    def _sync_session_buttons(self) -> None:
        """Offer exactly the actions that make sense for what is picked.

        Several recordings cannot be watched or encoded at once, but they can
        certainly be deleted at once, so the buttons part company here rather
        than all following the same "something is selected" flag.
        """

        sessions = self._selected_session_paths()
        count = len(sessions)
        exporting = self._timelapse_export is not None
        single = sessions[0] if count == 1 else None
        # An empty session folder can be opened and deleted but has nothing to
        # watch, so the two buttons that need frames check for them.
        has_frames = single is not None and bool(session_frames(single))
        self.open_session_button.setEnabled(single is not None)
        self.play_session_button.setEnabled(has_frames)
        self.export_session_button.setEnabled(has_frames and not exporting)
        self.delete_session_button.setEnabled(count > 0 and not exporting)
        self.timelapse_format_combo.setEnabled(not exporting)
        several = "Pick a single recording to do that with it."
        self.play_session_button.setToolTip(
            several if count > 1 else "Play the selected recording back inside RustPainter."
        )
        self.export_session_button.setToolTip(
            several
            if count > 1
            else (
                "Write the selected recording to a single video file you can "
                "keep, upload, or share."
            )
        )
        self.open_session_button.setToolTip(
            several if count > 1 else "Open the selected recording's folder"
        )
        self.delete_session_button.setToolTip(
            f"Delete the {count} selected recordings and every frame in them"
            if count > 1
            else "Delete the selected recording and every frame in it"
        )
        self._refresh_session_selection_label(sessions)

    def _refresh_session_selection_label(self, sessions: list[Path]) -> None:
        """Say what a multiple selection adds up to before it is deleted."""

        if len(sessions) < 2:
            self.timelapse_selection_label.setText("")
            return
        frames = [frame for session in sessions for frame in session_frames(session)]
        megabytes = sum(frame.stat().st_size for frame in frames) / (1024 * 1024)
        self.timelapse_selection_label.setText(
            f"{len(sessions)} recordings selected  •  {len(frames):,} frame"
            f"{'s' if len(frames) != 1 else ''}  •  {megabytes:.1f} MB"
        )

    @Slot()
    def _play_selected_session(self) -> None:
        """Open a modeless player for the selected recording."""

        session = self._selected_session_path()
        if session is None or not session.is_dir():
            return
        frames = session_frames(session)
        if not frames:
            QMessageBox.information(
                self,
                "Nothing to play",
                f"“{session.name}” has no captured frames yet.",
            )
            return
        try:
            from .timelapse_player import TimelapsePlayer

            player = TimelapsePlayer(
                session.name,
                frames,
                self,
                frame_rate=self.timelapse_speed_slider.value(),
            )
        except Exception as exc:
            LOGGER.exception("Could not open the timelapse player")
            QMessageBox.warning(self, "Could not play the recording", str(exc))
            return
        player.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        player.destroyed.connect(
            lambda _obj=None, ref=player: self._forget_timelapse_player(ref)
        )
        self._timelapse_players.append(player)
        player.show()
        player.raise_()
        player.activateWindow()
        player.play()
        LOGGER.info("Playing timelapse %s (%d frames)", session.name, len(frames))

    def _forget_timelapse_player(self, player: Any) -> None:
        try:
            self._timelapse_players.remove(player)
        except ValueError:
            pass

    @Slot()
    def _export_selected_session(self) -> None:
        """Encode the selected recording into a single video file."""

        if self._timelapse_export is not None:
            QMessageBox.information(
                self,
                "Export in progress",
                "One recording is already being saved. Wait for it to finish.",
            )
            return
        session = self._selected_session_path()
        if session is None or not session.is_dir():
            return
        frames = session_frames(session)
        if not frames:
            QMessageBox.information(
                self,
                "Nothing to export",
                f"“{session.name}” has no captured frames yet.",
            )
            return
        video_format = format_for(str(self.timelapse_format_combo.currentData()))
        suggested = (
            self._last_export_directory() / f"{session.name}{video_format.suffix}"
        )
        chosen, _filter = QFileDialog.getSaveFileName(
            self,
            "Save timelapse as video",
            str(suggested),
            f"{video_format.filter_text};;All files (*)",
        )
        if not chosen:
            return
        destination = Path(chosen)
        if not destination.suffix:
            destination = destination.with_suffix(video_format.suffix)
        worker = _TimelapseExportWorker(
            frames,
            destination,
            self.timelapse_speed_slider.value(),
            video_format.key,
        )
        worker.signals.progress.connect(self._on_export_progress)
        worker.signals.completed.connect(self._on_export_completed)
        worker.signals.failed.connect(self._on_export_failed)
        worker.signals.cancelled.connect(self._on_export_cancelled)
        self._timelapse_export = worker
        self.timelapse_export_progress.setRange(0, len(frames))
        self.timelapse_export_progress.setValue(0)
        self.timelapse_export_progress.setFormat(
            f"Saving {destination.name} — %v of %m frames"
        )
        self.timelapse_export_progress.setVisible(True)
        self._sync_session_buttons()
        LOGGER.info(
            "Exporting %d frames of %s to %s at %d fps",
            len(frames),
            session.name,
            destination,
            self.timelapse_speed_slider.value(),
        )
        self._timelapse_export_pool.start(worker)

    def _last_export_directory(self) -> Path:
        stored = self._settings.get("ui", {}).get("last_video_export_directory")
        if isinstance(stored, str) and stored:
            candidate = Path(stored)
            if candidate.is_dir():
                return candidate
        return Path.home()

    @Slot(int, int)
    def _on_export_progress(self, done: int, total: int) -> None:
        if self._timelapse_export is None:
            return
        self.timelapse_export_progress.setRange(0, total)
        self.timelapse_export_progress.setValue(done)

    @Slot(str)
    def _on_export_completed(self, destination: str) -> None:
        path = Path(destination)
        self._finish_export()
        self._settings.setdefault("ui", {})["last_video_export_directory"] = str(
            path.parent
        )
        self._schedule_settings_save()
        self.statusBar().showMessage(f"Timelapse saved to {path}", 10000)
        if (
            QMessageBox.information(
                self,
                "Timelapse saved",
                f"Saved to:\n{path}\n\nOpen the folder it is in?",
                QMessageBox.StandardButton.Open | QMessageBox.StandardButton.Close,
                QMessageBox.StandardButton.Close,
            )
            == QMessageBox.StandardButton.Open
        ):
            self._open_in_file_manager(path.parent)

    @Slot(str)
    def _on_export_failed(self, message: str) -> None:
        self._finish_export()
        QMessageBox.warning(self, "Could not save the timelapse", message)

    @Slot()
    def _on_export_cancelled(self) -> None:
        self._finish_export()
        self.statusBar().showMessage("Timelapse export cancelled", 5000)

    def _finish_export(self) -> None:
        self._timelapse_export = None
        self.timelapse_export_progress.setVisible(False)
        self.timelapse_export_progress.reset()
        if not self._closing:
            self._sync_session_buttons()

    @Slot()
    def _open_timelapse_folder(self) -> None:
        directory = self._timelapse_root()
        directory.mkdir(parents=True, exist_ok=True)
        self._open_in_file_manager(directory)

    @Slot()
    def _open_selected_session(self) -> None:
        session = self._selected_session_path()
        if session is not None and session.is_dir():
            self._open_in_file_manager(session)

    @staticmethod
    def _open_in_file_manager(directory: Path) -> None:
        from PySide6.QtCore import QUrl
        from PySide6.QtGui import QDesktopServices

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(directory)))

    @Slot()
    def _delete_selected_sessions(self) -> None:
        """Delete every picked recording, live ones excepted, after one prompt.

        Clearing out old recordings is a batch job, so it asks once for the
        whole batch and reports once for it.  A recording the current paint
        job is still writing to is dropped from the batch rather than taking
        the rest of it down with an error.
        """

        sessions = [
            session for session in self._selected_session_paths() if session.is_dir()
        ]
        if not sessions:
            return
        live = getattr(self._timelapse_recorder, "directory", None)
        held = [session for session in sessions if session == live]
        sessions = [session for session in sessions if session != live]
        if not sessions:
            QMessageBox.information(
                self,
                "Recording in progress",
                "That session is still being recorded. Let the paint job finish "
                "before deleting it.",
            )
            return
        frames = sum(len(session_frames(session)) for session in sessions)
        plural = "s" if frames != 1 else ""
        subject = (
            f"“{sessions[0].name}” and its {frames} frame{plural}"
            if len(sessions) == 1
            else f"{len(sessions)} recordings and their {frames:,} frame{plural}"
        )
        held_note = (
            "\n\nThe recording still being written by the current paint job is "
            "left alone."
            if held
            else ""
        )
        if (
            QMessageBox.question(
                self,
                "Delete recording" if len(sessions) == 1 else "Delete recordings",
                f"Delete {subject}? This cannot be undone.{held_note}",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.Cancel,
                QMessageBox.StandardButton.Cancel,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        import shutil

        deleted: list[Path] = []
        failure: str | None = None
        for session in sessions:
            try:
                shutil.rmtree(session)
            except OSError as exc:
                LOGGER.exception("Could not delete a timelapse session")
                failure = failure or f"{session.name}: {exc}"
                continue
            LOGGER.info("Deleted timelapse session %s", session.name)
            deleted.append(session)
        self._refresh_timelapse_sessions()
        if deleted:
            self.statusBar().showMessage(
                f"Deleted {deleted[0].name}"
                if len(deleted) == 1
                else f"Deleted {len(deleted)} recordings",
                5000,
            )
        if failure is not None:
            QMessageBox.warning(
                self,
                "Could not delete every recording"
                if deleted
                else "Could not delete the recording",
                failure,
            )

    def _set_idle_ui(self, detail: str = "No active paint job") -> None:
        self.progress_state_label.setText("Idle")
        self.progress_detail_label.setText(detail)
        self._set_active_progress_visible(False)
        self._set_state_badge("idle", "IDLE")
        self._update_start_availability()

    # --------------------------------------------------------------- debug mode

    @Slot()
    def _run_debug_action(self, action: str) -> None:
        if self._debug_running or self._countdown_callback_running or (
            self._countdown is not None and self._countdown.isVisible()
        ):
            self.statusBar().showMessage("Another operation is already in progress.", 4000)
            return
        required = {
            "canvas_tl": ("canvas",),
            "canvas_center": ("canvas",),
            "canvas_br": ("canvas",),
            "test_hue": ("hue_bar",),
            "test_sv": ("color_box",),
            "test_dot": ("canvas",),
            "test_stroke": ("canvas",),
        }.get(action, ())
        missing = [name for name in required if self._profile_rect(name) is None]
        if missing:
            QMessageBox.information(
                self,
                "Calibration needed",
                "Calibrate " + ", ".join(name.replace("_", " ") for name in missing) + " first.",
            )
            return
        if self._painter_is_active():
            QMessageBox.warning(self, "Painting is active", "Pause or stop the paint job first.")
            return
        if (
            not self.dry_run_check.isChecked()
            and not self._emergency_hotkey_available()
        ):
            QMessageBox.critical(
                self,
                "Emergency hotkey unavailable",
                "Real calibration tests are disabled because the global stop hotkey "
                "is not active. Choose three distinct, available hotkeys and try again.",
            )
            return
        if not self.dry_run_check.isChecked():
            try:
                self._validate_profile_on_virtual_screen()
            except ValueError as exc:
                QMessageBox.critical(self, "Calibration is outside the desktop", str(exc))
                return
        self._pending_start_cancelled = False
        with self._debug_input_gate:
            self._debug_abort_event.clear()
        if self.dry_run_check.isChecked():
            self._execute_debug_action(action)
            return
        self._pending_debug_action = action
        self._launch_countdown(
            2,
            lambda: self._execute_debug_action(self._pending_debug_action),
            hint=f"{self.abort_hotkey_combo.currentText()} cancels",
        )

    def _execute_debug_action(self, action: str) -> None:
        if self._pending_start_cancelled or self._closing:
            self._set_idle_ui("Debug action cancelled")
            return
        try:
            from app.color_mapping import map_hue_to_screen, map_sv_to_screen
            from app.input_controller import (
                DryRunInputController,
                create_system_input_controller,
            )
            from app.screen import foreground_window_matches

            dry_run = self.dry_run_check.isChecked()
            if not dry_run and not self._emergency_hotkey_available():
                raise RuntimeError(
                    "The global stop hotkey stopped before the debug action began."
                )
            expected_title = self.expected_window_edit.text().strip()
            expected_process = self.expected_process_edit.text().strip()
            require_foreground = not dry_run and self.focus_guard_check.isChecked()
            if require_foreground and not expected_title and not expected_process:
                raise RuntimeError(
                    "Foreground protection needs an expected window title or process name."
                )
            canvas = self._profile_rect("canvas")
            color_box = self._profile_rect("color_box")
            hue_bar = self._profile_rect("hue_bar")
            description = action
            operation = "move"
            start: tuple[float, float]
            end: tuple[float, float] | None = None
            if action == "canvas_tl" and canvas:
                start = (canvas.left, canvas.top)
                description = f"move to canvas top-left {start}"
            elif action == "canvas_center" and canvas:
                start = tuple(round(value) for value in canvas.center)
                description = f"move to canvas center {start}"
            elif action == "canvas_br" and canvas:
                start = (canvas.right - 1, canvas.bottom - 1)
                description = f"move to canvas bottom-right {start}"
            elif action == "test_hue" and hue_bar:
                start = map_hue_to_screen(
                    180.0, hue_bar, "bottom_to_top"
                )
                operation = "click"
                description = f"click 180° hue at ({start[0]:.1f}, {start[1]:.1f})"
            elif action == "test_sv" and color_box:
                start = map_sv_to_screen(
                    0.5,
                    0.75,
                    color_box,
                    "left_low",
                    "top_bright",
                )
                operation = "click"
                description = f"click S=0.50 V=0.75 at ({start[0]:.1f}, {start[1]:.1f})"
            elif action == "test_dot" and canvas:
                start = canvas.center
                operation = "click"
                description = f"paint dot at canvas center {canvas.center}"
            elif action == "test_stroke" and canvas:
                center_x, center_y = canvas.center
                half = max(3.0, min(40.0, canvas.width * 0.04))
                start = (center_x - half, center_y)
                end = (center_x + half, center_y)
                operation = "drag"
                description = (
                    f"paint stroke from ({center_x - half:.1f}, {center_y:.1f}) "
                    f"to ({center_x + half:.1f}, {center_y:.1f})"
                )
            else:
                raise ValueError(f"Unknown or unavailable debug action: {action}")

            hold_seconds = self.dot_duration_spin.value() / 1000.0
            speed = max(1.0, self.stroke_speed_spin.value())
            step_pixels = self.interpolation_spin.value()
            detailed_logging = bool(
                self._settings_document()
                .get("execution", {})
                .get("debug_mouse_logging", False)
            )

            with self._debug_input_gate:
                if self._pending_start_cancelled or self._debug_abort_event.is_set():
                    raise _DebugCancelled("cancelled before input")
                self._debug_abort_event.clear()
                controller = (
                    DryRunInputController(detailed_logging=detailed_logging)
                    if dry_run
                    else create_system_input_controller()
                )
                if self._pending_start_cancelled or self._debug_abort_event.is_set():
                    controller.release_all()
                    raise _DebugCancelled("cancelled before input")
                self._debug_controller = controller
                self._debug_running = True

            self._update_start_availability()

            def run_debug() -> None:
                def checkpoint() -> None:
                    if self._debug_abort_event.is_set() or self._closing:
                        raise _DebugCancelled("emergency stop requested")
                    if require_foreground and not foreground_window_matches(
                        title_contains=expected_title or None,
                        executable=expected_process or None,
                    ):
                        self._debug_abort_event.set()
                        raise _DebugCancelled("expected Rust window lost foreground")

                def guarded_move(point: tuple[float, float]) -> None:
                    with self._debug_input_gate:
                        checkpoint()
                        controller.move_mouse(*point)

                def guarded_down() -> None:
                    with self._debug_input_gate:
                        checkpoint()
                        controller.mouse_down()

                def interruptible_wait(seconds: float) -> None:
                    if controller.skip_timing or seconds <= 0:
                        return
                    deadline = time.monotonic() + seconds
                    while True:
                        remaining = deadline - time.monotonic()
                        if remaining <= 0:
                            return
                        if self._debug_abort_event.wait(min(0.01, remaining)):
                            raise _DebugCancelled("emergency stop requested")
                        with self._debug_input_gate:
                            checkpoint()

                status = "completed"
                message = description
                try:
                    if operation == "move":
                        guarded_move(start)
                    elif operation == "click":
                        guarded_move(start)
                        guarded_down()
                        try:
                            interruptible_wait(hold_seconds)
                        finally:
                            with self._debug_input_gate:
                                controller.mouse_up()
                    elif operation == "drag" and end is not None:
                        guarded_move(start)
                        guarded_down()
                        try:
                            distance = math.hypot(end[0] - start[0], end[1] - start[1])
                            steps = max(1, math.ceil(distance / step_pixels))
                            delay = (distance / speed) / steps
                            for index in range(1, steps + 1):
                                ratio = index / steps
                                point = (
                                    start[0] + (end[0] - start[0]) * ratio,
                                    start[1] + (end[1] - start[1]) * ratio,
                                )
                                guarded_move(point)
                                interruptible_wait(delay)
                        finally:
                            with self._debug_input_gate:
                                controller.mouse_up()
                except _DebugCancelled as exc:
                    status = "cancelled"
                    message = str(exc)
                except Exception as exc:
                    LOGGER.exception("Debug action failed")
                    status = "error"
                    message = str(exc)
                finally:
                    try:
                        with self._debug_input_gate:
                            controller.release_all()
                    except Exception as exc:
                        LOGGER.exception("Could not release debug input")
                        if status == "completed":
                            status = "error"
                            message = f"Could not release input: {exc}"
                    self._painter_bridge.debug_finished.emit(status, message)

            thread = threading.Thread(
                target=run_debug,
                name="RustPainterDebugWorker",
                daemon=True,
            )
            self._debug_thread = thread
            try:
                thread.start()
            except Exception:
                with self._debug_input_gate:
                    controller.release_all()
                    self._debug_controller = None
                self._debug_thread = None
                self._debug_running = False
                self._update_start_availability()
                raise
        except Exception as exc:
            LOGGER.exception("Debug action failed")
            QMessageBox.warning(self, "Debug action stopped", str(exc))

    @Slot(str, str)
    def _on_debug_finished(self, status: str, message: str) -> None:
        with self._debug_input_gate:
            controller = self._debug_controller
            if controller is not None:
                try:
                    # The worker already releases in ``finally``.  Retry here
                    # because SendInputController deliberately retains failed
                    # button-up state so a transient failure cannot be lost.
                    controller.release_all()
                    self._debug_controller = None
                except Exception as exc:
                    LOGGER.exception("Could not retry final debug-input release")
                    status = "error"
                    message = f"Could not release input: {exc}"
        # A controller is retained only when its final release failed.  Keep
        # real-input starts locked and the Stop button enabled until a later
        # abort/close retry succeeds; never overwrite the only record of a
        # potentially held mouse button.
        self._debug_running = self._debug_controller is not None
        self._debug_thread = None
        if self._closing:
            return
        if status == "completed":
            LOGGER.info(
                "Debug action completed: %s%s",
                message,
                " (dry run; no input emitted)" if self.dry_run_check.isChecked() else "",
            )
            self.statusBar().showMessage("Debug action completed", 5000)
        elif status == "cancelled":
            LOGGER.warning("Debug action cancelled: %s", message)
            self._set_idle_ui(f"Debug action cancelled: {message}")
        else:
            LOGGER.error("Debug action failed: %s", message)
            QMessageBox.warning(self, "Debug action stopped", message)
        self._update_start_availability()

    # -------------------------------------------------------------- shutdown

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self._closing = True
        self._pending_start_cancelled = True
        self._pending_paint = None
        self._process_timer.stop()
        self._settings_timer.stop()
        self._rust_monitor_timer.stop()
        self._timelapse_timer.stop()
        self._timelapse_recorder = None
        if self._timelapse_export is not None:
            # The worker deletes its half-written file on the way out, so the
            # user is never left with a video that stops mid-paint.
            self._timelapse_export.cancel()
            self._timelapse_export = None
        self._timelapse_export_pool.waitForDone(3000)
        for player in list(self._timelapse_players):
            try:
                player.close()
            except Exception:
                LOGGER.exception("Could not close a timelapse player")
        self._timelapse_players.clear()
        try:
            if self._calibration_preview is not None:
                self._calibration_preview.close()
                self._calibration_preview = None
        except Exception:
            LOGGER.exception("Could not close the calibration overlay")

        # Emergency input cleanup comes first and is isolated from every
        # ancillary shutdown step so a settings/hotkey error cannot skip it.
        self._debug_abort_event.set()
        try:
            with self._debug_input_gate:
                if self._debug_controller is not None:
                    self._debug_controller.release_all()
        except Exception:
            LOGGER.exception("Could not perform final debug-input release")
        if self._painter is not None:
            try:
                self._painter.shutdown(timeout=2.0)
            except Exception:
                LOGGER.exception("Could not shut down the painter")
            finally:
                try:
                    self._painter.input.release_all()
                except Exception:
                    LOGGER.exception("Could not perform final input release")
        try:
            if self._countdown is not None:
                self._countdown.reject()
        except Exception:
            LOGGER.exception("Could not close countdown")
        try:
            if self._hotkeys is not None:
                self._hotkeys.stop()
        except Exception:
            LOGGER.exception("Could not stop global hotkeys")
        try:
            self._save_settings()
        except Exception:
            LOGGER.exception("Could not save settings during shutdown")
        try:
            debug_thread = self._debug_thread
            if debug_thread is not None and debug_thread is not threading.current_thread():
                debug_thread.join(1500 / 1000.0)
                if debug_thread.is_alive():
                    LOGGER.warning("Debug worker is still stopping in the background")
            # Retry after the worker's own finally block.  SendInputController
            # retains failed button-up state specifically so shutdown still has
            # another opportunity to release it.
            with self._debug_input_gate:
                if self._debug_controller is not None:
                    self._debug_controller.release_all()
        except Exception:
            LOGGER.exception("Could not stop debug worker")
        try:
            self._load_pool.clear()
            if not self._load_pool.waitForDone(1500):
                LOGGER.warning("Waiting for the active image decode to finish before exit")
                self._load_pool.waitForDone()
        except Exception:
            LOGGER.exception("Could not stop image-loading workers")
        try:
            self._thread_pool.clear()
            if not self._thread_pool.waitForDone(1500):
                LOGGER.warning("Waiting for active image processing to finish before exit")
                self._thread_pool.waitForDone()
        except Exception:
            LOGGER.exception("Could not stop image workers")
        logging.getLogger("rust_painter").removeHandler(self._qt_log_handler)
        LOGGER.info("RustPainter closed")
        event.accept()
