"""Main PySide6 window for RustPainter."""

from __future__ import annotations

import logging
import math
import os
import sys
import threading
import time
from copy import deepcopy
from dataclasses import dataclass, replace
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
)
from PySide6.QtWidgets import (
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
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from app.color_calibration import ColorCorrectionModel
from app.image_processing import process_image, quantize_image
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
    ScreenRect,
    TransparencyMode,
)
from app.paint_plan import count_unmerged_strokes, generate_paint_plan
from app.paint_optimizer import (
    BrushCapabilities,
    OptimizationStatistics,
    mode_options,
    optimize_paint_plan,
    simplify_colors,
)
from app.profiles import Profile, ProfileStore
from app.settings import SettingsStore, default_settings

from .assets import icon as art_icon, pixmap as art_pixmap, tinted_pixmap
from .styles import ON_ACCENT, TEXT, badge_foreground, state_badge_style
from .widgets import (
    CalibrationStatus,
    ColorButton,
    CountdownDialog,
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

# All timing values one speed preset controls, in the spinbox units used below.
SPEED_PRESETS: dict[str, dict[str, float]] = {
    "Relaxed": {
        "stroke_speed": 400.0,
        "dot_ms": 40,
        "hue_ms": 120,
        "sv_ms": 120,
        "stroke_ms": 35,
        "color_ms": 180,
        "interp_px": 3.0,
    },
    "Standard": {
        "stroke_speed": 700.0,
        "dot_ms": 28,
        "hue_ms": 90,
        "sv_ms": 90,
        "stroke_ms": 18,
        "color_ms": 120,
        "interp_px": 4.0,
    },
    "Fast": {
        "stroke_speed": 1300.0,
        "dot_ms": 20,
        "hue_ms": 60,
        "sv_ms": 60,
        "stroke_ms": 10,
        "color_ms": 80,
        "interp_px": 6.0,
    },
    "Turbo": {
        "stroke_speed": 2200.0,
        "dot_ms": 12,
        "hue_ms": 45,
        "sv_ms": 45,
        "stroke_ms": 5,
        "color_ms": 50,
        "interp_px": 8.0,
    },
}

# Logical-pixel gap each stroke-merging mode may paint across.
MERGE_MODE_GAPS: dict[str, int | None] = {"off": 0, "balanced": 6, "maximum": None}

# Bounds on the pixel font size of a text layer. Layers keep their size as a
# fraction of the logical canvas height and derive pixels within these bounds,
# so a caption keeps its proportions when the resolution changes.
MIN_TEXT_SIZE = 4
MAX_TEXT_SIZE = 256

# Matches the limit the settings schema validates, so a sign that fills up
# says so instead of failing the next save.
MAX_TEXT_LAYERS = 20


@dataclass(slots=True)
class _ProcessResult:
    serial: int
    processed: ProcessedImage
    plan: PaintPlan
    simulation: Image.Image
    stroke_pixel_steps: int
    dot_count: int
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


@dataclass(frozen=True, slots=True)
class _TextOverlayOptions:
    """A small, worker-safe snapshot of the text controls.

    ``font_size`` is in logical canvas pixels, which is what both renderers
    need, but it is derived from ``size_ratio`` — the height of the text as a
    fraction of the canvas. The ratio is what survives a change of painting
    resolution, so text keeps the same size on the finished sign whether it was
    placed under the Very Fast or the Very High preset.
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


class _WorkerSignals(QObject):
    completed = Signal(object)
    failed = Signal(int, str)


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
            font = QFont(layer.font_family)
            font.setPixelSize(max(1, layer.font_size))
            font.setBold(layer.bold)
            font.setItalic(layer.italic)
            painter.setFont(font)
            painter.setPen(QColor(*layer.color))
            text_bounds = painter.fontMetrics().boundingRect(layer.text)
            bounds = QRectF(
                0.0,
                0.0,
                max(1.0, float(text_bounds.width() + 4)),
                max(1.0, float(text_bounds.height() + 4)),
            )
            bounds.moveCenter(QPointF(layer.x * width, layer.y * height))
            painter.drawText(bounds, int(Qt.AlignmentFlag.AlignCenter), layer.text)
    finally:
        painter.end()

    overlay = ImageQt.fromqimage(overlay_qt).convert("RGBA")
    overlay_mask = np.asarray(overlay.getchannel("A"), dtype=np.uint8) > 0
    if not np.any(overlay_mask):
        return processed

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
        self.signals = _WorkerSignals()

    @Slot()
    def run(self) -> None:
        try:
            base_processed = process_image(self.image, self.options)
            processed = _apply_text_overlays(base_processed, self.text_overlays)
            mode = PaintMode(self.paint_mode)
            optimization = None
            if mode is PaintMode.EXACT:
                plan = generate_paint_plan(processed, overpaint_gap=self.overpaint_gap)
                unmerged_stroke_count = (
                    plan.stroke_count
                    if self.overpaint_gap == 0
                    else count_unmerged_strokes(processed)
                )
                simulation_processed = base_processed
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
                if processed is base_processed:
                    simulation_processed = optimized_processed
                else:
                    # Text stays a live vector overlay in the preview, so the
                    # backdrop is simplified without the text baked in. Only
                    # the color simplification runs here - brush planning has
                    # no effect on how the target looks. Merge centers are
                    # derived without the text pixels, so a backdrop shade can
                    # differ very slightly from the painted plan's.
                    backdrop_image, backdrop_mask = simplify_colors(
                        base_processed, mode, options=optimizer_options
                    )
                    simulation_processed = ProcessedImage(
                        backdrop_image, backdrop_mask, base_processed.requested_colors
                    )
                processed = optimized_processed
            # Text stays as live vector items in the editor preview. The paint
            # plan above still uses the composited, palette-limited result.
            simulation = _build_simulation_image(
                simulation_processed, self.color_correction
            )
            stroke_pixel_steps = sum(
                max(0, stroke.pixel_count - 1)
                for group in plan.color_groups
                for stroke in group.strokes
            )
            dot_count = sum(
                stroke.pixel_count == 1
                for group in plan.color_groups
                for stroke in group.strokes
            )
            self.signals.completed.emit(
                _ProcessResult(
                    self.serial,
                    processed,
                    plan,
                    simulation,
                    stroke_pixel_steps,
                    dot_count,
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
        self._plan_stroke_pixel_steps = 0
        self._plan_dot_count = 0
        self._load_serial = 0
        self._load_pool = QThreadPool(self)
        self._load_pool.setMaxThreadCount(1)
        self._process_serial = 0
        self._thread_pool = QThreadPool(self)
        self._thread_pool.setMaxThreadCount(1)
        self._process_timer = QTimer(self)
        self._process_timer.setSingleShot(True)
        self._process_timer.setInterval(180)
        self._process_timer.timeout.connect(self._start_processing)
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
        self._calibration_preview: CalibrationPreviewOverlay | None = None
        self._applying_speed_preset = False
        # Seeded before the resolution controls exist, so this ratio is spelled
        # out against the default 256x128 canvas.
        self._text_layers = [
            _TextOverlayOptions("", "", 24, (255, 255, 255), size_ratio=24 / 128)
        ]
        self._selected_text_layer = 0
        self._syncing_text_controls = False
        self._plan_processing = False
        self._closing = False
        self._painter_bridge = _PainterBridge()
        self._painter_bridge.progress.connect(self._on_paint_progress)
        self._painter_bridge.state.connect(self._on_paint_state)
        self._painter_bridge.completed.connect(self._on_paint_complete)
        self._painter_bridge.error.connect(self._on_paint_error)
        self._painter_bridge.start_requested.connect(self._start_or_resume)
        self._painter_bridge.pause_requested.connect(self._pause_painting)
        self._painter_bridge.abort_requested.connect(self._abort_painting)
        self._painter_bridge.hotkey_error.connect(self._on_hotkey_error)
        self._painter_bridge.debug_finished.connect(self._on_debug_finished)

        self._build_ui()
        self._connect_processing_controls()
        self._install_logging_handler()
        self._initialize_services()
        self._update_quality_dimensions()
        self._update_start_availability()

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
        self.settings_nav_button = QPushButton("Settings")
        self.settings_nav_button.setObjectName("navButton")
        self.settings_nav_button.setCheckable(True)
        self.settings_nav_button.setAutoExclusive(True)
        self._set_icon(self.settings_nav_button, "settings", size=17)
        self.workspace_nav_button.clicked.connect(lambda: self.page_stack.setCurrentIndex(0))
        self.settings_nav_button.clicked.connect(lambda: self.page_stack.setCurrentIndex(1))

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
        self._set_state_badge("idle", "SAFE IDLE")

        layout.addWidget(mark)
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(self.workspace_nav_button)
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
        form.addRow("Background / alpha fill", self.background_combo)
        form.addRow("Custom", self.background_color_button)
        form.addRow("Transparent pixels", self.transparency_combo)
        layout.addWidget(composition)

        quality = QGroupBox("Palette and strokes")
        form = QFormLayout(quality)
        self.color_count_combo = NoWheelComboBox()
        for value in (8, 16, 24, 32, 48, 64, 96, 128, 256):
            self.color_count_combo.addItem(str(value), value)
        self.color_count_combo.setCurrentText("32")
        self.merge_combo = NoWheelComboBox()
        self.merge_combo.addItem("Off — exact strokes", "off")
        self.merge_combo.addItem("Balanced — small gaps", "balanced")
        self.merge_combo.addItem("Maximum — longest strokes", "maximum")
        self._set_combo_data(self.merge_combo, "balanced")
        self.merge_combo.setToolTip(
            "Lets early colors paint straight through pixels that later colors\n"
            "repaint anyway. The finished image is identical, but fragmented\n"
            "areas need far fewer strokes, so painting is much faster."
        )
        form.addRow("Maximum colors", self.color_count_combo)
        form.addRow("Stroke merging", self.merge_combo)
        layout.addWidget(quality)

        layout.addStretch(1)
        return content

    def _build_paint_settings(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(12)

        title = QLabel("Painting")
        title.setObjectName("pageTitle")
        note = QLabel("Tune brush behavior and input timing when a preset needs adjustment.")
        note.setObjectName("muted")
        layout.addWidget(title)
        layout.addWidget(note)

        speed_group = QGroupBox("Speed preset")
        speed_form = QFormLayout(speed_group)
        self.speed_preset_combo = NoWheelComboBox()
        self.speed_preset_combo.addItems([*SPEED_PRESETS.keys(), "Custom"])
        self.speed_preset_combo.setCurrentText("Standard")
        self.speed_preset_combo.setToolTip(
            "One-click timing profiles. Faster presets assume Rust keeps up with\n"
            "quick strokes; if rows bleed or strokes get gaps, step back down.\n"
            "Editing any value under Advanced Timing switches this to Custom."
        )
        speed_form.addRow("Preset", self.speed_preset_combo)
        speed_note = QLabel(
            "Turbo can outrun Rust's UI on slower machines — test it with a "
            "small image first."
        )
        speed_note.setWordWrap(True)
        speed_note.setObjectName("muted")
        speed_form.addRow("", speed_note)
        layout.addWidget(speed_group)

        advanced = QGroupBox("Advanced timing")
        advanced_layout = QFormLayout(advanced)
        self.pixel_spacing_spin = self._double_spin(0.25, 3.0, 1.0, 0.05, " ×")
        self.stroke_speed_spin = self._double_spin(10.0, 5000.0, 700.0, 10.0, " px/s")
        self.dot_duration_spin = self._int_spin(1, 1000, 28, " ms")
        self.hue_delay_spin = self._int_spin(0, 3000, 90, " ms")
        self.sv_delay_spin = self._int_spin(0, 3000, 90, " ms")
        self.brush_delay_spin = self._int_spin(0, 3000, 60, " ms")
        self.stroke_delay_spin = self._int_spin(0, 3000, 18, " ms")
        self.color_delay_spin = self._int_spin(0, 5000, 120, " ms")
        self.interpolation_spin = self._double_spin(1.0, 100.0, 4.0, 1.0, " px")
        advanced_layout.addRow("Logical spacing", self.pixel_spacing_spin)
        advanced_layout.addRow("Stroke speed", self.stroke_speed_spin)
        advanced_layout.addRow("Dot hold", self.dot_duration_spin)
        advanced_layout.addRow("After hue", self.hue_delay_spin)
        advanced_layout.addRow("After S/V", self.sv_delay_spin)
        advanced_layout.addRow("After brush", self.brush_delay_spin)
        advanced_layout.addRow("Between strokes", self.stroke_delay_spin)
        advanced_layout.addRow("Between colors", self.color_delay_spin)
        advanced_layout.addRow("Interpolation step", self.interpolation_spin)
        layout.addWidget(advanced)

        layout.addStretch(1)
        return content

    def _build_preview_area(self) -> QWidget:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(10, 4, 4, 4)
        layout.setSpacing(12)

        heading = QHBoxLayout()
        title = QLabel("PREVIEW")
        title.setObjectName("pageTitle")
        hint = QLabel("Click the preview or drop an image anywhere to open it")
        hint.setObjectName("muted")
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(hint)
        layout.addLayout(heading)

        tabs = QTabWidget()
        browse_hint = "Click here or drop an image"
        self.original_preview = PreviewLabel(
            "Browse an image to begin", hint=browse_hint
        )
        self.paint_preview = TextEditorPreview(
            "Paint simulation will appear here", hint=browse_hint
        )
        self.paint_preview.setToolTip(
            "Drag text to move it, drag its handles to resize it, double-click "
            "to edit it, press Ctrl+D or Ctrl+C to copy it, or press Delete to "
            "remove it."
        )
        for preview in (self.original_preview, self.paint_preview):
            preview.browseRequested.connect(self._browse_image)
            preview.imageDropped.connect(
                lambda dropped: self.load_image(Path(dropped))
            )
        tabs.addTab(self.original_preview, "Source")
        tabs.addTab(self.paint_preview, "Rust preview")
        self.preview_tabs = tabs
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
        status_glyph = QLabel()
        status_glyph.setFixedSize(20, 20)
        status_glyph.setScaledContents(True)
        status_glyph.setPixmap(art_pixmap("status", 40))
        progress_head.addWidget(status_glyph)
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

    def _set_active_progress_visible(self, active: bool) -> None:
        """Swap the plan panel for the enlarged progress readout while painting."""

        self.plan_progress_stack.setCurrentIndex(1 if active else 0)
        self.progress_frame.setVisible(not active)

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
        self.quality_combo = NoWheelComboBox()
        self.quality_combo.addItems([*QUALITY_LONG_EDGE.keys(), "Custom"])
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
        for combo in (
            self.scale_mode_combo,
            self.quality_combo,
            self.crop_alignment_combo,
            self.paint_mode_combo,
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
        for column, checkbox in enumerate(
            (self.remove_background_check, self.dither_check)
        ):
            quick_grid.addWidget(
                checkbox,
                4,
                column,
                alignment=Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter,
            )

        self.custom_resolution_panel = QFrame()
        self.custom_resolution_panel.setObjectName("inlinePanel")
        custom_layout = QHBoxLayout(self.custom_resolution_panel)
        custom_layout.setContentsMargins(10, 7, 10, 7)
        custom_layout.setSpacing(8)
        custom_layout.addWidget(QLabel("Custom resolution"))
        self.logical_width_spin = NoWheelSpinBox()
        self.logical_width_spin.setRange(8, 2048)
        self.logical_width_spin.setValue(256)
        self.logical_width_spin.setToolTip(
            "Logical width. Height follows the calibrated canvas aspect ratio."
        )
        self.logical_height_spin = NoWheelSpinBox()
        self.logical_height_spin.setRange(8, 2048)
        self.logical_height_spin.setValue(128)
        self.logical_height_spin.setToolTip(
            "Logical height. Width follows the calibrated canvas aspect ratio."
        )
        custom_layout.addStretch(1)
        custom_layout.addWidget(self.logical_width_spin)
        custom_layout.addWidget(QLabel("×"))
        custom_layout.addWidget(self.logical_height_spin)
        quick_grid.addWidget(self.custom_resolution_panel, 5, 0, 1, 2)

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
            "Touching the edges", BackgroundRemovalScope.CONNECTED.value
        )
        self.removal_scope_combo.addItem(
            "Anywhere in the image", BackgroundRemovalScope.EVERYWHERE.value
        )
        self.removal_scope_combo.setToolTip(
            "Edge matching keeps enclosed areas — the hole in an O, a white\n"
            "eye — painted. Anywhere also skips every matching inner pocket."
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
        quick_grid.addWidget(self.background_removal_panel, 6, 0, 1, 2)
        quick_grid.setColumnStretch(0, 1)
        quick_grid.setColumnStretch(1, 1)
        image_layout.addLayout(quick_grid)

        text_heading = QHBoxLayout()
        text_title = QLabel("Text overlay")
        text_title.setObjectName("sectionTitle")
        self.add_text_button = QPushButton("Add text")
        self.add_text_button.setObjectName("compactButton")
        self.duplicate_text_button = QPushButton("Duplicate")
        self.duplicate_text_button.setObjectName("compactButton")
        self.duplicate_text_button.setToolTip(
            "Copy the selected text layer. Ctrl+D or Ctrl+C does the same to\n"
            "the layer selected in the Rust preview."
        )
        self.remove_text_button = QPushButton("Remove")
        self.remove_text_button.setObjectName("compactButton")
        text_heading.addWidget(text_title)
        text_heading.addStretch(1)
        text_heading.addWidget(self.add_text_button)
        text_heading.addWidget(self.duplicate_text_button)
        text_heading.addWidget(self.remove_text_button)
        image_layout.addLayout(text_heading)

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
        text_grid.setColumnStretch(1, 1)
        text_grid.setColumnStretch(3, 1)
        image_layout.addWidget(self.text_options_panel)
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
        self.brush_slider_status = CalibrationStatus("Size track", optional=True)
        self.brush_preview_status = CalibrationStatus("Brush preview", optional=True)
        self.calibrate_canvas_button = QPushButton("Set")
        self.calibrate_color_box_button = QPushButton("Set")
        self.calibrate_hue_bar_button = QPushButton("Set")
        self.calibrate_brush_button = QPushButton("Set")
        self.calibrate_brush_preview_button = QPushButton("Set")
        entries = (
            (self.canvas_status, self.calibrate_canvas_button, "Calibrate canvas"),
            (self.color_box_status, self.calibrate_color_box_button, "Calibrate color box"),
            (self.hue_bar_status, self.calibrate_hue_bar_button, "Calibrate hue bar"),
            (self.brush_slider_status, self.calibrate_brush_button, "Calibrate size track"),
            (
                self.brush_preview_status,
                self.calibrate_brush_preview_button,
                "Calibrate brush preview",
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
            "Sizes the brush to a logical image cell before painting. Needs the "
            "Size track and the Brush preview tile calibrated."
        )
        self.show_calibration_check = QCheckBox("Show boxes on screen")
        self.show_calibration_check.setToolTip(
            "Draws labeled red outlines over every calibrated region so you can\n"
            "confirm they still line up with Rust's painting UI. The outlines\n"
            "are click-through and hide automatically while painting."
        )
        profile_layout.addWidget(self.apply_brush_check)
        profile_layout.addWidget(self.show_calibration_check)
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
        self.start_button = QPushButton("START PAINTING  •  F8")
        self.start_button.setObjectName("accent")
        self.start_button.setMinimumHeight(44)
        self._set_icon(self.start_button, "play", ON_ACCENT, size=22)
        run_buttons = QHBoxLayout()
        self.pause_button = QPushButton("Pause  •  F9")
        self.abort_button = QPushButton("Abort  •  F10")
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
        self.corner_abort_check = QCheckBox("Rapid move to a screen corner aborts")
        self.corner_abort_check.setChecked(True)
        self.mouse_pause_check = QCheckBox("Moving the mouse pauses instead of painting over it")
        self.mouse_pause_check.setChecked(True)
        self.mouse_pause_check.setToolTip(
            "Painting stops the moment you take the mouse back and releases the "
            "button, then continues from the same stroke when you resume, so an "
            "accidental nudge never costs the whole run."
        )
        self.verify_ui_check = QCheckBox("Compare calibration reference before start")
        self.start_hotkey_combo = self._hotkey_combo("F8")
        self.pause_hotkey_combo = self._hotkey_combo("F9")
        self.abort_hotkey_combo = self._hotkey_combo("F10")
        safety_form.addRow("Countdown", self.countdown_spin)
        safety_form.addRow("Focus guard", self.focus_guard_check)
        safety_form.addRow("Expected window", self.expected_window_edit)
        safety_form.addRow("Expected process", self.expected_process_edit)
        safety_form.addRow("Corner stop", self.corner_abort_check)
        safety_form.addRow("Mouse guard", self.mouse_pause_check)
        safety_form.addRow("UI check", self.verify_ui_check)
        safety_form.addRow("Start / resume", self.start_hotkey_combo)
        safety_form.addRow("Pause", self.pause_hotkey_combo)
        safety_form.addRow("Abort", self.abort_hotkey_combo)
        layout.addWidget(safety_group)
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
        glyph = QLabel()
        glyph.setFixedSize(22, 22)
        glyph.setScaledContents(True)
        glyph.setPixmap(art_pixmap(icon_name, 44))
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
        combo.addItems([f"F{number}" for number in range(5, 13)])
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
        self.crop_alignment_combo.currentIndexChanged.connect(self._schedule_processing)
        self.background_combo.currentIndexChanged.connect(self._on_background_changed)
        self.background_color_button.colorChanged.connect(self._schedule_processing)
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
        self.merge_combo.currentIndexChanged.connect(self._schedule_processing)
        self.paint_mode_combo.currentIndexChanged.connect(self._on_paint_mode_changed)
        self.add_text_button.clicked.connect(self._add_text_layer)
        self.duplicate_text_button.clicked.connect(self._duplicate_selected_text_layer)
        self.remove_text_button.clicked.connect(self._remove_text_layer)
        self.text_layer_combo.currentIndexChanged.connect(self._select_text_layer)
        self.text_edit.textChanged.connect(self._on_text_control_changed)
        self.text_font_combo.currentFontChanged.connect(self._on_text_control_changed)
        self.text_size_spin.valueChanged.connect(self._on_text_control_changed)
        self.text_color_button.colorChanged.connect(self._on_text_control_changed)
        self.text_bold_check.toggled.connect(self._on_text_control_changed)
        self.text_italic_check.toggled.connect(self._on_text_control_changed)
        self.paint_preview.layerMoved.connect(self._on_text_layer_moved)
        self.paint_preview.layerSelected.connect(self._select_text_layer)
        self.paint_preview.layerTextEdited.connect(self._on_canvas_text_edited)
        self.paint_preview.layerResized.connect(self._on_canvas_text_resized)
        self.paint_preview.layerDeleteRequested.connect(self._delete_text_layer)
        self.paint_preview.layerDuplicateRequested.connect(self._duplicate_text_layer)
        self.paint_preview.interactionFinished.connect(
            self._on_text_interaction_finished
        )
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

    def _current_overpaint_gap(self) -> int | None:
        return MERGE_MODE_GAPS.get(str(self.merge_combo.currentData()), 6)

    def _current_paint_mode(self) -> str:
        # A calibration chart measures the raw material response, so it is
        # always planned exactly, whatever the user's normal mode is.
        if self._painting_calibration_chart():
            return PaintMode.EXACT.value
        return str(self.paint_mode_combo.currentData() or PaintMode.BALANCED.value)

    def _brush_capabilities(self) -> BrushCapabilities:
        """What the optimizer may plan with, given the current calibration."""

        slider = self._profile_rect("brush_slider")
        preview = self._profile_rect("brush_preview")
        canvas = self._profile_rect("canvas")
        cell_pixels = 0.0
        if canvas is not None:
            cell_pixels = min(
                canvas.width / max(1, self.logical_width_spin.value()),
                canvas.height / max(1, self.logical_height_spin.value()),
            )
        return BrushCapabilities(
            sizing=bool(
                self.apply_brush_check.isChecked()
                and slider is not None
                and preview is not None
                # Spacing above 1.0 spreads stroke geometry while the brush
                # stays capped at one unspaced cell, so multi-cell bands would
                # leave unpainted rows; keep those plans single-cell.
                and self.pixel_spacing_spin.value() <= 1.0
            ),
            cell_pixels=cell_pixels,
        )

    @Slot()
    def _on_paint_mode_changed(self, *_args: Any) -> None:
        self._sync_paint_mode_dependent_controls()
        self._schedule_processing()

    def _sync_paint_mode_dependent_controls(self) -> None:
        """Stroke merging is superseded by the optimizer outside Exact mode."""

        exact = self.paint_mode_combo.currentData() == PaintMode.EXACT.value
        self.merge_combo.setEnabled(exact)

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

    def _sync_text_controls(self) -> None:
        if not self._text_layers:
            return
        layer = self._text_layers[self._selected_text_layer]
        self._syncing_text_controls = True
        try:
            self.text_edit.setText(layer.text)
            if layer.font_family:
                self.text_font_combo.setCurrentFont(QFont(layer.font_family))
            self.text_size_spin.setValue(layer.font_size)
            self.text_color_button.set_color(QColor(*layer.color))
            self.text_bold_check.setChecked(layer.bold)
            self.text_italic_check.setChecked(layer.italic)
        finally:
            self._syncing_text_controls = False

    @Slot()
    def _add_text_layer(self) -> None:
        if len(self._text_layers) >= MAX_TEXT_LAYERS:
            self.statusBar().showMessage(
                f"A sign can hold at most {MAX_TEXT_LAYERS} text layers", 4000
            )
            return
        color = self.text_color_button.color()
        offset = min(0.24, len(self._text_layers) * 0.06)
        font_size = self.text_size_spin.value()
        self._text_layers.append(
            _TextOverlayOptions(
                "",
                self.text_font_combo.currentFont().family(),
                font_size,
                (color.red(), color.green(), color.blue()),
                x=min(0.85, 0.5 + offset),
                y=min(0.85, 0.5 + offset),
                bold=self.text_bold_check.isChecked(),
                italic=self.text_italic_check.isChecked(),
                size_ratio=self._text_size_ratio(font_size),
            )
        )
        self._selected_text_layer = len(self._text_layers) - 1
        self._rebuild_text_layer_combo()
        self._sync_text_controls()
        self._refresh_text_editor_layers()
        self.text_edit.setFocus()
        self._schedule_settings_save()

    @Slot()
    def _duplicate_selected_text_layer(self) -> None:
        self._duplicate_text_layer(self._selected_text_layer)

    @Slot(int)
    def _duplicate_text_layer(self, index: int) -> None:
        """Insert a copy of one text layer, nudged clear of the original."""

        if not 0 <= index < len(self._text_layers):
            return
        if len(self._text_layers) >= MAX_TEXT_LAYERS:
            self.statusBar().showMessage(
                f"A sign can hold at most {MAX_TEXT_LAYERS} text layers", 4000
            )
            return
        source = self._text_layers[index]
        offset = self._text_size_ratio(source.font_size) * 0.5
        copy = replace(
            source,
            x=min(max(source.x + offset * 0.5, 0.0), 1.0),
            y=min(max(source.y + offset, 0.0), 1.0),
        )
        self._text_layers.insert(index + 1, copy)
        self._selected_text_layer = index + 1
        self._rebuild_text_layer_combo()
        self._sync_text_controls()
        self._refresh_text_editor_layers()
        self._schedule_processing()
        self._schedule_settings_save()

    @Slot()
    def _remove_text_layer(self) -> None:
        self._delete_text_layer(self._selected_text_layer)

    @Slot(int)
    def _delete_text_layer(self, index: int) -> None:
        """Drop one text layer, keeping a single empty layer to type into."""

        if not 0 <= index < len(self._text_layers):
            return
        if len(self._text_layers) > 1:
            self._text_layers.pop(index)
            selected = self._selected_text_layer
            if index < selected:
                selected -= 1
            self._selected_text_layer = min(max(selected, 0), len(self._text_layers) - 1)
        else:
            self._text_layers[0] = replace(self._text_layers[0], text="")
            self._selected_text_layer = 0
        self._rebuild_text_layer_combo()
        self._sync_text_controls()
        self._refresh_text_editor_layers()
        self._schedule_processing()
        self._schedule_settings_save()

    @Slot(int)
    def _select_text_layer(self, index: int) -> None:
        if not 0 <= index < len(self._text_layers):
            return
        self._selected_text_layer = index
        if self.text_layer_combo.currentIndex() != index:
            self.text_layer_combo.blockSignals(True)
            self.text_layer_combo.setCurrentIndex(index)
            self.text_layer_combo.blockSignals(False)
        self._sync_text_controls()
        self.paint_preview.select_layer(index)

    def _on_text_control_changed(self, *_args: Any) -> None:
        if self._syncing_text_controls or not self._text_layers:
            return
        current = self._text_layers[self._selected_text_layer]
        color = self.text_color_button.color()
        font_size = self.text_size_spin.value()
        self._text_layers[self._selected_text_layer] = replace(
            current,
            text=self.text_edit.text(),
            font_family=self.text_font_combo.currentFont().family(),
            font_size=font_size,
            color=(color.red(), color.green(), color.blue()),
            bold=self.text_bold_check.isChecked(),
            italic=self.text_italic_check.isChecked(),
            size_ratio=self._text_size_ratio(font_size),
        )
        self._rebuild_text_layer_combo()
        self._refresh_text_editor_layers()
        self._schedule_processing()
        self._schedule_settings_save()

    @Slot(int, float, float)
    def _on_text_layer_moved(self, index: int, x: float, y: float) -> None:
        if not 0 <= index < len(self._text_layers):
            return
        self._text_layers[index] = replace(
            self._text_layers[index],
            x=min(max(x, 0.0), 1.0),
            y=min(max(y, 0.0), 1.0),
        )
        self._selected_text_layer = index
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
        self._schedule_processing()
        self._schedule_settings_save()

    @Slot()
    def _on_text_interaction_finished(self) -> None:
        QTimer.singleShot(0, self._refresh_text_editor_layers)

    def _refresh_text_editor_layers(self) -> None:
        self.paint_preview.set_layers(
            self._text_layers,
            self._selected_text_layer,
        )

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
        current = self._speed_preset_values()
        for name, values in SPEED_PRESETS.items():
            if all(
                math.isclose(float(current[key]), float(expected), abs_tol=0.01)
                for key, expected in values.items()
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
        self._thread_pool.clear()
        self._original_image = None
        self._image_path = None
        self._processed = None
        self._plan = None
        self._plan_metric_source = None
        self._plan_stroke_pixel_steps = 0
        self._plan_dot_count = 0
        self.original_preview.clear_source("Decoding image…")
        self.paint_preview.clear_source("Waiting for the new image")
        self.image_name_label.setText(path.name)
        self.image_dimensions_label.setText("Loading…")
        self.processing_label.setText("Decoding image…")
        self._set_plan_processing(True)
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
        self.original_preview.set_source(self._pil_to_pixmap(result.preview))
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
        self._set_plan_processing(False)
        self._refresh_statistics()
        self._update_start_availability()
        QMessageBox.critical(self, "Could not load image", message)

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
        self._rescale_text_layers()
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
        width = max(8, min(2048, width))
        height = max(8, min(2048, height))
        self.logical_width_spin.blockSignals(True)
        self.logical_height_spin.blockSignals(True)
        self.logical_width_spin.setValue(width)
        self.logical_height_spin.setValue(height)
        self.logical_width_spin.blockSignals(False)
        self.logical_height_spin.blockSignals(False)

    @Slot()
    def _update_quality_dimensions(self) -> None:
        preset = self.quality_combo.currentText()
        custom = preset == "Custom"
        self.custom_resolution_panel.setVisible(custom)
        self.logical_width_spin.setEnabled(custom)
        self.logical_height_spin.setEnabled(custom)
        if not custom:
            aspect = self._canvas_aspect_ratio()
            longest = QUALITY_LONG_EDGE[preset]
            if aspect >= 1.0:
                width, height = longest, max(8, round(longest / aspect))
            else:
                width, height = max(8, round(longest * aspect)), longest
            self.logical_width_spin.blockSignals(True)
            self.logical_height_spin.blockSignals(True)
            self.logical_width_spin.setValue(width)
            self.logical_height_spin.setValue(height)
            self.logical_width_spin.blockSignals(False)
            self.logical_height_spin.blockSignals(False)
        else:
            self._sync_custom_resolution("width")
        self._rescale_text_layers()
        self._schedule_processing()

    @Slot()
    def _schedule_processing(self, *_args: Any) -> None:
        if self._original_image is None:
            return
        self._process_serial += 1
        self._plan = None
        self._processed = None
        self._plan_metric_source = None
        self._plan_stroke_pixel_steps = 0
        self._plan_dot_count = 0
        self._process_timer.start()
        self.processing_label.setText("Updating paint simulation…")
        self._set_plan_processing(True)
        self._refresh_statistics()
        self._update_start_availability()

    def _set_plan_processing(self, processing: bool) -> None:
        """Show or hide the visible signs that the plan is being recalculated."""

        self._plan_processing = processing
        if processing:
            self.processing_spinner.start()
        else:
            self.processing_spinner.stop()

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
            crop_alignment=CropAlignment(self.crop_alignment_combo.currentData()),
            color_count=int(self.color_count_combo.currentData()),
            dither=self.dither_check.isChecked(),
            background_color=background,
            transparency_mode=transparency,
            transparent_fill_color=transparent_fill,
            alpha_threshold=0,
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
        serial = self._process_serial
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
        )
        worker.signals.completed.connect(self._on_processing_complete)
        worker.signals.failed.connect(self._on_processing_failed)
        self._thread_pool.start(worker)

    @Slot(object)
    def _on_processing_complete(self, result: _ProcessResult) -> None:
        if result.serial != self._process_serial or self._closing:
            return
        self._set_plan_processing(False)
        self._processed = result.processed
        self._plan = result.plan
        self._plan_metric_source = result.plan
        self._plan_stroke_pixel_steps = result.stroke_pixel_steps
        self._plan_dot_count = result.dot_count
        self.paint_preview.set_source(self._pil_to_pixmap(result.simulation))
        if not self.paint_preview.is_interacting:
            self._refresh_text_editor_layers()
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
                "be painted" + merge_note
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
        if serial != self._process_serial or self._closing:
            return
        self._set_plan_processing(False)
        self.processing_label.setText(f"Could not process image: {message}")
        self._plan = None
        self._processed = None
        self._plan_metric_source = None
        self._refresh_statistics()
        self._update_start_availability()

    def _refresh_statistics(self, *_args: Any) -> None:
        plan = self._plan
        if plan is None:
            # While a recalculation is in flight the metrics read as pending
            # rather than absent, so the numbers do not just vanish.
            placeholder = "…" if self._plan_processing else "—"
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
        seconds = self._estimate_seconds(plan)
        self.analysis_time.value_label.setText(self._format_duration(seconds))  # type: ignore[attr-defined]

    def _estimate_seconds(self, plan: PaintPlan) -> float:
        canvas = self._profile_rect("canvas")
        cell_width = canvas.width / plan.width if canvas else 1.0
        if plan is self._plan_metric_source:
            stroke_pixel_steps = self._plan_stroke_pixel_steps
            dot_count = self._plan_dot_count
        else:
            stroke_pixel_steps = sum(
                max(0, stroke.pixel_count - 1)
                for group in plan.color_groups
                for stroke in group.strokes
            )
            dot_count = sum(
                stroke.pixel_count == 1
                for group in plan.color_groups
                for stroke in group.strokes
            )
        travel = stroke_pixel_steps * cell_width
        # One walk over the groups tracks everything the painter tracks: the
        # picker is selected once per run of same-color groups, and the slider
        # cache is keyed by diameter.
        sizing = (
            self.apply_brush_check.isChecked()
            and self._profile_rect("brush_slider") is not None
            and self._profile_rect("brush_preview") is not None
        )
        selections = 0
        previous_color: tuple[int, int, int] | None = None
        searched: set[int] = set()
        previous_diameter: int | None = None
        revisits = 0
        for group in plan.color_groups:
            if group.color != previous_color:
                selections += 1
                previous_color = group.color
            if sizing:
                diameter = max(1, group.brush_diameter)
                if diameter != previous_diameter:
                    if diameter in searched:
                        revisits += 1
                    else:
                        searched.add(diameter)
                    previous_diameter = diameter
        # A fresh diameter binary-searches the slider with ~7 preview
        # measurements, each of which also selects a temporary color; a
        # revisit replays one remembered click.
        settle = max(self.brush_delay_spin.value() / 1000.0, 0.16)
        click = self.dot_duration_spin.value() / 1000.0
        brush_seconds = len(searched) * 7 * (settle + click) + revisits * (
            settle + click
        )
        selections += len(searched)
        color_ms = selections * (
            self.hue_delay_spin.value()
            + self.sv_delay_spin.value()
            + 2 * self.dot_duration_spin.value()
        ) + len(plan.color_groups) * self.color_delay_spin.value()
        stroke_ms = plan.stroke_count * self.stroke_delay_spin.value()
        dot_ms = dot_count * self.dot_duration_spin.value()
        movement_seconds = travel / max(1.0, self.stroke_speed_spin.value())
        return (
            movement_seconds
            + (color_ms + stroke_ms + dot_ms) / 1000.0
            + brush_seconds
        )

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
        if sys.platform == "darwin":
            root = Path.home() / "Library" / "Application Support" / "RustPainter"
        else:
            local = os.environ.get("LOCALAPPDATA")
            root = Path(local) / "RustPainter" if local else Path.cwd() / "data"
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _initialize_services(self) -> None:
        data = self._local_data_directory()
        self._profile_store = ProfileStore(data / "profiles")
        self._settings_store = SettingsStore(data / "settings.json")
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
            lambda: self._begin_calibration("brush_slider", "clickable Size track")
        )
        self.calibrate_brush_preview_button.clicked.connect(
            lambda: self._begin_calibration(
                "brush_preview", "gray brush-preview tile"
            )
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
        self.move_to_rust_button.clicked.connect(self._move_calibration_to_rust_monitor)

        settings_controls = (
            self.scale_mode_combo,
            self.crop_alignment_combo,
            self.background_combo,
            self.transparency_combo,
            self.quality_combo,
            self.paint_mode_combo,
            self.logical_width_spin,
            self.logical_height_spin,
            self.color_count_combo,
            self.dither_check,
            self.merge_combo,
            self.show_calibration_check,
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
            self.apply_brush_check,
            self.countdown_spin,
            self.dry_run_check,
            self.focus_guard_check,
            self.expected_window_edit,
            self.expected_process_edit,
            self.corner_abort_check,
            self.mouse_pause_check,
            self.verify_ui_check,
            self.start_hotkey_combo,
            self.pause_hotkey_combo,
            self.abort_hotkey_combo,
        )
        for control in settings_controls:
            if isinstance(control, QComboBox):
                control.currentIndexChanged.connect(self._schedule_settings_save)
            elif isinstance(control, (QSpinBox, QDoubleSpinBox)):
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
            self._set_combo_data(
                self.crop_alignment_combo, image.get("crop_alignment", "center")
            )
            preset = str(image.get("quality_preset", "balanced")).replace("_", " ").title()
            if self.quality_combo.findText(preset) >= 0:
                self.quality_combo.setCurrentText(preset)
            self._set_combo_data(
                self.paint_mode_combo, str(image.get("paint_mode", "balanced"))
            )
            self.logical_width_spin.setValue(int(image.get("logical_width", 256)))
            self.logical_height_spin.setValue(int(image.get("logical_height", 128)))
            self._set_combo_data(self.color_count_combo, int(image.get("color_count", 32)))
            self.dither_check.setChecked(bool(image.get("dithering", False)))
            self._set_combo_data(
                self.background_combo, image.get("background_mode", "unpainted")
            )
            self.background_color_button.set_color(image.get("background_color", "#ffffff"))
            self._set_combo_data(
                self.transparency_combo,
                image.get("transparent_pixels", "leave_unpainted"),
            )
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
                image.get("background_removal_scope", "connected"),
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
                self._text_layers.append(
                    _TextOverlayOptions(
                        text=str(layer_value.get("text", "")),
                        font_family=str(layer_value.get("font_family", "")),
                        font_size=font_size,
                        color=(color.red(), color.green(), color.blue()),
                        x=float(layer_value.get("x", 0.5)),
                        y=float(layer_value.get("y", 0.5)),
                        bold=bool(layer_value.get("bold", False)),
                        italic=bool(layer_value.get("italic", False)),
                        size_ratio=float(layer_value.get("size_ratio", 0.0))
                        or self._text_size_ratio(font_size),
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
            self._rebuild_text_layer_combo()
            self._sync_text_controls()

            self.pixel_spacing_spin.setValue(
                float(painting.get("logical_pixel_spacing", 1.0))
            )
            self.stroke_speed_spin.setValue(
                float(painting.get("stroke_speed_pixels_per_second", 700.0))
            )
            self.dot_duration_spin.setValue(
                round(float(painting.get("mouse_down_duration_seconds", 0.028)) * 1000)
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
                round(float(painting.get("delay_after_brush_seconds", 0.06)) * 1000)
            )
            self.stroke_delay_spin.setValue(
                round(float(painting.get("delay_between_strokes_seconds", 0.018)) * 1000)
            )
            self.color_delay_spin.setValue(
                round(float(painting.get("delay_between_colors_seconds", 0.12)) * 1000)
            )
            self.interpolation_spin.setValue(
                float(painting.get("stroke_interpolation_step_pixels", 4.0))
            )
            self.apply_brush_check.setChecked(bool(painting.get("apply_brush_size", False)))
            merge_index = self.merge_combo.findData(
                str(painting.get("stroke_merge_mode", "balanced"))
            )
            self.merge_combo.setCurrentIndex(
                merge_index if merge_index >= 0 else self.merge_combo.findData("balanced")
            )
            self.speed_preset_combo.setCurrentText(self._detect_speed_preset())
            ui = settings.get("ui", {})
            self.show_calibration_check.setChecked(
                bool(ui.get("show_calibration_overlay", False))
            )

            self.countdown_spin.setValue(int(safety.get("countdown_seconds", 3)))
            self.corner_abort_check.setChecked(bool(safety.get("corner_abort_enabled", True)))
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
        self._refresh_text_editor_layers()

    def _settings_document(self) -> dict[str, Any]:
        current = self._settings.copy()
        current["image"] = {
            **current.get("image", {}),
            "scale_mode": self.scale_mode_combo.currentData(),
            "crop_alignment": self.crop_alignment_combo.currentData(),
            "quality_preset": self.quality_combo.currentText().lower().replace(" ", "_"),
            "paint_mode": str(self.paint_mode_combo.currentData() or "balanced"),
            "logical_width": self.logical_width_spin.value(),
            "logical_height": self.logical_height_spin.value(),
            "color_count": int(self.color_count_combo.currentData()),
            "dithering": self.dither_check.isChecked(),
            "background_mode": self.background_combo.currentData(),
            "background_color": self.background_color_button.color().name().upper(),
            "transparent_pixels": self.transparency_combo.currentData(),
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
            "brush_direction": "low_to_high",
            "stroke_merge_mode": str(self.merge_combo.currentData() or "balanced"),
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
            "corner_abort_enabled": self.corner_abort_check.isChecked(),
            "pause_on_mouse_move": self.mouse_pause_check.isChecked(),
            "require_rust_foreground": self.focus_guard_check.isChecked(),
            "expected_window_title_contains": self.expected_window_edit.text().strip(),
            "expected_process_name": self.expected_process_edit.text().strip(),
            "verify_calibrated_ui": self.verify_ui_check.isChecked(),
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
        brush_optional = not self.apply_brush_check.isChecked()
        self.brush_slider_status.set_calibrated(
            bool(status.get("brush_slider")), brush_optional
        )
        self.brush_preview_status.set_calibrated(
            bool(status.get("brush_preview")), brush_optional
        )
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
        self._refresh_display_warning()
        self._update_start_availability()

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
                    "brush_slider",
                    "brush_preview",
                ):
                    setattr(candidate, field, getattr(source, field, None))
                candidate.display = source.display
                if (
                    isinstance(source.metadata, dict)
                    and "color_correction" in source.metadata
                ):
                    candidate.metadata["color_correction"] = deepcopy(
                        source.metadata["color_correction"]
                    )
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
                "Abort or wait for the current operation before changing calibration.",
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
                "brush_slider",
                "brush_preview",
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
        elif field in {"brush_slider", "brush_preview"}:
            # These calibrations change what the optimizer may plan with.
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
        for name in ("canvas", "color_box", "hue_bar", "brush_slider", "brush_preview"):
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
                    ("Size track", getattr(profile, "brush_slider", None)),
                    ("Brush preview", getattr(profile, "brush_preview", None)),
                )
                if rect is not None
            ]
        busy = (
            self._painter_is_active()
            or self._debug_running
            or self._countdown_callback_running
            or bool(self._countdown and self._countdown.isVisible())
        )
        show = (
            not self._closing
            and self.show_calibration_check.isChecked()
            and bool(entries)
            and not busy
        )
        if not show:
            if self._calibration_preview is not None and self._calibration_preview.isVisible():
                self._calibration_preview.hide()
            return
        try:
            if self._calibration_preview is None:
                self._calibration_preview = CalibrationPreviewOverlay()
            self._calibration_preview.set_rectangles(entries)
            if not self._calibration_preview.isVisible():
                self._calibration_preview.show_overlay()
        except Exception:
            LOGGER.exception("Could not display the calibration overlay")

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
                "Pause or abort the active operation before capturing a reference.",
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
                "Finish or abort the current operation before preparing a color chart.",
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
            self._on_hotkey_error("Start, pause, and abort hotkeys must be different.")
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
            f"Abort  •  {self.abort_hotkey_combo.currentText()}"
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
        paused = False
        if self._painter is not None:
            try:
                from app.painter import PainterState

                paused = self._painter.state == PainterState.PAUSED
            except Exception:
                paused = False
        profile_ready = bool(self._current_profile and self._current_profile.is_ready)
        can_dry_run = self.dry_run_check.isChecked() and self._plan is not None
        can_start = (self._plan is not None and profile_ready) or can_dry_run or paused
        if (
            not self.dry_run_check.isChecked()
            and not self._emergency_hotkey_available()
            and not paused
        ):
            can_start = False
        self.start_button.setEnabled(
            can_start and not countdown_active and (not active or paused)
        )
        self.start_button.setText(
            f"RESUME PAINTING  •  {self.start_hotkey_combo.currentText()}"
            if paused
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
        self._set_job_controls_locked(job_locked)
        self._update_calibration_overlay()

    def _set_job_controls_locked(self, locked: bool) -> None:
        controls = (
            self.browse_button,
            self.scale_mode_combo,
            self.crop_alignment_combo,
            self.background_combo,
            self.background_color_button,
            self.transparency_combo,
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
            self.apply_brush_check,
            self.profile_combo,
            self.new_profile_button,
            self.rename_profile_button,
            self.delete_profile_button,
            self.calibrate_canvas_button,
            self.calibrate_color_box_button,
            self.calibrate_hue_bar_button,
            self.calibrate_brush_button,
            self.calibrate_brush_preview_button,
            self.countdown_spin,
            self.dry_run_check,
            self.focus_guard_check,
            self.expected_window_edit,
            self.expected_process_edit,
            self.corner_abort_check,
            self.mouse_pause_check,
            self.verify_ui_check,
            self.start_hotkey_combo,
            self.pause_hotkey_combo,
            self.abort_hotkey_combo,
            self.capture_reference_button,
            self.prepare_color_chart_button,
            self.measure_color_chart_button,
            self.clear_color_correction_button,
            *self.debug_buttons.values(),
        )
        for control in controls:
            control.setEnabled(not locked)
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
                "Real painting is disabled because the global abort hotkey is not active. "
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
                    apply_brush_size=self.apply_brush_check.isChecked()
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
    ) -> None:
        profile = profile or self._current_profile
        if profile is None:
            raise ValueError("No sign profile is selected")
        from app.screen import get_virtual_screen

        desktop = get_virtual_screen()
        names = ["canvas", "color_box", "hue_bar"]
        if apply_brush_size:
            names.extend(("brush_slider", "brush_preview"))
        for name in names:
            rectangle = getattr(profile, name, None)
            if rectangle is None:
                if name in {"brush_slider", "brush_preview"}:
                    raise ValueError(
                        "Automatic brush sizing requires both the Size slider and "
                        "gray brush-preview tile to be calibrated."
                    )
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
        dry_run = pending.dry_run
        if not dry_run and not self._emergency_hotkey_available():
            QMessageBox.critical(
                self,
                "Emergency hotkey unavailable",
                "Painting was cancelled because the global abort hotkey stopped "
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
            from app.painter import Painter, PainterSettings

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
            settings_document = pending.settings
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
                settings_document["safety"]["corner_abort_enabled"] = False
            settings = PainterSettings.from_mapping(settings_document)
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
            painter.configure(pending.plan, pending.profile, settings)
            if self._pending_start_cancelled:
                painter.shutdown(timeout=0.5)
                self._set_idle_ui("Start cancelled")
                return
            self._paint_generation = generation
            self._painter = painter
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
                "%s started: %d colors, %d strokes",
                "Dry run" if dry_run else "Painting",
                len(pending.plan.color_groups),
                pending.plan.stroke_count,
            )
            self._update_start_availability()
        except Exception as exc:
            LOGGER.exception("Could not start painting")
            self._on_paint_error(self._paint_generation, str(exc))

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
            brush_slider=None,
            brush_preview=None,
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
        elapsed = self._format_duration(progress.elapsed_seconds)
        self.active_detail_label.setText(f"{detail}  •  {elapsed} elapsed")

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
            "aborted": "ABORTED",
        }.get(value, value.upper())
        self._set_state_badge(value, badge_text)
        active = value in {"countdown", "running", "paused"}
        self._set_active_progress_visible(active)
        if active:
            self.active_progress_title.setText(
                {"countdown": "GET READY", "paused": "PAUSED"}.get(value, "PAINTING")
            )
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

    def _set_idle_ui(self, detail: str = "No active paint job") -> None:
        self.progress_state_label.setText("Idle")
        self.progress_detail_label.setText(detail)
        self._set_active_progress_visible(False)
        self._set_state_badge("idle", "SAFE IDLE")
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
            QMessageBox.warning(self, "Painting is active", "Pause or abort the paint job first.")
            return
        if (
            not self.dry_run_check.isChecked()
            and not self._emergency_hotkey_available()
        ):
            QMessageBox.critical(
                self,
                "Emergency hotkey unavailable",
                "Real calibration tests are disabled because the global abort hotkey "
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
            from app.screen import foreground_window_matches, get_virtual_screen

            dry_run = self.dry_run_check.isChecked()
            if not dry_run and not self._emergency_hotkey_available():
                raise RuntimeError(
                    "The global abort hotkey stopped before the debug action began."
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
            corner_enabled = not dry_run and self.corner_abort_check.isChecked()
            safety = self._settings_document().get("safety", {})
            corner_margin = int(safety.get("corner_abort_margin_pixels", 3))
            corner_distance = float(
                safety.get("corner_abort_minimum_distance_pixels", 80.0)
            )
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
                last_commanded: list[tuple[float, float] | None] = [None]

                def checkpoint() -> None:
                    if self._debug_abort_event.is_set() or self._closing:
                        raise _DebugCancelled("emergency stop requested")
                    if require_foreground and not foreground_window_matches(
                        title_contains=expected_title or None,
                        executable=expected_process or None,
                    ):
                        self._debug_abort_event.set()
                        raise _DebugCancelled("expected Rust window lost foreground")
                    if not corner_enabled:
                        return
                    cursor = controller.get_cursor_position()
                    desktop = get_virtual_screen()
                    near_x = (
                        cursor[0] <= desktop.left + corner_margin
                        or cursor[0] >= desktop.right - 1 - corner_margin
                    )
                    near_y = (
                        cursor[1] <= desktop.top + corner_margin
                        or cursor[1] >= desktop.bottom - 1 - corner_margin
                    )
                    prior = last_commanded[0]
                    displacement = (
                        math.inf
                        if prior is None
                        else math.hypot(cursor[0] - prior[0], cursor[1] - prior[1])
                    )
                    if near_x and near_y and displacement >= corner_distance:
                        self._debug_abort_event.set()
                        raise _DebugCancelled("mouse moved to emergency corner")

                def guarded_move(point: tuple[float, float]) -> None:
                    with self._debug_input_gate:
                        checkpoint()
                        controller.move_mouse(*point)
                        last_commanded[0] = point

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
        # real-input starts locked and the Abort button enabled until a later
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
