"""Multi-monitor rectangle calibration overlay.

Qt 6 lays out widgets in device-independent coordinates, while Windows input
automation consumes physical desktop coordinates.  The overlay therefore draws
with Qt coordinates but samples ``GetPhysicalCursorPos`` for the rectangle it
returns.  This keeps SendInput calibration accurate on common mixed-DPI setups.
"""

from __future__ import annotations

import ctypes
import logging
import math
import sys
from collections.abc import Callable
from ctypes import wintypes
from types import SimpleNamespace
from typing import Any

from PySide6.QtCore import QEventLoop, QPointF, QRect, QRectF, Qt, Signal
from PySide6.QtGui import (
    QColor,
    QCloseEvent,
    QGuiApplication,
    QKeyEvent,
    QMouseEvent,
    QPaintEvent,
    QPainter,
    QPen,
    QRegion,
)
from PySide6.QtWidgets import QApplication, QWidget

from .profiles import DisplayMetadata, MonitorMetadata, Rect


log = logging.getLogger(__name__)
PhysicalPositionProvider = Callable[[], tuple[int, int]]
RectangleEditedCallback = Callable[[str, Rect], None]


def _normalized_device_name(name: str) -> str:
    return name.strip().upper().removeprefix("\\\\.\\")


def physical_cursor_position() -> tuple[int, int]:
    """Return the cursor in Windows physical virtual-desktop coordinates."""

    if sys.platform == "win32":
        point = wintypes.POINT()
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        get_physical = getattr(user32, "GetPhysicalCursorPos", None)
        if get_physical is not None:
            get_physical.argtypes = (ctypes.POINTER(wintypes.POINT),)
            get_physical.restype = wintypes.BOOL
            if get_physical(ctypes.byref(point)):
                return int(point.x), int(point.y)
        get_cursor = user32.GetCursorPos
        get_cursor.argtypes = (ctypes.POINTER(wintypes.POINT),)
        get_cursor.restype = wintypes.BOOL
        if get_cursor(ctypes.byref(point)):
            return int(point.x), int(point.y)
    from PySide6.QtGui import QCursor

    cursor = QCursor.pos()
    return cursor.x(), cursor.y()


def enable_per_monitor_dpi_awareness() -> bool:
    """Request Per-Monitor-V2 DPI awareness before ``QApplication`` is created.

    Qt 6 normally requests a suitable mode itself.  This helper is useful for a
    frozen executable without a DPI-aware manifest.  ``False`` means the mode
    was already fixed (or could not be changed), not necessarily that DPI
    awareness is disabled.
    """

    if sys.platform != "win32" or QGuiApplication.instance() is not None:
        return False
    try:
        # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 == (HANDLE)-4
        context = ctypes.c_void_p(-4)
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        setter = user32.SetProcessDpiAwarenessContext
        setter.argtypes = (ctypes.c_void_p,)
        setter.restype = wintypes.BOOL
        if setter(context):
            return True
    except (AttributeError, OSError):
        pass
    try:
        # Older Windows 8.1 fallback: PROCESS_PER_MONITOR_DPI_AWARE == 2.
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        setter = shcore.SetProcessDpiAwareness
        setter.argtypes = (ctypes.c_int,)
        setter.restype = ctypes.c_long
        return setter(2) == 0
    except (AttributeError, OSError):
        return False


def _qrect_to_rect(value: QRect) -> Rect:
    return Rect(value.x(), value.y(), value.width(), value.height())


def _windows_monitors() -> dict[str, tuple[Rect, Rect, bool]]:
    """Enumerate true Win32 monitor/work rectangles keyed by display device."""

    if sys.platform != "win32":
        return {}

    class MonitorInfoExW(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
            ("szDevice", wintypes.WCHAR * 32),
        ]

    result: dict[str, tuple[Rect, Rect, bool]] = {}
    monitor_handle = getattr(wintypes, "HMONITOR", wintypes.HANDLE)
    device_context = getattr(wintypes, "HDC", wintypes.HANDLE)
    callback_type = ctypes.WINFUNCTYPE(
        wintypes.BOOL,
        monitor_handle,
        device_context,
        ctypes.POINTER(wintypes.RECT),
        wintypes.LPARAM,
    )
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    get_monitor_info = user32.GetMonitorInfoW
    get_monitor_info.argtypes = (monitor_handle, ctypes.POINTER(MonitorInfoExW))
    get_monitor_info.restype = wintypes.BOOL
    enumerate_monitors = user32.EnumDisplayMonitors
    enumerate_monitors.argtypes = (
        device_context,
        ctypes.POINTER(wintypes.RECT),
        callback_type,
        wintypes.LPARAM,
    )
    enumerate_monitors.restype = wintypes.BOOL

    def callback(handle: Any, _dc: Any, _rect: Any, _data: Any) -> bool:
        info = MonitorInfoExW()
        info.cbSize = ctypes.sizeof(info)
        if not get_monitor_info(handle, ctypes.byref(info)):
            return True
        monitor = info.rcMonitor
        work = info.rcWork
        name = _normalized_device_name(info.szDevice)
        result[name] = (
            Rect(
                int(monitor.left),
                int(monitor.top),
                int(monitor.right - monitor.left),
                int(monitor.bottom - monitor.top),
            ),
            Rect(
                int(work.left),
                int(work.top),
                int(work.right - work.left),
                int(work.bottom - work.top),
            ),
            bool(info.dwFlags & 1),
        )
        return True

    callback_pointer = callback_type(callback)
    try:
        enumerate_monitors(None, None, callback_pointer, 0)
    except (AttributeError, OSError):
        log.exception("Could not enumerate physical monitor geometry")
        return {}
    return result


def capture_display_metadata() -> DisplayMetadata:
    """Capture enough layout/scaling information to warn about stale profiles."""

    application = QGuiApplication.instance()
    if application is None:
        raise RuntimeError("A QApplication must exist before display metadata is captured")

    windows_monitors = _windows_monitors()
    qt_screens = QGuiApplication.screens()
    monitors: list[MonitorMetadata] = []
    primary = QGuiApplication.primaryScreen()

    for index, screen in enumerate(qt_screens):
        logical = _qrect_to_rect(screen.geometry())
        physical_match = windows_monitors.get(_normalized_device_name(screen.name()))
        if physical_match is None and index < len(windows_monitors):
            # Some Qt/platform combinations report a friendly model name rather
            # than DISPLAYn.  Screen order is a reasonable metadata-only fallback.
            physical_match = list(windows_monitors.values())[index]
        if physical_match is not None:
            physical, available, is_primary = physical_match
        else:
            ratio = max(float(screen.devicePixelRatio()), 1.0)
            physical = Rect(
                round(logical.left * ratio),
                round(logical.top * ratio),
                max(1, round(logical.width * ratio)),
                max(1, round(logical.height * ratio)),
            )
            available_qt = screen.availableGeometry()
            available = Rect(
                round(available_qt.x() * ratio),
                round(available_qt.y() * ratio),
                max(1, round(available_qt.width() * ratio)),
                max(1, round(available_qt.height() * ratio)),
            )
            is_primary = screen is primary

        monitors.append(
            MonitorMetadata(
                name=screen.name() or f"Display {index + 1}",
                rect=physical,
                available_rect=available,
                logical_rect=logical,
                device_pixel_ratio=float(screen.devicePixelRatio()),
                logical_dpi_x=float(screen.logicalDotsPerInchX()),
                logical_dpi_y=float(screen.logicalDotsPerInchY()),
                physical_dpi_x=float(screen.physicalDotsPerInchX()),
                physical_dpi_y=float(screen.physicalDotsPerInchY()),
                primary=is_primary,
            )
        )

    if monitors:
        left = min(item.rect.left for item in monitors)
        top = min(item.rect.top for item in monitors)
        right = max(item.rect.right for item in monitors)
        bottom = max(item.rect.bottom for item in monitors)
        virtual_screen = Rect(left, top, right - left, bottom - top)
    else:
        virtual_screen = None

    return DisplayMetadata(
        monitors=tuple(monitors),
        virtual_screen=virtual_screen,
        coordinate_space="physical" if windows_monitors else "logical",
    )


class CalibrationOverlay(QWidget):
    """Transparent virtual-desktop overlay that emits one physical ``Rect``."""

    selection_finished = Signal(object)
    selection_changed = Signal(object)
    cancelled = Signal()

    def __init__(
        self,
        instruction: str = "Drag a rectangle over the target area",
        parent: QWidget | None = None,
        *,
        minimum_size: int = 3,
        physical_position_provider: PhysicalPositionProvider | None = None,
        screen: Any | None = None,
    ) -> None:
        # A parented window can be clipped to its parent on some platforms.  Keep
        # this as a top-level tool while remembering the owner for API symmetry.
        super().__init__(None)
        self.owner = parent
        self.instruction = instruction
        self.minimum_size = max(1, int(minimum_size))
        self._physical_position_provider = physical_position_provider or physical_cursor_position
        self._start_local: QPointF | None = None
        self._current_local: QPointF | None = None
        self._start_physical: tuple[int, int] | None = None
        self._current_physical: tuple[int, int] | None = None
        self._result: Rect | None = None
        self._completed = False
        self._feedback = "Click and drag • Escape cancels"
        self._event_loop: QEventLoop | None = None

        flags = (
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
        )
        self.setWindowFlags(flags)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setCursor(Qt.CursorShape.CrossCursor)
        self.setWindowTitle("RustPainter Calibration")

        if screen is not None:
            # One overlay per monitor: a window confined to a single screen gets
            # that screen's DPI context, so it genuinely covers the whole
            # monitor. A single window stretched across mixed-DPI monitors is
            # scaled by only one monitor's DPI and leaves parts of the desktop
            # uncovered.
            self.setScreen(screen)
            self._virtual_geometry = QRect(screen.geometry())
            self.setGeometry(self._virtual_geometry)
        else:
            screens = QGuiApplication.screens()
            if not screens:
                raise RuntimeError("No screens are available for calibration")
            virtual_geometry = QRect(screens[0].geometry())
            for other in screens[1:]:
                virtual_geometry = virtual_geometry.united(other.geometry())
            self._virtual_geometry = virtual_geometry
            self.setGeometry(virtual_geometry)

    def enterEvent(self, event) -> None:  # noqa: N802 - Qt API
        # With one overlay per monitor, the window under the cursor must own
        # keyboard focus so Escape always cancels from wherever the user is.
        if not self._completed:
            self.activateWindow()
            self.setFocus(Qt.FocusReason.OtherFocusReason)
        super().enterEvent(event)

    @property
    def selected_rect(self) -> Rect | None:
        return self._result

    @property
    def selection_rect(self) -> Rect | None:
        """Alias used by a few UI integrations."""

        return self._result

    def exec(self) -> Rect | None:
        """Run a small nested event loop and return ``None`` on Escape/close."""

        if self._event_loop is not None:
            raise RuntimeError("CalibrationOverlay.exec() may only run once")
        if self._completed:
            return self._result
        self._event_loop = QEventLoop(self)
        self.selection_finished.connect(self._event_loop.quit)
        self.cancelled.connect(self._event_loop.quit)
        self.show()
        self.raise_()
        self.activateWindow()
        self.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        self._event_loop.exec()
        return self._result

    def cancel(self) -> None:
        if self._completed:
            return
        self._completed = True
        self._result = None
        self.cancelled.emit()
        self.close()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() == Qt.MouseButton.RightButton:
            self.cancel()
            return
        if event.button() != Qt.MouseButton.LeftButton:
            return
        self._start_local = event.position()
        self._current_local = event.position()
        self._start_physical = self._sample_physical(event)
        self._current_physical = self._start_physical
        self._feedback = "Release to accept • Escape cancels"
        self.grabMouse()
        self.update()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if self._start_local is None:
            self._current_local = event.position()
            self._current_physical = self._sample_physical(event)
            self.update()
            return
        self._current_local = event.position()
        self._current_physical = self._sample_physical(event)
        candidate = self._physical_rect()
        if candidate is not None:
            self.selection_changed.emit(candidate)
        self.update()
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() != Qt.MouseButton.LeftButton or self._start_local is None:
            return
        self.releaseMouse()
        self._current_local = event.position()
        self._current_physical = self._sample_physical(event)
        candidate = self._physical_rect()
        if (
            candidate is None
            or candidate.width < self.minimum_size
            or candidate.height < self.minimum_size
        ):
            self._feedback = f"Selection must be at least {self.minimum_size} x {self.minimum_size}"
            self._start_local = None
            self._start_physical = None
            self.update()
            event.accept()
            return
        self._result = candidate
        self._completed = True
        self.selection_finished.emit(candidate)
        self.close()
        event.accept()

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        if event.key() == Qt.Key.Key_Escape:
            self.cancel()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if not self._completed:
            self._completed = True
            self._result = None
            self.cancelled.emit()
        try:
            self.releaseMouse()
        except RuntimeError:
            pass
        event.accept()

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(4, 8, 7, 158))

        for screen in QGuiApplication.screens():
            screen_rect = QRect(screen.geometry())
            screen_rect.translate(-self._virtual_geometry.topLeft())
            painter.setPen(QPen(QColor(255, 255, 255, 55), 1, Qt.PenStyle.DashLine))
            painter.drawRect(screen_rect.adjusted(0, 0, -1, -1))
            painter.setPen(QColor(255, 255, 255, 105))
            painter.drawText(screen_rect.adjusted(12, 10, -10, -10), screen.name())

        selection = self._local_selection()
        if selection is not None and selection.width() > 0 and selection.height() > 0:
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Clear)
            painter.fillRect(selection, QColor(0, 0, 0, 0))
            painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceOver)
            painter.fillRect(selection, QColor(151, 203, 72, 28))
            painter.setPen(QPen(QColor(166, 220, 82), 2))
            painter.drawRect(selection)

        self._paint_banner(painter)
        self._paint_measurement(painter, selection)
        painter.end()

    def _paint_banner(self, painter: QPainter) -> None:
        width = min(680.0, max(320.0, self.width() - 40.0))
        banner = QRectF((self.width() - width) / 2.0, 22.0, width, 70.0)
        painter.setPen(QPen(QColor(166, 220, 82), 1))
        painter.setBrush(QColor(16, 20, 17, 232))
        painter.drawRoundedRect(banner, 8, 8)
        painter.setPen(QColor(242, 245, 239))
        font = painter.font()
        font.setPointSize(12)
        font.setBold(True)
        painter.setFont(font)
        painter.drawText(
            banner.adjusted(16, 8, -16, -30),
            Qt.AlignmentFlag.AlignCenter,
            self.instruction,
        )
        font.setPointSize(9)
        font.setBold(False)
        painter.setFont(font)
        painter.setPen(QColor(193, 199, 188))
        painter.drawText(
            banner.adjusted(16, 35, -16, -7),
            Qt.AlignmentFlag.AlignCenter,
            self._feedback,
        )

    def _paint_measurement(self, painter: QPainter, selection: QRectF | None) -> None:
        physical = self._physical_rect()
        if physical is not None:
            text = (
                f"Left {physical.left}   Top {physical.top}   "
                f"Width {physical.width}   Height {physical.height}"
            )
        elif self._current_physical is not None:
            x, y = self._current_physical
            text = f"X {x}   Y {y}"
        else:
            return

        metrics = painter.fontMetrics()
        width = metrics.horizontalAdvance(text) + 24
        height = metrics.height() + 14
        if selection is not None:
            x = min(max(selection.left(), 8.0), max(8.0, self.width() - width - 8.0))
            y = selection.bottom() + 10.0
            if y + height > self.height() - 8:
                y = selection.top() - height - 10.0
        else:
            x, y = 12.0, self.height() - height - 12.0
        box = QRectF(x, max(8.0, y), width, height)
        painter.setPen(QPen(QColor(166, 220, 82), 1))
        painter.setBrush(QColor(8, 11, 9, 235))
        painter.drawRoundedRect(box, 5, 5)
        painter.setPen(QColor(242, 245, 239))
        painter.drawText(box, Qt.AlignmentFlag.AlignCenter, text)

    def _local_selection(self) -> QRectF | None:
        if self._start_local is None or self._current_local is None:
            return None
        left = min(self._start_local.x(), self._current_local.x())
        top = min(self._start_local.y(), self._current_local.y())
        right = max(self._start_local.x(), self._current_local.x())
        bottom = max(self._start_local.y(), self._current_local.y())
        return QRectF(left, top, right - left, bottom - top)

    def _physical_rect(self) -> Rect | None:
        if self._start_physical is None or self._current_physical is None:
            return None
        start_x, start_y = self._start_physical
        end_x, end_y = self._current_physical
        width, height = abs(end_x - start_x), abs(end_y - start_y)
        if width == 0 or height == 0:
            return None
        return Rect(min(start_x, end_x), min(start_y, end_y), width, height)

    def _sample_physical(self, event: QMouseEvent) -> tuple[int, int]:
        try:
            x, y = self._physical_position_provider()
            if math.isfinite(x) and math.isfinite(y):
                return round(x), round(y)
        except (OSError, RuntimeError, TypeError, ValueError):
            log.warning("Physical cursor query failed; using Qt global coordinates", exc_info=True)
        position = event.globalPosition()
        return round(position.x()), round(position.y())


# SetWindowDisplayAffinity: the window shows on the monitor and nowhere
# else - screen captures, GDI BitBlt included, see what is behind it.
_WDA_EXCLUDEFROMCAPTURE = 0x11


def exclude_window_from_capture(widget: QWidget) -> bool:
    """Keep ``widget`` out of every screen capture, the app's own included.

    The painter reads the sign back off the screen - to verify strokes, to
    watch for the painting UI going away, to record the timelapse - and an
    overlay drawn over the sign would be read back with it.  With this
    affinity the overlay is seen by the player alone.  Returns False where
    the call is unavailable (not Windows, or older than 10 2004), in which
    case the caller must not draw over anything the painter captures.
    """

    if sys.platform != "win32":
        return False
    try:
        hwnd = int(widget.winId())
        return bool(
            ctypes.windll.user32.SetWindowDisplayAffinity(
                wintypes.HWND(hwnd), wintypes.DWORD(_WDA_EXCLUDEFROMCAPTURE)
            )
        )
    except Exception:
        log.warning("Could not exclude the overlay from screen capture", exc_info=True)
        return False


class _CalibrationPreviewWindow(QWidget):
    """One monitor's click-through window of labeled calibration outlines."""

    def __init__(self, screen: Any, monitor: Any) -> None:
        super().__init__(None)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setWindowTitle("RustPainter Calibration Preview")
        self.setScreen(screen)
        self.setGeometry(screen.geometry())
        self._monitor = monitor
        self._entries: list[tuple[str, Rect]] = []
        self._status: tuple[str, Rect] | None = None
        self._alerts: tuple[str, ...] = ()
        self._capture_excluded = False

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        self._capture_excluded = exclude_window_from_capture(self)
        if not self._capture_excluded:
            # The status is dropped rather than risk the painter reading it
            # back off the sign, so say why it is missing.
            log.warning(
                "This window could not be kept out of screen captures; "
                "the job's status will not be written on the sign"
            )

    def _on_monitor(self, rect: Rect) -> bool:
        physical = self._monitor.rect
        return (
            rect.left < physical.right
            and rect.right > physical.left
            and rect.top < physical.bottom
            and rect.bottom > physical.top
        )

    # Both setters are called on every progress tick of a running job, and
    # repainting a monitor-sized translucent window is not free, so a
    # repaint is asked for only when what is drawn actually changed.
    def set_entries(self, entries: list[tuple[str, Rect]]) -> None:
        mine = [(label, rect) for label, rect in entries if self._on_monitor(rect)]
        if mine == self._entries:
            return
        self._entries = mine
        self.update()

    def set_status(self, status: tuple[str, Rect] | None) -> None:
        """Show ``status`` - a word and the sign it belongs to - or nothing."""

        mine = status if status is not None and self._on_monitor(status[1]) else None
        if mine == self._status:
            return
        self._status = mine
        self.update()

    def set_alerts(self, alerts: tuple[str, ...]) -> None:
        alerts = tuple(alert for alert in alerts if alert)
        if alerts == self._alerts:
            return
        self._alerts = alerts
        self.update()

    def _map_physical_rect(self, rect: Rect) -> QRectF:
        """Convert a physical rectangle to this window's local coordinates."""

        physical = self._monitor.rect
        logical = self._monitor.logical_rect
        scale_x = logical.width / physical.width
        scale_y = logical.height / physical.height
        return QRectF(
            (rect.left - physical.left) * scale_x,
            (rect.top - physical.top) * scale_y,
            rect.width * scale_x,
            rect.height * scale_y,
        )

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        outline = QColor(255, 59, 48)
        font = painter.font()
        font.setPointSize(10)
        font.setBold(True)
        painter.setFont(font)
        metrics = painter.fontMetrics()

        for label, rect in self._entries:
            box = self._map_physical_rect(rect)
            painter.setPen(QPen(outline, 1))
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(box)

            tag_width = metrics.horizontalAdvance(label) + 16.0
            tag_height = metrics.height() + 6.0
            tag_top = box.top() - tag_height - 4.0
            if tag_top < 4.0:
                tag_top = box.top() + 4.0
            tag_left = min(max(4.0, box.left()), max(4.0, self.width() - tag_width - 4.0))
            tag = QRectF(tag_left, tag_top, tag_width, tag_height)
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(24, 8, 6, 220))
            painter.drawRoundedRect(tag, 5, 5)
            painter.setPen(QColor(255, 112, 102))
            painter.drawText(tag, Qt.AlignmentFlag.AlignCenter, label)
        # Without the capture exclusion the status would be read back off
        # the sign by the painter's own captures, so it is not drawn at all.
        if self._status is not None and self._capture_excluded:
            self._paint_status(painter, *self._status, self._alerts)
        painter.end()

    # The job's state in the app's orange, in the top-right corner of the
    # monitor the sign is on: the reassurance that the app is the one moving
    # the mouse, and what it is doing with it.  Out of the corner of the eye
    # rather than across the sign, so it never sits over the part being
    # painted or over the picture the player is trying to look at.
    _STATUS_COLOR = QColor(255, 147, 54)
    # Read from the corner of the eye, the color says it before the word
    # does: white while nothing is happening, green while the job is
    # painting, red while it waits on the user.
    _STATUS_WORD_COLORS = {
        "IDLE": QColor(240, 240, 240),
        "PAINTING": QColor(72, 214, 108),
        "PAUSED": QColor(255, 72, 60),
    }
    _STATUS_BACKDROP = QColor(24, 8, 6, 190)
    # Inset from the monitor's corner, as a share of the text height.
    _STATUS_INSET = 0.8

    def _paint_status(
        self,
        painter: QPainter,
        text: str,
        canvas: Rect,
        alerts: tuple[str, ...] = (),
    ) -> None:
        del canvas  # Only decides which monitor writes the status.
        if not text or self.width() <= 0 or self.height() <= 0:
            return
        font = painter.font()
        font.setBold(True)
        # Sized to the monitor rather than the sign: a corner label is read
        # by looking at it, so it only has to be legible, not large.
        pixel_size = max(14, min(36, int(self.height() * 0.026)))
        font.setPixelSize(pixel_size)
        painter.setFont(font)
        metrics = painter.fontMetrics()
        pad_x, pad_y = pixel_size * 0.5, pixel_size * 0.2
        pill = QRectF(
            0.0,
            0.0,
            metrics.horizontalAdvance(text) + pad_x * 2,
            metrics.height() + pad_y * 2,
        )
        inset = pixel_size * self._STATUS_INSET
        pill.moveTopRight(QPointF(self.width() - inset, inset))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(self._STATUS_BACKDROP)
        painter.drawRoundedRect(pill, pixel_size * 0.3, pixel_size * 0.3)
        painter.setPen(self._STATUS_WORD_COLORS.get(text, self._STATUS_COLOR))
        painter.drawText(pill, Qt.AlignmentFlag.AlignCenter, text)
        alert_top = pill.bottom() + pixel_size * 0.25
        for alert in alerts:
            alert_pill = QRectF(
                0.0,
                0.0,
                metrics.horizontalAdvance(alert) + pad_x * 2,
                metrics.height() + pad_y * 2,
            )
            alert_pill.moveTopRight(QPointF(self.width() - inset, alert_top))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(self._STATUS_BACKDROP)
            painter.drawRoundedRect(alert_pill, pixel_size * 0.3, pixel_size * 0.3)
            painter.setPen(QColor(255, 36, 36))
            painter.drawText(alert_pill, Qt.AlignmentFlag.AlignCenter, alert)
            alert_top = alert_pill.bottom() + pixel_size * 0.2


def _resize_edges(
    rect: Rect, x: int, y: int, tolerance: int
) -> tuple[bool, bool, bool, bool]:
    """Return the closest horizontal and vertical edges near ``(x, y)``."""

    tolerance = max(1, int(tolerance))
    left_distance = abs(x - rect.left)
    right_distance = abs(x - rect.right)
    top_distance = abs(y - rect.top)
    bottom_distance = abs(y - rect.bottom)
    left = left_distance <= tolerance and left_distance <= right_distance
    right = right_distance <= tolerance and right_distance < left_distance
    top = top_distance <= tolerance and top_distance <= bottom_distance
    bottom = bottom_distance <= tolerance and bottom_distance < top_distance
    return left, right, top, bottom


def _resized_rect(
    rect: Rect,
    edges: tuple[bool, bool, bool, bool],
    x: int,
    y: int,
    minimum_size: int,
) -> Rect:
    """Resize ``rect`` along ``edges``, keeping both dimensions usable."""

    left_edge, right_edge, top_edge, bottom_edge = edges
    minimum_size = max(1, int(minimum_size))
    left, right = rect.left, rect.right
    top, bottom = rect.top, rect.bottom
    if left_edge:
        left = min(int(x), right - minimum_size)
    elif right_edge:
        right = max(int(x), left + minimum_size)
    if top_edge:
        top = min(int(y), bottom - minimum_size)
    elif bottom_edge:
        bottom = max(int(y), top + minimum_size)
    return Rect(left, top, right - left, bottom - top)


class _CalibrationResizeHandle(QWidget):
    """An input-only border around one preview rectangle.

    Its center is removed from the native window mask, so clicks inside a
    calibrated region still reach Rust.  Only the edge and corner hit zones
    accept input.
    """

    rectangle_changing = Signal(str, object)
    rectangle_changed = Signal(str, object)

    _HIT_WIDTH = 7
    _MINIMUM_SIZE = 3

    def __init__(self, label: str, rect: Rect, screen: Any, monitor: Any) -> None:
        super().__init__(None)
        self._label = label
        self._rect = rect
        self._monitor = monitor
        self._drag_rect: Rect | None = None
        self._drag_edges: tuple[bool, bool, bool, bool] | None = None
        self._hover_edges = (False, False, False, False)
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        self.setMouseTracking(True)
        self.setWindowTitle(f"Resize {label} calibration")
        self.setScreen(screen)
        self._set_border_geometry(screen)

    def showEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        exclude_window_from_capture(self)

    def paintEvent(self, event: QPaintEvent) -> None:  # noqa: N802 - Qt API
        """Keep Windows' layered hit test alive and reveal hovered handles."""

        del event
        painter = QPainter(self)
        # Windows passes mouse input through fully transparent pixels in a
        # layered window.  Alpha 1 is visually imperceptible but makes the
        # masked border a real hit target; the interior is absent from the
        # window mask and remains genuinely click-through.
        painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_Source)
        painter.fillRect(self.rect(), QColor(255, 59, 48, 1))
        if any(self._hover_edges):
            painter.setCompositionMode(
                QPainter.CompositionMode.CompositionMode_SourceOver
            )
            painter.setPen(QPen(QColor(255, 88, 76, 235), 3))
            hit = self._HIT_WIDTH
            left, right, top, bottom = self._hover_edges
            x_left = hit
            x_right = self.width() - hit - 1
            y_top = hit
            y_bottom = self.height() - hit - 1
            if left:
                painter.drawLine(x_left, y_top, x_left, y_bottom)
            if right:
                painter.drawLine(x_right, y_top, x_right, y_bottom)
            if top:
                painter.drawLine(x_left, y_top, x_right, y_top)
            if bottom:
                painter.drawLine(x_left, y_bottom, x_right, y_bottom)
            if (left or right) and (top or bottom):
                center_x = x_left if left else x_right
                center_y = y_top if top else y_bottom
                painter.setBrush(QColor(255, 88, 76, 245))
                painter.drawRect(QRect(center_x - 4, center_y - 4, 8, 8))
        painter.end()

    def _set_hover_edges(
        self, edges: tuple[bool, bool, bool, bool]
    ) -> None:
        if edges == self._hover_edges:
            return
        self._hover_edges = edges
        self.update()

    def _set_border_geometry(self, screen: Any) -> None:
        physical = self._monitor.rect
        logical = self._monitor.logical_rect
        scale_x = logical.width / physical.width
        scale_y = logical.height / physical.height
        box = QRectF(
            screen.geometry().x() + (self._rect.left - physical.left) * scale_x,
            screen.geometry().y() + (self._rect.top - physical.top) * scale_y,
            self._rect.width * scale_x,
            self._rect.height * scale_y,
        )
        hit = self._HIT_WIDTH
        geometry = QRect(
            math.floor(box.left()) - hit,
            math.floor(box.top()) - hit,
            max(1, math.ceil(box.width()) + hit * 2),
            max(1, math.ceil(box.height()) + hit * 2),
        )
        self.setGeometry(geometry)
        outer = QRegion(self.rect())
        inner = self.rect().adjusted(hit * 2, hit * 2, -hit * 2, -hit * 2)
        self.setMask(outer.subtracted(QRegion(inner)) if inner.isValid() else outer)

    def _sample_physical(self, event: QMouseEvent) -> tuple[int, int]:
        try:
            x, y = physical_cursor_position()
            if math.isfinite(x) and math.isfinite(y):
                return round(x), round(y)
        except (OSError, RuntimeError, TypeError, ValueError):
            log.warning("Physical cursor query failed while resizing", exc_info=True)
        position = event.globalPosition()
        physical = self._monitor.rect
        logical = self._monitor.logical_rect
        return (
            round(
                physical.left
                + (position.x() - logical.left) * physical.width / logical.width
            ),
            round(
                physical.top
                + (position.y() - logical.top) * physical.height / logical.height
            ),
        )

    def _edges_at(self, x: int, y: int) -> tuple[bool, bool, bool, bool]:
        physical = self._monitor.rect
        logical = self._monitor.logical_rect
        tolerance = math.ceil(
            self._HIT_WIDTH
            * max(physical.width / logical.width, physical.height / logical.height)
        )
        return _resize_edges(self._rect, x, y, tolerance)

    @staticmethod
    def _cursor_for(edges: tuple[bool, bool, bool, bool]) -> Qt.CursorShape:
        left, right, top, bottom = edges
        if (left and top) or (right and bottom):
            return Qt.CursorShape.SizeFDiagCursor
        if (right and top) or (left and bottom):
            return Qt.CursorShape.SizeBDiagCursor
        if left or right:
            return Qt.CursorShape.SizeHorCursor
        if top or bottom:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.ArrowCursor

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() != Qt.MouseButton.LeftButton:
            return
        x, y = self._sample_physical(event)
        edges = self._edges_at(x, y)
        if not any(edges):
            return
        self._drag_rect = self._rect
        self._drag_edges = edges
        self._set_hover_edges(edges)
        self.grabMouse()
        event.accept()

    def mouseMoveEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        x, y = self._sample_physical(event)
        if self._drag_rect is None or self._drag_edges is None:
            edges = self._edges_at(x, y)
            self._set_hover_edges(edges)
            self.setCursor(self._cursor_for(edges))
            return
        self._rect = _resized_rect(
            self._drag_rect, self._drag_edges, x, y, self._MINIMUM_SIZE
        )
        self.rectangle_changing.emit(self._label, self._rect)
        event.accept()

    def mouseReleaseEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API
        if event.button() != Qt.MouseButton.LeftButton or self._drag_rect is None:
            return
        self.releaseMouse()
        x, y = self._sample_physical(event)
        assert self._drag_edges is not None
        self._rect = _resized_rect(
            self._drag_rect, self._drag_edges, x, y, self._MINIMUM_SIZE
        )
        self._drag_rect = None
        self._drag_edges = None
        self.rectangle_changing.emit(self._label, self._rect)
        self.rectangle_changed.emit(self._label, self._rect)
        event.accept()

    def leaveEvent(self, event: Any) -> None:  # noqa: N802 - Qt API
        if self._drag_rect is None:
            self._set_hover_edges((False, False, False, False))
            self.unsetCursor()
        super().leaveEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        try:
            self.releaseMouse()
        except RuntimeError:
            pass
        event.accept()


class CalibrationPreviewOverlay:
    """Labeled on-screen outlines of every calibrated rectangle.

    Manages one click-through window per monitor so every window renders in its
    own monitor's DPI context; a single window stretched across mixed-scaling
    monitors would be sized by only one monitor's DPI and drawn misaligned.
    """

    def __init__(self) -> None:
        self._windows: list[_CalibrationPreviewWindow] = []
        self._handles: list[_CalibrationResizeHandle] = []
        self._entries: list[tuple[str, Rect]] = []

        self._screen_monitors: list[tuple[Any, Any]] = []
        self._editable = False
        self._rectangle_edited: RectangleEditedCallback | None = None
        self._status: tuple[str, Rect] | None = None
        self._alerts: tuple[str, ...] = ()

    def set_resize_callback(self, callback: RectangleEditedCallback | None) -> None:
        self._rectangle_edited = callback

    def set_editable(self, editable: bool) -> None:
        editable = bool(editable)
        if editable == self._editable:
            return
        self._editable = editable
        self._rebuild_handles()

    def set_rectangles(self, entries: list[tuple[str, Rect | None]]) -> None:
        updated = [
            (str(label), rect) for label, rect in entries if rect is not None
        ]
        if updated == self._entries:
            return
        self._entries = updated
        for window in self._windows:
            window.set_entries(self._entries)
        self._rebuild_handles()

    def set_status(self, status: tuple[str, Rect] | None) -> None:
        """Write a job's state across the canvas, or wipe it with None."""

        self._status = status
        for window in self._windows:
            window.set_status(status)

    def set_alerts(self, alerts: tuple[str, ...]) -> None:
        self._alerts = tuple(alerts)
        for window in self._windows:
            window.set_alerts(self._alerts)

    def show_overlay(self) -> None:
        # Rebuild from the live monitor layout every time the overlay comes
        # back, so display/DPI changes made while hidden are always honored.
        self._destroy_windows()
        self._screen_monitors = []
        screens = QGuiApplication.screens()
        if not screens:
            return
        try:
            monitors: tuple[Any, ...] = capture_display_metadata().monitors
        except Exception:
            log.exception("Could not capture display metadata for the preview overlay")
            monitors = ()
        for index, screen in enumerate(screens):
            if index < len(monitors):
                monitor = monitors[index]
            else:
                geometry = screen.geometry()
                fallback = Rect(
                    geometry.x(), geometry.y(), geometry.width(), geometry.height()
                )
                monitor = SimpleNamespace(rect=fallback, logical_rect=fallback)
            window = _CalibrationPreviewWindow(screen, monitor)
            window.set_entries(self._entries)
            window.set_status(self._status)
            window.set_alerts(self._alerts)
            self._windows.append(window)
            self._screen_monitors.append((screen, monitor))
            window.show()
            window.raise_()
        self._rebuild_handles()

    def hide(self) -> None:
        for window in self._windows:
            window.hide()
        for handle in self._handles:
            handle.hide()

    def isVisible(self) -> bool:  # noqa: N802 - mirrors the QWidget API
        return any(window.isVisible() for window in self._windows)

    def close(self) -> None:
        self._destroy_windows()

    def _destroy_windows(self) -> None:
        self._destroy_handles()
        windows, self._windows = self._windows, []
        for window in windows:
            try:
                window.close()
            finally:
                window.deleteLater()

    def _destroy_handles(self) -> None:
        handles, self._handles = self._handles, []
        for handle in handles:
            try:
                handle.close()
            finally:
                handle.deleteLater()

    def _rebuild_handles(self) -> None:
        self._destroy_handles()
        if not self._editable or not self.isVisible():
            return
        for label, rect in self._entries:
            for screen, monitor in self._screen_monitors:
                physical = monitor.rect
                if not (
                    rect.left < physical.right
                    and rect.right > physical.left
                    and rect.top < physical.bottom
                    and rect.bottom > physical.top
                ):
                    continue
                handle = _CalibrationResizeHandle(label, rect, screen, monitor)
                handle.rectangle_changing.connect(self._preview_resize)
                handle.rectangle_changed.connect(self._finish_resize)
                self._handles.append(handle)
                handle.show()
                handle.raise_()

    def _replace_entry(self, label: str, rect: Rect) -> None:
        self._entries = [
            (entry_label, rect if entry_label == label else entry_rect)
            for entry_label, entry_rect in self._entries
        ]
        for window in self._windows:
            window.set_entries(self._entries)

    def _preview_resize(self, label: str, rect: Rect) -> None:
        self._replace_entry(label, rect)

    def _finish_resize(self, label: str, rect: Rect) -> None:
        self._replace_entry(label, rect)
        self._rebuild_handles()
        if self._rectangle_edited is not None:
            self._rectangle_edited(label, rect)


def select_screen_rect(
    parent: QWidget | None = None,
    instruction: str = "Drag a rectangle over the target area",
    *,
    minimum_size: int = 3,
) -> Rect | None:
    """Hide ``parent``, collect a rectangle, then restore the parent window.

    One overlay window is created per monitor so each window is rendered in its
    own monitor's DPI context. This keeps every monitor fully covered on
    mixed-scaling multi-monitor desktops; the selection itself is still sampled
    in physical coordinates.
    """

    application = QApplication.instance()
    if application is None:
        raise RuntimeError("A QApplication must exist before calibration")
    screens = QGuiApplication.screens()
    if not screens:
        raise RuntimeError("No screens are available for calibration")

    was_visible = parent is not None and parent.isVisible()
    previous_state = parent.windowState() if parent is not None else None
    if was_visible and parent is not None:
        parent.hide()
        QApplication.processEvents()

    overlays = [
        CalibrationOverlay(
            instruction=instruction,
            parent=parent,
            minimum_size=minimum_size,
            screen=screen,
        )
        for screen in screens
    ]
    loop = QEventLoop()
    state: dict[str, Any] = {"result": None, "done": False}

    def finish(rect: Rect | None) -> None:
        if state["done"]:
            return
        state["done"] = True
        state["result"] = rect
        loop.quit()

    for overlay in overlays:
        overlay.selection_finished.connect(finish)
        overlay.cancelled.connect(lambda: finish(None))

    try:
        cursor_x, cursor_y = physical_cursor_position()
        for overlay in overlays:
            overlay.show()
            overlay.raise_()
        # Give initial keyboard focus to the monitor the cursor is on; hovering
        # any other overlay moves focus there via enterEvent.
        focused = overlays[0]
        try:
            metadata = capture_display_metadata()
            for overlay, monitor in zip(overlays, metadata.monitors):
                rect = monitor.rect
                if (
                    rect.left <= cursor_x < rect.right
                    and rect.top <= cursor_y < rect.bottom
                ):
                    focused = overlay
                    break
        except Exception:
            log.warning("Could not resolve the cursor's monitor", exc_info=True)
        focused.activateWindow()
        focused.setFocus(Qt.FocusReason.ActiveWindowFocusReason)
        loop.exec()
        return state["result"]
    finally:
        for overlay in overlays:
            try:
                overlay.blockSignals(True)
                overlay.close()
            finally:
                overlay.deleteLater()
        if was_visible and parent is not None:
            parent.show()
            if previous_state is not None:
                parent.setWindowState(previous_state)
            parent.raise_()
            parent.activateWindow()


# Descriptive alias used by the main-window calibration button handler.
calibrate_rectangle = select_screen_rect


__all__ = [
    "CalibrationOverlay",
    "CalibrationPreviewOverlay",
    "calibrate_rectangle",
    "capture_display_metadata",
    "enable_per_monitor_dpi_awareness",
    "physical_cursor_position",
    "select_screen_rect",
]
