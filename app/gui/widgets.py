"""Small reusable Qt widgets used by the main window."""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import Qt, QTimer, Signal, QObject
from PySide6.QtGui import QColor, QCloseEvent, QPainter, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .assets import pixmap as art_pixmap, tinted_pixmap
from .styles import ACCENT, MUTED, SUCCESS, WARNING


class NoWheelComboBox(QComboBox):
    """Combo box that lets a surrounding scroll area keep the mouse wheel."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class NoWheelSpinBox(QSpinBox):
    """Integer editor that changes only through deliberate editing/buttons."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class NoWheelDoubleSpinBox(QDoubleSpinBox):
    """Float editor that changes only through deliberate editing/buttons."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class PreviewLabel(QLabel):
    """A pixmap label that keeps the source aspect ratio while resizing."""

    def __init__(
        self,
        placeholder: str,
        parent: QWidget | None = None,
        *,
        smooth: bool = True,
    ) -> None:
        super().__init__("", parent)
        self.setObjectName("preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 250)
        self._source = QPixmap()
        self._smooth = smooth
        self._placeholder = placeholder
        self._placeholder_art = art_pixmap("preview-placeholder", 96)

    def set_source(self, pixmap: QPixmap | None) -> None:
        self._source = pixmap or QPixmap()
        self._update_scaled()
        self.update()

    def clear_source(self, placeholder: str = "No image loaded") -> None:
        self._source = QPixmap()
        self.setPixmap(QPixmap())
        self._placeholder = placeholder
        self.update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._update_scaled()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().paintEvent(event)
        if not self._source.isNull():
            return
        # QLabel shows either a pixmap or text, so the empty state — artwork
        # above its caption — is drawn by hand.
        area = self.contentsRect()
        painter = QPainter(self)
        art = self._placeholder_art
        top = area.center().y() - (art.height() + 14) // 2
        painter.drawPixmap(area.center().x() - art.width() // 2, top, art)
        painter.setPen(QColor(MUTED))
        painter.drawText(
            area.adjusted(0, top + art.height() + 12 - area.top(), 0, 0),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
            self._placeholder,
        )
        painter.end()

    def _update_scaled(self) -> None:
        if self._source.isNull():
            return
        target = self.contentsRect().size()
        if target.width() <= 0 or target.height() <= 0:
            return
        self.setText("")
        self.setPixmap(
            self._source.scaled(
                target,
                Qt.AspectRatioMode.KeepAspectRatio,
                (
                    Qt.TransformationMode.SmoothTransformation
                    if self._smooth
                    else Qt.TransformationMode.FastTransformation
                ),
            )
        )


class ColorButton(QPushButton):
    """A button that stores and edits a QColor."""

    colorChanged = Signal(QColor)

    def __init__(self, color: QColor | str = "#ffffff", parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self.clicked.connect(self._choose)
        self._refresh()

    def color(self) -> QColor:
        return QColor(self._color)

    def set_color(self, color: QColor | str, emit: bool = False) -> None:
        candidate = QColor(color)
        if not candidate.isValid():
            return
        changed = candidate != self._color
        self._color = candidate
        self._refresh()
        if changed and emit:
            self.colorChanged.emit(QColor(self._color))

    def _choose(self) -> None:
        chosen = QColorDialog.getColor(self._color, self, "Choose background color")
        if chosen.isValid():
            self.set_color(chosen, emit=True)

    def _refresh(self) -> None:
        foreground = "#111111" if self._color.lightnessF() > 0.55 else "#ffffff"
        self.setText(self._color.name().upper())
        self.setStyleSheet(
            f"QPushButton {{ background: {self._color.name()}; color: {foreground}; "
            "font-weight: 700; border: 1px solid #5b3a22; border-radius: 8px; }"
        )


class CalibrationStatus(QWidget):
    """Compact label showing whether a profile rectangle exists."""

    def __init__(self, name: str, optional: bool = False, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._name = QLabel(name)
        self._value = QLabel()
        self.setMinimumWidth(0)
        self.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        # Ignored width lets the whole row shrink with the side panel, but the
        # name still needs a floor or the label collapses away entirely.
        self._name.setMinimumWidth(78)
        self._name.setSizePolicy(
            QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred
        )
        self._icon = QLabel()
        self._icon.setFixedSize(14, 14)
        self._icon.setScaledContents(True)
        layout.addWidget(self._name)
        layout.addStretch(1)
        layout.addWidget(self._icon)
        layout.addWidget(self._value)
        self.set_calibrated(False, optional)

    def set_calibrated(self, calibrated: bool, optional: bool = False) -> None:
        if calibrated:
            text, color, glyph = "Ready", SUCCESS, "check"
        elif optional:
            text, color, glyph = "Optional", MUTED, "status"
        else:
            text, color, glyph = "Needed", WARNING, "target"
        self._icon.setPixmap(tinted_pixmap(glyph, color, 28))
        self._value.setText(text)
        self._value.setStyleSheet(f"color: {color}; font-weight: 600;")


class CountdownDialog(QDialog):
    """Modal safety countdown that invokes a callback when it reaches zero."""

    def __init__(
        self,
        seconds: int,
        on_finished: Callable[[], None],
        parent: QWidget | None = None,
        *,
        hint: str = "F10 cancels",
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("RustPainter")
        self.setModal(False)
        self.setWindowFlag(Qt.WindowType.WindowStaysOnTopHint, True)
        self.setMinimumWidth(360)
        self._remaining = max(1, seconds)
        self._on_finished = on_finished
        self._completed = False
        self._cancelled = False

        layout = QVBoxLayout(self)
        title = QLabel("Switch to Rust now")
        title.setObjectName("sectionTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count = QLabel()
        self._count.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._count.setStyleSheet(
            f"font-size: 42pt; font-weight: 800; color: {ACCENT};"
        )
        detail = QLabel(hint)
        detail.setObjectName("muted")
        detail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        layout.addWidget(title)
        layout.addWidget(self._count)
        layout.addWidget(detail)
        layout.addWidget(cancel)

        self._timer = QTimer(self)
        self._timer.setInterval(1000)
        self._timer.timeout.connect(self._tick)
        self.finished.connect(lambda _result: self._timer.stop())
        self._render()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().showEvent(event)
        if not self._completed and not self._cancelled:
            self._timer.start()

    def reject(self) -> None:
        self._cancelled = True
        self._timer.stop()
        super().reject()

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        if not self._completed:
            self._cancelled = True
        self._timer.stop()
        super().closeEvent(event)

    def _render(self) -> None:
        self._count.setText(str(self._remaining))

    def _tick(self) -> None:
        if self._cancelled or self._completed:
            self._timer.stop()
            return
        self._remaining -= 1
        if self._remaining > 0:
            self._render()
            return
        self._timer.stop()
        self._completed = True
        self.accept()
        self._on_finished()


class _LogEmitter(QObject):
    message = Signal(str)


class QtLogHandler(logging.Handler):
    """Logging handler that safely forwards formatted records to the Qt thread."""

    def __init__(self, append: Callable[[str], None]) -> None:
        super().__init__()
        self.emitter = _LogEmitter()
        self.emitter.message.connect(append)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self.emitter.message.emit(self.format(record))
        except Exception:
            self.handleError(record)
