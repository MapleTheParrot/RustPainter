"""Small reusable Qt widgets used by the main window."""

from __future__ import annotations

import logging
from collections.abc import Callable

from PySide6.QtCore import QPointF, Qt, QTimer, Signal, QObject
from PySide6.QtGui import QColor, QCloseEvent, QFont, QPainter, QPixmap
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
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


class _MovableTextItem(QGraphicsTextItem):
    """A text item that reports its normalized center while being dragged."""

    def __init__(self, index: int, moved: Callable[[int, float, float], None]) -> None:
        super().__init__()
        self.index = index
        self._moved = moved
        self._syncing = False
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setCursor(Qt.CursorShape.OpenHandCursor)

    def set_center(self, x: float, y: float) -> None:
        bounds = self.boundingRect()
        self._syncing = True
        self.setPos(x - bounds.width() / 2.0, y - bounds.height() / 2.0)
        self._syncing = False

    def itemChange(self, change, value):  # noqa: N802 - Qt API
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and self.scene() is not None
        ):
            position = value
            bounds = self.boundingRect()
            scene = self.scene().sceneRect()
            if bounds.width() >= scene.width():
                position.setX(scene.left() - bounds.left())
            else:
                position.setX(
                    min(
                        max(position.x(), scene.left() - bounds.left()),
                        scene.right() - bounds.right(),
                    )
                )
            if bounds.height() >= scene.height():
                position.setY(scene.top() - bounds.top())
            else:
                position.setY(
                    min(
                        max(position.y(), scene.top() - bounds.top()),
                        scene.bottom() - bounds.bottom(),
                    )
                )
            return position
        result = super().itemChange(change, value)
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
            and not self._syncing
            and self.scene() is not None
        ):
            scene = self.scene().sceneRect()
            center = self.mapToScene(self.boundingRect().center())
            if scene.width() > 0 and scene.height() > 0:
                self._moved(
                    self.index,
                    (center.x() - scene.left()) / scene.width(),
                    (center.y() - scene.top()) / scene.height(),
                )
        return result


class TextEditorPreview(QGraphicsView):
    """Image preview with selectable, draggable text layers."""

    layerMoved = Signal(int, float, float)
    layerSelected = Signal(int)

    def __init__(self, placeholder: str, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("preview")
        self.setMinimumSize(300, 250)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setDragMode(QGraphicsView.DragMode.NoDrag)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._background = QGraphicsPixmapItem()
        self._background.setZValue(-100.0)
        self._scene.addItem(self._background)
        self._scene.selectionChanged.connect(self._on_selection_changed)
        self._source = QPixmap()
        self._items: list[_MovableTextItem] = []
        self._placeholder = placeholder
        self._placeholder_art = art_pixmap("preview-placeholder", 96)

    def set_source(self, pixmap: QPixmap | None) -> None:
        self._source = pixmap or QPixmap()
        self._background.setPixmap(self._source)
        if self._source.isNull():
            self._scene.setSceneRect(0, 0, 1, 1)
        else:
            self._scene.setSceneRect(0, 0, self._source.width(), self._source.height())
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self.viewport().update()

    def clear_source(self, placeholder: str = "No image loaded") -> None:
        self._placeholder = placeholder
        self.set_source(None)
        self.set_layers([], -1)

    def set_layers(self, layers: list[object], selected_index: int) -> None:
        for item in self._items:
            self._scene.removeItem(item)
        self._items.clear()
        if self._source.isNull():
            return
        width = float(self._source.width())
        height = float(self._source.height())
        for index, layer in enumerate(layers):
            text = str(getattr(layer, "text", ""))
            if not text.strip():
                continue
            item = _MovableTextItem(index, self.layerMoved.emit)
            item.setPlainText(text)
            font = QFont(str(getattr(layer, "font_family", "")))
            font.setPixelSize(max(1, int(getattr(layer, "font_size", 24))))
            font.setBold(bool(getattr(layer, "bold", False)))
            font.setItalic(bool(getattr(layer, "italic", False)))
            item.setFont(font)
            color = getattr(layer, "color", (255, 255, 255))
            item.setDefaultTextColor(QColor(*color))
            item.setZValue(float(index))
            self._scene.addItem(item)
            item.set_center(
                float(getattr(layer, "x", 0.5)) * width,
                float(getattr(layer, "y", 0.5)) * height,
            )
            item.setSelected(index == selected_index)
            self._items.append(item)

    def select_layer(self, index: int) -> None:
        for item in self._items:
            item.setSelected(item.index == index)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if not self._source.isNull():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().paintEvent(event)
        if not self._source.isNull():
            return
        area = self.viewport().rect()
        painter = QPainter(self.viewport())
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

    def _on_selection_changed(self) -> None:
        selected = self._scene.selectedItems()
        if selected and isinstance(selected[0], _MovableTextItem):
            self.layerSelected.emit(selected[0].index)


class ColorButton(QPushButton):
    """A button that stores and edits a QColor."""

    colorChanged = Signal(QColor)

    def __init__(
        self,
        color: QColor | str = "#ffffff",
        parent: QWidget | None = None,
        *,
        dialog_title: str = "Choose background color",
    ) -> None:
        super().__init__(parent)
        self._color = QColor(color)
        self._dialog_title = dialog_title
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
        original = QColor(self._color)
        dialog = QColorDialog(self._color, self)
        dialog.setWindowTitle(self._dialog_title)
        # The non-native dialog reliably emits currentColorChanged while its
        # controls are being moved, which makes the editor preview genuinely live.
        dialog.setOption(QColorDialog.ColorDialogOption.DontUseNativeDialog, True)
        dialog.currentColorChanged.connect(lambda color: self.set_color(color, emit=True))
        if dialog.exec() != QDialog.DialogCode.Accepted:
            self.set_color(original, emit=True)

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
