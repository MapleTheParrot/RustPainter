"""Small reusable Qt widgets used by the main window."""

from __future__ import annotations

import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

from PySide6.QtCore import (
    QEvent,
    QObject,
    QPointF,
    QRect,
    QRectF,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QColor, QCloseEvent, QFont, QPainter, QPainterPath, QPixmap
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
from .styles import ACCENT, ACCENT_SOFT, MUTED, SUCCESS, WARNING


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


IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tif", ".tiff"}

_HOVER_EVENTS = {QEvent.Type.Enter, QEvent.Type.Leave}


def dropped_image_path(event: Any) -> Path | None:
    """The first local image file carried by a drag/drop event, if any."""

    mime = event.mimeData()
    if mime is None or not mime.hasUrls():
        return None
    for url in mime.urls():
        if not url.isLocalFile():
            continue
        path = Path(url.toLocalFile())
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            return path
    return None


def _paint_placeholder(
    device: QWidget,
    area: QRect,
    art: QPixmap,
    caption: str,
    hint: str,
    active: bool,
) -> None:
    """Draw the empty-state artwork, its caption, and its click hint.

    QLabel and QGraphicsView both show either content or nothing, so the empty
    state is drawn by hand. ``active`` lights the text up while the pointer or a
    dragged file is over the preview, which is what tells the user the area is
    itself a button.
    """

    painter = QPainter(device)
    block = art.height() + 14
    top = area.center().y() - block // 2
    painter.drawPixmap(area.center().x() - art.width() // 2, top, art)
    caption_top = top + art.height() + 12
    painter.setPen(QColor(ACCENT if active else MUTED))
    painter.drawText(
        area.adjusted(0, caption_top - area.top(), 0, 0),
        int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
        caption,
    )
    if hint:
        font = painter.font()
        font.setPointSizeF(max(6.5, font.pointSizeF() - 1.0))
        painter.setFont(font)
        painter.setPen(QColor(ACCENT_SOFT if active else "#6f6157"))
        painter.drawText(
            area.adjusted(0, caption_top + 20 - area.top(), 0, 0),
            int(Qt.AlignmentFlag.AlignHCenter | Qt.AlignmentFlag.AlignTop),
            hint,
        )
    painter.end()


class _ImageDropTarget:
    """Click-to-browse and drag-and-drop behaviour shared by both previews.

    Each preview has to be a drop target in its own right. A QGraphicsView
    accepts drag events itself so that scene items can receive them, which
    means the Rust preview tab swallowed every drop before the window saw it.
    Handling drops here covers both tabs and lets the empty state double as a
    browse button.
    """

    # Declared by the QWidget subclasses; repeated here for the type checker.
    browseRequested: Any
    imageDropped: Any

    def _init_drop_target(self) -> None:
        self._drop_active = False
        self._hovering = False
        self.setAcceptDrops(True)
        # A QGraphicsView never sees Enter/Leave itself: those go to its
        # viewport, so the hover highlight watches whichever widget gets them.
        self._hover_source = getattr(self, "viewport", lambda: self)()
        self._hover_source.installEventFilter(self)

    @property
    def _is_empty(self) -> bool:
        return self._source.isNull()

    def _set_drop_active(self, active: bool) -> None:
        if active == self._drop_active:
            return
        self._drop_active = active
        # A dynamic property lets the stylesheet own the highlight colours.
        self.setProperty("dropActive", active)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def _refresh_empty_state(self) -> None:
        """Only the empty state is clickable, so the cursor follows it."""

        self.setCursor(
            Qt.CursorShape.PointingHandCursor
            if self._is_empty
            else Qt.CursorShape.ArrowCursor
        )
        self.update()

    @property
    def _placeholder_active(self) -> bool:
        return self._drop_active or (self._hovering and self._is_empty)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if watched is self._hover_source and event.type() in _HOVER_EVENTS:
            self._hovering = event.type() == QEvent.Type.Enter
            self.update()
        return super().eventFilter(watched, event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._is_empty and event.button() == Qt.MouseButton.LeftButton:
            self.browseRequested.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def dragEnterEvent(self, event) -> None:  # noqa: N802 - Qt API
        if dropped_image_path(event) is None:
            event.ignore()
            return
        self._set_drop_active(True)
        event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if dropped_image_path(event) is None:
            event.ignore()
            return
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._set_drop_active(False)
        event.accept()

    def dropEvent(self, event) -> None:  # noqa: N802 - Qt API
        path = dropped_image_path(event)
        self._set_drop_active(False)
        if path is None:
            event.ignore()
            return
        event.acceptProposedAction()
        self.imageDropped.emit(str(path))


class PreviewLabel(_ImageDropTarget, QLabel):
    """A pixmap label that keeps the source aspect ratio while resizing."""

    browseRequested = Signal()
    imageDropped = Signal(str)

    def __init__(
        self,
        placeholder: str,
        parent: QWidget | None = None,
        *,
        smooth: bool = True,
        hint: str = "",
    ) -> None:
        super().__init__("", parent)
        self.setObjectName("preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 250)
        self._source = QPixmap()
        self._smooth = smooth
        self._placeholder = placeholder
        self._hint = hint
        self._placeholder_art = art_pixmap("preview-placeholder", 96)
        self._init_drop_target()
        self._refresh_empty_state()

    def set_source(self, pixmap: QPixmap | None) -> None:
        self._source = pixmap or QPixmap()
        self._update_scaled()
        self._refresh_empty_state()

    def clear_source(self, placeholder: str = "No image loaded") -> None:
        self._source = QPixmap()
        self.setPixmap(QPixmap())
        self._placeholder = placeholder
        self._refresh_empty_state()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._update_scaled()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().paintEvent(event)
        if self._source.isNull():
            _paint_placeholder(
                self,
                self.contentsRect(),
                self._placeholder_art,
                self._placeholder,
                self._hint,
                self._placeholder_active,
            )

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
    """Movable text with direct editing and edge/corner resize handles."""

    def __init__(
        self,
        index: int,
        moved: Callable[[int, float, float], None],
        text_edited: Callable[[int, str], None],
        resized: Callable[[int, int], None],
        interaction_finished: Callable[[], None],
    ) -> None:
        super().__init__()
        self.index = index
        self._moved = moved
        self._text_edited = text_edited
        self._resized = resized
        self._interaction_finished = interaction_finished
        self._syncing = True
        self._editing = False
        self._dragging = False
        self._resize_handle: str | None = None
        self._resize_start = QPointF()
        self._resize_start_size = 24
        self._resize_start_width = 1.0
        self._resize_start_height = 1.0
        self._resize_center = QPointF()
        self.setFlags(
            QGraphicsItem.GraphicsItemFlag.ItemIsMovable
            | QGraphicsItem.GraphicsItemFlag.ItemIsSelectable
            | QGraphicsItem.GraphicsItemFlag.ItemSendsGeometryChanges
        )
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setAcceptHoverEvents(True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self.document().contentsChanged.connect(self._on_document_changed)

    @property
    def is_interacting(self) -> bool:
        return self._editing or self._dragging or self._resize_handle is not None

    @property
    def is_editing(self) -> bool:
        return self._editing

    def set_center(self, x: float, y: float) -> None:
        bounds = self.boundingRect()
        self._syncing = True
        self.setPos(x - bounds.width() / 2.0, y - bounds.height() / 2.0)
        self._syncing = False

    def finish_setup(self) -> None:
        self._syncing = False

    def _handle_rects(self) -> dict[str, QRectF]:
        bounds = super().boundingRect()
        size = max(4.0, min(9.0, max(1, self.font().pixelSize()) * 0.24))
        half = size / 2.0
        left, center_x, right = bounds.left(), bounds.center().x(), bounds.right()
        top, center_y, bottom = bounds.top(), bounds.center().y(), bounds.bottom()
        return {
            "top_left": QRectF(left - half, top - half, size, size),
            "top": QRectF(center_x - half, top - half, size, size),
            "top_right": QRectF(right - half, top - half, size, size),
            "right": QRectF(right - half, center_y - half, size, size),
            "bottom_right": QRectF(right - half, bottom - half, size, size),
            "bottom": QRectF(center_x - half, bottom - half, size, size),
            "bottom_left": QRectF(left - half, bottom - half, size, size),
            "left": QRectF(left - half, center_y - half, size, size),
        }

    def boundingRect(self) -> QRectF:  # noqa: N802 - Qt API
        bounds = super().boundingRect()
        size = max(4.0, min(9.0, max(1, self.font().pixelSize()) * 0.24))
        return bounds.adjusted(-size / 2.0, -size / 2.0, size / 2.0, size / 2.0)

    def shape(self) -> QPainterPath:
        path = super().shape()
        if self.isSelected() and not self._editing:
            for rectangle in self._handle_rects().values():
                path.addRect(rectangle)
        return path

    def _handle_at(self, position: QPointF) -> str | None:
        if not self.isSelected() or self._editing:
            return None
        for name, rectangle in self._handle_rects().items():
            if rectangle.contains(position):
                return name
        return None

    @staticmethod
    def _handle_cursor(handle: str | None) -> Qt.CursorShape:
        if handle in {"top_left", "bottom_right"}:
            return Qt.CursorShape.SizeFDiagCursor
        if handle in {"top_right", "bottom_left"}:
            return Qt.CursorShape.SizeBDiagCursor
        if handle in {"left", "right"}:
            return Qt.CursorShape.SizeHorCursor
        if handle in {"top", "bottom"}:
            return Qt.CursorShape.SizeVerCursor
        return Qt.CursorShape.OpenHandCursor

    def paint(self, painter, option, widget=None) -> None:
        super().paint(painter, option, widget)
        if not self.isSelected() or self._editing:
            return
        painter.save()
        painter.setPen(QColor("#fff1e2"))
        painter.setBrush(QColor(ACCENT))
        for rectangle in self._handle_rects().values():
            painter.drawRect(rectangle)
        painter.restore()

    def hoverMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.setCursor(self._handle_cursor(self._handle_at(event.pos())))
        super().hoverMoveEvent(event)

    def hoverLeaveEvent(self, event) -> None:  # noqa: N802 - Qt API
        self.setCursor(Qt.CursorShape.IBeamCursor if self._editing else Qt.CursorShape.OpenHandCursor)
        super().hoverLeaveEvent(event)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        handle = self._handle_at(event.pos())
        if handle is not None:
            self._resize_handle = handle
            self._resize_start = event.scenePos()
            self._resize_start_size = max(1, self.font().pixelSize())
            text_bounds = super().boundingRect()
            self._resize_start_width = max(1.0, text_bounds.width())
            self._resize_start_height = max(1.0, text_bounds.height())
            self._resize_center = self.mapToScene(text_bounds.center())
            event.accept()
            return
        self._dragging = not self._editing
        self.setCursor(
            Qt.CursorShape.IBeamCursor
            if self._editing
            else Qt.CursorShape.ClosedHandCursor
        )
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._resize_handle is None:
            super().mouseMoveEvent(event)
            return
        delta = event.scenePos() - self._resize_start
        horizontal = (
            -1
            if "left" in self._resize_handle
            else 1
            if "right" in self._resize_handle
            else 0
        )
        vertical = (
            -1
            if "top" in self._resize_handle
            else 1
            if "bottom" in self._resize_handle
            else 0
        )
        outward = delta.x() * horizontal + delta.y() * vertical
        axes = int(horizontal != 0) + int(vertical != 0)
        if axes > 1:
            outward /= axes
        extent = max(
            self._resize_start_width if horizontal else 0.0,
            self._resize_start_height if vertical else 0.0,
            1.0,
        )
        font_size = round(self._resize_start_size * (1.0 + 2.0 * outward / extent))
        self._apply_font_size(min(max(font_size, 4), 256))
        event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        was_interacting = self._dragging or self._resize_handle is not None
        if self._resize_handle is not None:
            self._resize_handle = None
            event.accept()
        else:
            super().mouseReleaseEvent(event)
        self._dragging = False
        self.setCursor(
            Qt.CursorShape.IBeamCursor
            if self._editing
            else Qt.CursorShape.OpenHandCursor
        )
        if was_interacting:
            self._interaction_finished()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        self._editing = True
        self._dragging = False
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, False)
        self.setTextInteractionFlags(Qt.TextInteractionFlag.TextEditorInteraction)
        self.setCursor(Qt.CursorShape.IBeamCursor)
        self.setFocus(Qt.FocusReason.MouseFocusReason)
        super().mouseDoubleClickEvent(event)
        self.update()

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Escape}:
            self.clearFocus()
            event.accept()
            return
        super().keyPressEvent(event)

    def focusOutEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().focusOutEvent(event)
        if not self._editing:
            return
        self._editing = False
        self.setTextInteractionFlags(Qt.TextInteractionFlag.NoTextInteraction)
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsMovable, True)
        self.setCursor(Qt.CursorShape.OpenHandCursor)
        self._interaction_finished()
        self.update()

    def _apply_font_size(self, font_size: int) -> None:
        if font_size == self.font().pixelSize():
            return
        font = self.font()
        font.setPixelSize(font_size)
        self._syncing = True
        self.setFont(font)
        self.set_center(self._resize_center.x(), self._resize_center.y())
        self._syncing = False
        self._resized(self.index, font_size)
        self.update()

    def _on_document_changed(self) -> None:
        if not self._syncing:
            self._text_edited(self.index, self.toPlainText())

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


class TextEditorPreview(_ImageDropTarget, QGraphicsView):
    """Image preview with selectable, editable, resizable text layers."""

    layerMoved = Signal(int, float, float)
    layerSelected = Signal(int)
    layerTextEdited = Signal(int, str)
    layerResized = Signal(int, int)
    layerDeleteRequested = Signal(int)
    layerDuplicateRequested = Signal(int)
    interactionFinished = Signal()
    browseRequested = Signal()
    imageDropped = Signal(str)

    def __init__(
        self,
        placeholder: str,
        parent: QWidget | None = None,
        *,
        hint: str = "",
    ) -> None:
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
        self._hint = hint
        self._placeholder_art = art_pixmap("preview-placeholder", 96)
        self._init_drop_target()
        self._refresh_empty_state()

    @property
    def is_interacting(self) -> bool:
        return any(item.is_interacting for item in self._items)

    @property
    def is_editing_text(self) -> bool:
        return any(item.is_editing for item in self._items)

    def set_source(self, pixmap: QPixmap | None) -> None:
        self._source = pixmap or QPixmap()
        self._background.setPixmap(self._source)
        if self._source.isNull():
            self._scene.setSceneRect(0, 0, 1, 1)
        else:
            self._scene.setSceneRect(0, 0, self._source.width(), self._source.height())
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)
        self._refresh_empty_state()
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
            item = _MovableTextItem(
                index,
                self.layerMoved.emit,
                self.layerTextEdited.emit,
                self.layerResized.emit,
                self.interactionFinished.emit,
            )
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
            item.finish_setup()
            self._items.append(item)

    def select_layer(self, index: int) -> None:
        for item in self._items:
            item.setSelected(item.index == index)

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if not self._source.isNull():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Edit the selected layer with the keyboard.

        Delete or Backspace removes it; Ctrl+D or Ctrl+C copies it. While a
        layer is being edited every one of those keys belongs to the text
        cursor instead, so the shortcuts only apply to a layer that is merely
        selected.
        """

        if not self.is_editing_text:
            selected = self.selected_index()
            control = bool(
                event.modifiers() & Qt.KeyboardModifier.ControlModifier
            )
            deletes = event.key() in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}
            duplicates = control and event.key() in {Qt.Key.Key_D, Qt.Key.Key_C}
            if selected is not None and (deletes or duplicates):
                event.accept()
                if deletes:
                    self.layerDeleteRequested.emit(selected)
                else:
                    self.layerDuplicateRequested.emit(selected)
                return
        super().keyPressEvent(event)

    def selected_index(self) -> int | None:
        for item in self._items:
            if item.isSelected():
                return item.index
        return None

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().paintEvent(event)
        if self._source.isNull():
            _paint_placeholder(
                self.viewport(),
                self.viewport().rect(),
                self._placeholder_art,
                self._placeholder,
                self._hint,
                self._placeholder_active,
            )

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
