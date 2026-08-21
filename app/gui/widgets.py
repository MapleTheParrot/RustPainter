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
from PySide6.QtGui import (
    QColor,
    QCloseEvent,
    QFont,
    QPainter,
    QPainterPath,
    QPen,
    QPixmap,
)
from PySide6.QtWidgets import (
    QColorDialog,
    QComboBox,
    QDialog,
    QDoubleSpinBox,
    QFontComboBox,
    QFrame,
    QGraphicsItem,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from .assets import pixmap as art_pixmap, tinted_pixmap
from .styles import ACCENT, ACCENT_SOFT, MUTED, SUCCESS, WARNING
from .text_render import TextStyle, draw_text


class NoWheelComboBox(QComboBox):
    """Combo box that lets a surrounding scroll area keep the mouse wheel."""

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API
        event.ignore()


class Spinner(QWidget):
    """A small rotating arc that signals background work is in flight.

    Hidden while stopped, so it can live permanently in a layout and simply be
    started when a worker kicks off and stopped when its result lands.
    """

    def __init__(
        self,
        diameter: int = 16,
        line_width: int = 3,
        color: str = ACCENT,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._line_width = max(1, line_width)
        self._color = QColor(color)
        self._angle = 0
        self._timer = QTimer(self)
        self._timer.setInterval(33)
        self._timer.timeout.connect(self._advance)
        self.setFixedSize(diameter, diameter)
        self.hide()

    @property
    def is_spinning(self) -> bool:
        return self._timer.isActive()

    def start(self) -> None:
        if not self._timer.isActive():
            self._timer.start()
        self.show()

    def stop(self) -> None:
        self._timer.stop()
        self.hide()

    def _advance(self) -> None:
        self._angle = (self._angle + 10) % 360
        self.update()

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        pen = painter.pen()
        pen.setColor(self._color)
        pen.setWidth(self._line_width)
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        painter.setPen(pen)
        margin = self._line_width // 2 + 1
        bounds = self.rect().adjusted(margin, margin, -margin, -margin)
        # Qt angles are in sixteenths of a degree, counterclockwise.
        painter.drawArc(bounds, -self._angle * 16, 300 * 16)
        painter.end()


class NoWheelFontComboBox(QFontComboBox):
    """Font picker that changes only when opened and chosen from deliberately.

    Scrolling the settings column with the pointer over this combo used to
    change the selected font; the wheel now scrolls the surrounding area, and
    the dropdown list still scrolls normally once it is open.
    """

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

# What makes a floating overlay recompute where it sits.
_RESIZE_EVENTS = {QEvent.Type.Resize, QEvent.Type.Show}

# How near a dragged text layer has to come, in screen pixels, before it jumps
# onto an alignment.  Small enough that ordinary dragging never feels magnetic,
# and Alt turns it off outright.
SNAP_DISTANCE_PIXELS = 7.0

# Arrow keys nudge a selection in whole logical canvas pixels.
_ARROW_STEPS: dict[Any, tuple[float, float]] = {
    Qt.Key.Key_Left: (-1.0, 0.0),
    Qt.Key.Key_Right: (1.0, 0.0),
    Qt.Key.Key_Up: (0.0, -1.0),
    Qt.Key.Key_Down: (0.0, 1.0),
}


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


def _paint_corner_chip(device: QWidget, area: QRect, text: str) -> None:
    """Draw a quiet pill in the bottom-right corner of a preview.

    It is the standing answer to "can I edit this?", shown before the user
    tries rather than after, and is deliberately dim enough to read as a label
    on the panel instead of as content in the image.  It sits at the bottom so
    a notice arriving along the top edge never lands on top of it.
    """

    painter = QPainter(device)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    font = painter.font()
    font.setPointSizeF(max(6.5, font.pointSizeF() - 1.0))
    font.setBold(True)
    painter.setFont(font)
    metrics = painter.fontMetrics()
    width = metrics.horizontalAdvance(text) + 16
    height = metrics.height() + 6
    pill = QRectF(
        area.right() - width - 8, area.bottom() - height - 8, width, height
    )
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(18, 14, 11, 205))
    painter.drawRoundedRect(pill, height / 2, height / 2)
    painter.setPen(QColor(MUTED))
    painter.drawText(pill, int(Qt.AlignmentFlag.AlignCenter), text)
    painter.end()


class InlineNotice(QFrame):
    """A self-dismissing message that floats over the widget it explains.

    Used where a modal dialog would be out of proportion to the mistake: the
    user did something reasonable in the wrong place, and what they need is a
    sentence saying so plus the one button that puts them where they meant to
    be.  It never blocks input, it hides itself again, and it follows whatever
    it covers when that is resized.
    """

    actionTriggered = Signal()

    def __init__(
        self,
        parent: QWidget,
        *,
        icon_name: str = "status",
        seconds: float = 6.0,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("inlineNotice")
        self.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(12, 8, 10, 8)
        layout.setSpacing(9)
        self._glyph = QLabel()
        self._glyph.setFixedSize(18, 18)
        self._glyph.setScaledContents(True)
        self._glyph.setPixmap(tinted_pixmap(icon_name, ACCENT, 36))
        self._message = QLabel("")
        self._message.setObjectName("inlineNoticeText")
        self._action = QPushButton("")
        self._action.setObjectName("compactButton")
        self._action.setCursor(Qt.CursorShape.PointingHandCursor)
        self._action.clicked.connect(self._trigger)
        layout.addWidget(self._glyph)
        layout.addWidget(self._message)
        layout.addWidget(self._action)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.setInterval(max(1, int(seconds * 1000)))
        self._timer.timeout.connect(self.hide)
        self.hide()
        parent.installEventFilter(self)

    def show_message(self, message: str, action: str = "") -> None:
        self._message.setText(message)
        self._action.setText(action)
        self._action.setVisible(bool(action))
        self._reposition()
        self.show()
        self.raise_()
        self._timer.start()

    def _trigger(self) -> None:
        self.hide()
        self.actionTriggered.emit()

    def hide(self) -> None:  # noqa: D102 - QWidget API
        self._timer.stop()
        super().hide()

    def _reposition(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        self.adjustSize()
        width = min(self.width(), max(120, parent.width() - 24))
        self.resize(width, self.height())
        self.move(max(12, (parent.width() - width) // 2), 14)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if (
            watched is self.parentWidget()
            and event.type() in _RESIZE_EVENTS
            and self.isVisible()
        ):
            self._reposition()
        return super().eventFilter(watched, event)


class BusyOverlay(QWidget):
    """A scrim and a card saying what the application is working on.

    A 16-pixel spinner beside a heading is easy to miss when the control that
    started the work is somewhere else entirely, and a job that takes a minute
    then reads as an application that has stopped responding.  This covers the
    content being recalculated instead, so the answer to "is anything
    happening?" is wherever the user is already looking.  It never takes the
    mouse, so editing continues underneath while the plan catches up, and it
    waits out a short delay first so a fast recalculation never flashes.
    """

    def __init__(self, parent: QWidget, *, delay_ms: int = 220) -> None:
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)
        self._top_inset = 0
        self._card = QFrame(self)
        self._card.setObjectName("busyCard")
        self._card.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        card_layout = QVBoxLayout(self._card)
        card_layout.setContentsMargins(22, 16, 22, 16)
        card_layout.setSpacing(8)
        heading = QHBoxLayout()
        heading.setSpacing(10)
        self._spinner = Spinner(20, 3)
        self._title = QLabel("Working...")
        self._title.setObjectName("busyTitle")
        heading.addWidget(self._spinner)
        heading.addWidget(self._title)
        heading.addStretch(1)
        card_layout.addLayout(heading)
        self._detail = QLabel("")
        self._detail.setObjectName("muted")
        card_layout.addWidget(self._detail)
        self._bar = QProgressBar()
        # A job of unknown length gets the sweeping bar, which says "running"
        # far more plainly than a bar frozen at zero would.
        self._bar.setRange(0, 0)
        self._bar.setTextVisible(False)
        self._bar.setFixedHeight(6)
        card_layout.addWidget(self._bar)
        self._delay = QTimer(self)
        self._delay.setSingleShot(True)
        self._delay.setInterval(max(0, delay_ms))
        self._delay.timeout.connect(self._reveal)
        self.hide()
        parent.installEventFilter(self)

    def begin(self, title: str, detail: str = "") -> None:
        """Arm the overlay; it appears only if the work outlasts the delay."""

        self._title.setText(title)
        self._detail.setText(detail)
        self._detail.setVisible(bool(detail))
        if self.isVisible():
            self._layout_card()
        elif not self._delay.isActive():
            self._delay.start()

    def end(self) -> None:
        self._delay.stop()
        self._spinner.stop()
        self.hide()

    @property
    def is_pending(self) -> bool:
        """Whether the overlay is showing or still waiting out its delay."""

        return self._delay.isActive() or self.isVisible()

    def set_top_inset(self, pixels: int) -> None:
        """Leave the top of the parent uncovered - a tab bar stays reachable."""

        self._top_inset = max(0, int(pixels))
        if self.isVisible():
            self._layout_card()

    def _reveal(self) -> None:
        parent = self.parentWidget()
        if parent is None or not parent.isVisible():
            return
        self._spinner.start()
        self._layout_card()
        self.show()
        self.raise_()

    def _layout_card(self) -> None:
        parent = self.parentWidget()
        if parent is None:
            return
        area = parent.rect().adjusted(0, self._top_inset, 0, 0)
        self.setGeometry(area)
        size = self._card.sizeHint()
        width = min(max(size.width(), 260), max(160, area.width() - 40))
        self._card.setFixedWidth(width)
        self._card.adjustSize()
        self._card.move(
            max(0, (self.width() - self._card.width()) // 2),
            max(0, (self.height() - self._card.height()) // 2),
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API
        del event
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(8, 6, 5, 150))
        painter.end()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API
        if (
            watched is self.parentWidget()
            and event.type() in _RESIZE_EVENTS
            and self.isVisible()
        ):
            self._layout_card()
        return super().eventFilter(watched, event)


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

        self.setCursor(self._idle_cursor())
        self.update()

    def _idle_cursor(self) -> Qt.CursorShape:
        return (
            Qt.CursorShape.PointingHandCursor
            if self._is_empty
            else Qt.CursorShape.ArrowCursor
        )

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
    """A pixmap label that keeps the source aspect ratio while resizing.

    ``read_only_chip`` marks the label as something to look at rather than
    something to edit: the chip says so up front, a press emits
    :attr:`editAttempted` so the window can offer the place where that edit
    does work, and a double-click emits :attr:`editElsewhereRequested` to be
    taken there outright.
    """

    browseRequested = Signal()
    imageDropped = Signal(str)
    editAttempted = Signal()
    editElsewhereRequested = Signal()

    def __init__(
        self,
        placeholder: str,
        parent: QWidget | None = None,
        *,
        smooth: bool = True,
        hint: str = "",
        read_only_chip: str = "",
    ) -> None:
        super().__init__("", parent)
        self.setObjectName("preview")
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setMinimumSize(300, 250)
        self._source = QPixmap()
        self._smooth = smooth
        self._placeholder = placeholder
        self._hint = hint
        self._read_only_chip = read_only_chip
        self._placeholder_art = art_pixmap("preview-placeholder", 96)
        self._init_drop_target()
        self._refresh_empty_state()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        # A press on a filled read-only preview is almost always someone
        # trying to edit here; say so instead of silently doing nothing.
        if (
            self._read_only_chip
            and not self._is_empty
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self.editAttempted.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API
        # Clicking once asks what is going on and gets an answer; insisting is
        # unambiguous enough to just be taken where the edit works.
        if self._read_only_chip and not self._is_empty:
            self.editElsewhereRequested.emit()
            event.accept()
            return
        super().mouseDoubleClickEvent(event)

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
        elif self._read_only_chip:
            _paint_corner_chip(self, self.contentsRect(), self._read_only_chip)

    def set_smooth(self, smooth: bool) -> None:
        """Choose between filtered scaling and one hard block per source pixel.

        Filtered is closer to how the game draws the sign's texture; blocky
        shows each planned cell as its own square, which is what inspecting
        the plan cell by cell needs.
        """

        smooth = bool(smooth)
        if smooth == self._smooth:
            return
        self._smooth = smooth
        self._update_scaled()

    def is_smooth(self) -> bool:
        return self._smooth

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
    """Movable text with direct editing and edge/corner resize handles.

    The item's own coordinates and font stay in logical canvas pixels; the
    view scales the whole item so the same text covers the same fraction of
    the sign however large the backdrop pixmap is.  The owning view reports
    where that sign canvas sits in scene coordinates, which is what movement
    is clamped to and what the emitted position fractions are relative to.
    """

    def __init__(self, index: int, owner: "TextEditorPreview") -> None:
        super().__init__()
        self.index = index
        self._owner = owner
        self._style = TextStyle()
        self._syncing = True
        self._editing = False
        self._dragging = False
        self._drag_origin = QPointF()
        self._drag_start_rect = QRectF()
        self._drag_start: list[tuple["_MovableTextItem", QPointF]] = []
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

    def set_style(self, style: TextStyle) -> None:
        """Adopt the layer's fill and outline, which the item draws itself."""

        if style == self._style:
            return
        # An outline reaches past the text box, so the painted area grows.
        self.prepareGeometryChange()
        self._style = style
        self.update()

    def set_center(self, x: float, y: float) -> None:
        scale = max(self.scale(), 1e-6)
        center = self.boundingRect().center()
        self._syncing = True
        self.setPos(x - center.x() * scale, y - center.y() * scale)
        self._syncing = False

    def finish_setup(self) -> None:
        self._syncing = False

    def text_rect(self) -> QRectF:
        """The text's own box, without the margin handles and outlines need."""

        return QGraphicsTextItem.boundingRect(self)

    def text_scene_rect(self) -> QRectF:
        return self.mapRectToScene(self.text_rect())

    def _canvas_bounds(self) -> QRectF:
        rect = self._owner.canvas_rect()
        if rect.width() > 0 and rect.height() > 0:
            return rect
        if self.scene() is not None:
            return self.scene().sceneRect()
        return QRectF(0.0, 0.0, 1.0, 1.0)

    def _handle_size(self) -> float:
        return max(4.0, min(9.0, max(1, self.font().pixelSize()) * 0.24))

    def _handle_rects(self) -> dict[str, QRectF]:
        bounds = self.text_rect()
        size = self._handle_size()
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
        margin = self._handle_size() / 2.0 + self._style.outline
        return self.text_rect().adjusted(-margin, -margin, margin, margin)

    def shape(self) -> QPainterPath:
        path = super().shape()
        if self._handles_visible:
            for rectangle in self._handle_rects().values():
                path.addRect(rectangle)
        return path

    @property
    def _handles_visible(self) -> bool:
        """Resize handles belong to a lone selection.

        With several layers selected a handle would be ambiguous - the size
        box in the side panel is what applies to all of them - so each member
        of the group just outlines itself instead.
        """

        return (
            self.isSelected()
            and not self._editing
            and len(self._owner.selected_items()) <= 1
        )

    def _handle_at(self, position: QPointF) -> str | None:
        if not self._handles_visible:
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
        if self._editing:
            # The document draws the caret and the selection highlight, which
            # the glyph renderer knows nothing about.
            super().paint(painter, option, widget)
        else:
            draw_text(
                painter,
                self.toPlainText(),
                self.font(),
                self.text_rect().center(),
                self._style,
            )
        if not self.isSelected() or self._editing:
            return
        painter.save()
        if self._handles_visible:
            painter.setPen(QColor("#fff1e2"))
            painter.setBrush(QColor(ACCENT))
            for rectangle in self._handle_rects().values():
                painter.drawRect(rectangle)
        else:
            marker = QPen(QColor(ACCENT))
            marker.setCosmetic(True)
            painter.setPen(marker)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(self.text_rect())
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
            text_bounds = self.text_rect()
            self._resize_start_width = max(1.0, text_bounds.width())
            self._resize_start_height = max(1.0, text_bounds.height())
            self._resize_center = self.mapToScene(text_bounds.center())
            event.accept()
            return
        self._owner.note_primary(self.index)
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier and not self._editing:
            # Ctrl is the only multi-select modifier the base class knows, so
            # Shift - the one every other editor uses - is answered here: it
            # adds this layer to the selection, or drops it back out of one.
            self.setSelected(not self.isSelected())
            event.accept()
        else:
            # Qt settles the selection here, so which layers a drag carries is
            # only known once the base class has had the press.
            super().mousePressEvent(event)
        # A click that just deselected has nothing left under it to drag.
        self._dragging = not self._editing and self.isSelected()
        if self._dragging:
            self._drag_origin = event.scenePos()
            self._drag_start_rect = self.text_scene_rect()
            moving = self._owner.selected_items()
            if self not in moving:
                moving = [self]
            self._drag_start = [(item, item.pos()) for item in moving]
        self.setCursor(
            Qt.CursorShape.IBeamCursor
            if self._editing
            else Qt.CursorShape.ClosedHandCursor
        )

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._resize_handle is not None:
            self._resize_to(event)
            return
        if not self._dragging or not self._drag_start:
            super().mouseMoveEvent(event)
            return
        delta = event.scenePos() - self._drag_origin
        guides: list[tuple[str, float]] = []
        # Alt is the usual way out when the wanted spot is next to a guide.
        if not event.modifiers() & Qt.KeyboardModifier.AltModifier:
            offset, guides = self._owner.snap_offset(
                self._drag_start_rect.translated(delta),
                {item for item, _ in self._drag_start},
            )
            delta += offset
        self._owner.set_snap_guides(guides)
        # Every selected layer takes the same step, so a group keeps its shape
        # even where one member is held back by the canvas edge.
        for item, start in self._drag_start:
            item.setPos(start + delta)
        event.accept()

    def _resize_to(self, event) -> None:
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
        # The drag arrives in scene pixels while the text metrics are logical;
        # a scaled item therefore shrinks the drag back into its own units.
        outward = (delta.x() * horizontal + delta.y() * vertical) / max(
            self.scale(), 1e-6
        )
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
        elif event.modifiers() & Qt.KeyboardModifier.ShiftModifier and not self._editing:
            # The base class settles the selection a second time on release,
            # collapsing it onto whatever was clicked; Shift asked for the
            # group the press just widened, so it is left standing.
            event.accept()
        else:
            super().mouseReleaseEvent(event)
        self._dragging = False
        self._drag_start = []
        self._owner.set_snap_guides([])
        self.setCursor(
            Qt.CursorShape.IBeamCursor
            if self._editing
            else Qt.CursorShape.OpenHandCursor
        )
        if was_interacting:
            self._owner.interactionFinished.emit()

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
        self._owner.interactionFinished.emit()
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
        self._owner.layerResized.emit(self.index, font_size)
        self.update()

    def _on_document_changed(self) -> None:
        if not self._syncing:
            self._owner.layerTextEdited.emit(self.index, self.toPlainText())

    def itemChange(self, change, value):  # noqa: N802 - Qt API
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionChange
            and self.scene() is not None
        ):
            position = value
            scale = max(self.scale(), 1e-6)
            bounds = self.text_rect()
            canvas = self._canvas_bounds()
            left, top = bounds.left() * scale, bounds.top() * scale
            width, height = bounds.width() * scale, bounds.height() * scale
            if width >= canvas.width():
                position.setX(canvas.left() - left)
            else:
                position.setX(
                    min(
                        max(position.x(), canvas.left() - left),
                        canvas.right() - left - width,
                    )
                )
            if height >= canvas.height():
                position.setY(canvas.top() - top)
            else:
                position.setY(
                    min(
                        max(position.y(), canvas.top() - top),
                        canvas.bottom() - top - height,
                    )
                )
            return position
        result = super().itemChange(change, value)
        if (
            change == QGraphicsItem.GraphicsItemChange.ItemPositionHasChanged
            and not self._syncing
            and self.scene() is not None
        ):
            canvas = self._canvas_bounds()
            center = self.mapToScene(self.text_rect().center())
            if canvas.width() > 0 and canvas.height() > 0:
                self._owner.layerMoved.emit(
                    self.index,
                    (center.x() - canvas.left()) / canvas.width(),
                    (center.y() - canvas.top()) / canvas.height(),
                )
        return result


def _closest_offset(
    anchors: tuple[float, ...], targets: list[float], threshold: float
) -> tuple[float, float | None]:
    """The smallest move that lands one of ``anchors`` on one of ``targets``.

    Anchors are offered centre first and ties are kept, so a layer that could
    align equally well by its centre or by an edge prefers its centre.
    """

    offset, distance, guide = 0.0, threshold, None
    for anchor in anchors:
        for target in targets:
            if abs(target - anchor) < distance:
                distance = abs(target - anchor)
                offset = target - anchor
                guide = target
    return offset, guide


class TextEditorPreview(_ImageDropTarget, QGraphicsView):
    """Image preview with selectable, editable, resizable text layers."""

    layerMoved = Signal(int, float, float)
    layerTextEdited = Signal(int, str)
    layerResized = Signal(int, int)
    layersDeleteRequested = Signal(object)
    layersDuplicateRequested = Signal(object)
    layerSelectionChanged = Signal()
    undoRequested = Signal()
    redoRequested = Signal()
    interactionFinished = Signal()
    browseRequested = Signal()
    imageDropped = Signal(str)
    cropFocusChanged = Signal(float, float)
    cropDragFinished = Signal()

    def __init__(
        self,
        placeholder: str,
        parent: QWidget | None = None,
        *,
        smooth: bool = False,
        hint: str = "",
    ) -> None:
        super().__init__(parent)
        self.setObjectName("preview")
        self.setMinimumSize(300, 250)
        if smooth:
            self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setFrameShape(QGraphicsView.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # A drag that starts on bare canvas sweeps up every layer it touches;
        # one that starts on a layer still moves that layer.
        self.setDragMode(QGraphicsView.DragMode.RubberBandDrag)
        self.setRubberBandSelectionMode(
            Qt.ItemSelectionMode.IntersectsItemBoundingRect
        )
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._background = QGraphicsPixmapItem()
        self._background.setZValue(-100.0)
        self._scene.addItem(self._background)
        self._scene.selectionChanged.connect(self._on_selection_changed)
        self._source = QPixmap()
        self._items: list[_MovableTextItem] = []
        self._canvas_rect: QRectF | None = None
        self._font_scale = 1.0
        self._primary: int | None = None
        self._syncing_selection = False
        # What a modifier-held rubber band has to give back; see mousePressEvent.
        self._carried_selection: set[int] = set()
        self._guides: list[tuple[str, float]] = []
        # Fill crops the sign out of the middle of the source; dragging picks
        # which part of it that is.  Held from the press so a rebuilt scene
        # mid-drag cannot make the image jump.
        self._crop_pannable = False
        self._crop_origin: QPointF | None = None
        self._crop_press_rect = QRectF()
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
        self._update_scene_rect()
        self._refresh_empty_state()
        self.viewport().update()

    def set_canvas_geometry(self, rect: QRectF | None, font_scale: float = 1.0) -> None:
        """Locate the sign canvas on the displayed pixmap.

        ``rect`` is in pixmap pixels and may extend past the pixmap (Fit
        letterboxing) or cover only part of it (Fill cropping); ``None`` means
        the whole pixmap is the canvas.  ``font_scale`` is how many pixmap
        pixels one logical canvas pixel spans, which is the scale applied to
        every text item so it covers the same fraction of the sign it will
        cover when painted.
        """

        self._canvas_rect = QRectF(rect) if rect is not None else None
        self._font_scale = max(float(font_scale), 1e-6)
        self._update_scene_rect()
        if self._crop_origin is None:
            self._refresh_empty_state()
        self.viewport().update()

    def set_crop_pannable(self, pannable: bool) -> None:
        """Allow or forbid dragging the sign's window over the source image."""

        if pannable == self._crop_pannable:
            return
        self._crop_pannable = pannable
        self._crop_origin = None
        self._refresh_empty_state()

    @property
    def is_panning_crop(self) -> bool:
        return self._crop_origin is not None

    def crop_margin(self) -> tuple[float, float]:
        """How far the kept region can still travel on each axis, in pixels."""

        if not self._crop_pannable or self._canvas_rect is None or self._source.isNull():
            return (0.0, 0.0)
        return (
            max(0.0, self._source.width() - self._canvas_rect.width()),
            max(0.0, self._source.height() - self._canvas_rect.height()),
        )

    def can_pan_crop(self) -> bool:
        """Whether there is any margin at all to drag the crop across."""

        margin_x, margin_y = self.crop_margin()
        return margin_x > 0.5 or margin_y > 0.5

    def _crop_focus_at(self, left: float, top: float) -> tuple[float, float]:
        margin_x, margin_y = self.crop_margin()
        return (
            min(max(left / margin_x, 0.0), 1.0) if margin_x > 0.0 else 0.5,
            min(max(top / margin_y, 0.0), 1.0) if margin_y > 0.0 else 0.5,
        )

    def canvas_rect(self) -> QRectF:
        if (
            self._canvas_rect is not None
            and self._canvas_rect.width() > 0
            and self._canvas_rect.height() > 0
        ):
            return QRectF(self._canvas_rect)
        return QRectF(0.0, 0.0, float(self._source.width()), float(self._source.height()))

    def _update_scene_rect(self) -> None:
        if self._source.isNull():
            self._scene.setSceneRect(0, 0, 1, 1)
            return
        # Fit letterboxing places canvas area beyond the pixmap edges; keeping
        # it inside the scene keeps text dropped there visible and reachable.
        rect = QRectF(0.0, 0.0, self._source.width(), self._source.height())
        if self._canvas_rect is not None:
            rect = rect.united(self._canvas_rect)
        self._scene.setSceneRect(rect)
        self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def clear_source(self, placeholder: str = "No image loaded") -> None:
        self._placeholder = placeholder
        self._canvas_rect = None
        self._font_scale = 1.0
        self._crop_origin = None
        self.set_source(None)
        self.set_layers([], ())

    def _idle_cursor(self) -> Qt.CursorShape:
        # An open hand over bare canvas is what tells the user the sign's
        # window can be dragged onto the part of the image they want.
        if not self._is_empty and self.can_pan_crop():
            return Qt.CursorShape.OpenHandCursor
        return super()._idle_cursor()

    def set_layers(
        self,
        layers: list[object],
        selected: object = (),
        primary: int | None = None,
    ) -> None:
        """Rebuild the layer items, restoring which of them were selected."""

        wanted = (
            {int(selected)} if isinstance(selected, int) else {int(i) for i in selected}
        )
        # Tearing the old items down drops their selection, and putting the new
        # ones up retakes it; neither is a choice the user just made.
        self._syncing_selection = True
        try:
            self._rebuild_items(layers, wanted, primary)
        finally:
            self._syncing_selection = False

    def _rebuild_items(
        self, layers: list[object], wanted: set[int], primary: int | None
    ) -> None:
        for item in self._items:
            self._scene.removeItem(item)
        self._items.clear()
        self._guides = []
        if primary is not None:
            self._primary = primary
        if self._source.isNull():
            return
        canvas = self.canvas_rect()
        for index, layer in enumerate(layers):
            text = str(getattr(layer, "text", ""))
            if not text.strip():
                continue
            item = _MovableTextItem(index, self)
            item.setPlainText(text)
            font = QFont(str(getattr(layer, "font_family", "")))
            font.setPixelSize(max(1, int(getattr(layer, "font_size", 24))))
            font.setBold(bool(getattr(layer, "bold", False)))
            font.setItalic(bool(getattr(layer, "italic", False)))
            item.setFont(font)
            color = getattr(layer, "color", (255, 255, 255))
            item.setDefaultTextColor(QColor(*color))
            item.set_style(TextStyle.from_layer(layer))
            item.setZValue(float(index))
            item.setScale(self._font_scale)
            self._scene.addItem(item)
            item.set_center(
                canvas.left() + float(getattr(layer, "x", 0.5)) * canvas.width(),
                canvas.top() + float(getattr(layer, "y", 0.5)) * canvas.height(),
            )
            item.setSelected(index in wanted)
            item.finish_setup()
            self._items.append(item)

    def selected_items(self) -> list[_MovableTextItem]:
        return [item for item in self._items if item.isSelected()]

    def selected_indices(self) -> list[int]:
        return [item.index for item in self.selected_items()]

    def select_layers(self, indices: object, primary: int | None = None) -> None:
        wanted = {int(index) for index in indices}
        if primary is not None:
            self._primary = primary
        for item in self._items:
            item.setSelected(item.index in wanted)

    def select_layer(self, index: int) -> None:
        self.select_layers([index], index)

    def note_primary(self, index: int) -> None:
        """Remember the layer touched last; it is the one the panel edits."""

        self._primary = index

    def primary_index(self) -> int | None:
        selected = self.selected_indices()
        if self._primary in selected:
            return self._primary
        return selected[0] if selected else None

    def selected_index(self) -> int | None:
        return self.primary_index()

    def snap_offset(
        self, rect: QRectF, moving: set
    ) -> tuple[QPointF, list[tuple[str, float]]]:
        """Nudge a dragged box onto the nearest canvas or layer alignment.

        The threshold is a distance on screen rather than one in the scene, so
        how sticky a guide feels does not change with the size of the artwork.
        """

        threshold = SNAP_DISTANCE_PIXELS / max(abs(self.transform().m11()), 1e-6)
        canvas = self.canvas_rect()
        xs = [canvas.center().x(), canvas.left(), canvas.right()]
        ys = [canvas.center().y(), canvas.top(), canvas.bottom()]
        for item in self._items:
            if item in moving:
                continue
            other = item.text_scene_rect()
            xs += [other.center().x(), other.left(), other.right()]
            ys += [other.center().y(), other.top(), other.bottom()]
        offset_x, guide_x = _closest_offset(
            (rect.center().x(), rect.left(), rect.right()), xs, threshold
        )
        offset_y, guide_y = _closest_offset(
            (rect.center().y(), rect.top(), rect.bottom()), ys, threshold
        )
        guides: list[tuple[str, float]] = []
        if guide_x is not None:
            guides.append(("vertical", guide_x))
        if guide_y is not None:
            guides.append(("horizontal", guide_y))
        return QPointF(offset_x, offset_y), guides

    def set_snap_guides(self, guides: list[tuple[str, float]]) -> None:
        if guides == self._guides:
            return
        self._guides = list(guides)
        self.viewport().update()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        if not self._source.isNull():
            self.fitInView(self._scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio)

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Remember a selection a modifier-held rubber band should keep.

        Qt clears the selection the moment a band starts, so a Shift-drag
        meant to gather a second group would otherwise throw the first one
        away.  What was selected is carried across the drag and handed back as
        the band makes its own picks.
        """

        extend = event.modifiers() & (
            Qt.KeyboardModifier.ShiftModifier | Qt.KeyboardModifier.ControlModifier
        )
        if (
            not extend
            and event.button() == Qt.MouseButton.LeftButton
            and self._starts_crop_pan(event)
        ):
            self._crop_origin = self.mapToScene(event.position().toPoint())
            self._crop_press_rect = QRectF(self._canvas_rect or QRectF())
            self.viewport().setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
            return
        self._carried_selection = set(self.selected_indices()) if extend else set()
        super().mousePressEvent(event)

    def _starts_crop_pan(self, event) -> bool:
        """A plain drag on bare canvas moves the crop, not a rubber band.

        A text layer under the pointer still wins - dragging a caption is what
        that gesture already means - and the band is reachable with Shift or
        Ctrl held, which is how several layers were always gathered anyway.
        """

        if self._source.isNull() or not self.can_pan_crop():
            return False
        return self.itemAt(event.position().toPoint()) in (None, self._background)

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._crop_origin is not None:
            moved = self.mapToScene(event.position().toPoint()) - self._crop_origin
            self.cropFocusChanged.emit(
                *self._crop_focus_at(
                    self._crop_press_rect.left() + moved.x(),
                    self._crop_press_rect.top() + moved.y(),
                )
            )
            event.accept()
            return
        super().mouseMoveEvent(event)
        if self._carried_selection and not self.rubberBandRect().isNull():
            for item in self._items:
                if item.index in self._carried_selection:
                    item.setSelected(True)

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API
        if self._crop_origin is not None:
            self._crop_origin = None
            self.viewport().unsetCursor()
            self._refresh_empty_state()
            self.cropDragFinished.emit()
            event.accept()
            return
        carried = self._carried_selection
        banding = not self.rubberBandRect().isNull()
        self._carried_selection = set()
        super().mouseReleaseEvent(event)
        if carried and banding:
            self.select_layers(carried | set(self.selected_indices()))

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API
        """Edit the selected layers with the keyboard.

        Delete or Backspace removes them, Ctrl+D or Ctrl+C copies them, Ctrl+A
        takes all of them, and the arrow keys nudge them a logical pixel at a
        time - ten with Shift held.  While a layer is being edited every one of
        those keys belongs to the text cursor instead, so they only apply to
        layers that are merely selected.

        Ctrl+Z and Ctrl+Y are the exception: they walk the text history even
        mid-edit, because typing into a layer is exactly when a mistake wants
        taking back, and the history records the typing too.
        """

        control = bool(event.modifiers() & Qt.KeyboardModifier.ControlModifier)
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if control and event.key() in {Qt.Key.Key_Z, Qt.Key.Key_Y}:
            event.accept()
            if event.key() == Qt.Key.Key_Y or shift:
                self.redoRequested.emit()
            else:
                self.undoRequested.emit()
            return
        if not self.is_editing_text:
            selected = self.selected_indices()
            key = event.key()
            if control and key == Qt.Key.Key_A and self._items:
                self.select_layers([item.index for item in self._items])
                event.accept()
                return
            if key == Qt.Key.Key_Escape and selected:
                self.select_layers([])
                event.accept()
                return
            if selected and key in {Qt.Key.Key_Delete, Qt.Key.Key_Backspace}:
                event.accept()
                self.layersDeleteRequested.emit(selected)
                return
            if selected and control and key in {Qt.Key.Key_D, Qt.Key.Key_C}:
                event.accept()
                self.layersDuplicateRequested.emit(selected)
                return
            if selected and key in _ARROW_STEPS:
                event.accept()
                self._nudge(
                    _ARROW_STEPS[key],
                    10.0
                    if event.modifiers() & Qt.KeyboardModifier.ShiftModifier
                    else 1.0,
                )
                return
        super().keyPressEvent(event)

    def _nudge(self, direction: tuple[float, float], steps: float) -> None:
        step = self._font_scale * steps
        for item in self.selected_items():
            item.setPos(
                item.pos().x() + direction[0] * step,
                item.pos().y() + direction[1] * step,
            )
        self.interactionFinished.emit()

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

    def drawForeground(self, painter: QPainter, rect: QRectF) -> None:  # noqa: N802
        """Mark the sign canvas, and whatever a drag is lined up on right now.

        Fill mode dims the cropped-away margins so text cannot be parked on
        pixels the sign never receives; the dashed border also outlines the
        letterbox area Fit adds beyond the source.  Snap guides are solid, and
        last only as long as the drag holding that alignment.
        """

        super().drawForeground(painter, rect)
        if self._source.isNull():
            return
        canvas = self.canvas_rect()
        pixmap_rect = QRectF(0.0, 0.0, self._source.width(), self._source.height())
        if self._canvas_rect is not None and not (
            canvas.contains(pixmap_rect) and pixmap_rect.contains(canvas)
        ):
            cropped = QPainterPath()
            cropped.addRect(pixmap_rect)
            kept = QPainterPath()
            kept.addRect(canvas.intersected(pixmap_rect))
            painter.fillPath(cropped.subtracted(kept), QColor(0, 0, 0, 110))
            border = QPen(QColor(ACCENT))
            border.setCosmetic(True)
            border.setStyle(Qt.PenStyle.DashLine)
            painter.setPen(border)
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(canvas)
            if self.can_pan_crop():
                self._draw_crop_grips(painter, canvas)
        if not self._guides:
            return
        guide = QPen(QColor(ACCENT))
        guide.setCosmetic(True)
        painter.setPen(guide)
        span = canvas.united(self._scene.sceneRect())
        for orientation, position in self._guides:
            if orientation == "vertical":
                painter.drawLine(
                    QPointF(position, span.top()), QPointF(position, span.bottom())
                )
            else:
                painter.drawLine(
                    QPointF(span.left(), position), QPointF(span.right(), position)
                )

    def _draw_crop_grips(self, painter: QPainter, canvas: QRectF) -> None:
        """Bracket the crop frame's corners so it reads as something to grab."""

        grip = QPen(QColor(ACCENT))
        grip.setCosmetic(True)
        grip.setWidth(3)
        painter.setPen(grip)
        arm = min(canvas.width(), canvas.height()) * 0.09
        for x, step_x in ((canvas.left(), 1.0), (canvas.right(), -1.0)):
            for y, step_y in ((canvas.top(), 1.0), (canvas.bottom(), -1.0)):
                painter.drawLine(
                    QPointF(x, y), QPointF(x + arm * step_x, y)
                )
                painter.drawLine(
                    QPointF(x, y), QPointF(x, y + arm * step_y)
                )

    def _on_selection_changed(self) -> None:
        if self._syncing_selection:
            return
        # Membership decides whether an item draws handles or a plain marker.
        for item in self._items:
            item.update()
        self.layerSelectionChanged.emit()


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
