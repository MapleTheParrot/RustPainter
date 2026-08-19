"""One text renderer for the layer editor and for the baked overlay.

The Source tab floats text layers as live graphics items while the Rust
preview and the paint plan bake those same layers into logical pixels.  Both
sides draw through :func:`draw_text` here, so a caption looks while it is
being dragged exactly the way the sign will receive it: the same line box, the
same gradient ramp, the same outline weight.

Every length in this module is in logical canvas pixels - the units a layer's
``font_size`` already uses.  The editor scales the whole item to the preview
instead of scaling the numbers, so nothing here has to know about zoom.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (
    QBrush,
    QColor,
    QFont,
    QFontMetricsF,
    QLinearGradient,
    QPainter,
    QPainterPath,
    QPen,
)

RGBColor = tuple[int, int, int]

# The ramps a gradient layer can run its two colors along.
GRADIENT_DIRECTIONS: tuple[str, ...] = ("vertical", "horizontal", "diagonal")

# An outline is drawn in logical pixels, so a few of them already reads as a
# heavy border on a small sign; the cap keeps one from swallowing the letters.
MAX_OUTLINE_WIDTH = 16


def _rgb(value: Any) -> RGBColor:
    red, green, blue = (int(channel) for channel in tuple(value)[:3])
    return (red, green, blue)


@dataclass(frozen=True, slots=True)
class TextStyle:
    """How a text layer is painted: its fill, its optional ramp, its outline."""

    color: RGBColor = (255, 255, 255)
    gradient: bool = False
    gradient_color: RGBColor = (255, 255, 255)
    gradient_direction: str = "vertical"
    outline_width: int = 0
    outline_color: RGBColor = (0, 0, 0)

    @classmethod
    def from_layer(cls, layer: Any) -> "TextStyle":
        """Read the style off any object with the layer attributes."""

        return cls(
            color=_rgb(getattr(layer, "color", (255, 255, 255))),
            gradient=bool(getattr(layer, "gradient", False)),
            gradient_color=_rgb(getattr(layer, "gradient_color", (255, 255, 255))),
            gradient_direction=str(getattr(layer, "gradient_direction", "vertical")),
            outline_width=int(getattr(layer, "outline_width", 0)),
            outline_color=_rgb(getattr(layer, "outline_color", (0, 0, 0))),
        )

    @property
    def outline(self) -> int:
        return min(max(int(self.outline_width), 0), MAX_OUTLINE_WIDTH)


def layer_font(layer: Any) -> QFont:
    """Build the font a layer describes, in logical canvas pixels."""

    font = QFont(str(getattr(layer, "font_family", "") or ""))
    font.setPixelSize(max(1, int(getattr(layer, "font_size", 24))))
    font.setBold(bool(getattr(layer, "bold", False)))
    font.setItalic(bool(getattr(layer, "italic", False)))
    return font


def text_lines(text: str) -> list[str]:
    """Split on any line ending a paste may have carried in."""

    return text.replace("\r\n", "\n").replace("\r", "\n").split("\n")


def _line_layout(text: str, font: QFont) -> tuple[QFontMetricsF, list[str], list[float]]:
    metrics = QFontMetricsF(font)
    lines = text_lines(text)
    return metrics, lines, [metrics.horizontalAdvance(line) for line in lines]


def text_size(text: str, font: QFont) -> tuple[float, float]:
    """The line box the text occupies, ignoring any outline around it.

    Height comes from the font rather than from the glyphs, so a caption does
    not hop up and down as the letters typed into it gain and lose descenders.
    """

    metrics, lines, widths = _line_layout(text, font)
    height = metrics.lineSpacing() * (len(lines) - 1) + metrics.ascent() + metrics.descent()
    return (max(widths, default=0.0), height)


def text_rect(text: str, font: QFont, center: QPointF) -> QRectF:
    """Where the line box lands when the text is centered on ``center``."""

    width, height = text_size(text, font)
    return QRectF(
        center.x() - width / 2.0, center.y() - height / 2.0, width, height
    )


def text_path(text: str, font: QFont, center: QPointF) -> QPainterPath:
    """The glyph outlines of centered, line-broken text."""

    metrics, lines, widths = _line_layout(text, font)
    spacing = metrics.lineSpacing()
    top = center.y() - text_size(text, font)[1] / 2.0
    baseline = top + metrics.ascent()
    path = QPainterPath()
    for line, width in zip(lines, widths):
        if line:
            path.addText(center.x() - width / 2.0, baseline, font, line)
        baseline += spacing
    return path


def _gradient_line(direction: str, rect: QRectF) -> tuple[float, float, float, float]:
    if direction == "horizontal":
        return (rect.left(), rect.top(), rect.right(), rect.top())
    if direction == "diagonal":
        return (rect.left(), rect.top(), rect.right(), rect.bottom())
    return (rect.left(), rect.top(), rect.left(), rect.bottom())


def fill_brush(style: TextStyle, rect: QRectF) -> QBrush:
    """The brush the letters are filled with over their own line box."""

    if not style.gradient:
        return QBrush(QColor(*style.color))
    # A zero-length ramp would paint the whole run in the first color.
    if rect.width() <= 0.0 or rect.height() <= 0.0:
        return QBrush(QColor(*style.color))
    gradient = QLinearGradient(*_gradient_line(style.gradient_direction, rect))
    gradient.setColorAt(0.0, QColor(*style.color))
    gradient.setColorAt(1.0, QColor(*style.gradient_color))
    return QBrush(gradient)


def draw_text(
    painter: QPainter,
    text: str,
    font: QFont,
    center: QPointF,
    style: TextStyle,
) -> None:
    """Draw one text layer centered on ``center`` in the painter's units."""

    if not text.strip():
        return
    path = text_path(text, font, center)
    if path.isEmpty():
        return
    painter.save()
    try:
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        outline = style.outline
        if outline > 0:
            # A stroke straddles the glyph edge, so asking for twice the width
            # leaves exactly the requested width standing outside the letter.
            pen = QPen(QColor(*style.outline_color))
            pen.setWidthF(outline * 2.0)
            pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
            pen.setCapStyle(Qt.PenCapStyle.RoundCap)
            painter.strokePath(path, pen)
        painter.fillPath(path, fill_brush(style, text_rect(text, font, center)))
    finally:
        painter.restore()
