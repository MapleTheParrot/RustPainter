"""Access to the baked rust-theme artwork under assets/ui.

Everything the UI draws comes from :data:`ASSET_ROOT`, which resolves both in a
source checkout and inside the PyInstaller bundle. Pixmaps and icons are cached
because the same handful of images is requested from many widgets, and the
stylesheet needs forward-slash URLs regardless of platform.
"""

from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QColor, QIcon, QPainter, QPainterPath, QPen, QPixmap


def _asset_root() -> Path:
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        return Path(bundle) / "assets" / "ui"
    return Path(__file__).resolve().parent.parent.parent / "assets" / "ui"


ASSET_ROOT = _asset_root()


def asset_path(name: str) -> Path:
    """Return the on-disk path of a baked asset (with or without extension)."""

    return ASSET_ROOT / (name if name.endswith(".png") else f"{name}.png")


def asset_url(name: str) -> str:
    """Return a stylesheet-safe url() body for a baked asset."""

    return asset_path(name).as_posix()


@lru_cache(maxsize=None)
def pixmap(name: str, size: int = 0) -> QPixmap:
    """Load an asset, optionally scaled to a square edge length."""

    source = QPixmap(str(asset_path(name)))
    if size <= 0 or source.isNull():
        return source
    return source.scaled(
        size,
        size,
        Qt.AspectRatioMode.KeepAspectRatio,
        Qt.TransformationMode.SmoothTransformation,
    )


@lru_cache(maxsize=None)
def tinted_pixmap(name: str, color: str, size: int = 0) -> QPixmap:
    """Recolour an asset while keeping its alpha, for state-coloured icons.

    The artwork is uniformly rust-orange; states such as the green "Ready"
    marker reuse the same shapes in a different hue rather than shipping a
    second render of each one.
    """

    base = pixmap(name, size)
    if base.isNull():
        return base
    result = QPixmap(base.size())
    result.fill(Qt.GlobalColor.transparent)
    painter = QPainter(result)
    painter.drawPixmap(0, 0, base)
    painter.setCompositionMode(QPainter.CompositionMode.CompositionMode_SourceIn)
    painter.fillRect(result.rect(), color)
    painter.end()
    return result


@lru_cache(maxsize=None)
def icon(name: str, size: int = 64, color: str | None = None) -> QIcon:
    """Return a cached QIcon for a baked asset, optionally recoloured."""

    source = tinted_pixmap(name, color, size) if color else pixmap(name, size)
    return QIcon(source)


def icon_size(edge: int) -> QSize:
    return QSize(edge, edge)


@lru_cache(maxsize=None)
def vector_icon(name: str, size: int = 64, color: str = "#f27a22") -> QIcon:
    """Draw small semantic controls as resolution-independent line icons.

    These intentionally stay code-native: profile, folder, refresh, and delete
    marks are simple geometry and should remain crisp at every UI scale rather
    than borrowing a textured illustration whose silhouette means something
    else.
    """

    edge = max(16, int(size))
    canvas = QPixmap(edge, edge)
    canvas.fill(Qt.GlobalColor.transparent)
    painter = QPainter(canvas)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
    painter.scale(edge / 64.0, edge / 64.0)
    pen = QPen(QColor(color), 4.0)
    pen.setCapStyle(Qt.PenCapStyle.RoundCap)
    pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
    painter.setPen(pen)
    painter.setBrush(Qt.BrushStyle.NoBrush)

    if name == "profile-add":
        painter.drawRoundedRect(QRectF(7, 10, 38, 44), 5, 5)
        painter.drawEllipse(QPointF(26, 25), 6, 6)
        painter.drawArc(QRectF(15, 31, 22, 15), 0, 180 * 16)
        painter.drawLine(QPointF(48, 37), QPointF(60, 37))
        painter.drawLine(QPointF(54, 31), QPointF(54, 43))
    elif name == "folder-open":
        path = QPainterPath()
        path.moveTo(6, 19)
        path.lineTo(25, 19)
        path.lineTo(31, 25)
        path.lineTo(58, 25)
        path.lineTo(54, 52)
        path.lineTo(10, 52)
        path.closeSubpath()
        painter.drawPath(path)
        painter.drawLine(QPointF(7, 19), QPointF(7, 47))
    elif name == "refresh":
        painter.drawArc(QRectF(10, 10, 44, 44), 35 * 16, 285 * 16)
        painter.drawLine(QPointF(49, 11), QPointF(54, 25))
        painter.drawLine(QPointF(54, 25), QPointF(40, 22))
    elif name == "delete":
        painter.drawRoundedRect(QRectF(17, 19, 30, 36), 3, 3)
        painter.drawLine(QPointF(13, 17), QPointF(51, 17))
        painter.drawLine(QPointF(24, 10), QPointF(40, 10))
        painter.drawLine(QPointF(27, 27), QPointF(27, 47))
        painter.drawLine(QPointF(37, 27), QPointF(37, 47))

    painter.end()
    return QIcon(canvas)
