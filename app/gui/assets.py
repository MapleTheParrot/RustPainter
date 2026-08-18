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

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QIcon, QPainter, QPixmap


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
