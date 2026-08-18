"""Coordinate conversions for calibrated physical-screen rectangles."""

from __future__ import annotations

from math import floor
from typing import Protocol, TypeAlias

from .models import ScreenRect, Stroke


ScreenPoint: TypeAlias = tuple[float, float]


class RectangleLike(Protocol):
    left: int
    top: int
    width: int
    height: int


def _validate_rect(rect: RectangleLike) -> None:
    if rect.width <= 0 or rect.height <= 0:
        raise ValueError("Rectangle width and height must be positive")


def _validate_logical_size(width: int, height: int) -> None:
    if width <= 0 or height <= 0:
        raise ValueError("Logical width and height must be positive")


def logical_pixel_center(
    logical_x: int,
    logical_y: int,
    logical_width: int,
    logical_height: int,
    canvas: RectangleLike,
) -> ScreenPoint:
    """Map a logical pixel to the center of its calibrated canvas cell.

    The arithmetic deliberately stays in signed coordinates so it also works on
    monitors positioned to the left or above the Windows primary monitor.
    """

    _validate_rect(canvas)
    _validate_logical_size(logical_width, logical_height)
    if not 0 <= logical_x < logical_width or not 0 <= logical_y < logical_height:
        raise ValueError("Logical pixel is outside the logical image")

    x = canvas.left + ((logical_x + 0.5) / logical_width) * canvas.width
    y = canvas.top + ((logical_y + 0.5) / logical_height) * canvas.height
    return x, y


def logical_to_screen(
    logical_x: int,
    logical_y: int,
    logical_width: int,
    logical_height: int,
    canvas: RectangleLike,
) -> ScreenPoint:
    """Compatibility name for :func:`logical_pixel_center`."""

    return logical_pixel_center(
        logical_x, logical_y, logical_width, logical_height, canvas
    )


def logical_stroke_to_screen(
    stroke: Stroke,
    logical_width: int,
    logical_height: int,
    canvas: RectangleLike,
) -> tuple[ScreenPoint, ScreenPoint]:
    """Map both inclusive logical stroke endpoints to pixel centers."""

    return (
        logical_pixel_center(
            stroke.start_x,
            stroke.start_y,
            logical_width,
            logical_height,
            canvas,
        ),
        logical_pixel_center(
            stroke.end_x,
            stroke.end_y,
            logical_width,
            logical_height,
            canvas,
        ),
    )


def normalized_point(
    rect: RectangleLike, u: float, v: float, *, clamp: bool = True
) -> ScreenPoint:
    """Map normalized coordinates to safe physical pixels inside ``rect``.

    Here ``0`` and ``1`` mean the centers of the first and last physical pixels,
    rather than the exclusive rectangle boundary.  This is appropriate for
    clicking calibrated controls such as the hue and saturation/value regions.
    """

    _validate_rect(rect)
    if clamp:
        u = min(max(float(u), 0.0), 1.0)
        v = min(max(float(v), 0.0), 1.0)
    elif not 0.0 <= u <= 1.0 or not 0.0 <= v <= 1.0:
        raise ValueError("Normalized coordinates must be in the range [0, 1]")

    return (
        rect.left + u * max(rect.width - 1, 0),
        rect.top + v * max(rect.height - 1, 0),
    )


def clamp_to_rect(x: float, y: float, rect: RectangleLike) -> ScreenPoint:
    """Clamp a point to valid physical mouse coordinates inside ``rect``."""

    _validate_rect(rect)
    return (
        min(max(float(x), rect.left), rect.left + rect.width - 1),
        min(max(float(y), rect.top), rect.top + rect.height - 1),
    )


def screen_to_logical_pixel(
    screen_x: float,
    screen_y: float,
    logical_width: int,
    logical_height: int,
    canvas: RectangleLike,
    *,
    clamp: bool = False,
) -> tuple[int, int]:
    """Return the logical cell containing a physical screen point."""

    _validate_rect(canvas)
    _validate_logical_size(logical_width, logical_height)
    u = (screen_x - canvas.left) / canvas.width
    v = (screen_y - canvas.top) / canvas.height
    logical_x = floor(u * logical_width)
    logical_y = floor(v * logical_height)
    if clamp:
        logical_x = min(max(logical_x, 0), logical_width - 1)
        logical_y = min(max(logical_y, 0), logical_height - 1)
    elif not 0 <= logical_x < logical_width or not 0 <= logical_y < logical_height:
        raise ValueError("Screen point is outside the canvas")
    return logical_x, logical_y


__all__ = [
    "RectangleLike",
    "ScreenPoint",
    "ScreenRect",
    "clamp_to_rect",
    "logical_pixel_center",
    "logical_stroke_to_screen",
    "logical_to_screen",
    "normalized_point",
    "screen_to_logical_pixel",
]
