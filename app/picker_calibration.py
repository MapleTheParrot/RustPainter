"""Shrink calibrated picker rectangles to the widgets actually drawn inside them.

A hand-dragged rectangle overshoots the hue bar or the saturation/value box by
a pixel or two, which sounds harmless until you notice where the mapping sends
the extremes.  Saturation 0, saturation 1, value 0, value 1, and hue 0 degrees
all land on the exact edges of the calibrated rectangle, so a single pixel of
overshoot puts those clicks on the panel behind the widget.  Rust ignores them,
the color silently stays whatever was selected before, and every gray in an
image paints with a leftover color instead of gray.

Measuring the drawn widget removes that whole class of failure without asking
anyone to drag more precisely than a hand can.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .models import ScreenRect

if TYPE_CHECKING:
    from PIL.Image import Image


# How far a line's mean color may drift from the outermost line and still count
# as more of the same flat panel.
_PANEL_TOLERANCE = 8.0

# The mean per-channel jump that has to appear where the panel meets the widget.
# This is the whole trick: a gradient blends into its neighbouring line, so the
# black end of the value axis reads as a smooth continuation, while panel meets
# widget as a step.  Rust's dark panel against that same black edge is the
# narrowest real gap there is, at roughly eighteen levels.
_EDGE_STEP = 12.0

# A hand overshoots by a pixel or three.  Refusing to consider more than this
# keeps a misread from eating a real slice of the gradient.
_MAX_TRIM_FRACTION = 0.05

# Discarding more than half of either axis means the capture almost certainly
# did not show the picker, so the calibrated rectangle is kept as measured.
_MIN_RETAINED = 0.5


def _panel_lines(lines: np.ndarray) -> int:
    """How many lines at the outer end of ``lines`` are panel behind the widget.

    ``lines`` runs from the outer edge inward.  Returns 0 unless the run of
    panel-colored lines ends in a step, which is what separates an overshooting
    rectangle from one that simply starts on a dark part of the gradient.
    """

    limit = max(1, int(len(lines) * _MAX_TRIM_FRACTION))
    outer = lines[0].mean(axis=0)
    trimmed = 0
    while trimmed < limit and np.abs(
        lines[trimmed].mean(axis=0) - outer
    ).sum() <= _PANEL_TOLERANCE:
        trimmed += 1
    if trimmed == 0 or trimmed >= len(lines):
        return 0
    step = float(
        np.abs(lines[trimmed].astype(np.float64) - lines[trimmed - 1]).mean()
    )
    return trimmed if step >= _EDGE_STEP else 0


def trim_to_widget(image: "Image", rect: ScreenRect) -> ScreenRect:
    """Return ``rect`` shrunk to the picker widget visible in ``image``.

    ``image`` must be a capture of exactly ``rect``.  The rectangle is returned
    unchanged whenever the measurement is not clearly trustworthy: too small a
    region to judge, no panel found, or so much trimmed that the capture cannot
    have been showing the picker at all.
    """

    if rect.width < 8 or rect.height < 8:
        return rect
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)
    if pixels.shape[:2] != (rect.height, rect.width):
        raise ValueError("Capture size does not match the calibrated rectangle")

    top = _panel_lines(pixels)
    bottom = _panel_lines(pixels[::-1])
    if top + bottom >= rect.height:
        return rect
    # Columns are measured on the rows that survived, so a panel corner cannot
    # drag a column's mean toward the panel color.
    rows = pixels[top : rect.height - bottom]
    left = _panel_lines(rows.transpose(1, 0, 2))
    right = _panel_lines(rows.transpose(1, 0, 2)[::-1])
    if left + right >= rect.width:
        return rect

    width = rect.width - left - right
    height = rect.height - top - bottom
    if width < rect.width * _MIN_RETAINED or height < rect.height * _MIN_RETAINED:
        return rect
    return ScreenRect(rect.left + left, rect.top + top, width, height)


__all__ = ["trim_to_widget"]
