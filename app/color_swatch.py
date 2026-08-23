"""Reading the selected color back off Rust's color panel.

Right of the hue bar the painting UI draws a large flat block in whatever
color is selected - the color the next stroke will paint in - with its hex
code beneath.  The painter picks colors with two blind clicks, one on the
hue bar and one on the saturation / value box, and a click the game
swallows leaves the color where it was: the whole group then goes down in
the previous group's color, which on a fine palette reads as a hole.  That
block is the receipt for the clicks.  It is unlit, unblurred and
pixel-flat, so one small capture after the clicks says whether the color
took, with no model of the sign's rendering in between.

The block is not calibrated by hand: it is found next to the hue bar while
a known vivid color is selected, as the flat patch of that color starting
at the bar's top edge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .models import ScreenRect

RGBColor = tuple[int, int, int]

# The color selected while looking for the block: saturated and mid-bright,
# unlike the dark panel around the block and unlike anything a plan is
# likely to have left selected.
LOCATOR_COLOR: RGBColor = (255, 128, 0)

# Pixels within this much per channel of the seed count as the block.
_FLAT_TOLERANCE = 14
# A row or column joins the block when this fraction of it is flat.
_FLAT_FRACTION = 0.92
# The block must be at least this much of the hue bar, in each direction, to
# be believed; the seed is expected to sit on it one bar-width to the right.
_MIN_WIDTH_FRACTION = 0.5
_MIN_HEIGHT_FRACTION = 0.25
# The block's border and rounded corners are left out of the reading.
_READ_INSET = 0.15
# How far the block may read from the color it is showing and still be that
# color.  The picker quantises to its own pixels (a 313 px hue bar is 1.15
# degrees a pixel, which moves a saturated red about 3 sRGB units), and the
# readings of seven colors landed within 13; a color group that is this
# close to the one asked for is invisible on the sign.
MATCH_TOLERANCE = 22.0
# A block that reads with more spread than this is not flat: something is
# drawn over it, or the rectangle is off it.
MAX_SPREAD = 12.0


@dataclass(frozen=True, slots=True)
class SwatchReading:
    color: RGBColor
    # The 90th-percentile per-channel deviation from the median, over the
    # read area: 0 on a flat block.
    spread: float

    def distance_to(self, color: RGBColor) -> float:
        return float(
            np.linalg.norm(np.asarray(self.color, dtype=np.float64) - np.asarray(color))
        )

    def matches(self, color: RGBColor, tolerance: float = MATCH_TOLERANCE) -> bool:
        return self.spread <= MAX_SPREAD and self.distance_to(color) <= tolerance

    @property
    def hex(self) -> str:
        return "#%02X%02X%02X" % self.color


def search_region(hue_bar: ScreenRect) -> ScreenRect:
    """Where the block can be: right of the hue bar, within its height."""

    return ScreenRect(
        hue_bar.left + hue_bar.width,
        hue_bar.top - max(2, hue_bar.height // 50),
        max(8, hue_bar.width * 4),
        hue_bar.height + 2 * max(2, hue_bar.height // 50),
    )


def locate_swatch(
    capture: Callable[[ScreenRect], object],
    hue_bar: ScreenRect,
    selected: RGBColor = LOCATOR_COLOR,
) -> ScreenRect | None:
    """Find the selected-color block beside ``hue_bar`` while ``selected`` is up.

    Returns the area to read it from - the block inset from its edges - or
    None when no flat block of about that color is there.
    """

    region = search_region(hue_bar)
    image = capture(region)
    pixels = np.asarray(image.convert("RGB"), dtype=np.int16)  # type: ignore[attr-defined]
    if pixels.shape[:2] != (region.height, region.width):
        raise ValueError("Capture size does not match the search region")
    seed_x = min(region.width - 1, hue_bar.width)
    seed_y = min(region.height - 1, int(region.height * 0.3))
    seed = pixels[seed_y, seed_x]
    if np.abs(seed - np.asarray(selected, dtype=np.int16)).max() > 2 * _FLAT_TOLERANCE + 20:
        # Not the selected color under the seed: the block is elsewhere, or
        # the color never took.  The seed is tried a little further in too.
        seed_x = min(region.width - 1, int(hue_bar.width * 1.6))
        seed = pixels[seed_y, seed_x]
        if np.abs(seed - np.asarray(selected, dtype=np.int16)).max() > 2 * _FLAT_TOLERANCE + 20:
            return None
    flat = (np.abs(pixels - seed).max(axis=2) <= _FLAT_TOLERANCE)
    left, right = _extent(flat[seed_y], seed_x)
    rows = flat[:, left : right + 1].mean(axis=1) >= _FLAT_FRACTION
    top, bottom = _extent(rows, seed_y)
    cols = flat[top : bottom + 1, :].mean(axis=0) >= _FLAT_FRACTION
    left, right = _extent(cols, seed_x)
    width = right - left + 1
    height = bottom - top + 1
    if width < hue_bar.width * _MIN_WIDTH_FRACTION or height < hue_bar.height * _MIN_HEIGHT_FRACTION:
        return None
    inset_x = int(round(width * _READ_INSET))
    inset_y = int(round(height * _READ_INSET))
    return ScreenRect(
        region.left + left + inset_x,
        region.top + top + inset_y,
        max(1, width - 2 * inset_x),
        max(1, height - 2 * inset_y),
    )


def _extent(line: np.ndarray, start: int) -> tuple[int, int]:
    """The run of True values in ``line`` containing ``start``."""

    low = start
    while low > 0 and line[low - 1]:
        low -= 1
    high = start
    while high < len(line) - 1 and line[high + 1]:
        high += 1
    return low, high


def read_swatch(capture: Callable[[ScreenRect], object], swatch: ScreenRect) -> SwatchReading:
    image = capture(swatch)
    pixels = np.asarray(image.convert("RGB"), dtype=np.float32).reshape(-1, 3)  # type: ignore[attr-defined]
    median = np.median(pixels, axis=0)
    deviation = np.abs(pixels - median).max(axis=1)
    spread = float(np.percentile(deviation, 90)) if len(deviation) else 0.0
    color = tuple(int(round(float(v))) for v in median)
    return SwatchReading(color=color, spread=spread)  # type: ignore[arg-type]


__all__ = [
    "LOCATOR_COLOR",
    "MATCH_TOLERANCE",
    "MAX_SPREAD",
    "SwatchReading",
    "locate_swatch",
    "read_swatch",
    "search_region",
]
