"""Detect Rust's painting controls from one monitor capture.

The adaptive hue bar is the anchor: its ordered rainbow is unusually distinct,
and the colour box and fixed controls have stable positions around it.  Canvas
detection is deliberately conservative; a dark or already-painted sign is left
for the existing manual selector instead of returning a confident-looking guess.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from PIL import Image

from .models import ScreenRect
from .picker_calibration import trim_to_widget
from .ui_guard import looks_like_hue_bar


@dataclass(frozen=True, slots=True)
class DetectedRegion:
    rect: ScreenRect
    confidence: float
    method: str


@dataclass(frozen=True, slots=True)
class SetupDetection:
    regions: dict[str, DetectedRegion]

    @property
    def missing_required(self) -> tuple[str, ...]:
        return tuple(
            name for name in ("canvas", "color_box", "hue_bar") if name not in self.regions
        )


def _components(mask: np.ndarray, minimum: int) -> list[tuple[int, int, int, int, int]]:
    """Bounding boxes of 8-connected true regions as x, y, width, height, area."""

    height, width = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    found: list[tuple[int, int, int, int, int]] = []
    for y, x in zip(*np.nonzero(mask & ~seen)):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        left = right = int(x)
        top = bottom = int(y)
        area = 0
        while stack:
            cy, cx = stack.pop()
            area += 1
            left, right = min(left, cx), max(right, cx)
            top, bottom = min(top, cy), max(bottom, cy)
            for ny in range(max(0, cy - 1), min(height, cy + 2)):
                for nx in range(max(0, cx - 1), min(width, cx + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        if area >= minimum:
            found.append((left, top, right - left + 1, bottom - top + 1, area))
    return found


def _working_copy(image: Image.Image, limit: int = 900) -> tuple[Image.Image, float]:
    longest = max(image.size)
    if longest <= limit:
        return image.convert("RGB"), 1.0
    scale = limit / longest
    size = (max(1, round(image.width * scale)), max(1, round(image.height * scale)))
    return image.convert("RGB").resize(size, Image.Resampling.BILINEAR), scale


def _screen_rect(
    box: tuple[int, int, int, int], scale: float, origin: ScreenRect
) -> ScreenRect:
    left, top, width, height = box
    return ScreenRect(
        origin.left + round(left / scale),
        origin.top + round(top / scale),
        max(1, round(width / scale)),
        max(1, round(height / scale)),
    )


def _local_rect(rect: ScreenRect, origin: ScreenRect) -> ScreenRect:
    return ScreenRect(
        rect.left - origin.left, rect.top - origin.top, rect.width, rect.height
    )


def _detect_hue_bar(
    original: Image.Image, working: Image.Image, scale: float, screen: ScreenRect
) -> DetectedRegion | None:
    hsv = np.asarray(working.convert("HSV"), dtype=np.uint8)
    saturated = (hsv[:, :, 1] > 140) & (hsv[:, :, 2] > 80)
    minimum = max(12, int(working.width * working.height * 0.00005))
    candidates = []
    for left, top, width, height, area in _components(saturated, minimum):
        if height < 25 or height < width * 2.5:
            continue
        if left + width / 2.0 < working.width * 0.55:
            continue
        density = area / max(1, width * height)
        if density < 0.35:
            continue
        margin = max(1, round(width * 0.12))
        box = (
            max(0, left - margin),
            max(0, top - margin),
            min(working.width, left + width + margin),
            min(working.height, top + height + margin),
        )
        crop = working.crop(box)
        if not looks_like_hue_bar(crop):
            continue
        score = density + min(1.0, height / max(1.0, width * 8.0))
        candidates.append((score, (left, top, width, height)))
    if not candidates:
        return None
    _, box = max(candidates, key=lambda item: item[0])
    rect = _screen_rect(box, scale, screen)
    # Include the thin widget border that the saturated component omits, then
    # let the existing conservative trimmer settle on the actual gradient.
    border = 1
    expanded = ScreenRect(
        rect.left - border,
        rect.top - border,
        rect.width + 2 * border,
        rect.height + 2 * border,
    )
    local = _local_rect(expanded, screen)
    bounds = (local.left, local.top, local.right, local.bottom)
    if all(
        (bounds[0] >= 0, bounds[1] >= 0, bounds[2] <= original.width, bounds[3] <= original.height)
    ):
        expanded = trim_to_widget(original.crop(bounds), expanded)
    return DetectedRegion(expanded, 0.97, "ordered adaptive hue spectrum")


def _picker_box(original: Image.Image, hue: ScreenRect, screen: ScreenRect) -> DetectedRegion:
    # Current Rust UI: the S/V square shares the hue bar's vertical extent.
    # Deriving its width from height is more stable than the thin strip's width,
    # whose arrow markers can join or separate from the saturated component.
    width = max(8, hue.height)
    gap = max(1, round(hue.height * 0.01))
    guessed = ScreenRect(hue.left - gap - width, hue.top, width, hue.height)
    local = _local_rect(guessed, screen)
    if 0 <= local.left and 0 <= local.top and local.right <= original.width and local.bottom <= original.height:
        guessed = trim_to_widget(
            original.crop((local.left, local.top, local.right, local.bottom)), guessed
        )
    return DetectedRegion(guessed, 0.91, "adjacent to the detected hue bar")


def _detect_canvas(
    working: Image.Image, scale: float, screen: ScreenRect, picker_left: int
) -> DetectedRegion | None:
    pixels = np.asarray(working.convert("RGB"), dtype=np.float32) / 255.0
    value = pixels.max(axis=2)
    # Keep the painting side of the UI and drop the narrow top toolbar. A bare
    # sign is a single large, light component against Rust's charcoal editor.
    local_picker = max(1, round((picker_left - screen.left) * scale))
    limit = min(working.width, local_picker)
    if limit < working.width * 0.25:
        return None
    region = value[:, :limit]
    threshold = max(0.25, min(0.55, float(np.median(region)) + 0.10))
    mask = region >= threshold
    # Bridge texture grain and small painted marks without joining distant UI.
    for _ in range(2):
        expanded = mask.copy()
        expanded[1:] |= mask[:-1]
        expanded[:-1] |= mask[1:]
        expanded[:, 1:] |= mask[:, :-1]
        expanded[:, :-1] |= mask[:, 1:]
        mask = expanded
    minimum = max(30, int(working.width * working.height * 0.004))
    choices = []
    for left, top, width, height, area in _components(mask, minimum):
        if width < working.width * 0.12 or height < working.height * 0.12:
            continue
        if top < working.height * 0.02 and height < working.height * 0.25:
            continue
        density = area / max(1, width * height)
        screen_share = width * height / max(1, working.width * working.height)
        score = screen_share * (0.5 + min(1.0, density))
        choices.append((score, (left + 2, top + 2, max(1, width - 4), max(1, height - 4)), density))
    if not choices:
        return None
    _, box, density = max(choices, key=lambda item: item[0])
    confidence = min(0.88, 0.62 + max(0.0, density - 0.45) * 0.45)
    return DetectedRegion(_screen_rect(box, scale, screen), confidence, "large sign surface")


def detect_painting_setup(image: Image.Image, screen: ScreenRect) -> SetupDetection:
    """Find the visible painting regions in a capture of ``screen``."""

    if image.size != (screen.width, screen.height):
        raise ValueError("Setup capture size does not match its screen rectangle")
    working, scale = _working_copy(image)
    hue = _detect_hue_bar(image, working, scale, screen)
    if hue is None:
        return SetupDetection({})
    color_box = _picker_box(image, hue.rect, screen)
    regions: dict[str, DetectedRegion] = {
        "hue_bar": hue,
        "color_box": color_box,
    }
    canvas = _detect_canvas(working, scale, screen, color_box.rect.left)
    if canvas is not None:
        regions["canvas"] = canvas

    # Fixed UI controls are useful for automatic brush sizing and safety. They
    # are lower-confidence inferred proposals and are always shown for review.
    h = hue.rect.height
    regions["brush_size_box"] = DetectedRegion(
        ScreenRect(
            hue.rect.right + round(h * 0.105),
            hue.rect.top - round(h * 0.87),
            max(1, round(h * 0.239)),
            max(1, round(h * 0.140)),
        ),
        0.68,
        "current Rust UI layout",
    )
    regions["clear_button"] = DetectedRegion(
        ScreenRect(
            screen.left + round(screen.width * 0.012),
            screen.top + round(screen.height * 0.022),
            max(1, round(screen.width * 0.0383)),
            max(1, round(screen.height * 0.0646)),
        ),
        0.66,
        "current Rust toolbar layout",
    )
    regions["download_button"] = DetectedRegion(
        ScreenRect(
            screen.left + round(screen.width * 0.0652),
            screen.top + round(screen.height * 0.025),
            max(1, round(screen.width * 0.0332)),
            max(1, round(screen.height * 0.0604)),
        ),
        0.64,
        "current Rust toolbar layout",
    )
    regions["save_button"] = DetectedRegion(
        ScreenRect(
            color_box.rect.left - round(h * 0.057),
            color_box.rect.bottom + round(h * 0.60),
            max(1, round(h * 1.55)),
            max(1, round(h * 0.159)),
        ),
        0.62,
        "current Rust action layout",
    )
    return SetupDetection(regions)


__all__ = ["DetectedRegion", "SetupDetection", "detect_painting_setup"]
