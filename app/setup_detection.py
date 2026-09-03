"""Detect Rust's painting controls from one monitor capture.

The adaptive hue bar is the anchor: its ordered rainbow is unusually distinct,
and the colour box and fixed controls have stable positions around it. Canvas
detection refines a loose bright component to its dense rectangular material
core, excluding sparse frames and hanging decoration. A dark or already-painted
sign is still left for the manual selector instead of returning a confident guess.
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
            name
            for name in (
                "canvas",
                "color_box",
                "hue_bar",
                "clear_button",
                "save_button",
                "download_button",
            )
            if name not in self.regions
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


def _narrow_hue_strip(crop: Image.Image) -> tuple[int, int] | None:
    """Return the longest run of columns that independently sweep the spectrum.

    At small UI scales, resizing a monitor capture can blend the thin hue strip
    into the saturated edge of the adjacent S/V square. Each hue-strip column
    still contains a complete ordered spectrum; the S/V square's columns do
    not. This separates the two without relying on an exact pixel gap.
    """

    candidates = np.array(
        [
            looks_like_hue_bar(
                crop.crop((column, 0, column + 1, crop.height)), reduced=True
            )
            for column in range(crop.width)
        ],
        dtype=bool,
    )
    run = _longest_run(candidates)
    if run is None or run[1] - run[0] < 2:
        return None
    return run


def _close_short_gaps(values: np.ndarray, maximum: int) -> np.ndarray:
    """Fill short false runs without joining genuinely separate edges."""

    closed = np.asarray(values, dtype=bool).copy()
    start = 0
    while start < len(closed):
        if closed[start]:
            start += 1
            continue
        end = start + 1
        while end < len(closed) and not closed[end]:
            end += 1
        if start > 0 and end < len(closed) and end - start <= maximum:
            closed[start:end] = True
        start = end
    return closed


def _longest_run(values: np.ndarray) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for index, value in enumerate(np.append(np.asarray(values, dtype=bool), False)):
        if value and start is None:
            start = index
        elif not value and start is not None:
            if best is None or index - start > best[1] - best[0]:
                best = (start, index)
            start = None
    return best


def _coverage_core(mask: np.ndarray) -> tuple[int, int, int, int, float] | None:
    """Find the dense rectangular core of a component-like material mask."""

    if mask.size == 0:
        return None
    left, top, right, bottom = 0, 0, mask.shape[1], mask.shape[0]
    for _ in range(2):
        current = mask[top:bottom, left:right]
        if current.size == 0:
            return None
        row_reference = float(np.percentile(current.mean(axis=1), 80))
        row_cutoff = max(0.62, row_reference * 0.76)
        rows = _close_short_gaps(
            current.mean(axis=1) >= row_cutoff,
            max(1, min(3, current.shape[0] // 100)),
        )
        row_run = _longest_run(rows)
        if row_run is None:
            return None
        top += row_run[0]
        bottom = top + row_run[1] - row_run[0]

        current = mask[top:bottom, left:right]
        column_reference = float(np.percentile(current.mean(axis=0), 80))
        column_cutoff = max(0.62, column_reference * 0.76)
        columns = _close_short_gaps(
            current.mean(axis=0) >= column_cutoff,
            max(1, min(3, current.shape[1] // 100)),
        )
        column_run = _longest_run(columns)
        if column_run is None:
            return None
        left += column_run[0]
        right = left + column_run[1] - column_run[0]

    core = mask[top:bottom, left:right]
    if core.size == 0:
        return None
    return left, top, right, bottom, float(core.mean())


def _rectangular_material_core(
    pixels: np.ndarray,
    brightness_mask: np.ndarray,
    box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], float] | None:
    """Refine a loose bright component to its consistent rectangular material."""

    left, top, width, height = box
    crop = pixels[top : top + height, left : left + width]
    bright = brightness_mask[top : top + height, left : left + width]
    if crop.size == 0:
        return None

    # The broad middle is overwhelmingly paintable material on an unpainted
    # sign. Its robust colour spread tolerates cloth/wood grain but rejects a
    # metal frame, hooks, shadows, and other differently shaded decoration.
    center = crop[
        height // 4 : max(height // 4 + 1, height - height // 4),
        width // 4 : max(width // 4 + 1, width - width // 4),
    ]
    samples = center.reshape(-1, 3)
    median = np.median(samples, axis=0)
    center_distance = np.linalg.norm(samples - median, axis=1)
    distance_limit = max(
        0.10,
        min(0.30, float(np.percentile(center_distance, 90)) * 1.8),
    )
    material = np.linalg.norm(crop - median, axis=2) <= distance_limit
    material &= bright

    refined = _coverage_core(material)
    if refined is None:
        return None
    core_left, core_top, core_right, core_bottom, density = refined
    core_width = core_right - core_left
    core_height = core_bottom - core_top
    if core_width < width * 0.55 or core_height < height * 0.55 or density < 0.68:
        return None
    return (
        (left + core_left, top + core_top, core_width, core_height),
        density,
    )


def _detect_hue_bar(
    working: Image.Image, scale: float, screen: ScreenRect
) -> DetectedRegion | None:
    hsv = np.asarray(working.convert("HSV"), dtype=np.uint8)
    saturated = (hsv[:, :, 1] > 140) & (hsv[:, :, 2] > 80)
    minimum = max(12, int(working.width * working.height * 0.00005))
    candidates = []
    for left, top, width, height, area in _components(saturated, minimum):
        # A 0.5 Rust UI scale on a high-resolution monitor can leave fewer
        # than 25 pixels after the monitor capture is reduced for analysis.
        if height < 12:
            continue
        density = area / max(1, width * height)
        if density < 0.35:
            continue
        crop = working.crop((left, top, left + width, top + height))
        # A normal hue strip is already narrow. Only split unusually broad
        # components, where downsampling has joined it to the colour square.
        strip = _narrow_hue_strip(crop) if width > height * 0.4 else None
        if not looks_like_hue_bar(crop, reduced=True) and strip is None:
            continue
        if strip is not None:
            strip_left, strip_right = strip
            left += strip_left
            width = strip_right - strip_left
        # The ordered-spectrum test is the position-independent discriminator.
        # At a low Rust UI scale the picker can sit near the screen centre, so
        # rejecting candidates outside an assumed right-side panel is unsafe.
        score = density + min(1.0, height / max(1.0, width * 8.0))
        candidates.append((score, (left, top, width, height)))
    if not candidates:
        return None
    _, box = max(candidates, key=lambda item: item[0])
    rect = _screen_rect(box, scale, screen)
    # Include the thin widget border that the saturated component omits. Do
    # not run the generic gradient trimmer here: it treats dark rainbow hues
    # as background and can shorten this vertical widget, which in turn makes
    # the derived square picker too small.
    border = 1
    expanded = ScreenRect(
        rect.left - border,
        rect.top - border,
        rect.width + 2 * border,
        rect.height + 2 * border,
    )
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
    for left, top, width, height, _area in _components(mask, minimum):
        if width < working.width * 0.12 or height < working.height * 0.12:
            continue
        if top < working.height * 0.02 and height < working.height * 0.25:
            continue
        refined = _rectangular_material_core(
            pixels, mask, (left, top, width, height)
        )
        if refined is None:
            continue
        core, density = refined
        core_left, core_top, core_width, core_height = core
        screen_share = core_width * core_height / max(
            1, working.width * working.height
        )
        rectangularity = core_width * core_height / max(1, width * height)
        score = screen_share * (0.6 + density) * (0.7 + 0.3 * rectangularity)
        # The brightness bridge grows every edge by two working pixels. An
        # equal inset returns the proposal to the detected material and adds a
        # small safety margin against unpaintable borders.
        box = (
            core_left + 2,
            core_top + 2,
            max(1, core_width - 4),
            max(1, core_height - 4),
        )
        choices.append((score, box, density, rectangularity))
    if not choices:
        return None
    _, box, density, rectangularity = max(choices, key=lambda item: item[0])
    confidence = min(
        0.91,
        0.58
        + max(0.0, density - 0.65) * 0.55
        + max(0.0, rectangularity - 0.60) * 0.25,
    )
    return DetectedRegion(
        _screen_rect(box, scale, screen),
        confidence,
        "consistent rectangular sign surface",
    )


def detect_painting_setup(image: Image.Image, screen: ScreenRect) -> SetupDetection:
    """Find the visible painting regions in a capture of ``screen``."""

    if image.size != (screen.width, screen.height):
        raise ValueError("Setup capture size does not match its screen rectangle")
    working, scale = _working_copy(image)
    hue = _detect_hue_bar(working, scale, screen)
    regions: dict[str, DetectedRegion] = {}
    picker_left = screen.right
    if hue is not None:
        color_box = _picker_box(image, hue.rect, screen)
        regions.update({"hue_bar": hue, "color_box": color_box})
        picker_left = color_box.rect.left
    canvas = _detect_canvas(working, scale, screen, picker_left)
    if canvas is not None:
        regions["canvas"] = canvas

    # The canvas can still be useful when the picker is obscured, rendered
    # too small, or simply changed by a Rust update. Show that partial result
    # for review instead of making the user start from nothing.
    if hue is None:
        return SetupDetection(regions)

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
