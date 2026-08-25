"""Small, dependency-free OCR for calibrated Rust HUD number boxes.

The hunger and thirst readouts contain only decimal digits.  Keeping the
reader deliberately narrow makes it practical to ship without a full OCR
engine: the calibrated crop is separated from its border colour, glyphs are
split into connected components, and each glyph is compared with normalized
digit templates.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont


_NORMALIZED_SIZE = (28, 40)


def _normalize(mask: np.ndarray) -> np.ndarray:
    points = np.argwhere(mask)
    if not len(points):
        return np.zeros((_NORMALIZED_SIZE[1], _NORMALIZED_SIZE[0]), dtype=bool)
    top, left = points.min(axis=0)
    bottom, right = points.max(axis=0) + 1
    cropped = Image.fromarray((mask[top:bottom, left:right] * 255).astype(np.uint8))
    target_height = 34
    target_width = max(
        2,
        min(24, round(cropped.width * target_height / max(1, cropped.height))),
    )
    resized = cropped.resize((target_width, target_height), Image.Resampling.BILINEAR)
    canvas = Image.new("L", _NORMALIZED_SIZE, 0)
    canvas.paste(
        resized,
        ((_NORMALIZED_SIZE[0] - target_width) // 2, (_NORMALIZED_SIZE[1] - target_height) // 2),
    )
    return np.asarray(canvas) >= 96


@lru_cache(maxsize=1)
def _templates() -> dict[int, tuple[np.ndarray, ...]]:
    font = ImageFont.load_default(size=48)
    result: dict[int, tuple[np.ndarray, ...]] = {}
    for digit in range(10):
        variants: list[np.ndarray] = []
        for stroke_width in (0, 1):
            image = Image.new("L", (80, 80), 0)
            draw = ImageDraw.Draw(image)
            draw.text(
                (40, 40),
                str(digit),
                font=font,
                fill=255,
                anchor="mm",
                stroke_width=stroke_width,
                stroke_fill=255,
            )
            variants.append(_normalize(np.asarray(image) > 0))
        result[digit] = tuple(variants)
    return result


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    boxes: list[tuple[int, int, int, int, int]] = []
    for y, x in np.argwhere(mask):
        if seen[y, x]:
            continue
        stack = [(int(y), int(x))]
        seen[y, x] = True
        pixels: list[tuple[int, int]] = []
        while stack:
            py, px = stack.pop()
            pixels.append((py, px))
            for ny in range(max(0, py - 1), min(height, py + 2)):
                for nx in range(max(0, px - 1), min(width, px + 2)):
                    if mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
        ys, xs = zip(*pixels)
        boxes.append((min(xs), min(ys), max(xs) + 1, max(ys) + 1, len(pixels)))
    if not boxes:
        return []
    tallest = max(bottom - top for _left, top, _right, bottom, _area in boxes)
    useful = [
        box
        for box in boxes
        if box[3] - box[1] >= max(3, tallest * 0.55) and box[4] >= 3
    ]
    useful.sort(key=lambda box: box[0])
    return [mask[top:bottom, left:right] for left, top, right, bottom, _area in useful]


def _dilate(mask: np.ndarray) -> np.ndarray:
    image = Image.fromarray((mask * 255).astype(np.uint8))
    return np.asarray(image.filter(ImageFilter.MaxFilter(3))) > 0


def _classify(glyph: np.ndarray) -> tuple[int, float]:
    normalized = _normalize(glyph)
    dilated_glyph = _dilate(normalized)
    best_digit, best_score = 0, -1.0
    for digit, variants in _templates().items():
        for template in variants:
            dilated_template = _dilate(template)
            glyph_coverage = float((normalized & dilated_template).sum()) / max(
                1, int(normalized.sum())
            )
            template_coverage = float((template & dilated_glyph).sum()) / max(
                1, int(template.sum())
            )
            score = (glyph_coverage + template_coverage) / 2.0
            if score > best_score:
                best_digit, best_score = digit, score
    return best_digit, best_score


def _otsu(values: np.ndarray) -> float:
    clipped = np.clip(values, 0, 255).astype(np.uint8)
    histogram = np.bincount(clipped.ravel(), minlength=256).astype(np.float64)
    total = histogram.sum()
    if total <= 0:
        return 0.0
    weights = np.arange(256, dtype=np.float64)
    total_mean = float((histogram * weights).sum())
    background_weight = 0.0
    background_sum = 0.0
    best_variance = -1.0
    threshold = 0
    for level in range(256):
        background_weight += histogram[level]
        if background_weight <= 0:
            continue
        foreground_weight = total - background_weight
        if foreground_weight <= 0:
            break
        background_sum += level * histogram[level]
        background_mean = background_sum / background_weight
        foreground_mean = (total_mean - background_sum) / foreground_weight
        variance = background_weight * foreground_weight * (
            background_mean - foreground_mean
        ) ** 2
        if variance > best_variance:
            best_variance = variance
            threshold = level
    return float(threshold)


def read_number(image: Any) -> int | None:
    """Read an integer from a tightly calibrated HUD crop.

    ``None`` means the crop did not contain a confidently readable number.
    Both bright and dark glyphs are considered, as are coloured HUD digits.
    """

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    if rgb.ndim != 3 or rgb.shape[0] < 3 or rgb.shape[1] < 2:
        return None
    border = np.concatenate((rgb[0], rgb[-1], rgb[:, 0], rgb[:, -1]), axis=0)
    background = np.median(border, axis=0)
    colour_distance = np.sqrt(((rgb - background) ** 2).sum(axis=2))
    gray = rgb.mean(axis=2)
    candidates = [
        colour_distance > max(14.0, _otsu(colour_distance)),
        gray > max(float(np.percentile(gray, 65)), _otsu(gray)),
        gray < min(float(np.percentile(gray, 35)), _otsu(gray)),
    ]
    best: tuple[float, int] | None = None
    for mask in candidates:
        ratio = float(mask.mean())
        if not 0.005 <= ratio <= 0.65:
            continue
        glyphs = _components(mask)
        if not 1 <= len(glyphs) <= 4:
            continue
        digits: list[str] = []
        confidence = 1.0
        for glyph in glyphs:
            digit, score = _classify(glyph)
            digits.append(str(digit))
            confidence = min(confidence, score)
        if confidence >= 0.57 and (best is None or confidence > best[0]):
            best = confidence, int("".join(digits))
    return None if best is None else best[1]


__all__ = ["read_number"]
