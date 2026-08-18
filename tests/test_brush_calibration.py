from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.brush_calibration import measure_brush_footprint


def _preview(size: tuple[int, int] = (120, 120)) -> Image.Image:
    rng = np.random.default_rng(42)
    noise = rng.integers(-5, 6, (size[1], size[0], 1), dtype=np.int16)
    background = np.full((size[1], size[0], 3), 72, dtype=np.int16) + noise
    return Image.fromarray(np.clip(background, 0, 255).astype(np.uint8), "RGB")


def test_measure_centered_square_brush_on_textured_preview() -> None:
    image = _preview()
    draw = ImageDraw.Draw(image)
    draw.rectangle((51, 52, 68, 67), fill=(210, 30, 230))

    footprint = measure_brush_footprint(image)

    assert footprint.left == 51
    assert footprint.top == 52
    assert footprint.width == 18
    assert footprint.height == 16
    assert footprint.diameter == 18
    assert footprint.confidence > 0.95


def test_measure_dark_circle_brush() -> None:
    image = _preview((100, 100))
    draw = ImageDraw.Draw(image)
    draw.ellipse((42, 42, 57, 57), fill=(5, 5, 8))

    footprint = measure_brush_footprint(image)

    assert 15 <= footprint.width <= 16
    assert 15 <= footprint.height <= 16


def test_measure_rejects_preview_without_brush_shape() -> None:
    with pytest.raises(ValueError, match="No brush shape"):
        measure_brush_footprint(_preview())
