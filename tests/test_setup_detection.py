from __future__ import annotations

import numpy as np
from PIL import Image

from app.models import ScreenRect
from app.setup_detection import detect_painting_setup


def _painting_screen() -> tuple[Image.Image, ScreenRect]:
    width, height = 1280, 720
    pixels = np.full((height, width, 3), 34, dtype=np.uint8)
    # Bright, lightly textured sign on the left.
    pixels[90:640, 180:780] = 188
    pixels[90:640:7, 180:780] = 181
    # Adaptive S/V square and hue strip on the right.
    box_left, box_top, box_size = 900, 300, 156
    hue_width = 26
    for y in range(box_size):
        value = 1.0 - y / (box_size - 1)
        for x in range(box_size):
            saturation = x / (box_size - 1)
            base = np.array([255, 45, 35], dtype=np.float32)
            rgb = (255.0 * (1.0 - saturation) + base * saturation) * value
            pixels[box_top + y, box_left + x] = np.clip(rgb, 0, 255)
    hue = Image.fromarray(np.zeros((box_size, hue_width, 3), dtype=np.uint8), "RGB")
    hsv = np.zeros((box_size, hue_width, 3), dtype=np.uint8)
    hsv[:, :, 0] = np.linspace(0, 255, box_size, dtype=np.uint8)[:, None]
    hsv[:, :, 1:] = 255
    hue = Image.fromarray(hsv, "HSV").convert("RGB")
    pixels[box_top : box_top + box_size, box_left + box_size + 3 : box_left + box_size + 3 + hue_width] = np.asarray(hue)
    return Image.fromarray(pixels, "RGB"), ScreenRect(-200, 50, width, height)


def test_detects_adaptive_picker_canvas_and_infers_fixed_controls() -> None:
    image, screen = _painting_screen()
    result = detect_painting_setup(image, screen)

    assert result.missing_required == ()
    assert result.regions["hue_bar"].confidence > 0.9
    hue = result.regions["hue_bar"].rect
    assert abs(hue.left - (screen.left + 1059)) <= 4
    assert abs(hue.top - (screen.top + 300)) <= 4
    color_box = result.regions["color_box"].rect
    assert abs(color_box.left - (screen.left + 900)) <= 8
    assert abs(color_box.width - 156) <= 8
    canvas = result.regions["canvas"].rect
    assert abs(canvas.left - (screen.left + 180)) <= 8
    assert abs(canvas.top - (screen.top + 90)) <= 8
    assert {"brush_size_box", "clear_button", "download_button", "save_button"} <= result.regions.keys()


def test_basic_palette_does_not_produce_confident_setup() -> None:
    image = Image.new("RGB", (800, 600), (35, 35, 35))
    result = detect_painting_setup(image, ScreenRect(0, 0, 800, 600))
    assert result.regions == {}
    assert result.missing_required == ("canvas", "color_box", "hue_bar")
