from __future__ import annotations

import numpy as np
from PIL import Image

from app.models import ScreenRect
from app.setup_detection import detect_painting_setup


def _add_adaptive_picker(
    pixels: np.ndarray,
    *,
    box_left: int = 900,
    box_top: int = 300,
    box_size: int = 156,
) -> None:
    hue_width = 26
    for y in range(box_size):
        value = 1.0 - y / (box_size - 1)
        for x in range(box_size):
            saturation = x / (box_size - 1)
            base = np.array([255, 45, 35], dtype=np.float32)
            rgb = (255.0 * (1.0 - saturation) + base * saturation) * value
            pixels[box_top + y, box_left + x] = np.clip(rgb, 0, 255)
    hsv = np.zeros((box_size, hue_width, 3), dtype=np.uint8)
    hsv[:, :, 0] = np.linspace(0, 255, box_size, dtype=np.uint8)[:, None]
    hsv[:, :, 1:] = 255
    hue = Image.fromarray(hsv, "HSV").convert("RGB")
    hue_left = box_left + box_size + 3
    pixels[
        box_top : box_top + box_size,
        hue_left : hue_left + hue_width,
    ] = np.asarray(hue)


def _painting_screen() -> tuple[Image.Image, ScreenRect]:
    width, height = 1280, 720
    pixels = np.full((height, width, 3), 34, dtype=np.uint8)
    # Bright, lightly textured sign on the left.
    pixels[90:640, 180:780] = 188
    pixels[90:640:7, 180:780] = 181
    _add_adaptive_picker(pixels)
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


def test_detects_adaptive_picker_when_low_scale_places_it_near_screen_center() -> None:
    pixels = np.full((720, 1280, 3), 34, dtype=np.uint8)
    pixels[100:620, 100:450] = 188
    _add_adaptive_picker(pixels, box_left=480)

    result = detect_painting_setup(
        Image.fromarray(pixels, "RGB"), ScreenRect(0, 0, 1280, 720)
    )

    assert result.missing_required == ()
    assert result.regions["hue_bar"].rect.left < 1280 * 0.55


def test_canvas_detection_uses_inner_material_instead_of_metal_frame() -> None:
    pixels = np.full((720, 1280, 3), 34, dtype=np.uint8)
    # A connected, bright-enough metal surround would enlarge the old
    # brightness-component bounding box beyond the paintable brown panel.
    pixels[80:635, 135:815] = (76, 81, 84)
    pixels[105:610, 170:780] = (151, 116, 82)
    pixels[105:610:9, 170:780] = (145, 109, 77)
    pixels[80:92, 180:760:90] = (190, 194, 196)
    pixels[623:635, 180:760:100] = (190, 194, 196)
    _add_adaptive_picker(pixels)

    result = detect_painting_setup(
        Image.fromarray(pixels, "RGB"), ScreenRect(0, 0, 1280, 720)
    )

    canvas = result.regions["canvas"].rect
    assert abs(canvas.left - 170) <= 7
    assert abs(canvas.top - 105) <= 7
    assert abs(canvas.right - 780) <= 7
    assert abs(canvas.bottom - 610) <= 7
    assert "rectangular" in result.regions["canvas"].method


def test_canvas_detection_discards_sparse_hanging_threads() -> None:
    pixels = np.full((720, 1280, 3), 34, dtype=np.uint8)
    pixels[80:560, 110:790] = (184, 180, 171)
    pixels[80:560:11, 110:790] = (177, 173, 165)
    # Same-material threads are attached to the canvas, so colour and connected
    # components alone cannot distinguish them. Their row coverage can.
    for x, length in ((190, 35), (420, 65), (615, 45), (740, 28)):
        pixels[560 : 560 + length, x - 2 : x + 3] = (184, 180, 171)
    _add_adaptive_picker(pixels)

    result = detect_painting_setup(
        Image.fromarray(pixels, "RGB"), ScreenRect(0, 0, 1280, 720)
    )

    canvas = result.regions["canvas"].rect
    assert abs(canvas.left - 110) <= 7
    assert abs(canvas.right - 790) <= 7
    assert abs(canvas.top - 80) <= 7
    assert abs(canvas.bottom - 560) <= 7
