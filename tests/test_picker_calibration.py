from __future__ import annotations

import numpy as np
from PIL import Image

from app.models import ScreenRect
from app.painter import Painter, PaintingTarget
from app.input_controller import MockInputController
from app.picker_calibration import trim_to_widget


PANEL = (21, 21, 12)


def _picker_capture(rect: ScreenRect, inset: int) -> Image.Image:
    """A saturation/value box drawn ``inset`` pixels inside its capture."""

    pixels = np.zeros((rect.height, rect.width, 3), dtype=np.uint8)
    pixels[:, :] = PANEL
    width = rect.width - 2 * inset
    height = rect.height - 2 * inset
    saturation = np.linspace(0.0, 1.0, width)[None, :, None]
    value = np.linspace(1.0, 0.0, height)[:, None, None]
    hue = np.array([1.0, 0.35, 0.0])[None, None, :]
    gradient = (1.0 - saturation) + saturation * hue
    pixels[inset : inset + height, inset : inset + width] = np.rint(
        gradient * value * 255.0
    ).astype(np.uint8)
    return Image.fromarray(pixels, mode="RGB")


def test_overshooting_rectangle_is_trimmed_to_the_widget() -> None:
    # A rectangle dragged one pixel wide on every side sends saturation 0 and
    # value 1 onto the panel, where Rust ignores the click entirely.
    rect = ScreenRect(2048, 760, 314, 317)

    trimmed = trim_to_widget(_picker_capture(rect, inset=1), rect)

    assert trimmed == ScreenRect(2049, 761, 312, 315)


def test_a_tight_rectangle_is_left_exactly_as_calibrated() -> None:
    rect = ScreenRect(2048, 760, 314, 317)

    assert trim_to_widget(_picker_capture(rect, inset=0), rect) == rect


def test_a_capture_without_a_picker_is_not_trimmed() -> None:
    # Rust showing anything but the painting UI must never shrink calibration.
    rect = ScreenRect(2048, 760, 314, 317)
    blank = Image.new("RGB", (rect.width, rect.height), PANEL)

    assert trim_to_widget(blank, rect) == rect


def test_a_sliver_of_content_is_rejected_rather_than_trusted() -> None:
    rect = ScreenRect(2048, 760, 314, 317)
    pixels = np.zeros((rect.height, rect.width, 3), dtype=np.uint8)
    pixels[:, :] = PANEL
    pixels[150:160, 100:110] = (255, 255, 255)

    assert trim_to_widget(Image.fromarray(pixels, mode="RGB"), rect) == rect


def test_a_mismatched_capture_is_refused() -> None:
    rect = ScreenRect(2048, 760, 314, 317)
    try:
        trim_to_widget(Image.new("RGB", (10, 10), PANEL), rect)
    except ValueError:
        return
    raise AssertionError("A capture of the wrong size must not be measured")


def test_the_painter_paints_through_the_measured_rectangles() -> None:
    # Profiles already on disk have to be corrected without recalibration, so
    # the measurement happens at paint time rather than at calibration time.
    box = ScreenRect(2048, 760, 314, 317)
    bar = ScreenRect(2365, 763, 51, 312)
    target = PaintingTarget(
        canvas=ScreenRect(100, 100, 400, 300), color_box=box, hue_bar=bar
    )
    controller = MockInputController()
    controller.emits_real_input = True

    def capture(rect) -> Image.Image:
        return _picker_capture(ScreenRect(rect.left, rect.top, rect.width, rect.height), 1)

    measured = Painter(controller, screen_capture=capture)._measured_picker_target(target)

    assert measured.color_box == ScreenRect(2049, 761, 312, 315)
    assert measured.hue_bar == ScreenRect(2366, 764, 49, 310)
    assert measured.canvas == target.canvas


def test_a_capture_failure_leaves_the_calibration_untouched() -> None:
    target = PaintingTarget(
        canvas=ScreenRect(100, 100, 400, 300),
        color_box=ScreenRect(2048, 760, 314, 317),
        hue_bar=ScreenRect(2365, 763, 51, 312),
    )
    controller = MockInputController()
    controller.emits_real_input = True

    def capture(_rect) -> Image.Image:
        raise OSError("screen capture unavailable")

    measured = Painter(controller, screen_capture=capture)._measured_picker_target(target)

    assert measured == target
