from __future__ import annotations

from PIL import Image

from app.input_controller import MockInputController
from app.models import ScreenRect
from app.painting_view_setup import (
    PaintingViewCapture,
    fit_canvas_view,
    layout_is_safe,
)
from app.setup_detection import DetectedRegion, SetupDetection


SCREEN = ScreenRect(0, 0, 1000, 700)


def _capture(canvas: ScreenRect | None) -> PaintingViewCapture:
    regions = {
        "hue_bar": DetectedRegion(ScreenRect(860, 210, 20, 180), 0.97, "test"),
        "color_box": DetectedRegion(ScreenRect(660, 210, 180, 180), 0.91, "test"),
        "brush_size_box": DetectedRegion(ScreenRect(865, 50, 80, 35), 0.68, "test"),
        "clear_button": DetectedRegion(ScreenRect(15, 15, 40, 40), 0.66, "test"),
        "download_button": DetectedRegion(ScreenRect(70, 15, 40, 40), 0.64, "test"),
        "save_button": DetectedRegion(ScreenRect(680, 480, 180, 35), 0.62, "test"),
    }
    if canvas is not None:
        regions["canvas"] = DetectedRegion(canvas, 0.78, "test")
    return PaintingViewCapture(
        Image.new("RGB", (SCREEN.width, SCREEN.height)),
        SCREEN,
        SetupDetection(regions),
    )


def test_incomplete_layout_never_emits_wheel_input() -> None:
    initial = _capture(None)
    controller = MockInputController()

    result = fit_canvas_view(controller, lambda: initial, initial, settle=lambda _s: None)

    assert result.zoom_steps == 0
    assert "missing canvas" in result.reason
    assert controller.events == []


def test_fit_keeps_last_safe_zoom_and_undoes_unsafe_step() -> None:
    initial = _capture(ScreenRect(120, 100, 300, 250))
    enlarged = _capture(ScreenRect(105, 85, 330, 275))
    overlaps_picker = _capture(ScreenRect(70, 55, 650, 540))
    observations = iter((enlarged, overlaps_picker))
    controller = MockInputController()

    result = fit_canvas_view(
        controller,
        lambda: next(observations),
        initial,
        max_steps=4,
        settle=lambda _s: None,
    )

    assert result.capture is enlarged
    assert result.zoom_steps == 1
    assert "overlaps" in result.reason
    assert [event.value for event in controller.events if event.kind == "wheel"] == [
        120,
        120,
        -120,
    ]


def test_fit_probes_the_other_wheel_direction_when_needed() -> None:
    initial = _capture(ScreenRect(120, 100, 300, 250))
    smaller = _capture(ScreenRect(135, 112, 270, 225))
    enlarged = _capture(ScreenRect(105, 85, 330, 275))
    observations = iter((smaller, enlarged))
    controller = MockInputController()

    result = fit_canvas_view(
        controller,
        lambda: next(observations),
        initial,
        max_steps=1,
        settle=lambda _s: None,
    )

    assert result.capture is enlarged
    assert result.zoom_steps == -1
    assert [event.value for event in controller.events if event.kind == "wheel"] == [
        120,
        -120,
        -120,
    ]


def test_layout_rejects_a_clipped_canvas_and_changed_aspect() -> None:
    clipped = _capture(ScreenRect(2, 100, 300, 250))
    valid, reason = layout_is_safe(clipped)
    assert not valid
    assert "screen edge" in reason

    changed = _capture(ScreenRect(120, 100, 360, 250))
    valid, reason = layout_is_safe(changed, expected_aspect=1.2)
    assert not valid
    assert "aspect ratio" in reason
