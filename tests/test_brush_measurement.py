from __future__ import annotations

import os

import pytest
from PIL import Image, ImageDraw

from app.brush_calibration import (
    BrushResponse,
    BrushResponseSet,
    build_brush_response,
    prime_spacing,
    prime_sweeps,
    probe_sites,
)
from app.input_controller import MockInputController
from app.models import ColorGroup, PaintPlan, ScreenRect, Stroke
from app.painter import Painter, PainterSettings, PainterState
from app.profiles import CalibrationProfile


CANVAS = ScreenRect(200, 150, 1400, 1100)
SLIDER = ScreenRect(1700, 400, 240, 20)
SHAPE_BUTTONS = {
    "square": ScreenRect(1700, 460, 40, 40),
    "circle": ScreenRect(1750, 460, 40, 40),
}
PROBES = 6
# Matches the exponent the painter uses to crowd the probes at the low end of
# the track, where a mis-sized brush does the most damage.
FRACTIONS = tuple(round((index / (PROBES - 1)) ** 1.7, 4) for index in range(PROBES))

_TIMEOUT_SCALE = float(os.environ.get("RUST_PAINTER_TEST_TIMEOUT_SCALE", "1"))


def _profile(*, shapes: bool = False) -> CalibrationProfile:
    profile = CalibrationProfile.new(
        "Measured",
        canvas=CANVAS,
        color_box=ScreenRect(1700, 150, 200, 200),
        hue_bar=ScreenRect(1950, 150, 20, 200),
        brush_slider=SLIDER,
    )
    if shapes:
        profile.square_shape_button = SHAPE_BUTTONS["square"]
        profile.circle_shape_button = SHAPE_BUTTONS["circle"]
    return profile


def _settings(**overrides: object) -> PainterSettings:
    values: dict[str, object] = {
        "countdown_seconds": 0.0,
        "mouse_down_duration_seconds": 0.0,
        "delay_after_hue_seconds": 0.0,
        "delay_after_saturation_value_seconds": 0.0,
        "delay_between_strokes_seconds": 0.0,
        "delay_between_colors_seconds": 0.0,
        "delay_after_brush_seconds": 0.0,
        "stroke_speed_pixels_per_second": 1_000_000.0,
        "stroke_interpolation_step_pixels": 4096.0,
        "corner_abort_enabled": False,
        "progress_callback_interval_seconds": 0.0,
        "safety_poll_interval_seconds": 0.002,
    }
    values.update(overrides)
    return PainterSettings(**values)  # type: ignore[arg-type]


def _canvas_capture(controller: MockInputController, shape_widths=None):
    """A tiny canvas model: replay the input log, then serve what it painted.

    Dabs are stamped long before their patches are captured, so answering from
    the Size track's *current* position would report the last probe everywhere.
    Walking the log instead records what each dab was actually painted with,
    which is what Rust's canvas does too.
    """

    def replay() -> dict[tuple[int, int], int]:
        position = (0, 0)
        fraction = 0.0
        shape: str | None = None
        dabs: dict[tuple[int, int], int] = {}
        for event in controller.events:
            if event.kind == "move":
                position = (event.x, event.y)
                if (
                    SLIDER.left <= event.x <= SLIDER.left + SLIDER.width - 1
                    and SLIDER.top <= event.y <= SLIDER.top + SLIDER.height - 1
                ):
                    fraction = (event.x - SLIDER.left) / float(SLIDER.width - 1)
                for name, button in SHAPE_BUTTONS.items():
                    if (
                        button.left <= event.x <= button.left + button.width - 1
                        and button.top <= event.y <= button.top + button.height - 1
                    ):
                        shape = name
            elif event.kind == "mouse_down":
                if (
                    CANVAS.left <= position[0] < CANVAS.left + CANVAS.width
                    and CANVAS.top <= position[1] < CANVAS.top + CANVAS.height
                ):
                    diameter = round(6 + fraction * 58)
                    if shape_widths:
                        diameter += shape_widths.get(shape, 0)
                    # Last dab at a point wins, exactly as paint does.
                    dabs[position] = diameter
        return dabs

    def capture(rect):
        center = (rect.left + rect.width // 2, rect.top + rect.height // 2)
        image = Image.new("RGB", (rect.width, rect.height), (250, 250, 250))
        for point, diameter in replay().items():
            if abs(point[0] - center[0]) <= 2 and abs(point[1] - center[1]) <= 2:
                left = (rect.width - diameter) // 2
                top = (rect.height - diameter) // 2
                ImageDraw.Draw(image).ellipse(
                    (left, top, left + diameter - 1, top + diameter - 1),
                    fill=(10, 10, 10),
                )
                break
        return image

    return capture


def _strokes(controller: MockInputController) -> int:
    return sum(1 for event in controller.events if event.kind == "mouse_down")


def test_probe_sites_stay_inside_the_canvas_and_apart() -> None:
    sites = probe_sites(CANVAS, PROBES)

    assert len(sites) == PROBES
    for site in sites:
        assert CANVAS.left <= site.prime.left
        assert site.prime.left + site.prime.width <= CANVAS.left + CANVAS.width
        assert CANVAS.top <= site.prime.top
        assert site.prime.top + site.prime.height <= CANVAS.top + CANVAS.height
    for index, site in enumerate(sites):
        for other in sites[index + 1 :]:
            apart = abs(site.point[0] - other.point[0]) >= site.prime.width or abs(
                site.point[1] - other.point[1]
            ) >= site.prime.height
            assert apart, "probe patches must not overlap"


def test_a_canvas_too_small_to_measure_is_refused_before_any_stroke() -> None:
    with pytest.raises(ValueError, match="too small"):
        probe_sites(ScreenRect(0, 0, 300, 120), PROBES)


def test_priming_sweeps_cover_the_square_at_any_spacing() -> None:
    for spacing in (6, 25, 51):
        sweeps = prime_sweeps(ScreenRect(0, 0, 200, 200), spacing)
        rows = [start[1] for start, _ in sweeps]

        assert rows[0] == 0 and rows[-1] == 199
        assert max(later - earlier for earlier, later in zip(rows, rows[1:])) <= spacing
        assert all(start[0] == 0 and end[0] == 199 for start, end in sweeps)


def test_sweep_spacing_follows_the_measured_brush() -> None:
    # A wide brush primes a patch in a handful of strokes; an unmeasurable one
    # falls back to sweeps close enough for any brush at all.
    assert prime_spacing(64.0) == 51
    assert prime_spacing(32.0) == 25
    assert prime_spacing(None) == 6
    assert prime_spacing(0.0) == 6
    assert len(prime_sweeps(ScreenRect(0, 0, 200, 200), prime_spacing(64.0))) == 5
    assert len(prime_sweeps(ScreenRect(0, 0, 200, 200), prime_spacing(None))) == 35


def test_the_curve_inverts_to_a_size_track_fraction() -> None:
    response = build_brush_response([(0.0, 6.0), (0.5, 30.0), (1.0, 64.0)])

    assert response.fraction_for(30.0) == pytest.approx(0.5)
    assert response.diameter_for(0.25) == pytest.approx(18.0)
    # Outside the measured range the answer is clamped, and the caller is left
    # to notice via largest_diameter rather than being handed a fiction.
    assert response.fraction_for(500.0) == pytest.approx(1.0)
    assert response.largest_diameter == 64.0


def test_a_track_that_does_not_change_the_brush_is_rejected() -> None:
    with pytest.raises(ValueError, match="did not change"):
        build_brush_response([(0.0, 12.0), (1.0, 12.5)])


def test_a_response_survives_a_round_trip_through_the_profile_document() -> None:
    response = build_brush_response([(0.0, 6.0), (0.5, 30.0), (1.0, 64.0)])

    assert BrushResponse.from_dict(response.to_dict()).samples == response.samples


def test_measuring_paints_probes_and_reads_the_brush_off_the_canvas() -> None:
    controller = MockInputController()
    painter = Painter(controller, screen_capture=_canvas_capture(controller))

    painter.configure_brush_measurement(_profile(), _settings(), probe_count=PROBES)
    assert painter.start()
    assert painter.wait(30.0 * _TIMEOUT_SCALE)

    assert painter.state is PainterState.COMPLETED
    measured = painter.brush_responses
    assert measured is not None
    # No shape buttons are calibrated, so one curve covers whatever brush Rust
    # has selected - which is the brush every pass will use.
    assert measured.shapes == (None,)
    curve = measured.for_shape(None)
    assert curve is not None and len(curve.samples) == PROBES
    for fraction, diameter in curve.samples:
        assert diameter == pytest.approx(round(6 + fraction * 58), abs=1.5)
    assert not controller.held_buttons


def test_the_run_sizes_the_widest_brush_before_priming_with_it() -> None:
    # Priming used to assume the worst about the brush and drag for a minute.
    # One dab at full size tells it the truth, and the sweeps space themselves.
    controller = MockInputController()
    painter = Painter(controller, screen_capture=_canvas_capture(controller))

    painter.configure_brush_measurement(_profile(), _settings(), probe_count=PROBES)
    assert painter.start()
    assert painter.wait(30.0 * _TIMEOUT_SCALE)

    assert painter.state is PainterState.COMPLETED
    # A 64px widest brush primes each of the six patches in five sweeps. The
    # worst-case spacing would have needed thirty-five apiece.
    assert _strokes(controller) < 80, "priming should not need hundreds of strokes"


def test_each_calibrated_shape_gets_its_own_curve() -> None:
    # A shape change can render a different footprint at the same slider
    # position, so a square measurement must never be reused for the circle.
    controller = MockInputController()
    painter = Painter(
        controller,
        screen_capture=_canvas_capture(controller, {"square": 0, "circle": -4}),
    )

    painter.configure_brush_measurement(
        _profile(shapes=True), _settings(), probe_count=PROBES
    )
    assert painter.start()
    assert painter.wait(40.0 * _TIMEOUT_SCALE)

    assert painter.state is PainterState.COMPLETED
    measured = painter.brush_responses
    assert measured is not None
    assert measured.shapes == ("square", "circle")
    square = measured.for_shape("square")
    circle = measured.for_shape("circle")
    assert square is not None and circle is not None
    assert square.largest_diameter - circle.largest_diameter == pytest.approx(4, abs=1.5)
    # With both shapes on file there is no honest answer for an unknown shape.
    assert measured.for_shape(None) is None


def test_a_pass_is_sized_from_its_own_shape_curve() -> None:
    square = build_brush_response([(0.0, 4.0), (1.0, 64.0)], shape="square")
    circle = build_brush_response([(0.0, 4.0), (1.0, 24.0)], shape="circle")
    measured = BrushResponseSet((square, circle))

    # 18px wants a third of the square track but well past half the circle's.
    assert measured.for_shape("square").fraction_for(18.0) == pytest.approx(
        0.2333, abs=0.01
    )
    assert measured.for_shape("circle").fraction_for(18.0) == pytest.approx(
        0.7, abs=0.01
    )


def test_a_measured_curve_replaces_the_preview_search_while_painting() -> None:
    # With a curve on the profile the painter must not capture the preview at
    # all: the whole point is that the tile answers the wrong question.
    profile = _profile()
    profile.metadata["brush_response"] = build_brush_response(
        [(0.0, 4.0), (0.5, 34.0), (1.0, 64.0)]
    ).to_dict()
    captures: list[object] = []

    def capture(rect):
        captures.append(rect)
        return Image.new("RGB", (rect.width, rect.height), (21, 21, 12))

    controller = MockInputController()
    painter = Painter(controller, screen_capture=capture)
    plan = PaintPlan(70, 55, (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),))

    assert painter.start(plan, profile, _settings(apply_brush_size=True))
    assert painter.wait(10.0 * _TIMEOUT_SCALE)

    assert painter.state is PainterState.COMPLETED
    assert captures == []
    # One 20px cell wants a 90% brush, which the curve reaches around a quarter
    # of the way along the track.
    slider_x = [
        event.x
        for event in controller.events
        if event.kind == "move"
        and SLIDER.left <= event.x <= SLIDER.left + SLIDER.width - 1
        and SLIDER.top <= event.y <= SLIDER.top + SLIDER.height - 1
    ]
    assert slider_x, "the Size track was never clicked"
    fraction = (slider_x[-1] - SLIDER.left) / float(SLIDER.width - 1)
    assert fraction == pytest.approx(0.24, abs=0.05)
