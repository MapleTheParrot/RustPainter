from __future__ import annotations

import os

import pytest
from PIL import Image, ImageDraw

from app.brush_calibration import (
    BrushResponse,
    BrushResponseSet,
    build_brush_response,
    prime_sweeps,
    probe_sites,
)
from app.input_controller import MockInputController
from app.models import ColorGroup, PaintPlan, ScreenRect, Stroke
from app.painter import Painter, PainterSettings, PainterState
from app.profiles import CalibrationProfile


CANVAS = ScreenRect(200, 150, 1400, 1100)
PROBES = 6
# Matches the exponent the painter uses to crowd the probes at the low end of
# the track, where a mis-set brush does the most damage.
FRACTIONS = tuple(round((index / (PROBES - 1)) ** 1.7, 4) for index in range(PROBES))

_TIMEOUT_SCALE = float(os.environ.get("RUST_PAINTER_TEST_TIMEOUT_SCALE", "1"))


def _profile() -> CalibrationProfile:
    return CalibrationProfile.new(
        "Measured",
        canvas=CANVAS,
        color_box=ScreenRect(1700, 150, 200, 200),
        hue_bar=ScreenRect(1950, 150, 20, 200),
        brush_slider=ScreenRect(1700, 400, 240, 20),
    )


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


def _canvas_capture(painted: dict[tuple[int, int], int]):
    """Serve any region over a probe as primed canvas with its dab at centre.

    The painter re-captures a tighter crop when a dab is too small to separate
    from the background, so this answers for whatever region it asks about
    rather than only the full patch.
    """

    def capture(rect):
        center = (rect.left + rect.width // 2, rect.top + rect.height // 2)
        for point, diameter in painted.items():
            if abs(center[0] - point[0]) <= 2 and abs(center[1] - point[1]) <= 2:
                image = Image.new("RGB", (rect.width, rect.height), (250, 250, 250))
                left = (rect.width - diameter) // 2
                top = (rect.height - diameter) // 2
                ImageDraw.Draw(image).ellipse(
                    (left, top, left + diameter - 1, top + diameter - 1),
                    fill=(10, 10, 10),
                )
                return image
        # Everything else the painter captures is the picker, which this test
        # keeps featureless so the calibration is used verbatim.
        return Image.new("RGB", (rect.width, rect.height), (21, 21, 12))

    return capture


def _measured_painter() -> tuple[Painter, MockInputController]:
    """A painter whose canvas paints a dab proportional to the Size track."""

    sites = probe_sites(CANVAS, PROBES)
    painted = {
        site.point: round(4 + fraction * 60)
        for site, fraction in zip(sites, FRACTIONS)
    }
    controller = MockInputController()
    return Painter(controller, screen_capture=_canvas_capture(painted)), controller


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


def test_priming_sweeps_cover_the_square() -> None:
    sweeps = prime_sweeps(ScreenRect(0, 0, 200, 200))
    rows = [start[1] for start, _ in sweeps]

    assert rows[0] == 0 and rows[-1] == 199
    assert max(later - earlier for earlier, later in zip(rows, rows[1:])) <= 6
    assert all(start[0] == 0 and end[0] == 199 for start, end in sweeps)


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
    painter, controller = _measured_painter()

    painter.configure_brush_measurement(_profile(), _settings(), probe_count=PROBES)
    assert painter.start()
    assert painter.wait(20.0 * _TIMEOUT_SCALE)

    assert painter.state is PainterState.COMPLETED
    measured = painter.brush_responses
    assert measured is not None
    # No shape buttons are calibrated, so one curve covers whatever brush Rust
    # has selected - which is the brush every pass will use.
    assert measured.shapes == (None,)
    curve = measured.for_shape(None)
    assert curve is not None and len(curve.samples) == PROBES
    for fraction, diameter in curve.samples:
        assert diameter == pytest.approx(round(4 + fraction * 60), abs=1.5)
    assert not controller.held_buttons


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
    plan = PaintPlan(
        70, 55, (ColorGroup((40, 80, 160), (Stroke(0, 0, 0, 0),), 1),)
    )

    assert painter.start(plan, profile, _settings(apply_brush_size=True))
    assert painter.wait(10.0 * _TIMEOUT_SCALE)

    assert painter.state is PainterState.COMPLETED
    assert captures == []
    # One 20px cell wants a 90% brush, which the curve reaches around a third
    # of the way along the track.
    slider_x = [
        event.x
        for event in controller.events
        if event.kind == "move" and 1700 <= event.x <= 1939 and 400 <= event.y <= 419
    ]
    assert slider_x, "the Size track was never clicked"
    fraction = (slider_x[-1] - 1700) / 239.0
    assert fraction == pytest.approx(0.24, abs=0.05)


def test_each_calibrated_shape_gets_its_own_curve() -> None:
    # A shape change can render a different footprint at the same slider
    # position, so a square measurement must never be reused for the circle.
    profile = _profile()
    profile.square_shape_button = ScreenRect(1700, 460, 40, 40)
    profile.circle_shape_button = ScreenRect(1750, 460, 40, 40)
    sites = probe_sites(CANVAS, PROBES * 2)
    shape_of = {}
    for index, site in enumerate(sites):
        shape_of[site.point] = "square" if index < PROBES else "circle"
    painted = {}
    for index, site in enumerate(sites):
        fraction = FRACTIONS[index % PROBES]
        # The circle reads two pixels narrower at every position.
        offset = 0 if shape_of[site.point] == "square" else -2
        painted[site.point] = round(6 + fraction * 60) + offset
    controller = MockInputController()
    painter = Painter(controller, screen_capture=_canvas_capture(painted))

    painter.configure_brush_measurement(profile, _settings(), probe_count=PROBES)
    assert painter.start()
    assert painter.wait(30.0 * _TIMEOUT_SCALE)

    assert painter.state is PainterState.COMPLETED
    measured = painter.brush_responses
    assert measured is not None
    assert measured.shapes == ("square", "circle")
    square = measured.for_shape("square")
    circle = measured.for_shape("circle")
    assert square is not None and circle is not None
    assert square.largest_diameter - circle.largest_diameter == pytest.approx(2, abs=1)
    # With both shapes on file there is no honest answer for an unknown shape.
    assert measured.for_shape(None) is None


def test_a_pass_is_sized_from_its_own_shape_curve() -> None:
    square = build_brush_response([(0.0, 4.0), (1.0, 64.0)], shape="square")
    circle = build_brush_response([(0.0, 4.0), (1.0, 24.0)], shape="circle")
    measured = BrushResponseSet((square, circle))

    # 18px wants a third of the square track but well past half the circle's.
    assert measured.for_shape("square").fraction_for(18.0) == pytest.approx(0.2333, abs=0.01)
    assert measured.for_shape("circle").fraction_for(18.0) == pytest.approx(0.7, abs=0.01)
