from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.brush_calibration import (
    BRUSH_SIZE_MAX,
    BRUSH_SIZE_MIN,
    BrushSizeModel,
    canonical_texture_rows,
    fit_brush_size_model,
    measure_stroke_band,
)


def _canvas(size: tuple[int, int] = (400, 200)) -> Image.Image:
    """A lit, grainy sign surface - never a flat color in the real capture."""

    rng = np.random.default_rng(42)
    noise = rng.integers(-6, 7, (size[1], size[0], 1), dtype=np.int16)
    surface = np.full((size[1], size[0], 3), 96, dtype=np.int16) + noise
    return Image.fromarray(np.clip(surface, 0, 255).astype(np.uint8), "RGB")


def _with_band(
    canvas: Image.Image,
    top: int,
    height: int,
    *,
    left: int = 60,
    right: int = 340,
    color: tuple[int, int, int] = (255, 0, 255),
) -> Image.Image:
    painted = canvas.copy()
    draw = ImageDraw.Draw(painted)
    draw.rectangle((left, top, right, top + height - 1), fill=color)
    return painted


def test_measures_the_thickness_of_a_painted_band() -> None:
    before = _canvas()
    after = _with_band(before, top=90, height=21)

    band = measure_stroke_band(before, after)

    assert band.height == pytest.approx(21.0)
    assert band.top == 90
    assert not band.clipped


def test_round_stroke_caps_do_not_inflate_the_measurement() -> None:
    """A circular brush tapers at both ends; the straight middle is the answer."""

    before = _canvas()
    after = before.copy()
    draw = ImageDraw.Draw(after)
    # A capsule: a 20px band with semicircular ends, the shape a round brush
    # dragged horizontally actually leaves behind.
    draw.rectangle((70, 90, 330, 109), fill=(0, 255, 0))
    draw.ellipse((60, 90, 79, 109), fill=(0, 255, 0))
    draw.ellipse((321, 90, 340, 109), fill=(0, 255, 0))

    band = measure_stroke_band(before, after)

    assert band.height == pytest.approx(20.0)


def test_band_touching_an_edge_reports_itself_as_clipped() -> None:
    before = _canvas()
    after = _with_band(before, top=0, height=200)

    band = measure_stroke_band(before, after)

    assert band.clipped


def test_unchanged_capture_is_rejected_rather_than_measured() -> None:
    canvas = _canvas()

    with pytest.raises(ValueError, match="did not change"):
        measure_stroke_band(canvas, canvas.copy())


def test_mismatched_captures_are_rejected() -> None:
    with pytest.raises(ValueError, match="identical dimensions"):
        measure_stroke_band(_canvas(), _canvas((300, 200)))


def test_fit_recovers_the_size_to_fraction_relationship() -> None:
    # A sign 128 texture rows tall: size N covers N/128 of it.
    samples = [(size, size / 128.0) for size in (60, 30, 12)]

    model = fit_brush_size_model(samples)

    assert model.slope == pytest.approx(1 / 128.0)
    assert model.intercept == pytest.approx(0.0, abs=1e-9)
    assert model.sign_pixel_rows == pytest.approx(128.0)
    assert model.size_for_fraction(10 / 128.0) == pytest.approx(10.0)


def test_fit_needs_two_distinct_sizes() -> None:
    with pytest.raises(ValueError, match="two different Size values"):
        fit_brush_size_model([(30, 0.2), (30, 0.2)])


def test_fit_rejects_a_brush_that_never_grew() -> None:
    """Digits that never reached the field leave every probe the same width."""

    with pytest.raises(ValueError, match="did not grow"):
        fit_brush_size_model([(12, 0.1), (30, 0.1), (60, 0.1)])


def test_requested_size_is_held_inside_the_field_range() -> None:
    model = fit_brush_size_model([(size, size / 128.0) for size in (60, 30, 12)])

    assert model.clamped_size_for_fraction(0.0001) == BRUSH_SIZE_MIN
    assert model.clamped_size_for_fraction(10.0) == BRUSH_SIZE_MAX
    assert model.clamped_size_for_fraction(25 / 128.0) == 25


def test_model_survives_a_round_trip_through_a_profile() -> None:
    model = fit_brush_size_model([(60, 0.47), (12, 0.094)])

    restored = BrushSizeModel.from_dict(model.to_dict())

    assert restored.slope == pytest.approx(model.slope)
    assert restored.intercept == pytest.approx(model.intercept)
    assert restored.samples == model.samples
    assert restored.captured_at == model.captured_at


def test_the_model_is_independent_of_how_close_the_camera_stands() -> None:
    """The whole point: a fraction of the sign does not change with zoom.

    The same sign measured on a 200px-tall canvas and a 600px-tall one yields
    the same model, so walking toward the sign never invalidates it.
    """

    near = fit_brush_size_model([(60, 120 / 600.0), (12, 24 / 600.0)])
    far = fit_brush_size_model([(60, 40 / 200.0), (12, 8 / 200.0)])

    assert near.slope == pytest.approx(far.slope)
    assert near.sign_pixel_rows == pytest.approx(far.sign_pixel_rows)


def test_a_faded_brush_rim_is_not_counted_as_coverage() -> None:
    """The rim changes color without hiding what was under it.

    Rust's brush fades out over its last texture pixel or so.  Counting that
    fade inflates every band by the same couple of pixels, which is invisible
    on a wide brush and doubles the answer on a narrow one - the bug that made
    automatic sizing pick 2 where 5 was needed.
    """

    before = _canvas()
    after = before.copy()
    draw = ImageDraw.Draw(after)
    solid = (255, 0, 255)
    # Half way between the sign and the stroke: changed, but not covered.
    rim = (175, 48, 175)
    draw.rectangle((60, 85, 340, 89), fill=rim)
    draw.rectangle((60, 90, 340, 109), fill=solid)
    draw.rectangle((60, 110, 340, 114), fill=rim)

    band = measure_stroke_band(before, after)

    assert band.height == pytest.approx(20.0)
    assert band.touched_height == pytest.approx(30.0)


def test_coverage_is_judged_against_the_color_the_stroke_rendered_as() -> None:
    """The sign's material shifts every color, so the commanded one proves nothing.

    A stroke that lands nowhere near the requested magenta is still fully
    covering wherever it is uniform - which is why the rim is found by
    comparing against what the stroke actually rendered as rather than against
    what the picker was asked for.
    """

    before = _canvas()
    after = before.copy()
    ImageDraw.Draw(after).rectangle((60, 90, 340, 109), fill=(175, 48, 175))

    band = measure_stroke_band(before, after)

    assert band.height == pytest.approx(20.0)
    assert band.touched_height == pytest.approx(20.0)


def test_canonical_rows_snap_measurement_noise_to_the_texture_size() -> None:
    """527 measured rows on a 512-row sign is noise, not a 527-row sign."""

    assert canonical_texture_rows(527.0) == 512
    assert canonical_texture_rows(495.0) == 512
    assert canonical_texture_rows(250.0) == 256
    assert canonical_texture_rows(1025.1) == 1024
    assert canonical_texture_rows(128.0) == 128


def test_canonical_rows_keep_a_measurement_far_from_every_power_of_two() -> None:
    """A sign that truly is not a power of two must not be bent into one."""

    assert canonical_texture_rows(100.0) == 100
    assert canonical_texture_rows(768.4) == 768


def test_canonical_rows_reject_nonsense_measurements() -> None:
    assert canonical_texture_rows(0.0) == 0
    assert canonical_texture_rows(-12.0) == 0
    assert canonical_texture_rows(float("nan")) == 0
    assert canonical_texture_rows(float("inf")) == 0
