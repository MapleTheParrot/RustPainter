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


def test_band_width_measures_the_horizontal_footprint() -> None:
    """The band sticks out half the brush width past each drag endpoint.

    Height alone cannot say how wide the brush paints: that is only equal on a
    canvas whose rectangle has exactly the sign texture's aspect ratio, which
    a hand-dragged rectangle never quite does.
    """

    before = _canvas()
    after = _with_band(before, top=90, height=21, left=50, right=349)

    band = measure_stroke_band(before, after)

    assert band.width == pytest.approx(300.0)
    assert not band.x_clipped


def test_band_running_into_a_side_reports_x_clipped() -> None:
    before = _canvas()
    after = _with_band(before, top=90, height=21, left=0, right=340)

    band = measure_stroke_band(before, after)

    assert band.x_clipped


def test_conversions_interpolate_between_the_measured_probes() -> None:
    """Inside the probed range the samples answer, not the fitted line.

    These are the actual probe measurements from a live 10x8 run: the fitted
    line misses the size-29.75 probe by over four pixels, and four pixels is
    the entire seam budget of a one-cell brush.  Interpolating between the
    bracketing probes sizes the brush from what the sign actually painted.
    """

    canvas_height = 1081.0
    samples = [
        (14.75, 66.0 / canvas_height),
        (29.75, 128.0 / canvas_height),
        (59.5, 271.0 / canvas_height),
        (100.0, 448.0 / canvas_height),
    ]
    model = fit_brush_size_model(samples)

    # Forward reads hit every probe exactly.
    for size, fraction in samples:
        assert model.fraction_for_size(size) == pytest.approx(fraction)
    # The inverse lands between the bracketing probes, where the global line
    # (which claims 132px at size 30.9) would have left a bare seam.
    wanted = 137.4 / canvas_height
    size = model.size_for_fraction(wanted)
    assert 29.75 < size < 59.5
    assert model.fraction_for_size(size) == pytest.approx(wanted)
    # Outside the probed range the global slope continues from the endpoint
    # sample, so the curve stays continuous instead of jumping onto the line.
    just_past = model.fraction_for_size(100.5)
    assert just_past == pytest.approx(448.0 / canvas_height + 0.5 * model.slope)


def test_noisy_samples_fall_back_to_the_fitted_line() -> None:
    """A band that shrank as the size grew cannot anchor an interpolation."""

    samples = [(12.0, 0.10), (30.0, 0.08), (60.0, 0.47)]
    model = fit_brush_size_model(samples)

    # Non-monotonic fractions: every conversion reads the affine fit.
    assert model.fraction_for_size(30.0) == pytest.approx(
        model.slope * 30.0 + model.intercept
    )


def test_horizontal_samples_fit_their_own_axis() -> None:
    """A sign whose texture is wider than tall paints wider than it does tall."""

    vertical = [(size, size / 320.0) for size in (12, 30, 60)]
    horizontal = [(size, size * 0.5 / 640.0) for size in (12, 30, 60)]
    model = fit_brush_size_model(vertical, samples_x=horizontal)

    assert model.has_horizontal_model
    assert model.sign_pixel_rows == pytest.approx(320.0)
    assert model.sign_pixel_columns == pytest.approx(1280.0)
    assert model.fraction_x_for_size(20) == pytest.approx(10.0 / 640.0)
    assert model.size_for_fraction_x(10.0 / 640.0) == pytest.approx(20.0)


def test_unusable_horizontal_samples_degrade_to_the_vertical_model() -> None:
    """A constant-width band never describes a brush; sizing falls back."""

    vertical = [(size, size / 320.0) for size in (12, 30, 60)]
    flat = [(size, 0.25) for size in (12, 30, 60)]
    model = fit_brush_size_model(vertical, samples_x=flat)

    assert not model.has_horizontal_model
    with pytest.raises(ValueError, match="no horizontal measurement"):
        model.fraction_x_for_size(20)


def test_horizontal_model_survives_a_round_trip_through_a_profile() -> None:
    model = fit_brush_size_model(
        [(60, 0.47), (12, 0.094)],
        samples_x=[(60, 0.235), (12, 0.047)],
    )

    restored = BrushSizeModel.from_dict(model.to_dict())

    assert restored.slope_x == pytest.approx(model.slope_x)
    assert restored.intercept_x == pytest.approx(model.intercept_x)
    assert restored.samples_x == model.samples_x


def test_profiles_written_before_horizontal_measurement_still_load() -> None:
    """Stored profiles predate ``slopeX``; they must load as vertical-only."""

    legacy = {
        "schemaVersion": 1,
        "slope": 0.003,
        "intercept": 0.0005,
        "samples": [[12.0, 0.036], [60.0, 0.18]],
        "capturedAt": "2026-08-20T06:58:45+00:00",
    }
    model = BrushSizeModel.from_dict(legacy)

    assert not model.has_horizontal_model
    assert model.sign_pixel_columns == 0.0


def test_band_centers_locate_the_stroke_for_bias_measurement() -> None:
    """Comparing where the band landed against where the drag was commanded is
    what measures the sign's rendering bias, so the centroid must be right."""

    before = _canvas()
    after = _with_band(before, top=90, height=21, left=50, right=349)

    band = measure_stroke_band(before, after)

    assert band.center_x == pytest.approx((50 + 349) / 2, abs=0.5)
    assert band.center_y == pytest.approx(90 + 10, abs=0.5)


def test_rendering_bias_survives_a_round_trip() -> None:
    model = fit_brush_size_model(
        [(60, 0.47), (12, 0.094)], bias=(-0.004, 0.0019)
    )

    restored = BrushSizeModel.from_dict(model.to_dict())

    assert restored.bias_x == pytest.approx(-0.004)
    assert restored.bias_y == pytest.approx(0.0019)


def test_implausible_bias_degrades_to_no_compensation() -> None:
    """A bias of a tenth of the sign is a broken measurement, not a convention."""

    model = fit_brush_size_model(
        [(60, 0.47), (12, 0.094)], bias=(0.1, float("nan"))
    )

    assert model.bias_x == 0.0
    assert model.bias_y == 0.0


def test_canonical_sizes_include_rusts_four_by_three_textures() -> None:
    """A live probe measured the large wooden sign at 318x238 texels: 320x240.

    Power-of-two-only snapping would bend 238 rows into 256 and misalign every
    native-resolution cell; the nearest canonical size within a tight
    tolerance is the right reading.
    """

    assert canonical_texture_rows(238.4) == 240
    assert canonical_texture_rows(318.4) == 320
    # Nearest candidate wins where families sit close together.
    assert canonical_texture_rows(250.0) == 256
    assert canonical_texture_rows(243.0) == 240
