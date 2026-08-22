"""Measure what Rust's numeric brush size means, in units the planner uses.

Rust's Size field takes a number, not a screen distance, and that number is in
the sign's own texture pixels: a size-20 brush covers the same slice of the
sign whether the camera is pressed against it or halfway across the base.  The
model here is therefore stored as a *fraction of the calibrated canvas* per
size unit, which is dimensionless and survives zoom, painting resolution, and
monitor changes - only switching to a different sign type invalidates it.

Measurement is a diff: capture the canvas, drag one stroke, capture again, and
the band that changed color is the brush footprint at that size.  Reading the
painted result rather than Rust's preview tile is the whole point - the
preview draws at its own scale, so matching a target against it can only ever
be a guess.
"""

from __future__ import annotations

import math

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image


BRUSH_SIZE_MODEL_SCHEMA = 1

# Probed against the live field with typed values and photographed readbacks:
# 0.99 clamps to 1.00, 150 clamps to 100.0, and 1.05 / 1.35 / 2.33 all hold
# exactly.  The field is continuous from 1.00 to 100.00 in hundredths.
BRUSH_SIZE_MIN = 1.0
BRUSH_SIZE_MAX = 100.0

# Sizes are quantized to this step before typing.  Finer steps than the
# measurement noise would express precision the model does not have.
BRUSH_SIZE_STEP = 0.05

# A stroke has to shift a pixel's color by at least this much to count as
# painted.  The sign is a lit, textured surface, so two captures of the same
# unpainted area still differ by a few levels of noise and compression.
_NOISE_FLOOR = 24.0

# The strongest change anywhere must clear this, otherwise the stroke never
# landed: wrong color selected, click swallowed, or the canvas already held
# that exact color.
_MIN_STROKE_CONTRAST = 40.0

# How far a pixel may sit from the stroke's rendered color and still count as
# painted, as a fraction of the full old-to-new color change.  Rust's brush
# fades out over roughly a pixel of sign texture, and those rim pixels change
# color without covering what was underneath - counting them inflates every
# band by the same couple of texture pixels, which is invisible on a wide brush
# and doubles the answer on a narrow one.
_SOLID_TOLERANCE = 0.30

# One size unit has to cover at least this much of the sign for the fit to
# describe a real brush.  Anything shallower implies a sign a hundred thousand
# rows tall, which is what a least-squares fit returns when every probe
# measured the same band - the digits never reached Rust's Size field.
_MIN_SLOPE = 1e-5

# Every edge length a deployable paintable declares in the game's own prefabs
# (tools/sign_sizes.json, read from the asset bundles with
# tools/dump_sign_sizes.py).  Rust's sign textures are far from all powers of
# two: picture frames are 205x256 and 256x192, the artist canvases 192x256
# and 256x640, the DLC frames 128x175 and 320x256.  This table is only the
# fallback for when the texel grid could not be measured on the sign: a
# fitted texel count carries a little band-measurement noise, so a count
# within the tolerance of its *nearest* candidate is that candidate, and a
# count far from every one is kept as measured - rounding it to a size the
# sign cannot be would misalign every row.
_CANONICAL_TEXTURE_SIZES = (
    128, 170, 175, 192, 205, 240, 256, 267, 320, 384, 512, 640, 1024,
)
_CANONICAL_TOLERANCE = 0.08


# Every distinct texture size those prefabs declare, as (columns, rows).
# Where a brush measurement has to stand in for a grid count, the sign's
# true size is one of these: the brush gives the rows only roughly, but the
# calibrated rectangle's shape and the rows together pick out one entry.
SIGN_TEXTURE_SIZES: tuple[tuple[int, int], ...] = (
    (128, 128), (128, 175), (128, 320), (128, 512), (170, 320), (192, 256),
    (205, 256), (256, 128), (256, 192), (256, 256), (256, 512), (256, 640),
    (256, 1024), (267, 320), (320, 240), (320, 256), (320, 384), (512, 256),
    (512, 512), (800, 160), (850, 300), (1024, 256), (1024, 512),
    (1200, 280), (1200, 360), (1320, 280),
)

# One Size unit is not quite one texel.  Read on three signs against their
# grid counts: 294 units for 240 rows (0.82), 330 for 256 (0.78), 649 for
# 512 (0.79).  The table lookup corrects the brush's row count by this
# before comparing, which is what separates 320x240 from 320x256 on a
# rectangle that fits either; nothing that lays out a stroke uses it.
TEXELS_PER_SIZE_UNIT = 0.8

# A rectangle is hand-dragged and a texture can sit under a frame, so the
# rectangle's shape matches the texture's only loosely (live: a 320x240
# texture under a rectangle of 1.20); and the corrected row count is trusted
# to within this factor either way.
_TABLE_ASPECT_TOLERANCE = 0.12
_TABLE_ROWS_TOLERANCE = 1.35


def sign_texture_size(measured_rows: float, aspect: float) -> tuple[int, int] | None:
    """The sign-table size a brush measurement and a rectangle shape point to.

    ``measured_rows`` is the brush model's row count in Size units; ``aspect``
    the calibrated rectangle's width over height.  Of the table entries
    shaped like the rectangle, the one whose rows are nearest the corrected
    count wins, provided it is within a plausible factor; ``None`` leaves the
    caller to the plain canonical snap.
    """

    if not np.isfinite(measured_rows) or measured_rows <= 0 or aspect <= 0:
        return None
    rows = measured_rows * TEXELS_PER_SIZE_UNIT
    shaped = [
        size
        for size in SIGN_TEXTURE_SIZES
        if abs(size[0] / size[1] / aspect - 1.0) <= _TABLE_ASPECT_TOLERANCE
    ]
    if not shaped:
        return None
    best = min(shaped, key=lambda size: abs(math.log(rows / size[1])))
    if abs(math.log(rows / best[1])) > math.log(_TABLE_ROWS_TOLERANCE):
        return None
    return best


def canonical_texture_rows(measured: float) -> int:
    """Snap a measured texel count to the canonical size it is within noise of.

    The distinction matters most at native resolution: planning 527 rows on a
    512-row sign guarantees fifteen collisions where two logical rows fight
    over one texel, while planning exactly 512 lines every cell up with its
    texel.  The measurement's job is to pick the right texture size, not to be
    believed to the last row.
    """

    if not np.isfinite(measured) or measured <= 0:
        return 0
    nearest = min(_CANONICAL_TEXTURE_SIZES, key=lambda size: abs(measured / size - 1.0))
    if abs(measured / nearest - 1.0) <= _CANONICAL_TOLERANCE:
        return nearest
    return int(round(measured))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class StrokeBand:
    """The painted band one calibration stroke left on the canvas."""

    top: int
    height: float
    changed_pixels: int
    clipped: bool
    # Everything the stroke touched, rim included.  Only ever a diagnostic: the
    # gap between it and ``height`` is how much of the brush fades out instead
    # of covering, which is what makes a narrow brush behave unlike a wide one.
    touched_height: float = 0.0
    # Horizontal extent of the solid band in capture pixels.  Subtracting the
    # drag length from it yields the brush's *horizontal* footprint, which a
    # sign whose texture aspect differs from the calibrated rectangle renders
    # wider or narrower than the vertical one - measured, not assumed.
    width: float = 0.0
    # Whether the band ran into the left or right edge of the capture, which
    # makes ``width`` a floor rather than a measurement.
    x_clipped: bool = False
    # Centroid of the solid band in capture pixels.  Compared against where
    # the stroke was commanded, this measures the sign's rendering bias: a
    # live probe showed Rust stamping the brush about a texel left and a
    # fraction of one down from the cursor, uniformly across the sign.
    center_x: float = 0.0
    center_y: float = 0.0

    @property
    def bottom(self) -> float:
        return self.top + self.height


def _band_thickness(mask: np.ndarray) -> float:
    """Median height of a horizontal band, ignoring its end caps.

    Reading the median of the per-column pixel counts rather than a bounding
    box keeps a round brush honest: it tapers at both ends of the drag, and a
    bounding box would report that taper as extra height.  The median describes
    the straight middle section, which is what decides whether adjacent rows
    collide.
    """

    columns = mask.sum(axis=0)
    peak_column = int(columns.max())
    if peak_column <= 0:
        return 0.0
    # Columns holding at least half the thickest column are the straight
    # section; the rest are the brush's end caps or stray noise.
    return float(np.median(columns[columns >= peak_column * 0.5]))


def measure_stroke_band(before: "Image", after: "Image") -> StrokeBand:
    """Height, in capture pixels, of the band a horizontal stroke actually covered.

    "Covered" is deliberately stricter than "changed".  Rust's brush fades out
    over its last texture pixel or so, and those rim pixels shift color without
    hiding what was underneath - a cell left under the rim still reads as
    unpainted on the finished sign.  So the band is measured as the pixels that
    ended up *the stroke's color*, found by taking the color the stroke rendered
    as and keeping what landed close to it.
    """

    before_pixels = np.asarray(before.convert("RGB"), dtype=np.float32)
    after_pixels = np.asarray(after.convert("RGB"), dtype=np.float32)
    if before_pixels.shape != after_pixels.shape:
        raise ValueError("Brush calibration captures must have identical dimensions")
    if before_pixels.ndim != 3 or before_pixels.shape[2] < 3:
        raise ValueError("Brush calibration captures must be RGB images")
    height, width = before_pixels.shape[:2]
    if width < 8 or height < 8:
        raise ValueError("The calibrated canvas is too small to measure a brush")

    change = np.linalg.norm(after_pixels - before_pixels, axis=2)
    peak = float(change.max())
    if peak < _MIN_STROKE_CONTRAST:
        raise ValueError(
            "The calibration stroke did not change the sign. Confirm the paint "
            "tool is selected, the sign is in view, and the canvas calibration "
            "covers only the sign."
        )
    touched = change >= max(_NOISE_FLOOR, peak * 0.25)
    if not touched.any():
        raise ValueError("No painted band was found in the calibration capture")

    # The stroke's rendered color, read from the pixels it changed hardest.  A
    # median shrugs off the rim and the sign's grain, and using the rendered
    # color rather than the commanded one means the sign's material and
    # lighting are already baked in.
    strong = change >= max(_NOISE_FLOOR, peak * 0.6)
    core_mask = strong if strong.any() else touched
    painted_color = np.median(after_pixels[core_mask], axis=0)
    full_change = float(np.median(change[core_mask]))
    to_painted = np.linalg.norm(after_pixels - painted_color, axis=2)
    solid = touched & (to_painted <= max(_NOISE_FLOOR, full_change * _SOLID_TOLERANCE))

    band_height = _band_thickness(solid)
    if band_height <= 0.0:
        raise ValueError(
            "The calibration stroke only blended the sign instead of covering it, "
            "so this brush size paints nothing solid."
        )

    rows = touched.any(axis=1)
    solid_columns = np.flatnonzero(solid.any(axis=0))
    band_width = (
        float(solid_columns[-1] - solid_columns[0] + 1) if solid_columns.size else 0.0
    )
    touched_columns = touched.any(axis=0)
    solid_rows, solid_cols = np.nonzero(solid)
    return StrokeBand(
        top=int(np.argmax(rows)),
        height=band_height,
        changed_pixels=int(solid.sum()),
        clipped=bool(rows[0] or rows[-1]),
        touched_height=_band_thickness(touched),
        width=band_width,
        x_clipped=bool(touched_columns[0] or touched_columns[-1]),
        center_x=float(solid_cols.mean()),
        center_y=float(solid_rows.mean()),
    )


def _interpolation_table(
    samples: Sequence[tuple[float, float]],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Sorted, strictly increasing (sizes, fractions) fit for interpolation.

    Returns empty tuples when the samples cannot anchor an interpolation - a
    duplicate size or a band that shrank as the size grew is measurement noise
    the affine fit absorbs better than a lookup table would.
    """

    ordered = sorted((float(size), float(fraction)) for size, fraction in samples)
    if len(ordered) < 2:
        return (), ()
    sizes = tuple(size for size, _ in ordered)
    fractions = tuple(fraction for _, fraction in ordered)
    if any(b <= a for a, b in zip(sizes, sizes[1:])):
        return (), ()
    if any(b <= a for a, b in zip(fractions, fractions[1:])):
        return (), ()
    return sizes, fractions


def _piecewise_forward(
    size: float, samples: Sequence[tuple[float, float]], slope: float
) -> float | None:
    """Fraction painted at ``size``, read from the nearest measurements.

    Inside the measured range the answer is a straight interpolation between
    the two bracketing probes - the fitted line can miss a probe by a few
    pixels where the sign's texel snapping bends the relationship, and a few
    pixels is the entire seam budget of a one-cell brush.  Outside the range
    the global slope continues from the endpoint sample, so the curve stays
    continuous instead of jumping onto the fitted line.
    """

    sizes, fractions = _interpolation_table(samples)
    if not sizes:
        return None
    if size <= sizes[0]:
        return fractions[0] + slope * (size - sizes[0])
    if size >= sizes[-1]:
        return fractions[-1] + slope * (size - sizes[-1])
    return float(np.interp(size, sizes, fractions))


def _piecewise_inverse(
    fraction: float, samples: Sequence[tuple[float, float]], slope: float
) -> float | None:
    """The Size number that paints ``fraction``, read from the measurements."""

    sizes, fractions = _interpolation_table(samples)
    if not sizes:
        return None
    if fraction <= fractions[0]:
        return sizes[0] + (fraction - fractions[0]) / slope
    if fraction >= fractions[-1]:
        return sizes[-1] + (fraction - fractions[-1]) / slope
    return float(np.interp(fraction, fractions, sizes))


def _quantized_field_size(size: float) -> float:
    """A raw size quantized to the field's step and held inside its range."""

    quantized = round(size / BRUSH_SIZE_STEP) * BRUSH_SIZE_STEP
    return float(min(BRUSH_SIZE_MAX, max(BRUSH_SIZE_MIN, quantized)))


@dataclass(frozen=True, slots=True)
class BrushSizeModel:
    """Map from Rust's Size number to a fraction of the sign.

    ``fraction = slope * size + intercept`` where ``fraction`` is the painted
    band height divided by the calibrated canvas height.  Both sides are
    dimensionless, which is what makes the model independent of how close the
    camera happens to be standing.  Within the measured range the conversions
    interpolate between the stored samples rather than reading the fitted
    line: the line summarizes the sign, the samples *are* the sign.

    ``slope_x`` describes the same brush along the canvas width.  The two only
    agree when the calibrated rectangle has exactly the sign texture's aspect
    ratio, which a hand-dragged rectangle never quite does - and on signs
    whose texture is not square at all, the axes differ by design.  ``0.0``
    means the horizontal footprint was never measured, in which case sizing
    falls back to assuming the footprint is square in screen pixels.
    """

    slope: float
    intercept: float
    samples: tuple[tuple[float, float], ...]
    captured_at: str = ""
    slope_x: float = 0.0
    intercept_x: float = 0.0
    samples_x: tuple[tuple[float, float], ...] = ()
    # Where the sign renders a stroke relative to where it was commanded, as
    # fractions of the canvas (positive = right/down).  A live probe measured
    # Rust stamping about a texel left and a fraction of one down; the painter
    # subtracts this bias from every artwork coordinate so the rendered image
    # lands centered on the calibrated rectangle instead of uniformly shifted.
    bias_x: float = 0.0
    bias_y: float = 0.0

    def __post_init__(self) -> None:
        if not np.isfinite(self.slope) or self.slope < _MIN_SLOPE:
            raise ValueError(
                "Brush size slope must be finite and describe a plausible sign"
            )
        if not np.isfinite(self.intercept):
            raise ValueError("Brush size intercept must be finite")
        if len(self.samples) < 2:
            raise ValueError("A brush size model needs at least two measurements")
        # A horizontal model is an upgrade, never a requirement: anything
        # implausible (an old profile, a corrupted value) degrades back to the
        # square-in-screen-pixels assumption instead of failing the model.
        if not np.isfinite(self.slope_x) or self.slope_x < _MIN_SLOPE:
            object.__setattr__(self, "slope_x", 0.0)
            object.__setattr__(self, "intercept_x", 0.0)
            object.__setattr__(self, "samples_x", ())
        if not np.isfinite(self.intercept_x):
            object.__setattr__(self, "intercept_x", 0.0)
        # A bias beyond a twentieth of the sign is not a rendering convention,
        # it is a measurement gone wrong; compensating it would shift the whole
        # artwork visibly, so an implausible value degrades to no compensation.
        for name in ("bias_x", "bias_y"):
            value = getattr(self, name)
            if not np.isfinite(value) or abs(value) > 0.05:
                object.__setattr__(self, name, 0.0)
        if not self.captured_at:
            object.__setattr__(self, "captured_at", _utc_now())

    @property
    def has_horizontal_model(self) -> bool:
        return self.slope_x >= _MIN_SLOPE

    def fraction_for_size(self, size: float) -> float:
        """Canvas-height fraction a given Size number paints."""

        interpolated = _piecewise_forward(float(size), self.samples, self.slope)
        if interpolated is not None:
            return interpolated
        return self.slope * float(size) + self.intercept

    def size_for_fraction(self, fraction: float) -> float:
        """The Size number that paints ``fraction`` of the canvas height."""

        interpolated = _piecewise_inverse(float(fraction), self.samples, self.slope)
        if interpolated is not None:
            return interpolated
        return (float(fraction) - self.intercept) / self.slope

    def fraction_x_for_size(self, size: float) -> float:
        """Canvas-width fraction a given Size number paints."""

        if not self.has_horizontal_model:
            raise ValueError("This brush model has no horizontal measurement")
        interpolated = _piecewise_forward(float(size), self.samples_x, self.slope_x)
        if interpolated is not None:
            return interpolated
        return self.slope_x * float(size) + self.intercept_x

    def size_for_fraction_x(self, fraction: float) -> float:
        """The Size number that paints ``fraction`` of the canvas width."""

        if not self.has_horizontal_model:
            raise ValueError("This brush model has no horizontal measurement")
        interpolated = _piecewise_inverse(float(fraction), self.samples_x, self.slope_x)
        if interpolated is not None:
            return interpolated
        return (float(fraction) - self.intercept_x) / self.slope_x

    def clamped_size_for_fraction(self, fraction: float) -> float:
        """``size_for_fraction`` quantized and held inside Rust's accepted range.

        The field takes hundredths, so the answer is a float: at the detail end
        of the scale the gap between 1.0 and 2.0 is the difference between a
        correct brush and one twice as wide as the cell it paints.
        """

        return _quantized_field_size(self.size_for_fraction(fraction))

    def clamped_size_for_fraction_x(self, fraction: float) -> float:
        """``size_for_fraction_x`` quantized and held inside the field's range."""

        return _quantized_field_size(self.size_for_fraction_x(fraction))

    @property
    def smallest_fraction(self) -> float:
        return self.fraction_for_size(BRUSH_SIZE_MIN)

    @property
    def largest_fraction(self) -> float:
        return self.fraction_for_size(BRUSH_SIZE_MAX)

    @property
    def fitted_range(self) -> tuple[float, float]:
        """Smallest and largest Size number this model was actually measured at.

        A line read well outside its own data is guesswork, however tidy its
        residuals looked: a couple of pixels of error is nothing on a wide brush
        and is the whole answer on a narrow one.
        """

        sizes = [size for size, _ in self.samples]
        return (min(sizes), max(sizes)) if sizes else (BRUSH_SIZE_MIN, BRUSH_SIZE_MAX)

    @property
    def sign_pixel_rows(self) -> float:
        """Rows in the sign's own texture, if one Size unit is one texture pixel.

        Only ever a diagnostic: it explains *why* a resolution is unreachable
        in terms a user can act on, and never plans a stroke.
        """

        return 1.0 / self.slope if self.slope > 0 else 0.0

    @property
    def sign_pixel_columns(self) -> float:
        """Columns in the sign's texture, when the horizontal axis was measured."""

        return 1.0 / self.slope_x if self.has_horizontal_model else 0.0

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schemaVersion": BRUSH_SIZE_MODEL_SCHEMA,
            "slope": self.slope,
            "intercept": self.intercept,
            "samples": [[float(size), float(fraction)] for size, fraction in self.samples],
            "capturedAt": self.captured_at,
        }
        if self.has_horizontal_model:
            value["slopeX"] = self.slope_x
            value["interceptX"] = self.intercept_x
            value["samplesX"] = [
                [float(size), float(fraction)] for size, fraction in self.samples_x
            ]
        if self.bias_x or self.bias_y:
            value["biasX"] = self.bias_x
            value["biasY"] = self.bias_y
        return value

    @staticmethod
    def _sample_pairs(raw: Any) -> tuple[tuple[float, float], ...]:
        samples: list[tuple[float, float]] = []
        for entry in raw or ():
            if isinstance(entry, Sequence) and len(entry) >= 2:
                samples.append((float(entry[0]), float(entry[1])))
        return tuple(samples)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BrushSizeModel":
        return cls(
            slope=float(value["slope"]),
            intercept=float(value.get("intercept", 0.0)),
            samples=cls._sample_pairs(value.get("samples")),
            captured_at=str(value.get("capturedAt") or value.get("captured_at") or ""),
            slope_x=float(value.get("slopeX", 0.0)),
            intercept_x=float(value.get("interceptX", 0.0)),
            samples_x=cls._sample_pairs(value.get("samplesX")),
            bias_x=float(value.get("biasX", 0.0)),
            bias_y=float(value.get("biasY", 0.0)),
        )


def format_brush_size(size: float) -> str:
    """The exact text typed into Rust's Size field: "1", "1.5", "2.35".

    Trailing zeros are trimmed because every keystroke is a chance for a
    15 FPS frame to drop it - and a dropped digit with the field unfocused
    is a hotbar key.
    """

    text = f"{size:.2f}".rstrip("0").rstrip(".")
    return text or "1"


def _fit_line(
    samples: Sequence[tuple[float, float]],
) -> tuple[float, float, list[tuple[float, float]]]:
    """Least-squares line through (size, fraction) samples.

    Raises ``ValueError`` when the samples cannot describe a brush: fewer than
    two distinct sizes, or a band that never grew with the number typed.
    """

    usable = [(float(size), float(fraction)) for size, fraction in samples]
    if len({size for size, _ in usable}) < 2:
        raise ValueError(
            "Brush calibration needs at least two different Size values to measure"
        )
    sizes = np.array([size for size, _ in usable], dtype=np.float64)
    fractions = np.array([fraction for _, fraction in usable], dtype=np.float64)
    design = np.stack([sizes, np.ones_like(sizes)], axis=1)
    (slope, intercept), *_ = np.linalg.lstsq(design, fractions, rcond=None)
    if not np.isfinite(slope) or slope < _MIN_SLOPE:
        raise ValueError(
            "The painted band did not grow with the Size value. Confirm the Size "
            "field accepts typed numbers and that Enter commits them."
        )
    return float(slope), float(intercept), usable


def fit_brush_size_model(
    samples: Sequence[tuple[float, float]],
    *,
    samples_x: Sequence[tuple[float, float]] = (),
    bias: tuple[float, float] = (0.0, 0.0),
    captured_at: str | None = None,
) -> BrushSizeModel:
    """Least-squares fit of Size number to painted canvas fraction.

    Rust's brush is expected to run through the origin, but the fit keeps an
    intercept so a minimum footprint or an off-by-one radius convention shows
    up in the constant instead of bending every stroke's size.

    ``samples_x`` are optional (size, fraction-of-canvas-width) measurements of
    the same strokes' horizontal footprint.  They are fitted the same way, but
    a failure there is silent: the vertical model still sizes a brush, just
    under the older square-in-screen-pixels assumption.

    ``bias`` is the sign's measured rendering offset as canvas fractions
    (positive = the paint landed right/down of the command); implausible
    values degrade to zero inside the model.
    """

    slope, intercept, usable = _fit_line(samples)
    slope_x = 0.0
    intercept_x = 0.0
    usable_x: list[tuple[float, float]] = []
    if samples_x:
        try:
            slope_x, intercept_x, usable_x = _fit_line(samples_x)
        except ValueError:
            slope_x, intercept_x, usable_x = 0.0, 0.0, []
    return BrushSizeModel(
        slope=slope,
        intercept=intercept,
        samples=tuple(usable),
        captured_at=captured_at or _utc_now(),
        slope_x=slope_x,
        intercept_x=intercept_x,
        samples_x=tuple(usable_x),
        bias_x=float(bias[0]),
        bias_y=float(bias[1]),
    )


__all__ = [
    "BRUSH_SIZE_MAX",
    "BRUSH_SIZE_MIN",
    "BRUSH_SIZE_STEP",
    "canonical_texture_rows",
    "format_brush_size",
    "BRUSH_SIZE_MODEL_SCHEMA",
    "BrushSizeModel",
    "StrokeBand",
    "fit_brush_size_model",
    "measure_stroke_band",
]
