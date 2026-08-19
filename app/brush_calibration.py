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

# Rust's sign textures come in power-of-two sizes.  A fitted row count carries
# a few percent of band-measurement noise - reading 527 rows on a 512-row sign
# is normal - so a measurement within this tolerance of a power of two *is*
# that power of two.  A measurement far from every one is kept as measured:
# rounding it to a size the sign cannot be would misalign every row.
_CANONICAL_TEXTURE_SIZES = (32, 64, 128, 256, 512, 1024, 2048)
_CANONICAL_TOLERANCE = 0.15


def canonical_texture_rows(measured: float) -> int:
    """Snap a measured texel count to the power of two it is within noise of.

    The distinction matters most at native resolution: planning 527 rows on a
    512-row sign guarantees fifteen collisions where two logical rows fight
    over one texel, while planning exactly 512 lines every cell up with its
    texel.  The measurement's job is to pick the right power of two, not to be
    believed to the last row.
    """

    if not np.isfinite(measured) or measured <= 0:
        return 0
    for size in _CANONICAL_TEXTURE_SIZES:
        if abs(measured / size - 1.0) <= _CANONICAL_TOLERANCE:
            return size
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
    return StrokeBand(
        top=int(np.argmax(rows)),
        height=band_height,
        changed_pixels=int(solid.sum()),
        clipped=bool(rows[0] or rows[-1]),
        touched_height=_band_thickness(touched),
    )


@dataclass(frozen=True, slots=True)
class BrushSizeModel:
    """Affine map from Rust's Size number to a fraction of the sign.

    ``fraction = slope * size + intercept`` where ``fraction`` is the painted
    band height divided by the calibrated canvas height.  Both sides are
    dimensionless, which is what makes the model independent of how close the
    camera happens to be standing.
    """

    slope: float
    intercept: float
    samples: tuple[tuple[float, float], ...]
    captured_at: str = ""

    def __post_init__(self) -> None:
        if not np.isfinite(self.slope) or self.slope < _MIN_SLOPE:
            raise ValueError(
                "Brush size slope must be finite and describe a plausible sign"
            )
        if not np.isfinite(self.intercept):
            raise ValueError("Brush size intercept must be finite")
        if len(self.samples) < 2:
            raise ValueError("A brush size model needs at least two measurements")
        if not self.captured_at:
            object.__setattr__(self, "captured_at", _utc_now())

    def fraction_for_size(self, size: float) -> float:
        """Canvas-height fraction a given Size number paints."""

        return self.slope * float(size) + self.intercept

    def size_for_fraction(self, fraction: float) -> float:
        """The Size number that paints ``fraction`` of the canvas height."""

        return (float(fraction) - self.intercept) / self.slope

    def clamped_size_for_fraction(self, fraction: float) -> float:
        """``size_for_fraction`` quantized and held inside Rust's accepted range.

        The field takes hundredths, so the answer is a float: at the detail end
        of the scale the gap between 1.0 and 2.0 is the difference between a
        correct brush and one twice as wide as the cell it paints.
        """

        size = round(self.size_for_fraction(fraction) / BRUSH_SIZE_STEP) * BRUSH_SIZE_STEP
        return float(min(BRUSH_SIZE_MAX, max(BRUSH_SIZE_MIN, size)))

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": BRUSH_SIZE_MODEL_SCHEMA,
            "slope": self.slope,
            "intercept": self.intercept,
            "samples": [[float(size), float(fraction)] for size, fraction in self.samples],
            "capturedAt": self.captured_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BrushSizeModel":
        raw_samples = value.get("samples") or ()
        samples: list[tuple[float, float]] = []
        for entry in raw_samples:
            if isinstance(entry, Sequence) and len(entry) >= 2:
                samples.append((float(entry[0]), float(entry[1])))
        return cls(
            slope=float(value["slope"]),
            intercept=float(value.get("intercept", 0.0)),
            samples=tuple(samples),
            captured_at=str(value.get("capturedAt") or value.get("captured_at") or ""),
        )


def format_brush_size(size: float) -> str:
    """The exact text typed into Rust's Size field: "1", "1.5", "2.35".

    Trailing zeros are trimmed because every keystroke is a chance for a
    15 FPS frame to drop it - and a dropped digit with the field unfocused
    is a hotbar key.
    """

    text = f"{size:.2f}".rstrip("0").rstrip(".")
    return text or "1"


def fit_brush_size_model(
    samples: Sequence[tuple[float, float]], *, captured_at: str | None = None
) -> BrushSizeModel:
    """Least-squares fit of Size number to painted canvas fraction.

    Rust's brush is expected to run through the origin, but the fit keeps an
    intercept so a minimum footprint or an off-by-one radius convention shows
    up in the constant instead of bending every stroke's size.
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
    return BrushSizeModel(
        slope=float(slope),
        intercept=float(intercept),
        samples=tuple(usable),
        captured_at=captured_at or _utc_now(),
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
