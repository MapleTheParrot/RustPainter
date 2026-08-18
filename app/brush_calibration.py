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

# Rust accepts 0 in the Size field, but a zero-width brush paints nothing and
# would read back as a failed stroke rather than a small one, so the usable
# range starts at one.
BRUSH_SIZE_MIN = 1
BRUSH_SIZE_MAX = 100

# A stroke has to shift a pixel's color by at least this much to count as
# painted.  The sign is a lit, textured surface, so two captures of the same
# unpainted area still differ by a few levels of noise and compression.
_NOISE_FLOOR = 24.0

# The strongest change anywhere must clear this, otherwise the stroke never
# landed: wrong color selected, click swallowed, or the canvas already held
# that exact color.
_MIN_STROKE_CONTRAST = 40.0

# One size unit has to cover at least this much of the sign for the fit to
# describe a real brush.  Anything shallower implies a sign a hundred thousand
# rows tall, which is what a least-squares fit returns when every probe
# measured the same band - the digits never reached Rust's Size field.
_MIN_SLOPE = 1e-5


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


@dataclass(frozen=True, slots=True)
class StrokeBand:
    """The painted band one calibration stroke left on the canvas."""

    top: int
    height: float
    changed_pixels: int
    clipped: bool

    @property
    def bottom(self) -> float:
        return self.top + self.height


def measure_stroke_band(before: "Image", after: "Image") -> StrokeBand:
    """Height, in capture pixels, of the band a horizontal stroke painted.

    Thickness is read as the median of the per-column pixel counts over the
    stroke's core rather than from a bounding box.  A round brush tapers at
    both ends of the drag and a bounding box would report that taper as extra
    height; the median describes the straight middle section, which is what
    actually decides whether adjacent rows collide.
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

    distance = np.linalg.norm(after_pixels - before_pixels, axis=2)
    peak = float(distance.max())
    if peak < _MIN_STROKE_CONTRAST:
        raise ValueError(
            "The calibration stroke did not change the sign. Confirm the paint "
            "tool is selected, the sign is in view, and the canvas calibration "
            "covers only the sign."
        )
    changed = distance >= max(_NOISE_FLOOR, peak * 0.4)

    columns = changed.sum(axis=0)
    peak_column = int(columns.max())
    if peak_column <= 0:
        raise ValueError("No painted band was found in the calibration capture")
    # Columns holding at least half the thickest column are the straight
    # section; the rest are the brush's end caps or stray noise.
    core = columns[columns >= peak_column * 0.5]
    band_height = float(np.median(core))

    rows = changed.any(axis=1)
    top = int(np.argmax(rows))
    return StrokeBand(
        top=top,
        height=band_height,
        changed_pixels=int(changed.sum()),
        clipped=bool(rows[0] or rows[-1]),
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
    samples: tuple[tuple[int, float], ...]
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

    def clamped_size_for_fraction(self, fraction: float) -> int:
        """``size_for_fraction`` rounded and held inside Rust's accepted range."""

        size = round(self.size_for_fraction(fraction))
        return int(min(BRUSH_SIZE_MAX, max(BRUSH_SIZE_MIN, size)))

    @property
    def smallest_fraction(self) -> float:
        return self.fraction_for_size(BRUSH_SIZE_MIN)

    @property
    def largest_fraction(self) -> float:
        return self.fraction_for_size(BRUSH_SIZE_MAX)

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
            "samples": [[int(size), float(fraction)] for size, fraction in self.samples],
            "capturedAt": self.captured_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BrushSizeModel":
        raw_samples = value.get("samples") or ()
        samples: list[tuple[int, float]] = []
        for entry in raw_samples:
            if isinstance(entry, Sequence) and len(entry) >= 2:
                samples.append((int(entry[0]), float(entry[1])))
        return cls(
            slope=float(value["slope"]),
            intercept=float(value.get("intercept", 0.0)),
            samples=tuple(samples),
            captured_at=str(value.get("capturedAt") or value.get("captured_at") or ""),
        )


def fit_brush_size_model(
    samples: Sequence[tuple[int, float]], *, captured_at: str | None = None
) -> BrushSizeModel:
    """Least-squares fit of Size number to painted canvas fraction.

    Rust's brush is expected to run through the origin, but the fit keeps an
    intercept so a minimum footprint or an off-by-one radius convention shows
    up in the constant instead of bending every stroke's size.
    """

    usable = [(int(size), float(fraction)) for size, fraction in samples]
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
    "BRUSH_SIZE_MODEL_SCHEMA",
    "BrushSizeModel",
    "StrokeBand",
    "fit_brush_size_model",
    "measure_stroke_band",
]
