"""Measure Rust's brush - in its preview tile, and on the canvas itself."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Mapping, Sequence

import numpy as np

from .models import ScreenRect

if TYPE_CHECKING:
    from PIL.Image import Image


BRUSH_RESPONSE_SCHEMA = 1

# Probe patches are primed and then measured, so the primed square is larger
# than the square the detector reads: the detector estimates its background
# from the patch border, which therefore has to be paint rather than canvas.
PROBE_PATCH_PIXELS = 176
PROBE_PRIME_PIXELS = 200

# How far apart priming sweeps run when the widest brush could not be measured.
# Any brush at least this wide still covers the square without gaps, at the cost
# of a lot of dragging - which is why the run measures the widest brush first
# and spaces its sweeps from that instead.
PROBE_PRIME_SPACING_PIXELS = 6

# Fraction of the measured widest brush to step between priming sweeps. Under
# one so consecutive sweeps overlap rather than meeting exactly, which absorbs
# both the measurement's own error and a soft brush edge.
PRIME_SWEEP_OVERLAP = 0.8


@dataclass(frozen=True, slots=True)
class BrushFootprint:
    """Detected brush bounds inside a calibrated preview capture."""

    left: int
    top: int
    width: int
    height: int
    confidence: float

    @property
    def diameter(self) -> float:
        """Conservative diameter used to match a logical paint cell."""

        return float(max(self.width, self.height))


def measure_brush_footprint(image: "Image") -> BrushFootprint:
    """Find the centered colored brush shape on Rust's gray preview tile.

    The calibrated region should contain only the gray preview tile.  Its edge
    pixels provide a robust background estimate; the connected foreground
    component nearest the tile center is treated as the brush footprint.
    """

    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("Brush preview capture must be an RGB image")
    height, width = pixels.shape[:2]
    if width < 8 or height < 8:
        raise ValueError("Brush preview calibration is too small")

    border_width = max(1, min(width, height) // 12)
    border_mask = np.zeros((height, width), dtype=np.bool_)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    border_pixels = pixels[border_mask]
    background = np.median(border_pixels, axis=0)
    distances = np.linalg.norm(pixels - background, axis=2)
    border_distances = distances[border_mask]
    # Rust's preview background has a subtle texture.  Keep its ordinary noise
    # out while still accepting dark, white, and saturated brush colors.
    threshold = max(24.0, float(np.percentile(border_distances, 98)) * 2.25)
    foreground = distances >= threshold

    components: list[tuple[int, int, int, int, int, float]] = []
    visited = np.zeros_like(foreground)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    for start_y, start_x in np.argwhere(foreground):
        y = int(start_y)
        x = int(start_x)
        if visited[y, x]:
            continue
        queue: deque[tuple[int, int]] = deque([(x, y)])
        visited[y, x] = True
        min_x = max_x = x
        min_y = max_y = y
        area = 0
        while queue:
            current_x, current_y = queue.popleft()
            area += 1
            min_x = min(min_x, current_x)
            max_x = max(max_x, current_x)
            min_y = min(min_y, current_y)
            max_y = max(max_y, current_y)
            for offset_x, offset_y in (
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),             (1, 0),
                (-1, 1),  (0, 1),  (1, 1),
            ):
                next_x = current_x + offset_x
                next_y = current_y + offset_y
                if (
                    0 <= next_x < width
                    and 0 <= next_y < height
                    and foreground[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_x, next_y))
        component_x = (min_x + max_x) / 2.0
        component_y = (min_y + max_y) / 2.0
        distance_to_center = (component_x - center_x) ** 2 + (
            component_y - center_y
        ) ** 2
        components.append((min_x, min_y, max_x, max_y, area, distance_to_center))

    minimum_area = max(4, round(width * height * 0.0005))
    viable = [component for component in components if component[4] >= minimum_area]
    if not viable:
        raise ValueError(
            "No brush shape was detected. Recalibrate only the gray preview tile "
            "and use a solid circle or square brush."
        )
    min_x, min_y, max_x, max_y, area, distance_to_center = min(
        viable,
        key=lambda component: (component[5], -component[4]),
    )
    detected_width = max_x - min_x + 1
    detected_height = max_y - min_y + 1
    if distance_to_center > (min(width, height) * 0.2) ** 2:
        raise ValueError(
            "The detected shape is not centered in the brush preview. "
            "Recalibrate the gray preview tile."
        )
    fill_ratio = area / float(detected_width * detected_height)
    confidence = min(1.0, max(0.0, fill_ratio))
    return BrushFootprint(
        left=min_x,
        top=min_y,
        width=detected_width,
        height=detected_height,
        confidence=confidence,
    )


@dataclass(frozen=True, slots=True)
class ProbeSite:
    """Where one test dab is stamped, and the region that measures it."""

    point: tuple[int, int]
    patch: ScreenRect
    prime: ScreenRect


def _square_columns(count: int) -> int:
    columns = 1
    while columns * columns < count:
        columns += 1
    return columns


def _centered(center_x: int, center_y: int, size: int) -> ScreenRect:
    return ScreenRect(center_x - size // 2, center_y - size // 2, size, size)


def probe_sites(canvas: ScreenRect, count: int) -> tuple[ProbeSite, ...]:
    """Lay ``count`` well-separated test dabs across a calibrated canvas.

    Every dab shares one canvas and one capture, so they spread over a grid
    rather than stack in one place.  Priming and measuring a small square per
    dab costs a fraction of covering the whole canvas, and the spacing keeps a
    large dab from reaching into its neighbour's patch.
    """

    if count < 2:
        raise ValueError("Brush measurement needs at least two probes")
    columns = _square_columns(count)
    rows = -(-count // columns)
    cell_width = canvas.width / columns
    cell_height = canvas.height / rows
    if min(cell_width, cell_height) < PROBE_PRIME_PIXELS:
        raise ValueError(
            "The calibrated canvas is too small to measure the brush on. "
            "Calibrate a larger sign before measuring the brush."
        )
    sites: list[ProbeSite] = []
    for index in range(count):
        column, row = index % columns, index // columns
        center_x = int(round(canvas.left + (column + 0.5) * cell_width))
        center_y = int(round(canvas.top + (row + 0.5) * cell_height))
        sites.append(
            ProbeSite(
                point=(center_x, center_y),
                patch=_centered(center_x, center_y, PROBE_PATCH_PIXELS),
                prime=_centered(center_x, center_y, PROBE_PRIME_PIXELS),
            )
        )
    return tuple(sites)


def probe_sites_within(
    canvas: ScreenRect, count: int, painted_mask: np.ndarray
) -> tuple[ProbeSite, ...]:
    """Up to ``count`` well-separated probe sites the plan will repaint.

    Measuring right before painting means the test dabs land on the user's
    sign, so every dab must sit where the plan paints over it afterwards.
    Candidates come from progressively denser grids; the selection keeps
    chosen sites a full priming square apart, which is the same separation the
    plain grid guarantees, so a large dab still cannot reach a neighbour's
    patch.  Fewer than ``count`` sites - possibly none - come back when the
    painted region is too small or too fragmented to host more.
    """

    mask = np.asarray(painted_mask, dtype=np.bool_)
    height, width = mask.shape
    if count < 1 or not mask.any() or height == 0 or width == 0:
        return ()

    def cells_under(rect: ScreenRect) -> np.ndarray:
        x0 = (rect.left - canvas.left) * width // canvas.width
        x1 = (rect.left + rect.width - 1 - canvas.left) * width // canvas.width
        y0 = (rect.top - canvas.top) * height // canvas.height
        y1 = (rect.top + rect.height - 1 - canvas.top) * height // canvas.height
        return mask[y0 : y1 + 1, x0 : x1 + 1]

    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    base_columns = _square_columns(count)
    base_rows = -(-count // base_columns)
    for factor in (1, 2, 3, 4):
        columns = base_columns * factor
        rows = base_rows * factor
        cell_width = canvas.width / columns
        cell_height = canvas.height / rows
        for index in range(columns * rows):
            center_x = int(round(canvas.left + (index % columns + 0.5) * cell_width))
            center_y = int(round(canvas.top + (index // columns + 0.5) * cell_height))
            if (center_x, center_y) in seen:
                continue
            seen.add((center_x, center_y))
            prime = _centered(center_x, center_y, PROBE_PRIME_PIXELS)
            if (
                prime.left < canvas.left
                or prime.top < canvas.top
                or prime.left + prime.width > canvas.left + canvas.width
                or prime.top + prime.height > canvas.top + canvas.height
            ):
                continue
            covered = cells_under(prime)
            if covered.size and bool(covered.all()):
                candidates.append((center_x, center_y))

    if not candidates:
        return ()
    # Farthest-point selection, seeded near the canvas center, spreads the
    # dabs as evenly as the painted region allows.
    canvas_center = (canvas.left + canvas.width / 2.0, canvas.top + canvas.height / 2.0)
    chosen = [
        min(
            candidates,
            key=lambda p: (p[0] - canvas_center[0]) ** 2 + (p[1] - canvas_center[1]) ** 2,
        )
    ]
    while len(chosen) < count:
        best: tuple[int, int] | None = None
        best_distance = -1.0
        for candidate in candidates:
            if candidate in chosen:
                continue
            nearest = min(
                (candidate[0] - point[0]) ** 2 + (candidate[1] - point[1]) ** 2
                for point in chosen
            )
            if nearest > best_distance:
                best_distance = nearest
                best = candidate
        if best is None or best_distance < PROBE_PRIME_PIXELS**2:
            break
        chosen.append(best)
    return tuple(
        ProbeSite(
            point=point,
            patch=_centered(point[0], point[1], PROBE_PATCH_PIXELS),
            prime=_centered(point[0], point[1], PROBE_PRIME_PIXELS),
        )
        for point in chosen
    )


def prime_spacing(widest_diameter: float | None) -> int:
    """How far apart priming sweeps may run for a brush of this width."""

    if widest_diameter is None or not widest_diameter > 0:
        return PROBE_PRIME_SPACING_PIXELS
    return max(PROBE_PRIME_SPACING_PIXELS, int(widest_diameter * PRIME_SWEEP_OVERLAP))


def prime_sweeps(
    square: ScreenRect, spacing: int = PROBE_PRIME_SPACING_PIXELS
) -> tuple[tuple[tuple[int, int], tuple[int, int]], ...]:
    """Horizontal strokes that cover ``square`` with the brush at full size.

    ``spacing`` comes from the measured widest brush, so a wide brush primes a
    patch in a handful of strokes instead of the dozens a worst-case guess
    would need.
    """

    step = max(1, int(spacing))
    sweeps: list[tuple[tuple[int, int], tuple[int, int]]] = []
    last = square.top + square.height - 1
    right = square.left + square.width - 1
    y = square.top
    while y < last:
        sweeps.append(((square.left, y), (right, y)))
        y += step
    sweeps.append(((square.left, last), (right, last)))
    return tuple(sweeps)


@dataclass(frozen=True, slots=True)
class BrushResponse:
    """Size-track fraction to painted diameter, measured on the canvas itself.

    The preview tile renders the brush at its own scale, which is not the
    canvas's scale, so a footprint measured there is in the wrong units: a
    brush matched against it lands too large or too small, and neighbouring
    cells bleed into each other.  Measuring what Rust actually paints settles
    the unit mismatch and the brush's soft edge in one step, and a stored curve
    means a job never has to stop and measure at all.
    """

    samples: tuple[tuple[float, float], ...]
    captured_at: str
    shape: str | None = None

    def __post_init__(self) -> None:
        if len(self.samples) < 2:
            raise ValueError("A brush response needs at least two probes")
        fractions = [fraction for fraction, _ in self.samples]
        diameters = [diameter for _, diameter in self.samples]
        if any(not 0.0 <= fraction <= 1.0 for fraction in fractions):
            raise ValueError("Size-track fractions must be between 0 and 1")
        if any(later <= earlier for earlier, later in zip(fractions, fractions[1:])):
            raise ValueError("Brush response samples must ascend by fraction")
        if any(not diameter > 0 for diameter in diameters):
            raise ValueError("Measured brush diameters must be positive")
        if max(diameters) - min(diameters) < 2.0:
            raise ValueError(
                "The Size track did not change the painted brush. Recalibrate the "
                "Size track and confirm a solid square or circle brush is selected."
            )

    def _curve(self) -> tuple[list[float], list[float]]:
        fractions = [fraction for fraction, _ in self.samples]
        # Noise can leave one probe a hair under its predecessor, which would
        # make the inversion ambiguous.  A running maximum keeps the curve
        # single valued without discarding a measurement.
        diameters: list[float] = []
        for _, diameter in self.samples:
            diameters.append(max(diameter, diameters[-1] if diameters else diameter))
        return fractions, diameters

    @property
    def largest_diameter(self) -> float:
        return self._curve()[1][-1]

    @property
    def smallest_diameter(self) -> float:
        return self._curve()[1][0]

    def diameter_for(self, fraction: float) -> float:
        fractions, diameters = self._curve()
        return float(np.interp(float(fraction), fractions, diameters))

    def fraction_for(self, diameter: float) -> float:
        """The Size-track fraction that paints closest to ``diameter``.

        Clamped to the measured range; a target outside it is the caller's to
        reject, with :attr:`largest_diameter` to report what was reachable.
        """

        fractions, diameters = self._curve()
        return float(np.interp(float(diameter), diameters, fractions))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": BRUSH_RESPONSE_SCHEMA,
            "samples": [
                [float(fraction), float(diameter)] for fraction, diameter in self.samples
            ],
            "capturedAt": self.captured_at,
            "shape": self.shape,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BrushResponse":
        raw = value.get("samples")
        if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes)):
            raise ValueError("Brush response samples are missing")
        samples = tuple(
            (float(pair[0]), float(pair[1]))
            for pair in raw
            if isinstance(pair, Sequence) and len(pair) == 2
        )
        shape = value.get("shape")
        return cls(
            samples=samples,
            captured_at=str(value.get("capturedAt", value.get("captured_at", ""))),
            shape=str(shape) if isinstance(shape, str) else None,
        )


@dataclass(frozen=True, slots=True)
class BrushResponseSet:
    """Every curve measured for a profile, one per brush shape.

    A shape change can render a different footprint at the same Size-track
    position, which is why the painter treats shape and diameter as one key.
    Keeping the curves separate means a measurement taken under the square
    brush is never quietly reused for the circle.
    """

    curves: tuple[BrushResponse, ...]

    def __post_init__(self) -> None:
        if not self.curves:
            raise ValueError("A brush response set needs at least one curve")
        shapes = [curve.shape for curve in self.curves]
        if len(set(shapes)) != len(shapes):
            raise ValueError("Brush response curves must each cover a distinct shape")

    def for_shape(self, shape: str | None) -> BrushResponse | None:
        """The curve to size a pass of ``shape`` with, or None to fall back."""

        for curve in self.curves:
            if curve.shape == shape:
                return curve
        # A profile with no shape buttons calibrated measures whatever brush
        # Rust has selected and paints with that same brush throughout, so its
        # single curve is the right one.  With several curves on file the shape
        # is genuinely ambiguous, and guessing is worse than measuring again.
        if shape is None and len(self.curves) == 1:
            return self.curves[0]
        return None

    @property
    def shapes(self) -> tuple[str | None, ...]:
        return tuple(curve.shape for curve in self.curves)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": BRUSH_RESPONSE_SCHEMA,
            "curves": [curve.to_dict() for curve in self.curves],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "BrushResponseSet":
        raw = value.get("curves")
        if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            return cls(
                curves=tuple(
                    BrushResponse.from_dict(curve)
                    for curve in raw
                    if isinstance(curve, Mapping)
                )
            )
        # A document written before curves were kept per shape.
        return cls(curves=(BrushResponse.from_dict(value),))


def build_brush_response(
    samples: Sequence[tuple[float, float]], *, shape: str | None = None
) -> BrushResponse:
    """Order and validate measured probes into a usable response curve."""

    ordered = tuple(
        sorted(((float(f), float(d)) for f, d in samples), key=lambda sample: sample[0])
    )
    return BrushResponse(
        samples=ordered,
        captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
        shape=shape,
    )


__all__ = [
    "BRUSH_RESPONSE_SCHEMA",
    "BrushFootprint",
    "BrushResponse",
    "BrushResponseSet",
    "ProbeSite",
    "build_brush_response",
    "measure_brush_footprint",
    "prime_spacing",
    "prime_sweeps",
    "probe_sites",
    "probe_sites_within",
]
