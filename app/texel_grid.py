"""Measure the sign's texel grid from where its brush actually stamps.

A sign is a texture, and paint lands in texture space: wherever the cursor is
inside a texel, the stamp covers that whole texel and nothing of the next.
Slide the cursor one screen pixel at a time and the stamp stays put, stays
put, then jumps by exactly one texel.  That staircase is the texture's own
grid showing through, and it can be measured without knowing what the Size
number means, what the sign is called, or anything the game keeps to itself.

The measurement is a ladder.  Stamps a few texels apart give the texel pitch
to a few percent; that is enough to count the texels between two stamps
further apart exactly, which gives the pitch to a fraction of a percent; and
so on out to the far side of the sign, where the pitch is known well enough
that the sign's width divided by it is an integer with nothing to round.
The sign's own edge in the capture - the quad the game draws the texture on -
then pins down which lattice line is texel zero, and the jumps say where the
cursor has to be to land on a given texel.

Everything here is pure arithmetic on captures and coordinates; the painter
owns the mouse, the captures and the order of operations.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, Callable, Mapping, Sequence

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image


TEXEL_GRID_SCHEMA = 1

# A stamp has to move the capture by at least this much, as an RGB distance,
# to be a stamp rather than the sign's grain or a compression wobble.
_NOISE_FLOOR = 24.0
_MIN_STAMP_CONTRAST = 40.0

# Within one stair of the staircase every stamp sits on the same texel, so
# their centres agree to capture noise.  A spread wider than this fraction of
# the coarse pitch means the stamps are not snapping to a grid at all.
_MAX_STAIR_SPREAD = 0.3

# How far a ladder stamp may land from a whole number of texels before the
# count is called ambiguous.  The ladder is built so the expected error is
# well under this, so hitting it means a stamp was misread, not mis-counted.
_MAX_LADDER_RESIDUAL = 0.35

# Locating a blurred one-texel stamp by its centroid is good to a fraction of a
# screen pixel; the ladder is planned against this figure.
_CENTROID_SIGMA_PIXELS = 0.45

# The fractional-texel error the next ladder rung may carry and still count
# unambiguously.  Half a texel is the cliff; a third leaves room for noise.
_LADDER_TARGET_ERROR = 0.3

# The sign's edge must sit within this many texels of a lattice line for the
# lattice and the edge to be describing the same texture.
_MAX_EDGE_MISFIT = 0.35

# How sharply the sign's quad must stand out from what surrounds it in the
# capture for its edge to be trusted, in luminance levels per pixel.
_MIN_EDGE_STEP = 18.0


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


# ------------------------------------------------------------------ locating


def stamp_diff(before: "Image", after: "Image") -> np.ndarray:
    """Per-pixel RGB distance between two captures of the same region."""

    before_pixels = np.asarray(before.convert("RGB"), dtype=np.float32)
    after_pixels = np.asarray(after.convert("RGB"), dtype=np.float32)
    if before_pixels.shape != after_pixels.shape:
        raise ValueError("Texel grid captures must have identical dimensions")
    return np.linalg.norm(after_pixels - before_pixels, axis=2)


def locate_stamps(
    diff: np.ndarray,
    points: Sequence[tuple[float, float]],
    window: float,
) -> list[tuple[float, float] | None]:
    """Centre of the stamp nearest each expected point, in capture pixels.

    Each stamp is searched for inside a square ``window`` pixels either side
    of where it was commanded, which is wide enough for the stamp to have
    snapped a texel away and narrow enough that neighbouring stamps stay out
    of each other's windows.  The centre is the diff-weighted centroid of the
    pixels that changed strongly; a bilinear-filtered texel blurs
    symmetrically, so the centroid is the texel's centre however soft its
    edges came out.  ``None`` marks a stamp that never landed.
    """

    height, width = diff.shape
    half = max(2, int(round(window)))
    found: list[tuple[float, float] | None] = []
    for x, y in points:
        left = max(0, int(round(x)) - half)
        right = min(width, int(round(x)) + half + 1)
        top = max(0, int(round(y)) - half)
        bottom = min(height, int(round(y)) + half + 1)
        if right - left < 3 or bottom - top < 3:
            found.append(None)
            continue
        patch = diff[top:bottom, left:right]
        peak = float(patch.max())
        if peak < _MIN_STAMP_CONTRAST:
            found.append(None)
            continue
        strong = patch >= max(_NOISE_FLOOR, peak * 0.5)
        weights = np.where(strong, patch, 0.0)
        total = float(weights.sum())
        if total <= 0.0:
            found.append(None)
            continue
        ys, xs = np.mgrid[top:bottom, left:right]
        # Pixel ``i`` covers ``[i, i + 1)``, so its centre is half a pixel in.
        centre = (
            float((weights * xs).sum() / total) + 0.5,
            float((weights * ys).sum() / total) + 0.5,
        )
        # A window that clipped one side of the blob pulls the centroid
        # toward its own middle; one pass re-centred on the first answer
        # takes the whole blob in.
        left = max(0, int(round(centre[0])) - half)
        right = min(width, int(round(centre[0])) + half + 1)
        top = max(0, int(round(centre[1])) - half)
        bottom = min(height, int(round(centre[1])) + half + 1)
        patch = diff[top:bottom, left:right]
        strong = patch >= max(_NOISE_FLOOR, peak * 0.5)
        weights = np.where(strong, patch, 0.0)
        total = float(weights.sum())
        if total > 0.0:
            ys, xs = np.mgrid[top:bottom, left:right]
            centre = (
                float((weights * xs).sum() / total) + 0.5,
                float((weights * ys).sum() / total) + 0.5,
            )
        found.append(centre)
    return found


def measure_scout(
    diff: np.ndarray, point: tuple[float, float], reach: float
) -> tuple[float, tuple[float, float]]:
    """Size of a lone stamp and where it landed, from a generous search.

    Returns ``(size, centre)`` in capture pixels: ``size`` is the longer side
    of the stamp's bounding box and ``centre`` its centroid.  Everything the
    later stamps are laid out from - how far apart they sit, how wide a
    window finds them, how far the stamp sits from the cursor - comes from
    this one measurement rather than from a guess.
    """

    height, width = diff.shape
    half = max(4, int(round(reach)))
    left = max(0, int(round(point[0])) - half)
    right = min(width, int(round(point[0])) + half + 1)
    top = max(0, int(round(point[1])) - half)
    bottom = min(height, int(round(point[1])) + half + 1)
    patch = diff[top:bottom, left:right]
    peak = float(patch.max()) if patch.size else 0.0
    if peak < _MIN_STAMP_CONTRAST:
        raise ValueError("The scout stamp did not change the sign")
    strong = patch >= max(_NOISE_FLOOR, peak * 0.5)
    ys, xs = np.nonzero(strong)
    size = float(max(xs.max() - xs.min() + 1, ys.max() - ys.min() + 1))
    weights = np.where(strong, patch, 0.0)
    total = float(weights.sum())
    grid_y, grid_x = np.mgrid[top:bottom, left:right]
    centre = (
        float((weights * grid_x).sum() / total) + 0.5,
        float((weights * grid_y).sum() / total) + 0.5,
    )
    return size, centre


# ----------------------------------------------------------------- staircase


@dataclass(frozen=True, slots=True)
class Staircase:
    """What sliding the cursor a pixel at a time revealed about one axis."""

    # Texel pitch in screen pixels, to a few percent.
    coarse_pitch: float
    # Stamp centres of the stairs, one per texel landed on, ascending.
    levels: tuple[float, ...]
    # Cursor positions where the stamp jumped to the next texel: midway
    # between the last cursor on one stair and the first on the next.
    jumps: tuple[float, ...]
    # How far, in cursor pixels, a jump could be from where it was bracketed.
    jump_uncertainty: float


def fit_staircase(
    cursor: Sequence[float], centres: Sequence[float | None]
) -> Staircase:
    """Read the texel pitch and the cursor-to-texel boundaries off a staircase.

    ``cursor`` are the commanded positions along the axis and ``centres`` the
    stamp centres they produced along the same axis, in the same units.  The
    centres cluster on the texels landed on; the gaps between clusters are the
    pitch and the cursor positions between clusters are the boundaries.
    """

    pairs = sorted(
        (float(c), float(m)) for c, m in zip(cursor, centres) if m is not None
    )
    if len(pairs) < 4:
        raise ValueError("Too few stamps landed to read a staircase")
    positions = np.array([c for c, _ in pairs])
    measured = np.array([m for _, m in pairs])
    # Consecutive stamps either share a texel (centres agree to noise) or
    # sit a texel apart.  A step well clear of the centroid noise and at
    # least half the largest step is a jump; anything smaller is noise.
    steps = np.abs(np.diff(measured))
    if steps.max() < 4.0 * _CENTROID_SIGMA_PIXELS:
        raise ValueError("The stamps never moved: no texel boundary was crossed")
    jump_here = steps > max(3.0 * _CENTROID_SIGMA_PIXELS, 0.5 * float(steps.max()))
    jump_sizes = steps[jump_here]
    if jump_sizes.max() > 1.5 * jump_sizes.min():
        raise ValueError(
            "Stamps jumped by unequal amounts: the cursor step skipped texels"
        )
    if int((~jump_here).sum()) < 2:
        raise ValueError(
            "Every stamp landed on a new texel: the cursor step is too coarse "
            "to see the grid"
        )
    # Levels: mean centre of each run between jumps.
    boundaries = np.flatnonzero(jump_here) + 1
    runs = np.split(np.arange(len(measured)), boundaries)
    levels = [float(measured[run].mean()) for run in runs]
    spreads = [float(np.ptp(measured[run])) for run in runs if len(run) > 1]
    coarse_pitch = float(np.median(np.diff(levels)))
    if coarse_pitch <= 0.0:
        raise ValueError("Stamps moved backwards along the axis")
    if spreads and max(spreads) > _MAX_STAIR_SPREAD * coarse_pitch:
        raise ValueError(
            "Stamps within one texel did not agree on where they landed: the "
            "brush is not snapping to a texel grid"
        )
    jumps = tuple(
        float((positions[index - 1] + positions[index]) / 2.0) for index in boundaries
    )
    jump_uncertainty = float(max(positions[index] - positions[index - 1] for index in boundaries)) / 2.0
    return Staircase(
        coarse_pitch=coarse_pitch,
        levels=tuple(levels),
        jumps=jumps,
        jump_uncertainty=jump_uncertainty,
    )


# -------------------------------------------------------------------- ladder


def ladder_offsets(
    coarse_pitch: float, relative_error: float, max_texels: int
) -> tuple[int, ...]:
    """Texel offsets for the ladder, each rung countable from the one before.

    Starting from a pitch known to ``relative_error``, a rung ``d`` texels out
    can be counted exactly while ``relative_error * d`` stays under the target;
    locating it then tightens the error to the centroid noise over its span.
    The rungs grow geometrically until one reaches ``max_texels``.
    """

    offsets: list[int] = []
    error = max(relative_error, 1e-6)
    while True:
        reach = int(_LADDER_TARGET_ERROR / error)
        if reach < 1:
            reach = 1
        if reach >= max_texels:
            if not offsets or offsets[-1] < max_texels:
                offsets.append(max_texels)
            break
        if offsets and reach <= offsets[-1]:
            reach = offsets[-1] + 1
        offsets.append(reach)
        error = _CENTROID_SIGMA_PIXELS / (reach * coarse_pitch)
        if len(offsets) > 12:
            break
    return tuple(offsets)


def refine_pitch(
    base: float,
    rungs: Sequence[tuple[int, float | None]],
    coarse_pitch: float,
) -> tuple[float, float]:
    """Tighten the pitch rung by rung; returns ``(pitch, worst residual)``.

    ``rungs`` are ``(intended texel offset, measured centre)`` pairs in the
    order they were planned.  Each centre is counted in texels with the pitch
    known so far, and the count must come out close to a whole number -
    otherwise the ladder broke and the measurement is not to be trusted.
    """

    pitch = float(coarse_pitch)
    worst = 0.0
    counted = 0
    for _intended, centre in rungs:
        if centre is None:
            continue
        span = float(centre) - float(base)
        texels = span / pitch
        count = int(round(texels))
        residual = abs(texels - count)
        if count < 1 or residual > _MAX_LADDER_RESIDUAL:
            raise ValueError(
                f"A ladder stamp landed {texels:.2f} texels out, too far from a "
                "whole number to count"
            )
        worst = max(worst, residual)
        pitch = span / count
        counted += 1
    if counted == 0:
        raise ValueError("No ladder stamp landed")
    return pitch, worst


# ------------------------------------------------------------------- edges


def find_quad_edges(
    capture: np.ndarray,
    expected: tuple[float, float, float, float],
    search: float,
) -> tuple[float | None, float | None, float | None, float | None]:
    """Sub-pixel left, top, right, bottom edges of the sign's quad in a capture.

    ``expected`` is where the calibrated rectangle puts them, in capture
    pixels, and ``search`` how far either side to look.  An edge is the
    strongest luminance step in its search band, read from a column or row
    profile averaged over the middle of the opposite axis so a stamp or a
    smudge cannot pass for it; a step too weak to be a quad edge reads as
    ``None`` rather than a guess.
    """

    pixels = np.asarray(capture, dtype=np.float32)
    if pixels.ndim == 3:
        luminance = pixels[:, :, :3].mean(axis=2)
    else:
        luminance = pixels
    height, width = luminance.shape
    left, top, right, bottom = expected

    def band(axis_profile: np.ndarray, at: float, rising: bool) -> float | None:
        # Gradient between neighbouring samples; an edge is its extreme.
        gradient = np.diff(axis_profile)
        lo = max(0, int(np.floor(at - search)))
        hi = min(len(gradient), int(np.ceil(at + search)) + 1)
        if hi - lo < 3:
            return None
        segment = gradient[lo:hi] if rising else -gradient[lo:hi]
        index = int(np.argmax(segment))
        if segment[index] < _MIN_EDGE_STEP:
            return None
        # Parabolic refinement between the neighbours; the step between
        # samples i and i + 1 sits on their shared boundary at i + 1.
        offset = 0.0
        if 0 < index < len(segment) - 1:
            a, b, c = segment[index - 1], segment[index], segment[index + 1]
            denominator = a - 2.0 * b + c
            if denominator < 0:
                offset = float(0.5 * (a - c) / denominator)
        return float(lo + index + 1.0 + offset)

    rows = slice(int(height * 0.2), max(int(height * 0.2) + 1, int(height * 0.8)))
    cols = slice(int(width * 0.2), max(int(width * 0.2) + 1, int(width * 0.8)))
    column_profile = luminance[rows, :].mean(axis=0)
    row_profile = luminance[:, cols].mean(axis=1)
    return (
        band(column_profile, left, rising=True),
        band(row_profile, top, rising=True),
        band(column_profile, right, rising=False),
        band(row_profile, bottom, rising=False),
    )


# --------------------------------------------------------------------- axis


@dataclass(frozen=True, slots=True)
class AxisFit:
    """One axis of the texel grid, in absolute screen pixels."""

    pitch: float
    # Leading edge of texel zero.
    origin: float
    count: int
    # Added to a texel's centre to get the cursor position that stamps it.
    aim_offset: float
    # Worst fractional-texel miss across the ladder; a confidence figure.
    residual: float
    # Whether ``origin`` and ``count`` came from the quad's edges (True) or
    # had to be taken from the calibrated rectangle (False).
    from_edges: bool


def fit_axis(
    staircase: Staircase,
    pitch: float,
    *,
    reference_centre: float,
    rect_low: float,
    rect_high: float,
    edge_low: float | None,
    edge_high: float | None,
    residual: float,
) -> AxisFit:
    """Place the lattice on the screen and count its texels along one axis.

    The lattice is fixed by one stamp centre and the pitch.  The sign's low
    edge picks which lattice line is texel zero; the high edge counts the
    texels.  Both edges have to fall on lattice lines - an edge that does not
    is not this texture's edge, and the calibrated rectangle is used instead.
    """

    phase = (reference_centre - pitch / 2.0) % pitch

    def nearest_line(position: float) -> float:
        return phase + round((position - phase) / pitch) * pitch

    from_edges = False
    if edge_low is not None and edge_high is not None:
        origin = nearest_line(edge_low)
        far = nearest_line(edge_high)
        low_misfit = abs(edge_low - origin) / pitch
        high_misfit = abs(edge_high - far) / pitch
        if low_misfit <= _MAX_EDGE_MISFIT and high_misfit <= _MAX_EDGE_MISFIT:
            from_edges = True
    if not from_edges:
        origin = nearest_line(rect_low)
        far = nearest_line(rect_high)
    count = int(round((far - origin) / pitch))
    if count < 2:
        raise ValueError("The sign measured fewer than two texels across")

    # Where the cursor has to be.  Each jump is where the cursor crossed into
    # the texel whose stamp centre is the next level; that texel's cursor
    # window runs one pitch from the jump, so its middle is half a pitch in.
    offsets: list[float] = []
    for jump, level in zip(staircase.jumps, staircase.levels[1:]):
        texel = int(round((level - origin) / pitch - 0.5))
        centre = origin + (texel + 0.5) * pitch
        offsets.append(jump + pitch / 2.0 - centre)
    aim_offset = float(np.median(offsets)) if offsets else 0.0
    return AxisFit(
        pitch=float(pitch),
        origin=float(origin),
        count=count,
        aim_offset=aim_offset,
        residual=float(residual),
        from_edges=from_edges,
    )


# -------------------------------------------------------------------- model


@dataclass(frozen=True, slots=True)
class TexelGridModel:
    """The sign's texture grid, measured, in absolute screen pixels.

    ``columns`` by ``rows`` texels, ``pitch`` screen pixels each, with texel
    (0, 0) starting at ``origin``.  ``aim`` is how far from a texel's centre
    the cursor has to be for the game to stamp that texel.  All of it is
    specific to where the sign sits on the screen right now; the painter
    measures it afresh on every job and the stored copy only informs the
    planner about the sign's resolution.
    """

    columns: int
    rows: int
    pitch_x: float
    pitch_y: float
    origin_x: float
    origin_y: float
    aim_x: float = 0.0
    aim_y: float = 0.0
    residual: float = 0.0
    from_edges: bool = False
    captured_at: str = ""

    def __post_init__(self) -> None:
        if self.columns < 2 or self.rows < 2:
            raise ValueError("A texel grid needs at least two texels each way")
        for name in ("pitch_x", "pitch_y"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError("Texel pitch must be a positive, finite number")
        for name in ("origin_x", "origin_y", "aim_x", "aim_y", "residual"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        if not self.captured_at:
            object.__setattr__(self, "captured_at", _utc_now())

    @property
    def width(self) -> float:
        return self.columns * self.pitch_x

    @property
    def height(self) -> float:
        return self.rows * self.pitch_y

    def registered_rect(self) -> SimpleNamespace:
        """The texture's extent on screen, as a rectangle strokes lay out on."""

        return SimpleNamespace(
            left=self.origin_x, top=self.origin_y, width=self.width, height=self.height
        )

    def agrees_with(self, canvas: Any, tolerance: float = 0.1) -> bool:
        """Whether this grid describes the sign the rectangle was drawn on.

        The grid is absolute, so a moved camera or a re-dragged rectangle
        makes it stale; a stale grid would lay the artwork out over the wrong
        part of the screen, so it has to sit on the rectangle within a little
        hand-drag slop to be used at all.
        """

        width = float(canvas.width)
        height = float(canvas.height)
        if width <= 0 or height <= 0:
            return False
        return (
            abs(self.origin_x - canvas.left) <= tolerance * width
            and abs(self.origin_y - canvas.top) <= tolerance * height
            and abs(self.width - width) <= tolerance * width
            and abs(self.height - height) <= tolerance * height
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": TEXEL_GRID_SCHEMA,
            "columns": self.columns,
            "rows": self.rows,
            "pitchX": self.pitch_x,
            "pitchY": self.pitch_y,
            "originX": self.origin_x,
            "originY": self.origin_y,
            "aimX": self.aim_x,
            "aimY": self.aim_y,
            "residual": self.residual,
            "fromEdges": self.from_edges,
            "capturedAt": self.captured_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TexelGridModel":
        return cls(
            columns=int(value["columns"]),
            rows=int(value["rows"]),
            pitch_x=float(value["pitchX"]),
            pitch_y=float(value["pitchY"]),
            origin_x=float(value["originX"]),
            origin_y=float(value["originY"]),
            aim_x=float(value.get("aimX", 0.0)),
            aim_y=float(value.get("aimY", 0.0)),
            residual=float(value.get("residual", 0.0)),
            from_edges=bool(value.get("fromEdges", False)),
            captured_at=str(value.get("capturedAt") or ""),
        )


# ------------------------------------------------------------- orchestration

# Stamps slid one step at a time along an axis, each on its own row or column
# so they can be told apart in one capture.
_STAIRCASE_STAMPS = 18
_MIN_STAIRCASE_STAMPS = 10
# Cursor steps per hinted texel: fine enough to bracket a boundary closely,
# coarse enough to cross a few of them in one staircase.
_STEPS_PER_TEXEL = 5
# Stamps are kept this many stamp-widths apart across the axis being measured.
_STAMP_SPACING = 3.0
# Texels of margin kept between any stamp and the sign's edge.
_EDGE_MARGIN_TEXELS = 2.0
# Coarsest the staircase's pitch can be trusted to, whatever its jumps say.
_MIN_COARSE_ERROR = 0.15


@dataclass(frozen=True, slots=True)
class GridProbePlan:
    """Where one batch of stamps goes; a sequence of absolute screen points."""

    points: tuple[tuple[float, float], ...]
    label: str


def measure_grid(
    canvas: Any,
    stamp_batch: "Callable[[GridProbePlan], np.ndarray]",
    *,
    pitch_hint: float,
    stamp_hint: float,
    edges: tuple[float | None, float | None, float | None, float | None] | None = None,
) -> TexelGridModel:
    """Measure both axes of the sign's texel grid.

    ``canvas`` is the calibrated rectangle in absolute screen pixels.
    ``stamp_batch`` dabs the given points in order and returns the diff of
    the canvas between before and after, indexed in canvas-capture pixels.
    ``pitch_hint`` and ``stamp_hint`` are rough screen-pixel sizes of a texel
    and of the smallest stamp, used only to lay the probes out.  ``edges`` are
    the sign quad's left, top, right, bottom in absolute screen pixels when
    they could be found in a capture; ``None`` entries fall back to the
    rectangle.
    """

    left, top = float(canvas.left), float(canvas.top)
    width, height = float(canvas.width), float(canvas.height)
    right, bottom = left + width, top + height
    edge_left, edge_top, edge_right, edge_bottom = edges or (None, None, None, None)

    def to_capture(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(x - left, y - top) for x, y in points]

    # Scout: one stamp, searched for generously, tells how big a stamp really
    # is and how far from the cursor it lands.  The hints only size the
    # search; nothing downstream depends on them.
    scout_point = (left + 0.25 * width, top + 0.20 * height)
    diff = stamp_batch(GridProbePlan((scout_point,), "scout"))
    stamp, scout_centre = measure_scout(
        diff,
        to_capture([scout_point])[0],
        max(30.0, 6.0 * max(stamp_hint, pitch_hint, 1.0)),
    )
    # Where a stamp sits relative to the cursor that made it, so the windows
    # that look for later stamps are centred on where they will actually be.
    shift = (
        scout_centre[0] + left - scout_point[0],
        scout_centre[1] + top - scout_point[1],
    )
    # A stamp is at least one texel, so stepping a fifth of the smaller of
    # the stamp and the hinted texel crosses a boundary every few stamps
    # without ever skipping one.
    step = max(1, int(min(stamp, 1.5 * max(pitch_hint, 1.0)) / _STEPS_PER_TEXEL))

    def layout(across_span: float) -> tuple[int, float]:
        """Staircase length and stamp spacing that fit across the sign.

        Comfortable spacing first; a sign too small for that packs the
        stamps closer, then shortens the staircase, and refuses only when
        even the shortest staircase cannot fit.
        """

        rungs = 8
        count = _STAIRCASE_STAMPS
        spacing = max(8.0, _STAMP_SPACING * stamp)
        if (count + rungs) * spacing > across_span:
            spacing = max(8.0, 2.0 * stamp)
            while (count + rungs) * spacing > across_span and count > _MIN_STAIRCASE_STAMPS:
                count -= 1
            if (count + rungs) * spacing > across_span:
                raise ValueError("The sign is too small for the texel probe to fit on")
        return count, spacing

    def shifted(points: Sequence[tuple[float, float]]) -> list[tuple[float, float]]:
        return [(x + shift[0], y + shift[1]) for x, y in points]

    def axis(
        name: str,
        along_start: float,
        across_start: float,
        along_high: float,
        across_high: float,
        make_point: "Callable[[float, float], tuple[float, float]]",
        pick: "Callable[[tuple[float, float]], float]",
        rect_low: float,
        rect_high: float,
        edge_low: float | None,
        edge_high: float | None,
    ) -> AxisFit:
        stamps, spacing = layout(across_high - across_start - 2.0 * stamp)
        window = max(3.0, spacing / 2.0 - 1.0)
        # Staircase: slide along the axis, one stamp per row across it.
        cursor = [along_start + index * step for index in range(stamps)]
        across = [across_start + index * spacing for index in range(stamps)]
        points = tuple(make_point(a, c) for a, c in zip(cursor, across))
        diff = stamp_batch(GridProbePlan(points, f"{name} staircase"))
        centres = locate_stamps(diff, to_capture(shifted(points)), window)
        along_centres = [
            (pick(c) + (left if name == "columns" else top)) if c is not None else None
            for c in centres
        ]
        staircase = fit_staircase(cursor, along_centres)
        coarse = staircase.coarse_pitch
        jumps = max(1, len(staircase.jumps))
        coarse_error = max(
            _MIN_COARSE_ERROR, 2.0 * _CENTROID_SIGMA_PIXELS / (coarse * np.sqrt(jumps))
        )
        base = staircase.levels[0]

        # Ladder: rungs further and further out along the axis, each on the
        # next row across it, until the far side of the sign.
        reach = int((along_high - _EDGE_MARGIN_TEXELS * coarse - base) / coarse)
        if reach < 2:
            raise ValueError(f"The sign is too narrow along its {name} to ladder")
        offsets = ladder_offsets(coarse, coarse_error, reach)
        # Rungs are commanded from the first staircase cursor, whose stamp is
        # ``base``; the count is read against ``base`` itself.
        rung_points = tuple(
            make_point(
                cursor[0] + offset * coarse,
                across_start + (stamps + index) * spacing,
            )
            for index, offset in enumerate(offsets)
        )
        diff = stamp_batch(GridProbePlan(rung_points, f"{name} ladder"))
        rung_centres = locate_stamps(diff, to_capture(shifted(rung_points)), window)
        rungs = [
            (offset, (pick(c) + (left if name == "columns" else top)) if c is not None else None)
            for offset, c in zip(offsets, rung_centres)
        ]
        pitch, residual = refine_pitch(base, rungs, coarse)
        return fit_axis(
            staircase,
            pitch,
            reference_centre=base,
            rect_low=rect_low,
            rect_high=rect_high,
            edge_low=edge_low,
            edge_high=edge_high,
            residual=residual,
        )

    columns = axis(
        "columns",
        along_start=left + 0.12 * width,
        across_start=top + 0.30 * height,
        along_high=right,
        across_high=bottom,
        make_point=lambda a, c: (a, c),
        pick=lambda c: c[0],
        rect_low=left,
        rect_high=right,
        edge_low=edge_left,
        edge_high=edge_right,
    )
    rows = axis(
        "rows",
        along_start=top + 0.10 * height,
        across_start=left + 0.45 * width,
        along_high=bottom,
        across_high=right,
        make_point=lambda a, c: (c, a),
        pick=lambda c: c[1],
        rect_low=top,
        rect_high=bottom,
        edge_low=edge_top,
        edge_high=edge_bottom,
    )
    return TexelGridModel(
        columns=columns.count,
        rows=rows.count,
        pitch_x=columns.pitch,
        pitch_y=rows.pitch,
        origin_x=columns.origin,
        origin_y=rows.origin,
        aim_x=columns.aim_offset,
        aim_y=rows.aim_offset,
        residual=max(columns.residual, rows.residual),
        from_edges=columns.from_edges and rows.from_edges,
    )


__all__ = [
    "TEXEL_GRID_SCHEMA",
    "GridProbePlan",
    "measure_grid",
    "AxisFit",
    "Staircase",
    "TexelGridModel",
    "find_quad_edges",
    "fit_axis",
    "fit_staircase",
    "ladder_offsets",
    "locate_stamps",
    "measure_scout",
    "refine_pitch",
    "stamp_diff",
]
