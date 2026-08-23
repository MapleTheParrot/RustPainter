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

import logging

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image


LOGGER = logging.getLogger("rust_painter.texel_grid")

TEXEL_GRID_SCHEMA = 1

# The game takes paint clicks only on the texture itself, frame or no frame:
# live, a click a pixel above the texture's top edge was swallowed and one on
# the last column's outer half-pixel was not.  The mouse is therefore held on
# whole pixels inside the rendered texture, with the calibrated rectangle -
# hand-dragged, so allowed this much slack - as the outer bound.
RECTANGLE_SLACK_PIXELS = 1.0
# How far inside a texel the visible quad's edge has to fall, on both sides,
# to count as cutting through it rather than lying on its lattice line.
_EDGE_CUT_MARGIN = 0.75

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
# screen pixel on a clean render.  The staircase measures the real scatter on
# the sign in front of it (a grainy, speckled canvas is noisier than that),
# and the ladder is planned against the larger of the two.
_CENTROID_SIGMA_PIXELS = 0.45
# The measured scatter is inflated by this much when planning rungs: a rung
# counted wrong by one texel is the one failure the ladder must not have.
_NOISE_SAFETY = 1.5

# The fractional-texel error the next ladder rung may carry and still count
# unambiguously.  Half a texel is the cliff; a quarter leaves room for noise.
_LADDER_TARGET_ERROR = 0.25

# The most rungs a ladder plans.  The growth floor in ``ladder_offsets``
# makes rung reach multiply by at least 1.35 per rung, so sixteen rungs span
# any sign the game has.
_MAX_LADDER_RUNGS = 16
# Rows across the sign reserved for rung stamps.  A ladder that needs more
# rungs than this wraps back over the same rows: a late rung sits tens to
# hundreds of texels along the axis from the early rung whose row it
# borrows, so their search windows can never overlap - the same sharing the
# staircase flights in one band have always used.
_LADDER_RUNG_ROWS = 8

# A dab now and then lands a whole texel from where the cursor was - seen
# live as one staircase stamp a texel behind its neighbours, most likely the
# game sampling the press a frame before the move.  Centroid noise is tiny
# beside that, so every ladder rung is stamped this many times on separate
# rows and the median is the rung; one stray stamp then changes nothing.
_STAMPS_PER_RUNG = 3

# The sign's extent is measured with dabs aimed at the texels the calibrated
# rectangle's edges fall in and at their neighbours, so the outermost texel
# that can be painted is seen whether the rectangle was dragged a little wide
# or a little narrow - or a frame is drawn over the edge of the texture.
# (Live, the visible quad edge sat 2.7 px inside the texture for exactly
# that reason, which is why the quad edge is logged but never trusted.)
# Texels tried at each edge, relative to the one the rectangle's edge falls
# in: one outward, and two inward - a stamp that lands a texel from the
# cursor makes the outermost texel reachable only from a texel further in.
_EXTENT_NEIGHBOURS = (-1, 0, 1, 2)
_EXTENT_DABS = len(_EXTENT_NEIGHBOURS)
# Each of those positions is dabbed this many times: a stray dab lands a
# texel inward, so the outermost stamp seen at an edge is the true edge as
# long as one copy was not stray.
_EXTENT_COPIES = 2

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
    max_extent: float | None = None,
) -> list[tuple[float, float] | None]:
    """Centre of the stamp nearest each expected point, in capture pixels.

    Each stamp is searched for inside a square ``window`` pixels either side
    of where it was commanded, which is wide enough for the stamp to have
    snapped a texel away and narrow enough that neighbouring stamps stay out
    of each other's windows.  The centre is the diff-weighted centroid of the
    pixels that changed strongly; a bilinear-filtered texel blurs
    symmetrically, so the centroid is the texel's centre however soft its
    edges came out.  ``None`` marks a stamp that never landed - or a window
    holding more than one stamp, which a strong region wider than
    ``max_extent`` pixels betrays: a neighbour's stamp that slipped a texel
    into this window would otherwise be averaged into a centre between the
    two, a position no texel has.
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
        if max_extent is not None and strong.any():
            rows_hit = np.flatnonzero(strong.any(axis=1))
            cols_hit = np.flatnonzero(strong.any(axis=0))
            extent = max(rows_hit[-1] - rows_hit[0] + 1, cols_hit[-1] - cols_hit[0] + 1)
            if extent > max_extent:
                found.append(None)
                continue
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
    # Stamp centre of the first stair, then of the stair after each jump, so
    # ``levels[1:]`` pairs with ``jumps``.
    levels: tuple[float, ...]
    # Cursor positions where the stamp jumped to the next texel: midway
    # between the last cursor on one stair and the first on the next.
    jumps: tuple[float, ...]
    # How far, in cursor pixels, a jump could be from where it was bracketed.
    jump_uncertainty: float
    # Scatter of stamp centres within a stair, in pixels: the centroid noise
    # of one stamp on this sign, measured rather than assumed.
    noise: float = 0.0


def fit_staircase(
    cursor: Sequence[float],
    centres: Sequence[float | None],
    pitch_hint: float | None = None,
    forced_pitch: float | None = None,
) -> Staircase:
    """Read the texel pitch and the cursor-to-texel boundaries off a staircase.

    ``cursor`` are the commanded positions along the axis and ``centres`` the
    stamp centres they produced along the same axis, in the same units.  The
    centres cluster on the texels landed on; the gaps between clusters are the
    pitch and the cursor positions between clusters are the boundaries.

    A dab now and then lands a texel from where its cursor was.  Each stamp
    is therefore first given a texel index, and a stamp whose index disagrees
    with both its neighbours is dropped before anything is read off the rest.
    """

    pairs = sorted(
        (float(c), float(m)) for c, m in zip(cursor, centres) if m is not None
    )
    if len(pairs) < 4:
        raise ValueError("Too few stamps landed to read a staircase")
    positions = np.array([c for c, _ in pairs])
    measured = np.array([m for _, m in pairs])
    # Consecutive stamps either share a texel (centres agree to noise) or
    # sit a texel apart - or two, around a stray.  The median of the steps
    # that clear the centroid noise is a first reading of the pitch, good
    # enough to give every stamp a texel index.
    steps = np.abs(np.diff(measured))
    # A single-texel jump has to clear the centroid noise to be seen at all.
    # The fixed multiple of sigma is right for coarse signs, but on a fine
    # pitch it can sit above the pitch itself: at 1.77 px per texel the 1.8 px
    # gate read genuine one-texel jumps as "never moved" and threw whole
    # staircases away.  When the caller knows roughly how big a texel is, the
    # gate never exceeds most of that.
    jump_gate = 4.0 * _CENTROID_SIGMA_PIXELS
    if pitch_hint is not None and pitch_hint > 0:
        jump_gate = min(jump_gate, 0.8 * pitch_hint)
    if steps.max() < jump_gate:
        raise ValueError("The stamps never moved: no texel boundary was crossed")
    jump_here = steps > 0.75 * jump_gate
    # On a DPI-scaled display the game's cursor map quantizes in steps
    # larger than a pixel, so a one-pixel slide sometimes crosses TWO texel
    # boundaries at once: the visible jumps are then a mixture of one- and
    # two-texel moves, and a plain median can land on the double, halving
    # every index.  Instead, candidate pitches are scored by how well they
    # explain every jump as a whole number of texels, worst fifth dropped
    # for strays, and the LARGEST candidate that explains the jumps wins -
    # any sub-multiple of the truth also "explains" them, a multiple never
    # does.
    moved = np.sort(steps[jump_here].astype(float))
    candidates = {float(np.median(moved)), float(np.median(moved)) / 2.0, float(moved[0])}
    if pitch_hint is not None and pitch_hint > 0:
        candidates.add(float(pitch_hint))
    candidates = {c for c in candidates if c > 0.3}

    def unexplained(candidate: float) -> float:
        counts = np.maximum(np.rint(moved / candidate), 1.0)
        residuals = np.sort(np.abs(moved / candidate - counts))
        kept = residuals[: max(1, int(np.ceil(0.8 * len(residuals))))]
        return float(kept.mean())

    if forced_pitch is not None and forced_pitch > 0:
        # A sibling staircase already read the pitch; indexing with it is
        # unambiguous even when this flight's own jumps are too confused by
        # the cursor quantization to bootstrap from.
        first_pitch = float(forced_pitch)
    else:
        first_pitch = None
        for candidate in sorted(candidates, reverse=True):
            if unexplained(candidate) <= 0.12:
                first_pitch = candidate
                break
        if first_pitch is None:
            first_pitch = min(candidates, key=unexplained)
    index = np.rint((measured - measured.min()) / first_pitch).astype(int)
    # A stamp whose texel disagrees with both neighbours is a stray dab.
    stray = np.zeros(len(index), dtype=bool)
    for i in range(1, len(index) - 1):
        if index[i] != index[i - 1] and index[i] != index[i + 1]:
            stray[i] = True
    if len(index) > 1:
        stray[0] = index[0] > index[1]
        stray[-1] = index[-1] < index[-2]
    if stray.any():
        positions, measured, index = positions[~stray], measured[~stray], index[~stray]
    if len(measured) < 4:
        raise ValueError("Too few clean stamps left to read a staircase")
    if np.any(np.diff(index) < 0):
        raise ValueError("Stamps moved backwards along the axis")
    def group() -> tuple[np.ndarray, list[np.ndarray], list[float], list[int]]:
        stair_breaks = np.flatnonzero(np.diff(index) != 0) + 1
        stair_runs = np.split(np.arange(len(measured)), stair_breaks)
        return (
            stair_breaks,
            stair_runs,
            [float(measured[run].mean()) for run in stair_runs],
            [int(index[run[0]]) for run in stair_runs],
        )

    boundaries, runs, levels, texels = group()
    if len(runs) < 2:
        raise ValueError("The stamps never moved: no texel boundary was crossed")
    per_texel = [
        (levels[i + 1] - levels[i]) / (texels[i + 1] - texels[i]) for i in range(len(runs) - 1)
    ]
    coarse_pitch = float(np.median(per_texel))
    if coarse_pitch <= 0.0:
        raise ValueError("Stamps moved backwards along the axis")
    # A sub-texel stamp near a boundary sometimes paints both texels and its
    # centroid sits between their centres; the stair it forms is off the
    # lattice and used to fail the whole staircase as "jumped by unequal
    # amounts".  Drop stairs that sit off the robust lattice and regroup.
    intercepts = [level - texel * coarse_pitch for level, texel in zip(levels, texels)]
    lattice_base = float(np.median(intercepts))
    off_lattice = [
        stair
        for stair, (level, texel) in enumerate(zip(levels, texels))
        if abs(level - (lattice_base + texel * coarse_pitch)) > 0.35 * coarse_pitch
    ]
    if off_lattice:
        bad = np.zeros(len(measured), dtype=bool)
        for stair in off_lattice:
            bad[runs[stair]] = True
        positions, measured, index = positions[~bad], measured[~bad], index[~bad]
        if len(measured) < 4:
            raise ValueError("Too few clean stamps left to read a staircase")
        boundaries, runs, levels, texels = group()
        if len(runs) < 2:
            raise ValueError("The stamps never moved: no texel boundary was crossed")
        per_texel = [
            (levels[i + 1] - levels[i]) / (texels[i + 1] - texels[i])
            for i in range(len(runs) - 1)
        ]
        coarse_pitch = float(np.median(per_texel))
    if not any(len(run) > 1 for run in runs):
        raise ValueError(
            "Every stamp landed on a new texel: the cursor step is too coarse "
            "to see the grid"
        )
    if max(per_texel) > 1.5 * min(per_texel):
        raise ValueError(
            "Stamps jumped by unequal amounts: the cursor step skipped texels"
        )
    spreads = [float(np.ptp(measured[run])) for run in runs if len(run) > 1]
    if spreads and max(spreads) > _MAX_STAIR_SPREAD * coarse_pitch:
        raise ValueError(
            "Stamps within one texel did not agree on where they landed: the "
            "brush is not snapping to a texel grid"
        )
    # Jumps: only between stairs one texel apart - a gap of two means the
    # stamps in between were dropped, and the boundary is not bracketed.
    jumps = []
    jump_levels = []
    for i, boundary in enumerate(boundaries):
        if texels[i + 1] - texels[i] == 1:
            jumps.append(float((positions[boundary - 1] + positions[boundary]) / 2.0))
            jump_levels.append(levels[i + 1])
    if not jumps:
        raise ValueError("No texel boundary was bracketed cleanly")
    jump_uncertainty = float(
        max(positions[boundary] - positions[boundary - 1] for boundary in boundaries)
    ) / 2.0
    deviations = np.concatenate(
        [measured[run] - measured[run].mean() for run in runs if len(run) > 1]
    )
    noise = float(np.sqrt((deviations**2).mean())) if deviations.size else 0.0
    return Staircase(
        coarse_pitch=coarse_pitch,
        levels=(levels[0], *jump_levels),
        jumps=tuple(jumps),
        jump_uncertainty=jump_uncertainty,
        noise=noise,
    )


# -------------------------------------------------------------------- ladder


def ladder_offsets(
    coarse_pitch: float,
    relative_error: float,
    max_texels: int,
    sigma: float = _CENTROID_SIGMA_PIXELS,
    copies: int = 1,
) -> tuple[int, ...]:
    """Texel offsets for the ladder, each rung countable from the one before.

    Starting from a pitch known to ``relative_error``, a rung ``d`` texels out
    can be counted exactly while ``relative_error * d`` stays under the target;
    locating it then tightens the error to the centroid noise over its span.
    The rungs grow geometrically until one reaches ``max_texels``.

    ``copies`` is how many stamps the caller will average per rung: the noise
    of a located rung shrinks with the square root of the copies.  On a fine
    pitch that factor decides whether the ladder grows at all: at 1.77 px per
    texel a single-stamp rung can only be trusted one texel further out than
    the last (the growth ratio ``target * pitch / sigma`` drops under one),
    and the murica XXL run's ladder stalled at 14 texels, extrapolating the
    pitch of a 1810 px sign from a 25 px lever - 0.5% low, which miscounted
    1024 texels as 1026.  A floor on the growth ratio keeps the ladder moving
    regardless; the per-rung counting error it allows stays well under the
    ``_MAX_LADDER_RESIDUAL`` gate that would call the count ambiguous.
    """

    offsets: list[int] = []
    effective_sigma = max(sigma / np.sqrt(max(copies, 1)), 1e-6)
    # How much further out the next rung may sit, as a multiple of the last:
    # the error a rung carries is ``growth * effective_sigma / pitch`` texels,
    # so even the floored growth keeps it under the target at any pitch the
    # staircase can read at all.
    growth = max(1.35, _LADDER_TARGET_ERROR * coarse_pitch / effective_sigma)
    error = max(relative_error, 1e-6)
    while True:
        reach = int(_LADDER_TARGET_ERROR / error)
        if reach < 2:
            reach = 2
        if reach >= max_texels:
            if not offsets or offsets[-1] < max_texels:
                offsets.append(max_texels)
            break
        if offsets and reach < int(np.ceil(offsets[-1] * growth)):
            reach = int(np.ceil(offsets[-1] * growth))
            if reach >= max_texels:
                offsets.append(max_texels)
                break
        if offsets and reach <= offsets[-1]:
            reach = offsets[-1] + 1
        offsets.append(reach)
        error = effective_sigma / (reach * coarse_pitch)
        if len(offsets) >= _MAX_LADDER_RUNGS:
            break
    return tuple(offsets)


@dataclass(frozen=True, slots=True)
class Ladder:
    """What the rungs settled: the pitch and the furthest counted rung."""

    pitch: float
    worst_residual: float
    far_span: float
    far_count: int


def refine_pitch(
    base: float,
    rungs: Sequence[tuple[int, Sequence[float]]],
    coarse_pitch: float,
) -> Ladder:
    """Tighten the pitch rung by rung.

    ``rungs`` are ``(intended texel offset, measured centres)`` pairs in the
    order they were planned, each rung stamped more than once.  The copies
    of a rung are counted in texels with the pitch known so far and the
    texel most of them landed on is the rung; a copy that strayed a texel is
    outvoted.  The count must come out close to a whole number - otherwise
    the ladder broke and the measurement is not to be trusted.
    """

    pitch = float(coarse_pitch)
    worst = 0.0
    counted = 0
    skipped = 0
    far_span = 0.0
    far_count = 0
    for intended, copies in rungs:
        copies = [float(c) for c in copies]
        if not copies:
            LOGGER.info("  rung %d texels out: no stamp found", intended)
            continue
        counts = [int(round((c - base) / pitch)) for c in copies]
        tally: dict[int, list[float]] = {}
        for count, centre in zip(counts, copies):
            tally.setdefault(count, []).append(centre)
        most = max(len(group) for group in tally.values())
        majority = [count for count, group in tally.items() if len(group) == most]
        # No majority: the copy nearest where the rung was aimed.
        count = min(majority, key=lambda c: abs(c - intended))
        centre = float(np.mean(tally[count]))
        span = centre - float(base)
        texels = span / pitch
        residual = abs(texels - count)
        LOGGER.info(
            "  rung %d texels out: %d of %d stamps agree, centre %.2f, %.3f "
            "texels by the pitch so far -> counted %d (off by %.2f), pitch now %.4f",
            intended,
            len(tally[count]),
            len(copies),
            centre,
            texels,
            count,
            residual,
            span / count if count else float("nan"),
        )
        if count < 1 or residual > _MAX_LADDER_RESIDUAL:
            # On a fine pitch the sub-texel stamp sometimes straddles two
            # texels and its centroid sits between their centres - seen live
            # at 1.77 px per texel, where one straddled rung used to abort
            # the whole measurement.  A rung that cannot be counted cleanly
            # is evidence about nothing; the rungs that can be still tighten
            # the pitch, and the sign-table snap has the final say on the
            # count.
            LOGGER.info(
                "  rung %d texels out: %.2f texels from a whole count - skipped",
                intended,
                residual,
            )
            skipped += 1
            continue
        worst = max(worst, residual)
        pitch = span / count
        counted += 1
        if abs(span) > abs(far_span):
            far_span, far_count = span, count
    if counted == 0:
        raise ValueError("No ladder stamp landed")
    if skipped > counted:
        raise ValueError(
            f"Only {counted} ladder rungs of {counted + skipped} could be "
            "counted; the stamps are not resolving the grid"
        )
    return Ladder(pitch=pitch, worst_residual=worst, far_span=far_span, far_count=far_count)


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
    # The cursor lattice: the cursor at ``aim_origin + (k + 0.5) * aim_pitch``
    # stamps texel ``k``.  Its pitch need not equal the rendered pitch.
    aim_origin: float
    aim_pitch: float
    # Worst fractional-texel miss across the ladder; a confidence figure.
    residual: float
    # Whether ``origin`` and ``count`` were pinned by stamps at the sign's
    # edges (True) or had to be taken from the calibrated rectangle (False).
    from_edges: bool
    # How far the cursor lattice slides along this axis per pixel across it:
    # the shear a sheared cursor map has, in pixels per pixel.
    aim_shear: float = 0.0
    # The fitted inverse map ``k = a + b * along + c * across + d * along *
    # across``, in phase-relative texels; the model re-anchors it.
    aim_coefficients: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)


def fit_axis(
    staircase: Staircase,
    ladder: Ladder,
    *,
    reference_centre: float,
    rect_low: float,
    rect_high: float,
    extent: tuple[float, float] | None,
    aim_origin: float,
    aim_pitch: float,
    edge_low: float | None = None,
    edge_high: float | None = None,
    aim_shear: float = 0.0,
    aim_coefficients: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0),
) -> AxisFit:
    """Place the lattice on the screen and count its texels along one axis.

    The lattice is fixed by one stamp centre and the ladder's pitch.  The
    sign's extent comes from stamps made at its edges: ``extent`` holds the
    centres of the outermost stamps that landed at the low and high ends,
    and the texels they sit on are the first and last texel - observed, not
    inferred.  Without them the calibrated rectangle supplies the count, to
    hand-drag precision.
    """

    pitch = ladder.pitch
    phase = (reference_centre - pitch / 2.0) % pitch

    def texel_of(position: float) -> int:
        return int(round((position - phase) / pitch - 0.5))

    def line(texel: int) -> float:
        return phase + texel * pitch

    from_edges = False
    if extent is not None:
        first, last = texel_of(extent[0]), texel_of(extent[1])
        # A texel the visible quad's edge cuts through - part in view, the
        # rest under the frame the UI draws over the texture's edge - may
        # show no stamp at all, so none seen is no evidence it is not there.
        # Textures end on lattice lines, so an edge inside a texel means
        # that texel exists.  (Live: the last column had 1.5 px in view and
        # 2.6 px under the frame, painted but invisible in the UI - and on
        # the sign in the world.)  An edge on a lattice line is the texture's
        # own edge, and nothing lies beyond it.
        # The quad edge is where the visible texture stops.  Every whole
        # texel between the outermost stamps and that edge exists, whether
        # or not a stamp could reach it (the cursor map can put an edge
        # texel's click outside the calibrated rectangle): extend while the
        # edge reaches at least a margin into the next texel out.  An edge
        # within the margin of a lattice line is that texel's own boundary -
        # noise in the edge detector must not conjure a texel - and the
        # margin scales with the pitch so a fine sign is not frozen by a
        # fixed 0.75 px (0.26 px window at 1.76 px per texel).
        margin = min(_EDGE_CUT_MARGIN, 0.2 * pitch)
        for _ in range(3):
            if edge_low is not None and edge_low < line(first) - margin:
                first -= 1
            else:
                break
        for _ in range(3):
            if edge_high is not None and edge_high > line(last + 1) + margin:
                last += 1
            else:
                break
        if last > first:
            origin = line(first)
            count = last - first + 1
            from_edges = True
    if not from_edges:
        first = int(round((rect_low - phase) / pitch))
        origin = line(first)
        count = int(round((rect_high - origin) / pitch))
    if count < 2:
        raise ValueError("The sign measured fewer than two texels across")
    residual = ladder.worst_residual

    # The cursor lattice was fitted against phase-relative texel indices;
    # re-anchor it on the sign's texel zero.
    return AxisFit(
        pitch=float(pitch),
        origin=float(origin),
        count=count,
        aim_origin=float(aim_origin + first * aim_pitch),
        aim_pitch=float(aim_pitch),
        aim_shear=float(aim_shear),
        # Re-anchored on the sign's texel zero: ``k' = k - first``.
        aim_coefficients=(
            float(aim_coefficients[0] - first),
            float(aim_coefficients[1]),
            float(aim_coefficients[2]),
            float(aim_coefficients[3]),
        ),
        residual=float(residual),
        from_edges=from_edges,
    )


def fit_cursor_lattice(
    boundaries: Sequence[tuple[float, int]], render_pitch: float
) -> tuple[float, float, float]:
    """The cursor lattice along one axis from staircase jumps.

    ``boundaries`` are ``(cursor position of a jump, rendered texel index the
    jump crossed into)`` pairs.  The cursor crosses into texel ``k`` at
    ``origin + k * pitch``; a least-squares line through the pairs gives
    both, and the spread of the jumps about it says how well one lattice
    describes the axis.  Returns ``(origin, pitch, rms)``.
    """

    if not boundaries:
        raise ValueError("No texel boundaries were bracketed")
    positions = np.array([b for b, _ in boundaries], dtype=np.float64)
    texels = np.array([k for _, k in boundaries], dtype=np.float64)
    if len(boundaries) >= 3 and np.ptp(texels) >= 8:
        design = np.stack([texels, np.ones_like(texels)], axis=1)
        (pitch, origin), *_ = np.linalg.lstsq(design, positions, rcond=None)
    else:
        pitch = float(render_pitch)
        origin = float(np.mean(positions - texels * pitch))
    fitted = origin + texels * pitch
    rms = float(np.sqrt(np.mean((positions - fitted) ** 2)))
    return float(origin), float(pitch), rms


def fit_cursor_map(
    boundaries: Sequence[tuple[float, float, int]],
    render_pitch: float,
    min_across_spread: float = 0.0,
) -> tuple[tuple[float, float, float, float], float]:
    """One axis of the cursor map from jumps seen at several places across it.

    ``boundaries`` are ``(along, across, texel)``: the cursor position along
    the axis at which a jump crossed into ``texel``, and where across the
    axis that staircase sat.  The cursor crosses into texel ``k`` where
    ``a + b * along + c * across + d * along * across == k``: a plane with a
    twist, so a boundary that drifts across the sign - a sheared cursor map
    - is followed, and so is a drift that itself changes along the sign - a
    keystoned one, the cursor being ray-cast onto a sign seen in
    perspective.  Returns ``((a, b, c, d), rms)`` with the rms in texels.
    Terms the jumps cannot support are left at zero: with one band there is
    no ``c``; the twist needs jumps spread both ways and enough of them.
    """

    if not boundaries:
        raise ValueError("No texel boundaries were bracketed")
    along = np.array([b[0] for b in boundaries], dtype=np.float64)
    across = np.array([b[1] for b in boundaries], dtype=np.float64)
    texels = np.array([b[2] for b in boundaries], dtype=np.float64)
    spread_along = np.ptp(texels) >= 8 and len(boundaries) >= 3
    # Flights in one band differ a little in where across they sat; only
    # bands genuinely apart can support a slope across the sign.
    spread_across = np.ptp(across) > max(min_across_spread, 1e-6) and len(boundaries) >= 4
    # The twist needs jumps at several places along the axis in more than
    # one band, and enough of them that it is not fitted to noise.
    bands = len(np.unique(np.round(across / max(1.0, min_across_spread / 4.0))))
    twist = spread_across and bands >= 3 and len(boundaries) >= 12 and np.ptp(texels) >= 40
    if not spread_along:
        b = 1.0 / float(render_pitch)
        a = float(np.mean(texels - b * along))
        fitted = a + b * along
        return (a, b, 0.0, 0.0), float(np.sqrt(np.mean((texels - fitted) ** 2)))
    columns = [np.ones_like(along), along]
    if spread_across:
        columns.append(across)
    if twist:
        columns.append(along * across)
    design = np.stack(columns, axis=1)
    coefficients, *_ = np.linalg.lstsq(design, texels, rcond=None)
    a, b = float(coefficients[0]), float(coefficients[1])
    c = float(coefficients[2]) if spread_across else 0.0
    d = float(coefficients[3]) if twist else 0.0
    fitted = design @ coefficients
    # The cursor lattice's pitch is deliberately fitted free rather than tied
    # to the rendered pitch: measured live, the game's cursor mapping runs a
    # slightly different scale from the rendered quad (0.1-0.2%), and the
    # staircase positions span most of the sign, which is lever enough.
    return (a, b, c, d), float(np.sqrt(np.mean((texels - fitted) ** 2)))


def pixel_span(
    texture_low: float, texture_high: float, rect_low: float, rect_high: float
) -> tuple[int, int]:
    """First and last whole pixel that is on the texture and near the rectangle.

    A pixel ``x`` is on the texture when ``texture_low <= x < texture_high``;
    the rectangle, hand-dragged, bounds it with a pixel of slack either side.
    """

    first = int(np.ceil(max(texture_low, rect_low - RECTANGLE_SLACK_PIXELS)))
    last = int(np.floor(min(texture_high - 1e-6, rect_high - 1.0 + RECTANGLE_SLACK_PIXELS)))
    return first, max(first, last)


# -------------------------------------------------------------------- model


@dataclass(frozen=True, slots=True)
class TexelGridModel:
    """The sign's texture grid, measured, in absolute screen pixels.

    Two lattices.  The *rendered* one - ``columns`` by ``rows`` texels,
    ``pitch`` screen pixels each, texel (0, 0) starting at ``origin`` - is
    where the texture's texels are drawn, and so where a capture is read
    back.  The *cursor* one - ``aim_origin`` and ``aim_pitch`` - is where the
    cursor has to be to stamp texel ``k``: ``aim_origin + (k + 0.5) *
    aim_pitch``.  The game maps the cursor and draws the texture over
    slightly different rectangles, so the two pitches differ by a fraction
    of a percent, which is a texel of drift across a sign.

    All of it is specific to where the sign sits on the screen right now;
    the painter measures it afresh on every job and the stored copy only
    informs the planner about the sign's resolution.
    """

    columns: int
    rows: int
    pitch_x: float
    pitch_y: float
    origin_x: float
    origin_y: float
    aim_origin_x: float = float("nan")
    aim_origin_y: float = float("nan")
    aim_pitch_x: float = float("nan")
    aim_pitch_y: float = float("nan")
    # Shear of the cursor map: the cursor lattice along x slides by
    # ``aim_shear_x`` pixels per pixel of screen y (absolute), and vice
    # versa; ``aim_origin`` is the lattice's intercept at y = 0.  Zero for a
    # map that is a plain lattice.
    aim_shear_x: float = 0.0
    aim_shear_y: float = 0.0
    # The full inverse cursor map, per axis: ``u = ax + bx * x + cx * y + dx
    # * x * y`` gives the texel column the cursor at ``(x, y)`` stamps, and
    # likewise ``v`` from ``(ay, by, cy, dy)`` with the roles of x and y
    # swapped.  All-zero means "use the lattice and shear" (an old record).
    aim_map_x: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    aim_map_y: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # Per-column / per-row aim corrections in screen pixels, measured by the
    # aim audit (one dot per index, displaced landings nudged back).  On a
    # DPI-scaled display the game reads the cursor in quantized steps the
    # smooth map above cannot represent: a texel is 1.42 game pixels at 125%
    # scale, and in the phase bands where the map's rounded pixel falls in
    # the wrong step, every dab lands one texel over.  Empty means unaudited.
    aim_nudge_x: tuple[float, ...] = ()
    aim_nudge_y: tuple[float, ...] = ()
    residual: float = 0.0
    from_edges: bool = False
    captured_at: str = ""

    def __post_init__(self) -> None:
        if self.columns < 2 or self.rows < 2:
            raise ValueError("A texel grid needs at least two texels each way")
        for name in ("pitch_x", "pitch_y"):
            if not np.isfinite(getattr(self, name)) or getattr(self, name) <= 0.0:
                raise ValueError("Texel pitch must be a positive, finite number")
        for name in ("origin_x", "origin_y", "residual", "aim_shear_x", "aim_shear_y"):
            if not np.isfinite(getattr(self, name)):
                raise ValueError(f"{name} must be finite")
        # A grid without a measured cursor lattice aims at the rendered one.
        if not np.isfinite(self.aim_origin_x) or not np.isfinite(self.aim_pitch_x):
            object.__setattr__(self, "aim_origin_x", self.origin_x)
            object.__setattr__(self, "aim_pitch_x", self.pitch_x)
        if not np.isfinite(self.aim_origin_y) or not np.isfinite(self.aim_pitch_y):
            object.__setattr__(self, "aim_origin_y", self.origin_y)
            object.__setattr__(self, "aim_pitch_y", self.pitch_y)
        if not self.captured_at:
            object.__setattr__(self, "captured_at", _utc_now())

    @property
    def width(self) -> float:
        return self.columns * self.pitch_x

    @property
    def height(self) -> float:
        return self.rows * self.pitch_y

    def registered_rect(self) -> SimpleNamespace:
        """Where the texels are drawn: the rectangle a capture is read on."""

        return SimpleNamespace(
            left=self.origin_x, top=self.origin_y, width=self.width, height=self.height
        )

    def aim_rect(self) -> SimpleNamespace:
        """Where the cursor stamps them: the rectangle strokes are laid out on.

        A plan of ``columns`` by ``rows`` cells laid out over this rectangle
        puts every cell's centre exactly where the cursor has to be to stamp
        that cell's texel.
        """

        return SimpleNamespace(
            left=self.aim_origin_x,
            top=self.aim_origin_y,
            width=self.columns * self.aim_pitch_x,
            height=self.rows * self.aim_pitch_y,
        )

    def cursor_point(self, u: float, v: float) -> tuple[float, float]:
        """Where the cursor has to be to stamp texel coordinates ``(u, v)``.

        ``u`` and ``v`` are continuous texel coordinates - ``(k + 0.5, m +
        0.5)`` is the middle of texel ``(k, m)``'s cursor window.  The map
        is affine: each axis's lattice plus its shear against the other
        axis, so the two are solved together.
        """

        # x = ox + u * px + sx * y;  y = oy + v * py + sy * x  (screen-absolute)
        ox, oy = self.aim_origin_x, self.aim_origin_y
        px, py = self.aim_pitch_x, self.aim_pitch_y
        sx, sy = self.aim_shear_x, self.aim_shear_y
        bx = ox + u * px
        by = oy + v * py
        determinant = 1.0 - sx * sy
        if abs(determinant) < 1e-9:
            x, y = bx, by
        else:
            x = (bx + sx * by) / determinant
            y = (by + sy * bx) / determinant
        if self.aim_map_x[1] == 0.0 or self.aim_map_y[1] == 0.0:
            return self._nudged(x, y, u, v)
        # The full map, with its twist: ``u = a + b x + c y + d x y`` solves
        # for x at a given y in closed form, and the two axes are iterated
        # from the affine answer - the twist is a few thousandths, so three
        # rounds settle it to far below a pixel.
        ax, bx_, cx, dx = self.aim_map_x
        ay, by_, cy, dy = self.aim_map_y
        for _ in range(3):
            x = (u - ax - cx * y) / (bx_ + dx * y)
            y = (v - ay - cy * x) / (by_ + dy * x)
        return self._nudged(x, y, u, v)

    def _nudged(self, x: float, y: float, u: float, v: float) -> tuple[float, float]:
        """Apply the audit's per-index aim corrections, when any were measured."""

        if self.aim_nudge_x:
            index = min(max(int(u), 0), len(self.aim_nudge_x) - 1)
            x += self.aim_nudge_x[index]
        if self.aim_nudge_y:
            index = min(max(int(v), 0), len(self.aim_nudge_y) - 1)
            y += self.aim_nudge_y[index]
        return x, y

    def clamp_rect(self, canvas: Any) -> SimpleNamespace:
        """Where the mouse may go: whole pixels on the rendered texture.

        The game takes paint clicks on the texture and nowhere else, and the
        texture is known to a fraction of a pixel; the calibrated rectangle
        still bounds it, being what the user vouched for, with a pixel of
        slack for the hand that dragged it.  In the rectangle convention the
        last usable coordinate is ``left + width - 1``.
        """

        left, right = pixel_span(
            self.origin_x, self.origin_x + self.width, canvas.left, canvas.left + canvas.width
        )
        top, bottom = pixel_span(
            self.origin_y, self.origin_y + self.height, canvas.top, canvas.top + canvas.height
        )
        return SimpleNamespace(
            left=float(left),
            top=float(top),
            width=float(max(1, right - left + 1)),
            height=float(max(1, bottom - top + 1)),
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
        value = {
            "schemaVersion": TEXEL_GRID_SCHEMA,
            "columns": self.columns,
            "rows": self.rows,
            "pitchX": self.pitch_x,
            "pitchY": self.pitch_y,
            "originX": self.origin_x,
            "originY": self.origin_y,
            "aimOriginX": self.aim_origin_x,
            "aimOriginY": self.aim_origin_y,
            "aimPitchX": self.aim_pitch_x,
            "aimPitchY": self.aim_pitch_y,
            "aimShearX": self.aim_shear_x,
            "aimShearY": self.aim_shear_y,
            "aimMapX": list(self.aim_map_x),
            "aimMapY": list(self.aim_map_y),
            "residual": self.residual,
            "fromEdges": self.from_edges,
            "capturedAt": self.captured_at,
        }
        if self.aim_nudge_x:
            value["aimNudgeX"] = list(self.aim_nudge_x)
        if self.aim_nudge_y:
            value["aimNudgeY"] = list(self.aim_nudge_y)
        return value

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "TexelGridModel":
        return cls(
            columns=int(value["columns"]),
            rows=int(value["rows"]),
            pitch_x=float(value["pitchX"]),
            pitch_y=float(value["pitchY"]),
            origin_x=float(value["originX"]),
            origin_y=float(value["originY"]),
            aim_origin_x=float(value.get("aimOriginX", float("nan"))),
            aim_origin_y=float(value.get("aimOriginY", float("nan"))),
            aim_pitch_x=float(value.get("aimPitchX", float("nan"))),
            aim_pitch_y=float(value.get("aimPitchY", float("nan"))),
            aim_shear_x=float(value.get("aimShearX", 0.0)),
            aim_shear_y=float(value.get("aimShearY", 0.0)),
            aim_map_x=tuple(float(v) for v in value.get("aimMapX", (0.0, 0.0, 0.0, 0.0))),
            aim_map_y=tuple(float(v) for v in value.get("aimMapY", (0.0, 0.0, 0.0, 0.0))),
            aim_nudge_x=tuple(float(v) for v in value.get("aimNudgeX", ())),
            aim_nudge_y=tuple(float(v) for v in value.get("aimNudgeY", ())),
            residual=float(value.get("residual", 0.0)),
            from_edges=bool(value.get("fromEdges", False)),
            captured_at=str(value.get("capturedAt") or ""),
        )


# ------------------------------------------------------------- orchestration

# Stamps slid one step at a time along an axis, each on its own row or column
# so they can be told apart in one capture.
_STAIRCASE_STAMPS = 18
_MIN_STAIRCASE_STAMPS = 8
# Where along the axis the staircases sit, as fractions of the sign.  The
# first is the long one the ladder grows from; the others are shorter and
# exist to measure the *cursor* lattice: live, the game mapped the cursor
# over the visible quad while drawing the texture a few pixels wider under
# its frame, so the cursor pitch was 0.4% off the rendered pitch - over a
# texel of drift across the sign, which one offset cannot describe.
_STAIRCASE_POSITIONS = (0.12, 0.50, 0.85)
_SHORT_STAIRCASE_STAMPS = 10
# The short staircases are repeated in bands across the axis, so a boundary
# along one axis is seen at several places along the other.  Live, the
# cursor map was sheared: the column boundaries sat up to three pixels
# further left at the bottom of the sign than at the top (a cursor ray-cast
# onto the sign in the world, while the canvas is drawn flat), which a
# lattice measured in one band cannot know.
# The least a staircase should span, in texels, so it brackets a boundary
# or two however few stamps a small sign leaves room for.
_STAIRCASE_SPAN_TEXELS = 2.2
# Cursor steps per hinted texel: fine enough to bracket a boundary closely,
# coarse enough to cross a few of them in one staircase.
_STEPS_PER_TEXEL = 5
# Stamps are kept this many stamp-widths apart across the axis being measured.
_STAMP_SPACING = 3.0
# A located blob wider than this many scout-stamp widths is two stamps, not
# one, and is discarded rather than read as a centre between them.
_STAMP_EXTENT_LIMIT = 1.6
# Texels of margin kept between any stamp and the sign's edge.
_EDGE_MARGIN_TEXELS = 2.0
# Coarsest the staircase's pitch can be trusted to, whatever its jumps say.
_MIN_COARSE_ERROR = 0.15


@dataclass(frozen=True, slots=True)
class GridProbePlan:
    """Where one batch of stamps goes; a sequence of absolute screen points."""

    points: tuple[tuple[float, float], ...]
    label: str


def _snap_axis_to_count(
    fit: AxisFit,
    true_count: int,
    edge_low: float | None,
    edge_high: float | None,
    name: str,
) -> AxisFit:
    """Re-label ``fit`` so it spans exactly ``true_count`` texels.

    The probe counts the texels it can SEE, but the sign's frame is drawn
    over the texture's outer pixels, so one or two edge texels can be
    invisible however well the probe works - and a probe whose ladder was
    starved (fine pitch, old builds) could also miscount outright.  When the
    sign's true texture size is known, the missing (or surplus) texels are
    assigned to the low and high ends so that the frame overhang each end
    implies stays physically small, and the lattice keeps its measured pitch
    and phase - only the labelling moves.
    """

    hidden = true_count - fit.count
    if hidden == 0:
        return fit
    span_low = min(0, hidden)
    span_high = max(0, hidden)
    best_low = hidden // 2
    best_score = None
    for hidden_low in range(span_low, span_high + 1):
        if (hidden_low < 0) != (hidden < 0) and hidden_low != 0:
            continue
        hidden_high = hidden - hidden_low
        if (hidden_high < 0) != (hidden < 0) and hidden_high != 0:
            continue
        origin = fit.origin - hidden_low * fit.pitch
        end = origin + true_count * fit.pitch
        # Frame overhang each end would have: texture edge to visible edge.
        overhang_low = None if edge_low is None else edge_low - origin
        overhang_high = None if edge_high is None else end - edge_high
        score = 0.0
        for overhang in (overhang_low, overhang_high):
            if overhang is None:
                continue
            if overhang < -0.75:
                score += 100.0 + (-overhang)
            elif overhang > 4.5:
                score += 100.0 + overhang
            else:
                score += abs(overhang)
        if best_score is None or score < best_score:
            best_score = score
            best_low = hidden_low
    hidden_low = best_low
    LOGGER.warning(
        "%s: probe counted %d texels but the sign's texture has %d - snapping, "
        "%+d texel(s) at the low edge, %+d at the high (frame-hidden or "
        "miscounted)",
        name,
        fit.count,
        true_count,
        hidden_low,
        hidden - hidden_low,
    )
    return AxisFit(
        pitch=fit.pitch,
        origin=fit.origin - hidden_low * fit.pitch,
        count=true_count,
        aim_origin=fit.aim_origin - hidden_low * fit.aim_pitch,
        aim_pitch=fit.aim_pitch,
        aim_shear=fit.aim_shear,
        # ``k = a + b x + c y + d x y`` labels move with the texels:
        # texel zero moves ``hidden_low`` outward, so every index grows.
        aim_coefficients=(
            fit.aim_coefficients[0] + hidden_low,
            fit.aim_coefficients[1],
            fit.aim_coefficients[2],
            fit.aim_coefficients[3],
        ),
        residual=fit.residual,
        from_edges=fit.from_edges,
    )


def snap_to_texture_sizes(
    columns: AxisFit,
    rows: AxisFit,
    texture_sizes: Sequence[tuple[int, int]],
    edges: tuple[float | None, float | None, float | None, float | None],
    tolerance_texels: int = 5,
) -> tuple[AxisFit, AxisFit, tuple[int, int] | None]:
    """Snap both axes to the nearest known sign texture size, if one is near.

    ``texture_sizes`` are (columns, rows) entries read from the game's own
    bundles.  A measured count within ``tolerance_texels`` of an entry on
    both axes is taken to BE that entry: sign textures only come in those
    sizes, and the probe can miss a frame-covered edge texel or two however
    carefully it stamps.  Farther than that, the measurement is left alone -
    a modded or unknown sign is not forced into the table.
    """

    edge_left, edge_top, edge_right, edge_bottom = edges
    best: tuple[int, tuple[int, int]] | None = None
    for width, height in texture_sizes:
        miss = max(abs(columns.count - width), abs(rows.count - height))
        if miss <= tolerance_texels and (best is None or miss < best[0]):
            best = (miss, (width, height))
    if best is None:
        return columns, rows, None
    width, height = best[1]
    return (
        _snap_axis_to_count(columns, width, edge_left, edge_right, "columns"),
        _snap_axis_to_count(rows, height, edge_top, edge_bottom, "rows"),
        (width, height),
    )


# Below this rendered pitch the aim audit runs after a successful probe: a
# texel this small is at most a couple of game-side cursor steps wide on a
# DPI-scaled display, and the smooth cursor map misplaces whole phase bands
# of columns.  Coarser signs measured 1200/1200 exact without it.
AIM_AUDIT_MAX_PITCH = 2.5
# A landing this far (in texels) from the consensus is a displaced dab.
_AUDIT_DISPLACED_TEXELS = 0.5
# The most pixels one index's aim may be nudged by the audit.
_AUDIT_MAX_NUDGE_PIXELS = 2.0


def audit_cursor_map(
    grid: TexelGridModel,
    stamp_batch: "Callable[[GridProbePlan], np.ndarray]",
    canvas: Any,
    *,
    rounds: int = 3,
) -> TexelGridModel:
    """Stamp one dot per column and per row; nudge the aims that landed off.

    The cursor map is fitted smooth, but on a DPI-scaled display the game
    reads the cursor in quantized steps: at 125% scale a 1.77 px texel is
    1.42 game pixels, and in the phase bands where the smooth map's rounded
    pixel falls in the wrong step every dab lands one texel over - measured
    live as vertical bands of misplaced detail and unfilled holes.  One dot
    per index shows exactly which aims are wrong; a one-pixel nudge flips
    them back, and a verify round re-stamps just the corrected indexes.
    """

    from dataclasses import replace as _replace

    left, top = float(canvas.left), float(canvas.top)
    nudge_x = [0.0] * grid.columns
    nudge_y = [0.0] * grid.rows

    def stagger(k: int, other_count: int) -> int:
        return 4 + (k * 37) % max(1, other_count - 8)

    def aim_for(u_texel: int, v_texel: int) -> tuple[float, float]:
        x, y = grid.cursor_point(u_texel + 0.5, v_texel + 0.5)
        return x + nudge_x[u_texel], y + nudge_y[v_texel]

    def landing(diff: np.ndarray, u_texel: int, v_texel: int, along_x: bool) -> float | None:
        cx = grid.origin_x + (u_texel + 0.5) * grid.pitch_x - left
        cy = grid.origin_y + (v_texel + 0.5) * grid.pitch_y - top
        x0 = int(round(cx)) - 3
        y0 = int(round(cy)) - 3
        window = diff[max(0, y0) : y0 + 7, max(0, x0) : x0 + 7]
        if window.size == 0 or window.max() < _NOISE_FLOOR:
            return None
        ys, xs = np.nonzero(window >= max(_NOISE_FLOOR, 0.5 * float(window.max())))
        weight = window[ys, xs].astype(float)
        mx = float((xs * weight).sum() / weight.sum()) + max(0, x0)
        my = float((ys * weight).sum() / weight.sum()) + max(0, y0)
        if along_x:
            return (mx - cx) / grid.pitch_x
        return (my - cy) / grid.pitch_y

    def audit_axis(along_x: bool) -> int:
        count = grid.columns if along_x else grid.rows
        other = grid.rows if along_x else grid.columns
        nudges = nudge_x if along_x else nudge_y
        name = "columns" if along_x else "rows"
        targets = list(range(count))
        corrected = 0
        for round_index in range(rounds):
            if not targets:
                break
            plan_points = []
            for k in targets:
                if along_x:
                    u_texel, v_texel = k, stagger(k, other)
                else:
                    u_texel, v_texel = stagger(k, other), k
                plan_points.append(aim_for(u_texel, v_texel))
            diff = stamp_batch(
                GridProbePlan(tuple(plan_points), f"{name} aim audit {round_index + 1}")
            )
            offsets: dict[int, float | None] = {}
            for k in targets:
                if along_x:
                    u_texel, v_texel = k, stagger(k, other)
                else:
                    u_texel, v_texel = stagger(k, other), k
                offsets[k] = landing(diff, u_texel, v_texel, along_x)
            finite = [value for value in offsets.values() if value is not None]
            if not finite:
                LOGGER.warning("%s aim audit saw no stamps; leaving the aims alone", name)
                return corrected
            if round_index == 0:
                # The consensus landing offset comes from the full sweep; a
                # verify round re-stamps only the corrected few, which are no
                # crowd to take a median over.
                baseline = float(np.median(finite))
            displaced = []
            for k, value in offsets.items():
                if value is None:
                    continue
                relative = value - baseline
                if abs(relative) < _AUDIT_DISPLACED_TEXELS:
                    continue
                proposed = nudges[k] - float(np.sign(relative))
                if abs(proposed) > _AUDIT_MAX_NUDGE_PIXELS:
                    continue
                nudges[k] = proposed
                displaced.append(k)
            if round_index == 0:
                corrected = len(displaced)
            LOGGER.info(
                "%s aim audit round %d: %d of %d dots landed a texel off%s",
                name,
                round_index + 1,
                len(displaced),
                len(targets),
                "; nudging and re-checking" if displaced and round_index + 1 < rounds else "",
            )
            targets = displaced
        return corrected

    corrected_x = audit_axis(along_x=True)
    corrected_y = audit_axis(along_x=False)
    if corrected_x == 0 and corrected_y == 0:
        return grid
    LOGGER.info(
        "Aim audit corrected %d column aims and %d row aims by whole pixels",
        corrected_x,
        corrected_y,
    )
    return _replace(
        grid, aim_nudge_x=tuple(nudge_x), aim_nudge_y=tuple(nudge_y)
    )


def measure_grid(
    canvas: Any,
    stamp_batch: "Callable[[GridProbePlan], np.ndarray]",
    *,
    pitch_hint: float,
    stamp_hint: float,
    edges: tuple[float | None, float | None, float | None, float | None] | None = None,
    texture_sizes: "Sequence[tuple[int, int]] | None" = None,
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
    # without ever skipping one - and the whole staircase has to span a
    # couple of texels, or a short one on a coarse sign sees no boundary.
    texel_guess = min(stamp, 1.5 * max(pitch_hint, 1.0))
    step = max(1, int(texel_guess / _STEPS_PER_TEXEL))

    @dataclass(frozen=True, slots=True)
    class Layout:
        stamps: int  # stamps in the long staircase
        spacing: float  # pixels between stamps across the axis
        rung_copies: int  # stamps per ladder rung
        extent_copies: int  # stamps per edge dab position
        short: int  # stamps in each short staircase
        extra_bands: int  # short-staircase bands beyond the first

        @property
        def rows(self) -> int:
            return (
                self.stamps
                + _LADDER_RUNG_ROWS * self.rung_copies
                + _EXTENT_DABS * self.extent_copies
                + self.extra_bands * self.short
            )

    def layout(across_span: float) -> "Layout":
        """How the probe's stamps share the rows across the axis.

        Everything the probe wants, in the order it gives things up when a
        sign is too small: extra bands across the sign come before a third
        stamp per rung, because a sheared cursor map (seen live) can only be
        measured from several bands, while a third rung stamp only guards
        against a rare stray dab.  A sign too small even for the leanest
        layout is refused.
        """

        preferences = (
            dict(stamps=18, rung_copies=3, extent_copies=2, short=10, extra_bands=2),
            dict(stamps=18, rung_copies=2, extent_copies=2, short=10, extra_bands=2),
            dict(stamps=16, rung_copies=2, extent_copies=1, short=8, extra_bands=2),
            dict(stamps=14, rung_copies=2, extent_copies=1, short=8, extra_bands=1),
            dict(stamps=12, rung_copies=2, extent_copies=1, short=8, extra_bands=1),
            dict(stamps=12, rung_copies=1, extent_copies=1, short=8, extra_bands=1),
            dict(stamps=12, rung_copies=1, extent_copies=1, short=8, extra_bands=0),
            dict(stamps=10, rung_copies=1, extent_copies=1, short=8, extra_bands=0),
            dict(stamps=8, rung_copies=1, extent_copies=1, short=8, extra_bands=0),
        )
        for spacing in (max(8.0, _STAMP_SPACING * stamp), max(8.0, 2.0 * stamp)):
            rows = int(across_span // spacing)
            for preference in preferences:
                candidate = Layout(spacing=spacing, **preference)
                if candidate.rows <= rows:
                    return candidate
        raise ValueError("The sign is too small for the texel probe to fit on")

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
        plan_rows = layout(across_high - across_start - 2.0 * stamp)
        stamps, spacing = plan_rows.stamps, plan_rows.spacing
        copies_per_rung, extent_copies = plan_rows.rung_copies, plan_rows.extent_copies
        window = max(3.0, spacing / 2.0 - 1.0)
        stride = max(step, int(np.ceil(_STAIRCASE_SPAN_TEXELS * texel_guess / stamps)))
        # Staircases: slide along the axis, one stamp per row across it, at
        # several places along the sign and in several bands across it.
        # Flights in a band share rows - their windows are far apart along
        # the axis - so one capture reads them all.  The first flight is the
        # long one the ladder grows from, in the band the ladder and extent
        # rows follow; the other bands sit beyond those rows.
        along_extent = along_high - along_start
        short = plan_rows.short
        starts = [along_start] + [
            along_start + fraction * along_extent * 0.92
            for fraction in _STAIRCASE_POSITIONS[1:]
        ]
        # The first band, the ladder and the extent rows come first; the
        # extra bands spread over whatever is left, the last one as far
        # across the sign as it will go, for the longest lever on the shear.
        probe_rows = stamps + _LADDER_RUNG_ROWS * copies_per_rung + _EXTENT_DABS * extent_copies
        band_starts = [across_start]
        across_free = across_start + probe_rows * spacing
        across_last = across_high - 2.0 * stamp - short * spacing
        if plan_rows.extra_bands == 1:
            band_starts.append(across_last)
        elif plan_rows.extra_bands >= 2:
            band_starts.append((across_free + across_last) / 2.0)
            band_starts.append(across_last)
        flights: list[tuple[list[float], float]] = []
        all_points: list[tuple[float, float]] = []
        for band_index, band in enumerate(band_starts):
            for flight_index, start in enumerate(starts):
                length = stamps if band_index == 0 and flight_index == 0 else short
                cursor = [start + index * stride for index in range(length)]
                across_of = [band + index * spacing for index in range(length)]
                flights.append((cursor, float(np.mean(across_of))))
                all_points.extend(make_point(a, c) for a, c in zip(cursor, across_of))
        diff = stamp_batch(GridProbePlan(tuple(all_points), f"{name} staircases"))
        centres = locate_stamps(diff, to_capture(shifted(all_points)), window, max_extent=_STAMP_EXTENT_LIMIT * stamp)
        along_all = [
            (pick(c) + (left if name == "columns" else top)) if c is not None else None
            for c in centres
        ]
        staircases: list[tuple[Staircase, float]] = []
        first_flight_fit: Staircase | None = None
        first_flight_error: ValueError | None = None
        consumed = 0
        flight_data: list[tuple[list[float], list[float | None], float]] = []
        for cursor, across_mean in flights:
            flight_centres = along_all[consumed : consumed + len(cursor)]
            consumed += len(cursor)
            flight_data.append((cursor, flight_centres, across_mean))
        for flight_index, (cursor, flight_centres, across_mean) in enumerate(flight_data):
            try:
                fit = fit_staircase(cursor, flight_centres, pitch_hint=texel_guess)
            except ValueError as exc:
                if flight_index == 0:
                    first_flight_error = exc
                else:
                    LOGGER.info("%s staircase %d unusable: %s", name, flight_index + 1, exc)
                continue
            if flight_index == 0:
                first_flight_fit = fit
            staircases.append((fit, across_mean))
        if first_flight_fit is None:
            # The long flight could not bootstrap itself - on a fine pitch a
            # DPI-quantized cursor sometimes confuses its jump mixture - but
            # a sibling flight's pitch can index it unambiguously.
            if not staircases:
                raise first_flight_error or ValueError("No staircase was usable")
            sibling_pitch = float(np.median([s.coarse_pitch for s, _ in staircases]))
            cursor, flight_centres, across_mean = flight_data[0]
            try:
                first_flight_fit = fit_staircase(
                    cursor,
                    flight_centres,
                    pitch_hint=texel_guess,
                    forced_pitch=sibling_pitch,
                )
            except ValueError as exc:
                raise ValueError(
                    f"the long staircase failed ({first_flight_error}) and could "
                    f"not be indexed with the sibling pitch either ({exc})"
                ) from exc
            LOGGER.info(
                "%s staircase 1 re-read with the sibling flights' pitch %.3f px",
                name,
                sibling_pitch,
            )
            staircases.insert(0, (first_flight_fit, across_mean))
        staircase = first_flight_fit
        cursor = flights[0][0]
        coarse = staircase.coarse_pitch
        jumps = max(1, len(staircase.jumps))
        sigma = max(_CENTROID_SIGMA_PIXELS, _NOISE_SAFETY * staircase.noise)
        coarse_error = max(_MIN_COARSE_ERROR, 2.0 * sigma / (coarse * np.sqrt(jumps)))
        base = staircase.levels[0]
        LOGGER.info(
            "%s staircase: %d stamps, %d jumps, coarse pitch %.3f px, stamp "
            "scatter %.2f px (planning with %.2f), jumps at %s",
            name,
            len(cursor),
            len(staircase.jumps),
            coarse,
            staircase.noise,
            sigma,
            ", ".join(f"{j:.1f}" for j in staircase.jumps),
        )

        # Ladder: rungs further and further out along the axis, each on the
        # next row across it, until the far side of the sign.
        reach = int((along_high - _EDGE_MARGIN_TEXELS * coarse - base) / coarse)
        if reach < 2:
            raise ValueError(f"The sign is too narrow along its {name} to ladder")
        offsets = ladder_offsets(
            coarse, coarse_error, reach, sigma=sigma, copies=copies_per_rung
        )
        LOGGER.info(
            "%s ladder: rungs at %s texels, %d stamps each",
            name,
            ", ".join(map(str, offsets)),
            copies_per_rung,
        )
        # Rungs are commanded from the first staircase cursor, whose stamp is
        # ``base``; the count is read against ``base`` itself.  Each rung is
        # stamped several times on its own rows and read as their median, so
        # a dab that lands a texel astray cannot miscount the ladder.
        rung_slots = _LADDER_RUNG_ROWS * copies_per_rung
        rung_points = tuple(
            make_point(
                cursor[0] + offset * coarse,
                across_start
                + (stamps + (index * copies_per_rung + copy) % rung_slots) * spacing,
            )
            for index, offset in enumerate(offsets)
            for copy in range(copies_per_rung)
        )
        diff = stamp_batch(GridProbePlan(rung_points, f"{name} ladder"))
        rung_centres = locate_stamps(diff, to_capture(shifted(rung_points)), window, max_extent=_STAMP_EXTENT_LIMIT * stamp)
        along_rungs = [
            (pick(c) + (left if name == "columns" else top)) if c is not None else None
            for c in rung_centres
        ]
        rungs: list[tuple[int, list[float]]] = []
        for index, offset in enumerate(offsets):
            copies = [
                value
                for value in along_rungs[index * copies_per_rung : (index + 1) * copies_per_rung]
                if value is not None
            ]
            rungs.append((offset, copies))
        ladder = refine_pitch(base, rungs, coarse)

        # The rendered lattice, from the ladder.
        pitch = ladder.pitch
        phase = (base - pitch / 2.0) % pitch

        def texel_of(position: float) -> int:
            return int(round((position - phase) / pitch - 0.5))

        # The cursor map along this axis, from every jump of every
        # staircase: the cursor position that crossed into a texel, where
        # across the axis it was seen, and which texel it was.  Fitted as a
        # plane, so a boundary that slides across the sign is followed.
        boundaries = [
            (jump, across_mean, texel_of(level))
            for flight, across_mean in staircases
            for jump, level in zip(flight.jumps, flight.levels[1:])
        ]
        (plane_a, plane_b, plane_c, plane_d), plane_rms = fit_cursor_map(
            boundaries, pitch, min_across_spread=0.2 * (across_high - across_start)
        )
        # k = a + b * along + c * across (+ d * along * across): summarised as
        # a lattice and a shear at the middle of the sign for the log and the
        # edge dabs; the painter uses the full map.
        across_mid = (across_start + across_high) / 2.0
        slope_mid = plane_b + plane_d * across_mid
        aim_pitch = 1.0 / slope_mid
        aim_origin = -plane_a / slope_mid
        aim_shear = -plane_c / slope_mid
        LOGGER.info(
            "%s cursor map: %d boundaries from %d staircases in %d bands, pitch "
            "%.4f px (rendered %.4f), origin %.2f, shear %+.5f px/px, twist %+.2e, "
            "jumps %.2f texel rms off the fit",
            name,
            len(boundaries),
            len(staircases),
            len(band_starts),
            aim_pitch,
            pitch,
            aim_origin,
            aim_shear,
            plane_d,
            plane_rms,
        )
        # Where the extent row sits across the axis, for aiming on it.
        extent_across = across_start + (
            stamps + min(len(offsets) * copies_per_rung, _LADDER_RUNG_ROWS * copies_per_rung)
        ) * spacing

        # Texels known to exist: the first stair's and the ladder's far rung's.
        known_low = min(texel_of(base), texel_of(base + ladder.far_span))
        known_high = max(texel_of(base), texel_of(base + ladder.far_span))

        def cursor_for(texel: int, side: str) -> float | None:
            """Where to click to stamp ``texel``, or None if nowhere will do.

            The click has to land on the texture.  If ``texel`` exists the
            texture covers everything from it to the texels the ladder
            already stamped, so the cursor - which may sit a texel or two
            from the texel it stamps - is held inside that stretch, and
            inside the calibrated rectangle as the outer bound.
            """

            # The texture certainly covers the candidate texel (if it exists),
            # the texels the ladder stamped, and everything up to the quad
            # edge seen in the capture; a cursor window that stamps a texel
            # from one or two texels over (a stamp offset) is still inside
            # that stretch.
            if side == "low":
                texture_low = min(phase + texel * pitch, edge_low if edge_low is not None else np.inf)
                texture_high = max(phase + (known_high + 1) * pitch, edge_high if edge_high is not None else -np.inf)
            else:
                texture_low = min(phase + known_low * pitch, edge_low if edge_low is not None else np.inf)
                texture_high = max(phase + (texel + 1) * pitch, edge_high if edge_high is not None else -np.inf)
            # The same rule the painter clamps strokes by, so a texel counted
            # here is one the painter can reach.
            low, high = pixel_span(texture_low, texture_high, rect_low, rect_high)
            if high < low:
                return None
            aimed = (texel + 0.5 - plane_a - plane_c * extent_across) / (
                plane_b + plane_d * extent_across
            )
            return float(min(max(int(np.floor(aimed + 0.5)), low), high))

        # Extent: dab at the texels the rectangle's edges fall in and their
        # neighbours, aimed with the cursor lattice and held inside the
        # clickable area.  The outermost stamp that appears at each end sits
        # on the first and last paintable texel - whether the rectangle was
        # dragged a little wide or a little narrow.
        low_guess = texel_of(rect_low + 1.0)
        high_guess = texel_of(rect_high - 2.0)
        extent_row = extent_across
        extent_plan: list[tuple[str, tuple[float, float]]] = []
        for index, neighbour in enumerate(_EXTENT_NEIGHBOURS):
            for copy in range(extent_copies):
                across_here = extent_row + (index * extent_copies + copy) * spacing
                for side, texel in (("low", low_guess + neighbour), ("high", high_guess - neighbour)):
                    along = cursor_for(texel, side)
                    if along is not None:
                        extent_plan.append((side, make_point(along, across_here)))
        extent_points = tuple(point for _side, point in extent_plan)
        diff = stamp_batch(GridProbePlan(extent_points, f"{name} extent"))
        extent_centres = locate_stamps(
            diff,
            to_capture(shifted(extent_points)),
            max(window, ladder.pitch),
            max_extent=_STAMP_EXTENT_LIMIT * stamp,
        )
        along_extent = [
            (pick(c) + (left if name == "columns" else top)) if c is not None else None
            for c in extent_centres
        ]
        lows = [v for (side, _p), v in zip(extent_plan, along_extent) if side == "low" and v is not None]
        highs = [v for (side, _p), v in zip(extent_plan, along_extent) if side == "high" and v is not None]
        extent = (min(lows), max(highs)) if lows and highs else None
        LOGGER.info(
            "%s extent: low-edge stamps at %s, high-edge stamps at %s (quad edge in "
            "the capture: %s to %s)",
            name,
            ", ".join(f"{v:.1f}" for v in lows) or "none",
            ", ".join(f"{v:.1f}" for v in highs) or "none",
            "?" if edge_low is None else f"{edge_low:.1f}",
            "?" if edge_high is None else f"{edge_high:.1f}",
        )

        fit = fit_axis(
            staircase,
            ladder,
            reference_centre=base,
            rect_low=rect_low,
            rect_high=rect_high,
            extent=extent,
            aim_origin=aim_origin,
            aim_pitch=aim_pitch,
            edge_low=edge_low,
            edge_high=edge_high,
            aim_shear=aim_shear,
            aim_coefficients=(plane_a, plane_b, plane_c, plane_d),
        )
        LOGGER.info(
            "%s: %d texels of %.4f px from %.2f; cursor %.4f px from %.2f (%s)",
            name,
            fit.count,
            fit.pitch,
            fit.origin,
            fit.aim_pitch,
            fit.aim_origin,
            "edge stamps" if fit.from_edges else "rectangle",
        )
        return fit

    columns = axis(
        "columns",
        along_start=left + 0.12 * width,
        across_start=top + 0.04 * height,
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
        across_start=left + 0.04 * width,
        along_high=bottom,
        across_high=right,
        make_point=lambda a, c: (c, a),
        pick=lambda c: c[1],
        rect_low=top,
        rect_high=bottom,
        edge_low=edge_top,
        edge_high=edge_bottom,
    )
    if texture_sizes:
        columns, rows, snapped = snap_to_texture_sizes(
            columns,
            rows,
            texture_sizes,
            (edge_left, edge_top, edge_right, edge_bottom),
        )
        if snapped is not None:
            LOGGER.info(
                "Texel grid snapped to the sign table: %dx%d texels",
                snapped[0],
                snapped[1],
            )
    return TexelGridModel(
        columns=columns.count,
        rows=rows.count,
        pitch_x=columns.pitch,
        pitch_y=rows.pitch,
        origin_x=columns.origin,
        origin_y=rows.origin,
        aim_origin_x=columns.aim_origin,
        aim_origin_y=rows.aim_origin,
        aim_pitch_x=columns.aim_pitch,
        aim_pitch_y=rows.aim_pitch,
        aim_shear_x=columns.aim_shear,
        aim_shear_y=rows.aim_shear,
        aim_map_x=columns.aim_coefficients,
        aim_map_y=rows.aim_coefficients,
        residual=max(columns.residual, rows.residual),
        from_edges=columns.from_edges and rows.from_edges,
    )


__all__ = [
    "RECTANGLE_SLACK_PIXELS",
    "pixel_span",
    "TEXEL_GRID_SCHEMA",
    "GridProbePlan",
    "measure_grid",
    "AxisFit",
    "Ladder",
    "Staircase",
    "TexelGridModel",
    "find_quad_edges",
    "fit_axis",
    "fit_cursor_lattice",
    "fit_cursor_map",
    "fit_staircase",
    "ladder_offsets",
    "locate_stamps",
    "measure_scout",
    "refine_pitch",
    "stamp_diff",
]
