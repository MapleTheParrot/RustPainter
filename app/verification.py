"""Compare the painted canvas against the plan and build touch-up strokes.

Open-loop painting misses for reasons no calibration can fully remove: a
dropped click, a frame hitch while a stroke was moving, a picker click landing
one hue off.  Reading the sign back and repainting only the cells that came
out wrong is what actually closes the gap between the preview and the sign.

The sign is a lit, textured 3D surface, so captured colors never equal the
palette values exactly.  Every comparison here is therefore *relative*: a cell
counts as wrong only when its captured color sits decisively closer to
something else than to what its own color looks like on this sign - a global
lighting shift moves every color together and changes nothing.

Two things came out of reading a real five-hour painting back:

- Almost every genuine miss was a *hole*: a short stroke the game never
  registered, leaving bare sign where a run of cells should be.  Holes are
  found by comparing against the bare sign itself rather than hoping the
  wood happens to resemble some other palette entry.
- Almost every false alarm was a *twin*: two palette entries a few units
  apart that the sign renders identically, with the material shift pushing
  both past the line between them.  Colors are therefore compared against
  what each one actually looks like on the sign, read from the capture, so
  two entries that render alike can never be confused for one another.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .models import ColorGroup, PaintPlan, RGBColor
from .paint_optimizer import _srgb_to_lab
from .paint_plan import merge_runs_across_gaps


# A cell is repainted only when its captured Lab color is this much closer to
# something else than to its own color's rendering.  Below the margin the
# reading is ambiguous - sign texture, a cell boundary, compression - and
# repainting on ambiguity would oscillate instead of converge.
CLASSIFICATION_MARGIN_DELTA_E = 5.0

# Mismatches beyond this fraction of covered cells mean the capture itself is
# untrustworthy (the sign is occluded, the camera moved, a menu is open).
UNRELIABLE_CAPTURE_FRACTION = 0.4

# A color's observed rendering is trusted only while it stays within this
# distance of the lighting-normalized nominal value.  Further than that the
# whole group was painted as something else - a picker click that missed -
# and trusting its rendering would declare the mistake correct.
OBSERVED_COLOR_TRUST_DELTA_E = 20.0

# A color's rendering is measured only from at least this many cells; fewer
# and the median is one or two readings, which may themselves be the misses.
OBSERVED_COLOR_MIN_CELLS = 6

# A cell this far from every color's rendering looks like nothing the plan
# painted - bare sign when no bare reference is available, or a stray mark.
# Comfortably above the texture noise of a correctly painted cell, which the
# per-color spread raises further on a grainy sign: a cell is only ever
# called wrong once it sits this many spreads away from its own color, so a
# grain excursion across the midline between two near-identical colors is
# left alone.
UNEXPLAINED_DELTA_E = 12.0
SPREAD_MULTIPLIER = 5.0

# A bare reference is read from this many plan-unpainted cells at least;
# fewer and it comes from the capture of the freshly cleared sign instead.
BARE_REFERENCE_MIN_CELLS = 16

# A capture of the sign taken before painting stands in for the cleared one
# only if it looks bare: nine cells in ten within this distance of the
# median color.  A bare artist canvas reads about 6 on this measure and a
# painted one 40 and up, so the threshold sits well clear of both; the odd
# speck or dab left on a cleared sign lands in the tenth that is ignored.
BARE_CAPTURE_SPREAD_DELTA_E = 12.0
BARE_CAPTURE_SPREAD_PERCENTILE = 90.0

# Below this many screen pixels per logical cell the capture cannot tell one
# cell's color from its neighbours', so recoloring single cells is guesswork
# and only holes are repaired.
RECOLOR_MIN_CELL_PIXELS = 2.0

# Painting goes wrong in two shapes.  A stroke the game dropped is a hole.  A
# picker click that missed paints *every* cell of that color alike, so a
# color is either almost entirely right or almost entirely wrong: a color
# with at least this fraction of its cells read as wrong is taken to have
# been painted wrong, and all of it is repainted.
WRONG_COLOR_GROUP_FRACTION = 0.5

# Wrong-color verdicts that are neither - sprinkled a few per color through
# many colors - are the capture failing to resolve cells, not the painter
# failing to paint them; nothing in the painting loop miscolors one cell in
# five at random.  Past this fraction of the sign they are set aside, and
# the pass fills holes only.  Left in, a 512-wide sign read at two screen
# pixels per cell turned into a second painting's worth of strokes.
SCATTERED_WRONG_COLOR_FRACTION = 0.05

# ...unless there are few enough that repainting them is cheap anyway: below
# this many cells a touch-up is under a minute, and a stroke the game placed
# a cell off really does leave a few wrong cells in otherwise right colors.
SCATTERED_WRONG_COLOR_MIN_CELLS = 500


@dataclass(frozen=True, slots=True)
class Mismatch:
    """Which cells need repainting, and why, from one capture."""

    cells: np.ndarray
    blank: int
    wrong_color: int
    unexplained: int
    # Wrong-color verdicts set aside as implausible (see
    # SCATTERED_WRONG_COLOR_FRACTION); not counted in ``wrong_color``.
    discarded: int = 0

    @property
    def count(self) -> int:
        return int(self.cells.sum())


def plan_expectations(plan: PaintPlan) -> tuple[np.ndarray, np.ndarray]:
    """Replay the plan's coverage: the color index each cell ends up with.

    Returns ``(indices, palette)`` where ``indices`` holds a row into
    ``palette`` per cell and ``-1`` where no stroke reaches.  Groups are
    replayed in painting order, so a cell crossed early and repainted late
    reports its final color - the same invariant the planner builds on.
    """

    palette: list[RGBColor] = []
    palette_index: dict[RGBColor, int] = {}
    indices = np.full((plan.height, plan.width), -1, dtype=np.int32)
    for group in plan.color_groups:
        index = palette_index.get(group.color)
        if index is None:
            index = len(palette)
            palette.append(group.color)
            palette_index[group.color] = index
        radius = (max(1, group.brush_diameter) - 1) // 2
        for stroke in group.strokes:
            x0 = min(stroke.start_x, stroke.end_x)
            x1 = max(stroke.start_x, stroke.end_x)
            y0 = min(stroke.start_y, stroke.end_y)
            y1 = max(stroke.start_y, stroke.end_y)
            indices[
                max(0, y0 - radius) : min(plan.height, y1 + radius + 1),
                max(0, x0) : min(plan.width, x1 + 1),
            ] = index
    return indices, np.array(palette, dtype=np.uint8).reshape(-1, 3)


def sample_cell_colors(
    capture_rgb: np.ndarray,
    logical_width: int,
    logical_height: int,
    *,
    centers: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """One robust color per logical cell, read from a canvas capture.

    Samples a 3x3 median around each cell center, which shrugs off the odd
    noisy pixel and the sign's grain without blurring across cell borders the
    way an area resize would.  When cells are under three pixels across the
    window would reach into the neighbours, so the centre pixel alone is
    read instead.

    ``centers`` overrides where the cells are: ``(xs, ys)`` in capture pixels,
    one per column and one per row.  A measured texel grid puts the cells on
    the texture's own lattice rather than spread evenly over the rectangle,
    and reading them anywhere else would sample a neighbour's paint.
    """

    pixels = np.asarray(capture_rgb, dtype=np.float32)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("Canvas capture must be an RGB array")
    height, width = pixels.shape[:2]
    if centers is not None:
        raw_x, raw_y = centers
        if len(raw_x) != logical_width or len(raw_y) != logical_height:
            raise ValueError("Cell centres must match the logical size")
        centers_x = np.clip(np.floor(np.asarray(raw_x, dtype=np.float64)).astype(np.int64), 1, max(1, width - 2))
        centers_y = np.clip(np.floor(np.asarray(raw_y, dtype=np.float64)).astype(np.int64), 1, max(1, height - 2))
    else:
        centers_y = np.clip(
            ((np.arange(logical_height) + 0.5) * height / logical_height).astype(np.int64),
            1,
            max(1, height - 2),
        )
        centers_x = np.clip(
            ((np.arange(logical_width) + 0.5) * width / logical_width).astype(np.int64),
            1,
            max(1, width - 2),
        )
    if min(height / max(1, logical_height), width / max(1, logical_width)) < 3.0:
        return pixels[centers_y][:, centers_x, :3].copy()
    neighborhood = np.stack(
        [
            pixels[centers_y + dy][:, centers_x + dx, :3]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
    )
    return np.median(neighborhood, axis=0)


def fit_capture_lighting(
    sampled: np.ndarray,
    indices: np.ndarray,
    palette: np.ndarray,
    *,
    min_cells: int = OBSERVED_COLOR_MIN_CELLS,
) -> np.ndarray | None:
    """Fit the sign's global material response as one affine RGB transform.

    Live testing showed a painting that matched its plan perfectly still
    classified 76% wrong: the lit sign compresses dark colors, and with a
    tightly clustered palette that compression pushes a correct cell closer to
    a neighbouring palette entry than to its own.  One global transform
    fitted from the capture itself absorbs lighting and material - it cannot
    absorb per-cell painting mistakes, because a dozen parameters cannot bend
    thousands of cells individually.

    The fit sees one point per color - the median of that color's cells - so
    a large group painted wrong weighs no more than a small one, and the
    model is only as rich as the palette can pin down: a full affine mix
    needs eight colors, a per-channel gain and offset four, and fewer than
    that leave the capture as it is.  Colors that fit far worse than the
    rest are dropped and the fit repeated, so a color painted as something
    else cannot drag the transform toward hiding itself.  Returns the
    4x3 coefficient matrix, or ``None`` when there is too little to fit.
    """

    points: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    flat = np.asarray(sampled, dtype=np.float64).reshape(-1, 3)
    flat_indices = np.asarray(indices).reshape(-1)
    for index in range(len(palette)):
        cells = flat[flat_indices == index]
        if len(cells) >= min_cells:
            points.append(np.median(cells, axis=0))
            targets.append(np.asarray(palette[index], dtype=np.float64))
    if len(points) < 4:
        return None
    captured = np.array(points)
    wanted = np.array(targets)
    full = len(points) >= 8

    def fit(rows: np.ndarray) -> np.ndarray:
        coefficients = np.zeros((4, 3))
        if full:
            design = np.hstack([captured[rows], np.ones((int(rows.sum()), 1))])
            coefficients, *_ = np.linalg.lstsq(design, wanted[rows], rcond=None)
            return coefficients
        for channel in range(3):
            design = np.column_stack(
                [captured[rows, channel], np.ones(int(rows.sum()))]
            )
            gain_offset, *_ = np.linalg.lstsq(design, wanted[rows, channel], rcond=None)
            coefficients[channel, channel] = gain_offset[0]
            coefficients[3, channel] = gain_offset[1]
        return coefficients

    everything = np.ones(len(points), dtype=np.bool_)
    coefficients = fit(everything)
    design = np.hstack([captured, np.ones((len(points), 1))])
    residuals = np.linalg.norm(design @ coefficients - wanted, axis=1)
    typical = float(np.median(residuals))
    keep = residuals <= max(3.0 * typical, CLASSIFICATION_MARGIN_DELTA_E)
    if keep.sum() >= (8 if full else 4) and keep.sum() < len(points):
        coefficients = fit(keep)
    return coefficients


def apply_capture_lighting(
    samples: np.ndarray, coefficients: np.ndarray | None
) -> np.ndarray:
    """Map captured colors into nominal space with a fitted transform."""

    if coefficients is None:
        return np.asarray(samples, dtype=np.float32)
    flat = np.asarray(samples, dtype=np.float64).reshape(-1, 3)
    design = np.hstack([flat, np.ones((len(flat), 1))])
    corrected = np.clip(design @ coefficients, 0.0, 255.0)
    return corrected.reshape(np.shape(samples)).astype(np.float32)


def normalize_capture_lighting(
    sampled: np.ndarray, indices: np.ndarray, palette: np.ndarray
) -> np.ndarray:
    """Undo the sign's global material response before classifying cells."""

    coefficients = fit_capture_lighting(sampled, indices, palette)
    if coefficients is None:
        return sampled
    return apply_capture_lighting(sampled, coefficients)


def observed_palette_lab(
    sampled_lab: np.ndarray,
    indices: np.ndarray,
    palette_lab: np.ndarray,
    *,
    trust_delta_e: float = OBSERVED_COLOR_TRUST_DELTA_E,
    min_cells: int = OBSERVED_COLOR_MIN_CELLS,
) -> np.ndarray:
    """What each plan color actually looks like on this sign.

    The median Lab of the cells planned in a color is that color's rendering:
    most of them are painted right, and a median ignores the ones that are
    not.  A rendering implausibly far from the nominal color means the whole
    group went down as something else, so the nominal value is kept for it
    and its cells stay flagged.
    """

    observed = palette_lab.astype(np.float64).copy()
    flat_indices = indices.reshape(-1)
    flat_samples = sampled_lab.reshape(-1, 3)
    for index in range(len(palette_lab)):
        cells = flat_samples[flat_indices == index]
        if len(cells) < min_cells:
            continue
        median = np.median(cells, axis=0)
        if np.linalg.norm(median - palette_lab[index]) <= trust_delta_e:
            observed[index] = median
    return observed


def bare_reference_lab(
    sampled_lab: np.ndarray,
    indices: np.ndarray,
    bare_sampled_lab: np.ndarray | None = None,
    *,
    min_cells: int = BARE_REFERENCE_MIN_CELLS,
) -> np.ndarray | None:
    """The bare sign's color, as well as this capture can know it.

    Cells the plan never paints show the bare sign under the same lighting
    as the painting, so they are the best reference when there are enough of
    them.  Otherwise a capture of the freshly cleared sign stands in; its
    lighting may have drifted since, which the decisive margin tolerates.
    """

    uncovered = indices < 0
    if int(uncovered.sum()) >= min_cells:
        return np.median(sampled_lab[uncovered].reshape(-1, 3), axis=0)
    if bare_sampled_lab is not None and bare_sampled_lab.size:
        return np.median(bare_sampled_lab.reshape(-1, 3), axis=0)
    return None


def capture_looks_bare(
    sampled: np.ndarray,
    *,
    max_spread: float = BARE_CAPTURE_SPREAD_DELTA_E,
    percentile: float = BARE_CAPTURE_SPREAD_PERCENTILE,
) -> bool:
    """Whether per-cell readings of a sign describe one unpainted surface.

    ``sampled`` is the per-cell RGB reading of a capture (see
    :func:`sample_cell_colors`).  A sign that still carries an earlier
    artwork spreads its cells across many colors; a bare one is a single
    color with grain, a vignette, and perhaps a few specks, so nearly every
    cell sits close to the median.
    """

    cells = np.asarray(sampled, dtype=np.float32).reshape(-1, 3)
    if len(cells) < BARE_REFERENCE_MIN_CELLS:
        return False
    lab = _srgb_to_lab(cells)
    median = np.median(lab, axis=0)
    distance = np.sqrt(((lab - median.reshape(1, 3)) ** 2).sum(axis=1))
    return bool(np.percentile(distance, percentile) <= max_spread)


def classify_cells(
    sampled: np.ndarray,
    indices: np.ndarray,
    palette: np.ndarray,
    *,
    bare_sampled: np.ndarray | None = None,
    recolor: bool = True,
    margin: float = CLASSIFICATION_MARGIN_DELTA_E,
) -> Mismatch:
    """Decide which cells need repainting from one capture.

    ``sampled`` and ``bare_sampled`` are per-cell RGB readings (the latter of
    the cleared sign, optional).  The capture's lighting is normalized first,
    then every covered cell is measured against its own color's rendering,
    the nearest other rendering, and the bare sign:

    - *blank*: decisively nearer the bare sign than its own color - a stroke
      the game never registered.  Always repainted.
    - *wrong color*: decisively nearer another color's rendering - a picker
      click that missed.  Repainted only when ``recolor`` is set, because
      recoloring single cells with a brush or a capture too coarse to
      resolve them does more harm than good.
    - *unexplained*: far from every rendering and not known to be bare -
      treated as a hole, since that is what it almost always is.
    """

    covered = indices >= 0
    empty = Mismatch(np.zeros(indices.shape, dtype=np.bool_), 0, 0, 0)
    if len(palette) == 0 or not covered.any():
        return empty
    coefficients = fit_capture_lighting(sampled, indices, palette)
    normalized = apply_capture_lighting(sampled, coefficients)
    sampled_lab = _srgb_to_lab(normalized.reshape(-1, 3)).reshape(*indices.shape, 3)
    palette_lab = _srgb_to_lab(palette.astype(np.float32))
    observed = observed_palette_lab(sampled_lab, indices, palette_lab)

    own_index = np.where(covered, indices, 0)
    own, nearest_other = _own_and_nearest_other(sampled_lab, own_index, observed)

    # How far a correctly painted cell of each color wanders from that
    # color's rendering, measured from the capture itself: the sign's grain,
    # and on a large sign its lighting gradient.  Every "wrong" verdict has
    # to clear several of these, or a grain excursion across the midline
    # between two colors rendered a few units apart would be repainted.
    flat_own = own[covered]
    flat_indices = indices[covered]
    spread = np.zeros(len(palette))
    for index in range(len(palette)):
        residuals = flat_own[flat_indices == index]
        if len(residuals) >= OBSERVED_COLOR_MIN_CELLS:
            spread[index] = 1.4826 * float(
                np.median(np.abs(residuals - np.median(residuals)))
            )
    far_from_own = own > np.maximum(margin, SPREAD_MULTIPLIER * spread)[own_index]

    wrong = covered & far_from_own & (own - nearest_other > margin)

    bare_lab = None
    if bare_sampled is not None:
        bare_lab = bare_reference_lab(
            sampled_lab,
            indices,
            _srgb_to_lab(
                apply_capture_lighting(bare_sampled, coefficients).reshape(-1, 3)
            ),
        )
    else:
        bare_lab = bare_reference_lab(sampled_lab, indices, None)
    if bare_lab is not None:
        to_bare = np.sqrt(((sampled_lab - bare_lab.reshape(1, 1, 3)) ** 2).sum(axis=2))
        # Bare: decisively nearer the bare sign than its own color, and at
        # least as near it as any other color's rendering - a cell sitting
        # squarely on another color is a wrong color, not a hole, even when
        # that color happens to be nearer the wood than its own.
        blank = (
            covered
            & (own - to_bare > margin)
            & (to_bare - nearest_other < margin)
        )
    else:
        blank = np.zeros(indices.shape, dtype=np.bool_)

    unexplained_threshold = np.maximum(UNEXPLAINED_DELTA_E, SPREAD_MULTIPLIER * spread)[
        own_index
    ]
    unexplained = (
        covered
        & (own > unexplained_threshold)
        & (nearest_other > unexplained_threshold)
        & ~blank
    )

    wrong = wrong & ~blank & ~unexplained
    wrong, discarded = _plausible_wrong_color(wrong, indices, covered, len(palette))
    cells = blank | unexplained | (wrong if recolor else np.zeros_like(wrong))
    return Mismatch(
        cells=cells,
        blank=int(blank.sum()),
        wrong_color=int(wrong.sum()),
        unexplained=int(unexplained.sum()),
        discarded=discarded,
    )


def _plausible_wrong_color(
    wrong: np.ndarray, indices: np.ndarray, covered: np.ndarray, colors: int
) -> tuple[np.ndarray, int]:
    """Keep wrong-color verdicts that describe how painting actually fails.

    Colors read as mostly wrong were painted wrong and are kept whole.  The
    rest are scattered single cells; a few are tolerated, but once they cover
    more of the sign than :data:`SCATTERED_WRONG_COLOR_FRACTION` the capture
    is not resolving cells and they are all set aside.
    """

    total_wrong = int(wrong.sum())
    if total_wrong == 0 or colors == 0:
        return wrong, 0
    flat_indices = indices[covered]
    per_color_cells = np.bincount(flat_indices, minlength=colors)
    per_color_wrong = np.bincount(flat_indices, weights=wrong[covered], minlength=colors)
    whole = per_color_wrong >= WRONG_COLOR_GROUP_FRACTION * np.maximum(per_color_cells, 1)
    concentrated = wrong & whole[np.where(covered, indices, 0)] & covered
    scattered = wrong & ~concentrated
    scattered_count = int(scattered.sum())
    allowed = max(
        SCATTERED_WRONG_COLOR_MIN_CELLS,
        SCATTERED_WRONG_COLOR_FRACTION * int(covered.sum()),
    )
    if scattered_count <= allowed:
        return wrong, 0
    return concentrated, scattered_count


def mismatched_cells(
    sampled: np.ndarray,
    indices: np.ndarray,
    palette: np.ndarray,
    *,
    margin: float = CLASSIFICATION_MARGIN_DELTA_E,
) -> np.ndarray:
    """Cells whose captured color decisively belongs to a different plan color.

    Kept as the plain two-way comparison against nominal palette values for
    callers that bring already-normalized samples; painting uses
    :func:`classify_cells`, which also knows about bare sign and twins.
    """

    if len(palette) == 0:
        return np.zeros(indices.shape, dtype=np.bool_)
    palette_lab = _srgb_to_lab(palette.astype(np.float32)).astype(np.float64)
    sampled_lab = _srgb_to_lab(
        np.asarray(sampled, dtype=np.float32).reshape(-1, 3)
    ).reshape(*indices.shape, 3)
    covered = indices >= 0
    own, nearest_other = _own_and_nearest_other(
        sampled_lab, np.where(covered, indices, 0), palette_lab
    )
    return covered & (own - nearest_other > margin)


def _own_and_nearest_other(
    sampled_lab: np.ndarray, own_index: np.ndarray, colors_lab: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Per cell, the distance to its own color and to the nearest other one.

    Streams over the colors rather than forming a cells-by-colors table: a
    two-thousand-wide sign has over a million cells, and the table would be
    hundreds of megabytes for a handful of distances per cell.
    """

    samples = np.asarray(sampled_lab, dtype=np.float64)
    own = np.sqrt(((samples - colors_lab[own_index]) ** 2).sum(axis=2))
    nearest_other = np.full(own_index.shape, np.inf)
    for index in range(len(colors_lab)):
        distance = np.sqrt(((samples - colors_lab[index]) ** 2).sum(axis=2))
        distance[own_index == index] = np.inf
        np.minimum(nearest_other, distance, out=nearest_other)
    return own, nearest_other


def touch_up_plan(
    mismatch: np.ndarray, indices: np.ndarray, palette: np.ndarray
) -> PaintPlan:
    """Single-cell strokes that repaint exactly the mismatched cells.

    Strokes never cross cells of other colors - a touch-up must fix cells, not
    introduce a second generation of collateral overpainting.
    """

    height, width = indices.shape
    groups: list[ColorGroup] = []
    for index in np.unique(indices[mismatch]):
        must = mismatch & (indices == index)
        strokes = merge_runs_across_gaps(must, None, 0)
        if not strokes:
            continue
        color: RGBColor = tuple(int(channel) for channel in palette[index])  # type: ignore[assignment]
        groups.append(
            ColorGroup(
                color=color,
                strokes=tuple(strokes),
                pixel_count=int(must.sum()),
            )
        )
    groups.sort(key=lambda group: -group.pixel_count)
    return PaintPlan(width=width, height=height, color_groups=tuple(groups))


__all__ = [
    "BARE_CAPTURE_SPREAD_DELTA_E",
    "BARE_CAPTURE_SPREAD_PERCENTILE",
    "BARE_REFERENCE_MIN_CELLS",
    "CLASSIFICATION_MARGIN_DELTA_E",
    "Mismatch",
    "OBSERVED_COLOR_TRUST_DELTA_E",
    "RECOLOR_MIN_CELL_PIXELS",
    "SCATTERED_WRONG_COLOR_FRACTION",
    "SCATTERED_WRONG_COLOR_MIN_CELLS",
    "SPREAD_MULTIPLIER",
    "UNEXPLAINED_DELTA_E",
    "UNRELIABLE_CAPTURE_FRACTION",
    "WRONG_COLOR_GROUP_FRACTION",
    "apply_capture_lighting",
    "bare_reference_lab",
    "capture_looks_bare",
    "classify_cells",
    "fit_capture_lighting",
    "mismatched_cells",
    "normalize_capture_lighting",
    "observed_palette_lab",
    "plan_expectations",
    "sample_cell_colors",
    "touch_up_plan",
]
