"""Artist-style paint planning: large fills first, smaller brushes for detail.

The classic planner in :mod:`app.paint_plan` reproduces the quantized image
row by row with a one-cell brush.  This module plans like a painter instead:
perceptually indistinguishable colors are merged, insignificant specks are
absorbed by their surroundings, and the largest color regions are painted
first with the largest brush that cannot spill anywhere harmful - sweeping
freely across pixels that a later color repaints anyway.  Only the remaining
detail falls back to single-cell strokes.  Exact mode never enters this
module, so the raw pipeline keeps its behavior untouched.

Correctness rests on one invariant shared with the classic ``overpaint_gap``
merging: colors are painted most-common first, and a stroke may touch only
cells of its own color or of colors painted later.  Every cell's final visit
therefore paints its final color, whatever was smeared across it earlier.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from PIL import Image

from .models import (
    ColorGroup,
    PaintMode,
    PaintPlan,
    RGBColor,
    Stroke,
)
from .paint_plan import (
    PaintPlanTiming,
    PlanImage,
    _as_rgb_and_mask,
    _ordered_color_index_map,
    analyze_paint_plan,
    merge_runs_across_gaps,
)


# Fallback ceiling for profiles whose brush reach has not been measured yet.
# Once ``BrushCapabilities.max_brush_pixels`` is known it replaces this
# entirely, because the measured value is what the Size field can really do
# rather than what has been observed to work elsewhere.
_ASSUMED_MAX_BRUSH_PIXELS = 64.0

# Rough cost of one extra stroke in cells of mouse travel: the inter-stroke
# delay, the button press, and the hop to the stroke's start point.
_STROKE_OVERHEAD_CELLS = 10


@dataclass(frozen=True, slots=True)
class BrushCapabilities:
    """What the calibrated profile lets the planner physically do.

    ``sizing`` requires the calibrated Size value box and a measured brush
    model plus the automatic-brush-sizing option; without it every stroke stays
    one cell.  ``cell_pixels`` is the physical size of one logical cell and
    ``max_brush_pixels`` the widest band the profile actually measured Rust
    painting, which together keep every planned brush inside what the Size
    field can reach.  Zero means unknown for either.
    """

    sizing: bool = False
    cell_pixels: float = 0.0
    max_brush_pixels: float = 0.0


@dataclass(frozen=True, slots=True)
class OptimizerOptions:
    """Internal tuning for one optimization mode.

    Kept as plain data so individual values can later be surfaced as advanced
    user settings without restructuring the pipeline.
    """

    # Colors closer than this CIE76 delta-E merge into one paint pass.
    merge_tolerance: float
    # Connected regions smaller than this many cells may be absorbed by a
    # neighbor, but only when the contrast between them stays below the limit.
    min_region_area: int
    region_contrast_limit: float
    # Largest run of later-painted cells a detail stroke may paint across.
    overpaint_gap: int
    # Descending odd brush diameters (in logical cells) tried above one cell.
    brush_diameters: tuple[int, ...]
    # Execution costs expressed in cells of mouse travel, so a candidate pass
    # can be weighed directly against the detail strokes it would replace.
    # Searching a fresh diameter measures the preview repeatedly; revisiting a
    # diameter replays one remembered slider click.
    resize_cost_cells: int
    revisit_cost_cells: int


MODE_OPTIONS: dict[PaintMode, OptimizerOptions] = {
    PaintMode.QUALITY: OptimizerOptions(
        merge_tolerance=2.5,
        min_region_area=3,
        region_contrast_limit=6.0,
        overpaint_gap=4,
        brush_diameters=(3,),
        resize_cost_cells=450,
        revisit_cost_cells=110,
    ),
    PaintMode.BALANCED: OptimizerOptions(
        merge_tolerance=5.0,
        min_region_area=6,
        region_contrast_limit=11.0,
        overpaint_gap=8,
        brush_diameters=(5, 3),
        resize_cost_cells=400,
        revisit_cost_cells=90,
    ),
    PaintMode.FAST: OptimizerOptions(
        merge_tolerance=9.0,
        min_region_area=14,
        region_contrast_limit=18.0,
        overpaint_gap=24,
        brush_diameters=(7, 5, 3),
        resize_cost_cells=350,
        revisit_cost_cells=80,
    ),
}


@dataclass(frozen=True, slots=True)
class OptimizationStatistics:
    """What the optimizer changed and how visible the change should be."""

    mode: str
    input_colors: int
    output_colors: int
    stroke_count: int
    brush_size_changes: int
    mean_delta_e: float
    similarity_percent: float


@dataclass(slots=True)
class OptimizedPlan:
    """An executable plan plus the exact image it will reproduce."""

    plan: PaintPlan
    image: Image.Image
    paint_mask: np.ndarray
    statistics: OptimizationStatistics


def mode_options(mode: PaintMode | str, *, preserve_dither: bool = False) -> OptimizerOptions:
    """The preset for a mode, optionally adjusted for a dithered source.

    Dithering builds gradients out of deliberate single-cell speckle, which is
    exactly what region cleanup would erase, so cleanup is disabled and color
    merging is reined in when the source was dithered on purpose.
    """

    options = MODE_OPTIONS[PaintMode(mode)]
    if preserve_dither:
        options = replace(
            options,
            min_region_area=0,
            merge_tolerance=min(options.merge_tolerance, 2.5),
        )
    return options


# ---------------------------------------------------------------------- color


def _srgb_to_lab(colors: np.ndarray) -> np.ndarray:
    """Convert ``(n, 3)`` 8-bit sRGB rows to CIE Lab under D65."""

    rgb = np.asarray(colors, dtype=np.float32).reshape(-1, 3) / 255.0
    linear = np.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055) ** 2.4)
    matrix = np.array(
        [
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ],
        dtype=np.float32,
    )
    xyz = linear @ matrix.T
    xyz /= np.array([0.95047, 1.0, 1.08883], dtype=np.float32)
    epsilon = 216.0 / 24389.0
    kappa = 24389.0 / 27.0
    transfer = np.where(xyz > epsilon, np.cbrt(xyz), (kappa * xyz + 16.0) / 116.0)
    lab = np.empty_like(transfer)
    lab[:, 0] = 116.0 * transfer[:, 1] - 16.0
    lab[:, 1] = 500.0 * (transfer[:, 0] - transfer[:, 1])
    lab[:, 2] = 200.0 * (transfer[:, 1] - transfer[:, 2])
    return lab


def merge_similar_colors(
    rgb: np.ndarray, mask: np.ndarray, tolerance: float
) -> np.ndarray:
    """Collapse painted colors within ``tolerance`` delta-E onto one center.

    Popular colors become the centers, so a JPEG halo of near-whites lands on
    the dominant white rather than each shade earning its own paint pass.
    """

    if tolerance <= 0:
        return rgb
    painted = rgb[mask]
    if painted.size == 0:
        return rgb
    unique, inverse, counts = np.unique(
        painted.reshape(-1, 3), axis=0, return_inverse=True, return_counts=True
    )
    if len(unique) <= 1:
        return rgb
    lab = _srgb_to_lab(unique)
    replacement = np.arange(len(unique))
    center_indices: list[int] = []
    for index in np.argsort(-counts, kind="stable"):
        candidate = int(index)
        if center_indices:
            centers = lab[center_indices]
            distances = np.sqrt(((centers - lab[candidate]) ** 2).sum(axis=1))
            nearest = int(np.argmin(distances))
            if float(distances[nearest]) <= tolerance:
                replacement[candidate] = center_indices[nearest]
                continue
        center_indices.append(candidate)
    if len(center_indices) == len(unique):
        return rgb
    merged = rgb.copy()
    merged[mask] = unique[replacement][inverse.reshape(-1)]
    return merged


# -------------------------------------------------------------------- regions


def _label_regions(index_map: np.ndarray) -> tuple[np.ndarray, int]:
    """4-connected component labels for painted cells; ``-1`` elsewhere.

    Rows are first split into same-color runs, then runs touching vertically
    with the same color are unioned.  Working on runs instead of pixels keeps
    the Python union-find small even for noisy photographic sources.
    """

    height, width = index_map.shape
    flat = index_map.reshape(-1)
    previous = np.empty_like(flat)
    previous[0] = -2
    previous[1:] = flat[:-1]
    row_start = np.zeros(flat.size, dtype=np.bool_)
    row_start[::width] = True
    starts = (flat >= 0) & (row_start | (flat != previous))
    run_count = int(starts.sum())
    if run_count == 0:
        return np.full(index_map.shape, -1, dtype=np.int32), 0
    run_of_pixel = (np.cumsum(starts) - 1).reshape(height, width)

    parent = np.arange(run_count, dtype=np.int64)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    painted = index_map >= 0
    touching = painted[1:] & painted[:-1] & (index_map[1:] == index_map[:-1])
    pairs = np.unique(
        np.stack((run_of_pixel[:-1][touching], run_of_pixel[1:][touching]), axis=1),
        axis=0,
    )
    for upper, lower in pairs:
        root_upper, root_lower = find(int(upper)), find(int(lower))
        if root_upper != root_lower:
            parent[root_lower] = root_upper
    roots = np.array([find(index) for index in range(run_count)], dtype=np.int64)
    _, label_of_run = np.unique(roots, return_inverse=True)
    labels = np.full(flat.shape, -1, dtype=np.int32)
    labels[flat >= 0] = label_of_run.reshape(-1)[run_of_pixel.reshape(-1)[flat >= 0]].astype(
        np.int32
    )
    return labels.reshape(index_map.shape), int(label_of_run.max()) + 1


def absorb_insignificant_regions(
    rgb: np.ndarray,
    mask: np.ndarray,
    min_area: int,
    contrast_limit: float,
    *,
    max_rounds: int = 3,
) -> np.ndarray:
    """Recolor tiny low-contrast regions to their dominant similar neighbor.

    Area alone never condemns a region: a tiny black pupil on a white face has
    huge contrast and survives, while a lone near-white speck inside white is
    absorbed.  Absorption only flows uphill in area, so two adjacent specks
    cannot trade colors forever.
    """

    if min_area <= 1 or contrast_limit <= 0 or not mask.any():
        return rgb
    result = rgb
    for _round in range(max_rounds):
        index_map, colors, _counts = _ordered_color_index_map(result, mask, "frequency")
        if len(colors) <= 1:
            break
        labels, label_count = _label_regions(index_map)
        if label_count == 0:
            break
        areas = np.bincount(labels[labels >= 0].reshape(-1), minlength=label_count)
        if not (areas < min_area).any():
            break
        color_array = np.array(colors, dtype=np.uint8)
        lab = _srgb_to_lab(color_array)
        label_color = np.zeros(label_count, dtype=np.int64)
        flat_labels = labels.reshape(-1)
        flat_index = index_map.reshape(-1)
        valid = flat_labels >= 0
        label_color[flat_labels[valid]] = flat_index[valid]

        contact_pairs: list[np.ndarray] = []
        for side_a, side_b in (
            (labels[:, :-1], labels[:, 1:]),
            (labels[:-1, :], labels[1:, :]),
        ):
            differs = (side_a >= 0) & (side_b >= 0) & (side_a != side_b)
            if differs.any():
                stacked = np.stack((side_a[differs], side_b[differs]), axis=1)
                contact_pairs.append(stacked)
                contact_pairs.append(stacked[:, ::-1])
        if not contact_pairs:
            break
        pairs = np.concatenate(contact_pairs)
        unique_pairs, contacts = np.unique(pairs, axis=0, return_counts=True)

        absorbed_color = label_color.copy()
        changed = False
        boundaries = np.searchsorted(
            unique_pairs[:, 0], np.arange(label_count + 1)
        )
        for label in np.flatnonzero(areas < min_area):
            begin, end = boundaries[label], boundaries[label + 1]
            if begin == end:
                continue
            neighbors = unique_pairs[begin:end, 1]
            neighbor_contacts = contacts[begin:end]
            own_area = areas[label]
            # Only grow into something at least as established, so absorption
            # terminates instead of ping-ponging between two specks.
            eligible = (areas[neighbors] >= min_area) | (areas[neighbors] > own_area)
            if not eligible.any():
                continue
            neighbors = neighbors[eligible]
            neighbor_contacts = neighbor_contacts[eligible]
            deltas = np.sqrt(
                ((lab[label_color[neighbors]] - lab[label_color[label]]) ** 2).sum(axis=1)
            )
            within = deltas <= contrast_limit
            if not within.any():
                continue
            candidates = np.flatnonzero(within)
            best = candidates[int(np.argmax(neighbor_contacts[candidates]))]
            absorbed_color[label] = label_color[neighbors[best]]
            changed = True
        if not changed:
            break
        updated = result.copy()
        updated[mask] = color_array[absorbed_color[labels[mask]]]
        result = updated
    return result


# ------------------------------------------------------------- brush planning


def _brush_is_achievable(diameter: int, capabilities: BrushCapabilities) -> bool:
    """Whether Rust's Size field can render a ``diameter``-cell pass.

    With no cell size known there is nothing to compare against, so every
    candidate stays.
    """

    cell = capabilities.cell_pixels
    if cell <= 0:
        return True
    ceiling = capabilities.max_brush_pixels
    if ceiling <= 0:
        ceiling = _ASSUMED_MAX_BRUSH_PIXELS
    return diameter * cell <= ceiling


def _safety_offsets(diameter: int) -> np.ndarray:
    """Neighbourhood that must be paintable for a brush centered on a cell.

    The reach includes one guard cell beyond the nominal footprint so slider
    rounding and sub-pixel alignment can never push paint onto a cell that is
    only repainted earlier - or never repainted at all.  Planning always uses
    the square brush's spill reach (the worst case), so whichever solid shape
    is actually selected in Rust stays safe.
    """

    coverage_radius = (diameter - 1) // 2
    reach = coverage_radius + 1
    span = np.arange(-reach, reach + 1)
    grid_y, grid_x = np.meshgrid(span, span, indexing="ij")
    return np.stack((grid_y.reshape(-1), grid_x.reshape(-1)), axis=1)


def _erode(allowed: np.ndarray, offsets: np.ndarray) -> np.ndarray:
    """Cells whose whole offset neighbourhood is allowed (edges count as not)."""

    height, width = allowed.shape
    radius = int(np.abs(offsets).max()) if offsets.size else 0
    if radius == 0:
        return allowed.copy()
    padded = np.zeros((height + 2 * radius, width + 2 * radius), dtype=np.bool_)
    padded[radius : radius + height, radius : radius + width] = allowed
    result = allowed.copy()
    for offset_y, offset_x in offsets:
        result &= padded[
            radius + offset_y : radius + offset_y + height,
            radius + offset_x : radius + offset_x + width,
        ]
    return result


def _plan_brush_pass(
    safe: np.ndarray, uncovered: np.ndarray, diameter: int
) -> tuple[list[Stroke], int]:
    """Sweep rows of safe centers, covering the band ``diameter`` cells tall.

    ``uncovered`` is consumed in place.  Rows advance ``diameter - 1`` cells
    after a stroke so adjacent bands overlap by one row, which hides the seam
    if the physical brush lands slightly under its target size.
    """

    coverage_radius = (diameter - 1) // 2
    height, width = safe.shape
    split_gap = max(4, 2 * diameter)
    strokes: list[Stroke] = []
    covered = 0
    y = 0
    while y < height:
        row_safe = safe[y]
        if not row_safe.any():
            y += 1
            continue
        band_top = max(0, y - coverage_radius)
        band_bottom = min(height, y + coverage_radius + 1)
        useful = uncovered[band_top:band_bottom].any(axis=0)
        emitted = False
        columns = np.flatnonzero(row_safe)
        run_breaks = np.flatnonzero(np.diff(columns) > 1)
        run_starts = columns[np.concatenate(([0], run_breaks + 1))]
        run_ends = columns[np.concatenate((run_breaks, [columns.size - 1]))]
        for run_start, run_end in zip(run_starts, run_ends):
            span_useful = np.flatnonzero(useful[run_start : run_end + 1])
            if span_useful.size == 0:
                continue
            gaps = np.diff(span_useful) - 1
            boundaries = np.flatnonzero(gaps > split_gap)
            starts = span_useful[np.concatenate(([0], boundaries + 1))] + run_start
            ends = (
                span_useful[np.concatenate((boundaries, [span_useful.size - 1]))]
                + run_start
            )
            for start_x, end_x in zip(starts, ends):
                strokes.append(Stroke(int(start_x), int(y), int(end_x), int(y)))
                block = uncovered[band_top:band_bottom, start_x : end_x + 1]
                covered += int(block.sum())
                block[:] = False
                emitted = True
        y += max(1, diameter - 1) if emitted else 1
    return strokes, covered


def _plan_detail_runs(
    uncovered: np.ndarray, allowed: np.ndarray, max_gap: int
) -> list[Stroke]:
    """Single-cell runs over the remaining target, crossing allowed gaps.

    A gap may only be painted through when every cell in it is repainted later
    (or is this color anyway), and when the gap is short enough to be worth it.
    """

    blocked_cumulative = np.cumsum(~allowed, axis=1) if max_gap > 0 else None
    return merge_runs_across_gaps(uncovered, blocked_cumulative, max_gap)


def _serpentine(strokes: list[Stroke]) -> list[Stroke]:
    """Order horizontal strokes top-to-bottom, alternating sweep direction.

    Ending each row where the next row begins keeps the mouse from jumping
    back across the canvas between rows.
    """

    by_row: dict[int, list[Stroke]] = {}
    for stroke in strokes:
        by_row.setdefault(stroke.start_y, []).append(stroke)
    ordered: list[Stroke] = []
    reverse = False
    for y in sorted(by_row):
        row = sorted(by_row[y], key=lambda stroke: min(stroke.start_x, stroke.end_x))
        if reverse:
            ordered.extend(
                Stroke(max(s.start_x, s.end_x), y, min(s.start_x, s.end_x), y)
                for s in reversed(row)
            )
        else:
            ordered.extend(
                Stroke(min(s.start_x, s.end_x), y, max(s.start_x, s.end_x), y)
                for s in row
            )
        reverse = not reverse
    return ordered


def _row_run_count(cells: np.ndarray) -> int:
    """How many horizontal runs the cells split into (one stroke each)."""

    previous = np.zeros_like(cells)
    previous[:, 1:] = cells[:, :-1]
    return int((cells & ~previous).sum())


def _evaluate_pass(
    allowed: np.ndarray,
    uncovered: np.ndarray,
    diameter: int,
) -> tuple[np.ndarray, list[Stroke], int, float]:
    """Plan one candidate pass and score its benefit over painting as detail.

    The benefit is measured in cells of mouse travel: what the covered cells
    would have cost as single-cell runs, minus what the pass itself travels.
    Fixed per-pass costs (a brush resize) are the caller's to subtract, since
    they depend on what is already selected.
    """

    safe = _erode(allowed, _safety_offsets(diameter))
    trial = uncovered.copy()
    strokes, covered = _plan_brush_pass(safe, trial, diameter)
    replaced_runs = _row_run_count(uncovered & ~trial)
    pass_travel = sum(stroke.pixel_count for stroke in strokes)
    benefit = float(
        covered
        + _STROKE_OVERHEAD_CELLS * replaced_runs
        - pass_travel
        - _STROKE_OVERHEAD_CELLS * len(strokes)
    )
    return trial, strokes, covered, benefit


# ----------------------------------------------------------------- main entry


def simplify_colors(
    source: PlanImage,
    mode: PaintMode | str,
    *,
    options: OptimizerOptions | None = None,
    paint_mask: np.ndarray | None = None,
) -> tuple[Image.Image, np.ndarray]:
    """Run only the color simplification a mode would apply, no brush planning.

    Returns the merged/cleaned RGBA image and its paint mask.  This is what a
    preview backdrop needs, at a fraction of a full plan's cost.
    """

    resolved_mode = PaintMode(mode)
    options = options or MODE_OPTIONS[resolved_mode]
    rgb, mask = _as_rgb_and_mask(source, paint_mask)
    rgb = merge_similar_colors(rgb, mask, options.merge_tolerance)
    rgb = absorb_insignificant_regions(
        rgb, mask, options.min_region_area, options.region_contrast_limit
    )
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    image = Image.fromarray(np.dstack((rgb, alpha)), mode="RGBA")
    return image, np.asarray(mask, dtype=np.bool_).copy()


def optimize_paint_plan(
    source: PlanImage,
    mode: PaintMode | str,
    *,
    capabilities: BrushCapabilities | None = None,
    options: OptimizerOptions | None = None,
    paint_mask: np.ndarray | None = None,
    timing: PaintPlanTiming | None = None,
) -> OptimizedPlan:
    """Build an optimized plan plus the exact image it reproduces.

    Exact mode deliberately has no path through here; callers keep using
    :func:`app.paint_plan.generate_paint_plan` for it.
    """

    resolved_mode = PaintMode(mode)
    if resolved_mode is PaintMode.EXACT:
        raise ValueError("Exact mode uses generate_paint_plan, not the optimizer")
    capabilities = capabilities or BrushCapabilities()
    options = options or MODE_OPTIONS[resolved_mode]

    rgb, mask = _as_rgb_and_mask(source, paint_mask)
    original_rgb = rgb
    rgb = merge_similar_colors(rgb, mask, options.merge_tolerance)
    rgb = absorb_insignificant_regions(
        rgb, mask, options.min_region_area, options.region_contrast_limit
    )
    index_map, colors, _counts = _ordered_color_index_map(rgb, mask, "frequency")
    height, width = mask.shape

    diameters: tuple[int, ...] = options.brush_diameters if capabilities.sizing else ()
    largest_useful = max(1, min(height, width) // 3)
    diameters = tuple(
        diameter
        for diameter in diameters
        if 1 < diameter <= largest_useful
        and _brush_is_achievable(diameter, capabilities)
    )

    groups: list[ColorGroup] = []
    last_diameter = 1
    # Mirrors the painter's brush changes: each switch to a new diameter costs
    # a click and a typed number, so the planner only pays for one when the
    # wider brush saves more travel than the switch costs.
    searched_diameters: set[int] = set()
    for color_index, color in enumerate(colors):
        allowed = index_map >= color_index
        uncovered = index_map == color_index
        for diameter in diameters:
            remaining_cells = int(uncovered.sum())
            if remaining_cells < 32:
                break
            if diameter == last_diameter:
                switch_cost = 0.0
            elif diameter in searched_diameters:
                switch_cost = float(options.revisit_cost_cells)
            else:
                switch_cost = float(options.resize_cost_cells)
            trial, strokes, covered, benefit = _evaluate_pass(
                allowed, uncovered, diameter
            )
            if not strokes or benefit <= switch_cost:
                continue
            uncovered = trial
            groups.append(
                ColorGroup(
                    color=color,
                    strokes=tuple(_serpentine(strokes)),
                    pixel_count=covered,
                    brush_diameter=diameter,
                )
            )
            last_diameter = diameter
            searched_diameters.add(diameter)
        if uncovered.any():
            detail = _plan_detail_runs(uncovered, allowed, options.overpaint_gap)
            groups.append(
                ColorGroup(
                    color=color,
                    strokes=tuple(_serpentine(detail)),
                    pixel_count=int(uncovered.sum()),
                    brush_diameter=1,
                )
            )
            last_diameter = 1

    plan = PaintPlan(
        width=width,
        height=height,
        color_groups=tuple(groups),
        unpainted_pixels=int(mask.size - np.count_nonzero(mask)),
    )
    plan = replace(plan, _statistics=analyze_paint_plan(plan, timing))

    statistics = _build_statistics(
        resolved_mode, original_rgb, rgb, mask, colors, plan
    )
    alpha = np.where(mask, 255, 0).astype(np.uint8)
    image = Image.fromarray(np.dstack((rgb, alpha)), mode="RGBA")
    return OptimizedPlan(
        plan=plan,
        image=image,
        paint_mask=np.asarray(mask, dtype=np.bool_).copy(),
        statistics=statistics,
    )


def _build_statistics(
    mode: PaintMode,
    original_rgb: np.ndarray,
    final_rgb: np.ndarray,
    mask: np.ndarray,
    colors: list[RGBColor],
    plan: PaintPlan,
) -> OptimizationStatistics:
    painted_original = original_rgb[mask]
    if painted_original.size:
        input_colors = len(np.unique(painted_original.reshape(-1, 3), axis=0))
        deltas = np.sqrt(
            (
                (_srgb_to_lab(painted_original) - _srgb_to_lab(final_rgb[mask])) ** 2
            ).sum(axis=1)
        )
        mean_delta_e = float(deltas.mean())
    else:
        input_colors = 0
        mean_delta_e = 0.0

    size_changes = 0
    previous_diameter = 1
    for group in plan.color_groups:
        if group.brush_diameter != previous_diameter:
            size_changes += 1
            previous_diameter = group.brush_diameter

    return OptimizationStatistics(
        mode=mode.value,
        input_colors=input_colors,
        output_colors=len(colors),
        stroke_count=plan.stroke_count,
        brush_size_changes=size_changes,
        mean_delta_e=mean_delta_e,
        similarity_percent=max(0.0, 100.0 - 2.0 * mean_delta_e),
    )


__all__ = [
    "BrushCapabilities",
    "MODE_OPTIONS",
    "OptimizationStatistics",
    "OptimizedPlan",
    "OptimizerOptions",
    "absorb_insignificant_regions",
    "merge_similar_colors",
    "mode_options",
    "optimize_paint_plan",
    "simplify_colors",
]
