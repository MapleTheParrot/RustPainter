"""Generate reliable color-grouped horizontal paint strokes."""

from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot
from typing import Literal, TypeAlias

import numpy as np
from PIL import Image

from .models import (
    ColorGroup,
    PaintPlan,
    PaintStatistics,
    ProcessedImage,
    RGBColor,
    Stroke,
)


PlanImage: TypeAlias = ProcessedImage | Image.Image | np.ndarray
ColorOrder: TypeAlias = Literal["first_seen", "frequency", "rgb"]


@dataclass(frozen=True, slots=True)
class PaintPlanTiming:
    """Simple logical-space inputs for a preview-only time estimate."""

    stroke_speed_pixels_per_second: float = 80.0
    point_duration_seconds: float = 0.03
    delay_between_strokes_seconds: float = 0.02
    color_selection_seconds: float = 0.12
    delay_between_colors_seconds: float = 0.10

    def __post_init__(self) -> None:
        values = (
            self.stroke_speed_pixels_per_second,
            self.point_duration_seconds,
            self.delay_between_strokes_seconds,
            self.color_selection_seconds,
            self.delay_between_colors_seconds,
        )
        if self.stroke_speed_pixels_per_second <= 0 or any(v < 0 for v in values[1:]):
            raise ValueError("Timing values must be non-negative and speed must be positive")


def _as_rgb_and_mask(
    source: PlanImage, paint_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(source, ProcessedImage):
        rgba = np.asarray(source.image.convert("RGBA"), dtype=np.uint8)
        default_mask = np.asarray(source.paint_mask, dtype=np.bool_)
        if default_mask.shape != rgba.shape[:2]:
            raise ValueError("Processed image mask dimensions do not match its image")
        default_mask = default_mask & (rgba[:, :, 3] > 0)
    elif isinstance(source, Image.Image):
        rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8)
        default_mask = rgba[:, :, 3] > 0
    else:
        array = np.asarray(source)
        if array.ndim != 3 or array.shape[2] not in (3, 4):
            raise ValueError("Plan image must have shape (height, width, 3 or 4)")
        if not np.issubdtype(array.dtype, np.number):
            raise ValueError("Plan image channels must be numeric")
        if np.any(array < 0) or np.any(array > 255):
            raise ValueError("Plan image channels must be in the range 0..255")
        array = array.astype(np.uint8, copy=False)
        if array.shape[2] == 3:
            rgba = np.dstack(
                (array, np.full(array.shape[:2], 255, dtype=np.uint8))
            )
        else:
            rgba = array
        default_mask = rgba[:, :, 3] > 0

    if rgba.shape[0] == 0 or rgba.shape[1] == 0:
        raise ValueError("Plan image dimensions must be positive")

    if paint_mask is None:
        mask = default_mask
    else:
        mask = np.asarray(paint_mask, dtype=np.bool_)
        if mask.shape != rgba.shape[:2]:
            raise ValueError("Paint mask dimensions must match the image")
        # Alpha zero always remains a final safety backstop.
        mask = mask & (rgba[:, :, 3] > 0)
    return rgba[:, :, :3], mask


def group_horizontal_runs(
    source: PlanImage,
    paint_mask: np.ndarray | None = None,
) -> dict[RGBColor, tuple[Stroke, ...]]:
    """Return maximal same-color runs, scanned top-to-bottom and left-to-right."""

    rgb, mask = _as_rgb_and_mask(source, paint_mask)
    height, width = mask.shape
    grouped: dict[RGBColor, list[Stroke]] = {}

    for y in range(height):
        x = 0
        while x < width:
            if not mask[y, x]:
                x += 1
                continue
            color: RGBColor = tuple(int(channel) for channel in rgb[y, x])  # type: ignore[assignment]
            start_x = x
            x += 1
            while (
                x < width
                and mask[y, x]
                and bool(np.array_equal(rgb[y, x], rgb[y, start_x]))
            ):
                x += 1
            grouped.setdefault(color, []).append(Stroke(start_x, y, x - 1, y))

    return {color: tuple(strokes) for color, strokes in grouped.items()}


def horizontal_runs_for_color(
    source: PlanImage,
    color: RGBColor,
    paint_mask: np.ndarray | None = None,
) -> tuple[Stroke, ...]:
    return group_horizontal_runs(source, paint_mask).get(color, ())


def generate_horizontal_runs(
    source: PlanImage,
    paint_mask: np.ndarray | None = None,
    *,
    color_order: ColorOrder = "first_seen",
) -> tuple[ColorGroup, ...]:
    """Build color groups containing their maximal horizontal runs."""

    grouped = group_horizontal_runs(source, paint_mask)
    groups = [
        ColorGroup(
            color=color,
            strokes=strokes,
            pixel_count=sum(stroke.pixel_count for stroke in strokes),
        )
        for color, strokes in grouped.items()
    ]
    if color_order == "frequency":
        groups.sort(key=lambda group: (-group.pixel_count, group.color))
    elif color_order == "rgb":
        groups.sort(key=lambda group: group.color)
    elif color_order != "first_seen":
        raise ValueError(f"Unknown color order: {color_order!r}")
    return tuple(groups)


def _ordered_color_index_map(
    rgb: np.ndarray,
    mask: np.ndarray,
    color_order: ColorOrder,
) -> tuple[np.ndarray, list[RGBColor], np.ndarray]:
    """Label every painted pixel with its color's paint-order index.

    Returns ``(index_map, colors, pixel_counts)`` where ``index_map`` holds the
    paint-order index for painted pixels and ``-1`` elsewhere, and ``colors``
    lists RGB tuples in the exact order they will be painted.
    """

    packed = (
        (rgb[:, :, 0].astype(np.int32) << 16)
        | (rgb[:, :, 1].astype(np.int32) << 8)
        | rgb[:, :, 2].astype(np.int32)
    )
    flat = np.where(mask, packed, -1).ravel()
    painted = flat[flat >= 0]
    if painted.size == 0:
        return np.full(mask.shape, -1, dtype=np.int32), [], np.zeros(0, dtype=np.int64)

    unique_values, first_seen, counts = np.unique(
        painted, return_index=True, return_counts=True
    )
    if color_order == "frequency":
        # Primary key: descending pixel count; tiebreak: ascending RGB, which
        # matches tuple ordering because the packing preserves channel order.
        order = np.lexsort((unique_values, -counts))
    elif color_order == "rgb":
        order = np.argsort(unique_values, kind="stable")
    elif color_order == "first_seen":
        order = np.argsort(first_seen, kind="stable")
    else:
        raise ValueError(f"Unknown color order: {color_order!r}")

    rank_of_sorted = np.empty(len(unique_values), dtype=np.int32)
    rank_of_sorted[order] = np.arange(len(unique_values), dtype=np.int32)

    index_map = np.full(flat.shape, -1, dtype=np.int32)
    painted_positions = flat >= 0
    index_map[painted_positions] = rank_of_sorted[
        np.searchsorted(unique_values, flat[painted_positions])
    ]
    colors: list[RGBColor] = [
        (int(value >> 16 & 0xFF), int(value >> 8 & 0xFF), int(value & 0xFF))
        for value in unique_values[order]
    ]
    return index_map.reshape(mask.shape), colors, counts[order]


def merge_runs_across_gaps(
    must: np.ndarray,
    barrier_cumulative: np.ndarray | None,
    max_gap: int,
) -> list[Stroke]:
    """Turn required cells into horizontal strokes, crossing harmless gaps.

    A run may extend across up to ``max_gap`` consecutive non-required cells,
    but never across a barrier.  ``barrier_cumulative`` is a per-row cumulative
    count of barrier cells (or ``None`` when ``max_gap`` is zero), so a gap is
    tested with two lookups instead of a scan.
    """

    strokes: list[Stroke] = []
    for y in np.flatnonzero(must.any(axis=1)):
        columns = np.flatnonzero(must[y])
        if columns.size == 1:
            x = int(columns[0])
            strokes.append(Stroke(x, int(y), x, int(y)))
            continue
        gap_lengths = np.diff(columns) - 1
        split = gap_lengths > max_gap
        if barrier_cumulative is not None:
            row = barrier_cumulative[y]
            split = split | (row[columns[1:] - 1] - row[columns[:-1]] > 0)
        boundaries = np.flatnonzero(split)
        starts = columns[np.concatenate(([0], boundaries + 1))]
        ends = columns[np.concatenate((boundaries, [columns.size - 1]))]
        strokes.extend(
            Stroke(int(start), int(y), int(end), int(y))
            for start, end in zip(starts, ends)
        )
    return strokes


def _runs_for_color(
    index_map: np.ndarray,
    color_index: int,
    max_gap: int,
) -> list[Stroke]:
    """Merge one color's horizontal runs, optionally crossing later colors.

    A run may extend across up to ``max_gap`` consecutive pixels belonging to
    colors painted *later* (they repaint themselves afterwards). Pixels that are
    unpainted or belong to already-painted colors always split runs.
    """

    must = index_map == color_index
    barrier_cumulative: np.ndarray | None = None
    if max_gap > 0:
        barrier_cumulative = np.cumsum(index_map < color_index, axis=1)
    return merge_runs_across_gaps(must, barrier_cumulative, max_gap)


def count_unmerged_strokes(
    source: PlanImage,
    paint_mask: np.ndarray | None = None,
) -> int:
    """The stroke count of an exact, unmerged plan - without building it.

    Equal to the number of maximal same-color horizontal runs, which is just
    the number of cells that start one.
    """

    rgb, mask = _as_rgb_and_mask(source, paint_mask)
    continues_run = np.zeros(mask.shape, dtype=np.bool_)
    continues_run[:, 1:] = (
        mask[:, 1:] & mask[:, :-1] & np.all(rgb[:, 1:] == rgb[:, :-1], axis=2)
    )
    return int((mask & ~continues_run).sum())


def generate_merged_color_groups(
    source: PlanImage,
    paint_mask: np.ndarray | None = None,
    *,
    color_order: ColorOrder = "frequency",
    overpaint_gap: int | None = 0,
) -> tuple[ColorGroup, ...]:
    """Build ordered color groups with optional gap-crossing stroke merging.

    ``overpaint_gap`` is the largest number of consecutive later-painted pixels
    a stroke may paint through; ``0`` reproduces exact maximal runs and ``None``
    removes the limit entirely.
    """

    rgb, mask = _as_rgb_and_mask(source, paint_mask)
    index_map, colors, pixel_counts = _ordered_color_index_map(rgb, mask, color_order)
    max_gap = int(mask.shape[1]) if overpaint_gap is None else max(0, int(overpaint_gap))
    groups = []
    for color_index, color in enumerate(colors):
        strokes = _runs_for_color(index_map, color_index, max_gap)
        groups.append(
            ColorGroup(
                color=color,
                strokes=tuple(strokes),
                pixel_count=int(pixel_counts[color_index]),
            )
        )
    return tuple(groups)


def plan_painted_mask(plan: PaintPlan) -> np.ndarray:
    """Cells the plan will paint over, including multi-cell band coverage.

    Uses each pass's nominal footprint - rows within its coverage radius - so
    the result never counts on accidental spill.  Anything placed under a True
    cell (a probe dab, existing paint) is repainted before the job finishes.
    """

    mask = np.zeros((plan.height, plan.width), dtype=np.bool_)
    for group in plan.color_groups:
        radius = (max(1, group.brush_diameter) - 1) // 2
        for stroke in group.strokes:
            x0 = min(stroke.start_x, stroke.end_x)
            x1 = max(stroke.start_x, stroke.end_x)
            y0 = min(stroke.start_y, stroke.end_y)
            y1 = max(stroke.start_y, stroke.end_y)
            mask[
                max(0, y0 - radius) : min(plan.height, y1 + radius + 1),
                max(0, x0) : min(plan.width, x1 + 1),
            ] = True
    return mask


def _estimate_mouse_travel(groups: tuple[ColorGroup, ...]) -> float:
    travel = 0.0
    for group in groups:
        previous_end: tuple[int, int] | None = None
        for stroke in group.strokes:
            if previous_end is not None:
                travel += hypot(
                    stroke.start_x - previous_end[0],
                    stroke.start_y - previous_end[1],
                )
            travel += stroke.logical_length
            previous_end = (stroke.end_x, stroke.end_y)
    return travel


def analyze_paint_plan(
    plan: PaintPlan,
    timing: PaintPlanTiming | None = None,
) -> PaintStatistics:
    """Calculate deterministic counts and an intentionally approximate duration."""

    timing = timing or PaintPlanTiming()
    travel = _estimate_mouse_travel(plan.color_groups)
    estimated_seconds = travel / timing.stroke_speed_pixels_per_second
    for group in plan.color_groups:
        estimated_seconds += timing.color_selection_seconds
        for stroke in group.strokes:
            if stroke.logical_length == 0:
                estimated_seconds += timing.point_duration_seconds
            estimated_seconds += timing.delay_between_strokes_seconds
        estimated_seconds += timing.delay_between_colors_seconds

    return PaintStatistics(
        logical_width=plan.width,
        logical_height=plan.height,
        unique_colors=len(plan.color_groups),
        stroke_count=plan.stroke_count,
        painted_pixels=plan.painted_pixels,
        unpainted_pixels=plan.unpainted_pixels,
        estimated_mouse_travel=travel,
        estimated_seconds=estimated_seconds,
    )


def generate_paint_plan(
    source: PlanImage,
    paint_mask: np.ndarray | None = None,
    *,
    color_order: ColorOrder = "frequency",
    timing: PaintPlanTiming | None = None,
    overpaint_gap: int | None = 0,
) -> PaintPlan:
    """Generate a complete color-grouped plan for an image.

    ``overpaint_gap`` merges same-color strokes across small islands of colors
    that are painted later anyway; the final image is unchanged, but fragmented
    regions need far fewer mouse strokes. Groups are always emitted in painting
    order, and the painter must execute them in that order for merging to be
    correct.
    """

    rgb, final_mask = _as_rgb_and_mask(source, paint_mask)
    groups = generate_merged_color_groups(
        rgb,
        final_mask,
        color_order=color_order,
        overpaint_gap=overpaint_gap,
    )
    height, width = final_mask.shape
    plan = PaintPlan(
        width=width,
        height=height,
        color_groups=groups,
        unpainted_pixels=int(final_mask.size - np.count_nonzero(final_mask)),
    )
    return replace(plan, _statistics=analyze_paint_plan(plan, timing))


# Names that read naturally in worker/controller code.
build_paint_plan = generate_paint_plan
create_paint_plan = generate_paint_plan


__all__ = [
    "ColorOrder",
    "PaintPlanTiming",
    "analyze_paint_plan",
    "build_paint_plan",
    "count_unmerged_strokes",
    "create_paint_plan",
    "generate_horizontal_runs",
    "generate_merged_color_groups",
    "generate_paint_plan",
    "group_horizontal_runs",
    "horizontal_runs_for_color",
    "merge_runs_across_gaps",
    "plan_painted_mask",
]
