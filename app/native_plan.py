"""Lay any plan out on the sign's own texels, one row of texels per stroke.

The one stamp the game was measured to make exactly is the smallest brush:
a press paints the single texel under the cursor, a drag paints the texel
centres its path crosses, whatever the screen resolution.  Every wider
brush has a soft rim whose partial coverage depends on where inside a
texel the cursor sits - a sub-pixel the cursor cannot be placed to at
under two screen pixels per texel - and that rim is exactly the one-texel
strip of half-transparent canvas a finished sign showed along every band.

So a plan is executed at texel resolution with that one brush.  A plan
drawn at some other resolution - a coarse "very high" preset, a plan with
a three-cell fill brush - is converted here: every cell maps onto the
block of texels it covers, every fill band onto its texel rows, and each
texel row becomes one stroke.  Colour order is kept, which is what the
planners' overpainting rests on.  Nothing is estimated about coverage: the
strokes ARE the texels.
"""

from __future__ import annotations

from dataclasses import replace

from .models import ColorGroup, PaintPlan, Stroke


def cell_span(index: int, cells: int, texels: int) -> tuple[int, int]:
    """The texels a cell covers along one axis, inclusive, tiling exactly.

    ``cells`` logical cells share ``texels`` texels: cell ``i`` starts at
    ``floor(i * texels / cells)`` and ends before the next cell starts, so
    no texel belongs to two cells and none to none.
    """

    start = (index * texels) // cells
    end = ((index + 1) * texels) // cells - 1
    return start, max(start, end)


def band_rows(stroke: Stroke, diameter: int, height: int) -> tuple[int, int]:
    """The rows (cells) a stroke of a ``diameter``-cell brush covers, inclusive."""

    radius = (max(1, diameter) - 1) // 2
    top = min(stroke.start_y, stroke.end_y) - radius
    bottom = max(stroke.start_y, stroke.end_y) + radius
    return max(0, top), min(height - 1, bottom)


def is_native(plan: PaintPlan, columns: int, rows: int) -> bool:
    """Whether a plan already is one texel per cell with the smallest brush."""

    return (plan.width, plan.height) == (columns, rows) and all(
        group.brush_diameter <= 1 for group in plan.color_groups
    )


def nativize_plan(plan: PaintPlan, columns: int, rows: int) -> PaintPlan:
    """The same painting as texel-row strokes on a ``columns`` x ``rows`` sign.

    Horizontal strokes (and dabs) become one stroke per texel row of the
    block they cover; a stroke's sweep direction is kept so a serpentine
    order stays serpentine.  A diagonal stroke - which no planner in this
    application emits - is rasterised cell by cell.
    """

    if is_native(plan, columns, rows):
        return plan
    if plan.width <= 0 or plan.height <= 0 or columns <= 0 or rows <= 0:
        raise ValueError("Plan and sign dimensions must be positive")
    groups: list[ColorGroup] = []
    for group in plan.color_groups:
        strokes: list[Stroke] = []
        covered = 0
        for stroke in group.strokes:
            top_cell, bottom_cell = band_rows(stroke, group.brush_diameter, plan.height)
            first_row = cell_span(top_cell, plan.height, rows)[0]
            last_row = cell_span(bottom_cell, plan.height, rows)[1]
            if stroke.start_y == stroke.end_y or stroke.start_x == stroke.end_x:
                left_cell = min(stroke.start_x, stroke.end_x)
                right_cell = max(stroke.start_x, stroke.end_x)
                if stroke.start_y != stroke.end_y:
                    # A vertical run: its own cells' rows, one stroke each.
                    left_cell = right_cell = stroke.start_x
                first_col = cell_span(left_cell, plan.width, columns)[0]
                last_col = cell_span(right_cell, plan.width, columns)[1]
                forward = stroke.end_x >= stroke.start_x
                for row in range(first_row, last_row + 1):
                    if forward:
                        strokes.append(Stroke(first_col, row, last_col, row))
                    else:
                        strokes.append(Stroke(last_col, row, first_col, row))
                    covered += last_col - first_col + 1
                continue
            # Diagonal: every cell along it, as its own block of rows.
            steps = max(abs(stroke.end_x - stroke.start_x), abs(stroke.end_y - stroke.start_y))
            for step in range(steps + 1):
                cx = round(stroke.start_x + (stroke.end_x - stroke.start_x) * step / steps)
                cy = round(stroke.start_y + (stroke.end_y - stroke.start_y) * step / steps)
                c0, c1 = cell_span(cx, plan.width, columns)
                r0, r1 = cell_span(cy, plan.height, rows)
                for row in range(r0, r1 + 1):
                    strokes.append(Stroke(c0, row, c1, row))
                    covered += c1 - c0 + 1
        groups.append(
            ColorGroup(
                color=group.color,
                strokes=tuple(strokes),
                pixel_count=covered,
                brush_diameter=1,
            )
        )
    unpainted = columns * rows - _painted_texels(groups, columns, rows)
    native = PaintPlan(
        width=columns,
        height=rows,
        color_groups=tuple(groups),
        unpainted_pixels=int(unpainted),
    )
    return native


def _painted_texels(groups: list[ColorGroup], columns: int, rows: int) -> int:
    import numpy as np

    covered = np.zeros((rows, columns), dtype=np.bool_)
    for group in groups:
        for stroke in group.strokes:
            x0, x1 = sorted((stroke.start_x, stroke.end_x))
            covered[stroke.start_y, x0 : x1 + 1] = True
    return int(covered.sum())


def stroke_index_map(plan: PaintPlan, native: PaintPlan) -> list[int]:
    """For each native stroke, the index of the original stroke it came from."""

    # Both plans keep group order and, within a group, stroke order; a
    # native group's strokes are the original's expanded in sequence.
    mapping: list[int] = []
    original_index = 0
    for group, native_group in zip(plan.color_groups, native.color_groups):
        rows_per = _rows_per_stroke(plan, native, group)
        for stroke, count in zip(group.strokes, rows_per):
            mapping.extend([original_index] * count)
            original_index += 1
        assert len(native_group.strokes) == sum(rows_per)
    return mapping


def _rows_per_stroke(plan: PaintPlan, native: PaintPlan, group: ColorGroup) -> list[int]:
    counts = []
    for stroke in group.strokes:
        top_cell, bottom_cell = band_rows(stroke, group.brush_diameter, plan.height)
        first_row = cell_span(top_cell, plan.height, native.height)[0]
        last_row = cell_span(bottom_cell, plan.height, native.height)[1]
        if stroke.start_y == stroke.end_y or stroke.start_x == stroke.end_x:
            counts.append(last_row - first_row + 1)
        else:
            steps = max(abs(stroke.end_x - stroke.start_x), abs(stroke.end_y - stroke.start_y))
            total = 0
            for step in range(steps + 1):
                cy = round(stroke.start_y + (stroke.end_y - stroke.start_y) * step / steps)
                r0, r1 = cell_span(cy, plan.height, native.height)
                total += r1 - r0 + 1
            counts.append(total)
    return counts


__all__ = [
    "band_rows",
    "cell_span",
    "is_native",
    "nativize_plan",
    "stroke_index_map",
]
