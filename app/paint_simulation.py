"""Render what the physical brush will actually paint, not the logical grid.

The processed image shows the plan's *target*: every cell filled exactly.  The
painter, however, executes strokes with a physical brush whose diameter comes
from the measured response curve - including its clamping, so a Size track
whose smallest dab is wider than a cell paints that dab anyway.  Stamping each
stroke with that footprint, in painting order, turns the preview into a
prediction of the sign instead of a wish.
"""

from __future__ import annotations

from math import ceil, floor, sqrt

import numpy as np

from .brush_calibration import BrushResponse, BrushResponseSet
from .models import BrushShape, PaintPlan


def painted_diameter(
    response: BrushResponse | None,
    cell: float,
    diameter_cells: int,
    spacing: float = 1.0,
) -> float:
    """The diameter the painter will actually get for one pass.

    Mirrors the sizing logic in ``Painter._apply_brush_size``: the same target
    for the same pass, pushed through the same measured curve - whose
    interpolation clamps to the measured range exactly like the slider does.
    """

    spacing = min(spacing, 1.0)
    if diameter_cells <= 1:
        target = cell * spacing * 0.90
    else:
        target = cell * spacing * diameter_cells
    if response is None:
        return target
    return response.diameter_for(response.fraction_for(target))


def _stamp_horizontal(
    rgb: np.ndarray,
    painted: np.ndarray,
    color: tuple[int, int, int],
    x0: float,
    x1: float,
    y: float,
    radius: float,
    square: bool,
) -> None:
    """Stamp one horizontal drag (or dab) of the brush onto the buffers."""

    height, width = painted.shape
    left = max(0, floor(min(x0, x1) - radius))
    right = min(width, ceil(max(x0, x1) + radius) + 1)
    top = max(0, floor(y - radius))
    bottom = min(height, ceil(y + radius) + 1)
    if left >= right or top >= bottom:
        return
    ys = np.arange(top, bottom, dtype=np.float32)[:, None] - y
    xs = np.arange(left, right, dtype=np.float32)[None, :]
    beyond = np.maximum(np.maximum(min(x0, x1) - xs, xs - max(x0, x1)), 0.0)
    if square:
        mask = (np.abs(ys) <= radius) & (beyond <= radius)
    else:
        mask = beyond**2 + ys**2 <= radius**2
    rgb[top:bottom, left:right][mask] = color
    painted[top:bottom, left:right][mask] = True


def simulate_painted_plan(
    plan: PaintPlan,
    canvas_width: int,
    canvas_height: int,
    responses: BrushResponseSet | None,
    *,
    spacing: float = 1.0,
    max_pixels: int = 1_600_000,
) -> tuple[np.ndarray, np.ndarray]:
    """Stamp every stroke with its physical footprint, in painting order.

    Returns ``(rgb, painted)`` at canvas scale (downscaled uniformly when the
    canvas exceeds ``max_pixels``, so previews stay cheap).  Cells no stroke
    reaches stay unpainted; cells a too-wide brush spills onto are painted -
    both exactly as they will be on the sign.
    """

    if canvas_width <= 0 or canvas_height <= 0:
        raise ValueError("Canvas dimensions must be positive")
    scale = min(1.0, sqrt(max_pixels / float(canvas_width * canvas_height)))
    width = max(1, round(canvas_width * scale))
    height = max(1, round(canvas_height * scale))
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    painted = np.zeros((height, width), dtype=np.bool_)
    cell = min(canvas_width / plan.width, canvas_height / plan.height)
    center_x = width / 2.0
    center_y = height / 2.0

    for group in plan.color_groups:
        response = responses.for_shape(group.brush_shape) if responses else None
        diameter = painted_diameter(
            response, cell, max(1, group.brush_diameter), spacing
        )
        radius = max(0.5, diameter * scale / 2.0)
        square = group.brush_shape == BrushShape.SQUARE.value
        for stroke in group.strokes:
            x0 = (stroke.start_x + 0.5) / plan.width * width
            x1 = (stroke.end_x + 0.5) / plan.width * width
            y0 = (stroke.start_y + 0.5) / plan.height * height
            if spacing != 1.0:
                # Mirrors the painter's _space_and_clamp: stroke geometry
                # contracts or spreads about the canvas center.
                x0 = center_x + (x0 - center_x) * spacing
                x1 = center_x + (x1 - center_x) * spacing
                y0 = center_y + (y0 - center_y) * spacing
            _stamp_horizontal(
                rgb, painted, group.color, x0, x1, y0, radius, square
            )
    return rgb, painted


__all__ = ["painted_diameter", "simulate_painted_plan"]
