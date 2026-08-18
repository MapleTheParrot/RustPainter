"""Compare the painted canvas against the plan and build touch-up strokes.

Open-loop painting misses for reasons no calibration can fully remove: a
dropped click, a frame hitch while a stroke was moving, a picker click landing
one hue off.  Reading the sign back and repainting only the cells that came
out wrong is what actually closes the gap between the preview and the sign.

The sign is a lit, textured 3D surface, so captured colors never equal the
palette values exactly.  Every comparison here is therefore *relative*: a cell
counts as wrong only when its captured color sits decisively closer to a
different plan color than to its own - a global lighting shift moves every
color together and changes nothing.
"""

from __future__ import annotations

import numpy as np

from .models import ColorGroup, PaintPlan, RGBColor
from .paint_optimizer import _srgb_to_lab
from .paint_plan import merge_runs_across_gaps


# A cell is repainted only when its captured Lab color is this much closer to
# another plan color than to its own.  Below the margin the reading is
# ambiguous - sign texture, a cell boundary, compression - and repainting on
# ambiguity would oscillate instead of converge.
CLASSIFICATION_MARGIN_DELTA_E = 5.0

# Mismatches beyond this fraction of covered cells mean the capture itself is
# untrustworthy (the sign is occluded, the camera moved, a menu is open).
UNRELIABLE_CAPTURE_FRACTION = 0.4


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
    capture_rgb: np.ndarray, logical_width: int, logical_height: int
) -> np.ndarray:
    """One robust color per logical cell, read from a canvas capture.

    Samples a 3x3 median around each cell center, which shrugs off the odd
    noisy pixel and the sign's grain without blurring across cell borders the
    way an area resize would.
    """

    pixels = np.asarray(capture_rgb, dtype=np.float32)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("Canvas capture must be an RGB array")
    height, width = pixels.shape[:2]
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
    neighborhood = np.stack(
        [
            pixels[centers_y + dy][:, centers_x + dx]
            for dy in (-1, 0, 1)
            for dx in (-1, 0, 1)
        ]
    )
    return np.median(neighborhood, axis=0)


def normalize_capture_lighting(
    sampled: np.ndarray, indices: np.ndarray, palette: np.ndarray
) -> np.ndarray:
    """Undo the sign's global material response before classifying cells.

    Live testing showed a painting that matched its plan perfectly still
    classified 76% wrong: the lit sign compresses dark colors, and with a
    tightly clustered palette that compression pushes a correct cell closer to
    a neighbouring palette entry than to its own.  One global affine transform
    fitted from the capture itself absorbs lighting and material - it cannot
    absorb per-cell painting mistakes, because twelve parameters cannot bend
    thousands of cells individually.

    The fit runs twice, the second time without the worst quartile of
    residuals, so a minority of genuinely wrong cells does not drag the
    transform toward hiding themselves.
    """

    covered = indices >= 0
    count = int(covered.sum())
    if count < 24 or len(palette) < 2:
        return sampled
    captured = sampled[covered].reshape(-1, 3).astype(np.float64)
    wanted = palette[indices[covered]].astype(np.float64)
    design = np.hstack([captured, np.ones((len(captured), 1))])

    def fit(rows: np.ndarray) -> np.ndarray:
        coefficients, *_ = np.linalg.lstsq(design[rows], wanted[rows], rcond=None)
        return coefficients

    everything = np.ones(len(captured), dtype=np.bool_)
    coefficients = fit(everything)
    residuals = np.linalg.norm(design @ coefficients - wanted, axis=1)
    keep = residuals <= np.percentile(residuals, 75)
    if keep.sum() >= 24:
        coefficients = fit(keep)

    corrected = sampled.astype(np.float64).copy()
    corrected[covered] = np.clip(design @ coefficients, 0.0, 255.0)
    return corrected.astype(np.float32)


def mismatched_cells(
    sampled: np.ndarray,
    indices: np.ndarray,
    palette: np.ndarray,
    *,
    margin: float = CLASSIFICATION_MARGIN_DELTA_E,
) -> np.ndarray:
    """Cells whose captured color decisively belongs to a different plan color."""

    if len(palette) == 0:
        return np.zeros(indices.shape, dtype=np.bool_)
    palette_lab = _srgb_to_lab(palette.astype(np.float32))
    sampled_lab = _srgb_to_lab(sampled.reshape(-1, 3)).reshape(-1, 1, 3)
    distances = np.sqrt(
        ((sampled_lab - palette_lab.reshape(1, -1, 3)) ** 2).sum(axis=2)
    ).reshape(*indices.shape, len(palette))
    covered = indices >= 0
    own = np.take_along_axis(
        distances, np.where(covered, indices, 0)[..., None], axis=2
    )[..., 0]
    nearest = distances.min(axis=2)
    return covered & (own - nearest > margin)


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
    "CLASSIFICATION_MARGIN_DELTA_E",
    "UNRELIABLE_CAPTURE_FRACTION",
    "mismatched_cells",
    "normalize_capture_lighting",
    "plan_expectations",
    "sample_cell_colors",
    "touch_up_plan",
]
