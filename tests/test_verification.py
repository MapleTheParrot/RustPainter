from __future__ import annotations

import numpy as np

from app.models import ColorGroup, PaintPlan, Stroke
from app.verification import (
    mismatched_cells,
    plan_expectations,
    sample_cell_colors,
    touch_up_plan,
)


RED = (200, 30, 30)
BLUE = (30, 60, 200)


def _two_band_plan() -> PaintPlan:
    return PaintPlan(
        8,
        6,
        (
            ColorGroup(RED, tuple(Stroke(0, y, 7, y) for y in range(3)), 24),
            ColorGroup(BLUE, tuple(Stroke(0, y, 7, y) for y in range(3, 6)), 24),
        ),
    )


def _capture_for(indices: np.ndarray, palette: np.ndarray, cell: int = 20) -> np.ndarray:
    height, width = indices.shape
    capture = np.zeros((height * cell, width * cell, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            index = indices[y, x]
            color = palette[index] if index >= 0 else (0, 0, 0)
            capture[y * cell : (y + 1) * cell, x * cell : (x + 1) * cell] = color
    return capture


def test_expectations_replay_painting_order_and_band_coverage() -> None:
    plan = PaintPlan(
        9,
        9,
        (
            # A 3-cell band centered on row 4 covers rows 3-5.
            ColorGroup(RED, (Stroke(0, 4, 8, 4),), 27, brush_diameter=3),
            # A later single-cell stroke wins its cells back.
            ColorGroup(BLUE, (Stroke(2, 4, 5, 4),), 4),
        ),
    )
    indices, palette = plan_expectations(plan)
    assert [tuple(color) for color in palette] == [RED, BLUE]
    assert (indices[3, :] == 0).all() and (indices[5, :] == 0).all()
    assert (indices[4, 2:6] == 1).all()
    assert indices[4, 0] == 0 and indices[4, 8] == 0
    assert (indices[0, :] == -1).all()


def test_sampling_reads_cell_centers_not_borders() -> None:
    plan = _two_band_plan()
    indices, palette = plan_expectations(plan)
    capture = _capture_for(indices, palette)
    sampled = sample_cell_colors(capture, plan.width, plan.height)
    assert np.allclose(sampled[0, 0], RED)
    assert np.allclose(sampled[5, 7], BLUE)


def test_a_global_lighting_shift_is_never_a_mismatch() -> None:
    plan = _two_band_plan()
    indices, palette = plan_expectations(plan)
    capture = _capture_for(indices, palette) * 0.7 + 20.0  # darker, lifted sign
    sampled = sample_cell_colors(capture, plan.width, plan.height)
    assert not mismatched_cells(sampled, indices, palette).any()


def test_cells_painted_the_wrong_plan_color_are_flagged_and_repainted() -> None:
    plan = _two_band_plan()
    indices, palette = plan_expectations(plan)
    wrong = indices.copy()
    wrong[1, 2] = 1  # a red cell that came out blue
    wrong[1, 3] = 1
    wrong[4, 6] = 0  # a blue cell that came out red
    capture = _capture_for(wrong, palette)
    sampled = sample_cell_colors(capture, plan.width, plan.height)
    mismatch = mismatched_cells(sampled, indices, palette)
    assert mismatch.sum() == 3
    assert mismatch[1, 2] and mismatch[1, 3] and mismatch[4, 6]

    touch = touch_up_plan(mismatch, indices, palette)
    painted = {
        (group.color, stroke.start_x, stroke.start_y, stroke.end_x, stroke.end_y)
        for group in touch.color_groups
        for stroke in group.strokes
    }
    # Adjacent wrong cells merge into one stroke; strokes never leave the
    # mismatched cells.
    assert painted == {
        (RED, 2, 1, 3, 1),
        (BLUE, 6, 4, 6, 4),
    }
    assert sum(group.pixel_count for group in touch.color_groups) == 3


def test_uncovered_cells_are_ignored() -> None:
    plan = PaintPlan(8, 6, (ColorGroup(RED, (Stroke(0, 0, 7, 0),), 8),))
    indices, palette = plan_expectations(plan)
    capture = np.zeros((120, 160, 3), dtype=np.float32)  # everything black
    capture[0:20] = RED
    sampled = sample_cell_colors(capture, plan.width, plan.height)
    assert not mismatched_cells(sampled, indices, palette).any()
