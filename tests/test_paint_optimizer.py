from __future__ import annotations

import numpy as np
import pytest

from app.models import PaintMode, PaintPlan
from app.paint_plan import count_unmerged_strokes, generate_merged_color_groups
from app.paint_optimizer import (
    BrushCapabilities,
    absorb_insignificant_regions,
    merge_similar_colors,
    mode_options,
    optimize_paint_plan,
    simplify_colors,
)


def _rgb(width: int, height: int, color: tuple[int, int, int]) -> np.ndarray:
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[:] = color
    return image


def _full_mask(image: np.ndarray) -> np.ndarray:
    return np.ones(image.shape[:2], dtype=np.bool_)


def _simulate(plan: PaintPlan, footprint: str) -> np.ndarray:
    """Rasterize a plan in order; ``-1`` marks never-painted cells.

    ``cover`` paints the minimal band every solid brush guarantees; ``spill``
    paints the worst-case reach the planner's safety margin allows, so it
    checks that overshoot can only ever land where a later pass repaints.
    """

    canvas = np.full((plan.height, plan.width, 3), -1, dtype=np.int32)
    for group in plan.color_groups:
        radius = (group.brush_diameter - 1) // 2
        color = np.array(group.color, dtype=np.int32)
        for stroke in group.strokes:
            y = stroke.start_y
            x0, x1 = sorted((stroke.start_x, stroke.end_x))
            # A single-cell brush is sized to underfill its cell, so only the
            # multi-cell brushes have a worst-case spill beyond their band.
            if footprint == "cover" or group.brush_diameter == 1:
                top = max(0, y - radius)
                bottom = min(plan.height, y + radius + 1)
                canvas[top:bottom, x0 : x1 + 1] = color
                continue
            reach = radius + 1
            if group.brush_shape == "circle":
                limit = (radius + 1.25) ** 2
                for cy in range(max(0, y - reach), min(plan.height, y + reach + 1)):
                    for cx in range(
                        max(0, x0 - reach), min(plan.width, x1 + reach + 1)
                    ):
                        gap_x = max(0, x0 - cx, cx - x1)
                        if gap_x * gap_x + (cy - y) * (cy - y) <= limit:
                            canvas[cy, cx] = color
            else:
                top = max(0, y - reach)
                bottom = min(plan.height, y + reach + 1)
                left = max(0, x0 - reach)
                right = min(plan.width, x1 + reach + 1)
                canvas[top:bottom, left:right] = color
    return canvas


def _assert_plan_paints_target(
    plan: PaintPlan, target: np.ndarray, mask: np.ndarray
) -> None:
    covered = _simulate(plan, "cover")
    assert np.array_equal(covered[mask], target[mask].astype(np.int32)), (
        "minimal brush coverage must reproduce the optimized target exactly"
    )
    spilled = _simulate(plan, "spill")
    assert np.array_equal(spilled[mask], target[mask].astype(np.int32)), (
        "worst-case brush spill must always be repainted by a later pass"
    )
    outside = ~mask
    if outside.any():
        assert (spilled[outside] == -1).all(), (
            "no brush footprint may reach an unpainted cell"
        )


def test_merge_collapses_invisible_shades_and_keeps_contrast() -> None:
    image = _rgb(8, 4, (255, 255, 255))
    image[0, 0] = (254, 255, 255)
    image[0, 1] = (255, 254, 253)
    image[2, 2] = (40, 40, 40)
    merged = merge_similar_colors(image, _full_mask(image), tolerance=2.5)
    assert (merged[0, 0] == (255, 255, 255)).all()
    assert (merged[0, 1] == (255, 255, 255)).all()
    assert (merged[2, 2] == (40, 40, 40)).all()


def test_absorb_speck_but_keep_high_contrast_dot() -> None:
    image = _rgb(20, 12, (255, 255, 255))
    image[3, 4] = (250, 250, 250)  # invisible speck
    image[8, 14] = (0, 0, 0)  # deliberate dark dot
    cleaned = absorb_insignificant_regions(
        image, _full_mask(image), min_area=4, contrast_limit=8.0
    )
    assert (cleaned[3, 4] == (255, 255, 255)).all()
    assert (cleaned[8, 14] == (0, 0, 0)).all()


def test_dither_preset_disables_region_cleanup() -> None:
    options = mode_options(PaintMode.FAST, preserve_dither=True)
    assert options.min_region_area == 0
    assert options.merge_tolerance <= 2.5


def test_exact_mode_is_rejected() -> None:
    image = _rgb(8, 8, (10, 20, 30))
    with pytest.raises(ValueError):
        optimize_paint_plan(image, PaintMode.EXACT)


def test_without_sizing_every_group_stays_single_cell() -> None:
    image = _rgb(64, 32, (200, 60, 40))
    image[8:24, 10:40] = (30, 70, 200)
    result = optimize_paint_plan(image, PaintMode.BALANCED)
    assert result.plan.color_groups
    assert all(group.brush_diameter == 1 for group in result.plan.color_groups)
    assert all(group.brush_shape is None for group in result.plan.color_groups)
    _assert_plan_paints_target(
        result.plan, np.asarray(result.image.convert("RGB")), result.paint_mask
    )


def test_large_region_earns_a_larger_brush() -> None:
    image = _rgb(96, 64, (240, 240, 240))
    image[20:44, 30:70] = (200, 30, 30)
    result = optimize_paint_plan(
        image,
        PaintMode.BALANCED,
        capabilities=BrushCapabilities(sizing=True, square=True),
    )
    diameters = {group.brush_diameter for group in result.plan.color_groups}
    assert max(diameters) > 1
    assert result.plan.stroke_count < 96 * 64 // 4
    _assert_plan_paints_target(
        result.plan, np.asarray(result.image.convert("RGB")), result.paint_mask
    )


def test_layered_plan_reproduces_target_with_both_shapes() -> None:
    rng = np.random.default_rng(7)
    image = _rgb(96, 64, (60, 120, 220))  # sky
    image[40:64, :] = (60, 160, 60)  # ground
    image[10:30, 12:34] = (200, 40, 40)  # building
    image[16:24, 60:84] = (250, 240, 100)  # sun block
    for _ in range(30):  # scattered detail
        y = int(rng.integers(0, 64))
        x = int(rng.integers(0, 96))
        image[y, x] = (20, 20, 20)
    result = optimize_paint_plan(
        image,
        PaintMode.FAST,
        capabilities=BrushCapabilities(sizing=True, square=True, circle=True),
    )
    target = np.asarray(result.image.convert("RGB"))
    _assert_plan_paints_target(result.plan, target, result.paint_mask)
    statistics = result.statistics
    assert statistics.output_colors <= statistics.input_colors
    assert 0.0 <= statistics.similarity_percent <= 100.0
    used_shapes = {
        group.brush_shape
        for group in result.plan.color_groups
        if group.brush_diameter > 1
    }
    assert used_shapes <= {"square", "circle"}


def test_unknown_shape_plans_conservatively() -> None:
    image = _rgb(80, 48, (255, 255, 255))
    image[8:40, 8:72] = (10, 10, 10)
    result = optimize_paint_plan(
        image,
        PaintMode.BALANCED,
        capabilities=BrushCapabilities(sizing=True),
    )
    assert all(group.brush_shape is None for group in result.plan.color_groups)
    _assert_plan_paints_target(
        result.plan, np.asarray(result.image.convert("RGB")), result.paint_mask
    )


def test_partial_mask_is_never_touched() -> None:
    image = _rgb(64, 48, (90, 90, 200))
    image[10:30, 10:50] = (240, 220, 60)
    mask = _full_mask(image)
    mask[:, 52:] = False  # letterboxed edge stays unpainted
    mask[36:44, 4:20] = False  # transparent hole
    result = optimize_paint_plan(
        image,
        PaintMode.BALANCED,
        capabilities=BrushCapabilities(sizing=True, square=True),
        paint_mask=mask,
    )
    assert np.array_equal(result.paint_mask, mask)
    _assert_plan_paints_target(
        result.plan, np.asarray(result.image.convert("RGB")), result.paint_mask
    )


def test_unmerged_stroke_count_matches_the_built_plan() -> None:
    rng = np.random.default_rng(5)
    image = rng.integers(0, 4, (24, 40))[:, :, None].repeat(3, axis=2) * 60
    image = image.astype(np.uint8)
    mask = _full_mask(image)
    mask[:, 30:] = False
    built = sum(
        len(group.strokes)
        for group in generate_merged_color_groups(image, mask, overpaint_gap=0)
    )
    assert count_unmerged_strokes(image, mask) == built


def test_simplify_colors_matches_the_full_optimizer_target() -> None:
    image = _rgb(48, 24, (250, 250, 250))
    image[4, 4] = (252, 250, 250)
    image[10:20, 10:30] = (30, 30, 30)
    simplified, mask = simplify_colors(image, PaintMode.BALANCED)
    full = optimize_paint_plan(image, PaintMode.BALANCED)
    assert np.array_equal(
        np.asarray(simplified.convert("RGB")), np.asarray(full.image.convert("RGB"))
    )
    assert np.array_equal(mask, full.paint_mask)


def test_oversized_cells_cap_the_brush() -> None:
    image = _rgb(64, 32, (20, 20, 20))
    result = optimize_paint_plan(
        image,
        PaintMode.FAST,
        # 40px cells: even a 3-cell brush would exceed the slider's safe range.
        capabilities=BrushCapabilities(sizing=True, square=True, cell_pixels=40.0),
    )
    assert all(group.brush_diameter == 1 for group in result.plan.color_groups)


def test_a_measured_top_end_overrides_the_guessed_slider_range() -> None:
    image = _rgb(64, 32, (20, 20, 20))
    result = optimize_paint_plan(
        image,
        PaintMode.FAST,
        # The guess would forbid 40px cells, but this Size track was measured
        # reaching 130px, which comfortably paints a 3-cell (120px) band.
        capabilities=BrushCapabilities(
            sizing=True, square=True, cell_pixels=40.0, max_brush_pixels=130.0
        ),
    )
    assert max(group.brush_diameter for group in result.plan.color_groups) == 3


def test_a_measured_minimum_rules_out_overshooting_passes() -> None:
    image = _rgb(64, 32, (20, 20, 20))
    result = optimize_paint_plan(
        image,
        PaintMode.FAST,
        # The smallest dab this track paints is 30px, so on 4px cells even a
        # 5-cell band (20px nominal, one guard cell each side) would spill
        # beyond the erosion's safety margin; only 7-cell passes may stay.
        capabilities=BrushCapabilities(
            sizing=True,
            square=True,
            cell_pixels=4.0,
            min_brush_pixels=30.0,
            max_brush_pixels=64.0,
        ),
    )
    diameters = {group.brush_diameter for group in result.plan.color_groups}
    assert 3 not in diameters and 5 not in diameters
