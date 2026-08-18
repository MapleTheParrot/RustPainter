from __future__ import annotations

import numpy as np
import pytest

from app.brush_calibration import BrushResponseSet, build_brush_response
from app.models import ColorGroup, PaintPlan, Stroke
from app.paint_simulation import painted_diameter, simulate_painted_plan


def _plan(groups: tuple[ColorGroup, ...], width: int = 10, height: int = 10) -> PaintPlan:
    return PaintPlan(width, height, groups)


def test_painted_diameter_reproduces_the_curve_clamp() -> None:
    response = build_brush_response([(0.0, 30.0), (1.0, 64.0)])
    # A 20px cell asks for 18px, but the track never paints under 30px: the
    # simulation must predict the same oversized dab the painter will get.
    assert painted_diameter(response, 20.0, 1) == pytest.approx(30.0)
    # Without a curve the target itself is the best available prediction.
    assert painted_diameter(None, 20.0, 1) == pytest.approx(18.0)
    assert painted_diameter(None, 20.0, 3) == pytest.approx(60.0)


def test_an_oversized_brush_visibly_bleeds_past_its_cell() -> None:
    responses = BrushResponseSet(
        (build_brush_response([(0.0, 40.0), (1.0, 64.0)]),)
    )
    plan = _plan(
        (ColorGroup((255, 0, 0), (Stroke(5, 5, 5, 5),), 1),)
    )
    rgb, painted = simulate_painted_plan(plan, 200, 200, responses)
    ys, xs = np.nonzero(painted)
    # One 20px cell, but the smallest dab is 40px: the footprint must span
    # roughly twice the cell, not politely stop at its edge.
    assert xs.max() - xs.min() + 1 >= 36
    assert ys.max() - ys.min() + 1 >= 36
    assert (rgb[painted] == (255, 0, 0)).all()


def test_painting_order_decides_the_final_color() -> None:
    responses = BrushResponseSet(
        (build_brush_response([(0.0, 10.0), (1.0, 64.0)]),)
    )
    plan = _plan(
        (
            ColorGroup((0, 0, 255), (Stroke(2, 5, 7, 5),), 6),
            ColorGroup((0, 255, 0), (Stroke(5, 5, 5, 5),), 1),
        )
    )
    rgb, painted = simulate_painted_plan(plan, 200, 200, responses)
    center = rgb[110, 110]
    assert (center == (0, 255, 0)).all(), "the later group must overpaint"
    assert (rgb[110, 50] == (0, 0, 255)).all()


def test_square_passes_fill_their_corners_and_circles_do_not() -> None:
    square_curve = build_brush_response([(0.0, 10.0), (1.0, 64.0)], shape="square")
    circle_curve = build_brush_response([(0.0, 10.0), (1.0, 64.0)], shape="circle")
    responses = BrushResponseSet((square_curve, circle_curve))
    for shape, corner_painted in (("square", True), ("circle", False)):
        plan = _plan(
            (
                ColorGroup(
                    (200, 200, 0),
                    (Stroke(5, 5, 5, 5),),
                    9,
                    brush_diameter=3,
                    brush_shape=shape,
                ),
            )
        )
        _rgb, painted = simulate_painted_plan(plan, 200, 200, responses)
        ys, xs = np.nonzero(painted)
        center_y = (ys.min() + ys.max()) / 2
        radius = (ys.max() - ys.min()) / 2
        offset = int(radius * 0.85)
        corner = painted[int(center_y - offset), int(np.median(xs)) - offset]
        assert bool(corner) is corner_painted, shape


def test_a_well_fitted_brush_renders_solid_coverage_without_seams() -> None:
    # Detail strokes are deliberately sized a shade under their cell; Rust's
    # soft brush edge closes that seam on the sign, so the preview must not
    # show a grid of gapped dots where the sign will look solid.
    responses = BrushResponseSet(
        # The curve can hit any target, so every row paints at 0.9 cells.
        (build_brush_response([(0.0, 4.0), (1.0, 64.0)]),)
    )
    plan = _plan(
        (
            ColorGroup(
                (120, 40, 40),
                tuple(Stroke(0, y, 9, y) for y in range(10)),
                100,
            ),
        )
    )
    _rgb, painted = simulate_painted_plan(plan, 200, 200, responses)
    assert painted.all(), "full-coverage plans must preview with no checker gaps"


def test_only_overshoot_extends_past_the_nominal_cells() -> None:
    responses = BrushResponseSet(
        (build_brush_response([(0.0, 4.0), (1.0, 64.0)]),)
    )
    plan = _plan((ColorGroup((10, 200, 10), (Stroke(3, 5, 6, 5),), 4),))
    _rgb, painted = simulate_painted_plan(plan, 200, 200, responses)
    ys, xs = np.nonzero(painted)
    # A reachable 18px target on a 20px cell: coverage is exactly the four
    # nominal cells, with no bleed into the neighbouring rows or columns.
    assert ys.min() >= 100 and ys.max() <= 120
    assert xs.min() >= 60 and xs.max() <= 140


def test_the_simulation_downscales_a_huge_canvas() -> None:
    responses = BrushResponseSet(
        (build_brush_response([(0.0, 10.0), (1.0, 64.0)]),)
    )
    plan = _plan((ColorGroup((1, 2, 3), (Stroke(0, 0, 9, 0),), 10),))
    rgb, painted = simulate_painted_plan(
        plan, 4000, 2000, responses, max_pixels=200_000
    )
    assert rgb.shape[0] * rgb.shape[1] <= 220_000
    assert painted.any()
