"""Any plan becomes one Size-1 stroke per texel row, covering the same texels."""

from __future__ import annotations

import numpy as np

from app.models import ColorGroup, PaintPlan, Stroke
from app.native_plan import cell_span, is_native, nativize_plan, stroke_index_map
from app.verification import plan_layers


def _coverage(plan: PaintPlan) -> np.ndarray:
    indices, _under, _palette = plan_layers(plan)
    return indices


def test_cells_tile_the_texels_exactly() -> None:
    spans = [cell_span(i, 257, 512) for i in range(257)]
    assert spans[0][0] == 0 and spans[-1][1] == 511
    assert all(b[0] == a[1] + 1 for a, b in zip(spans, spans[1:]))
    assert cell_span(3, 512, 1024) == (6, 7)


def test_a_native_single_brush_plan_is_left_alone() -> None:
    plan = PaintPlan(4, 2, (ColorGroup((1, 2, 3), (Stroke(0, 0, 3, 0),), 4),))
    assert is_native(plan, 4, 2)
    assert nativize_plan(plan, 4, 2) is plan


def test_a_coarse_plan_expands_to_the_rows_of_every_cell_in_painting_order() -> None:
    red, blue = (200, 0, 0), (0, 0, 200)
    plan = PaintPlan(
        4,
        2,
        (
            ColorGroup(red, (Stroke(0, 0, 3, 0), Stroke(3, 1, 1, 1)), 7),
            ColorGroup(blue, (Stroke(2, 1, 2, 1),), 1),
        ),
    )
    native = nativize_plan(plan, 8, 4)
    assert (native.width, native.height) == (8, 4)
    assert [g.color for g in native.color_groups] == [red, blue]
    first = native.color_groups[0].strokes
    assert first[:2] == (Stroke(0, 0, 7, 0), Stroke(0, 1, 7, 1))
    # the leftward stroke stays leftward on both of its rows
    assert first[2:] == (Stroke(7, 2, 2, 2), Stroke(7, 3, 2, 3))
    assert native.color_groups[1].strokes == (Stroke(4, 2, 5, 2), Stroke(4, 3, 5, 3))
    assert all(g.brush_diameter == 1 for g in native.color_groups)
    # The final colour of every texel is the final colour of its cell.
    coarse = _coverage(plan)
    fine = _coverage(native)
    for v in range(4):
        for u in range(8):
            assert fine[v, u] == coarse[v // 2, u // 2]
    assert stroke_index_map(plan, native) == [0, 0, 1, 1, 2, 2]


def test_fill_bands_cover_their_texel_rows_and_non_integer_ratios_still_tile() -> None:
    plan = PaintPlan(
        6,
        5,
        (ColorGroup((9, 9, 9), (Stroke(1, 2, 4, 2),), 12, brush_diameter=3),),
    )
    native = nativize_plan(plan, 13, 11)  # 13/6 and 11/5 texels per cell
    coarse = _coverage(plan)
    fine = _coverage(native)
    row_cell = {t: i for i in range(5) for t in range(*cell_span(i, 5, 11)) }
    row_cell.update({cell_span(i, 5, 11)[1]: i for i in range(5)})
    col_cell = {t: i for i in range(6) for t in range(*cell_span(i, 6, 13))}
    col_cell.update({cell_span(i, 6, 13)[1]: i for i in range(6)})
    for v in range(11):
        for u in range(13):
            assert fine[v, u] == coarse[row_cell[v], col_cell[u]]
    assert native.unpainted_pixels == int((fine < 0).sum())
