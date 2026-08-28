"""The export sweep reads a cursor map pixel by pixel, without fitting."""

from __future__ import annotations

import math

import numpy as np
import pytest

from app.cursor_map import (
    AxisTable,
    SweepError,
    attribute_sweep,
    check_lattice,
    grid_from_tables,
    lane_offsets,
    lattice_targets,
    sweep_positions,
)
from app.texel_grid import TexelGridModel


def _simulate_sweep(
    origin: float,
    pitch: float,
    count: int,
    positions: list[int],
    lanes: int,
    *,
    along_x: bool,
    other: int = 40,
    jitter=None,
    anchor: int | None = None,
):
    """The alpha plane of a sign swept along one axis, one texel per press."""

    shape = (other, count) if along_x else (count, other)
    alpha = np.zeros(shape, dtype=int)
    lane_lines = [5 + 3 * k for k in range(lanes + 1)]
    presses = [(position, lane_lines[i % lanes]) for i, position in enumerate(positions)]
    if anchor is not None:
        presses.append((anchor, lane_lines[lanes]))
    for position, line in presses:
        boundary_shift = jitter(position) if jitter else 0.0
        texel = math.floor((position + boundary_shift - origin) / pitch)
        if not 0 <= texel < count:
            continue
        if along_x:
            alpha[line, texel] = 255
        else:
            alpha[texel, line] = 255
    return alpha


def test_a_sweep_reads_every_pixel_to_its_texel_including_beyond_the_rectangle() -> None:
    origin, pitch, count = 235.379, 1.77312, 1024
    left, width = 236, 1810
    positions = sweep_positions(left, width, 8)
    anchor = left + width // 2
    alpha = _simulate_sweep(origin, pitch, count, positions, 8, along_x=True, anchor=anchor)
    table = attribute_sweep(
        alpha, positions, 8, along_x=True, interior=(left + 40, left + width - 41), anchor=anchor
    )
    assert table.count == count
    assert table.unreachable == ()
    assert abs(table.pitch - pitch) < 0.002
    assert abs(table.origin - origin) < 0.5
    for position, texel in zip(table.positions, table.texels):
        expected = math.floor((position - origin) / pitch)
        assert texel == (expected if 0 <= expected < count else None)
    runs = table.runs()
    assert runs[0][0] >= left - 8 and runs[-1][1] > left + width  # the last texels lie past the rectangle


def test_a_sweep_survives_boundary_jitter_because_nothing_is_fitted() -> None:
    origin, pitch, count = -1176.1, 1.77344, 512
    top, height = -1174, 909
    positions = sweep_positions(top, height, 8)
    rng = np.random.default_rng(3)
    shifts = {p: float(rng.uniform(-0.3, 0.3)) for p in positions}
    anchor = top + height // 2
    alpha = _simulate_sweep(
        origin, pitch, count, positions, 8, along_x=False, jitter=shifts.get, anchor=anchor
    )
    table = attribute_sweep(
        alpha, positions, 8, along_x=False, interior=(top + 40, top + height - 41), anchor=anchor
    )
    assert table.unreachable == ()
    for position, texel in zip(table.positions, table.texels):
        expected = math.floor((position + shifts[position] - origin) / pitch)
        assert texel == (expected if 0 <= expected < count else None)


def test_a_lane_stride_narrower_than_a_texel_is_refused_so_the_caller_can_widen_it() -> None:
    origin, pitch, count = 100.0, 12.0, 64  # a small sign shown huge
    positions = sweep_positions(100, 768, 8)
    alpha = _simulate_sweep(origin, pitch, count, positions, 8, along_x=True)
    with pytest.raises(SweepError, match="share a texel"):
        attribute_sweep(alpha, positions, 8, along_x=True, interior=(140, 828))
    positions = sweep_positions(100, 768, 8)
    anchor = 100 + 384
    alpha = _simulate_sweep(origin, pitch, count, positions, 16, along_x=True, other=80, anchor=anchor)
    table = attribute_sweep(alpha, positions, 16, along_x=True, interior=(140, 828), anchor=anchor)
    assert table.unreachable == ()
    assert abs(table.origin - origin) < 0.5


def test_a_wrong_lane_count_in_the_export_is_an_error() -> None:
    positions = sweep_positions(236, 1810, 8)
    alpha = _simulate_sweep(235.4, 1.7731, 1024, positions, 8, along_x=True)
    alpha[0, 3] = 255  # a stray mark on a ninth row
    with pytest.raises(SweepError, match="9 rows"):
        attribute_sweep(alpha, positions, 8, along_x=True, interior=(276, 2005))


def test_without_an_anchor_a_margin_that_costs_every_lane_a_press_is_ambiguous() -> None:
    origin, pitch, count = 235.379, 1.77312, 1024
    positions = sweep_positions(236, 1810, 8)
    alpha = _simulate_sweep(origin, pitch, count, positions, 8, along_x=True)
    table = attribute_sweep(alpha, positions, 8, along_x=True, interior=(276, 2005))
    # a lattice one stride over explains the export just as well...
    assert abs(table.origin - origin) > 1.0
    anchor = 236 + 905
    alpha = _simulate_sweep(origin, pitch, count, positions, 8, along_x=True, anchor=anchor)
    table = attribute_sweep(alpha, positions, 8, along_x=True, interior=(276, 2005), anchor=anchor)
    # ...until one press at a known pixel says which lattice is real.
    assert abs(table.origin - origin) < 0.5


def test_unreachable_texels_are_reported_not_hidden() -> None:
    table = AxisTable(positions=(0, 1, 2, 3), texels=(0, 1, 1, 3), count=4, pitch=1.0, origin=0.0)
    assert table.unreachable == (2,)
    with pytest.raises(SweepError, match="answered to no pixel"):
        table.runs()


def test_the_grid_aims_at_run_middles_and_ends_drags_one_pixel_past_the_first() -> None:
    x = AxisTable(positions=tuple(range(10, 20)), texels=(0, 0, 1, 1, 2, 3, 3, 4, 4, 5), count=6, pitch=1.7, origin=9.5)
    y = AxisTable(positions=tuple(range(50, 58)), texels=(0, 0, 1, 2, 2, 3, 3, 4), count=5, pitch=1.7, origin=49.5)
    grid = grid_from_tables(x, y)
    assert grid.swept
    assert grid.aim_columns[0] == (10, 11) and grid.aim_columns[2] == (14, 14)
    assert grid.aim_pixel(0, 0) == (10, 50)
    assert grid.aim_pixel(2, 2) == (14, 53)
    assert grid.drag_end_x(3, +1) == 16 and grid.drag_end_x(3, -1) == 15
    assert grid.drag_end_x(2, +1) == 15 and grid.drag_end_x(2, -1) == 13
    assert grid.drag_end_y(4, +1) == 58 and grid.drag_end_y(1, -1) == 51
    # cursor_point keeps the continuous convention: the centre is the middle
    assert grid.cursor_point(2.5, 2.5) == (14.0, 53.5)
    clamp = grid.clamp_rect(None)
    assert (clamp.left, clamp.top) == (9.0, 49.0)
    assert clamp.left + clamp.width - 1 == 20.0 and clamp.top + clamp.height - 1 == 58.0
    again = TexelGridModel.from_dict(grid.to_dict())
    assert again.aim_columns == grid.aim_columns and again.aim_rows == grid.aim_rows


def test_the_lattice_check_reports_the_dabs_that_missed() -> None:
    targets = lattice_targets(1024, 512, 4, 2)
    assert targets[0] == (0, 0) and targets[-1] == (1023, 511)
    alpha = np.zeros((512, 1024), dtype=int)
    for u, v in targets:
        alpha[v, u] = 255
    alpha[511, 1023] = 0
    exact, wrong = check_lattice(alpha, targets)
    assert exact == len(targets) - 1 and wrong == [(1023, 511)]


def test_lanes_sit_apart_in_the_middle_of_the_sign() -> None:
    offsets = lane_offsets(909, 8)
    assert offsets[0] == round(909 * 0.3)
    assert all(b - a >= 8 for a, b in zip(offsets, offsets[1:]))
    assert offsets[-1] < 909
