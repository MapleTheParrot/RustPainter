from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from app.models import ScreenRect
from app.texel_grid import (
    GridProbePlan,
    TexelGridModel,
    find_quad_edges,
    fit_staircase,
    ladder_offsets,
    measure_grid,
    stamp_diff,
)


class SimulatedSign:
    """A texture drawn on a quad with bilinear filtering, stamped per texel.

    ``cursor_shift`` is where the game's cursor-to-texel mapping sits relative
    to the texel lattice, and ``stamp_offset`` how many texels away from the
    cursor's texel the stamp actually lands - both of which a live sign was
    seen to do, and both of which the probe has to measure rather than
    assume.  ``snap=False`` paints a stamp centred wherever the cursor is,
    which is what a sign with no texel grid would look like.
    """

    def __init__(
        self,
        columns: int,
        rows: int,
        *,
        origin: tuple[float, float],
        pitch: tuple[float, float],
        cursor_shift: tuple[float, float] = (0.0, 0.0),
        stamp_offset: tuple[int, int] = (0, 0),
        cursor_pitch: tuple[float, float] | None = None,
        cursor_shear: tuple[float, float] = (0.0, 0.0),
        snap: bool = True,
        seed: int = 1,
    ) -> None:
        rng = np.random.default_rng(seed)
        self.columns, self.rows = columns, rows
        self.origin, self.pitch = origin, pitch
        # The game maps the cursor over a slightly different rectangle than
        # it draws the texture on, so the cursor's texel pitch can differ.
        self.cursor_pitch = cursor_pitch or pitch
        # A cursor ray-cast onto the sign in the world can be sheared
        # against the flat canvas: column boundaries slide with y.
        self.cursor_shear = cursor_shear
        self.cursor_shift, self.stamp_offset, self.snap = cursor_shift, stamp_offset, snap
        base = np.array([214, 204, 186], dtype=np.float32)
        self.texture = base + rng.normal(0.0, 4.0, (rows, columns, 3)).astype(np.float32)
        self.dabs = 0
        # A continuous stamp needs a finer canvas; keep unsnapped paint at
        # four samples per texel.
        self._fine = 4
        self._overlay = np.zeros((rows * self._fine, columns * self._fine, 4), np.float32)

    @property
    def quad(self) -> tuple[float, float, float, float]:
        ox, oy = self.origin
        return ox, oy, ox + self.columns * self.pitch[0], oy + self.rows * self.pitch[1]

    def dab(self, x: float, y: float, color=(200, 30, 160)) -> None:
        ox, oy = self.origin
        px, py = self.pitch
        # The game takes paint clicks on the texture and nowhere else.
        if not (ox <= x < ox + self.columns * px and oy <= y < oy + self.rows * py):
            self.dabs += 1
            return
        if self.snap:
            cx, cy = self.cursor_pitch
            sx, sy = self.cursor_shear
            column = math.floor((x - ox - self.cursor_shift[0] - sx * (y - oy)) / cx) + self.stamp_offset[0]
            row = math.floor((y - oy - self.cursor_shift[1] - sy * (x - ox)) / cy) + self.stamp_offset[1]
            if 0 <= column < self.columns and 0 <= row < self.rows:
                self.texture[row, column] = color
        else:
            fx = (x - ox) / px * self._fine
            fy = (y - oy) / py * self._fine
            radius = 0.6 * self._fine
            ys, xs = np.mgrid[0 : self.rows * self._fine, 0 : self.columns * self._fine]
            inside = (xs + 0.5 - fx) ** 2 + (ys + 0.5 - fy) ** 2 <= radius**2
            self._overlay[inside, :3] = color
            self._overlay[inside, 3] = 1.0
        self.dabs += 1

    def _texel_image(self) -> np.ndarray:
        if not self.snap and self._overlay[:, :, 3].any():
            fine = np.repeat(np.repeat(self.texture, self._fine, 0), self._fine, 1)
            alpha = self._overlay[:, :, 3:4]
            return fine * (1 - alpha) + self._overlay[:, :, :3] * alpha
        return self.texture

    def capture(self, rect) -> Image.Image:
        """Render the quad into a capture of ``rect`` with a dark panel around it."""

        image = np.full((rect.height, rect.width, 3), 58.0, dtype=np.float32)
        texels = self._texel_image()
        rows, columns = texels.shape[:2]
        ox, oy = self.origin
        px = self.pitch[0] * self.columns / columns
        py = self.pitch[1] * self.rows / rows
        xs = np.arange(rect.width) + rect.left + 0.5
        ys = np.arange(rect.height) + rect.top + 0.5
        u = (xs - ox) / px - 0.5
        v = (ys - oy) / py - 0.5
        inside_x = (xs >= ox) & (xs <= ox + self.columns * self.pitch[0])
        inside_y = (ys >= oy) & (ys <= oy + self.rows * self.pitch[1])
        u0 = np.clip(np.floor(u).astype(int), 0, columns - 1)
        u1 = np.clip(u0 + 1, 0, columns - 1)
        fu = np.clip(u - np.floor(u), 0, 1)[None, :, None]
        v0 = np.clip(np.floor(v).astype(int), 0, rows - 1)
        v1 = np.clip(v0 + 1, 0, rows - 1)
        fv = np.clip(v - np.floor(v), 0, 1)[:, None, None]
        top = texels[v0][:, u0] * (1 - fu) + texels[v0][:, u1] * fu
        bottom = texels[v1][:, u0] * (1 - fu) + texels[v1][:, u1] * fu
        rendered = top * (1 - fv) + bottom * fv
        mask = inside_y[:, None] & inside_x[None, :]
        image[mask] = rendered[mask]
        return Image.fromarray(np.clip(np.rint(image), 0, 255).astype(np.uint8), "RGB")


def _probe(sign: SimulatedSign, canvas: ScreenRect, *, with_edges: bool = True):
    colors = [(200, 30, 160), (30, 200, 60), (40, 90, 230), (240, 180, 20)]
    batches = {"count": 0}

    def stamp_batch(plan: GridProbePlan) -> np.ndarray:
        before = sign.capture(canvas)
        color = colors[batches["count"] % len(colors)]
        batches["count"] += 1
        for x, y in plan.points:
            # The painter keeps the mouse on the texture, within a pixel of
            # the calibrated rectangle; the harness keeps the same promise.
            x = min(max(math.floor(x), canvas.left - 1), canvas.left + canvas.width)
            y = min(max(math.floor(y), canvas.top - 1), canvas.top + canvas.height)
            sign.dab(x, y, color)
        after = sign.capture(canvas)
        return stamp_diff(before, after)

    edges = None
    if with_edges:
        margin = 12
        wide = ScreenRect(
            canvas.left - margin, canvas.top - margin, canvas.width + 2 * margin, canvas.height + 2 * margin
        )
        found = find_quad_edges(
            np.asarray(sign.capture(wide), dtype=np.float32),
            (margin, margin, margin + canvas.width, margin + canvas.height),
            margin - 2,
        )
        edges = tuple(None if e is None else e + (wide.left if i % 2 == 0 else wide.top) for i, e in enumerate(found))
    pitch_hint = canvas.height / 300.0
    return measure_grid(
        canvas, stamp_batch, pitch_hint=pitch_hint, stamp_hint=pitch_hint, edges=edges
    )


def _assert_aim_lands(sign: SimulatedSign, grid: TexelGridModel) -> None:
    """Aiming at texel centres plus the measured offset must paint those texels."""

    for column, row in ((3, 5), (grid.columns // 2, grid.rows // 3), (grid.columns - 4, grid.rows - 3), (2, grid.rows - 2), (grid.columns - 3, 2)):
        x, y = grid.cursor_point(column + 0.5, row + 0.5)
        before = sign.texture.copy()
        sign.dab(math.floor(x), math.floor(y), (1, 2, 3))
        changed = np.argwhere(np.any(sign.texture != before, axis=2))
        assert changed.tolist() == [[row, column]], (column, row, changed.tolist())


def test_grid_is_counted_exactly_from_a_hand_dragged_rectangle() -> None:
    sign = SimulatedSign(320, 320, origin=(286.3, 135.7), pitch=(4.4125, 4.3906))
    # The rectangle is a couple of pixels outside the quad on every side, as
    # one dragged to the visible edge would be.
    canvas = ScreenRect(284, 134, 1417, 1409)
    grid = _probe(sign, canvas)
    assert (grid.columns, grid.rows) == (320, 320)
    assert abs(grid.origin_x - 286.3) < 0.3
    assert abs(grid.origin_y - 135.7) < 0.3
    assert abs(grid.pitch_x - 4.4125) < 0.002
    assert abs(grid.pitch_y - 4.3906) < 0.002
    assert grid.from_edges
    _assert_aim_lands(sign, grid)


def test_a_stamp_offset_and_cursor_shift_are_measured_into_the_aim() -> None:
    # Stamps land a texel left and the cursor mapping sits a third of a
    # texel off the lattice - the live sign's behaviour.
    sign = SimulatedSign(
        256, 128, origin=(400.0, 300.2), pitch=(5.5, 5.5),
        cursor_shift=(1.8, -2.1), stamp_offset=(-1, 0),
    )
    canvas = ScreenRect(398, 299, 1412, 706)
    grid = _probe(sign, canvas)
    # With stamps landing a texel left of the cursor, the texture's last
    # column can only be reached from outside the calibrated rectangle - but
    # the column exists, and the count is the texture's true extent: the
    # quad edge proves the texel, and counting it is what lets a measured
    # grid snap onto the game's own texture table (1024 must read as 1024,
    # not as however many texels the frame and the cursor map left visible).
    # The painter clamps the unreachable column's clicks at paint time.
    assert (grid.columns, grid.rows) == (256, 128)
    assert abs(grid.pitch_x - 5.5) < 0.002
    assert grid.from_edges
    _assert_aim_lands(sign, grid)


def test_without_a_visible_quad_the_edge_stamps_count_the_texels() -> None:
    sign = SimulatedSign(128, 64, origin=(100.0, 100.0), pitch=(9.0, 9.0))
    canvas = ScreenRect(98, 101, 1154, 575)
    grid = _probe(sign, canvas, with_edges=False)
    assert (grid.columns, grid.rows) == (128, 64)
    # The extent comes from stamps at the edges, not from seeing the quad.
    assert grid.from_edges
    assert abs(grid.origin_x - 100.0) < 0.3


def test_a_sign_that_does_not_snap_is_refused() -> None:
    sign = SimulatedSign(200, 200, origin=(100.0, 100.0), pitch=(6.0, 6.0), snap=False)
    canvas = ScreenRect(100, 100, 1200, 1200)
    with pytest.raises(ValueError):
        _probe(sign, canvas, with_edges=False)


def test_staircase_reads_pitch_and_boundaries() -> None:
    pitch = 4.4
    cursor = list(range(0, 16))
    centres = [math.floor((c - 1.3) / pitch) * pitch + pitch / 2 + 0.1 for c in cursor]
    stair = fit_staircase(cursor, centres)
    assert abs(stair.coarse_pitch - pitch) < 0.01
    # Boundaries at 1.3, 5.7, 10.1, 14.5 bracketed to half a pixel.
    assert len(stair.jumps) == 4
    for jump, expected in zip(stair.jumps, (1.5, 5.5, 10.5, 14.5)):
        assert abs(jump - expected) <= 0.5


def test_ladder_reaches_the_far_side_in_a_few_rungs() -> None:
    offsets = ladder_offsets(4.4, 0.15, 300)
    assert offsets[-1] == 300
    assert len(offsets) <= 9
    assert all(b > a for a, b in zip(offsets, offsets[1:]))


def test_grid_model_round_trips_and_checks_against_the_canvas() -> None:
    grid = TexelGridModel(
        columns=320, rows=320, pitch_x=4.4, pitch_y=4.4, origin_x=286.0, origin_y=135.0,
        aim_origin_x=287.2, aim_origin_y=134.6, aim_pitch_x=4.39, aim_pitch_y=4.41,
    )
    assert TexelGridModel.from_dict(grid.to_dict()) == grid
    assert grid.agrees_with(ScreenRect(286, 135, 1412, 1410))
    assert not grid.agrees_with(ScreenRect(600, 135, 1412, 1410))
    rect = grid.registered_rect()
    assert (rect.left, rect.top) == (286.0, 135.0)
    assert abs(rect.width - 1408.0) < 1e-9


def test_a_cursor_lattice_unlike_the_rendered_one_is_measured_and_aimed() -> None:
    """Live, the cursor mapped over the visible quad (1294 px) while the texture
    was drawn 1299 px wide under the frame: pitches 4.044 against 4.060, more
    than a texel of drift across the sign.  One aim offset cannot paint that
    sign right; the measured cursor lattice can."""

    sign = SimulatedSign(
        320, 240,
        origin=(342.33, 178.45), pitch=(4.0602, 4.5133),
        cursor_pitch=(1294 / 320, 1078 / 240), cursor_shift=(6.67, 2.55),
        stamp_offset=(-1, 0),
    )
    canvas = ScreenRect(342, 177, 1299, 1085)
    grid = _probe(sign, canvas)
    assert abs(grid.pitch_x - 4.0602) < 0.002
    assert abs(grid.aim_pitch_x - 1294 / 320) < 0.004
    assert abs(grid.aim_pitch_y - 1078 / 240) < 0.004
    assert grid.rows == 240
    _assert_aim_lands(sign, grid)


def test_a_sheared_cursor_map_is_measured_and_aimed() -> None:
    """Live, the column boundaries sat three pixels further left at the
    bottom of the sign than at the top - a cursor ray-cast onto the sign in
    the world, against a canvas drawn flat - and a lattice measured in one
    band put a sixth of a test lattice's dots a texel over in the corner."""

    sign = SimulatedSign(
        320, 240,
        origin=(342.33, 178.45), pitch=(4.0602, 4.5133),
        cursor_pitch=(4.0546, 4.5134), cursor_shift=(6.9, -3.0),
        cursor_shear=(-0.0028, 0.0012),
    )
    canvas = ScreenRect(342, 177, 1299, 1085)
    grid = _probe(sign, canvas)
    assert (grid.columns, grid.rows) == (320, 240)
    # The shear summary shares its variance with the fitted twist; what is
    # painted uses the full map, which the aim check below holds to a texel.
    assert abs(grid.aim_shear_x - (-0.0028)) < 0.0012
    assert abs(grid.aim_shear_y - 0.0012) < 0.0012
    _assert_aim_lands(sign, grid)


class StraySign(SimulatedSign):
    """A sign where every Nth dab lands a texel short along one axis.

    Live, one staircase stamp landed a texel behind its neighbours along the
    axis being walked - the cursor had been within half a pixel of a texel
    boundary, and the game (rounding the cursor through a 1.25 DPI scale)
    put it on the other side.  The slip is along one axis at a time.
    """

    def __init__(self, *args, stray_every: int = 7, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.stray_every = stray_every

    def dab(self, x: float, y: float, color=(200, 30, 160)) -> None:
        if self.dabs % self.stray_every == self.stray_every - 1:
            if (self.dabs // self.stray_every) % 2 == 0:
                x -= self.pitch[0]
            else:
                y -= self.pitch[1]
        super().dab(x, y, color)


def test_a_stray_dab_cannot_miscount_the_ladder() -> None:
    """Live, one staircase stamp landed a whole texel back; a rung doing the
    same would have counted the sign a texel short, and did.  Three stamps
    per rung and the edges as a check make one stray dab harmless."""

    sign = StraySign(320, 240, origin=(345.0, 181.0), pitch=(4.04375, 4.49167), stray_every=7)
    canvas = ScreenRect(342, 177, 1299, 1085)
    grid = _probe(sign, canvas)
    assert (grid.columns, grid.rows) == (320, 240)
    assert abs(grid.pitch_x - 4.04375) < 0.002
    assert abs(grid.pitch_y - 4.49167) < 0.002
    assert abs(grid.origin_x - 345.0) < 0.3
