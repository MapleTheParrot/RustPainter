"""Measure where the cursor stamps every texel, from the sign's own export.

The painter used to infer the cursor-to-texel map from staircases of
stamps read off screen captures at under two pixels per texel, then fit a
lattice through the jumps.  Measured against the game's exported texture,
that fit's pitch was 0.4% off along one axis - two texels of aim error at
either edge of a 1024-wide sign, wrong-texel dabs across whole bands of
columns, and edge columns the clamp never let the cursor reach.

This module does not fit anything.  With the smallest brush, which stamps
exactly the one texel under the cursor, it presses once at every whole
screen pixel across the sign along each axis, exports the texture, and
reads which texel each press painted.  The result is a table: for every
texel column the first and last x that stamp it, and likewise for rows.
Aiming then means looking a texel up.  There is no rounding, no phase, no
audit; a pixel that is in the table was seen to stamp its texel.

A sweep is one press per pixel of the sign's width plus its height, in
lanes so that consecutive presses never share a texel: about 2,700 presses
on the largest sign, under a minute at the press timing the game was
measured to accept.  The tables generalise by construction: they are a
function of the screen geometry alone, measured on whatever sign is open.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np

from .texel_grid import TexelGridModel

LOGGER = logging.getLogger("rust_painter.cursor_map")

# Pixels swept beyond each edge of the calibrated rectangle.  The rectangle
# is hand-dragged just inside the canvas, and the game maps the cursor over
# a slightly wider quad than it draws (live: the last two texel columns of
# an XXL sign answered only to pixels 3-5 beyond the rectangle).
SWEEP_MARGIN_PIXELS = 8
# Lanes across the axis being swept: consecutive presses go to successive
# lanes, so two presses on one lane are this many pixels apart along the
# axis and cannot share a texel unless a texel is wider than that.
DEFAULT_LANES = 8
# Extra lanes beyond the sweep's, each with one press at the anchor pixel.
ANCHOR_LANES = 3
# The most lanes tried before giving up on a sign whose texels are wider
# than any lane stride.
MAX_LANES = 64


class SweepError(ValueError):
    """The sweep's export could not be read as one texel per press."""


@dataclass(frozen=True, slots=True)
class AxisTable:
    """One axis of the cursor map: which texel each swept pixel stamps."""

    positions: tuple[int, ...]
    texels: tuple[int | None, ...]
    count: int  # texels along this axis
    pitch: float  # least-squares summary, for logs and screen readers
    origin: float

    def runs(self) -> tuple[tuple[int, int], ...]:
        """(first, last) pixel per texel; raises if any texel has none."""

        first: dict[int, int] = {}
        last: dict[int, int] = {}
        for position, texel in zip(self.positions, self.texels):
            if texel is None:
                continue
            first.setdefault(texel, position)
            last[texel] = position
        missing = [k for k in range(self.count) if k not in first]
        if missing:
            raise SweepError(
                f"{len(missing)} texel(s) answered to no pixel: {missing[:12]}"
            )
        return tuple((first[k], last[k]) for k in range(self.count))

    @property
    def unreachable(self) -> tuple[int, ...]:
        seen = {t for t in self.texels if t is not None}
        return tuple(k for k in range(self.count) if k not in seen)


def lane_offsets(across_size: int, lanes: int) -> list[int]:
    """Where across the axis the lanes sit, relative to the rectangle's start.

    Spread over the middle of the sign, far enough apart that lanes land on
    different texel rows even on a coarse sign, and never near an edge.
    """

    gap = max(8, int(round(across_size / 40.0)))
    start = int(round(across_size * 0.3))
    return [start + k * gap for k in range(lanes)]


def sweep_positions(low: int, size: int, margin: int = SWEEP_MARGIN_PIXELS) -> list[int]:
    return list(range(low - margin, low + size + margin))


def attribute_sweep(
    alpha: np.ndarray,
    positions: Sequence[int],
    lanes: int,
    *,
    along_x: bool,
    interior: tuple[int, int],
    anchor: int | None = None,
) -> AxisTable:
    """Read one sweep's export: which texel did the press at each pixel paint?

    ``alpha`` is the export's alpha plane after the sweep on a cleared sign.
    Press ``i`` went to lane ``i % lanes``; the lanes are the ``lanes``
    lines (rows for an x sweep) carrying paint, in order.  Within a lane
    the presses are ``lanes`` pixels apart, so - as long as a texel is
    narrower than that - each press painted its own texel and the painted
    texels along the lane are in press order.  Presses outside the texture
    paint nothing; those happen only at the ends, so a lane's texels are a
    contiguous slice of its presses.  The slice's offset is found from the
    lattice every lane shares: ``interior`` bounds the pixels certainly on
    the texture, whose in-order pairs fix each lane's pitch and origin.

    That fixes the lattice only up to a whole lane stride: every lane
    losing its first press to the margin looks exactly like no lane losing
    any, the lattice a stride over.  ``anchor`` breaks the tie - one extra
    lane beyond the others carrying a single press at that pixel; the
    texel it painted says which candidate is the truth.
    """

    alpha = np.asarray(alpha)
    lines = np.nonzero((alpha > 0).sum(axis=1 if along_x else 0))[0]
    axis_name = "x" if along_x else "y"
    anchor_texel: int | None = None
    if anchor is not None:
        # The anchor lanes lie beyond the sweep's and carry one press each;
        # any that landed will do, and those that landed must agree.
        counts = [int(((alpha[line, :] if along_x else alpha[:, line]) > 0).sum()) for line in lines]
        if len(lines) < lanes + 1 or any(c != 1 for c in counts[lanes:]) or len(lines) > lanes + ANCHOR_LANES:
            raise SweepError(
                f"the {axis_name} sweep's export shows paint on {len(lines)} "
                f"{'rows' if along_x else 'columns'} ({counts}); expected {lanes} lanes "
                f"and up to {ANCHOR_LANES} single anchor presses"
            )
        marks = sorted(
            {
                int(np.nonzero((alpha[line, :] if along_x else alpha[:, line]) > 0)[0][0])
                for line in lines[lanes:]
            }
        )
        if len(marks) != 1:
            raise SweepError(f"the {axis_name} sweep's anchor presses disagree: texels {marks}")
        anchor_texel = marks[0]
        lines = lines[:lanes]
    elif len(lines) != lanes:
        raise SweepError(
            f"the {axis_name} sweep's export shows paint on {len(lines)} "
            f"{'rows' if along_x else 'columns'}, not the {lanes} lanes swept"
        )
    count = alpha.shape[1] if along_x else alpha.shape[0]
    positions = [int(p) for p in positions]
    stride = lanes
    per_lane: list[tuple[np.ndarray, np.ndarray]] = []
    fits: list[list[tuple[float, float]]] = []
    for k, line in enumerate(lines):
        painted = alpha[line, :] if along_x else alpha[:, line]
        texels = np.nonzero(painted > 0)[0].astype(np.int64)
        presses = np.array(positions[k::lanes], dtype=np.int64)
        if len(texels) < 4:
            raise SweepError(f"lane {k} of the {axis_name} sweep painted only {len(texels)} texels")
        # Presses in the interior all land, one texel each, in order: the
        # pitch is the press stride over the texels one stride advances.
        inside = (presses >= interior[0]) & (presses <= interior[1])
        n_inside = int(inside.sum())
        if n_inside < 4:
            raise SweepError("too few presses inside the rectangle to read a lattice")
        if len(texels) < 0.8 * n_inside:
            raise SweepError(
                f"texels along {axis_name} are wider than the {stride} px lane stride: "
                "presses on one lane can share a texel"
            )
        per_lane.append((texels, presses))
        # Candidate (pitch, origin) pairs for each possible number of leading
        # presses that missed the texture (0..4); the true one agrees with
        # the other lanes', the rest are a stride apart.
        candidates: list[tuple[float, float]] = []
        for lost in range(0, 5):
            n = min(len(texels), len(presses) - lost)
            if n < 4:
                break
            pairs_p = presses[lost : lost + n].astype(np.float64)
            pairs_t = texels[:n].astype(np.float64)
            keep = (pairs_p >= interior[0]) & (pairs_p <= interior[1])
            if keep.sum() < 4:
                continue
            design = np.stack([pairs_t[keep] + 0.5, np.ones(int(keep.sum()))], axis=1)
            (pitch, origin), *_ = np.linalg.lstsq(design, pairs_p[keep], rcond=None)
            if pitch <= 0:
                continue
            candidates.append((float(pitch), float(origin)))
        if not candidates:
            raise SweepError(f"lane {k} of the {axis_name} sweep could not be fitted")
        fits.append(candidates)
    # Consensus: the origin most lanes have a candidate within a pixel of.
    all_origins = [origin for lane in fits for _pitch, origin in lane]
    best_origin = None
    best_support = -1
    for candidate in all_origins:
        support = sum(
            1 for lane in fits if any(abs(o - candidate) <= 1.0 for _p, o in lane)
        )
        if support > best_support:
            best_support, best_origin = support, candidate
    assert best_origin is not None
    if best_support < max(2, (lanes + 1) // 2):
        raise SweepError(
            f"the {axis_name} sweep's lanes do not agree on a lattice "
            f"({best_support} of {lanes})"
        )
    chosen = []
    for lane in fits:
        chosen.append(min(lane, key=lambda po: abs(po[1] - best_origin)))
    pitch = float(np.median([p for p, _o in chosen]))
    origin = float(np.median([o for _p, o in chosen]))
    if stride < pitch:
        raise SweepError(
            f"texels are {pitch:.2f} px wide along {axis_name} but the lane "
            f"stride is {stride} px: presses on one lane can share a texel"
        )
    if anchor is not None and anchor_texel is not None:
        # The lattice is known up to a stride; the anchor press says which.
        candidates = [origin + shift * stride for shift in range(-6, 7)]
        matching = [
            o for o in candidates if int(np.floor((anchor - o) / pitch)) == anchor_texel
        ]
        if not matching:
            raise SweepError(
                f"no lattice a stride apart puts the {axis_name} anchor press at "
                f"{anchor} on texel {anchor_texel}"
            )
        origin = min(matching, key=lambda o: abs(o - origin))
    table: dict[int, int] = {}
    for texels, presses in per_lane:
        painted_set = set(int(t) for t in texels)
        taken: set[int] = set()
        for p in presses:
            continuous = (float(p) - origin) / pitch
            predicted = int(np.floor(continuous))
            fraction = continuous - predicted
            # The lattice says which texel this press stamped to well under
            # a pixel; the export says whether that texel was painted.  A
            # boundary a fraction off the lattice shows as the neighbour
            # painted instead, but only for a press near that boundary, so
            # the neighbour on the near side is the one fallback - never a
            # texel another press already claimed.
            candidates = [predicted]
            if fraction > 0.7:
                candidates.append(predicted + 1)
            elif fraction < 0.3:
                candidates.append(predicted - 1)
            for candidate in candidates:
                if candidate in painted_set and candidate not in taken:
                    table[int(p)] = candidate
                    taken.add(candidate)
                    break
    values = [table.get(p) for p in positions]
    # Each texel's pixels must be contiguous, and texels must not go
    # backwards along the axis: anything else is a misread export.
    last_texel = -1
    seen_end: dict[int, int] = {}
    for position, texel in zip(positions, values):
        if texel is None:
            continue
        if texel < last_texel:
            raise SweepError(
                f"the {axis_name} sweep read texel {texel} at {position} after "
                f"texel {last_texel}"
            )
        if texel in seen_end and seen_end[texel] != position - 1:
            raise SweepError(
                f"texel {texel} along {axis_name} answered to pixels that are "
                "not contiguous"
            )
        seen_end[texel] = position
        last_texel = texel
    return AxisTable(
        positions=tuple(positions),
        texels=tuple(values),
        count=int(count),
        pitch=pitch,
        origin=origin,
    )


def unread_positions(table: AxisTable) -> list[int]:
    """Pixels between the first and last read ones whose press left no texel.

    A press the game dropped, or a pixel on a boundary the lattice put on
    the wrong side; either way its texel is unknown and worth a second press.
    """

    known = [i for i, t in enumerate(table.texels) if t is not None]
    if not known:
        return []
    first, last = known[0], known[-1]
    return [table.positions[i] for i in range(first, last + 1) if table.texels[i] is None]


def lane_line(alpha: np.ndarray, along_x: bool, lane: int) -> int:
    """The row (or column) the sweep's ``lane``-th lane painted."""

    lines = np.nonzero((np.asarray(alpha) > 0).sum(axis=1 if along_x else 0))[0]
    return int(lines[lane])


def fill_in_sweep(table: AxisTable, alpha: np.ndarray, line: int, *, along_x: bool) -> AxisTable:
    """Read the fill-in export: the unread pixels were pressed again on ``line``.

    Only texels not already in the table are new; each is claimed by the
    unread pixel the lattice puts nearest, so a dropped press's pixel gets
    its texel and a pixel the sign truly ignores stays unread.
    """

    alpha = np.asarray(alpha)
    painted = alpha[line, :] if along_x else alpha[:, line]
    fresh = [int(t) for t in np.nonzero(painted > 0)[0] if t not in set(table.texels)]
    if not fresh:
        return table
    texels = list(table.texels)
    unread = [i for i, t in enumerate(texels) if t is None]
    for t in fresh:
        expected = table.origin + (t + 0.5) * table.pitch
        best = min(unread, key=lambda i: abs(table.positions[i] - expected), default=None)
        if best is not None and abs(table.positions[best] - expected) <= table.pitch:
            texels[best] = t
            unread.remove(best)
    return AxisTable(
        positions=table.positions,
        texels=tuple(texels),
        count=table.count,
        pitch=table.pitch,
        origin=table.origin,
    )


def grid_from_tables(x: AxisTable, y: AxisTable) -> TexelGridModel:
    """A grid model whose aim is the measured tables; the lattice is a summary."""

    columns = x.runs()
    rows = y.runs()
    # Where the texture is drawn is not measured here; the cursor lattice
    # is within a couple of pixels of it (live: 1.6 px), which is all a
    # reader of screen captures gets.
    return TexelGridModel(
        columns=x.count,
        rows=y.count,
        pitch_x=x.pitch,
        pitch_y=y.pitch,
        origin_x=x.origin,
        origin_y=y.origin,
        aim_origin_x=x.origin,
        aim_origin_y=y.origin,
        aim_pitch_x=x.pitch,
        aim_pitch_y=y.pitch,
        aim_columns=columns,
        aim_rows=rows,
        residual=0.0,
        from_edges=True,
    )


def lattice_targets(columns: int, rows: int, nx: int = 24, ny: int = 12) -> list[tuple[int, int]]:
    """A lattice of texels spread over the sign, edges included, to check a map."""

    targets = []
    for j in range(ny):
        for i in range(nx):
            u = round(i * (columns - 1) / (nx - 1))
            v = round(j * (rows - 1) / (ny - 1))
            targets.append((u, v))
    return targets


def check_lattice(alpha: np.ndarray, targets: Sequence[tuple[int, int]]) -> tuple[int, list[tuple[int, int]]]:
    """How many lattice dabs landed exactly on their texel, and which did not."""

    alpha = np.asarray(alpha)
    exact = 0
    wrong: list[tuple[int, int]] = []
    for u, v in targets:
        if alpha[v, u] >= 250:
            exact += 1
        else:
            wrong.append((u, v))
    return exact, wrong


__all__ = [
    "ANCHOR_LANES",
    "AxisTable",
    "DEFAULT_LANES",
    "fill_in_sweep",
    "lane_line",
    "unread_positions",
    "MAX_LANES",
    "SWEEP_MARGIN_PIXELS",
    "SweepError",
    "attribute_sweep",
    "check_lattice",
    "grid_from_tables",
    "lane_offsets",
    "lattice_targets",
    "sweep_positions",
]
