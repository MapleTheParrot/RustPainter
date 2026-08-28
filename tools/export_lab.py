"""Measure the sign through Rust's own export, texel by texel.

Every earlier probe read the sign off the screen at under two pixels per
texel, where a one-texel stamp is a blur of a few grey levels and half the
measurements are the detector's, not the game's.  The download button
writes the texture itself - alpha 255 where paint landed, 0 where it never
did - so these experiments stamp, export, and read the truth back.

    python tools/export_lab.py footprint --out diagnostic/lab1
    python tools/export_lab.py cursor-map --out diagnostic/lab1 --size 1
    python tools/export_lab.py timing --out diagnostic/lab1 --size 1
    python tools/export_lab.py drags --out diagnostic/lab1 --size 1

Escape aborts, as does a hand on the mouse or Rust losing focus; F7 too.
The sign is cleared before each experiment and left painted afterwards.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import math
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.input_controller import MouseButton, create_system_input_controller  # noqa: E402
from app.models import ScreenRect  # noqa: E402
from app.painter import _high_resolution_timer  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.screen import capture_region  # noqa: E402
from app.sign_export import ExportWatcher, SignExport  # noqa: E402
from app.texel_grid import TexelGridModel  # noqa: E402
from tools._safety import Aborted, Guard, countdown  # noqa: E402
from tools.decimal_probe import _data_directory, _focus_rust  # noqa: E402
from tools.press_timing_probe import (  # noqa: E402
    FOCUS_SETTLE_SECONDS,
    hide_overlays,
    require_painting_ui,
    restore_overlays,
    select_color,
    set_brush_size,
)

VK_F7 = 0x76

COLORS = (
    (230, 40, 40),
    (40, 90, 230),
    (40, 190, 70),
    (240, 190, 30),
    (210, 50, 200),
    (30, 200, 210),
    (250, 130, 40),
    (140, 70, 230),
)


class Lab:
    """The sign, the mouse, and the export, with the safety rails."""

    def __init__(self, out: Path, profile_hint: str | None = None, budget: float = 1800.0):
        self.out = out
        out.mkdir(parents=True, exist_ok=True)
        store = ProfileStore(_data_directory() / "profiles")
        profiles = store.list_profiles()
        if profile_hint:
            matches = [p for p in profiles if profile_hint.lower() in p.name.lower()]
            if len(matches) != 1:
                raise SystemExit(f"profile {profile_hint!r} matched {len(matches)}")
            self.profile = matches[0]
        else:
            self.profile = store.get_default() or profiles[0]
        p = self.profile
        if p.download_button is None or p.clear_button is None or p.brush_size_box is None:
            raise SystemExit("The profile needs the download, clear and Size controls calibrated")
        self.canvas = ScreenRect(p.canvas.left, p.canvas.top, p.canvas.width, p.canvas.height)
        # Off every widget: hovering the picker between bursts of presses
        # made the game drop most of every third burst (measured).
        self.park = (int(p.canvas.left - 40), int(p.canvas.top + p.canvas.height / 2))
        stored = p.metadata.get("texel_grid")
        self.grid = TexelGridModel.from_dict(stored) if stored else None
        self.overlays = hide_overlays()
        _focus_rust()
        self.input = create_system_input_controller()
        self.guard = Guard(self.input, budget_seconds=budget)
        time.sleep(FOCUS_SETTLE_SECONDS)
        require_painting_ui(p)
        self.watcher = ExportWatcher()
        self.exports = 0
        self.lut: dict | None = None
        self.last_anti_afk = time.monotonic()

    # ------------------------------------------------------------ controls

    def check(self) -> None:
        if ctypes.windll.user32.GetAsyncKeyState(VK_F7) & 0x8000:
            raise Aborted("Stopped: F7")
        self.guard.check()

    def close(self) -> None:
        try:
            self.input.release_all()
        except Exception:
            pass
        restore_overlays(self.overlays)

    def clear(self) -> None:
        c = self.profile.clear_button
        self.guard.click(c.left + c.width / 2, c.top + c.height / 2, settle=0.9)
        self.guard.commanded(*self.input.get_cursor_position())
        require_painting_ui(self.profile)
        self.guard.park(self.park)

    def size(self, value: float) -> None:
        set_brush_size(self.guard, self.profile, value)

    def color(self, rgb) -> None:
        select_color(self.guard, self.profile, rgb)

    def move(self, x: float, y: float) -> None:
        self.check()
        self.input.move_mouse(x, y)
        self.guard.commanded(x, y)

    def dab(self, x: int, y: int, hold: float = 0.07, gap: float = 0.02) -> None:
        self.move(x, y)
        time.sleep(0.002)
        self.input.mouse_down(MouseButton.LEFT)
        t = time.perf_counter()
        try:
            remaining = hold - (time.perf_counter() - t)
            if remaining > 0:
                time.sleep(remaining)
        finally:
            self.input.mouse_up(MouseButton.LEFT)
        if gap > 0:
            time.sleep(gap)

    def drag(self, points, seconds_per_step: float, dwell: float = 0.07) -> None:
        """Press at the first point, glide through the rest, dwell, release."""

        self.move(*points[0])
        time.sleep(0.004)
        self.input.mouse_down(MouseButton.LEFT)
        try:
            for x, y in points[1:]:
                self.move(x, y)
                if seconds_per_step > 0:
                    time.sleep(seconds_per_step)
            if dwell > 0:
                time.sleep(dwell)
        finally:
            self.input.mouse_up(MouseButton.LEFT)
        time.sleep(0.03)

    def line(self, start, end, hold: float = 0.07, lead: float = 0.05) -> None:
        """Rust's Shift line: press, jump, release with Shift held throughout."""

        self.move(*start)
        self.input.key_down("SHIFT")
        try:
            time.sleep(lead)
            self.input.mouse_down(MouseButton.LEFT)
            try:
                time.sleep(hold)
                self.move(*end)
                time.sleep(hold)
            finally:
                self.input.mouse_up(MouseButton.LEFT)
            time.sleep(lead)
        finally:
            self.input.key_up("SHIFT")
        time.sleep(0.03)

    def anti_afk(self) -> None:
        """Save the sign, jump, and reopen it with E - the app's own break.

        A server that kicks idle players watches for movement; a player who
        has stood at a sign for half an hour has made none.  Nothing touches
        the mouse while the UI is closed: the game owns the cursor then.
        """

        b = self.profile.save_button
        if b is None:
            raise SystemExit("The profile needs the Save button calibrated for the anti-AFK break")
        print("anti-AFK break: saving, jumping, reopening the sign", flush=True)
        self.move(b.left + b.width / 2, b.top + b.height / 2)
        time.sleep(0.05)
        self.input.mouse_down(MouseButton.LEFT)
        time.sleep(0.09)
        self.input.mouse_up(MouseButton.LEFT)
        time.sleep(0.5)
        self.input.press_key("SPACE", hold_seconds=0.1)
        time.sleep(2.0)
        self.input.press_key("E", hold_seconds=0.1)
        time.sleep(1.5)
        self.guard.commanded(*self.input.get_cursor_position())
        require_painting_ui(self.profile)
        self.guard.park(self.park, settle=0.4)
        self.last_anti_afk = time.monotonic()

    def export(self, tag: str) -> SignExport:
        b = self.profile.download_button
        # The game uploads the sign in the background and the download reads
        # the server's copy: exporting within 0.3 s of a burst of presses
        # lost the last half-second of them (measured); 3 s lost none.
        self.guard.park(self.park, settle=3.0)
        self.watcher.snapshot()
        self.guard.park((b.left + b.width // 2, b.top + b.height // 2), settle=0.1)
        self.guard.click(b.left + b.width / 2, b.top + b.height / 2, settle=0.3)
        self.guard.park(self.park, settle=0.1)
        export = self.watcher.collect(keep_copy_in=self.out)
        if export is None:
            raise SystemExit("No export appeared on the desktop")
        self.exports += 1
        target = self.out / f"export_{self.exports:02d}_{tag}.png"
        Path(export.source).replace(target)
        export = SignExport(export.rgb, export.painted, str(target))
        self.guard.commanded(*self.input.get_cursor_position())
        print(
            f"export {tag}: {export.columns}x{export.rows}, "
            f"{int(export.painted.sum())} texels painted -> {target.name}",
            flush=True,
        )
        return export

    def capture(self, tag: str):
        self.guard.park(self.park, settle=0.3)
        image = capture_region(self.canvas).convert("RGB")
        image.save(self.out / f"screen_{tag}.png")
        return image

    # ------------------------------------------------------------ geometry

    @property
    def columns(self) -> int:
        return self.grid.columns if self.grid else 1024

    @property
    def rows(self) -> int:
        return self.grid.rows if self.grid else 512

    def aim(self, u: float, v: float) -> tuple[int, int]:
        """A screen pixel that should stamp texel (u, v), from the stored grid."""

        if self.lut is not None:
            try:
                return lut_aim(self.lut, int(u), int(v))
            except KeyError:
                pass
        if self.grid is None:
            x = self.canvas.left + (u + 0.5) * self.canvas.width / self.columns
            y = self.canvas.top + (v + 0.5) * self.canvas.height / self.rows
            return math.floor(x + 0.5), math.floor(y + 0.5)
        x, y = self.grid.cursor_point(u + 0.5, v + 0.5)
        return math.floor(x + 0.5), math.floor(y + 0.5)


def alpha_of(export: SignExport) -> np.ndarray:
    from PIL import Image

    return np.asarray(Image.open(export.source).convert("RGBA"))[:, :, 3].astype(int)


# ------------------------------------------------------------- experiments


def footprint(lab: Lab, args) -> None:
    """What does one press paint, at each Size?  Read exactly from the export."""

    sizes = [float(s) for s in args.sizes.split(",")]
    lab.clear()
    placements = []
    rows, cols = lab.rows, lab.columns
    # Each Size gets a row band; dots are spread across the columns and
    # kept well apart so even a Size-10 stamp stays its own.
    band = max(14, (rows - 28) // max(1, len(sizes)))
    for i, size in enumerate(sizes):
        v = 14 + i * band
        if v >= rows - 14:
            break
        lab.size(size)
        lab.color(COLORS[i % len(COLORS)])
        for k in range(args.dots):
            u = 20 + k * ((cols - 40) // max(1, args.dots - 1))
            x, y = lab.aim(u, v)
            lab.dab(x, y, hold=args.hold / 1000.0)
            placements.append((size, u, v, x, y))
    export = lab.export("footprint")
    a = alpha_of(export)
    report = {}
    for size in sizes:
        mine = [p for p in placements if p[0] == size]
        if not mine:
            continue
        boxes = []
        for _s, u, v, x, y in mine:
            top, left = max(0, v - 12), max(0, u - 12)
            win = a[top : v + 13, left : u + 13]
            ys, xs = np.nonzero(win > 0)
            if len(ys) == 0:
                boxes.append(None)
                continue
            full = int((win >= 250).sum())
            any_ = int((win > 0).sum())
            h = int(ys.max() - ys.min() + 1)
            w = int(xs.max() - xs.min() + 1)
            wgt = win[ys, xs].astype(float)
            cx = (xs * wgt).sum() / wgt.sum() - (u - left)
            cy = (ys * wgt).sum() / wgt.sum() - (v - top)
            boxes.append(
                {
                    "w": w,
                    "h": h,
                    "full": full,
                    "any": any_,
                    "cx": round(float(cx), 2),
                    "cy": round(float(cy), 2),
                    "own": int(a[v, u]),
                    "alphas": sorted({int(t) for t in win[ys, xs]})[:12],
                }
            )
        landed = [b for b in boxes if b]
        summary = {
            "dots": len(mine),
            "landed": len(landed),
            "median_w": float(np.median([b["w"] for b in landed])) if landed else None,
            "median_h": float(np.median([b["h"] for b in landed])) if landed else None,
            "median_full": float(np.median([b["full"] for b in landed])) if landed else None,
            "median_any": float(np.median([b["any"] for b in landed])) if landed else None,
            "own_full": sum(1 for b in landed if b["own"] >= 250),
            "mean_offset_texels": (
                [
                    round(float(np.mean([b["cx"] for b in landed])), 2),
                    round(float(np.mean([b["cy"] for b in landed])), 2),
                ]
                if landed
                else None
            ),
            "detail": boxes,
        }
        report[str(size)] = summary
        print(
            f"Size {size:5.2f}: {summary['landed']}/{summary['dots']} landed, "
            f"{summary['own_full']} with the aimed texel at full alpha; footprint "
            f"{summary['median_w']}x{summary['median_h']} texels, {summary['median_full']} "
            f"at alpha>=250 of {summary['median_any']} touched; centre offset "
            f"{summary['mean_offset_texels']} texels",
            flush=True,
        )
        for b in landed[:2]:
            print(f"      e.g. {b}", flush=True)
    (lab.out / "footprint.json").write_text(json.dumps(report, indent=2))


def _sweep_axis(lab: Lab, args, along_x: bool) -> dict:
    """Dab at every screen pixel along one axis; which texel took each dab?"""

    canvas = lab.canvas
    cols, rows = lab.columns, lab.rows
    lanes = 8
    if along_x:
        positions = list(range(canvas.left - 2, canvas.left + canvas.width + 2))
        lane_texels = [rows // 4 + 6 * k for k in range(lanes)]
    else:
        positions = list(range(canvas.top - 2, canvas.top + canvas.height + 2))
        lane_texels = [cols // 3 + 6 * k for k in range(lanes)]
    lab.color(COLORS[1 if along_x else 0])
    dabs = []
    for i, pos in enumerate(positions):
        lane = lane_texels[i % lanes]
        if along_x:
            _, y = lab.aim(0, lane)
            x = pos
        else:
            x, _ = lab.aim(lane, 0)
            y = pos
        lab.dab(x, y, hold=args.hold / 1000.0, gap=args.gap / 1000.0)
        dabs.append((pos, lane, x, y))
    export = lab.export("sweep_x" if along_x else "sweep_y")
    a = alpha_of(export)
    table = {}
    unattributed = 0
    multi = 0
    for pos, lane, x, y in dabs:
        if along_x:
            line = a[lane, :]
            expected = (
                (x - lab.grid.origin_x) / lab.grid.pitch_x - 0.5
                if lab.grid
                else (x - canvas.left) / canvas.width * cols
            )
        else:
            line = a[:, lane]
            expected = (
                (y - lab.grid.origin_y) / lab.grid.pitch_y - 0.5
                if lab.grid
                else (y - canvas.top) / canvas.height * rows
            )
        lo = max(0, int(expected) - 3)
        hi = min(len(line), int(expected) + 4)
        hits = [k for k in range(lo, hi) if line[k] > 0]
        if not hits:
            table[pos] = None
            unattributed += 1
            continue
        strong = [k for k in hits if line[k] >= 250]
        pick = strong if strong else hits
        if len(pick) > 1:
            multi += 1
        table[pos] = int(min(pick, key=lambda k: abs(k - expected)))
    values = [table[p] for p in positions]
    runs = []
    start = 0
    for i in range(1, len(values) + 1):
        if i == len(values) or values[i] != values[start]:
            runs.append((positions[start], positions[i - 1], values[start]))
            start = i
    lengths = [r[1] - r[0] + 1 for r in runs if r[2] is not None]
    axis = "x" if along_x else "y"
    print(
        f"{axis} sweep: {len(positions)} dabs, {unattributed} unattributed, "
        f"{multi} ambiguous; run lengths {sorted(Counter(lengths).items())}",
        flush=True,
    )
    seen = sorted({v for v in values if v is not None})
    missing = (
        [k for k in range(seen[0], seen[-1] + 1) if k not in set(seen)] if seen else []
    )
    if seen:
        print(
            f"  texels reached: {seen[0]}..{seen[-1]} ({len(seen)} of "
            f"{seen[-1] - seen[0] + 1}); unreachable: {missing[:30]}",
            flush=True,
        )
    backwards = sum(
        1
        for i in range(1, len(values))
        if values[i] is not None and values[i - 1] is not None and values[i] < values[i - 1]
    )
    print(f"  backwards steps: {backwards}", flush=True)
    return {
        "positions": positions,
        "texel": values,
        "runs": runs,
        "unattributed": unattributed,
        "ambiguous": multi,
        "missing": missing,
    }


def cursor_map(lab: Lab, args) -> None:
    lab.clear()
    lab.size(args.size)
    result = {"size": args.size, "hold_ms": args.hold}
    result["x"] = _sweep_axis(lab, args, along_x=True)
    lab.clear()
    result["y"] = _sweep_axis(lab, args, along_x=False)
    (lab.out / "cursor_map.json").write_text(json.dumps(result))


def timing(lab: Lab, args) -> None:
    """Drop rate as a function of hold and gap, read exactly."""

    holds = [float(h) for h in args.holds.split(",")]
    gaps = [float(g) for g in args.gaps.split(",")]
    lab.clear()
    lab.size(args.size)
    cols, rows = lab.columns, lab.rows
    batches = [(h, gaps[0]) for h in holds] + [(holds[0], g) for g in gaps[1:]]
    placements = []
    n = args.dots
    for b, (hold, gap) in enumerate(batches):
        lab.color(COLORS[b % len(COLORS)])
        v0 = 10 + b * max(8, (rows - 20) // len(batches))
        for k in range(n):
            u = 8 + (k * (cols - 16)) // n
            v = v0 + (k % 4) * 2
            x, y = lab.aim(u, v)
            lab.dab(x, y, hold=hold / 1000.0, gap=gap / 1000.0)
            placements.append((b, u, v))
    export = lab.export("timing")
    a = alpha_of(export)
    report = []
    for b, (hold, gap) in enumerate(batches):
        mine = [(u, v) for bb, u, v in placements if bb == b]
        landed = sum(
            1 for u, v in mine if a[max(0, v - 1) : v + 2, max(0, u - 1) : u + 2].max() > 0
        )
        report.append({"hold_ms": hold, "gap_ms": gap, "dots": len(mine), "landed": landed})
        print(f"hold {hold:5.1f} ms gap {gap:5.1f} ms: {landed}/{len(mine)} landed", flush=True)
    (lab.out / "timing.json").write_text(json.dumps(report, indent=2))


def drags(lab: Lab, args) -> None:
    """Coverage of drags and lines along rows, at several rates."""

    rates = [float(r) for r in args.rates.split(",")]
    lab.clear()
    lab.size(args.size)
    cols = lab.columns
    pitch = lab.grid.pitch_x if lab.grid else lab.canvas.width / cols
    plan = []
    row = 20
    step_px = args.step_texels * pitch
    for i, rate in enumerate(rates):
        lab.color(COLORS[i % len(COLORS)])
        for _rep in range(args.reps):
            u0, u1 = 4, cols - 5
            x0, y0 = lab.aim(u0, row)
            x1, _ = lab.aim(u1, row)
            n = max(1, int(math.ceil((x1 - x0) / step_px)))
            pts = [(math.floor(x0 + (x1 - x0) * k / n + 0.5), y0) for k in range(n + 1)]
            per_step = (u1 - u0) / rate / n
            lab.drag(pts, per_step, dwell=0.07)
            plan.append(("drag", rate, row, u0, u1))
            row += 6
    lab.color(COLORS[len(rates) % len(COLORS)])
    for _rep in range(args.reps):
        u0, u1 = 4, cols - 5
        x0, y0 = lab.aim(u0, row)
        x1, _ = lab.aim(u1, row)
        lab.line((x0, y0), (x1, y0))
        plan.append(("line", 0.0, row, u0, u1))
        row += 6
    export = lab.export("drags")
    a = alpha_of(export)
    report = []
    for kind, rate, v, u0, u1 in plan:
        band = a[max(0, v - 2) : v + 3, u0 : u1 + 1]
        own = a[v, u0 : u1 + 1]
        covered_full = int((own >= 250).sum())
        covered_any = int((own > 0).sum())
        holes = [u0 + k for k in range(u1 - u0 + 1) if own[k] < 250]
        rows_touched = int(((band > 0).sum(axis=1) > 0).sum())
        report.append(
            {
                "kind": kind,
                "rate": rate,
                "row": v,
                "texels": u1 - u0 + 1,
                "full": covered_full,
                "any": covered_any,
                "rows_touched": rows_touched,
                "holes": holes[:40],
            }
        )
        print(
            f"{kind} {rate:6.0f} texel/s row {v}: {covered_full}/{u1 - u0 + 1} full alpha, "
            f"{covered_any} touched, rows touched {rows_touched}, holes {len(holes)}: {holes[:20]}",
            flush=True,
        )
    (lab.out / "drags.json").write_text(json.dumps(report, indent=2))


def cadence(lab: Lab, args) -> None:
    """Replay the sweep cadences that dropped presses, with variations.

    Each variant stamps a stream of dabs in its own row band; the in-game
    FPS counter is photographed during each so a frame-rate collapse shows
    up next to the drop count.
    """

    from PIL import Image

    lab.clear()
    lab.size(args.size)
    cols, rows = lab.columns, lab.rows
    hud = ScreenRect(lab.canvas.left - 87, lab.canvas.top + rows * 0 - 1174 + 1360 - (-1174 - lab.canvas.top) - 1174 + 1174, 260, 80)
    # The FPS counter sits at the bottom-left of the Rust window: 260x80 px
    # above the window's bottom edge, which is the canvas top minus 266
    # plus 1440 on this layout; measured off a screenshot.
    hud = ScreenRect(149, -80, 260, 80)
    variants = [
        # name, lanes, dx per dab (px), settle after move, hold, gap, lane_axis
        ("x-sweep replica", 8, 1, 0.002, args.hold / 1000, args.gap / 1000, "x"),
        ("y-sweep replica", 8, 1, 0.002, args.hold / 1000, args.gap / 1000, "y"),
        ("x-sweep, 40ms settle after move", 8, 1, 0.040, args.hold / 1000, args.gap / 1000, "x"),
        ("x-sweep, 80ms gap", 8, 1, 0.002, args.hold / 1000, 0.080, "x"),
        ("x-sweep, 2 lanes", 2, 1, 0.002, args.hold / 1000, args.gap / 1000, "x"),
        ("x-sweep, 30ms hold", 8, 1, 0.002, 0.030, args.gap / 1000, "x"),
    ]
    n = args.dots
    report = []
    band = 0
    for name, lanes, step, settle, hold, gap, axis in variants:
        lab.color(COLORS[band % len(COLORS)])
        placements = []
        if axis == "x":
            lane_texels = [12 + band * 60 + 6 * k for k in range(lanes)]
            x0 = lab.canvas.left + 60
            for i in range(n):
                lane = lane_texels[i % lanes]
                _, y = lab.aim(0, lane)
                x = x0 + i * step
                lab.move(x, y)
                time.sleep(settle)
                lab.dab(x, y, hold=hold, gap=gap)
                placements.append((x, y, lane))
                if i == n // 2:
                    capture_region(hud).save(lab.out / f"hud_{band}.png")
        else:
            lane_texels = [12 + band * 60 + 6 * k for k in range(lanes)]
            y0 = lab.canvas.top + 40
            for i in range(n):
                lane = lane_texels[i % lanes]
                x, _ = lab.aim(lane + 400, 0)
                y = y0 + i * step
                lab.move(x, y)
                time.sleep(settle)
                lab.dab(x, y, hold=hold, gap=gap)
                placements.append((x, y, lane))
                if i == n // 2:
                    capture_region(hud).save(lab.out / f"hud_{band}.png")
        report.append((name, axis, lanes, placements, band))
        band += 1
    export = lab.export("cadence")
    a = alpha_of(export)
    out = []
    for name, axis, lanes, placements, band in report:
        per_lane = []
        lane_texels = sorted({p[2] for p in placements})
        for lane in lane_texels:
            if axis == "x":
                painted = int((a[lane, :] > 0).sum())
            else:
                painted = int((a[:, lane + 400] > 0).sum())
            per_lane.append(painted)
        dabs = len(placements)
        landed = sum(per_lane)
        out.append({"variant": name, "dabs": dabs, "landed": landed, "per_lane": per_lane})
        print(f"{name:36s}: {landed}/{dabs} landed; per lane {per_lane}", flush=True)
    (lab.out / "cadence.json").write_text(json.dumps(out, indent=2))


def _load_lut(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def lut_aim(lut: dict, u: int, v: int) -> tuple[int, int]:
    """The middle screen pixel of texel column u and row v from a sweep table."""

    def middle(axis: dict, k: int) -> int:
        pos = [p for p, t in zip(axis["positions"], axis["texel"]) if t == k]
        if not pos:
            raise KeyError(k)
        return pos[len(pos) // 2]

    return middle(lut["x"], u), middle(lut["y"], v)


def edge(lab: Lab, args) -> None:
    """Can the outermost texels be stamped from just outside the rectangle?"""

    lut = _load_lut(args.lut)
    lab.clear()
    lab.size(args.size)
    lab.color(COLORS[0])
    c = lab.canvas
    plan = []
    rows = [40, 200, 300, 470]
    for v in rows:
        _, y = lut_aim(lut, 0, v)
        for x in range(c.left + c.width - 4, c.left + c.width + 10):
            lab.dab(x, y, hold=args.hold / 1000.0)
            plan.append(("right", x, y, v))
        for x in range(c.left - 8, c.left + 3):
            lab.dab(x, y, hold=args.hold / 1000.0)
            plan.append(("left", x, y, v))
    lab.color(COLORS[1])
    cols = [60, 400, 700, 990]
    for u in cols:
        x, _ = lut_aim(lut, u, 0)
        for y in range(c.top + c.height - 4, c.top + c.height + 10):
            lab.dab(x, y, hold=args.hold / 1000.0)
            plan.append(("bottom", x, y, u))
        for y in range(c.top - 8, c.top + 3):
            lab.dab(x, y, hold=args.hold / 1000.0)
            plan.append(("top", x, y, u))
    export = lab.export("edge")
    a = alpha_of(export)
    for side in ("right", "left"):
        for v in rows:
            painted = [int(u) for u in np.nonzero(a[v, :] > 0)[0] if (u >= 1000 if side == "right" else u < 24)]
            print(f"{side} edge, row {v}: painted texels {painted}", flush=True)
    for side in ("bottom", "top"):
        for u in cols:
            painted = [int(v) for v in np.nonzero(a[:, u] > 0)[0] if (v >= 490 if side == "bottom" else v < 24)]
            print(f"{side} edge, col {u}: painted texels {painted}", flush=True)


def lattice(lab: Lab, args) -> None:
    """Aim a lattice of dabs through the sweep tables; did every one land on its texel?"""

    lut = _load_lut(args.lut)
    lab.clear()
    lab.size(args.size)
    lab.color(COLORS[2])
    cols, rows = lab.columns, lab.rows
    targets = []
    nx, ny = args.lattice_x, args.lattice_y
    for j in range(ny):
        for i in range(nx):
            u = round(1 + i * (cols - 3) / (nx - 1))
            v = round(1 + j * (rows - 3) / (ny - 1))
            targets.append((u, v))
    for u, v in targets:
        try:
            x, y = lut_aim(lut, u, v)
        except KeyError:
            continue
        lab.dab(x, y, hold=args.hold / 1000.0)
    export = lab.export("lattice")
    a = alpha_of(export)
    exact = 0
    off = []
    missing = []
    for u, v in targets:
        if a[v, u] >= 250:
            exact += 1
            continue
        win = a[max(0, v - 2) : v + 3, max(0, u - 2) : u + 3]
        ys, xs = np.nonzero(win > 0)
        if len(ys) == 0:
            missing.append((u, v))
        else:
            off.append((u, v, [(int(x) - min(2, u), int(y) - min(2, v)) for y, x in zip(ys, xs)]))
    print(f"lattice: {exact}/{len(targets)} exact, {len(off)} displaced {off[:12]}, {len(missing)} missing {missing[:12]}", flush=True)
    (lab.out / "lattice.json").write_text(json.dumps({"exact": exact, "total": len(targets), "off": off, "missing": missing}))


def bands(lab: Lab, args) -> None:
    """Which Size drags a band of 2 or 3 rows at full alpha, whatever the row's phase?"""

    lut = _load_lut(args.lut)
    lab.clear()
    cols, rows = lab.columns, lab.rows
    trials = [(2.0, "boundary"), (2.5, "boundary"), (3.0, "centre"), (3.35, "centre"), (3.5, "centre"), (3.75, "centre"), (4.0, "centre"), (4.5, "centre"), (5.0, "centre")]
    plan = []
    row = 12
    for i, (size, aim) in enumerate(trials):
        lab.size(size)
        lab.color(COLORS[i % len(COLORS)])
        for _rep in range(args.reps):
            u0, u1 = 6, cols // 2
            x0, y0 = lut_aim(lut, u0, row)
            x1, _ = lut_aim(lut, u1, row)
            if aim == "boundary":
                _, y1 = lut_aim(lut, u0, row + 1)
                y = (y0 + y1) // 2
            else:
                y = y0
            lab.drag([(x0, y), (x1, y)], 0.0, dwell=0.03)
            plan.append((size, aim, row, u0, u1, y))
            row += 9
    export = lab.export("bands")
    a = alpha_of(export)
    out = []
    for size, aim, v, u0, u1, y in plan:
        cover = {}
        for dv in range(-3, 5):
            r = v + dv
            if 0 <= r < rows:
                line = a[r, u0 : u1 + 1]
                cover[dv] = (int((line >= 250).sum()), int((line > 0).sum()))
        n = u1 - u0 + 1
        desc = " ".join(f"{dv:+d}:{f}/{t}" for dv, (f, t) in cover.items() if t)
        print(f"Size {size:4.2f} {aim:8s} row {v:3d} (y {y}): of {n} texels per row -> {desc}", flush=True)
        out.append({"size": size, "aim": aim, "row": v, "n": n, "cover": {str(k): list(v2) for k, v2 in cover.items()}})
    (lab.out / "bands.json").write_text(json.dumps(out, indent=2))


def dragloss(lab: Lab, args) -> None:
    """How often is a whole drag lost, per drag mechanic?  Size 1, one row each."""

    lut = _load_lut(args.lut)
    lab.clear()
    lab.size(1.0)
    cols, rows = lab.columns, lab.rows
    variants = [
        ("jump: press, jump, 30ms, release", dict(settle=0.0, pre=0.0, steps=1, dwell=0.03)),
        ("settled: 20ms before and after press", dict(settle=0.02, pre=0.02, steps=1, dwell=0.03)),
        ("stepped: 8 moves, 70ms dwell", dict(settle=0.004, pre=0.0, steps=8, dwell=0.07)),
        ("jump, 5ms dwell", dict(settle=0.0, pre=0.0, steps=1, dwell=0.005)),
    ]
    plan = []
    row = 8
    n = args.dots
    for i, (name, m) in enumerate(variants):
        lab.color(COLORS[i % len(COLORS)])
        for _k in range(n):
            u0, u1 = 6, cols // 2
            x0, y = lut_aim(lut, u0, row)
            x1, _ = lut_aim(lut, u1, row)
            lab.move(x0, y)
            time.sleep(m["settle"])
            lab.input.mouse_down(MouseButton.LEFT)
            time.sleep(m["pre"])
            for step in range(1, m["steps"] + 1):
                lab.move(round(x0 + (x1 - x0) * step / m["steps"]), y)
            time.sleep(m["dwell"])
            lab.input.mouse_up(MouseButton.LEFT)
            time.sleep(0.02)
            plan.append((name, row, u0, u1))
            row += 2
            if row >= rows - 2:
                break
    export = lab.export("dragloss")
    a = alpha_of(export)
    for name, _m in variants:
        mine = [(r, u0, u1) for nm, r, u0, u1 in plan if nm == name]
        full = sum(1 for r, u0, u1 in mine if (a[r, u0 : u1 + 1] >= 250).all())
        partial = sum(1 for r, u0, u1 in mine if (a[r, u0 : u1 + 1] > 0).any() and not (a[r, u0 : u1 + 1] >= 250).all())
        nothing = len(mine) - full - partial
        print(f"{name:40s}: {full}/{len(mine)} rows complete, {partial} partial, {nothing} lost entirely", flush=True)


def dragend(lab: Lab, args) -> None:
    """Where must a drag end so its last texel is painted and the next is not?"""

    lut = _load_lut(args.lut)
    lab.clear()
    cols, rows = lab.columns, lab.rows

    def run_of(u):
        pos = [p for p, t in zip(lut["x"]["positions"], lut["x"]["texel"]) if t == u]
        return pos[0], pos[-1]

    plan = []
    row = 6
    for size, aim in ((1.0, "centre"), (2.0, "boundary")):
        lab.size(size)
        for direction in (1, -1):
            for offset_name in ("first-1", "first", "first+1", "last", "last+1"):
                lab.color(COLORS[len(plan) % len(COLORS)])
                for k in range(args.dots):
                    u1 = 300 + k * 7 if direction == 1 else 700 - k * 7
                    u0 = u1 - direction * 40
                    x0, y = lut_aim(lut, u0, row)
                    if aim == "boundary":
                        _, y1 = lut_aim(lut, u0, row + 1)
                        y = (y + y1) // 2
                    first, last = run_of(u1)
                    if direction == -1:
                        first, last = last, first  # "first" is the first pixel met when travelling
                    e = {"first-1": first - direction, "first": first, "first+1": first + direction, "last": last, "last+1": last + direction}[offset_name]
                    lab.move(x0, y)
                    time.sleep(0.004)
                    lab.input.mouse_down(MouseButton.LEFT)
                    lab.move(e, y)
                    time.sleep(0.03)
                    lab.input.mouse_up(MouseButton.LEFT)
                    time.sleep(0.02)
                    plan.append((size, direction, offset_name, row, u0, u1))
                    row += 3 if size > 1 else 2
    export = lab.export("dragend")
    a = alpha_of(export)
    from collections import Counter
    summary = {}
    for size, direction, offset_name, v, u0, u1 in plan:
        key = (size, direction, offset_name)
        end_ok = a[v, u1] >= 250
        over = a[v, u1 + direction] > 0
        body = (a[v, min(u0, u1) : max(u0, u1) + 1] >= 250).all() if not over else None
        c = summary.setdefault(key, Counter())
        c["n"] += 1
        c["end painted"] += int(end_ok)
        c["overshoot"] += int(over)
        c["body full"] += int(bool(body)) if body is not None else 0
    for key, c in summary.items():
        print(f"Size {key[0]} dir {key[1]:+d} end at {key[2]:8s}: end texel painted {c['end painted']}/{c['n']}, next texel touched {c['overshoot']}/{c['n']}", flush=True)


def edges2(lab: Lab, args) -> None:
    """The texture's four edges across the whole sign: how straight are they on screen?"""

    lab.clear()
    lab.size(1.0)
    c = lab.canvas
    xs = [c.left + 10 + k * (c.width - 20) // 29 for k in range(30)]
    ys = [c.top + 10 + k * (c.height - 20) // 19 for k in range(20)]
    plan = []
    lab.color(COLORS[0])
    for x in xs:
        for y in range(c.top + c.height - 6, c.top + c.height + 4):
            lab.dab(x, y, hold=0.02, gap=0.005)
            plan.append(("bottom", x, y))
        for y in range(c.top - 4, c.top + 6):
            lab.dab(x, y, hold=0.02, gap=0.005)
            plan.append(("top", x, y))
    lab.color(COLORS[1])
    for y in ys:
        for x in range(c.left - 4, c.left + 6):
            lab.dab(x, y, hold=0.02, gap=0.005)
            plan.append(("left", x, y))
        for x in range(c.left + c.width - 6, c.left + c.width + 6):
            lab.dab(x, y, hold=0.02, gap=0.005)
            plan.append(("right", x, y))
    export = lab.export("edges2")
    a = alpha_of(export)
    rows, cols = a.shape
    # bottom/top: for each x, which rows got paint and the lowest/highest y that painted the edge row
    for side in ("bottom", "top"):
        print(side, flush=True)
        for x in xs:
            # find the texel column of this x from any painted texel in the edge rows
            edge_row = rows - 1 if side == "bottom" else 0
            painted_cols = np.nonzero(a[edge_row, :] > 0)[0]
            # closest painted column to the expected one from the lab5 lattice
            u_expected = int((x - 235.4) / 1.7731)
            near = [u for u in painted_cols if abs(u - u_expected) <= 2]
            print(f"   x {x}: edge-row texels painted at columns {near}; rows painted in this column band: {sorted(set(int(v) for u in near for v in np.nonzero(a[:, u] > 0)[0] if (v > rows - 8 if side == 'bottom' else v < 8)))}", flush=True)
    for side in ("left", "right"):
        print(side, flush=True)
        for y in ys:
            edge_col = cols - 1 if side == "right" else 0
            painted_rows = np.nonzero(a[:, edge_col] > 0)[0]
            v_expected = int((y + 1176.09) / 1.7735)
            near = [v for v in painted_rows if abs(v - v_expected) <= 2]
            print(f"   y {y}: edge-column texels painted at rows {near}; columns painted in this row band: {sorted(set(int(u) for v in near for u in np.nonzero(a[v, :] > 0)[0] if (u > cols - 8 if side == 'right' else u < 8)))}", flush=True)


def edges3(lab: Lab, args) -> None:
    """Where exactly is each texture edge, across the sign?  One press per column."""

    lut = _load_lut(args.lut)
    lab.clear()
    lab.size(1.0)
    c = lab.canvas

    def col_of(x):
        i = lut["x"]["positions"].index(x) if x in lut["x"]["positions"] else None
        return lut["x"]["texel"][i] if i is not None else None

    def row_of(y):
        i = lut["y"]["positions"].index(y) if y in lut["y"]["positions"] else None
        return lut["y"]["texel"][i] if i is not None else None

    plan = []
    lab.color(COLORS[0])
    bases = [c.left + 12 + k * (c.width - 60) // 24 for k in range(25)]
    for x0 in bases:
        for k in range(12):
            x = x0 + 4 * k
            lab.dab(x, c.top + c.height - 8 + k, hold=0.02, gap=0.005)
            plan.append(("bottom", x, c.top + c.height - 8 + k))
            lab.dab(x, c.top - 4 + k, hold=0.02, gap=0.005)
            plan.append(("top", x, c.top - 4 + k))
    lab.color(COLORS[1])
    ybases = [c.top + 12 + k * (c.height - 60) // 17 for k in range(18)]
    for y0 in ybases:
        for k in range(12):
            y = y0 + 4 * k
            lab.dab(c.left - 4 + k, y, hold=0.02, gap=0.005)
            plan.append(("left", c.left - 4 + k, y))
            lab.dab(c.left + c.width - 8 + k, y, hold=0.02, gap=0.005)
            plan.append(("right", c.left + c.width - 8 + k, y))
    export = lab.export("edges3")
    a = alpha_of(export)
    rows, cols = a.shape
    out = []
    for side, x, y in plan:
        if side in ("bottom", "top"):
            u = col_of(x)
            if u is None:
                continue
            band = range(rows - 6, rows) if side == "bottom" else range(0, 6)
            hit = [v for v in band if a[v, u] > 0]
            out.append((side, x, y, hit))
        else:
            v = row_of(y)
            if v is None:
                continue
            band = range(cols - 6, cols) if side == "right" else range(0, 6)
            hit = [u for u in band if a[v, u] > 0]
            out.append((side, x, y, hit))
    (lab.out / "edges3.json").write_text(json.dumps(out))
    # summarise: for each base, the first coordinate that paints the outermost texel and the first that paints nothing
    for side in ("bottom", "top", "left", "right"):
        print(side, flush=True)
        groups: dict = {}
        for s_, x, y, hit in out:
            if s_ != side:
                continue
            key = (x - (x - c.left - 12) % 4) if side in ("bottom", "top") else (y - (y - c.top - 12) % 4)
            groups.setdefault(key, []).append(((y if side in ("bottom", "top") else x), hit))
        for key in sorted(groups):
            seq = sorted(groups[key])
            desc = " ".join(f"{p}:{h[0] if h else '-'}" for p, h in seq)
            print(f"   base {key}: {desc}", flush=True)


def stream(lab: Lab, args) -> None:
    """A long fast stream of presses: does the tail get lost, and does waiting help?"""

    lut = _load_lut(args.lut)
    lab.clear()
    lab.size(1.0)
    lab.color(COLORS[0])
    n = args.dots
    hold, gap = args.hold / 1000.0, args.gap / 1000.0
    if args.tail == "bottom":
        # the last 200 presses walk the bottom rows, ending on row 511
        texels = [(8 + (i % 250) * 4, 8 + (i // 250) * 2) for i in range(n - 200)]
        texels += [(8 + (i % 100) * 10, 496 + (i // 100) * 2 + (1 if i >= 100 else 0)) for i in range(200)]
        texels = texels[:-100] + [(8 + i * 10, 511 if i % 2 else 509) for i in range(100)]
    elif args.tail == "corner":
        texels = [(8 + (i % 250) * 4, 8 + (i // 250) * 2) for i in range(n - 100)]
        texels += [(1 + (i % 10), 480 + (i // 10) * 3) for i in range(100)]
    elif args.tail == "top":
        texels = [(8 + (i % 250) * 4, 40 + (i // 250) * 2) for i in range(n - 100)]
        texels += [(8 + i * 10, 2 if i % 2 else 0) for i in range(100)]
    elif args.tail == "right":
        texels = [(8 + (i % 250) * 4, 40 + (i // 250) * 2) for i in range(n - 100)]
        texels += [(1014 + (i % 10), 200 + (i // 10) * 3) for i in range(100)]
    else:
        texels = [(8 + (i % 250) * 4, 8 + (i // 250) * 2) for i in range(n)]
    if args.after:
        texels += [(400 + (i % 50) * 4, 300 + (i // 50) * 2) for i in range(args.after)]
    t0 = time.perf_counter()
    for u, v in texels:
        x, y = lut_aim(lut, u, v)
        lab.dab(x, y, hold=hold, gap=gap)
    print(f"{n} presses in {time.perf_counter() - t0:.1f}s; waiting {args.settle}s before the export", flush=True)
    lab.guard.park(lab.park, settle=args.settle)
    before = np.asarray(lab.capture("stream_before_export"), dtype=np.float32)
    export = lab.export("stream")
    after = np.asarray(lab.capture("stream_after_export"), dtype=np.float32)
    a = alpha_of(export)
    missing = [i for i, (u, v) in enumerate(texels) if a[v, u] == 0]
    # were the missing dabs visible on screen before / after the export?
    bare = np.asarray(lab.capture("stream_bare_ref"), dtype=np.float32) * 0 + 200.0  # placeholder, replaced below
    def visible(img, u, v):
        x, y = lut_aim(lut, u, v)
        px, py = x - lab.canvas.left, y - lab.canvas.top
        patch = img[max(0, py - 1) : py + 2, max(0, px - 1) : px + 2]
        return float(np.linalg.norm(patch.reshape(-1, 3) - np.array([195, 179, 162], dtype=np.float32), axis=1).max())
    if missing:
        vis_before = [round(visible(before, *texels[i])) for i in missing[:12]]
        vis_after = [round(visible(after, *texels[i])) for i in missing[:12]]
        ok_before = [round(visible(before, *texels[i])) for i in range(max(0, missing[0] - 6), missing[0])]
        print(f"   screen contrast at the first missing dabs before export: {vis_before}; after: {vis_after}; at the 6 landed dabs just before them: {ok_before}", flush=True)
    print(f"missing {len(missing)} of {n}: indices {missing[:20]}{' ...' if len(missing) > 20 else ''}", flush=True)
    if missing:
        print(f"   first missing index {missing[0]}, last {missing[-1]}", flush=True)


def bursts(lab: Lab, args) -> None:
    """Fast bursts each followed by the cursor leaving the sign: is the last press lost?"""

    lut = _load_lut(args.lut)
    lab.clear()
    lab.size(1.0)
    lab.color(COLORS[0])
    hold, gap = args.hold / 1000.0, args.gap / 1000.0
    n_bursts, per = 20, 30
    texels = []
    for b in range(n_bursts):
        burst = [(8 + (i * 4) % 1000, 20 + b * 12 + (i // 250)) for i in range(per)]
        if args.pick:
            lab.color(COLORS[b % len(COLORS)])
            if args.pick_wait > 0:
                lab.guard.park((600, -150), settle=args.pick_wait)
        for u, v in burst:
            x, y = lut_aim(lut, u, v)
            lab.dab(x, y, hold=hold, gap=gap)
        if args.tail == "slow":
            # repeat the last press of the burst, held a frame
            x, y = lut_aim(lut, *burst[-1])
            lab.dab(x, y, hold=0.07, gap=0.02)
        elif args.tail == "linger":
            time.sleep(0.5)
        texels.append(burst)
        if args.park == "picker":
            lab.guard.park(lab.park, settle=0.5)
        elif args.park == "dark":
            lab.guard.park((600, -150), settle=0.5)
        elif args.park == "prelinger":
            lab.guard.park(lab.park, settle=0.5)
            x, y = lut_aim(lut, 500, 300)
            lab.move(x, y)
            time.sleep(0.5)
        elif args.park == "none":
            time.sleep(0.5)
    export = lab.export("bursts")
    a = alpha_of(export)
    lost = {}
    for b, burst in enumerate(texels):
        missing = [i for i, (u, v) in enumerate(burst) if a[v, u] == 0]
        if missing:
            lost[b] = missing
    print(f"tail={args.tail}: bursts with losses {len(lost)}/{n_bursts}: {lost}", flush=True)


EXPERIMENTS = {
    "footprint": footprint,
    "cursor-map": cursor_map,
    "timing": timing,
    "drags": drags,
    "cadence": cadence,
    "edge": edge,
    "lattice": lattice,
    "bands": bands,
    "dragloss": dragloss,
    "dragend": dragend,
    "edges2": edges2,
    "edges3": edges3,
    "stream": stream,
    "bursts": bursts,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment", choices=tuple(EXPERIMENTS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--sizes", type=str, default="1,1.25,1.5,1.75,2,2.5,3,3.35,4,6,8,10")
    parser.add_argument("--size", type=float, default=1.0)
    parser.add_argument("--dots", type=int, default=8)
    parser.add_argument("--hold", type=float, default=70.0)
    parser.add_argument("--gap", type=float, default=20.0)
    parser.add_argument("--holds", type=str, default="70,40,30,20,12,6")
    parser.add_argument("--gaps", type=str, default="20,12,8,5,2,0")
    parser.add_argument("--rates", type=str, default="250,600,900,1500,3000")
    parser.add_argument("--reps", type=int, default=2)
    parser.add_argument("--step-texels", type=float, default=1.0)
    parser.add_argument("--countdown", type=int, default=3)
    parser.add_argument("--lut", type=Path, default=Path("diagnostic/lab5/lut.json"))
    parser.add_argument("--lattice-x", type=int, default=32)
    parser.add_argument("--lattice-y", type=int, default=16)
    parser.add_argument("--budget", type=float, default=1800.0)
    parser.add_argument("--settle", type=float, default=0.3)
    parser.add_argument("--tail", type=str, default="rows")
    parser.add_argument("--after", type=int, default=0)
    parser.add_argument("--park", type=str, default="picker")
    parser.add_argument("--pick", action="store_true")
    parser.add_argument("--pick-wait", type=float, default=0.0)
    parser.add_argument("--anti-afk", action="store_true", help="save, jump and reopen the sign first")
    args = parser.parse_args()
    lab = Lab(args.out, args.profile, budget=args.budget)
    if args.lut and Path(args.lut).exists() and args.experiment not in ("cursor-map",):
        lab.lut = _load_lut(args.lut)
        print(f"aiming through {args.lut}", flush=True)
    try:
        countdown(args.countdown, args.experiment)
        if args.anti_afk:
            lab.anti_afk()
        EXPERIMENTS[args.experiment](lab, args)
    except Aborted as stop:
        print(stop, flush=True)
        return 1
    finally:
        lab.close()
    return 0


if __name__ == "__main__":
    with _high_resolution_timer():
        raise SystemExit(main())
