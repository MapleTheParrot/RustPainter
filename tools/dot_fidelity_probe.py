"""Score single-texel dot placement and drag row purity on the live sign.

The murica post-mortem measured half the fine detail landing a texel off.
This stamps isolated Size-1.00 dots at known texels through the painter's
exact aim arithmetic (cursor map, half-up rounding), captures the sign, and
reports how many landed on their texel; then draws full-width drags along
single rows at the painter's own pace and reports how much of each drag
stayed on its row.

    python tools/dot_fidelity_probe.py --out diagnostic/dots1

Escape aborts; the sign is cleared at the end.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.input_controller import create_system_input_controller
from app.models import ScreenRect
from app.painter import _high_resolution_timer
from app.profiles import ProfileStore
from app.screen import capture_region
from app.texel_grid import TexelGridModel
from tools._safety import Aborted, Guard, countdown
from tools.decimal_probe import _data_directory, _focus_rust
from tools.press_timing_probe import (
    FOCUS_SETTLE_SECONDS,
    hide_overlays,
    require_painting_ui,
    restore_overlays,
    select_color,
)

DOT_COLOR = (30, 30, 150)
DRAG_COLOR = (150, 30, 30)
PRESS_SECONDS = 0.07
DRAG_TEXELS_PER_SECOND = 250.0


def _write_brush_size(guard: Guard, controller, box, text: str) -> None:
    guard.click(box.left + box.width // 2, box.top + box.height // 2, settle=0.12)
    for key in ("BACKSPACE",) * 6 + ("DELETE",) * 6:
        guard.check()
        controller.press_key(key, hold_seconds=0.03)
        time.sleep(0.02)
    for char in text:
        guard.check()
        controller.press_key(0xBE if char == "." else char, hold_seconds=0.03)
        time.sleep(0.02)
    controller.press_key("ENTER", hold_seconds=0.03)
    time.sleep(0.2)


def _dab(guard: Guard, controller, x: int, y: int) -> None:
    guard.check()
    controller.move_mouse(x, y)
    guard.commanded(x, y)
    time.sleep(0.004)
    controller.mouse_down()
    time.sleep(PRESS_SECONDS)
    controller.mouse_up()
    time.sleep(0.02)


def _drag(guard: Guard, controller, start, end, pitch: float) -> None:
    guard.check()
    controller.move_mouse(*start)
    guard.commanded(*start)
    time.sleep(0.02)
    controller.mouse_down()
    try:
        distance = math.hypot(end[0] - start[0], end[1] - start[1])
        step = pitch
        steps = max(1, int(math.ceil(distance / step)))
        delay = (distance / (DRAG_TEXELS_PER_SECOND * pitch)) / steps
        for i in range(1, steps + 1):
            ratio = i / steps
            point = (
                math.floor(start[0] + (end[0] - start[0]) * ratio + 0.5),
                math.floor(start[1] + (end[1] - start[1]) * ratio + 0.5),
            )
            controller.move_mouse(*point)
            guard.commanded(*point)
            time.sleep(delay)
        time.sleep(PRESS_SECONDS)
    finally:
        controller.mouse_up()
    time.sleep(0.03)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    store = ProfileStore(_data_directory() / "profiles")
    profile = next(p for p in store.list_profiles() if "xxl" in p.name.lower())
    grid = TexelGridModel.from_dict(profile.metadata["texel_grid"])
    canvas = ScreenRect(
        profile.canvas.left, profile.canvas.top, profile.canvas.width, profile.canvas.height
    )
    clear = profile.clear_button

    def aim(u: float, v: float) -> tuple[int, int]:
        x, y = grid.cursor_point(u + 0.5, v + 0.5)
        return math.floor(x + 0.5), math.floor(y + 0.5)

    # Isolated dots: patches spread over the sign, every 3rd texel, plus the
    # exact corners and edge rows/columns the old grid could never reach.
    dot_texels: list[tuple[int, int]] = []
    for px in (6, 256, 508, 760, 1000):
        for py in (6, 130, 250, 380, 490):
            for du in range(0, 15, 3):
                for dv in range(0, 9, 3):
                    u, v = px + du, py + dv
                    if u < grid.columns and v < grid.rows:
                        dot_texels.append((u, v))
    edge_texels = [(0, 0), (1, 1), (0, 255), (1023, 0), (1023, 511), (0, 511), (512, 0), (512, 511), (0, 250), (1023, 250)]

    overlays = hide_overlays()
    _focus_rust()
    controller = create_system_input_controller()
    guard = Guard(controller, budget_seconds=420)
    time.sleep(FOCUS_SETTLE_SECONDS)
    require_painting_ui(profile)
    countdown(2, "dot fidelity probe")
    results: dict = {}
    try:
        before = capture_region(canvas).convert("RGB")
        before.save(args.out / "before.png")
        _write_brush_size(guard, controller, profile.brush_size_box, "1.00")
        select_color(guard, profile, DOT_COLOR)
        for u, v in dot_texels + edge_texels:
            _dab(guard, controller, *aim(u, v))
        time.sleep(0.5)
        after_dots = capture_region(canvas).convert("RGB")
        after_dots.save(args.out / "after_dots.png")

        # Drags: full-width rows.
        select_color(guard, profile, DRAG_COLOR)
        drag_rows = [30, 120, 256, 380, 470]
        for v in drag_rows:
            start = aim(3, v)
            end = aim(grid.columns - 4, v)
            _drag(guard, controller, start, end, grid.pitch_x)
        time.sleep(0.5)
        after_drags = capture_region(canvas).convert("RGB")
        after_drags.save(args.out / "after_drags.png")

        # ---------------- scoring (pure image math from here) ----------------
        base = np.asarray(before, dtype=int)
        dots_img = np.asarray(after_dots, dtype=int)
        drags_img = np.asarray(after_drags, dtype=int)

        def texel_center_capture(u: float, v: float) -> tuple[float, float]:
            return (
                grid.origin_x + (u + 0.5) * grid.pitch_x - canvas.left,
                grid.origin_y + (v + 0.5) * grid.pitch_y - canvas.top,
            )

        diff = np.abs(dots_img - base).max(axis=2)
        exact = 0
        displaced = []
        missing = []
        for u, v in dot_texels:
            cx, cy = texel_center_capture(u, v)
            window = diff[
                max(0, int(cy - 6)) : int(cy + 7), max(0, int(cx - 6)) : int(cx + 7)
            ]
            if window.max() < 40:
                missing.append((u, v))
                continue
            ys, xs = np.nonzero(window > max(40.0, 0.5 * window.max()))
            weight = window[ys, xs].astype(float)
            mx = float((xs * weight).sum() / weight.sum()) + max(0, int(cx - 6))
            my = float((ys * weight).sum() / weight.sum()) + max(0, int(cy - 6))
            du = (mx - cx) / grid.pitch_x
            dv = (my - cy) / grid.pitch_y
            landed_off = (round(du), round(dv))
            if landed_off == (0, 0):
                exact += 1
            else:
                displaced.append({"texel": [u, v], "off": list(landed_off),
                                  "subpixel": [round(du, 2), round(dv, 2)]})
        results["dots"] = {
            "total": len(dot_texels),
            "exact": exact,
            "displaced": len(displaced),
            "missing": len(missing),
            "displaced_detail": displaced[:40],
            "missing_detail": missing[:20],
        }
        print(f"DOTS: {exact}/{len(dot_texels)} exact, {len(displaced)} displaced, "
              f"{len(missing)} missing", flush=True)

        # Edge dots scored separately (the old grid left col 0 / rows 0-1 bare).
        edge_report = []
        for u, v in edge_texels:
            cx, cy = texel_center_capture(u, v)
            lo_x, lo_y = max(0, int(cx - 5)), max(0, int(cy - 5))
            window = diff[lo_y : int(cy + 6), lo_x : int(cx + 6)]
            edge_report.append({"texel": [u, v], "painted": bool(window.max() >= 40)})
        results["edges"] = edge_report
        print("EDGES:", ", ".join(f"{r['texel']}={'Y' if r['painted'] else 'n'}" for r in edge_report), flush=True)

        # Drag row purity.
        drag_diff = np.abs(drags_img - dots_img).max(axis=2)
        purity = []
        for v in drag_rows:
            _, cy = texel_center_capture(0, v)
            band = drag_diff[int(cy - 3 * grid.pitch_y) : int(cy + 3 * grid.pitch_y + 1), :]
            ys, xs = np.nonzero(band > 60)
            if len(ys) == 0:
                purity.append({"row": v, "painted_px": 0})
                continue
            offsets = (ys + int(cy - 3 * grid.pitch_y) - cy) / grid.pitch_y
            rows_hit = np.round(offsets).astype(int)
            on_row = float((rows_hit == 0).mean())
            purity.append({
                "row": v,
                "painted_px": int(len(ys)),
                "on_row_fraction": round(on_row, 3),
                "row_histogram": {int(k): int((rows_hit == k).sum()) for k in np.unique(rows_hit)},
            })
            print(f"DRAG row {v}: {len(ys)} px painted, {on_row * 100:.1f}% on row", flush=True)
        results["drags"] = purity

        guard.click(clear.left + clear.width // 2, clear.top + clear.height // 2, settle=0.8)
        print("canvas cleared", flush=True)
    except Aborted as stop:
        print(stop, flush=True)
        results["aborted"] = str(stop)
    finally:
        try:
            controller.release_all()
        except Exception:
            pass
        restore_overlays(overlays)
    (args.out / "result.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    with _high_resolution_timer():
        raise SystemExit(main())
