"""Does a small brush overwrite paint already on the sign, or only tint it?

Seven thousand presses on the largest sign landed at every hold, yet a
nine-hour job on it lost a third of its dabs - and lost them in a pattern no
dropped press can produce: a dab over earlier paint "missed" half the time
whatever its colour, a dab over bare canvas only when its colour was close
to the canvas.  That is not a press the game never saw.  That is a stamp the
game painted and that did not cover what was under it.

This paints a few blocks of one colour, lets them settle, then puts dabs of
a strongly contrasting colour on them and on the bare canvas beside them at
the sizes a fine plan uses, and reads how far each dab moved its patch
toward the new colour: 100% is an opaque stamp, 0% is nothing.

    python tools/overpaint_probe.py --out diagnostic/over1 --sizes 1,1.15,1.5,2,3

Needs the painting UI open and the Size box calibrated.  Escape, a moved
mouse, or Rust losing focus stop it.

Result on the XXL sign, 2026-08-23: every dab visible at every size over
paint and over bare canvas (40/40 each); a Size 1 dab moves its patch 72%
of the way to its color over paint and 55% over the canvas's texture, 1.5
and up about 89% - the stamp's soft edge at 1.77 px a texel, not
transparency.  The lost paint was swallowed picker clicks (see
app/color_swatch.py), not presses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.input_controller import MouseButton, create_system_input_controller  # noqa: E402
from app.models import ScreenRect  # noqa: E402
from app.painter import _high_resolution_timer  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.screen import capture_region  # noqa: E402
from tools._safety import Aborted, Guard, countdown  # noqa: E402
from tools.decimal_probe import _data_directory, _focus_rust  # noqa: E402
from tools.press_timing_probe import (  # noqa: E402
    FOCUS_SETTLE_SECONDS,
    hide_overlays,
    paint_dot,
    require_painting_ui,
    restore_overlays,
    select_color,
    set_brush_size,
)

BASE = (30, 60, 220)  # the block: blue
DAB = (240, 200, 30)  # the dabs: yellow, far from both the blue and the canvas


def fill_block(guard, rect: ScreenRect, size_px: float) -> None:
    """Drag a big brush back and forth across a rectangle to paint it solid."""

    y = rect.top + size_px / 2
    while y < rect.top + rect.height:
        guard.check()
        guard.input.move_mouse(rect.left, y)
        guard.commanded(rect.left, y)
        time.sleep(0.004)
        guard.input.mouse_down(MouseButton.LEFT)
        x = rect.left
        while x < rect.left + rect.width:
            x += 6
            guard.input.move_mouse(min(x, rect.left + rect.width), y)
            time.sleep(0.004)
        time.sleep(0.08)
        guard.input.mouse_up(MouseButton.LEFT)
        guard.commanded(min(x, rect.left + rect.width), y)
        time.sleep(0.03)
        y += size_px * 0.6


def coverage(before: np.ndarray, after: np.ndarray, canvas: ScreenRect, point, target, patch=7):
    """How far the pixels around a dab moved toward the target colour, 0-1."""

    cx, cy = int(point[0] - canvas.left), int(point[1] - canvas.top)
    h = patch // 2
    b = before[cy - h : cy + h + 1, cx - h : cx + h + 1].reshape(-1, 3)
    a = after[cy - h : cy + h + 1, cx - h : cx + h + 1].reshape(-1, 3)
    t = np.asarray(target, dtype=np.float32)
    # Per pixel, the projection of the change onto the before->target line.
    direction = t - b
    length = np.linalg.norm(direction, axis=1)
    ok = length > 20
    if not ok.any():
        return float("nan")
    moved = ((a - b) * direction).sum(axis=1) / np.maximum(length, 1e-6) ** 2
    return float(np.clip(moved[ok].max(), 0.0, 1.5))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--sizes", type=str, default="1,1.15,1.5,2,3")
    parser.add_argument("--dots", type=int, default=40, help="dabs per size per surface")
    parser.add_argument("--hold", type=float, default=70.0)
    parser.add_argument("--countdown", type=int, default=5)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    sizes = [float(v) for v in args.sizes.split(",") if v.strip()]

    store = ProfileStore(_data_directory() / "profiles")
    profiles = store.list_profiles()
    if args.profile:
        wanted = args.profile.strip().lower()
        matches = [p for p in profiles if wanted in p.name.strip().lower()]
        if len(matches) != 1:
            raise SystemExit(f"--profile {args.profile!r} is ambiguous or unknown")
        profile = matches[0]
    else:
        profile = store.get_default() or profiles[0]
    canvas = ScreenRect(
        profile.canvas.left, profile.canvas.top, profile.canvas.width, profile.canvas.height
    )
    park = (
        int(profile.color_box.left + profile.color_box.width / 2),
        int(profile.color_box.top + profile.color_box.height / 2),
    )
    # One horizontal strip per size: a painted block on the left half, bare
    # canvas on the right half, dabs along the middle of each.
    strip_h = (canvas.height - 40) / len(sizes)
    spacing = 14.0

    overlays = hide_overlays()
    _focus_rust()
    controller = create_system_input_controller()
    guard = Guard(controller, budget_seconds=900)
    time.sleep(FOCUS_SETTLE_SECONDS)
    require_painting_ui(profile)
    countdown(args.countdown, "measuring overpaint")

    results = []
    try:
        if profile.clear_button is not None:
            clear = profile.clear_button
            guard.click(clear.left + clear.width / 2, clear.top + clear.height / 2, settle=0.9)
            guard.commanded(*controller.get_cursor_position())
            require_painting_ui(profile)
            guard.park(park)
        # The blocks, with a wide brush.
        set_brush_size(guard, profile, 20.0)
        select_color(guard, profile, BASE)
        blocks = []
        for i in range(len(sizes)):
            top = canvas.top + 20 + i * strip_h
            block = ScreenRect(canvas.left + 30, int(top + 8), int(canvas.width / 2 - 60), int(strip_h - 16))
            fill_block(guard, block, 36.0)
            blocks.append(block)
        guard.park(park, settle=0.7)
        before = np.asarray(capture_region(canvas).convert("RGB"), dtype=np.float32)
        capture_region(canvas).save(args.out / "blocks.png")
        select_color(guard, profile, DAB)
        for i, size in enumerate(sizes):
            set_brush_size(guard, profile, size)
            block = blocks[i]
            y = int(block.top + block.height / 2)
            on_paint = [(int(block.left + 20 + k * spacing), y) for k in range(args.dots)]
            on_bare = [
                (int(canvas.left + canvas.width / 2 + 40 + k * spacing), y) for k in range(args.dots)
            ]
            for p in on_paint + on_bare:
                paint_dot(guard, p, args.hold / 1000.0, 0.0)
            guard.park(park, settle=0.7)
            after = np.asarray(capture_region(canvas).convert("RGB"), dtype=np.float32)
            cov_paint = [coverage(before, after, canvas, p, DAB) for p in on_paint]
            cov_bare = [coverage(before, after, canvas, p, DAB) for p in on_bare]
            row = {
                "size": size,
                "over_paint_median": round(float(np.nanmedian(cov_paint)), 3),
                "over_paint_landed": int(sum(c > 0.3 for c in cov_paint)),
                "over_bare_median": round(float(np.nanmedian(cov_bare)), 3),
                "over_bare_landed": int(sum(c > 0.3 for c in cov_bare)),
                "dots": args.dots,
            }
            results.append(row)
            print(
                f"  size {size:5.2f}: over paint moved {100 * row['over_paint_median']:4.0f}% toward the colour "
                f"({row['over_paint_landed']}/{args.dots} visible) | over bare {100 * row['over_bare_median']:4.0f}% "
                f"({row['over_bare_landed']}/{args.dots} visible)",
                flush=True,
            )
            before = after
        capture_region(canvas).save(args.out / "final.png")
    except Aborted as stop:
        print(stop, flush=True)
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
