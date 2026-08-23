"""Watch Rust's own FPS counter while the painter's input pattern runs.

Seven bands of dots at every hold from 8 ms up landed without a single loss
on the largest sign - measured at 59 FPS, on a sign just opened.  The
nine-hour run that lost a third of its presses was photographed at 15 FPS,
hours in.  Every job starts near 60.  So the question is not how long a
press must be held but what drives the game from 59 FPS to 15, and whether
presses start to drop as it does.

This streams presses at the painter's real cadence - a dab, the stroke gap,
the next dab - for a set time, and between them reads the FPS figure Rust
draws in its corner, by template-matching the digits off the screen.  It
reports the FPS trace and, at the end, how many of the dots landed, so a
fall in frame rate can be matched against a rise in dropped presses.

    python tools/fps_under_load.py --out diagnostic/load1 --seconds 120

Needs the profile's painting UI open; Escape, a moved mouse, or Rust losing
focus stop it.  The sign is left covered in dots.

Result on the XXL sign, 2026-08-22: 1,195 presses in two minutes, none
dropped, 56-58 FPS throughout.  The lost paint was swallowed picker clicks
(see app/color_swatch.py), not presses.
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
    require_painting_ui,
    restore_overlays,
    score_band,
    select_color,
)

# Where Rust draws its performance readout, on this screen: the "NN FPS
# (MM.Mms)" line sits in the lower-left corner of the primary monitor.
HUD_RECT = ScreenRect(0, 1330, 400, 60)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--seconds", type=float, default=120.0)
    parser.add_argument("--hold", type=float, default=70.0, help="press hold in ms")
    parser.add_argument("--gap", type=float, default=20.0, help="between presses, ms")
    parser.add_argument("--spacing", type=float, default=14.0)
    parser.add_argument("--countdown", type=int, default=4)
    parser.add_argument("--clear", action="store_true", help="click Rust's clear control first")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

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
    margin = 16.0
    columns = int((canvas.width - 2 * margin) // args.spacing)
    rows = int((canvas.height - 2 * margin) // args.spacing)
    points = [
        (round(canvas.left + margin + c * args.spacing), round(canvas.top + margin + r * args.spacing))
        for r in range(rows)
        for c in range(columns)
    ]
    print(f"profile {profile.name!r}: up to {len(points)} dots for {args.seconds:.0f}s", flush=True)

    overlays = hide_overlays()
    _focus_rust()
    controller = create_system_input_controller()
    guard = Guard(controller, budget_seconds=args.seconds + 120)
    time.sleep(FOCUS_SETTLE_SECONDS)
    require_painting_ui(profile)
    countdown(args.countdown, "streaming presses")
    if args.clear and profile.clear_button is not None:
        clear = profile.clear_button
        guard.click(clear.left + clear.width / 2, clear.top + clear.height / 2, settle=0.9)
        guard.commanded(*controller.get_cursor_position())
        require_painting_ui(profile)
        guard.park(park)
        print("sign cleared", flush=True)

    trace = []
    painted_points = []
    try:
        select_color(guard, profile, (230, 40, 40))
        guard.park(park, settle=0.6)
        before = np.asarray(capture_region(canvas).convert("RGB"), dtype=np.float32)
        hud_before = capture_region(HUD_RECT)
        hud_before.save(args.out / "hud_start.png")
        started = time.monotonic()
        last_hud = started
        index = 0
        while time.monotonic() - started < args.seconds and index < len(points):
            point = points[index]
            guard.check()
            controller.move_mouse(*point)
            guard.commanded(*point)
            time.sleep(0.002)
            controller.mouse_down(MouseButton.LEFT)
            time.sleep(args.hold / 1000.0)
            controller.mouse_up(MouseButton.LEFT)
            painted_points.append(point)
            index += 1
            time.sleep(args.gap / 1000.0)
            now = time.monotonic()
            if now - last_hud >= 5.0:
                last_hud = now
                hud = capture_region(HUD_RECT)
                hud.save(args.out / f"hud_{int(now - started):04d}.png")
                trace.append({"t": round(now - started, 1), "presses": index})
                print(f"  t={now - started:5.0f}s presses={index:5d}", flush=True)
        elapsed = time.monotonic() - started
        guard.park(park, settle=0.7)
        after_image = capture_region(canvas).convert("RGB")
        after = np.asarray(after_image, dtype=np.float32)
        after_image.save(args.out / "sign_after.png")
        found, missing = score_band(before, after, canvas, painted_points, 9)
        total = len(found) + len(missing)
        # Where in the stream the misses were: first third, middle, last.
        thirds = [0, 0, 0]
        for x, y, _moved in missing:
            k = painted_points.index((x, y))
            thirds[min(2, 3 * k // max(1, len(painted_points)))] += 1
        summary = {
            "seconds": round(elapsed, 1),
            "presses": total,
            "painted": len(found),
            "dropped": len(missing),
            "drop_rate": round(len(missing) / max(1, total), 4),
            "dropped_by_third": thirds,
            "ms_per_press": round(1000.0 * elapsed / max(1, total), 1),
            "hold_ms": args.hold,
            "gap_ms": args.gap,
            "trace": trace,
        }
        print(json.dumps({k: v for k, v in summary.items() if k != "trace"}, indent=2), flush=True)
        (args.out / "result.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Aborted as stop:
        print(stop, flush=True)
    finally:
        try:
            controller.release_all()
        except Exception:
            pass
        restore_overlays(overlays)
    return 0


if __name__ == "__main__":
    with _high_resolution_timer():
        raise SystemExit(main())
