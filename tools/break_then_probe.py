"""Does the anti-AFK break change how Rust takes presses afterwards?

A nine-hour run lost more presses with every hour - a tenth in its first
hour, two thirds in its fifth - while fresh measurements on the same sign
drop none at any hold.  Eighteen anti-AFK breaks ran through that job, each
a Save, a jump, and the sign reopened with E.  This performs one such break
exactly as the painter does and measures the drop rate at the painter's own
hold before it and after it, then again after several more, so a change the
break leaves behind in the game shows as a rate that climbs with the count.

    python tools/break_then_probe.py --out diagnostic/break1 --breaks 1,3

Needs the profile's painting UI open and the Save button calibrated.  The
sign is saved by each break, as a real job's would be.  Escape, a moved
mouse, or Rust losing focus stop it.

Result on the XXL sign, 2026-08-22: 730/730 dots after one break and after
four; the breaks change nothing.  The lost paint was swallowed picker
clicks (see app/color_swatch.py), not presses.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.input_controller import create_system_input_controller  # noqa: E402
from app.models import ScreenRect  # noqa: E402
from app.painter import _high_resolution_timer  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.screen import capture_region  # noqa: E402
from app.ui_guard import looks_like_hue_bar  # noqa: E402
from tools._safety import Aborted, Guard, countdown  # noqa: E402
from tools.decimal_probe import _data_directory, _focus_rust  # noqa: E402
from tools.press_timing_probe import (  # noqa: E402
    BAND_COLORS,
    FOCUS_SETTLE_SECONDS,
    band_points,
    hide_overlays,
    paint_dot,
    require_painting_ui,
    restore_overlays,
    score_band,
    select_color,
)


def anti_afk_break(guard: Guard, profile) -> None:
    """Save, jump, reopen with E - the painter's own sequence and timings."""

    save = profile.save_button
    guard.check()
    # Save closes the painting UI and Rust takes the cursor back to the
    # middle of the screen.  The painter drops its mouse-movement baseline
    # before the click for exactly this reason, and so must the guard here:
    # nothing below asks it to judge the mouse until the sign is open again.
    guard._commanded = None
    guard.input.click(save.left + save.width / 2, save.top + save.height / 2, hold_seconds=0.09)
    time.sleep(0.5)
    guard.input.press_key("SPACE", hold_seconds=0.1)
    time.sleep(2.0)
    guard.input.press_key("E", hold_seconds=0.1)
    time.sleep(1.0)
    deadline = time.monotonic() + 4.0
    rect = profile.hue_bar
    region = ScreenRect(rect.left, rect.top, rect.width, rect.height)
    while time.monotonic() < deadline:
        if looks_like_hue_bar(capture_region(region)):
            break
        time.sleep(0.5)
    else:
        raise Aborted("the painting UI did not come back after the break")
    guard.commanded(*guard.input.get_cursor_position())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--breaks", type=str, default="1,3", help="breaks before each later band")
    parser.add_argument("--hold", type=float, default=70.0)
    parser.add_argument("--countdown", type=int, default=4)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    stages = [0] + [int(v) for v in args.breaks.split(",") if v.strip()]

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
    if profile.save_button is None:
        raise SystemExit("The profile needs its Save button calibrated")
    canvas = ScreenRect(
        profile.canvas.left, profile.canvas.top, profile.canvas.width, profile.canvas.height
    )
    park = (
        int(profile.color_box.left + profile.color_box.width / 2),
        int(profile.color_box.top + profile.color_box.height / 2),
    )
    bands = [band_points(canvas, i, len(stages), 24.0) for i in range(len(stages))]

    overlays = hide_overlays()
    _focus_rust()
    controller = create_system_input_controller()
    guard = Guard(controller, budget_seconds=900)
    time.sleep(FOCUS_SETTLE_SECONDS)
    require_painting_ui(profile)
    countdown(args.countdown, "measuring across anti-AFK breaks")

    results = []
    try:
        for stage, (breaks, points) in enumerate(zip(stages, bands)):
            for n in range(breaks):
                anti_afk_break(guard, profile)
                print(f"  break {n + 1} of {breaks} done", flush=True)
            select_color(guard, profile, BAND_COLORS[stage % len(BAND_COLORS)])
            guard.park(park, settle=0.6)
            before = np.asarray(capture_region(canvas).convert("RGB"), dtype=np.float32)
            held = [paint_dot(guard, p, args.hold / 1000.0, 0.0) for p in points]
            guard.park(park, settle=0.6)
            after = np.asarray(capture_region(canvas).convert("RGB"), dtype=np.float32)
            found, missing = score_band(before, after, canvas, points, 9)
            total = len(found) + len(missing)
            rate = len(missing) / max(1, total)
            results.append(
                {
                    "breaks_before": breaks,
                    "breaks_total": sum(stages[: stage + 1]),
                    "dots": total,
                    "dropped": len(missing),
                    "drop_rate": round(rate, 4),
                    "actual_hold_ms": round(float(np.median(held)) * 1000, 1),
                }
            )
            print(
                f"  after {sum(stages[: stage + 1])} break(s): {len(found)}/{total} painted, "
                f"{100 * rate:.1f}% dropped",
                flush=True,
            )
            capture_region(canvas).save(args.out / f"stage_{stage}.png")
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
