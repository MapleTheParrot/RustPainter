"""Find the selected-color block beside the hue bar and time its updates.

Selects a vivid locator color, finds the block with ``app.color_swatch``,
then cycles through colors reading the block at a ladder of delays after
the saturation / value click, so the painter knows how soon after its
clicks the block tells the truth.  Also reads it between the hue click and
the S/V click, which is what a swallowed S/V click would leave behind.

    python tools/swatch_probe.py --profile xxl

Needs the painting UI open; Escape, a moved mouse, or Rust losing focus
stop it.  Nothing is painted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.color_mapping import map_rgb_to_picker  # noqa: E402
from app.color_swatch import LOCATOR_COLOR, locate_swatch, read_swatch  # noqa: E402
from app.input_controller import create_system_input_controller  # noqa: E402
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
    select_color,
)

COLORS = [
    (230, 40, 40),
    (40, 90, 230),
    (240, 200, 30),
    (20, 20, 20),
    (120, 200, 120),
    (200, 120, 180),
    (90, 60, 30),
    (250, 250, 250),
]
LADDER_MS = (0, 20, 40, 70, 100, 150, 250)


def picker_points(profile, color):
    coordinates = map_rgb_to_picker(
        color,
        profile.hue_bar,
        profile.color_box,
        hue_direction="bottom_to_top",
        saturation_direction="left_low",
        value_direction="top_bright",
    )
    out = []
    for point, rect in (
        (coordinates.hue, profile.hue_bar),
        (coordinates.saturation_value, profile.color_box),
    ):
        x = min(max(point[0], rect.left + rect.width * 0.02), rect.left + rect.width * 0.98)
        y = min(max(point[1], rect.top + rect.height * 0.02), rect.top + rect.height * 0.98)
        out.append((round(x), round(y)))
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--countdown", type=int, default=3)
    args = parser.parse_args()

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
    hue_bar = ScreenRect(
        profile.hue_bar.left, profile.hue_bar.top, profile.hue_bar.width, profile.hue_bar.height
    )
    park = (
        int(profile.color_box.left + profile.color_box.width / 2),
        int(profile.color_box.top + profile.color_box.height / 2),
    )

    overlays = hide_overlays()
    _focus_rust()
    controller = create_system_input_controller()
    guard = Guard(controller, budget_seconds=300)
    time.sleep(FOCUS_SETTLE_SECONDS)
    require_painting_ui(profile)
    countdown(args.countdown, "reading the color swatch")
    results = {"swatch": None, "colors": []}
    try:
        select_color(guard, profile, LOCATOR_COLOR)
        guard.park(park, settle=0.3)
        swatch = locate_swatch(capture_region, hue_bar, LOCATOR_COLOR)
        print(f"swatch: {swatch}", flush=True)
        if swatch is None:
            raise Aborted("no swatch found")
        results["swatch"] = [swatch.left, swatch.top, swatch.width, swatch.height]
        reading = read_swatch(capture_region, swatch)
        print(f"  locator reads {reading.hex} spread {reading.spread:.1f}", flush=True)
        for color in COLORS:
            hue_point, sv_point = picker_points(profile, color)
            before = read_swatch(capture_region, swatch)
            guard.check()
            controller.click(*hue_point, hold_seconds=0.09)
            guard.commanded(*hue_point)
            time.sleep(0.09)
            after_hue = read_swatch(capture_region, swatch)
            guard.check()
            controller.click(*sv_point, hold_seconds=0.09)
            guard.commanded(*sv_point)
            released = time.monotonic()
            ladder = []
            for ms in LADDER_MS:
                while (time.monotonic() - released) * 1000 < ms:
                    time.sleep(0.001)
                r = read_swatch(capture_region, swatch)
                ladder.append((ms, r.hex, round(r.distance_to(color), 1), round(r.spread, 1)))
            row = {
                "asked": "#%02X%02X%02X" % color,
                "before": before.hex,
                "after_hue": after_hue.hex,
                "after_hue_distance": round(after_hue.distance_to(color), 1),
                "ladder": ladder,
            }
            results["colors"].append(row)
            print(
                f"  {row['asked']}: before {before.hex}, after hue {after_hue.hex} "
                f"(d={row['after_hue_distance']}), after S/V: "
                + " ".join(f"{ms}ms={h}(d={d},s={s})" for ms, h, d, s in ladder),
                flush=True,
            )
    except Aborted as stop:
        print(stop, flush=True)
    finally:
        try:
            controller.release_all()
        except Exception:
            pass
        restore_overlays(overlays)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    with _high_resolution_timer():
        raise SystemExit(main())
