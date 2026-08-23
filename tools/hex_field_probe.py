"""Test the picker's hex text field and the hue bar's true edge behavior.

Nothing is painted: only the color panel is exercised.  Answers three
questions the murica post-mortem raised:

1. Does the hex box beside the swatch accept a typed color?  If it does,
   every pick can be exact and the 2% click inset stops mattering.
2. What hue does the bar deliver when clicked at its outermost pixels,
   with no inset at all - how wide is the dead zone really?
3. Does the S/V box's exact corner register (pure white), or is the 2%
   inset there load-bearing?

    python tools/hex_field_probe.py --out diagnostic/hexprobe

Escape aborts; a moved mouse or Rust losing focus stops it.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

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

# Colors the click path cannot reach (the reason this probe exists).
HEX_TRIALS = [
    ("C71B1C", (199, 27, 28)),   # flag red, hue 359.7 - clamped to 351 in the murica run
    ("FFFFFF", (255, 255, 255)), # the unreachable S/V corner
    ("320D12", (50, 13, 18)),    # control: the color already in the field
    ("1E90FF", (30, 144, 255)),  # mid-gamut control
]


def _field_rect(hue_bar) -> ScreenRect:
    """The hex box sits under the swatch strip, right of the hue bar."""

    return ScreenRect(hue_bar.left + hue_bar.width + 2, hue_bar.top + 258, 128, 42)


def _type_hex(guard: Guard, controller, field: ScreenRect, digits: str) -> None:
    center = (field.left + field.width // 2, field.top + field.height // 2)
    guard.click(*center, settle=0.15)
    for key in ("BACKSPACE",) * 8 + ("DELETE",) * 8:
        guard.check()
        controller.press_key(key, hold_seconds=0.03)
        time.sleep(0.02)
    for char in digits:
        guard.check()
        controller.press_key(char, hold_seconds=0.03)
        time.sleep(0.02)
    guard.check()
    controller.press_key("ENTER", hold_seconds=0.03)
    time.sleep(0.25)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--countdown", type=int, default=3)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    store = ProfileStore(_data_directory() / "profiles")
    profiles = store.list_profiles()
    profile = next((p for p in profiles if "xxl" in p.name.lower()), None) or (
        store.get_default() or profiles[0]
    )
    hue_bar = ScreenRect(
        profile.hue_bar.left, profile.hue_bar.top, profile.hue_bar.width, profile.hue_bar.height
    )
    field = _field_rect(hue_bar)
    park = (
        int(profile.color_box.left + profile.color_box.width / 2),
        int(profile.color_box.top + profile.color_box.height / 2),
    )
    results: dict = {"field_rect": [field.left, field.top, field.width, field.height]}

    overlays = hide_overlays()
    _focus_rust()
    controller = create_system_input_controller()
    guard = Guard(controller, budget_seconds=240)
    time.sleep(FOCUS_SETTLE_SECONDS)
    require_painting_ui(profile)
    countdown(args.countdown, "probing the hex field")
    try:
        # Locate the swatch with the app's own locator color first.
        select_color(guard, profile, LOCATOR_COLOR)
        guard.park(park, settle=0.3)
        swatch = locate_swatch(capture_region, hue_bar, LOCATOR_COLOR)
        if swatch is None:
            raise Aborted("no swatch found")
        results["swatch"] = [swatch.left, swatch.top, swatch.width, swatch.height]
        capture_region(field).save(args.out / "field_before.png")

        # --- Focus check: click the field, type ONE character, look for change.
        center = (field.left + field.width // 2, field.top + field.height // 2)
        guard.click(*center, settle=0.15)
        before = capture_region(field)
        controller.press_key("BACKSPACE", hold_seconds=0.03)
        time.sleep(0.05)
        controller.press_key("1", hold_seconds=0.03)
        time.sleep(0.15)
        after = capture_region(field)
        changed = list(before.convert("RGB").getdata()) != list(after.convert("RGB").getdata())
        after.save(args.out / "field_after_probe_key.png")
        results["field_editable"] = changed
        print(f"hex field editable: {changed}", flush=True)
        # Defocus with a harmless click on the S/V box: ESCAPE would close
        # the whole painting UI (learned the hard way).
        guard.click(*park, settle=0.15)

        # --- Hex trials (only meaningful if the field took the keystroke).
        results["hex"] = []
        if changed:
            for digits, rgb in HEX_TRIALS:
                _type_hex(guard, controller, field, digits)
                guard.park(park, settle=0.25)
                reading = read_swatch(capture_region, swatch)
                row = {
                    "typed": digits,
                    "swatch": reading.hex,
                    "distance": round(reading.distance_to(rgb), 1),
                }
                results["hex"].append(row)
                print(f"  typed {digits}: swatch {reading.hex} (d={row['distance']})", flush=True)

        # --- Hue bar edge ladder, no inset: top pixels then bottom pixels.
        results["hue_edges"] = []
        x = hue_bar.left + hue_bar.width // 2
        tops = [hue_bar.top + dy for dy in (0, 1, 2, 3, 4, 5, 6, 8, 10)]
        bottoms = [hue_bar.top + hue_bar.height - 1 - dy for dy in (0, 1, 2, 3, 4, 5, 6, 8, 10)]
        # Alternate a mid-bar reference click between edge clicks so a dropped
        # edge click is distinguishable from a registered one (swatch changes
        # back and forth only when clicks land).
        mid_y = hue_bar.top + hue_bar.height // 2
        for y in tops + bottoms:
            guard.click(x, mid_y, settle=0.12)
            mid_read = read_swatch(capture_region, swatch)
            guard.click(x, y, settle=0.12)
            edge_read = read_swatch(capture_region, swatch)
            registered = edge_read.hex != mid_read.hex
            row = {"y": y, "offset": y - hue_bar.top, "swatch": edge_read.hex,
                   "registered": registered}
            results["hue_edges"].append(row)
            print(f"  hue click y={y} (top+{y - hue_bar.top}): swatch {edge_read.hex}"
                  f" registered={registered}", flush=True)

        # --- S/V corner, no inset: pure white then pure black.
        box = profile.color_box
        results["sv_corners"] = []
        for name, (px, py) in (
            ("white_corner", (box.left, box.top)),
            ("white_corner+1", (box.left + 1, box.top + 1)),
            ("white_corner+2", (box.left + 2, box.top + 2)),
            ("black_corner", (box.left, box.top + box.height - 1)),
        ):
            guard.click(px, py, settle=0.12)
            reading = read_swatch(capture_region, swatch)
            results["sv_corners"].append({"name": name, "swatch": reading.hex})
            print(f"  {name} ({px},{py}): swatch {reading.hex}", flush=True)
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
    print(json.dumps(results, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    with _high_resolution_timer():
        raise SystemExit(main())
