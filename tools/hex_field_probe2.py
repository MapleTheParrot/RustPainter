"""Round 2: can a typed hex color actually COMMIT in Rust's picker?

Round 1 established the field is editable but a bare six-digit value never
took.  The displayed content starts with '#', so this round types the '#'
too (Shift+3 via raw SendInput) and tries the plausible commit gestures.

    python tools/hex_field_probe2.py --out diagnostic/hexprobe2
"""

from __future__ import annotations

import argparse
import ctypes
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

VK_SHIFT = 0x10
KEYEVENTF_KEYUP = 0x0002
KEYEVENTF_SCANCODE = 0x0008


def _scan_event(vk: int, up: bool) -> None:
    scan = ctypes.windll.user32.MapVirtualKeyW(vk, 0)
    flags = (KEYEVENTF_KEYUP if up else 0) | (KEYEVENTF_SCANCODE if scan else 0)
    ctypes.windll.user32.keybd_event(vk, scan, flags, 0)


def type_hash() -> None:
    """Shift+3 = '#' on a US layout."""

    _scan_event(VK_SHIFT, up=False)
    time.sleep(0.02)
    _scan_event(ord("3"), up=False)
    time.sleep(0.03)
    _scan_event(ord("3"), up=True)
    time.sleep(0.02)
    _scan_event(VK_SHIFT, up=True)
    time.sleep(0.03)


def clear_and_type(guard: Guard, controller, field: ScreenRect, text: str) -> None:
    center = (field.left + field.width // 2, field.top + field.height // 2)
    guard.click(*center, settle=0.15)
    for key in ("BACKSPACE",) * 9 + ("DELETE",) * 9:
        guard.check()
        controller.press_key(key, hold_seconds=0.03)
        time.sleep(0.02)
    for char in text:
        guard.check()
        if char == "#":
            type_hash()
        else:
            controller.press_key(char, hold_seconds=0.03)
        time.sleep(0.02)
    time.sleep(0.1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--countdown", type=int, default=2)
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
    field = ScreenRect(hue_bar.left + hue_bar.width + 2, hue_bar.top + 258, 128, 42)

    overlays = hide_overlays()
    _focus_rust()
    controller = create_system_input_controller()
    guard = Guard(controller, budget_seconds=240)
    time.sleep(FOCUS_SETTLE_SECONDS)
    require_painting_ui(profile)
    countdown(args.countdown, "hex commit trials")
    results: dict = {"trials": []}
    try:
        select_color(guard, profile, LOCATOR_COLOR)
        swatch = locate_swatch(capture_region, hue_bar, LOCATOR_COLOR)
        if swatch is None:
            raise Aborted("no swatch found")
        defocus = (swatch.left + swatch.width // 2, swatch.top + swatch.height // 2)

        trials = [
            ("hash_enter", "#C71B1C", "enter"),
            ("hash_click_away", "#1E90FF", "click"),
            ("bare_click_away", "2ECC40", "click"),
            ("hash_tab", "#FF00FF", "tab"),
        ]
        for name, text, commit in trials:
            clear_and_type(guard, controller, field, text)
            capture_region(field).save(args.out / f"field_{name}_typed.png")
            if commit == "enter":
                controller.press_key("ENTER", hold_seconds=0.03)
            elif commit == "tab":
                controller.press_key(0x09, hold_seconds=0.03)  # VK_TAB; not in the name map
            else:
                guard.click(*defocus, settle=0.1)
            time.sleep(0.3)
            capture_region(field).save(args.out / f"field_{name}_committed.png")
            reading = read_swatch(capture_region, swatch)
            want = tuple(int(text.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4))
            row = {"name": name, "typed": text, "commit": commit,
                   "swatch": reading.hex, "distance": round(reading.distance_to(want), 1)}
            results["trials"].append(row)
            print(f"  {name}: typed {text}, commit {commit} -> swatch {reading.hex} "
                  f"(d={row['distance']})", flush=True)
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
