"""Round 3: does a committed hex value change what actually PAINTS?

Types a hex color, paints one dot, types another, paints a second dot,
captures both, then clears the sign.  The dots' colors are the verdict:
if they follow the typed hexes the field works and only the side swatch
lags; if they stay the picker color the field is display-only.

    python tools/hex_field_probe3.py --out diagnostic/hexprobe3
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402

from app.color_swatch import LOCATOR_COLOR, locate_swatch, read_swatch  # noqa: E402
from app.input_controller import create_system_input_controller  # noqa: E402
from app.models import ScreenRect  # noqa: E402
from app.painter import _high_resolution_timer  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.screen import capture_region  # noqa: E402
from tools._safety import Aborted, Guard, countdown  # noqa: E402
from tools.decimal_probe import _data_directory, _focus_rust  # noqa: E402
from tools.hex_field_probe2 import clear_and_type  # noqa: E402
from tools.press_timing_probe import (  # noqa: E402
    FOCUS_SETTLE_SECONDS,
    hide_overlays,
    require_painting_ui,
    restore_overlays,
    select_color,
)


def dot_color(canvas_img, canvas, point, radius=4):
    a = np.asarray(canvas_img.convert("RGB"), dtype=float)
    x = point[0] - canvas.left
    y = point[1] - canvas.top
    patch = a[y - radius:y + radius + 1, x - radius:x + radius + 1].reshape(-1, 3)
    # The dot is the pixels farthest from the blank-canvas beige.
    blank = np.array([188.0, 174.0, 159.0])
    d = np.abs(patch - blank).max(axis=1)
    hot = patch[d > 60]
    return hot.mean(axis=0).round(0).tolist() if len(hot) else None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
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
    canvas = ScreenRect(
        profile.canvas.left, profile.canvas.top, profile.canvas.width, profile.canvas.height
    )
    clear = profile.clear_button

    overlays = hide_overlays()
    _focus_rust()
    controller = create_system_input_controller()
    guard = Guard(controller, budget_seconds=180)
    time.sleep(FOCUS_SETTLE_SECONDS)
    require_painting_ui(profile)
    countdown(2, "hex paint trial")
    results: dict = {"dots": []}
    try:
        select_color(guard, profile, LOCATOR_COLOR)
        swatch = locate_swatch(capture_region, hue_bar, LOCATOR_COLOR)
        if swatch is None:
            raise Aborted("no swatch found")

        trials = [
            ("#C71B1C", (199, 27, 28), (canvas.left + 400, canvas.top + 400)),
            ("#1E90FF", (30, 144, 255), (canvas.left + 460, canvas.top + 400)),
        ]
        for text, rgb, point in trials:
            clear_and_type(guard, controller, field, text)
            controller.press_key("ENTER", hold_seconds=0.03)
            time.sleep(0.3)
            guard.click(*point, settle=0.35)
            reading = read_swatch(capture_region, swatch)
            results["dots"].append({"typed": text, "point": list(point),
                                    "swatch_after": reading.hex})
            print(f"  typed {text}, dotted at {point}, swatch now {reading.hex}", flush=True)

        time.sleep(0.4)
        shot = capture_region(canvas)
        shot.save(args.out / "canvas_dots.png")
        for row, (text, rgb, point) in zip(results["dots"], trials):
            measured = dot_color(shot, canvas, point)
            row["dot_rgb"] = measured
            print(f"  {text}: dot painted {measured} (asked {list(rgb)})", flush=True)

        # Clear the sign.
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
