"""Does Rust's SIZE field accept decimals, and do they change the painted band?

The field displays two decimal places (1.00), while automatic sizing types
integers clamped to a minimum of 1.  If Rust honours fractional sizes, the
"strokes too small" refusals at high painting resolution are this program's
fault, not the game's.  Two questions, answered with pixels:

1. Type ``0.50`` - does the field display it?  (captured, not assumed)
2. Paint strokes at 0.25 / 0.50 / 1 / 2 / 4 - do the fractional bands come
   out narrower than the size-1 band?

    python tools/decimal_probe.py --out DIRECTORY
"""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.color_mapping import map_rgb_to_picker  # noqa: E402
from app.input_controller import create_system_input_controller  # noqa: E402
from app.models import ScreenRect  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.screen import capture_region, foreground_window_matches  # noqa: E402
from tools._safety import Aborted, Guard, RUST  # noqa: E402


VK_OEM_PERIOD = 0xBE

LADDER = ("0.25", "0.50", "1", "2", "4")


def _data_directory() -> Path:
    override = os.environ.get("RUST_PAINTER_DATA_DIR")
    if override:
        return Path(override).expanduser()
    local = os.environ.get("LOCALAPPDATA")
    return Path(local) / "RustPainter" if local else Path.cwd() / "data"


def _focus_rust(timeout_seconds: float = 20.0) -> None:
    """Bring the Rust window forward, or wait for a person to do it."""

    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, "Rust")
    if hwnd:
        user32.ShowWindow(hwnd, 9)  # SW_RESTORE
        user32.SetForegroundWindow(hwnd)
        time.sleep(0.5)
    deadline = time.monotonic() + timeout_seconds
    while not foreground_window_matches(RUST):
        if time.monotonic() > deadline:
            raise Aborted("Rust never became the focused window")
        print("  waiting for Rust to be focused...", flush=True)
        time.sleep(1.0)


def _rect(value) -> ScreenRect:
    return ScreenRect(value.left, value.top, value.width, value.height)


def _type_size(guard: Guard, box: ScreenRect, text: str) -> None:
    guard.click(box.left + box.width / 2, box.top + box.height / 2)
    for key in ("BACKSPACE",) * 5 + ("DELETE",) * 5:
        guard.press(key)
    for char in text:
        guard.press(VK_OEM_PERIOD if char == "." else char)
    guard.press("ENTER")
    time.sleep(0.2)


def run(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    store = ProfileStore(_data_directory() / "profiles")
    profile = store.get_default() or store.list_profiles()[0]
    canvas = _rect(profile.canvas)
    box = _rect(profile.brush_size_box)
    park = (
        int(profile.color_box.left + profile.color_box.width / 2),
        int(profile.color_box.top + profile.color_box.height / 2),
    )

    _focus_rust()
    guard = Guard(create_system_input_controller(), budget_seconds=90)

    # ---- 1. type each value and photograph the field itself
    for text in LADDER:
        _type_size(guard, box, text)
        guard.park(park, settle=0.25)
        capture_region(box).convert("RGB").save(
            output / f"sizebox_{text.replace('.', '_')}.png"
        )

    # ---- 2. paint the ladder and measure each band
    coordinates = map_rgb_to_picker(
        (255, 0, 255),
        profile.hue_bar,
        profile.color_box,
        hue_direction="bottom_to_top",
        saturation_direction="left_low",
        value_direction="top_bright",
    )
    guard.click(*coordinates.hue)
    guard.click(*coordinates.saturation_value)

    guard.park(park)
    before = capture_region(canvas).convert("RGB")
    before.save(output / "canvas_before.png")

    rows = [
        int(round(canvas.top + canvas.height * (index + 1) / (len(LADDER) + 1)))
        for index in range(len(LADDER))
    ]
    for text, y in zip(LADDER, rows):
        print(f"  size {text} at y={y}", flush=True)
        _type_size(guard, box, text)
        start = (int(canvas.left + canvas.width * 0.20), y)
        end = (int(canvas.left + canvas.width * 0.80), y)
        guard.drag(start, end)

    guard.park(park)
    after = capture_region(canvas).convert("RGB")
    after.save(output / "canvas_after.png")

    # ---- analysis: median per-row change across the stroke middle
    before_px = np.asarray(before, dtype=np.float32)
    after_px = np.asarray(after, dtype=np.float32)
    left = int(canvas.width * 0.30)
    right = int(canvas.width * 0.70)
    half = int(canvas.height / (len(LADDER) + 1) / 2)

    report = []
    print()
    print(f"{'size':>6} {'w@25%':>7} {'w@50%':>7} {'w@75%':>7} {'peak':>7}")
    for text, y in zip(LADDER, rows):
        local = y - canvas.top
        band = np.median(
            np.linalg.norm(
                after_px[local - half : local + half, left:right]
                - before_px[local - half : local + half, left:right],
                axis=2,
            ),
            axis=1,
        )
        peak = float(band.max())
        widths = {
            f: float((band >= peak * f).sum()) if peak > 0 else 0.0
            for f in (0.25, 0.50, 0.75)
        }
        report.append({"size": text, "peak": peak, **{str(k): v for k, v in widths.items()}})
        print(
            f"{text:>6} {widths[0.25]:>7.1f} {widths[0.50]:>7.1f} "
            f"{widths[0.75]:>7.1f} {peak:>7.1f}"
        )
        crop = after.crop((left, max(0, local - half), left + 240, local + half))
        crop.resize((crop.width, crop.height * 6), Image.NEAREST).save(
            output / f"stroke_{text.replace('.', '_')}.png"
        )

    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote results to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)


if __name__ == "__main__":
    main()
