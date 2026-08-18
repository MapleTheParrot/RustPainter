"""Pair every painted band with a photograph of the Size value that painted it.

The first probe left two open questions: three strokes all committed as
"1.00" measured 8, 8, and 3 px, and none of the color clicks registered.
Rust was running at 15 FPS, so a 10 ms click can start and end inside one
frame.  This probe holds every click across a frame, verifies the color
actually changed, and captures the Size field immediately before each stroke
so no band's size is ever inferred.

Also answers: does the field hold 1.50, or is it integers with a floor of 1?
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

# Rust ran at 15 FPS (67 ms/frame); a press-release inside one frame can be
# sampled as nothing.  Held this long, every click spans at least one frame.
CLICK_HOLD = 0.09

LADDER = ("1", "1.50", "2", "1", "3", "1")


def _data_directory() -> Path:
    override = os.environ.get("RUST_PAINTER_DATA_DIR")
    if override:
        return Path(override).expanduser()
    local = os.environ.get("LOCALAPPDATA")
    return Path(local) / "RustPainter" if local else Path.cwd() / "data"


def _focus_rust(timeout_seconds: float = 20.0) -> None:
    user32 = ctypes.windll.user32
    hwnd = user32.FindWindowW(None, "Rust")
    if hwnd:
        user32.ShowWindow(hwnd, 9)
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
    guard = Guard(create_system_input_controller(), budget_seconds=150)

    def click(x: float, y: float, settle: float = 0.15) -> None:
        guard.check()
        guard.input.click(round(x), round(y), hold_seconds=CLICK_HOLD)
        guard.commanded(x, y)
        time.sleep(settle)

    def type_size(text: str) -> None:
        click(box.left + box.width / 2, box.top + box.height / 2)
        for key in ("BACKSPACE",) * 5 + ("DELETE",) * 5:
            guard.check()
            guard.input.press_key(key, hold_seconds=0.03)
            time.sleep(0.03)
        for char in text:
            guard.check()
            guard.input.press_key(
                VK_OEM_PERIOD if char == "." else char, hold_seconds=0.03
            )
            time.sleep(0.03)
        guard.input.press_key("ENTER", hold_seconds=0.03)
        time.sleep(0.25)

    # ---- select magenta with frame-safe clicks, and verify it took
    coordinates = map_rgb_to_picker(
        (255, 0, 255),
        profile.hue_bar,
        profile.color_box,
        hue_direction="bottom_to_top",
        saturation_direction="left_low",
        value_direction="top_bright",
    )
    click(*coordinates.hue, settle=0.25)
    click(*coordinates.saturation_value, settle=0.25)
    guard.park(park)
    picker = capture_region(profile.color_box).convert("RGB")
    picker.save(output / "picker_after_selection.png")

    before = capture_region(canvas).convert("RGB")
    before.save(output / "canvas_before.png")

    rows = [
        int(round(canvas.top + canvas.height * (index + 1) / (len(LADDER) + 1)))
        for index in range(len(LADDER))
    ]
    box_shots: list[Image.Image] = []
    for index, (text, y) in enumerate(zip(LADDER, rows)):
        print(f"  [{index}] type {text}, stroke at y={y}", flush=True)
        type_size(text)
        guard.park(park, settle=0.3)
        shot = capture_region(box).convert("RGB")
        shot.save(output / f"box_{index}_{text.replace('.', '_')}.png")
        box_shots.append(shot)
        start = (int(canvas.left + canvas.width * 0.20), y)
        end = (int(canvas.left + canvas.width * 0.80), y)
        guard.drag(start, end, duration_seconds=0.6)

    guard.park(park)
    after = capture_region(canvas).convert("RGB")
    after.save(output / "canvas_after.png")

    # ---- analysis
    before_px = np.asarray(before, dtype=np.float32)
    after_px = np.asarray(after, dtype=np.float32)
    left = int(canvas.width * 0.30)
    right = int(canvas.width * 0.70)
    half = int(canvas.height / (len(LADDER) + 1) / 2)

    report = []
    print()
    print(f"{'idx':>3} {'typed':>6} {'w@25%':>7} {'w@50%':>7} {'w@75%':>7} {'peak':>7}")
    for index, (text, y) in enumerate(zip(LADDER, rows)):
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
        report.append({"index": index, "typed": text, "peak": peak,
                       **{str(k): v for k, v in widths.items()}})
        print(
            f"{index:>3} {text:>6} {widths[0.25]:>7.1f} {widths[0.50]:>7.1f} "
            f"{widths[0.75]:>7.1f} {peak:>7.1f}"
        )

    # one sheet of all box captures, in order
    scale = 3
    w = max(s.width for s in box_shots) * scale
    h = sum(s.height * scale + 4 for s in box_shots)
    sheet = Image.new("RGB", (w, h), (30, 30, 30))
    offset = 0
    for shot in box_shots:
        sheet.paste(shot.resize((shot.width * scale, shot.height * scale), Image.LANCZOS), (0, offset))
        offset += shot.height * scale + 4
    sheet.save(output / "boxes_in_order.png")

    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nWrote results to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    run(parser.parse_args().out)


if __name__ == "__main__":
    main()
