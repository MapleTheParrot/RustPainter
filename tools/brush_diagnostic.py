"""Paint a ladder of test strokes in Rust and report what each Size number covers.

Automatic brush sizing has now been wrong twice in different ways, and both
times the argument was conducted over a log line and a screenshot.  This script
exists so the next round is conducted over pixels: it paints one stroke per Size
value down the sign, captures the canvas, and writes both the raw images and a
per-stroke edge profile to an output directory.

It uses the application's own calibration, input, and capture code, so whatever
it measures is what the painter would measure.

    python tools/brush_diagnostic.py --out DIRECTORY [--profile NAME]

Rust must be focused with the sign's painting interface open, exactly as if
painting were about to start.  The strokes are real paint: clear the sign
afterwards.  Escape aborts between steps.
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
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.color_mapping import map_rgb_to_picker  # noqa: E402
from app.input_controller import create_system_input_controller  # noqa: E402
from app.models import ScreenRect  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.screen import (  # noqa: E402
    ForegroundRequirement,
    capture_region,
    foreground_window_matches,
)


# The range automatic sizing actually operates in for a detail brush.  Anything
# wider was already measured correctly; the argument is entirely about the
# bottom of the scale.
LADDER = (1, 2, 3, 4, 5, 6, 8, 10)

STROKE_COLOR = (255, 0, 255)


def _data_directory() -> Path:
    override = os.environ.get("RUST_PAINTER_DATA_DIR")
    if override:
        return Path(override).expanduser()
    local = os.environ.get("LOCALAPPDATA")
    return Path(local) / "RustPainter" if local else Path.cwd() / "data"


def _aborted() -> bool:
    return bool(ctypes.windll.user32.GetAsyncKeyState(0x1B) & 0x8000)


RUST = ForegroundRequirement(title_contains="Rust", executable="RustClient.exe")


def _checkpoint() -> None:
    """Stop the moment Escape is pressed or Rust stops being the focused window.

    Every step synthesizes clicks and digits.  Sent anywhere but Rust they are
    at best noise and at worst typing into someone else's application, so this
    fails closed rather than carrying on blind.
    """

    if _aborted():
        raise SystemExit("Aborted with Escape")
    if not foreground_window_matches(RUST):
        raise SystemExit("Rust is no longer the focused window; stopped")


def _rect(value) -> ScreenRect:
    return ScreenRect(value.left, value.top, value.width, value.height)


class Session:
    def __init__(self, profile, controller) -> None:
        self.profile = profile
        self.input = controller
        self.canvas = _rect(profile.canvas)
        self.box = _rect(profile.brush_size_box)
        self.park = (
            int(round(profile.color_box.left + profile.color_box.width / 2)),
            int(round(profile.color_box.top + profile.color_box.height / 2)),
        )

    def select_color(self, rgb) -> None:
        coordinates = map_rgb_to_picker(
            rgb,
            self.profile.hue_bar,
            self.profile.color_box,
            hue_direction="bottom_to_top",
            saturation_direction="left_low",
            value_direction="top_bright",
        )
        for point in (coordinates.hue, coordinates.saturation_value):
            _checkpoint()
            self.input.click(round(point[0]), round(point[1]))
            time.sleep(0.12)

    def type_size(self, size: int) -> None:
        _checkpoint()
        self.input.click(
            self.box.left + self.box.width // 2, self.box.top + self.box.height // 2
        )
        time.sleep(0.12)
        for key in ("BACKSPACE",) * 4 + ("DELETE",) * 4:
            self.input.press_key(key)
        for digit in str(size):
            self.input.press_key(digit)
        self.input.press_key("ENTER")
        time.sleep(0.20)

    def stroke(self, y: int) -> None:
        _checkpoint()
        start = (int(self.canvas.left + self.canvas.width * 0.20), y)
        end = (int(self.canvas.left + self.canvas.width * 0.80), y)
        self.input.drag(start, end, duration_seconds=0.35, step_pixels=3.0)
        time.sleep(0.12)

    def capture(self) -> Image.Image:
        _checkpoint()
        self.input.move_mouse(*self.park)
        time.sleep(0.45)
        return capture_region(self.canvas).convert("RGB")


def _edge_profile(
    before: np.ndarray, after: np.ndarray, top: int, bottom: int, left: int, right: int
) -> np.ndarray:
    """Median colour change per row across the stroke's straight middle."""

    change = np.linalg.norm(
        after[top:bottom, left:right] - before[top:bottom, left:right], axis=2
    )
    return np.median(change, axis=1)


def _width_at(profile: np.ndarray, fraction: float) -> float:
    """Rows whose change clears ``fraction`` of this stroke's strongest change."""

    peak = float(profile.max())
    if peak <= 0:
        return 0.0
    return float((profile >= peak * fraction).sum())


def run(output: Path, profile_name: str | None, countdown: int) -> None:
    store = ProfileStore(_data_directory() / "profiles")
    profiles = store.list_profiles()
    if not profiles:
        raise SystemExit("No calibration profiles found")
    if profile_name:
        matches = [p for p in profiles if p.name.lower() == profile_name.lower()]
        if not matches:
            names = ", ".join(p.name for p in profiles)
            raise SystemExit(f"No profile named {profile_name!r}. Available: {names}")
        profile = matches[0]
    else:
        profile = store.get_default() or profiles[0]

    for name in ("canvas", "color_box", "hue_bar", "brush_size_box"):
        if getattr(profile, name, None) is None:
            raise SystemExit(f"Profile {profile.name!r} has no {name} calibration")

    output.mkdir(parents=True, exist_ok=True)
    session = Session(profile, create_system_input_controller())
    canvas = session.canvas
    rows = [
        int(round(canvas.top + canvas.height * (index + 1) / (len(LADDER) + 1)))
        for index in range(len(LADDER))
    ]

    print(f"Profile: {profile.name}")
    print(f"  canvas   {canvas.left},{canvas.top} {canvas.width}x{canvas.height}")
    print(f"  size box {session.box.left},{session.box.top} "
          f"{session.box.width}x{session.box.height}")
    print(f"Painting sizes {', '.join(str(s) for s in LADDER)} down the sign.")
    print("Focus Rust with the sign open. Escape aborts.")
    for remaining in range(countdown, 0, -1):
        print(f"  starting in {remaining}...", flush=True)
        time.sleep(1.0)
        if _aborted():
            raise SystemExit("Aborted with Escape")
    _checkpoint()

    before = session.capture()
    before.save(output / "canvas_before.png")
    session.select_color(STROKE_COLOR)
    for size, y in zip(LADDER, rows):
        print(f"  size {size} at y={y}", flush=True)
        session.type_size(size)
        session.stroke(y)
    after = session.capture()
    after.save(output / "canvas_after.png")

    before_pixels = np.asarray(before, dtype=np.float32)
    after_pixels = np.asarray(after, dtype=np.float32)
    left = int(canvas.width * 0.30)
    right = int(canvas.width * 0.70)
    half = int(canvas.height / (len(LADDER) + 1) / 2)

    report: list[dict] = []
    annotated = after.copy()
    marker = ImageDraw.Draw(annotated)
    print()
    print(f"{'size':>4} {'w@25%':>7} {'w@50%':>7} {'w@75%':>7} {'w@90%':>7} {'peak':>7}")
    for size, y in zip(LADDER, rows):
        local_y = y - canvas.top
        top = max(0, local_y - half)
        bottom = min(canvas.height, local_y + half)
        profile_rows = _edge_profile(
            before_pixels, after_pixels, top, bottom, left, right
        )
        widths = {
            f"width_at_{int(f * 100)}": _width_at(profile_rows, f)
            for f in (0.25, 0.50, 0.75, 0.90)
        }
        peak = float(profile_rows.max())
        report.append({"size": size, "y": y, "peak_change": peak, **widths,
                       "row_profile": [round(float(v), 1) for v in profile_rows]})
        print(
            f"{size:>4} {widths['width_at_25']:>7.1f} {widths['width_at_50']:>7.1f} "
            f"{widths['width_at_75']:>7.1f} {widths['width_at_90']:>7.1f} {peak:>7.1f}"
        )
        marker.rectangle((left, top, right, bottom - 1), outline=(0, 255, 0))
        # A 6x vertical blow-up of one stroke, so the rim is visible by eye.
        crop = after.crop((left, top, left + 240, bottom))
        crop.resize((crop.width, crop.height * 6), Image.NEAREST).save(
            output / f"stroke_size_{size:02d}.png"
        )
    annotated.save(output / "canvas_annotated.png")

    cell_note = {
        "canvas": [canvas.left, canvas.top, canvas.width, canvas.height],
        "ladder": list(LADDER),
        "strokes": report,
    }
    (output / "report.json").write_text(json.dumps(cell_note, indent=2), encoding="utf-8")
    print()
    print(f"Wrote images and report.json to {output}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--profile", default=None)
    parser.add_argument("--countdown", type=int, default=6)
    arguments = parser.parse_args()
    run(arguments.out, arguments.profile, arguments.countdown)


if __name__ == "__main__":
    main()
