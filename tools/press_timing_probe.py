"""Measure how long a press has to be held before Rust stops dropping it.

Every timing constant in this application is inferred.  The frame floor was
reasoned from a 15 FPS reading of the paint UI, the long-drag cap from how two
speed presets looked, the stroke gap from nothing much - and a nine-hour sign
then came back with a third of its presses never painted, because the game's
frames on a large sign are longer than the hold that was reasoned for a small
one.  This tool measures the thing itself.

It paints bands of well-separated dots, one band per candidate hold, and counts
how many of them the sign actually took.  That gives the drop rate as a
function of the hold on *this* sign, at *this* resolution, with *this*
machine's frame rate - which is the number the painter's press hold, its
starting point and its cap should all be set from.

    python tools/press_timing_probe.py --out diagnostic/press1
    python tools/press_timing_probe.py --out diagnostic/press2 --holds 40,70,110,160
    python tools/press_timing_probe.py --out diagnostic/press3 --length 6

The sign is cleared before and left painted afterwards; clear it by hand or
run a real job over it.  Escape aborts, as does moving the mouse or letting
Rust lose focus.  Nothing is typed into Rust's Size field: the brush stays
whatever it is set to, which is all a dot needs.

What comes out, per hold: how many dots of that band appeared, and the frame
time implied by the loss (a press of H ms that is dropped a fraction D of the
time has been landing in frames of about H / (1 - D) ms).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.color_mapping import map_rgb_to_picker  # noqa: E402
from app.input_controller import MouseButton, create_system_input_controller  # noqa: E402
from app.models import ScreenRect  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.screen import capture_region  # noqa: E402
from tools._safety import Aborted, Guard, countdown  # noqa: E402
from tools.decimal_probe import _data_directory, _focus_rust  # noqa: E402


# Holds worth measuring, in milliseconds: under the current floor, at it, and
# up through the range the game's own frames have been suspected of reaching.
DEFAULT_HOLDS = (40, 55, 70, 90, 110, 140, 180)

# A dot is counted as painted when its patch of the sign moved this far in
# sRGB from the same patch before the band went down.  Well above the sign's
# grain (a bare artist canvas reads about 6) and far below a real dot.
APPEARED_DELTA = 30.0

# Dots are spread this many screen pixels apart so neighbouring ones cannot be
# confused even with a brush a few texels wide and an aim half a texel off.
DOT_SPACING_PIXELS = 24.0

# Read a square this wide around each dot's commanded point, so a dot that
# landed a texel or two off is still its own dot and not a miss.
DOT_PATCH_PIXELS = 9

# Between the last dot of a band and the capture that scores it: the game has
# to present the frame carrying it, which at its slowest is a quarter second.
BAND_SETTLE_SECONDS = 0.6

# One saturated colour per band, so a band painted over an earlier one is
# still told apart by eye in the saved captures.
BAND_COLORS = (
    (230, 40, 40),
    (40, 90, 230),
    (40, 190, 70),
    (240, 190, 30),
    (210, 50, 200),
    (30, 200, 210),
    (250, 130, 40),
    (140, 70, 230),
)


def select_color(guard: Guard, profile, color) -> None:
    """Click the picker for one colour, holding each click across a frame."""

    directions = profile.picker_directions
    coordinates = map_rgb_to_picker(
        color,
        profile.hue_bar,
        profile.color_box,
        hue_direction=getattr(directions, "hue", "bottom_to_top"),
        saturation_direction=getattr(directions, "saturation", "left_low"),
        value_direction=getattr(directions, "value", "top_bright"),
    )
    for point, rect in (
        (coordinates.hue, profile.hue_bar),
        (coordinates.saturation_value, profile.color_box),
    ):
        # The outermost pixels of each widget ignore clicks; the painter pulls
        # every picker click 2% inward and so does this.
        x = min(
            max(point[0], rect.left + rect.width * 0.02),
            rect.left + rect.width * 0.98,
        )
        y = min(
            max(point[1], rect.top + rect.height * 0.02),
            rect.top + rect.height * 0.98,
        )
        guard.check()
        guard.input.click(round(x), round(y), hold_seconds=0.09)
        guard.commanded(x, y)
        time.sleep(0.09)


def band_points(canvas: ScreenRect, band: int, bands: int, spacing: float) -> list:
    """The commanded points of one band's dots, inset from the sign's edges."""

    margin = max(spacing, 12.0)
    top = canvas.top + margin + band * (canvas.height - 2 * margin) / bands
    bottom = canvas.top + margin + (band + 1) * (canvas.height - 2 * margin) / bands
    rows = max(1, int((bottom - top - spacing) // spacing))
    columns = max(1, int((canvas.width - 2 * margin) // spacing))
    return [
        (
            round(canvas.left + margin + column * spacing),
            round(top + spacing / 2 + row * spacing),
        )
        for row in range(rows)
        for column in range(columns)
    ]


def paint_dot(guard: Guard, point, hold_seconds: float, length_pixels: float) -> None:
    """One press at ``point``, held ``hold_seconds``; a short drag if asked."""

    guard.check()
    guard.input.move_mouse(*point)
    guard.commanded(*point)
    time.sleep(0.004)
    guard.input.mouse_down(MouseButton.LEFT)
    pressed_at = time.monotonic()
    try:
        if length_pixels > 0:
            steps = max(1, int(length_pixels))
            for step in range(1, steps + 1):
                guard.input.move_mouse(point[0] + step * length_pixels / steps, point[1])
            guard.commanded(point[0] + length_pixels, point[1])
        remaining = hold_seconds - (time.monotonic() - pressed_at)
        if remaining > 0:
            time.sleep(remaining)
    finally:
        guard.input.mouse_up(MouseButton.LEFT)
    time.sleep(0.02)


def score_band(before: np.ndarray, after: np.ndarray, canvas: ScreenRect, points, patch: int):
    """How many of a band's dots changed the sign, and where the misses were."""

    half = patch // 2
    found = []
    missing = []
    for x, y in points:
        cx, cy = int(x - canvas.left), int(y - canvas.top)
        window = (
            slice(max(0, cy - half), min(after.shape[0], cy + half + 1)),
            slice(max(0, cx - half), min(after.shape[1], cx + half + 1)),
        )
        if before[window].size == 0:
            continue
        moved = np.linalg.norm(after[window] - before[window], axis=2).max()
        (found if moved >= APPEARED_DELTA else missing).append((x, y, float(moved)))
    return found, missing


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--holds",
        type=str,
        default=",".join(str(h) for h in DEFAULT_HOLDS),
        help="press holds to measure, in milliseconds",
    )
    parser.add_argument(
        "--length",
        type=float,
        default=0.0,
        help="drag this many screen pixels during the press (0 = a dab)",
    )
    parser.add_argument("--spacing", type=float, default=DOT_SPACING_PIXELS)
    parser.add_argument("--countdown", type=int, default=4)
    parser.add_argument("--budget", type=float, default=1200.0)
    parser.add_argument(
        "--no-clear", action="store_true", help="paint over the sign as it is"
    )
    args = parser.parse_args()
    holds = [float(value) / 1000.0 for value in args.holds.split(",") if value.strip()]
    if not holds:
        parser.error("--holds needs at least one value")
    args.out.mkdir(parents=True, exist_ok=True)

    store = ProfileStore(_data_directory() / "profiles")
    profile = store.get_default() or store.list_profiles()[0]
    if profile.canvas is None or profile.color_box is None or profile.hue_bar is None:
        raise SystemExit("The profile needs its canvas, colour box and hue bar calibrated")
    canvas = ScreenRect(
        profile.canvas.left,
        profile.canvas.top,
        profile.canvas.width,
        profile.canvas.height,
    )
    park = (
        int(profile.color_box.left + profile.color_box.width / 2),
        int(profile.color_box.top + profile.color_box.height / 2),
    )
    bands = [band_points(canvas, index, len(holds), args.spacing) for index in range(len(holds))]
    print(
        f"profile {profile.name!r}: {canvas.width}x{canvas.height} canvas, "
        f"{len(holds)} bands of {len(bands[0])} dots "
        f"({'dabs' if args.length <= 0 else f'{args.length:.0f} px drags'})",
        flush=True,
    )

    _focus_rust()
    controller = create_system_input_controller()
    guard = Guard(controller, budget_seconds=args.budget)
    countdown(args.countdown, "measuring the press hold")

    if not args.no_clear and profile.clear_button is not None:
        clear = profile.clear_button
        guard.click(clear.left + clear.width / 2, clear.top + clear.height / 2, settle=0.9)
        guard.park(park)
        print("sign cleared", flush=True)

    results = []
    try:
        for index, hold in enumerate(holds):
            points = bands[index]
            color = BAND_COLORS[index % len(BAND_COLORS)]
            select_color(guard, profile, color)
            guard.park(park, settle=BAND_SETTLE_SECONDS)
            before = np.asarray(capture_region(canvas).convert("RGB"), dtype=np.float32)
            started = time.monotonic()
            for point in points:
                paint_dot(guard, point, hold, args.length)
            elapsed = time.monotonic() - started
            guard.park(park, settle=BAND_SETTLE_SECONDS)
            after_image = capture_region(canvas).convert("RGB")
            after = np.asarray(after_image, dtype=np.float32)
            found, missing = score_band(before, after, canvas, points, DOT_PATCH_PIXELS)
            total = len(found) + len(missing)
            rate = len(missing) / total if total else 0.0
            frame = hold * 1000.0 / (1.0 - rate) if rate < 1.0 else float("inf")
            results.append(
                {
                    "hold_ms": round(hold * 1000.0, 1),
                    "dots": total,
                    "painted": len(found),
                    "dropped": len(missing),
                    "drop_rate": round(rate, 4),
                    "implied_frame_ms": round(frame, 1) if np.isfinite(frame) else None,
                    "seconds_per_press": round(elapsed / max(1, len(points)), 4),
                    "color": list(color),
                }
            )
            print(
                f"  hold {hold * 1000:5.0f} ms: {len(found):4d}/{total:4d} painted, "
                f"{100 * rate:5.1f}% dropped"
                + (f", frames ~{frame:.0f} ms" if np.isfinite(frame) and rate > 0 else "")
                + f"  ({elapsed / max(1, len(points)) * 1000:.0f} ms per press)",
                flush=True,
            )
            after_image.save(args.out / f"band_{index}_{int(hold * 1000)}ms.png")
    except Aborted as stop:
        print(stop, flush=True)
    finally:
        try:
            controller.release_all()
        except Exception:
            pass

    document = {
        "profile": profile.name,
        "canvas": {
            "left": canvas.left,
            "top": canvas.top,
            "width": canvas.width,
            "height": canvas.height,
        },
        "dragPixels": args.length,
        "spacingPixels": args.spacing,
        "bands": results,
    }
    clean = [row for row in results if row["drop_rate"] <= 0.01]
    if clean:
        document["cleanestHoldMs"] = min(row["hold_ms"] for row in clean)
        print(
            f"\nThe shortest hold this sign did not drop: "
            f"{document['cleanestHoldMs']:.0f} ms",
            flush=True,
        )
    else:
        print("\nEvery hold measured lost dots; raise the range with --holds", flush=True)
    frames = [row["implied_frame_ms"] for row in results if row["implied_frame_ms"]]
    if frames:
        document["impliedFrameMsMedian"] = round(float(np.median(frames)), 1)
        print(f"Implied frame time (median): {document['impliedFrameMsMedian']:.0f} ms", flush=True)
    (args.out / "result.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
    print(f"written to {args.out / 'result.json'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
