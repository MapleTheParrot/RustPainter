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
import ctypes
import ctypes.wintypes as wintypes
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.brush_calibration import format_brush_size  # noqa: E402
from app.color_mapping import map_rgb_to_picker  # noqa: E402
from app.painter import _high_resolution_timer  # noqa: E402
from app.input_controller import MouseButton, create_system_input_controller  # noqa: E402
from app.models import ScreenRect  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.screen import capture_region  # noqa: E402
from app.ui_guard import looks_like_hue_bar  # noqa: E402
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

# After Rust is handed the foreground by a script, before it is clicked.
FOCUS_SETTLE_SECONDS = 1.5

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


# The GUI's calibration overlay draws red outlines over the sign.  They are
# click-through and the painter hides them while it works, but a script
# driving the mouse itself has to, or every capture is scored against a
# rectangle with a red border burnt across it.
_OVERLAY_TITLE = "RustPainter Calibration Preview"
# The GUI also keeps one small top-level window per calibrated rectangle -
# "Resize Canvas calibration - RustPainter" and so on - carrying the drag
# handles around its border.  Those are NOT click-through: a press on one
# grabs the handle, and a probe that dabs along a sign's edge with them up
# resizes the profile's rectangles instead of painting (measured live: a
# sweep dragged the canvas rectangle to three pixels wide).
_HANDLE_TITLE_SUFFIX = " calibration - RustPainter"
_SW_HIDE, _SW_SHOWNA = 0, 8


def _overlay_windows() -> list:
    user32 = ctypes.windll.user32
    found: list = []

    def visit(hwnd, _lparam):
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length:
            buffer = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buffer, length + 1)
            if _OVERLAY_TITLE in buffer.value or buffer.value.endswith(
                _HANDLE_TITLE_SUFFIX
            ):
                found.append(hwnd)
        return True

    prototype = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    user32.EnumWindows.argtypes = [prototype, wintypes.LPARAM]
    user32.EnumWindows(prototype(visit), 0)
    return found


def hide_overlays() -> list:
    windows = _overlay_windows()
    for hwnd in windows:
        ctypes.windll.user32.ShowWindow(hwnd, _SW_HIDE)
    if windows:
        print(f"hid {len(windows)} calibration overlay window(s)", flush=True)
    return windows


def restore_overlays(windows) -> None:
    for hwnd in windows:
        ctypes.windll.user32.ShowWindow(hwnd, _SW_SHOWNA)


def require_painting_ui(profile) -> None:
    """Refuse to paint unless Rust's painting UI is actually on the screen.

    Every dot below is a left click at a point that is only a sign while the
    painting interface is open.  With it closed those clicks land in the
    world, on whatever the player happens to be holding, a thousand times
    over.  The hue bar is the cheapest thing to recognise: a saturated strip
    running through the spectrum, which nothing else on the screen is.
    """

    rect = profile.hue_bar
    region = ScreenRect(rect.left, rect.top, rect.width, rect.height)
    if not looks_like_hue_bar(capture_region(region)):
        raise SystemExit(
            "Rust's painting UI is not on the screen (no hue bar at the "
            f"calibrated {region.left},{region.top} {region.width}x{region.height}).  "
            "Open the sign's painting interface and run this again."
        )


def set_brush_size(guard: Guard, profile, size: float) -> None:
    """Type a Size into Rust's field the way the painter does: click it,
    clear it from both sides of the caret, type, Enter."""

    box = profile.brush_size_box
    if box is None:
        raise SystemExit("--size needs the profile's Size value box calibrated")
    guard.check()
    guard.input.click(box.left + box.width / 2, box.top + box.height / 2, hold_seconds=0.09)
    guard.commanded(box.left + box.width / 2, box.top + box.height / 2)
    time.sleep(0.12)
    for key in ("BACKSPACE",) * 6 + ("DELETE",) * 6:
        guard.input.press_key(key, hold_seconds=0.03)
        time.sleep(0.02)
    for char in format_brush_size(size):
        guard.input.press_key(0xBE if char == "." else char, hold_seconds=0.03)
        time.sleep(0.02)
    guard.input.press_key("ENTER", hold_seconds=0.03)
    time.sleep(0.15)
    print(f"brush size set to {format_brush_size(size)}", flush=True)


def select_color(guard: Guard, profile, color) -> None:
    """Click the picker for one colour, holding each click across a frame."""

    # The same fixed directions the painter uses for every profile.
    coordinates = map_rgb_to_picker(
        color,
        profile.hue_bar,
        profile.color_box,
        hue_direction=getattr(profile, "hue_direction", "bottom_to_top"),
        saturation_direction=getattr(profile, "saturation_direction", "left_low"),
        value_direction=getattr(profile, "value_direction", "top_bright"),
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


def paint_dot(guard: Guard, point, hold_seconds: float, length_pixels: float) -> float:
    """One press at ``point``, held ``hold_seconds``; a short drag if asked.

    Returns how long the button was really down.  A hold asked for in
    milliseconds is not the hold delivered - Windows' scheduler rounds a
    sleep to its timer resolution - and a measurement of the game has to be
    made against what the game was actually given.
    """

    guard.check()
    guard.input.move_mouse(*point)
    guard.commanded(*point)
    time.sleep(0.002)
    guard.input.mouse_down(MouseButton.LEFT)
    pressed_at = time.perf_counter()
    try:
        if length_pixels > 0:
            steps = max(1, int(length_pixels))
            for step in range(1, steps + 1):
                guard.input.move_mouse(point[0] + step * length_pixels / steps, point[1])
            guard.commanded(point[0] + length_pixels, point[1])
        remaining = hold_seconds - (time.perf_counter() - pressed_at)
        if remaining > 0:
            time.sleep(remaining)
        held = time.perf_counter() - pressed_at
    finally:
        guard.input.mouse_up(MouseButton.LEFT)
    time.sleep(0.015)
    return held


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
        "--profile",
        type=str,
        default=None,
        help="profile name to measure on (default: the app's selected profile)",
    )
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
    parser.add_argument(
        "--size",
        type=float,
        default=None,
        help="type this brush Size into Rust first (default: leave the brush as it is)",
    )
    parser.add_argument(
        "--color-offset",
        type=int,
        default=0,
        help="start the band colours this far into the palette, to differ from an earlier test",
    )
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
    profiles = store.list_profiles()
    if args.profile:
        wanted = args.profile.strip().lower()
        matches = [p for p in profiles if p.name.strip().lower() == wanted]
        if not matches:
            matches = [p for p in profiles if wanted in p.name.strip().lower()]
        if len(matches) != 1:
            names = ", ".join(repr(p.name) for p in profiles)
            raise SystemExit(
                f"--profile {args.profile!r} matched {len(matches)} profiles; choose one of: {names}"
            )
        profile = matches[0]
    else:
        profile = store.get_default() or profiles[0]
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

    overlays = hide_overlays()
    _focus_rust()
    controller = create_system_input_controller()
    guard = Guard(controller, budget_seconds=args.budget)
    # Rust wants a moment after it is given the foreground programmatically:
    # a click sent too soon after has been seen closing the painting UI
    # instead of pressing what it was aimed at.
    time.sleep(FOCUS_SETTLE_SECONDS)
    require_painting_ui(profile)
    countdown(args.countdown, "measuring the press hold")

    if not args.no_clear and profile.clear_button is not None:
        clear = profile.clear_button
        guard.click(clear.left + clear.width / 2, clear.top + clear.height / 2, settle=0.9)
        # Rust takes the cursor back whenever it closes or reopens anything,
        # so the guard's idea of where the script left the mouse has to be
        # re-established before it is asked to judge a hand on the mouse.
        guard.commanded(*controller.get_cursor_position())
        guard.park(park)
        print("sign cleared", flush=True)

    results = []
    try:
        require_painting_ui(profile)
        if args.size is not None:
            set_brush_size(guard, profile, args.size)
        for index, hold in enumerate(holds):
            points = bands[index]
            color = BAND_COLORS[(index + args.color_offset) % len(BAND_COLORS)]
            select_color(guard, profile, color)
            guard.park(park, settle=BAND_SETTLE_SECONDS)
            before = np.asarray(capture_region(canvas).convert("RGB"), dtype=np.float32)
            started = time.monotonic()
            held = [paint_dot(guard, point, hold, args.length) for point in points]
            elapsed = time.monotonic() - started
            actual = float(np.median(held)) * 1000.0
            guard.park(park, settle=BAND_SETTLE_SECONDS)
            after_image = capture_region(canvas).convert("RGB")
            after = np.asarray(after_image, dtype=np.float32)
            found, missing = score_band(before, after, canvas, points, DOT_PATCH_PIXELS)
            total = len(found) + len(missing)
            rate = len(missing) / total if total else 0.0
            frame = actual / (1.0 - rate) if rate < 1.0 else float("inf")
            results.append(
                {
                    "hold_ms": round(hold * 1000.0, 1),
                    "actual_hold_ms": round(actual, 2),
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
                f"  hold {hold * 1000:5.0f} ms (really {actual:5.1f}): "
                f"{len(found):4d}/{total:4d} painted, "
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
        restore_overlays(overlays)

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
        document["cleanestHoldMs"] = min(row["actual_hold_ms"] for row in clean)
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
    # Windows rounds a sleep to the current timer resolution - 15.6 ms by
    # default, which is longer than most of the holds worth measuring.  The
    # painter raises the resolution for exactly this reason and so must
    # anything measuring it.
    with _high_resolution_timer():
        raise SystemExit(main())
