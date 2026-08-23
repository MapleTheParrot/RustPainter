"""Measure the brush and the texel grid on the live sign, and nothing else.

Runs the application's own measurement job - the one every paint job runs
first - against the game, with the same safety rails plus a panic key:

* F7 or Escape aborts at once (polled from a watchdog thread)
* the Painter pauses itself if Rust loses focus or the mouse is touched
* the screen corners abort, as in the app

    python tools/live_grid_probe.py --out diagnostic/grid1 [--clear]

It prints the painter's log, saves the canvas before and after, and writes the
measured grid and brush model as JSON.  ``--clear`` clicks the sign's clear
control afterwards, which a measurement-only job does not do by itself.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.input_controller import create_system_input_controller  # noqa: E402
from app.models import ScreenRect  # noqa: E402
from app.painter import Painter, PainterSettings, PainterState  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.screen import capture_region  # noqa: E402
from app.settings import SettingsStore  # noqa: E402
from tools.decimal_probe import _data_directory, _focus_rust  # noqa: E402


VK_F7 = 0x76
VK_ESCAPE = 0x1B


def _panic_watch(painter: Painter, stop: threading.Event) -> None:
    user32 = ctypes.windll.user32
    while not stop.is_set():
        if user32.GetAsyncKeyState(VK_F7) & 0x8000 or user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
            print("PANIC KEY - aborting", flush=True)
            painter.abort("panic key")
            return
        time.sleep(0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clear", action="store_true", help="clear the sign afterwards")
    parser.add_argument("--clear-first", action="store_true", help="clear the sign before probing")
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=300.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.out / "probe.log", encoding="utf-8"),
        ],
    )

    data = _data_directory()
    store = ProfileStore(data / "profiles")
    profile = store.get_default() or store.list_profiles()[0]
    if profile.brush_size_box is None or profile.clear_button is None:
        raise SystemExit("The profile needs the Size box and clear control calibrated")
    document = SettingsStore(data / "settings.json").load()
    settings = PainterSettings.from_mapping(document)
    from dataclasses import replace as _replace

    settings = _replace(
        settings,
        countdown_seconds=args.countdown,
        require_foreground=True,
        apply_brush_size=True,
        measure_texel_grid=True,
        pause_on_mouse_move=True,
    )
    canvas = ScreenRect(profile.canvas.left, profile.canvas.top, profile.canvas.width, profile.canvas.height)
    print(f"profile {profile.name}: canvas {canvas}", flush=True)

    _focus_rust()
    if args.clear_first:
        controller = create_system_input_controller()
        clear = profile.clear_button
        controller.click(clear.left + clear.width // 2, clear.top + clear.height // 2, hold_seconds=0.09)
        time.sleep(0.8)
        print("sign cleared before probing", flush=True)
    capture_region(canvas).convert("RGB").save(args.out / "sign_before.png")

    painter = Painter(create_system_input_controller())
    # One cell per roughly 300 rows: the brush probes bracket a one-texel brush.
    painter.configure_brush_measurement(profile, settings, cell_fraction=1.0 / 300.0)
    stop = threading.Event()
    watchdog = threading.Thread(target=_panic_watch, args=(painter, stop), daemon=True)
    watchdog.start()
    print("F7 or Escape aborts.  Starting.", flush=True)
    if not painter.start():
        raise SystemExit("Measurement did not start")
    deadline = time.monotonic() + args.timeout
    while painter.is_alive and time.monotonic() < deadline:
        time.sleep(0.25)
    if painter.is_alive:
        painter.abort("timeout")
    stop.set()
    state = painter.state
    print(f"painter: {state.value} ({painter.state_reason})", flush=True)

    time.sleep(0.5)
    capture_region(canvas).convert("RGB").save(args.out / "sign_after.png")
    model = painter.measured_brush_size_model
    grid = painter.measured_texel_grid
    result = {
        "state": state.value,
        "reason": painter.state_reason,
        "canvas": [canvas.left, canvas.top, canvas.width, canvas.height],
        "brush_model": model.to_dict() if model else None,
        "texel_grid": grid.to_dict() if grid else None,
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    if model:
        print(f"brush: size 1 = {model.fraction_for_size(1.0) * canvas.height:.2f}px tall"
              f" (~{model.sign_pixel_rows:.0f} rows by the old inference)")
    if grid:
        print(f"GRID: {grid.columns}x{grid.rows} texels, pitch {grid.pitch_x:.4f}x{grid.pitch_y:.4f}px, "
              f"origin ({grid.origin_x:.2f}, {grid.origin_y:.2f}), cursor {grid.aim_pitch_x:.4f}x{grid.aim_pitch_y:.4f}px "
              f"from ({grid.aim_origin_x:.2f}, {grid.aim_origin_y:.2f}), "
              f"residual {grid.residual:.3f}, {'edges' if grid.from_edges else 'rectangle'}")
    else:
        print("GRID: not measured (see the warning in the log)")

    if args.clear and state is PainterState.COMPLETED:
        controller = create_system_input_controller()
        clear = profile.clear_button
        controller.click(clear.left + clear.width // 2, clear.top + clear.height // 2, hold_seconds=0.09)
        time.sleep(0.8)
        capture_region(canvas).convert("RGB").save(args.out / "sign_cleared.png")
        print("sign cleared", flush=True)
    return 0 if state is PainterState.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
