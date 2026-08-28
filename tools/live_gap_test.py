"""Paint an image with the real painter, export the sign, and map every gap.

The whole application path - brush measurement, the export sweep of the
cursor map, the timing probe, the artwork, the export audit and touch-up -
runs on the live sign, exactly as a job from the GUI does.  Afterwards the
sign's texture is exported once more and every texel the plan covers is
checked: alpha 255 is painted, anything less is a gap.  The count, the
spatial pattern and a magnified gap map go to ``--out``.

    python tools/live_gap_test.py --out diagnostic/gap1 --image PaintTests/USFlag.png
    python tools/live_gap_test.py --out diagnostic/gap2 --image PaintTests/USFlag.png \\
        --window 0,0,256,128 --window 768,384,256,128

``--window x,y,w,h`` (texels) restricts the artwork to those windows, so a
short run still exercises the far corners.  ``--logical WxH`` plans at a
coarser resolution to exercise the native conversion.  F7 or Escape abort.
"""

from __future__ import annotations

import argparse
import ctypes
import json
import logging
import os
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.image_processing import process_image  # noqa: E402
from app.input_controller import create_system_input_controller  # noqa: E402
from app.models import ImageProcessOptions, PaintMode, ScreenRect  # noqa: E402
from app.paint_optimizer import BrushCapabilities, mode_options, optimize_paint_plan  # noqa: E402
from app.paint_plan import generate_paint_plan  # noqa: E402
from app.painter import Painter, PainterSettings, PainterState  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.settings import SettingsStore  # noqa: E402
from app.sign_export import ExportWatcher  # noqa: E402
from app.verification import plan_layers  # noqa: E402
from tools.decimal_probe import _data_directory, _focus_rust  # noqa: E402
from tools.press_timing_probe import hide_overlays, restore_overlays  # noqa: E402

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


def build_plan(image_path: Path, width: int, height: int, mode: str, colors: int, windows, cell_pixels: float):
    image = Image.open(image_path)
    options = ImageProcessOptions(
        logical_width=width,
        logical_height=height,
        scale_mode="stretch",
        color_count=colors,
        sharpen="light",
    )
    processed = process_image(image, options)
    if windows:
        mask = np.zeros((height, width), dtype=bool)
        for x, y, w, h in windows:
            mask[y : y + h, x : x + w] = True
        processed.paint_mask = np.asarray(processed.paint_mask, dtype=bool) & mask
    if mode == "exact":
        return generate_paint_plan(processed, overpaint_gap=0)
    capabilities = BrushCapabilities(sizing=True, cell_pixels=cell_pixels, max_brush_pixels=60.0)
    optimized = optimize_paint_plan(
        processed, PaintMode(mode), capabilities=capabilities, options=mode_options(PaintMode(mode))
    )
    return optimized.plan


def gap_report(alpha: np.ndarray, rgb: np.ndarray, plan, out: Path) -> dict:
    indices, _under, palette = plan_layers(plan)
    covered = indices >= 0
    gaps = covered & (alpha < 250)
    bare = covered & (alpha == 0)
    rows, cols = np.nonzero(gaps)
    report = {
        "covered_texels": int(covered.sum()),
        "gaps": int(gaps.sum()),
        "fully_bare": int(bare.sum()),
        "partial": int(gaps.sum() - bare.sum()),
    }
    if len(rows):
        from collections import Counter

        report["rows_with_gaps"] = int(len(set(rows.tolist())))
        report["cols_with_gaps"] = int(len(set(cols.tolist())))
        report["top_rows"] = Counter(rows.tolist()).most_common(8)
        report["top_cols"] = Counter(cols.tolist()).most_common(8)
        h, w = gaps.shape
        report["blocks_8x4"] = [
            [int(gaps[i * h // 4 : (i + 1) * h // 4, j * w // 8 : (j + 1) * w // 8].sum()) for j in range(8)]
            for i in range(4)
        ]
        runs = Counter()
        for y in set(rows.tolist()):
            line = gaps[y]
            x = 0
            while x < w:
                if line[x]:
                    start = x
                    while x < w and line[x]:
                        x += 1
                    runs[x - start] += 1
                else:
                    x += 1
        report["horizontal_run_lengths"] = sorted(runs.items())[:12]
        report["first_gaps"] = [(int(c), int(r)) for r, c in list(zip(rows, cols))[:40]]
    vis = (rgb * 0.35).astype(np.uint8)
    vis[~covered] = (40, 40, 40)
    vis[gaps] = (255, 0, 255)
    vis[bare] = (255, 255, 0)
    Image.fromarray(vis).save(out / "gap_map.png")
    Image.fromarray(vis).resize((vis.shape[1] * 2, vis.shape[0] * 2), Image.NEAREST).save(out / "gap_map_2x.png")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--logical", type=str, default=None, help="WxH cells; default: the sign's texels")
    parser.add_argument("--mode", type=str, default="quality", choices=("exact", "quality", "balanced", "fast"))
    parser.add_argument("--colors", type=int, default=0)
    parser.add_argument("--window", action="append", default=[], help="x,y,w,h in plan cells")
    parser.add_argument("--verify-passes", type=int, default=3)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=6 * 3600.0)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(args.out / "paint.log", encoding="utf-8")],
    )

    data = _data_directory()
    store = ProfileStore(data / "profiles")
    profile = store.get_default() or store.list_profiles()[0]
    document = SettingsStore(data / "settings.json").load()
    settings = replace(
        PainterSettings.from_mapping(document),
        countdown_seconds=args.countdown,
        require_foreground=True,
        apply_brush_size=True,
        measure_texel_grid=True,
        verify_passes=args.verify_passes,
        pause_on_mouse_move=True,
        confirm_strokes=False,
        anti_afk_enabled=True,
        anti_afk_interval_seconds=25 * 60.0,
    )
    canvas = ScreenRect(profile.canvas.left, profile.canvas.top, profile.canvas.width, profile.canvas.height)
    grid_value = profile.metadata.get("texel_grid") or {}
    columns = int(grid_value.get("columns", 1024))
    rows = int(grid_value.get("rows", 512))
    if args.logical:
        width, height = (int(v) for v in args.logical.lower().split("x"))
    else:
        width, height = columns, rows
    windows = [tuple(int(v) for v in w.split(",")) for w in args.window]
    plan = build_plan(args.image, width, height, args.mode, args.colors, windows, canvas.width / width)
    print(
        f"plan: {plan.width}x{plan.height} cells, {len(plan.color_groups)} colors, "
        f"{plan.stroke_count} strokes, {plan.painted_pixels} painted cells",
        flush=True,
    )

    overlays = hide_overlays()
    _focus_rust()
    controller = create_system_input_controller()
    painter = Painter(controller)
    stop = threading.Event()
    threading.Thread(target=_panic_watch, args=(painter, stop), daemon=True).start()
    print("F7 or Escape aborts.  Starting.", flush=True)
    started = time.monotonic()
    try:
        if not painter.start(plan, profile, settings):
            raise SystemExit("Painting did not start")
        while painter.is_alive and time.monotonic() - started < args.timeout:
            time.sleep(0.5)
        if painter.is_alive:
            painter.abort("timeout")
    finally:
        stop.set()
    state = painter.state
    elapsed = time.monotonic() - started
    print(f"painter: {state.value} ({painter.state_reason}) in {elapsed:.0f}s", flush=True)
    executed = painter.executed_plan or plan
    result = {
        "state": state.value,
        "reason": painter.state_reason,
        "seconds": round(elapsed, 1),
        "plan": [plan.width, plan.height, plan.stroke_count],
        "executed": [executed.width, executed.height, executed.stroke_count],
        "press_hold_ms": (painter.measured_press_hold_seconds or 0) * 1000,
        "stroke_gap_ms": (painter.measured_stroke_gap_seconds or 0) * 1000,
    }
    grid = painter.measured_texel_grid
    if grid is not None:
        result["grid"] = {"columns": grid.columns, "rows": grid.rows, "swept": grid.swept, "pitch": [grid.pitch_x, grid.pitch_y]}

    # The final, independent export.
    time.sleep(0.6)
    watcher = ExportWatcher()
    watcher.snapshot()
    button = profile.download_button
    controller.click(button.left + button.width / 2, button.top + button.height / 2, hold_seconds=0.09)
    export = watcher.collect(keep_copy_in=args.out)
    restore_overlays(overlays)
    if export is None:
        print("no final export could be read", flush=True)
        (args.out / "result.json").write_text(json.dumps(result, indent=2, default=str))
        return 1
    Path(export.source).replace(args.out / "final_export.png")
    image = np.asarray(Image.open(args.out / "final_export.png").convert("RGBA"))
    alpha = image[:, :, 3].astype(int)
    rgb = image[:, :, :3].astype(np.float32)
    if (executed.width, executed.height) != (alpha.shape[1], alpha.shape[0]):
        print(f"executed plan is {executed.width}x{executed.height} but the export is {alpha.shape[1]}x{alpha.shape[0]}", flush=True)
        result["gap_report"] = None
    else:
        report = gap_report(alpha, rgb, executed, args.out)
        result["gap_report"] = report
        print("GAP MAP:", json.dumps(report, default=str), flush=True)
    (args.out / "result.json").write_text(json.dumps(result, indent=2, default=str))
    return 0 if state is PainterState.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
