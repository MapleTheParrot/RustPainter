"""Paint one detail-heavy crop of the murica image through the full pipeline.

The crop covers the character's face and hair - the region the murica XXL
run scrambled worst - planned at the sign's native 1024x512 on the measured
grid, quantized and dithered the way the app does it, optimized the way the
app optimizes it, and painted by the real Painter with all of its guards.
The sign is captured before and after; scoring happens offline.

    python tools/crop_paint_test.py --out diagnostic/croppaint1

Escape aborts (the Painter pauses on a touched mouse; F7 panic-aborts).
"""

from __future__ import annotations

import argparse
import ctypes
import json
import sys
import threading
import time
from dataclasses import replace
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image

from app.image_processing import load_image, process_image
from app.input_controller import create_system_input_controller
from app.models import ImageProcessOptions, PaintMode, ScreenRect
from app.paint_optimizer import BrushCapabilities, mode_options, optimize_paint_plan
from app.brush_calibration import BrushSizeModel
from app.painter import Painter, PainterSettings, PainterState
from app.profiles import ProfileStore
from app.models import ProcessedImage
from app.screen import capture_region
from app.settings import SettingsStore
from app.texel_grid import TexelGridModel
from tools.decimal_probe import _data_directory, _focus_rust

CROP = (440, 80, 640, 200)  # cells x0, y0, x1, y1 - the face and hair
VK_F7 = 0x76


def _panic(painter: Painter, stop: threading.Event) -> None:
    user32 = ctypes.windll.user32
    while not stop.is_set():
        if user32.GetAsyncKeyState(VK_F7) & 0x8000:
            print("PANIC KEY - aborting", flush=True)
            painter.abort("panic key")
            return
        time.sleep(0.02)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    # Probe + aim audit + paint + verification: half an hour was mid-touch-up.
    parser.add_argument("--timeout", type=float, default=3600.0)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    data = _data_directory()
    store = ProfileStore(data / "profiles")
    profile = next(p for p in store.list_profiles() if "xxl" in p.name.lower())
    grid = TexelGridModel.from_dict(profile.metadata["texel_grid"])
    model = BrushSizeModel.from_dict(profile.metadata["brush_size_model"])
    canvas = ScreenRect(
        profile.canvas.left, profile.canvas.top, profile.canvas.width, profile.canvas.height
    )

    # The plan: full-sign processing at the measured native grid, painted
    # only inside the crop.
    source = load_image(Path(r"c:\Users\yehey\Desktop\RustPainter\murica.png"))
    processed = process_image(
        source,
        ImageProcessOptions(
            logical_width=grid.columns,
            logical_height=grid.rows,
            color_count=224,
            dither=True,
        ),
    )
    rgb = np.asarray(processed.image.convert("RGB"), dtype=np.uint8)
    mask = np.zeros((grid.rows, grid.columns), dtype=bool)
    x0, y0, x1, y1 = CROP
    mask[y0:y1, x0:x1] = np.asarray(processed.paint_mask, dtype=bool)[y0:y1, x0:x1]
    cropped = ProcessedImage(Image.fromarray(rgb, "RGB"), mask, processed.requested_colors)

    capabilities = BrushCapabilities(
        sizing=True,
        cell_pixels=min(grid.pitch_x, grid.pitch_y),
        max_brush_pixels=model.largest_fraction * canvas.height,
    )
    optimized = optimize_paint_plan(
        cropped,
        PaintMode.BALANCED,
        capabilities=capabilities,
        options=mode_options(PaintMode.BALANCED, preserve_dither=True),
    )
    plan = optimized.plan
    promised = np.zeros((grid.rows, grid.columns, 3), dtype=np.uint8)
    opt_rgb = np.asarray(optimized.image.convert("RGB"), dtype=np.uint8)
    opt_mask = np.asarray(optimized.paint_mask, dtype=bool)
    promised[opt_mask] = opt_rgb[opt_mask]
    Image.fromarray(promised[y0:y1, x0:x1], "RGB").save(args.out / "promised_crop.png")
    print(
        f"plan: {plan.width}x{plan.height}, {len(plan.color_groups)} groups, "
        f"{plan.stroke_count} strokes in the crop",
        flush=True,
    )

    if "--plan-only" in sys.argv:
        print("plan built; stopping before any input (--plan-only)")
        return 0

    document = SettingsStore(data / "settings.json").load()
    settings = PainterSettings.from_mapping(document)
    settings = replace(
        settings,
        countdown_seconds=2.0,
        require_foreground=True,
        apply_brush_size=True,
        measure_texel_grid=True,
        pause_on_mouse_move=True,
        verify_passes=1,
    )

    _focus_rust()
    time.sleep(1.0)
    before = capture_region(canvas).convert("RGB")
    before.save(args.out / "before.png")

    painter = Painter(create_system_input_controller())
    stop = threading.Event()
    threading.Thread(target=_panic, args=(painter, stop), daemon=True).start()
    started = time.monotonic()
    if not painter.start(plan, profile, settings):
        raise SystemExit("Painting did not start")
    deadline = time.monotonic() + args.timeout
    while painter.is_alive and time.monotonic() < deadline:
        time.sleep(0.5)
    if painter.is_alive:
        painter.abort("timeout")
    stop.set()
    elapsed = time.monotonic() - started
    print(f"painter: {painter.state.value} ({painter.state_reason}) in {elapsed:.0f}s, "
          f"{painter.progress.completed_strokes} strokes", flush=True)

    time.sleep(0.6)
    after = capture_region(canvas).convert("RGB")
    after.save(args.out / "after.png")
    (args.out / "result.json").write_text(
        json.dumps(
            {
                "state": painter.state.value,
                "seconds": round(elapsed, 1),
                "strokes": painter.progress.completed_strokes,
                "crop": list(CROP),
                "grid": {"columns": grid.columns, "rows": grid.rows},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0 if painter.state is PainterState.COMPLETED else 1


if __name__ == "__main__":
    raise SystemExit(main())
