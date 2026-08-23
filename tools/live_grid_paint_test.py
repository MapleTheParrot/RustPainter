"""Paint a dot lattice on the live sign and check every dot hit its texel.

Runs a real paint job - brush probes, texel grid probe, wipe, then the
artwork - with a plan of single-texel dots every ``--every`` texels in both
directions, in ``--groups`` colors dealt out in turn.  Afterwards the sign is
captured and each dot's rendered centre is compared with where the measured
grid says that texel is.  A dot a texel off shows as a residual of a whole
pitch; a dot the game never painted shows as missing.

    python tools/live_grid_paint_test.py --out diagnostic/dots1 [--every 8]

With ``--confirm`` the painter checks each color as it goes down, exactly as
a real job does: the log then carries one line per color with how many dots
the game dropped on the first reading, and the press hold climbing while
they keep dropping - which on the largest sign is the measurement that
matters.  ``--groups 8 --hold 70`` reads the drop rate at the floor and then
at each raised hold; ``--confirm-rounds 1`` measures without repainting.

F7 or Escape aborts; the painter pauses on focus loss or a touched mouse.
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

import numpy as np
from PIL import ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.input_controller import create_system_input_controller  # noqa: E402
from app.models import ColorGroup, PaintPlan, ScreenRect, Stroke  # noqa: E402
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


def dot_plan(columns: int, rows: int, every: int, colors) -> PaintPlan:
    """Single-texel dots on a lattice, dealt out to the colors in turn."""

    groups = []
    count = len(colors)
    for which, color in enumerate(colors):
        strokes = tuple(
            Stroke(x, y, x, y)
            for y in range(every // 2, rows, every)
            for x in range(every // 2, columns, every)
            if ((x // every) + (y // every)) % count == which
        )
        groups.append(ColorGroup(color, strokes, len(strokes)))
    return PaintPlan(columns, rows, tuple(groups))


# Distinct, saturated colors that read clearly against any bare sign.
DOT_COLORS = (
    (30, 60, 220),
    (220, 40, 40),
    (30, 170, 60),
    (240, 180, 20),
    (200, 40, 200),
    (20, 190, 200),
    (250, 120, 30),
    (120, 60, 220),
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--every", type=int, default=8)
    parser.add_argument("--columns", type=int, default=320)
    parser.add_argument("--rows", type=int, default=240)
    parser.add_argument("--countdown", type=float, default=3.0)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--groups", type=int, default=2, help="colors the dots are dealt to")
    parser.add_argument(
        "--hold",
        type=float,
        default=None,
        help="press hold in ms (default: the settings file's, floored)",
    )
    parser.add_argument(
        "--confirm", action="store_true", help="check each color as it goes down, as a job does"
    )
    parser.add_argument("--confirm-rounds", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.groups <= len(DOT_COLORS):
        parser.error(f"--groups must be between 1 and {len(DOT_COLORS)}")
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
    base = PainterSettings.from_mapping(document)
    settings = PainterSettings(
        **{
            **{f: getattr(base, f) for f in base.__dataclass_fields__},
            "countdown_seconds": args.countdown,
            "require_foreground": True,
            "apply_brush_size": True,
            "measure_texel_grid": True,
            "verify_passes": 0,
            "pause_on_mouse_move": True,
            "confirm_strokes": bool(args.confirm),
            "confirm_max_rounds": args.confirm_rounds,
            **(
                {"mouse_down_duration_seconds": args.hold / 1000.0}
                if args.hold is not None
                else {}
            ),
        }
    )
    canvas = ScreenRect(profile.canvas.left, profile.canvas.top, profile.canvas.width, profile.canvas.height)
    colors = DOT_COLORS[: args.groups]
    plan = dot_plan(args.columns, args.rows, args.every, colors)
    print(
        f"plan: {plan.width}x{plan.height}, {plan.stroke_count} dots in {len(colors)} colors; "
        f"hold {settings.mouse_down_duration_seconds * 1000:.0f} ms, "
        f"{'checking each color' if args.confirm else 'no per-color check'}",
        flush=True,
    )

    _focus_rust()
    # The blank canvas carries coloured specks of its own; scoring against a
    # capture of it cleared, rather than against a flat colour, keeps them
    # from pulling a dot's measured centre.
    controller = create_system_input_controller()
    clear = profile.clear_button
    controller.click(clear.left + clear.width // 2, clear.top + clear.height // 2, hold_seconds=0.09)
    time.sleep(0.8)
    bare = np.asarray(capture_region(canvas).convert("RGB"), dtype=np.float32)
    painter = Painter(controller)
    stop = threading.Event()
    threading.Thread(target=_panic_watch, args=(painter, stop), daemon=True).start()
    print("F7 or Escape aborts.  Starting.", flush=True)
    if not painter.start(plan, profile, settings):
        raise SystemExit("Painting did not start")
    started = time.monotonic()
    deadline = started + args.timeout
    while painter.is_alive and time.monotonic() < deadline:
        time.sleep(0.5)
    if painter.is_alive:
        painter.abort("timeout")
    stop.set()
    state = painter.state
    print(f"painter: {state.value} ({painter.state_reason}) in {time.monotonic() - started:.0f}s", flush=True)

    time.sleep(0.6)
    after = capture_region(canvas).convert("RGB")
    after.save(args.out / "sign_after.png")
    grid = painter.measured_texel_grid
    confirmation = painter.confirmation_summary.to_dict()
    print("per-color check:", json.dumps(confirmation), flush=True)
    result: dict = {
        "state": state.value,
        "grid": grid.to_dict() if grid else None,
        "confirmation": confirmation,
        "hold_ms": settings.mouse_down_duration_seconds * 1000.0,
    }
    if grid is None or state is not PainterState.COMPLETED:
        (args.out / "result.json").write_text(json.dumps(result, indent=2))
        print("no grid or job not completed; nothing to score", flush=True)
        return 1

    # Score: every dot's rendered centre against the grid's prediction.
    pixels = np.asarray(after, dtype=np.float32)
    distance = np.linalg.norm(pixels - bare, axis=2)
    residuals = []
    missing = []
    missing_per_color = []
    sheet = after.copy()
    draw = ImageDraw.Draw(sheet)
    half = int(max(4, round(1.5 * max(grid.pitch_x, grid.pitch_y))))
    for group in plan.color_groups:
        missing_before = len(missing)
        for stroke in group.strokes:
            px = grid.origin_x + (stroke.start_x + 0.5) * grid.pitch_x - canvas.left
            py = grid.origin_y + (stroke.start_y + 0.5) * grid.pitch_y - canvas.top
            x0, y0 = int(round(px)) - half, int(round(py)) - half
            patch = distance[max(0, y0) : y0 + 2 * half + 1, max(0, x0) : x0 + 2 * half + 1]
            if patch.size == 0 or patch.max() < 40:
                missing.append((stroke.start_x, stroke.start_y))
                draw.ellipse((px - 3, py - 3, px + 3, py + 3), outline=(255, 255, 0))
                continue
            strong = patch >= max(24.0, patch.max() * 0.5)
            ys, xs = np.nonzero(strong)
            w = patch[ys, xs]
            cx = (w * xs).sum() / w.sum() + max(0, x0) + 0.5
            cy = (w * ys).sum() / w.sum() + max(0, y0) + 0.5
            residuals.append((cx - px, cy - py))
            draw.line((px - 2, py, px + 2, py), fill=(0, 255, 0))
            draw.line((px, py - 2, px, py + 2), fill=(0, 255, 0))
        missing_per_color.append(
            {
                "color": list(group.color),
                "dots": len(group.strokes),
                "missing": len(missing) - missing_before,
            }
        )
    sheet.save(args.out / "scored.png")
    res = np.array(residuals) if residuals else np.zeros((0, 2))
    off_x = int((np.abs(res[:, 0]) > grid.pitch_x / 2).sum()) if len(res) else 0
    off_y = int((np.abs(res[:, 1]) > grid.pitch_y / 2).sum()) if len(res) else 0
    summary = {
        "dots": plan.stroke_count,
        "found": len(residuals),
        "missing": len(missing),
        "mean_residual_px": [float(res[:, 0].mean()), float(res[:, 1].mean())] if len(res) else None,
        "rms_residual_px": [float(np.sqrt((res[:, 0] ** 2).mean())), float(np.sqrt((res[:, 1] ** 2).mean()))] if len(res) else None,
        "max_abs_residual_px": [float(np.abs(res[:, 0]).max()), float(np.abs(res[:, 1]).max())] if len(res) else None,
        "off_by_half_texel_or_more": [off_x, off_y],
        "missing_per_color": missing_per_color,
        "missing_cells": missing[:50],
    }
    result["score"] = summary
    (args.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(summary, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
