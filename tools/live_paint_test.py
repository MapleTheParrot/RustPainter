"""End-to-end in-game test: measure, paint an image, score the sign against the plan.

Runs the application's real pipeline - processing, optimization, the Painter
itself with its own safety guards - against the live game, then captures the
sign and scores it cell-by-cell with the same relative-Lab comparison the
verification pass uses.  Every run leaves images on disk so a disagreement is
settled by looking, not remembering.

    python tools/live_paint_test.py --image PaintTests/PurpleCat.jpg \
        --mode balanced --quality 64 --out diagnostic/run1 [--measure] [--clear]

Escape aborts; the Painter pauses itself if Rust loses focus or the mouse moves.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.image_processing import load_image, process_image  # noqa: E402
from app.input_controller import create_system_input_controller  # noqa: E402
from app.models import ImageProcessOptions, PaintMode, ScreenRect  # noqa: E402
from app.paint_optimizer import (  # noqa: E402
    BrushCapabilities,
    mode_options,
    optimize_paint_plan,
)
from app.paint_plan import generate_paint_plan  # noqa: E402
from app.painter import Painter, PainterSettings, PainterState  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.screen import capture_region  # noqa: E402
from app.verification import (  # noqa: E402
    mismatched_cells,
    plan_expectations,
    sample_cell_colors,
)
from tools._safety import Guard  # noqa: E402
from tools.decimal_probe2 import _data_directory, _focus_rust  # noqa: E402


TRASH_BUTTON = (80, 80)


def _rect(value) -> ScreenRect:
    return ScreenRect(value.left, value.top, value.width, value.height)


def _settings(**overrides) -> PainterSettings:
    values = dict(
        countdown_seconds=0.0,
        require_foreground=True,
        verify_passes=1,
        apply_brush_size=True,
    )
    values.update(overrides)
    return PainterSettings(**values)


def _wait(painter: Painter, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while painter.is_alive and time.monotonic() < deadline:
        time.sleep(0.5)
    if painter.is_alive:
        painter.abort("test timeout")
        raise SystemExit(f"Painter still running after {timeout:.0f}s; aborted")


def clear_canvas(guard: Guard, canvas: ScreenRect, park) -> None:
    guard.check()
    guard.input.click(*TRASH_BUTTON, hold_seconds=0.09)
    guard.commanded(*TRASH_BUTTON)
    time.sleep(0.6)
    guard.park(park)


def measure(store: ProfileStore, profile, cell_fraction: float, output: Path) -> None:
    painter = Painter(create_system_input_controller())
    painter.configure_brush_measurement(
        profile, _settings(), cell_fraction=cell_fraction
    )
    if not painter.start():
        raise SystemExit("Measurement did not start")
    _wait(painter, 240)
    if painter.state is not PainterState.COMPLETED:
        raise SystemExit(f"Measurement ended {painter.state}: {painter.state_reason}")
    model = painter.measured_brush_size_model
    assert model is not None
    profile.metadata["brush_size_model"] = model.to_dict()
    store.save(profile)
    print(
        f"measured: slope {model.slope:.6f} intercept {model.intercept:.6f} "
        f"(~{model.sign_pixel_rows:.0f} sign rows)"
    )
    (output / "model.json").write_text(json.dumps(model.to_dict(), indent=2))


def paint_and_score(profile, plan, expected_palette_image, output: Path, timeout: float) -> dict:
    canvas = _rect(profile.canvas)
    before = capture_region(canvas).convert("RGB")
    before.save(output / "sign_before.png")

    painter = Painter(create_system_input_controller())
    errors: list[str] = []
    painter.set_callbacks(on_error=lambda exc: errors.append(str(exc)))
    started = time.monotonic()
    if not painter.start(plan, profile, _settings()):
        raise SystemExit("Painting did not start")
    _wait(painter, timeout)
    elapsed = time.monotonic() - started
    state = painter.state
    print(f"painter finished: {state.value} in {elapsed:.0f}s "
          f"({painter.progress.completed_strokes} strokes)")
    if errors:
        print("painter error:", errors[0])

    time.sleep(0.5)
    after = capture_region(canvas).convert("RGB")
    after.save(output / "sign_after.png")

    indices, palette = plan_expectations(plan)
    sampled = sample_cell_colors(
        np.asarray(after, dtype=np.float32), plan.width, plan.height
    )
    mismatch = mismatched_cells(sampled, indices, palette)
    covered_mask = indices >= 0
    covered = int(covered_mask.sum())
    wrong = int(mismatch.sum())
    match_percent = 100.0 * (1 - wrong / covered) if covered else 0.0

    # The sign's material shifts every color, so also score after fitting one
    # global affine transform from capture to plan - the number that reflects
    # what a person comparing preview and sign actually sees.
    lit_match = 0.0
    if covered:
        X = sampled[covered_mask]
        Y = palette[indices[covered_mask]].astype(np.float32)
        design = np.hstack([X, np.ones((len(X), 1), np.float32)])
        coefficients, *_ = np.linalg.lstsq(design, Y, rcond=None)
        corrected = sampled.copy()
        corrected[covered_mask] = np.clip(design @ coefficients, 0, 255)
        lit_match = 100.0 * (
            1 - mismatched_cells(corrected, indices, palette).sum() / covered
        )
    print(f"cells: {covered} covered, {wrong} raw-wrong -> {match_percent:.1f}% raw, "
          f"{lit_match:.1f}% lighting-normalized")

    # Side-by-side: what the plan promises vs what the sign shows.
    scale = max(1, 640 // plan.width)
    promised = expected_palette_image.resize(
        (plan.width * scale, plan.height * scale), Image.NEAREST
    )
    shown = after.resize(promised.size)
    sheet = Image.new("RGB", (promised.width * 2 + 8, promised.height), (24, 24, 24))
    sheet.paste(promised, (0, 0))
    sheet.paste(shown, (promised.width + 8, 0))
    sheet.save(output / "promised_vs_painted.png")

    return {
        "state": state.value,
        "seconds": round(elapsed, 1),
        "strokes": painter.progress.completed_strokes,
        "covered_cells": covered,
        "wrong_cells": wrong,
        "match_percent": round(match_percent, 2),
        "lighting_normalized_percent": round(lit_match, 2),
        "errors": errors,
    }


def build_plan(image_path: Path, mode: str, long_edge: int, colors: int, profile, model):
    source = load_image(image_path)
    aspect = profile.canvas.width / profile.canvas.height
    if aspect >= 1.0:
        width, height = long_edge, max(8, round(long_edge / aspect))
    else:
        width, height = max(8, round(long_edge * aspect)), long_edge
    processed = process_image(
        source,
        ImageProcessOptions(logical_width=width, logical_height=height, color_count=colors),
    )
    canvas = profile.canvas
    cell = min(canvas.width / width, canvas.height / height)
    if mode == "exact":
        plan = generate_paint_plan(processed, overpaint_gap=6)
    else:
        capabilities = BrushCapabilities(
            sizing=True,
            cell_pixels=cell,
            max_brush_pixels=model.largest_fraction * canvas.height if model else 0.0,
        )
        optimized = optimize_paint_plan(
            processed,
            PaintMode(mode),
            capabilities=capabilities,
            options=mode_options(PaintMode(mode)),
        )
        plan = optimized.plan
    indices, palette = plan_expectations(plan)
    rgb = np.zeros((plan.height, plan.width, 3), dtype=np.uint8)
    covered = indices >= 0
    rgb[covered] = palette[indices[covered]]
    return plan, Image.fromarray(rgb, "RGB")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--mode", default="balanced",
                        choices=("exact", "quality", "balanced", "fast"))
    parser.add_argument("--quality", type=int, default=64, help="long edge in cells")
    parser.add_argument("--colors", type=int, default=32)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--measure", action="store_true")
    parser.add_argument("--clear", action="store_true")
    parser.add_argument("--timeout", type=float, default=600.0)
    arguments = parser.parse_args()

    arguments.out.mkdir(parents=True, exist_ok=True)
    store = ProfileStore(_data_directory() / "profiles")
    profile = store.get_default() or store.list_profiles()[0]
    if profile.clear_button is None:
        # Painting measures the brush itself and wipes the probes with this
        # control.  The harness has always known where the trash icon is, so
        # it stands in for a profile calibrated before the field existed.
        profile.clear_button = ScreenRect(
            TRASH_BUTTON[0] - 6, TRASH_BUTTON[1] - 6, 12, 12
        )
    canvas = _rect(profile.canvas)
    park = (
        int(profile.color_box.left + profile.color_box.width / 2),
        int(profile.color_box.top + profile.color_box.height / 2),
    )

    _focus_rust()
    guard = Guard(create_system_input_controller(), budget_seconds=30)

    if arguments.clear:
        clear_canvas(guard, canvas, park)
        print("canvas cleared")

    if arguments.measure:
        # Painting now measures the brush on every run, so this only exists to
        # score a measurement on its own, or to seed a model for the planner
        # before the first paint of a new sign.
        cell = min(canvas.width / 256, canvas.height / 214)
        measure(store, profile, cell / canvas.height, arguments.out)
        profile = store.get_default() or store.list_profiles()[0]
        # The probe strokes would otherwise sit inside the painting's margins.
        clear_canvas(guard, canvas, park)
        print("canvas cleared after measurement")

    from app.brush_calibration import BrushSizeModel

    stored = profile.metadata.get("brush_size_model")
    model = BrushSizeModel.from_dict(stored) if isinstance(stored, dict) else None
    if model is None:
        raise SystemExit("No brush size model; run with --measure first")

    plan, promised = build_plan(
        arguments.image, arguments.mode, arguments.quality, arguments.colors,
        profile, model,
    )
    print(f"plan: {plan.width}x{plan.height}, {len(plan.color_groups)} groups, "
          f"{plan.stroke_count} strokes")
    result = paint_and_score(profile, plan, promised, arguments.out, arguments.timeout)
    result.update(mode=arguments.mode, quality=arguments.quality, colors=arguments.colors)
    (arguments.out / "result.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
