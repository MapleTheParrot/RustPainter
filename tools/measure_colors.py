"""Paint the 32-swatch chart in game, measure the material response, store it.

The same flow as the GUI's Prepare/Measure Calibration Chart buttons, driven
headlessly: paint the chart in exact mode, capture the sign, fit the affine
material model, and save it into the profile.  Afterwards the preview renders
through the model and painting compensates commanded colors.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.color_calibration import (  # noqa: E402
    build_calibration_chart,
    fit_color_correction,
    sample_painted_chart,
)
from app.image_processing import process_image  # noqa: E402
from app.input_controller import create_system_input_controller  # noqa: E402
from app.models import ImageProcessOptions, ScreenRect  # noqa: E402
from app.paint_plan import generate_paint_plan  # noqa: E402
from app.painter import Painter, PainterState  # noqa: E402
from app.profiles import ProfileStore  # noqa: E402
from app.screen import capture_region  # noqa: E402
from tools._safety import Guard  # noqa: E402
from tools.decimal_probe import _data_directory, _focus_rust  # noqa: E402
from tools.live_paint_test import _settings, _wait, clear_canvas  # noqa: E402


def main() -> None:
    output = Path("diagnostic/colorchart")
    output.mkdir(parents=True, exist_ok=True)
    store = ProfileStore(_data_directory() / "profiles")
    profile = store.get_default() or store.list_profiles()[0]
    canvas = ScreenRect(
        profile.canvas.left, profile.canvas.top,
        profile.canvas.width, profile.canvas.height,
    )
    park = (
        int(profile.color_box.left + profile.color_box.width / 2),
        int(profile.color_box.top + profile.color_box.height / 2),
    )

    _focus_rust()
    guard = Guard(create_system_input_controller(), budget_seconds=30)
    clear_canvas(guard, canvas, park)
    print("canvas cleared")

    # The chart painted like any image, in exact mode, raw material response.
    chart = build_calibration_chart()
    aspect = canvas.width / canvas.height
    width = 64
    height = max(8, round(width / aspect))
    processed = process_image(
        chart,
        ImageProcessOptions(
            logical_width=width, logical_height=height,
            scale_mode="stretch", color_count=32,
        ),
    )
    plan = generate_paint_plan(processed, overpaint_gap=0)
    print(f"chart plan: {plan.width}x{plan.height}, {plan.stroke_count} strokes")

    # Chart painting must command raw colors, so strip any stored correction.
    chart_profile = type(profile).from_dict(profile.to_dict())
    chart_profile.metadata.pop("color_correction", None)

    painter = Painter(create_system_input_controller())
    if not painter.start(plan, chart_profile, _settings(verify_passes=0)):
        raise SystemExit("Chart painting did not start")
    _wait(painter, 500)
    if painter.state is not PainterState.COMPLETED:
        raise SystemExit(f"Chart ended {painter.state}: {painter.state_reason}")
    print("chart painted")

    guard = Guard(create_system_input_controller(), budget_seconds=30)
    guard.park(park)
    time.sleep(0.5)
    capture = capture_region(canvas).convert("RGB")
    capture.save(output / "chart_capture.png")
    processed.image.save(output / "chart_commanded.png")

    requested, observed = sample_painted_chart(capture, processed.image)
    model = fit_color_correction(requested, observed)
    print(f"fit: rmse {model.fit_rmse * 255:.1f} RGB levels over {model.sample_count} swatches")
    print("black renders as:", model.predict((0, 0, 0)))
    print("white renders as:", model.predict((255, 255, 255)))

    profile.metadata["color_correction"] = model.to_dict()
    store.save(profile)
    (output / "model.json").write_text(json.dumps(model.to_dict(), indent=2))
    print("stored color correction in profile:", profile.name)


if __name__ == "__main__":
    main()
