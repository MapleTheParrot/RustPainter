from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.color_calibration import (
    ColorCorrectionModel,
    build_calibration_chart,
    fit_color_correction,
    sample_painted_chart,
)


def _render_material(image: Image.Image) -> Image.Image:
    source = np.asarray(image.convert("RGB"), dtype=np.float64) / 255.0
    matrix = np.asarray(
        (
            (0.70, 0.04, 0.01),
            (0.03, 0.76, 0.02),
            (0.01, 0.06, 0.68),
        )
    )
    offset = np.asarray((0.06, 0.04, 0.08))
    rendered = source @ matrix.T + offset
    return Image.fromarray(
        np.rint(np.clip(rendered, 0.0, 1.0) * 255).astype(np.uint8),
        "RGB",
    )


def test_chart_sampling_and_affine_fit_recover_material_response() -> None:
    command_chart = build_calibration_chart(width=800, height=400)
    painted_chart = _render_material(command_chart)

    commanded, observed = sample_painted_chart(painted_chart, command_chart)
    model = fit_color_correction(commanded, observed)

    assert len(commanded) == 32
    assert model.sample_count == 32
    assert model.fit_rmse < 0.01
    command = (90, 150, 210)
    visible = model.predict(command)
    recovered = model.correct(visible)
    assert recovered == pytest.approx(command, abs=3)


def test_color_correction_round_trip_serialization() -> None:
    chart = build_calibration_chart(width=160, height=80)
    commanded, observed = sample_painted_chart(_render_material(chart), chart)
    model = fit_color_correction(commanded, observed)

    restored = ColorCorrectionModel.from_dict(model.to_dict())

    assert restored == model
    assert restored.correct((100, 80, 140)) == model.correct((100, 80, 140))


def test_fit_rejects_capture_with_no_color_variation() -> None:
    commanded = [(index * 8, 255 - index * 8, index * 4) for index in range(32)]
    observed = [(70, 70, 70)] * 32

    with pytest.raises(ValueError, match="too little color variation"):
        fit_color_correction(commanded, observed)
