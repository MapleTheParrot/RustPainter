"""Measured correction for colors rendered through Rust sign materials."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Iterable, Mapping, Sequence, TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image


CHART_COLUMNS = 8
CHART_ROWS = 4
COLOR_CORRECTION_SCHEMA = 1


def calibration_chart_colors() -> tuple[tuple[int, int, int], ...]:
    """Return 32 stable samples spanning grayscale, hue, saturation, and value."""

    colors: list[tuple[int, int, int]] = [
        (level, level, level) for level in (0, 36, 73, 109, 146, 182, 219, 255)
    ]
    for value in (0.52, 1.0):
        for saturation in (0.55, 1.0):
            for hue_degrees in (0, 60, 120, 180, 240, 300):
                rgb = colorsys.hsv_to_rgb(hue_degrees / 360.0, saturation, value)
                colors.append(tuple(round(channel * 255) for channel in rgb))
    # A fixed permutation distributes grays, darks, brights, and hues across
    # the whole sign so a plank or lighting gradient is not mistaken for a
    # channel-response difference.
    return tuple(colors[(index * 13) % len(colors)] for index in range(len(colors)))


def build_calibration_chart(
    *, width: int = 800, height: int = 400
) -> "Image":
    """Create the 8x4 swatch image used by the normal paint pipeline."""

    if width < CHART_COLUMNS or height < CHART_ROWS:
        raise ValueError("Calibration chart dimensions are too small")
    from PIL import Image as PillowImage
    from PIL import ImageDraw

    image = PillowImage.new("RGB", (width, height))
    draw = ImageDraw.Draw(image)
    for index, color in enumerate(calibration_chart_colors()):
        column = index % CHART_COLUMNS
        row = index // CHART_COLUMNS
        left = round(column * width / CHART_COLUMNS)
        right = round((column + 1) * width / CHART_COLUMNS)
        top = round(row * height / CHART_ROWS)
        bottom = round((row + 1) * height / CHART_ROWS)
        draw.rectangle((left, top, right - 1, bottom - 1), fill=color)
    return image


def sample_painted_chart(
    capture: "Image",
    commanded_image: "Image",
) -> tuple[tuple[tuple[int, int, int], ...], tuple[tuple[int, int, int], ...]]:
    """Sample matching interiors from a painted canvas and its logical image."""

    screen = np.asarray(capture.convert("RGB"), dtype=np.uint8)
    commands = np.asarray(commanded_image.convert("RGB"), dtype=np.uint8)
    screen_height, screen_width = screen.shape[:2]
    command_height, command_width = commands.shape[:2]
    if screen_width < CHART_COLUMNS * 4 or screen_height < CHART_ROWS * 4:
        raise ValueError("Captured sign canvas is too small for color calibration")

    requested: list[tuple[int, int, int]] = []
    observed: list[tuple[int, int, int]] = []
    for row in range(CHART_ROWS):
        for column in range(CHART_COLUMNS):
            command_x = min(
                command_width - 1,
                int((column + 0.5) * command_width / CHART_COLUMNS),
            )
            command_y = min(
                command_height - 1,
                int((row + 0.5) * command_height / CHART_ROWS),
            )
            requested.append(tuple(int(value) for value in commands[command_y, command_x]))

            # Use only the central half of each large painted swatch. This
            # rejects brush edges, neighboring overlap, and most plank seams.
            tile_left = column * screen_width / CHART_COLUMNS
            tile_right = (column + 1) * screen_width / CHART_COLUMNS
            tile_top = row * screen_height / CHART_ROWS
            tile_bottom = (row + 1) * screen_height / CHART_ROWS
            left = max(0, round(tile_left + (tile_right - tile_left) * 0.25))
            right = min(
                screen_width,
                round(tile_right - (tile_right - tile_left) * 0.25),
            )
            top = max(0, round(tile_top + (tile_bottom - tile_top) * 0.25))
            bottom = min(
                screen_height,
                round(tile_bottom - (tile_bottom - tile_top) * 0.25),
            )
            region = screen[top:bottom, left:right].reshape(-1, 3)
            if not region.size:
                raise ValueError("A calibration swatch capture region was empty")
            observed.append(
                tuple(int(round(value)) for value in np.median(region, axis=0))
            )
    return tuple(requested), tuple(observed)


@dataclass(frozen=True, slots=True)
class ColorCorrectionModel:
    """Affine forward model of picker RGB to observed sign RGB."""

    forward_matrix: tuple[tuple[float, float, float, float], ...]
    fit_rmse: float
    sample_count: int
    captured_at: str

    def __post_init__(self) -> None:
        matrix = np.asarray(self.forward_matrix, dtype=np.float64)
        if matrix.shape != (3, 4) or not np.isfinite(matrix).all():
            raise ValueError("Color correction matrix must be a finite 3x4 matrix")
        if not np.isfinite(self.fit_rmse) or self.fit_rmse < 0:
            raise ValueError("Color correction error must be finite and non-negative")
        if self.sample_count < 4:
            raise ValueError("Color correction needs at least four samples")

    def predict(self, picker_rgb: Sequence[int | float]) -> tuple[int, int, int]:
        command = np.asarray(tuple(picker_rgb), dtype=np.float64) / 255.0
        if command.shape != (3,):
            raise ValueError("RGB color must contain three channels")
        matrix = np.asarray(self.forward_matrix, dtype=np.float64)
        predicted = matrix[:, :3] @ command + matrix[:, 3]
        return tuple(int(value) for value in np.rint(np.clip(predicted, 0.0, 1.0) * 255))

    def correct(self, desired_rgb: Sequence[int | float]) -> tuple[int, int, int]:
        """Invert the measured material response for a desired visible color."""

        desired = np.asarray(tuple(desired_rgb), dtype=np.float64) / 255.0
        if desired.shape != (3,):
            raise ValueError("RGB color must contain three channels")
        matrix = np.asarray(self.forward_matrix, dtype=np.float64)
        command = np.linalg.pinv(matrix[:, :3]) @ (desired - matrix[:, 3])
        return tuple(int(value) for value in np.rint(np.clip(command, 0.0, 1.0) * 255))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": COLOR_CORRECTION_SCHEMA,
            "forwardMatrix": [list(row) for row in self.forward_matrix],
            "fitRmse": self.fit_rmse,
            "sampleCount": self.sample_count,
            "capturedAt": self.captured_at,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ColorCorrectionModel":
        matrix = value.get("forwardMatrix", value.get("forward_matrix"))
        if not isinstance(matrix, Sequence) or isinstance(matrix, (str, bytes)):
            raise ValueError("Color correction forwardMatrix is missing")
        rows = tuple(tuple(float(item) for item in row) for row in matrix)
        return cls(
            forward_matrix=rows,
            fit_rmse=float(value.get("fitRmse", value.get("fit_rmse", 0.0))),
            sample_count=int(value.get("sampleCount", value.get("sample_count", 0))),
            captured_at=str(value.get("capturedAt", value.get("captured_at", ""))),
        )


def fit_color_correction(
    commanded: Iterable[Sequence[int | float]],
    observed: Iterable[Sequence[int | float]],
    *,
    regularization: float = 0.04,
) -> ColorCorrectionModel:
    """Fit a stable affine material-response model from painted swatches."""

    commands = np.asarray(tuple(tuple(color) for color in commanded), dtype=np.float64)
    results = np.asarray(tuple(tuple(color) for color in observed), dtype=np.float64)
    if commands.shape != results.shape or commands.ndim != 2 or commands.shape[1] != 3:
        raise ValueError("Commanded and observed samples must be matching RGB arrays")
    if len(commands) < 8:
        raise ValueError("At least eight color samples are required")
    if not np.isfinite(commands).all() or not np.isfinite(results).all():
        raise ValueError("Color samples must be finite")
    commands = np.clip(commands / 255.0, 0.0, 1.0)
    results = np.clip(results / 255.0, 0.0, 1.0)
    if float(np.mean(np.std(results, axis=0))) < 0.08:
        raise ValueError(
            "The captured chart has too little color variation. Confirm the completed "
            "chart is visible and the canvas calibration is correct."
        )

    design = np.column_stack((commands, np.ones(len(commands))))
    penalty = np.diag((regularization, regularization, regularization, 0.0))
    target = design.T @ results + penalty @ np.vstack((np.eye(3), np.zeros((1, 3))))
    coefficients = np.linalg.solve(design.T @ design + penalty, target)
    predicted = design @ coefficients
    rmse = float(np.sqrt(np.mean((predicted - results) ** 2)))
    forward = coefficients.T
    if np.linalg.matrix_rank(forward[:, :3]) < 3:
        raise ValueError("The captured color response cannot be inverted reliably")
    if float(np.linalg.cond(forward[:, :3])) > 30.0:
        raise ValueError("The captured color response is too unstable to invert reliably")
    if rmse > 0.22:
        raise ValueError(
            "The painted chart is too inconsistent to calibrate reliably "
            f"(fit error {rmse * 255:.1f} RGB levels)."
        )
    return ColorCorrectionModel(
        forward_matrix=tuple(tuple(float(value) for value in row) for row in forward),
        fit_rmse=rmse,
        sample_count=len(commands),
        captured_at=datetime.now(timezone.utc).isoformat(timespec="seconds"),
    )


__all__ = [
    "CHART_COLUMNS",
    "CHART_ROWS",
    "ColorCorrectionModel",
    "build_calibration_chart",
    "calibration_chart_colors",
    "fit_color_correction",
    "sample_painted_chart",
]
