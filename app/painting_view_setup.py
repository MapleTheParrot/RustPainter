"""Confidence-gated fitting for Rust's sign painting view."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import time

from PIL import Image

from .input_controller import InputController
from .models import ScreenRect
from .setup_detection import SetupDetection


WHEEL_NOTCH = 120


@dataclass(frozen=True, slots=True)
class PaintingViewCapture:
    image: Image.Image
    screen: ScreenRect
    detection: SetupDetection


@dataclass(frozen=True, slots=True)
class CanvasFitResult:
    capture: PaintingViewCapture
    zoom_steps: int
    reason: str

    @property
    def auto_zoomed(self) -> bool:
        return self.zoom_steps != 0


def _inside(inner: ScreenRect, outer: ScreenRect, margin: int = 0) -> bool:
    return (
        inner.left >= outer.left + margin
        and inner.top >= outer.top + margin
        and inner.right <= outer.right - margin
        and inner.bottom <= outer.bottom - margin
    )


def _overlap(first: ScreenRect, second: ScreenRect) -> bool:
    return not (
        first.right <= second.left
        or second.right <= first.left
        or first.bottom <= second.top
        or second.bottom <= first.top
    )


def layout_is_safe(
    capture: PaintingViewCapture,
    *,
    expected_aspect: float | None = None,
    edge_margin: int = 8,
) -> tuple[bool, str]:
    """Say whether a complete detected layout is safe to enlarge and save."""

    detection = capture.detection
    missing = detection.missing_required
    if missing:
        return False, "missing " + ", ".join(missing)
    canvas = detection.regions["canvas"]
    if canvas.confidence < 0.60:
        return False, "canvas confidence is too low"
    if detection.regions["hue_bar"].confidence < 0.85:
        return False, "hue bar confidence is too low"
    if detection.regions["color_box"].confidence < 0.80:
        return False, "colour box confidence is too low"
    if not _inside(canvas.rect, capture.screen, edge_margin):
        return False, "canvas is too close to a screen edge"
    if expected_aspect and (
        abs(canvas.rect.aspect_ratio - expected_aspect) > expected_aspect * 0.08
    ):
        return False, "canvas aspect ratio changed"

    for name, region in detection.regions.items():
        if name != "canvas" and not _inside(region.rect, capture.screen):
            return False, f"{name} is outside the screen"
        if name != "canvas" and _overlap(canvas.rect, region.rect):
            return False, f"canvas overlaps {name}"
    return True, "complete layout detected"


def fit_canvas_view(
    controller: InputController,
    observe: Callable[[], PaintingViewCapture],
    initial: PaintingViewCapture,
    *,
    expected_aspect: float | None = None,
    max_steps: int = 12,
    settle: Callable[[float], None] = time.sleep,
    checkpoint: Callable[[], None] | None = None,
) -> CanvasFitResult:
    """Zoom to the largest safe detected canvas, undoing the first unsafe step.

    Both wheel directions are probed because Rust and user input settings may
    disagree about which direction zooms in. No wheel input is emitted unless
    the initial full layout is confidently detected.
    """

    safe, reason = layout_is_safe(initial, expected_aspect=expected_aspect)
    if not safe or max_steps <= 0:
        return CanvasFitResult(initial, 0, reason)

    def check() -> None:
        if checkpoint is not None:
            checkpoint()

    def turn(delta: int, target: PaintingViewCapture) -> None:
        check()
        controller.move_mouse(*target.detection.regions["canvas"].rect.center)
        check()
        controller.scroll_wheel(delta)
        settle(0.45)
        check()

    baseline_canvas = initial.detection.regions["canvas"].rect
    baseline_area = baseline_canvas.width * baseline_canvas.height
    direction = 0
    current = initial
    for candidate_direction in (WHEEL_NOTCH, -WHEEL_NOTCH):
        turn(candidate_direction, initial)
        trial = observe()
        trial_safe, _ = layout_is_safe(trial, expected_aspect=expected_aspect)
        trial_canvas = trial.detection.regions.get("canvas")
        trial_area = (
            trial_canvas.rect.width * trial_canvas.rect.height if trial_canvas else 0
        )
        if trial_safe and trial_area > baseline_area * 1.01:
            direction = candidate_direction
            current = trial
            break
        turn(-candidate_direction, initial)

    if direction == 0:
        return CanvasFitResult(initial, 0, "canvas could not be enlarged safely")

    steps = 1
    while steps < max_steps:
        turn(direction, current)
        trial = observe()
        trial_safe, stop_reason = layout_is_safe(
            trial, expected_aspect=expected_aspect
        )
        current_canvas = current.detection.regions["canvas"].rect
        trial_canvas = trial.detection.regions.get("canvas")
        current_area = current_canvas.width * current_canvas.height
        trial_area = trial_canvas.rect.width * trial_canvas.rect.height if trial_canvas else 0
        if not trial_safe or trial_area <= current_area * 1.005:
            turn(-direction, current)
            reason = stop_reason if not trial_safe else "canvas stopped getting larger"
            return CanvasFitResult(current, steps * (1 if direction > 0 else -1), reason)
        current = trial
        steps += 1

    return CanvasFitResult(
        current,
        steps * (1 if direction > 0 else -1),
        "automatic zoom limit reached",
    )


__all__ = [
    "CanvasFitResult",
    "PaintingViewCapture",
    "fit_canvas_view",
    "layout_is_safe",
]
