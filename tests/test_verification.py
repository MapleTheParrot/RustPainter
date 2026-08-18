from __future__ import annotations

import os

import numpy as np
from PIL import Image

from app.input_controller import MockInputController
from app.models import ColorGroup, PaintPlan, ScreenRect, Stroke
from app.painter import Painter, PainterSettings, PainterState
from app.profiles import CalibrationProfile
from app.verification import (
    mismatched_cells,
    plan_expectations,
    sample_cell_colors,
    touch_up_plan,
)

_TIMEOUT_SCALE = float(os.environ.get("RUST_PAINTER_TEST_TIMEOUT_SCALE", "1"))


RED = (200, 30, 30)
BLUE = (30, 60, 200)


def _two_band_plan() -> PaintPlan:
    return PaintPlan(
        8,
        6,
        (
            ColorGroup(RED, tuple(Stroke(0, y, 7, y) for y in range(3)), 24),
            ColorGroup(BLUE, tuple(Stroke(0, y, 7, y) for y in range(3, 6)), 24),
        ),
    )


def _capture_for(indices: np.ndarray, palette: np.ndarray, cell: int = 20) -> np.ndarray:
    height, width = indices.shape
    capture = np.zeros((height * cell, width * cell, 3), dtype=np.float32)
    for y in range(height):
        for x in range(width):
            index = indices[y, x]
            color = palette[index] if index >= 0 else (0, 0, 0)
            capture[y * cell : (y + 1) * cell, x * cell : (x + 1) * cell] = color
    return capture


def test_expectations_replay_painting_order_and_band_coverage() -> None:
    plan = PaintPlan(
        9,
        9,
        (
            # A 3-cell band centered on row 4 covers rows 3-5.
            ColorGroup(RED, (Stroke(0, 4, 8, 4),), 27, brush_diameter=3),
            # A later single-cell stroke wins its cells back.
            ColorGroup(BLUE, (Stroke(2, 4, 5, 4),), 4),
        ),
    )
    indices, palette = plan_expectations(plan)
    assert [tuple(color) for color in palette] == [RED, BLUE]
    assert (indices[3, :] == 0).all() and (indices[5, :] == 0).all()
    assert (indices[4, 2:6] == 1).all()
    assert indices[4, 0] == 0 and indices[4, 8] == 0
    assert (indices[0, :] == -1).all()


def test_sampling_reads_cell_centers_not_borders() -> None:
    plan = _two_band_plan()
    indices, palette = plan_expectations(plan)
    capture = _capture_for(indices, palette)
    sampled = sample_cell_colors(capture, plan.width, plan.height)
    assert np.allclose(sampled[0, 0], RED)
    assert np.allclose(sampled[5, 7], BLUE)


def test_a_global_lighting_shift_is_never_a_mismatch() -> None:
    plan = _two_band_plan()
    indices, palette = plan_expectations(plan)
    capture = _capture_for(indices, palette) * 0.7 + 20.0  # darker, lifted sign
    sampled = sample_cell_colors(capture, plan.width, plan.height)
    assert not mismatched_cells(sampled, indices, palette).any()


def test_cells_painted_the_wrong_plan_color_are_flagged_and_repainted() -> None:
    plan = _two_band_plan()
    indices, palette = plan_expectations(plan)
    wrong = indices.copy()
    wrong[1, 2] = 1  # a red cell that came out blue
    wrong[1, 3] = 1
    wrong[4, 6] = 0  # a blue cell that came out red
    capture = _capture_for(wrong, palette)
    sampled = sample_cell_colors(capture, plan.width, plan.height)
    mismatch = mismatched_cells(sampled, indices, palette)
    assert mismatch.sum() == 3
    assert mismatch[1, 2] and mismatch[1, 3] and mismatch[4, 6]

    touch = touch_up_plan(mismatch, indices, palette)
    painted = {
        (group.color, stroke.start_x, stroke.start_y, stroke.end_x, stroke.end_y)
        for group in touch.color_groups
        for stroke in group.strokes
    }
    # Adjacent wrong cells merge into one stroke; strokes never leave the
    # mismatched cells.
    assert painted == {
        (RED, 2, 1, 3, 1),
        (BLUE, 6, 4, 6, 4),
    }
    assert sum(group.pixel_count for group in touch.color_groups) == 3


def test_uncovered_cells_are_ignored() -> None:
    plan = PaintPlan(8, 6, (ColorGroup(RED, (Stroke(0, 0, 7, 0),), 8),))
    indices, palette = plan_expectations(plan)
    capture = np.zeros((120, 160, 3), dtype=np.float32)  # everything black
    capture[0:20] = RED
    sampled = sample_cell_colors(capture, plan.width, plan.height)
    assert not mismatched_cells(sampled, indices, palette).any()


class _RealishInputController(MockInputController):
    """A mock that claims to emit real input, so the verify pass runs."""

    emits_real_input = True


def _held_travel(controller: MockInputController) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    """Start and end of every press, so a dab can be told from a drag."""

    spans: list[tuple[tuple[int, int], tuple[int, int]]] = []
    position = (0, 0)
    start: tuple[int, int] | None = None
    for event in controller.events:
        if event.kind == "move":
            position = (event.x, event.y)
        elif event.kind == "mouse_down":
            start = position
        elif event.kind == "mouse_up" and start is not None:
            spans.append((start, position))
            start = None
    return spans


def test_verification_repaints_exactly_the_cells_that_came_out_wrong() -> None:
    # The painted sign is captured after the job; three cells read decisively
    # as the *other* plan color and must be repainted - two adjacent ones as a
    # single stroke, the lone one as a dab. Nothing else may be touched up.
    red, blue = (200, 30, 30), (30, 60, 200)
    canvas = ScreenRect(200, 150, 1400, 1100)
    profile = CalibrationProfile.new(
        "Verified",
        canvas=canvas,
        color_box=ScreenRect(1700, 150, 200, 200),
        hue_bar=ScreenRect(1950, 150, 20, 200),
    )
    corrupted = {(5, 5), (6, 5), (40, 40)}  # logical (x, y)

    def capture(rect):
        if (rect.width, rect.height) != (canvas.width, canvas.height):
            return Image.new("RGB", (rect.width, rect.height), (120, 120, 120))
        pixels = np.zeros((rect.height, rect.width, 3), dtype=np.uint8)
        for y in range(55):
            for x in range(70):
                expected = red if y < 27 else blue
                if (x, y) in corrupted:
                    expected = blue if expected == red else red
                pixels[y * 20 : (y + 1) * 20, x * 20 : (x + 1) * 20] = expected
        return Image.fromarray(pixels, "RGB")

    controller = _RealishInputController()
    painter = Painter(controller, screen_capture=capture)
    plan = PaintPlan(
        70,
        55,
        (
            ColorGroup(red, tuple(Stroke(0, y, 69, y) for y in range(27)), 27 * 70),
            ColorGroup(blue, tuple(Stroke(0, y, 69, y) for y in range(27, 55)), 28 * 70),
        ),
    )
    settings = PainterSettings(
        countdown_seconds=0.0,
        mouse_down_duration_seconds=0.0,
        delay_after_hue_seconds=0.0,
        delay_after_saturation_value_seconds=0.0,
        delay_between_strokes_seconds=0.0,
        delay_between_colors_seconds=0.0,
        delay_after_brush_seconds=0.0,
        stroke_speed_pixels_per_second=1_000_000.0,
        stroke_interpolation_step_pixels=4096.0,
        corner_abort_enabled=False,
        progress_callback_interval_seconds=0.0,
        safety_poll_interval_seconds=0.002,
    )

    assert painter.start(plan, profile, settings)
    assert painter.wait(90.0 * _TIMEOUT_SCALE)

    assert painter.state is PainterState.COMPLETED
    spans = [
        (start, end)
        for start, end in _held_travel(controller)
        if canvas.left <= start[0] < canvas.left + canvas.width
        and canvas.top <= start[1] < canvas.top + canvas.height
    ]
    center = lambda x, y: (canvas.left + x * 20 + 10, canvas.top + y * 20 + 10)
    # The two adjacent wrong cells merge into one touch-up drag.
    assert ((center(5, 5), center(6, 5)) in spans) or (
        (center(6, 5), center(5, 5)) in spans
    )
    # The lone wrong cell is repainted as a dab.
    assert (center(40, 40), center(40, 40)) in spans
    # The main plan painted 55 row strokes; only two touch-up strokes follow.
    assert len(spans) == 57


def test_lighting_normalization_recovers_a_globally_shifted_capture() -> None:
    """A lit sign compresses and tints everything; verification must see past it.

    Live testing: a painting that matched its plan perfectly classified 76%
    wrong because the material squeezed the dark palette together.  One global
    transform, fitted from the capture itself, absorbs exactly that.
    """

    from app.verification import normalize_capture_lighting

    rng = np.random.default_rng(7)
    # As tight as the real sign's palette: dark purples a few units apart.
    palette = np.array(
        [[20, 10, 40], [32, 22, 58], [45, 30, 90], [26, 40, 52]], dtype=np.uint8
    )
    indices = rng.integers(0, len(palette), (30, 40)).astype(np.int32)
    truth = palette[indices].astype(np.float32)
    # The material response: crush the darks toward a warm ambient and mix
    # channels, which reorders which palette entry each cell sits closest to.
    captured = truth * 0.3 + np.array([95.0, 78.0, 70.0])
    captured[..., 0] += truth[..., 2] * 0.35
    captured[..., 2] -= truth[..., 1] * 0.2

    raw_wrong = mismatched_cells(captured, indices, palette).sum()
    corrected = normalize_capture_lighting(captured, indices, palette)
    fixed_wrong = mismatched_cells(corrected, indices, palette).sum()

    assert raw_wrong > indices.size * 0.3  # the shift really broke classification
    assert fixed_wrong == 0


def test_lighting_normalization_cannot_hide_genuinely_wrong_cells() -> None:
    """Twelve parameters cannot bend individual cells onto their targets."""

    from app.verification import normalize_capture_lighting

    rng = np.random.default_rng(11)
    palette = np.array(
        [[20, 10, 40], [45, 30, 90], [80, 60, 150], [200, 190, 210]], dtype=np.uint8
    )
    indices = rng.integers(0, len(palette), (30, 40)).astype(np.int32)
    truth = palette[indices].astype(np.float32)
    captured = truth * 0.55 + np.array([60.0, 45.0, 40.0])
    # A block of cells was painted the wrong color entirely.
    wrong_block = np.zeros(indices.shape, dtype=np.bool_)
    wrong_block[5:10, 5:15] = True
    swapped = palette[(indices + 2) % len(palette)].astype(np.float32)
    captured[wrong_block] = swapped[wrong_block] * 0.55 + np.array([60.0, 45.0, 40.0])

    corrected = normalize_capture_lighting(captured, indices, palette)
    mismatch = mismatched_cells(corrected, indices, palette)

    inside = mismatch[wrong_block].mean()
    outside = mismatch[~wrong_block].mean()
    assert inside > 0.9  # the wrong block is still caught
    assert outside < 0.05  # the correct cells are not dragged down with it


def test_lighting_normalization_leaves_tiny_captures_untouched() -> None:
    from app.verification import normalize_capture_lighting

    palette = np.array([[10, 10, 10], [200, 200, 200]], dtype=np.uint8)
    indices = np.zeros((2, 3), dtype=np.int32)
    sampled = np.full((2, 3, 3), 90.0, dtype=np.float32)

    corrected = normalize_capture_lighting(sampled, indices, palette)

    assert np.array_equal(corrected, sampled)
