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
        # The capture stub never changes, so a second pass would repaint the
        # same three cells again; one pass is what this test is counting.
        verify_passes=1,
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

    from app.verification import classify_cells

    raw_wrong = mismatched_cells(captured, indices, palette).sum()
    corrected = normalize_capture_lighting(captured, indices, palette)
    fixed_wrong = mismatched_cells(corrected, indices, palette).sum()

    assert raw_wrong > indices.size * 0.3  # the shift really broke classification
    # Four colors pin down a per-channel response, which takes most of the
    # damage out of the plain comparison; the painting path, which also
    # reads each color's rendering off the sign, sees nothing wrong at all.
    assert fixed_wrong < raw_wrong * 0.25
    assert classify_cells(captured, indices, palette).count == 0


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

    from app.verification import classify_cells

    mismatch = classify_cells(captured, indices, palette).cells

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


# --- classify_cells: holes, twins, and what can safely be recolored ---------

WOOD = (186, 172, 156)  # the bare sign, as captured on a real run
DARK = (40, 40, 40)
NAVY = (30, 60, 200)


def _indices(rows: int, cols: int, *, uncovered_rows: int = 0) -> np.ndarray:
    indices = np.zeros((rows, cols), dtype=np.int32)
    indices[: rows // 2] = 0
    indices[rows // 2 :] = 1
    if uncovered_rows:
        indices[:uncovered_rows] = -1
    return indices


def _render(indices: np.ndarray, palette: np.ndarray, *, holes=(), uncovered=WOOD) -> np.ndarray:
    """Per-cell RGB samples of a sign painted exactly to plan, bar the holes."""

    sampled = np.zeros((*indices.shape, 3), dtype=np.float32)
    for index, color in enumerate(palette):
        sampled[indices == index] = color
    sampled[indices < 0] = uncovered
    for y, x in holes:
        sampled[y, x] = WOOD
    return sampled


def test_a_hole_is_found_against_the_bare_sign_the_plan_left_unpainted() -> None:
    """Bare wood under a dark cell is nearer that cell's own color than any
    other palette entry, so the old two-way comparison never flagged it."""

    from app.verification import classify_cells

    palette = np.array([DARK, NAVY], dtype=np.uint8)
    indices = _indices(20, 30, uncovered_rows=4)
    holes = {(6, 3), (6, 4), (6, 5), (15, 20)}
    sampled = _render(indices, palette, holes=holes)

    # The old two-way comparison: bare wood under a dark cell is nearer dark
    # than navy, so the holes in the dark half were invisible to it.
    old = mismatched_cells(sampled, indices, palette)
    assert not old[6].any()
    verdict = classify_cells(sampled, indices, palette)
    assert verdict.blank == len(holes)
    assert verdict.wrong_color == 0
    assert {tuple(cell) for cell in np.argwhere(verdict.cells)} == holes


def test_a_hole_is_found_against_the_capture_of_the_cleared_sign() -> None:
    from app.verification import classify_cells

    palette = np.array([DARK, NAVY], dtype=np.uint8)
    indices = _indices(20, 30)  # the plan covers the whole sign
    holes = {(2, 7), (2, 8), (17, 1)}
    sampled = _render(indices, palette, holes=holes)
    # The cleared sign was captured under different lighting.
    bare = np.full((*indices.shape, 3), WOOD, dtype=np.float32) * 0.8

    verdict = classify_cells(sampled, indices, palette, bare_sampled=bare)
    assert {tuple(cell) for cell in np.argwhere(verdict.cells)} == holes
    assert verdict.blank == len(holes)


def test_a_hole_with_no_bare_reference_is_still_repainted_as_unexplained() -> None:
    from app.verification import classify_cells

    palette = np.array([DARK, NAVY], dtype=np.uint8)
    indices = _indices(20, 30)
    holes = {(3, 3), (12, 12)}
    sampled = _render(indices, palette, holes=holes)

    verdict = classify_cells(sampled, indices, palette)
    assert verdict.blank == 0
    assert verdict.unexplained == len(holes)
    assert {tuple(cell) for cell in np.argwhere(verdict.cells)} == holes


def test_twin_colors_the_sign_renders_alike_are_never_confused() -> None:
    """The real false alarms: (234,234,234) beside (223,213,209), both coming
    back as one warm off-white, and a quarter of the sign 'repainted'."""

    from app.verification import classify_cells

    palette = np.array([(234, 234, 234), (223, 213, 209)], dtype=np.uint8)
    rng = np.random.default_rng(3)
    indices = rng.integers(0, 2, (30, 40)).astype(np.int32)
    # The material renders both as the same color, give or take grain.
    sampled = np.full((*indices.shape, 3), (226, 220, 214), dtype=np.float32)
    sampled += rng.normal(0.0, 1.5, sampled.shape).astype(np.float32)

    verdict = classify_cells(sampled, indices, palette)
    assert verdict.count == 0


def test_a_whole_group_painted_the_wrong_color_is_still_caught() -> None:
    """Reading each color's rendering off the sign must not let a missed
    picker click declare itself correct."""

    from app.verification import classify_cells

    red, blue, green = (200, 30, 30), (30, 60, 200), (30, 180, 60)
    palette = np.array([red, blue, green], dtype=np.uint8)
    indices = np.zeros((12, 12), dtype=np.int32)
    indices[4:8] = 1
    indices[8:] = 2
    sampled = _render(indices, palette)
    sampled[indices == 0] = green  # every red cell came out green

    verdict = classify_cells(sampled, indices, palette)
    assert verdict.wrong_color == int((indices == 0).sum())
    assert np.array_equal(verdict.cells, indices == 0)


def test_holes_only_mode_leaves_wrong_colors_alone() -> None:
    """A brush wider than a cell cannot recolor one cell without smearing
    its neighbours, so a plan that fine only gets its holes filled."""

    from app.verification import classify_cells

    red, blue, green = (200, 30, 30), (30, 60, 200), (30, 180, 60)
    palette = np.array([red, blue, green], dtype=np.uint8)
    indices = np.zeros((12, 12), dtype=np.int32)
    indices[4:8] = 1
    indices[8:] = 2
    sampled = _render(indices, palette, holes={(5, 5)})
    sampled[0, :] = green  # one row of red came out green

    verdict = classify_cells(sampled, indices, palette, recolor=False)
    assert verdict.wrong_color == 12
    assert verdict.blank + verdict.unexplained == 1
    assert {tuple(cell) for cell in np.argwhere(verdict.cells)} == {(5, 5)}


def test_cells_under_three_pixels_are_read_from_their_centre_pixel() -> None:
    """A 3x3 median over two-pixel cells reads the neighbours, not the cell."""

    capture = np.zeros((4, 8, 3), dtype=np.float32)
    capture[:, 0::4] = capture[:, 1::4] = (255, 0, 0)
    capture[:, 2::4] = capture[:, 3::4] = (0, 0, 255)
    sampled = sample_cell_colors(capture, 4, 2)
    assert sampled.shape == (2, 4, 3)
    assert tuple(sampled[0, 0]) == (255, 0, 0)
    assert tuple(sampled[0, 1]) == (0, 0, 255)
    assert tuple(sampled[1, 2]) == (255, 0, 0)


def test_scattered_wrong_colors_at_scale_are_capture_noise_not_misses() -> None:
    """Nothing in the painting loop miscolors one cell in four at random
    through otherwise-right colors; a reading that says so is the capture
    failing to resolve cells, and acting on it is a second painting."""

    from app.verification import (
        SCATTERED_WRONG_COLOR_MIN_CELLS,
        classify_cells,
    )

    red, blue, green = (200, 30, 30), (30, 60, 200), (30, 180, 60)
    palette = np.array([red, blue, green], dtype=np.uint8)
    indices = np.zeros((60, 60), dtype=np.int32)
    indices[20:40] = 1
    indices[40:] = 2
    sampled = _render(indices, palette, holes={(3, 3), (50, 7)})
    # Every fourth cell of every color reads as one of the other colors.
    rng = np.random.default_rng(7)
    scattered = rng.random(indices.shape) < 0.25
    others = {0: green, 1: red, 2: blue}
    for index, color in others.items():
        sampled[scattered & (indices == index)] = color
    assert int(scattered.sum()) > SCATTERED_WRONG_COLOR_MIN_CELLS

    verdict = classify_cells(sampled, indices, palette)
    assert verdict.wrong_color == 0
    assert verdict.discarded == int(scattered.sum()) - int(scattered[3, 3]) - int(
        scattered[50, 7]
    )
    # The holes are still filled.
    assert verdict.count == 2
    assert verdict.cells[3, 3] and verdict.cells[50, 7]


def test_a_wrong_whole_color_is_kept_even_when_other_colors_are_noisy() -> None:
    """A missed picker click paints a whole color wrong; that is repainted
    whole even when the capture is also sprinkling noise through the rest."""

    from app.verification import classify_cells

    red, blue, green = (200, 30, 30), (30, 60, 200), (30, 180, 60)
    palette = np.array([red, blue, green], dtype=np.uint8)
    indices = np.zeros((60, 60), dtype=np.int32)
    indices[20:40] = 1
    indices[40:] = 2
    sampled = _render(indices, palette)
    sampled[indices == 0] = green  # all of red came out green
    rng = np.random.default_rng(11)
    noisy = (rng.random(indices.shape) < 0.3) & (indices > 0)
    sampled[noisy & (indices == 1)] = red
    sampled[noisy & (indices == 2)] = blue

    verdict = classify_cells(sampled, indices, palette)
    assert verdict.wrong_color == int((indices == 0).sum())
    assert np.array_equal(verdict.cells, indices == 0)
    assert verdict.discarded == int(noisy.sum())


def test_a_few_scattered_wrong_cells_are_still_repainted() -> None:
    """A stroke the game placed a cell off leaves a handful of wrong cells in
    a right color; below the floor they are cheap to fix and are fixed."""

    from app.verification import classify_cells

    red, blue = (200, 30, 30), (30, 60, 200)
    palette = np.array([red, blue], dtype=np.uint8)
    indices = np.zeros((40, 40), dtype=np.int32)
    indices[20:] = 1
    sampled = _render(indices, palette)
    for x in range(8):
        sampled[19, x] = blue

    verdict = classify_cells(sampled, indices, palette)
    assert verdict.wrong_color == 8
    assert verdict.discarded == 0
    assert verdict.count == 8
