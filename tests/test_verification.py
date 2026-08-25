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
        progress_callback_interval_seconds=0.0,
        safety_poll_interval_seconds=0.002,
        # The capture stub never changes, so a second pass would repaint the
        # same three cells again; one pass is what this test is counting.
        verify_passes=1,
        confirm_strokes=False,
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

    from app.verification import classify_cells, normalize_capture_lighting

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


def test_a_bare_sign_looks_bare_and_a_painted_one_does_not() -> None:
    """A sign before painting stands in for the cleared one only if it is one
    surface: grain, a vignette and a few specks pass, an earlier artwork
    does not.  Measured live, a bare artist canvas reads about 6 and a
    painted one 40-60 on the percentile the check uses."""

    from app.verification import capture_looks_bare

    rng = np.random.default_rng(7)
    bare = np.array([214, 204, 186], dtype=np.float32) + rng.normal(0.0, 6.0, (64, 96, 3))
    bare = np.clip(bare, 0, 255).astype(np.float32)
    assert capture_looks_bare(bare)

    specked = bare.copy()
    specked[::9, ::7] = (30, 90, 230)  # dabs left on the sign, well under a tenth
    assert capture_looks_bare(specked)

    painted = bare.copy()
    painted[:, :48] = (40, 40, 40)  # half the sign carries an earlier picture
    assert not capture_looks_bare(painted)

    assert not capture_looks_bare(bare[:2, :4])  # too few cells to say


# ------------------------------------------------ checking colors as they land


def test_plan_layers_record_what_each_final_stroke_painted_over() -> None:
    """On a gap-merged plan the first color runs under the later ones, so a
    dropped final stroke leaves that color showing, not the sign."""

    from app.verification import plan_layers

    plan = PaintPlan(
        6,
        1,
        (
            ColorGroup(RED, (Stroke(0, 0, 5, 0),), 6),
            ColorGroup(BLUE, (Stroke(2, 0, 3, 0),), 2),
            # Red again over a cell already red: its last real change stays
            # the first stroke, with bare sign under it.
            ColorGroup(RED, (Stroke(5, 0, 5, 0),), 1),
        ),
    )
    indices, underpaint, palette = plan_layers(plan)
    assert [tuple(color) for color in palette] == [RED, BLUE]
    assert indices.tolist() == [[0, 0, 1, 1, 0, 0]]
    assert underpaint.tolist() == [[-1, -1, 0, 0, -1, -1]]


def test_a_cell_reading_as_its_underpaint_is_a_hole_not_capture_noise() -> None:
    """A missed stroke on a merged plan shows the color that ran under it.
    Read against the final colors alone that is a "wrong color", which at
    scale is discarded as noise; read against the underpaint it is the hole
    it is, and is repainted."""

    from app.verification import (
        SCATTERED_WRONG_COLOR_MIN_CELLS,
        classify_cells,
        plan_layers,
    )

    black, navy, cream = (0, 0, 0), (8, 12, 40), (240, 225, 200)
    width = height = 60
    strokes_black = tuple(Stroke(0, y, width - 1, y) for y in range(height))
    strokes_navy = tuple(Stroke(0, y, width - 1, y) for y in range(10, 50))
    strokes_cream = tuple(Stroke(0, y, width - 1, y) for y in range(50, 60))
    plan = PaintPlan(
        width,
        height,
        (
            ColorGroup(black, strokes_black, width * height),
            ColorGroup(navy, strokes_navy, width * 40),
            ColorGroup(cream, strokes_cream, width * 10),
        ),
    )
    indices, underpaint, palette = plan_layers(plan)
    sampled = _render(indices, palette)
    # Every fifth navy cell and every fifth cream cell lost its stroke and
    # shows the black underneath - far more than the scattered allowance.
    rng = np.random.default_rng(3)
    lost = (rng.random(indices.shape) < 0.2) & (indices > 0)
    sampled[lost] = black
    assert int(lost.sum()) > SCATTERED_WRONG_COLOR_MIN_CELLS

    without = classify_cells(sampled, indices, palette)
    assert without.count == 0 and without.discarded == int(lost.sum())

    with_layers = classify_cells(sampled, indices, palette, underpaint=underpaint)
    assert with_layers.discarded == 0
    assert with_layers.blank == int(lost.sum())
    assert np.array_equal(with_layers.cells, lost)


def test_confirm_cells_tells_hits_from_misses_and_skips_the_invisible() -> None:
    from app.verification import MIN_CONFIRM_DISTANCE, confirm_cells

    wood = np.array(WOOD, dtype=np.float32)
    before = np.broadcast_to(wood, (1, 4, 3)).copy()
    expected = np.array(RED, dtype=np.float32)
    after = before.copy()
    after[0, 0] = RED  # took the color
    after[0, 1] = WOOD  # stayed bare
    after[0, 2] = (RED[0] - 20, RED[1] + 15, RED[2] + 10)  # near enough
    judge = np.array([[True, True, True, False]])
    hit, judged = confirm_cells(before, after, expected, judge)
    assert judged.tolist() == [[True, True, True, False]]
    assert hit.tolist() == [[True, False, True, False]]

    # A color the sign already shows within the visibility floor is not
    # judged at all: its absence would be as invisible as its presence.
    near = np.array(RED, dtype=np.float32) + 5.0
    assert np.linalg.norm(near - expected) < MIN_CONFIRM_DISTANCE
    before_near = np.broadcast_to(near, (1, 1, 3)).copy()
    _hit, judged = confirm_cells(before_near, before_near, expected, np.array([[True]]))
    assert not judged.any()


def test_confirm_cells_sees_through_the_bilinear_blend_of_a_fine_sign() -> None:
    """At under two pixels per texel the sampled pixel carries up to half of
    a neighbour.  A lost texel whose neighbours all took the color moves
    almost halfway toward it and would pass a plain "did it move" test;
    modelled as a blend it is still a miss, and a painted texel whose pixel
    leans the same way is still a hit."""

    from app.verification import CellBlend, confirm_cells

    pitch = 2.0
    width, height = 7, 7
    # Every cell's centre sits 0.45 px before its sampled pixel's centre on
    # both axes, so the pixel reads over a fifth of the next cell along
    # each and the cell's own share is not much over a half.
    centers = np.arange(width) * pitch + 0.05
    blend = CellBlend.from_centers(centers, centers, pitch, pitch)
    assert np.allclose(blend.fraction_x, 0.45 / pitch, atol=0.01)
    assert (blend.step_x == 1).all()
    own, _mixed = blend.neighbour_share(np.zeros((height, width, 3)))
    assert 0.5 < float(own[3, 3, 0]) < 0.65

    wood = np.array(WOOD, dtype=np.float64)
    red = np.array(RED, dtype=np.float64)
    texels = np.broadcast_to(wood, (height, width, 3)).copy()
    texels[:] = red  # every texel painted...
    texels[3, 3] = wood  # ...but one, lost in the middle

    def rendered(cells: np.ndarray) -> np.ndarray:
        """What the capture reads: each pixel a bilinear mix per the blend."""

        own, mixed = blend.neighbour_share(cells)
        return own * cells + mixed

    before = rendered(np.broadcast_to(wood, (height, width, 3)).copy())
    after = rendered(texels)
    # The lost texel's pixel still moved most of the way a naive "did it
    # move toward the color" threshold would ask for.
    moved = np.linalg.norm(after[3, 3] - before[3, 3])
    assert moved > 0.35 * np.linalg.norm(red - wood)

    judge = np.ones((height, width), dtype=bool)
    hit, judged = confirm_cells(before, after, red, judge, blend=blend)
    assert judged.all()
    assert not hit[3, 3]
    assert hit.sum() == height * width - 1


def test_repaint_runs_bridge_own_and_later_cells_but_never_earlier_ones() -> None:
    from app.verification import repaint_runs

    # Row: own own EARLIER own own BARE own LATER own
    indices = np.array([[1, 1, 0, 1, 1, -1, 1, 2, 1]], dtype=np.int32)
    missed = np.array([[True, False, False, True, True, False, True, False, True]])
    runs = repaint_runs(missed, indices, 1, max_gap=2)
    spans = sorted((run.start_x, run.end_x) for run in runs)
    # Cell 0 cannot reach cell 3 across the earlier color; 3-4 stand alone
    # from 6 across bare sign; 6 reaches 8 across the later color.
    assert spans == [(0, 0), (3, 4), (6, 8)]


def test_sign_rendering_fit_maps_nominal_colors_to_what_the_capture_shows() -> None:
    from app.verification import apply_capture_lighting, fit_sign_rendering

    palette = np.array(
        [(0, 0, 0), (255, 255, 255), (200, 30, 30), (30, 60, 200), (30, 180, 60)],
        dtype=np.uint8,
    )
    indices = np.repeat(np.arange(5, dtype=np.int32), 10).reshape(5, 10)
    # The sign darkens and warms everything by one affine transform.
    sampled = _render(indices, palette) * 0.8 + np.array([20, 10, 0], dtype=np.float32)
    coefficients = fit_sign_rendering(sampled, indices, palette)
    assert coefficients is not None
    predicted = apply_capture_lighting(
        np.array([[200, 30, 30]], dtype=np.float32), coefficients
    )
    assert np.allclose(predicted[0], (200 * 0.8 + 20, 30 * 0.8 + 10, 30 * 0.8), atol=1.0)
    # Too few colors to fit on: nothing, and the caller uses the nominal value.
    assert fit_sign_rendering(sampled[:3], indices[:3], palette) is None


def test_a_band_brush_is_repainted_along_its_own_planned_strokes() -> None:
    """A three-row band laid on another row would reach rows the planner
    never cleared for it, so the strokes that covered a missed cell go
    down again whole instead."""

    from app.verification import repaint_runs

    indices = np.full((7, 10), 1, dtype=np.int32)
    strokes = (Stroke(0, 1, 9, 1), Stroke(0, 4, 9, 4))  # bands over rows 0-2 and 3-5
    missed = np.zeros((7, 10), dtype=bool)
    missed[5, 3] = True  # in the second band's bottom row
    again = repaint_runs(missed, indices, 1, strokes=strokes, radius=1)
    assert again == [Stroke(0, 4, 9, 4)]
    assert repaint_runs(np.zeros((7, 10), dtype=bool), indices, 1, strokes=strokes, radius=1) == []


def test_confirm_cells_learns_the_sign_rendering_from_the_cells_that_took() -> None:
    """Nominal dark red over a black underpaint: the sign renders the red
    warmer and lighter than its nominal value, close enough to the
    underpaint's rendering that the nominal prediction would call the real
    hits misses.  Enough cells that plainly took it redefine the expected
    color and the close calls are decided right."""

    from app.verification import confirm_cells

    underpaint = np.array((35, 30, 25), dtype=np.float32)  # black, as rendered
    nominal = np.array((48, 6, 7), dtype=np.float32)
    rendered = np.array((60, 28, 20), dtype=np.float32)  # what the sign shows
    # Nearer the underpaint than the nominal: the naive call would be "miss".
    assert np.linalg.norm(rendered - underpaint) < np.linalg.norm(rendered - nominal)
    before = np.broadcast_to(underpaint, (4, 10, 3)).copy()
    after = before.copy()
    took = np.zeros((4, 10), dtype=bool)
    took[:3] = True  # thirty cells took the color, ten did not
    after[took] = rendered
    judge = np.ones((4, 10), dtype=bool)
    hit, judged = confirm_cells(before, after, nominal, judge)
    assert judged.all()
    assert np.array_equal(hit, took)


def test_a_stored_bare_colour_finds_holes_a_covered_plan_hides() -> None:
    """A plan that paints every cell leaves no wood to read, so a job that
    never saw the sign cleared cannot tell a hole from a wrong colour - and
    on a sign too fine to recolour, wrong colours are left alone, so the
    holes go unrepaired.  One remembered colour restores the blank test."""

    import numpy as np

    from app.verification import classify_cells

    rows, cols = 24, 24
    palette = np.array(
        [[20, 20, 20], [200, 40, 40], [190, 175, 159]], dtype=np.float32
    )
    # Every cell is painted (no -1), and one palette colour happens to sit
    # near the wood - exactly the case that defeats colour alone.
    indices = np.zeros((rows, cols), dtype=np.int64)
    indices[:, 12:] = 1
    indices[0, :] = 2
    sampled = palette[indices].astype(np.float32)
    holes = [(5, 3), (5, 4), (9, 17), (14, 20), (18, 7)]
    wood = np.array([190, 175, 159], dtype=np.float32)
    for y, x in holes:
        sampled[y, x] = wood

    blind = classify_cells(sampled, indices, palette, bare_sampled=None, recolor=False)
    seeing = classify_cells(
        sampled,
        indices,
        palette,
        bare_sampled=np.full((rows, cols, 3), wood),
        recolor=False,
    )
    assert blind.blank == 0
    assert seeing.blank == len(holes)
    for y, x in holes:
        assert seeing.cells[y, x]
    # Nothing else is disturbed: the row genuinely painted the wood colour
    # is not read as a hole, because it is not far from its own colour.
    assert not seeing.cells[0].any()
    assert int(seeing.cells.sum()) == len(holes)


def test_classify_export_reads_blank_and_wrong_texels_exactly() -> None:
    import numpy as np

    from app.verification import classify_export

    palette = np.array([[20, 20, 20], [200, 40, 40]], dtype=np.float32)
    indices = np.zeros((6, 8), dtype=np.int64); indices[:, 4:] = 1
    rgb = palette[indices].astype(np.float32)
    painted = np.ones((6, 8), dtype=bool)
    painted[2, 1] = False                 # never painted
    rgb[3, 6] = (20, 20, 20)              # a red cell painted black
    rgb[:, 0] += 5                        # the picker's small deviation is not a mistake
    verdict = classify_export(rgb, painted, indices, palette)
    assert verdict.blank == 1 and verdict.wrong_color == 1 and verdict.discarded == 0
    assert verdict.cells[2, 1] and verdict.cells[3, 6]
    assert int(verdict.cells.sum()) == 2
