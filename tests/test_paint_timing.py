from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models import ColorGroup, PaintPlan, Stroke
from app.paint_timing import (
    CHECK_CAPTURE_SECONDS_DEFAULT,
    CHECK_REPAINT_FRACTION_DEFAULT,
    DEFAULT_STROKE_OVERHEAD_SECONDS,
    LEARN_PRIOR_COLORS,
    LEARN_PRIOR_PAINT_SECONDS,
    LEARN_PRIOR_STROKES,
    MAX_STROKE_OVERHEAD_SECONDS,
    TOUCH_UP_FRACTION_DEFAULT,
    LONG_DRAG_MAX_TEXELS_PER_SECOND,
    MIN_PRESS_SECONDS,
    PACE_PRIOR_SECONDS,
    PICKER_CLICK_HOLD_SECONDS,
    SETTLE_FLOOR_SECONDS,
    SHIFT_LINE_MIN_TEXELS,
    SHIFT_LINE_MODIFIER_LEAD_SECONDS,
    SHORT_RUN_TEXELS,
    STROKE_GAP_FLOOR_SECONDS,
    TIMING_FLOORS,
    LearnedTiming,
    PlanProfile,
    PlanWorkSchedule,
    StrokeTiming,
    estimate_plan_seconds,
    fields_below_floor,
    floored,
    remaining_seconds,
    stroke_pace,
)
from app.painter import PainterSettings


def _timing(**overrides: object) -> StrokeTiming:
    values: dict[str, object] = {
        "stroke_speed_pixels_per_second": 2200.0,
        "mouse_down_duration_seconds": 0.012,
        "delay_between_strokes_seconds": 0.005,
        "delay_after_hue_seconds": 0.045,
        "delay_after_saturation_value_seconds": 0.045,
        "delay_between_colors_seconds": 0.05,
        "delay_after_brush_seconds": 0.06,
        "overhead_seconds": 0.01,
    }
    values.update(overrides)
    return StrokeTiming(**values)  # type: ignore[arg-type]


def _plan() -> PaintPlan:
    # Two colors: one with a long run and a dab, a second pass of the same
    # color (no re-selection), then a different color with a wider brush.
    red = ColorGroup((200, 0, 0), (Stroke(0, 0, 29, 0), Stroke(0, 1, 0, 1)), 31)
    red_again = ColorGroup((200, 0, 0), (Stroke(5, 2, 6, 2),), 2)
    blue = ColorGroup((0, 0, 200), (Stroke(0, 3, 1, 3),), 2, brush_diameter=3)
    return PaintPlan(30, 4, (red, red_again, blue))


def test_short_strokes_cost_the_frame_hold_not_their_length() -> None:
    timing = _timing()
    dab = timing.stroke_seconds(0)
    short = timing.stroke_seconds(10)  # 4.5 ms of travel at 2200 px/s
    # The 5 ms between strokes is run at its floor, like the 12 ms hold.
    assert dab == pytest.approx(MIN_PRESS_SECONDS + STROKE_GAP_FLOOR_SECONDS + 0.01)
    assert short == pytest.approx(dab)
    # Only a drag longer than the hold costs more than the hold.  With no
    # texel pitch to pace by, the drag runs at the set speed.
    long = timing.stroke_seconds(2200)
    assert long == pytest.approx(1.0 + STROKE_GAP_FLOOR_SECONDS + 0.01)


def test_a_long_drag_is_priced_at_the_capped_rate() -> None:
    timing = _timing()
    pitch = 3.0
    capped = LONG_DRAG_MAX_TEXELS_PER_SECOND * pitch  # 750 px/s, under 2200
    long = timing.stroke_seconds(2200, pitch)
    assert long == pytest.approx(2200 / capped + STROKE_GAP_FLOOR_SECONDS + 0.01)
    # A run inside the short-run allowance is not capped - and is under the
    # hold anyway.
    short = timing.stroke_seconds(SHORT_RUN_TEXELS * pitch, pitch)
    assert short == pytest.approx(timing.stroke_seconds(0))


def test_stroke_pace_caps_only_long_drags_on_real_input() -> None:
    pitch = 4.0

    def pace(distance: float, **overrides: object):
        values: dict[str, object] = {
            "speed_pixels_per_second": 2200.0,
            "step_pixels": 8.0,
            "texel_pitch_pixels": pitch,
        }
        values.update(overrides)
        return stroke_pace(distance, **values)  # type: ignore[arg-type]

    assert pace(0.0).move_seconds == 0.0
    run = pace(SHORT_RUN_TEXELS * pitch)
    assert run.step_pixels == 8.0
    assert run.move_seconds == pytest.approx(SHORT_RUN_TEXELS * pitch / 2200.0)
    drag = pace(400.0)
    assert drag.step_pixels == pitch
    assert drag.move_seconds == pytest.approx(
        400.0 / (LONG_DRAG_MAX_TEXELS_PER_SECOND * pitch)
    )
    # A setting slower than the cap is left alone.
    slow = pace(400.0, speed_pixels_per_second=300.0, step_pixels=2.0)
    assert slow.step_pixels == 2.0
    assert slow.move_seconds == pytest.approx(400.0 / 300.0)
    # Mock input has no game to pace for.
    mock = pace(400.0, real_input=False)
    assert mock.step_pixels == 8.0
    assert mock.move_seconds == pytest.approx(400.0 / 2200.0)
    # An unknown pitch cannot cap anything.
    assert pace(400.0, texel_pitch_pixels=float("nan")).step_pixels == 8.0


def test_every_floored_setting_is_lifted_and_named() -> None:
    turbo = PainterSettings(
        mouse_down_duration_seconds=0.012,
        delay_after_hue_seconds=0.045,
        delay_after_saturation_value_seconds=0.045,
        delay_after_brush_seconds=0.06,
        delay_between_strokes_seconds=0.005,
        delay_between_colors_seconds=0.05,
    )
    assert fields_below_floor(turbo) == tuple(TIMING_FLOORS)
    for name, floor in TIMING_FLOORS.items():
        assert floored(turbo, name) == floor
        assert floored(turbo, name, real_input=False) == getattr(turbo, name)
    relaxed = PainterSettings(
        mouse_down_duration_seconds=0.08,
        delay_after_hue_seconds=0.12,
        delay_after_saturation_value_seconds=0.12,
        delay_after_brush_seconds=0.12,
        delay_between_strokes_seconds=0.035,
        delay_between_colors_seconds=0.18,
    )
    assert fields_below_floor(relaxed) == ()
    assert floored(relaxed, "delay_after_hue_seconds") == 0.12


def test_mock_input_skips_the_holds() -> None:
    timing = _timing(real_input=False)
    assert timing.stroke_seconds(0) == pytest.approx(0.012 + 0.005 + 0.01)
    assert timing.color_change_seconds() == pytest.approx(2 * 0.012 + 0.09 + 0.02)
    assert timing.group_gap_seconds() == 0.05


def test_color_change_prices_the_held_picker_clicks() -> None:
    timing = _timing()
    # The 45 ms settles are under a frame and run at the floor.
    assert timing.color_change_seconds() == pytest.approx(
        2 * PICKER_CLICK_HOLD_SECONDS + 2 * SETTLE_FLOOR_SECONDS + 2 * 0.01
    )
    assert timing.group_gap_seconds() == SETTLE_FLOOR_SECONDS
    assert _timing(delay_after_hue_seconds=0.12).color_change_seconds() == pytest.approx(
        2 * PICKER_CLICK_HOLD_SECONDS + 0.12 + SETTLE_FLOOR_SECONDS + 2 * 0.01
    )


def test_profile_and_schedule_agree_and_count_changes_like_the_painter() -> None:
    plan = _plan()
    timing = _timing()
    profile = PlanProfile.from_plan(plan)
    assert profile.stroke_count == plan.stroke_count == 4

    schedule = PlanWorkSchedule(plan, timing, 4.0, sizing=True)
    assert profile.seconds(timing, 4.0, sizing=True) == pytest.approx(schedule.total)
    assert estimate_plan_seconds(plan, timing, 4.0, sizing=True) == pytest.approx(
        schedule.total
    )

    # Red is selected once for its two consecutive groups; blue once more.
    # The brush is typed twice: diameter 1 for red, 3 for blue.
    expected = (
        2 * timing.color_change_seconds()
        + 2 * timing.brush_change_seconds()
        + 3 * timing.group_gap_seconds()
        + timing.stroke_seconds(29 * 4.0, 4.0)
        + timing.stroke_seconds(0)
        + timing.stroke_seconds(4.0)
        + timing.stroke_seconds(4.0)
    )
    assert schedule.total == pytest.approx(expected)
    # Group costs land where the painter charges them.
    assert schedule.group_cost(0) == pytest.approx(
        timing.color_change_seconds()
        + timing.brush_change_seconds()
        + timing.group_gap_seconds()
    )
    assert schedule.group_cost(1) == pytest.approx(timing.group_gap_seconds())
    assert schedule.stroke_cost(0, 1) == pytest.approx(timing.stroke_seconds(0))

    # Without sizing the brush is never retyped.
    assert PlanWorkSchedule(plan, timing, 4.0, sizing=False).total == pytest.approx(
        expected - 2 * timing.brush_change_seconds()
    )
    # A measured texel pitch paces the long run; the cell width stands in
    # for it until then.
    fine = PlanWorkSchedule(plan, timing, 4.0, sizing=True, texel_pitch_pixels=1.0)
    assert fine.total == pytest.approx(
        expected
        - timing.stroke_seconds(29 * 4.0, 4.0)
        + timing.stroke_seconds(29 * 4.0, 1.0)
    )
    assert fine.total > schedule.total


def test_timing_from_settings_mirrors_the_painter_settings() -> None:
    settings = PainterSettings(
        stroke_speed_pixels_per_second=900.0,
        delay_between_strokes_seconds=0.03,
    )
    timing = StrokeTiming.from_settings(settings, overhead_seconds=0.02)
    assert timing.stroke_speed_pixels_per_second == 900.0
    assert timing.delay_between_strokes_seconds == 0.03
    assert timing.overhead_seconds == 0.02
    assert timing.real_input


def test_remaining_is_the_model_first_and_the_measured_pace_later() -> None:
    # Nothing done yet: the prediction itself.
    assert remaining_seconds(0.0, 0.0, 600.0) == 600.0
    # Half the predicted work done in exactly the predicted time: half left.
    assert remaining_seconds(300.0, 300.0, 600.0) == pytest.approx(300.0)
    # Running at twice the predicted pace, long past the prior: about twice.
    slow = remaining_seconds(600.0, 300.0, 600.0)
    assert slow == pytest.approx(300.0 * (600.0 + PACE_PRIOR_SECONDS) / (300.0 + PACE_PRIOR_SECONDS))
    assert 550.0 < slow < 600.0
    # One slow first stroke does not blow the estimate up.
    early = remaining_seconds(2.0, 0.1, 600.0)
    assert early < 600.0 * 1.2
    assert remaining_seconds(10.0, 5.0, 0.0) is None
    assert remaining_seconds(10.0, 700.0, 600.0) == 0.0


def test_learned_timing_weighs_each_run_by_its_strokes_and_round_trips(
    tmp_path: Path,
) -> None:
    learned = LearnedTiming()
    assert learned.overhead_seconds == DEFAULT_STROKE_OVERHEAD_SECONDS
    # Too few strokes to believe.
    assert not learned.observe(predicted_seconds=10.0, actual_seconds=20.0, strokes=50)
    # 1000 strokes ran 10 s longer than predicted: this run measured the
    # overhead at 10 ms over the default, and counts for 1000 strokes
    # against the default's prior.
    assert learned.observe(predicted_seconds=80.0, actual_seconds=90.0, strokes=1000)
    measured = DEFAULT_STROKE_OVERHEAD_SECONDS + 0.01
    assert learned.overhead_seconds == pytest.approx(
        (DEFAULT_STROKE_OVERHEAD_SECONDS * LEARN_PRIOR_STROKES + measured * 1000)
        / (LEARN_PRIOR_STROKES + 1000)
    )
    assert learned.samples == 1

    path = tmp_path / "timing.json"
    learned.save(path)
    loaded = LearnedTiming.load(path)
    assert loaded.overhead_seconds == pytest.approx(learned.overhead_seconds, abs=1e-5)
    assert loaded.samples == 1
    assert loaded.history and loaded.history[0]["strokes"] == 1000
    assert loaded.history[0]["measured_overhead_seconds"] == pytest.approx(measured, abs=1e-5)

    # A run much faster than predicted cannot push the overhead negative,
    # and a wildly slow one cannot push it past the ceiling.
    floor = LearnedTiming()
    floor.observe(predicted_seconds=900.0, actual_seconds=100.0, strokes=1000)
    assert floor.overhead_seconds < DEFAULT_STROKE_OVERHEAD_SECONDS
    assert floor.history[-1]["measured_overhead_seconds"] == 0.0
    ceiling = LearnedTiming()
    ceiling.observe(predicted_seconds=100.0, actual_seconds=9000.0, strokes=1000)
    assert ceiling.history[-1]["measured_overhead_seconds"] == MAX_STROKE_OVERHEAD_SECONDS
    assert ceiling.overhead_seconds <= MAX_STROKE_OVERHEAD_SECONDS


def test_a_short_late_run_barely_moves_what_long_runs_measured() -> None:
    """The failure this replaces: one 229-stroke run that ran 24 s late
    lifted the overhead from 19 ms to 92 ms, and every estimate after it
    by hours."""

    learned = LearnedTiming()
    for _ in range(3):
        learned.observe(predicted_seconds=1800.0, actual_seconds=1790.0, strokes=16000)
    settled = learned.overhead_seconds
    assert abs(settled - DEFAULT_STROKE_OVERHEAD_SECONDS) < 0.002
    learned.observe(predicted_seconds=73.0, actual_seconds=97.0, strokes=229)
    assert learned.overhead_seconds - settled < 0.001
    # And a run's weight is capped: an overnight sign is one big vote, not
    # the only one.
    learned.observe(predicted_seconds=33000.0, actual_seconds=33000.0 + 0.05 * 300000, strokes=300000)
    assert learned.overhead_seconds < settled + 0.05 * 0.5


def test_checks_and_touch_up_are_learned_as_their_own_costs() -> None:
    learned = LearnedTiming()
    assert learned.check_capture_seconds == CHECK_CAPTURE_SECONDS_DEFAULT
    assert learned.check_repaint_fraction == CHECK_REPAINT_FRACTION_DEFAULT
    assert learned.touch_up_fraction == TOUCH_UP_FRACTION_DEFAULT
    assert learned.check_samples == 0 and learned.touch_up_samples == 0
    # Before any run: the defaults, shaped like the painter's work.
    assert learned.check_seconds(10, 100.0) == pytest.approx(
        10 * CHECK_CAPTURE_SECONDS_DEFAULT + 100.0 * CHECK_REPAINT_FRACTION_DEFAULT
    )
    assert learned.touch_up_seconds(100.0) == pytest.approx(100.0 * TOUCH_UP_FRACTION_DEFAULT)

    # A run of 600 s painting, 20 colors checked at 1 s a capture, 60 s of
    # repainting from the checks, and a 30 s touch-up.
    assert learned.observe(
        predicted_seconds=600.0,
        actual_seconds=600.0,
        strokes=5000,
        colors_checked=20,
        check_capture_seconds=20.0,
        check_repaint_seconds=60.0,
        touch_up_seconds=30.0,
    )
    assert learned.check_samples == 1 and learned.touch_up_samples == 1
    assert learned.check_capture_seconds == pytest.approx(
        (CHECK_CAPTURE_SECONDS_DEFAULT * LEARN_PRIOR_COLORS + 1.0 * 20) / (LEARN_PRIOR_COLORS + 20)
    )
    assert learned.check_repaint_fraction == pytest.approx(
        (CHECK_REPAINT_FRACTION_DEFAULT * LEARN_PRIOR_PAINT_SECONDS + 0.1 * 600)
        / (LEARN_PRIOR_PAINT_SECONDS + 600)
    )
    assert learned.touch_up_fraction == pytest.approx(
        (TOUCH_UP_FRACTION_DEFAULT * LEARN_PRIOR_PAINT_SECONDS + 0.05 * 600)
        / (LEARN_PRIOR_PAINT_SECONDS + 600)
    )
    # A run without checks, stopped before the touch-up, teaches neither.
    assert learned.observe(predicted_seconds=300.0, actual_seconds=310.0, strokes=3000)
    assert learned.check_samples == 1 and learned.touch_up_samples == 1
    # Too few colors checked to say what a capture costs.
    assert learned.observe(
        predicted_seconds=300.0,
        actual_seconds=310.0,
        strokes=3000,
        colors_checked=2,
        check_capture_seconds=9.0,
    )
    assert learned.check_samples == 1
    # A clean sign's touch-up is a few seconds of capture: a real, small sample.
    assert learned.observe(
        predicted_seconds=600.0, actual_seconds=600.0, strokes=5000, touch_up_seconds=2.0
    )
    assert learned.touch_up_samples == 2
    assert learned.touch_up_fraction < TOUCH_UP_FRACTION_DEFAULT


def test_an_older_timing_file_is_read_as_measurements(tmp_path: Path) -> None:
    """Files from before per-run measurements were kept hold the overhead
    after each run was blended in; each run's own measurement is the
    overhead before it plus its residual."""

    path = tmp_path / "timing.json"
    path.write_text(
        json.dumps(
            {
                "overhead_seconds": 0.092,
                "samples": 2,
                "history": [
                    {
                        "predicted_seconds": 1774.0,
                        "actual_seconds": 1730.1,
                        "strokes": 15893,
                        "overhead_seconds": 0.0117,
                    },
                    {
                        "predicted_seconds": 73.2,
                        "actual_seconds": 97.4,
                        "strokes": 229,
                        "overhead_seconds": 0.092,
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    loaded = LearnedTiming.load(path)
    first, second = loaded.history
    assert first["measured_overhead_seconds"] == pytest.approx(
        DEFAULT_STROKE_OVERHEAD_SECONDS + (1730.1 - 1774.0) / 15893, abs=1e-5
    )
    assert second["measured_overhead_seconds"] == pytest.approx(
        0.0117 + (97.4 - 73.2) / 229, abs=1e-5
    )
    # The stored 92 ms is not believed; the long run outweighs the short one.
    assert loaded.overhead_seconds < 0.02
    assert loaded.check_capture_seconds == CHECK_CAPTURE_SECONDS_DEFAULT
    assert loaded.touch_up_fraction == TOUCH_UP_FRACTION_DEFAULT


def test_learned_timing_survives_a_damaged_file(tmp_path: Path) -> None:
    path = tmp_path / "timing.json"
    path.write_text("{not json", encoding="utf-8")
    assert LearnedTiming.load(path).overhead_seconds == DEFAULT_STROKE_OVERHEAD_SECONDS
    path.write_text(json.dumps({"overhead_seconds": 9.0, "samples": "x"}), encoding="utf-8")
    assert LearnedTiming.load(path).overhead_seconds == DEFAULT_STROKE_OVERHEAD_SECONDS
    assert LearnedTiming.load(tmp_path / "missing.json").samples == 0


def test_a_line_priced_stroke_is_flat_and_far_under_the_capped_drag() -> None:
    """A Shift-click line costs its two presses however long the run is."""

    timing = StrokeTiming.from_settings(PainterSettings())
    pitch = 1.77
    long_run = 500 * pitch
    drag = timing.stroke_seconds(long_run, pitch)
    line = timing.stroke_seconds(long_run, pitch, line_tool=True)
    assert line < drag / 4
    press = max(0.07, MIN_PRESS_SECONDS)
    expected = (
        2 * press
        + 2 * SHIFT_LINE_MODIFIER_LEAD_SECONDS
        + STROKE_GAP_FLOOR_SECONDS
        + 2 * timing.overhead_seconds
    )
    assert line == pytest.approx(expected)
    # Flat: the game draws the span, so its length does not appear.
    assert line == pytest.approx(
        timing.stroke_seconds(4 * long_run, pitch, line_tool=True)
    )
    # A dab keeps its ordinary price whatever the flag says.
    assert timing.stroke_seconds(0.0, pitch, line_tool=True) == pytest.approx(
        timing.stroke_seconds(0.0, pitch)
    )


def test_work_schedule_prices_long_straight_runs_as_lines_only_when_proven() -> None:
    plan = PaintPlan(
        1000,
        10,
        (
            ColorGroup(
                (10, 20, 30),
                (Stroke(0, 2, 999, 2), Stroke(5, 4, 20, 4), Stroke(3, 1, 8, 7)),
                1,
            ),
        ),
    )
    timing = StrokeTiming.from_settings(PainterSettings())
    pitch = 2.0
    as_drags = PlanWorkSchedule(
        plan, timing, 2.0, sizing=False, texel_pitch_pixels=pitch
    )
    as_lines = PlanWorkSchedule(
        plan,
        timing,
        2.0,
        sizing=False,
        texel_pitch_pixels=pitch,
        line_min_pixels=SHIFT_LINE_MIN_TEXELS * pitch,
    )
    # The 1000-texel row is priced as a line, and only it: the short run is
    # under the threshold and the diagonal cannot be drawn straight.
    assert as_lines.stroke_cost(0, 0) < as_drags.stroke_cost(0, 0) / 4
    assert as_lines.stroke_cost(0, 1) == pytest.approx(as_drags.stroke_cost(0, 1))
    assert as_lines.stroke_cost(0, 2) == pytest.approx(as_drags.stroke_cost(0, 2))
    assert as_lines.total < as_drags.total


def test_a_measured_dab_press_prices_dabs_and_lines_but_never_drags() -> None:
    """The probe's hold replaces the frame floor for stationary presses only."""

    settings = PainterSettings()
    plain = StrokeTiming.from_settings(settings)
    measured = StrokeTiming.from_settings(settings, dab_press_seconds=0.024)
    pitch = 1.77

    saved = max(0.07, MIN_PRESS_SECONDS) - 0.024
    assert plain.stroke_seconds(0.0, pitch) - measured.stroke_seconds(
        0.0, pitch
    ) == pytest.approx(saved)
    long_run = 500 * pitch
    assert plain.stroke_seconds(
        long_run, pitch, line_tool=True
    ) - measured.stroke_seconds(long_run, pitch, line_tool=True) == pytest.approx(
        2 * saved
    )
    # A drag's dwell is not covered by the measurement and keeps its floor.
    assert measured.stroke_seconds(long_run, pitch) == pytest.approx(
        plain.stroke_seconds(long_run, pitch)
    )
    # Mock input has no frames to miss and no measured hold to honor.
    fake = StrokeTiming.from_settings(settings, real_input=False, dab_press_seconds=0.024)
    bare = StrokeTiming.from_settings(settings, real_input=False)
    assert fake.stroke_seconds(0.0, pitch) == pytest.approx(bare.stroke_seconds(0.0, pitch))
