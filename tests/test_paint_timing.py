from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.models import ColorGroup, PaintPlan, Stroke
from app.paint_timing import (
    DEFAULT_STROKE_OVERHEAD_SECONDS,
    MAX_STROKE_OVERHEAD_SECONDS,
    MIN_PRESS_SECONDS,
    PACE_PRIOR_SECONDS,
    PICKER_CLICK_HOLD_SECONDS,
    LearnedTiming,
    PlanProfile,
    PlanWorkSchedule,
    StrokeTiming,
    estimate_plan_seconds,
    remaining_seconds,
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
    assert dab == pytest.approx(MIN_PRESS_SECONDS + 0.005 + 0.01)
    assert short == pytest.approx(dab)
    # Only a drag longer than the hold costs more than the hold.
    long = timing.stroke_seconds(2200)
    assert long == pytest.approx(1.0 + 0.005 + 0.01)


def test_mock_input_skips_the_holds() -> None:
    timing = _timing(real_input=False)
    assert timing.stroke_seconds(0) == pytest.approx(0.012 + 0.005 + 0.01)
    assert timing.color_change_seconds() == pytest.approx(2 * 0.012 + 0.09 + 0.02)


def test_color_change_prices_the_held_picker_clicks() -> None:
    timing = _timing()
    assert timing.color_change_seconds() == pytest.approx(
        2 * PICKER_CLICK_HOLD_SECONDS + 0.09 + 2 * 0.01
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
        + 3 * timing.delay_between_colors_seconds
        + timing.stroke_seconds(29 * 4.0)
        + timing.stroke_seconds(0)
        + timing.stroke_seconds(4.0)
        + timing.stroke_seconds(4.0)
    )
    assert schedule.total == pytest.approx(expected)
    # Group costs land where the painter charges them.
    assert schedule.group_cost(0) == pytest.approx(
        timing.color_change_seconds()
        + timing.brush_change_seconds()
        + timing.delay_between_colors_seconds
    )
    assert schedule.group_cost(1) == pytest.approx(timing.delay_between_colors_seconds)
    assert schedule.stroke_cost(0, 1) == pytest.approx(timing.stroke_seconds(0))

    # Without sizing the brush is never retyped.
    assert PlanWorkSchedule(plan, timing, 4.0, sizing=False).total == pytest.approx(
        expected - 2 * timing.brush_change_seconds()
    )


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


def test_learned_timing_folds_in_the_residual_and_round_trips(tmp_path: Path) -> None:
    learned = LearnedTiming()
    assert learned.overhead_seconds == DEFAULT_STROKE_OVERHEAD_SECONDS
    # Too few strokes to believe.
    assert not learned.observe(predicted_seconds=10.0, actual_seconds=20.0, strokes=50)
    # 1000 strokes ran 10 s longer than predicted: 10 ms/stroke, blended.
    assert learned.observe(predicted_seconds=80.0, actual_seconds=90.0, strokes=1000)
    assert learned.overhead_seconds == pytest.approx(
        DEFAULT_STROKE_OVERHEAD_SECONDS + 0.7 * 0.01
    )
    assert learned.samples == 1

    path = tmp_path / "timing.json"
    learned.save(path)
    loaded = LearnedTiming.load(path)
    assert loaded.overhead_seconds == pytest.approx(learned.overhead_seconds, abs=1e-5)
    assert loaded.samples == 1
    assert loaded.history and loaded.history[0]["strokes"] == 1000

    # A run much faster than predicted cannot push the overhead negative,
    # and a wildly slow one cannot push it past the ceiling.
    loaded.observe(predicted_seconds=900.0, actual_seconds=100.0, strokes=1000)
    assert loaded.overhead_seconds == 0.0
    loaded.observe(predicted_seconds=100.0, actual_seconds=9000.0, strokes=1000)
    assert loaded.overhead_seconds == MAX_STROKE_OVERHEAD_SECONDS


def test_learned_timing_survives_a_damaged_file(tmp_path: Path) -> None:
    path = tmp_path / "timing.json"
    path.write_text("{not json", encoding="utf-8")
    assert LearnedTiming.load(path).overhead_seconds == DEFAULT_STROKE_OVERHEAD_SECONDS
    path.write_text(json.dumps({"overhead_seconds": 9.0, "samples": "x"}), encoding="utf-8")
    assert LearnedTiming.load(path).overhead_seconds == DEFAULT_STROKE_OVERHEAD_SECONDS
    assert LearnedTiming.load(tmp_path / "missing.json").samples == 0
