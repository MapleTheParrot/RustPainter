"""Predict how long a plan takes to paint, from the painter's own timing rules.

The painter does not spend its time moving the mouse.  Rust samples its
painting UI at about 15 FPS, so every press is held for at least one frame
(:data:`MIN_PRESS_SECONDS`) and every picker click a little longer; at any
speed setting worth using, almost every stroke is shorter than that hold.
Measured on real runs the cost of a stroke is therefore nearly flat -
around 85 ms whether it paints one cell or thirty - and an estimate built
from mouse travel alone lands five times short.

Everything here is derived from the same constants the painter executes
with, so the estimate and the job cannot drift apart.  The one quantity that
is not a constant, the per-stroke overhead of input calls, checkpoints and
timer slack, is learned from completed runs (:class:`LearnedTiming`) and
defaults to the value measured on the machine this was written on.
"""

from __future__ import annotations

import json
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .models import PaintPlan, RGBColor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .painter import PainterSettings


LOGGER = logging.getLogger("rust_painter.timing")

# Rust has been observed running its painting UI at 15 FPS (a 67 ms frame).
# Everything the game is asked to notice - a press, a picker click, the
# color a click just changed - has to survive until the next frame samples
# it, which makes a frame the floor under several of the timing settings.
# Each floor is a cliff rather than a slope: below it the event is simply
# not seen, above it extra waiting buys nothing.
FRAME_SECONDS = 1.0 / 15.0

# A press and release inside one frame can be sampled as nothing, so the
# painter keeps every stroke's button down until the press has lasted this
# long.  Slightly over a frame, so a press that starts late in one frame
# still straddles the boundary into the next.
MIN_PRESS_SECONDS = 0.07

# Picker clicks are rare (a handful per color change) and a dropped one
# paints a whole color wrong, so they are held longer still.
PICKER_CLICK_HOLD_SECONDS = 0.09

# After a picker or Size-field click, and before the picker is touched at
# all between colors, the UI must get a frame in which to apply the click:
# the S/V box is laid out for the hue just chosen, and the next stroke takes
# the color that is current when it is sampled.  One frame plus margin.
SETTLE_FLOOR_SECONDS = 0.07

# Between a stroke's release and the next press.  The game orders these as
# events rather than sampling them - a frame here is not needed, and would
# cost half again on every stroke - so this is only slack for the scheduler
# to deliver the release before the cursor jumps to the next stroke.
STROKE_GAP_FLOOR_SECONDS = 0.02

# A run of up to this many texels is a dab that moved: the press is held for
# a frame at its far end anyway, so it goes at whatever speed is set and
# cannot overshoot.  Longer drags are the ones the game samples mid-flight.
SHORT_RUN_TEXELS = 3.0

# The fastest a long drag crosses the sign, in texels per second, and the
# most of the sign the cursor skips between input events on one.  Above the
# rate the game has been seen to paint past a stroke's ends and skip texels
# in the middle; at it, with an event on every texel, it paints exactly the
# run.  Measured on the 320x240 artist canvas: Relaxed's ~130 texels/s is
# clean and Turbo's ~730 is not.
LONG_DRAG_MAX_TEXELS_PER_SECOND = 250.0
LONG_DRAG_MAX_STEP_TEXELS = 1.0

# Which timing settings have a floor, and what it is.  Every value is read
# through :func:`floored` at its point of use under real input, so a preset
# or a typed value below its floor is lifted - the painter logs once per
# job which ones were.
TIMING_FLOORS: dict[str, float] = {
    "mouse_down_duration_seconds": MIN_PRESS_SECONDS,
    "delay_after_hue_seconds": SETTLE_FLOOR_SECONDS,
    "delay_after_saturation_value_seconds": SETTLE_FLOOR_SECONDS,
    "delay_after_brush_seconds": SETTLE_FLOOR_SECONDS,
    "delay_between_colors_seconds": SETTLE_FLOOR_SECONDS,
    "delay_between_strokes_seconds": STROKE_GAP_FLOOR_SECONDS,
}


def floored(settings: "PainterSettings", name: str, *, real_input: bool = True) -> float:
    """A timing setting, lifted to its floor when the input is real."""

    value = float(getattr(settings, name))
    if not real_input:
        return value
    return max(value, TIMING_FLOORS.get(name, 0.0))


def fields_below_floor(settings: "PainterSettings") -> tuple[str, ...]:
    """The timing settings a real run lifts, in declaration order."""

    return tuple(
        name
        for name, floor in TIMING_FLOORS.items()
        if float(getattr(settings, name)) < floor
    )


@dataclass(frozen=True, slots=True)
class StrokePace:
    """How a drag of one length is driven: the event spacing and the time."""

    step_pixels: float
    move_seconds: float


def stroke_pace(
    distance_pixels: float,
    *,
    speed_pixels_per_second: float,
    step_pixels: float,
    texel_pitch_pixels: float,
    real_input: bool = True,
    max_texels_per_second: float = LONG_DRAG_MAX_TEXELS_PER_SECOND,
    max_step_texels: float = LONG_DRAG_MAX_STEP_TEXELS,
) -> StrokePace:
    """Pick the speed and interpolation step for a drag of this length.

    Short runs take the settings as they are - the frame hold at their far
    end is what makes them land, and there is nothing for them to overshoot.
    A long drag is capped at :data:`LONG_DRAG_MAX_TEXELS_PER_SECOND` and its
    step at one texel, so every texel on the way gets its own input event and
    the cursor never crosses more of the sign per frame than the game paints
    faithfully.  Mock input has no game behind it and is not capped.
    """

    speed = max(float(speed_pixels_per_second), 1e-9)
    step = max(float(step_pixels), 1e-9)
    pitch = float(texel_pitch_pixels)
    if (
        real_input
        and math.isfinite(pitch)
        and pitch > 0.0
        and distance_pixels > SHORT_RUN_TEXELS * pitch
    ):
        speed = min(speed, max_texels_per_second * pitch)
        step = min(step, max_step_texels * pitch)
    return StrokePace(step_pixels=step, move_seconds=max(0.0, distance_pixels) / speed)

# The frame floor above assumes the 15 FPS the paint UI was measured at on a
# 320x240 sign.  On a 1024x512 sign the same UI was found dropping a third of
# all presses.  Measured on the largest sign: 7,473 presses at holds from
# 8 ms to 160 ms, at brush sizes 1 and 10, under two minutes of continuous
# painting and across anti-AFK breaks, and not one was dropped.  The
# painter checks each color as it goes down anyway (see
# ``Painter._confirm_group``), since a stroke can still go wrong, but the
# press hold is never lengthened for it: the speckled signs were colors
# picked wrong, not presses lost.
CONFIRM_MIN_JUDGED_CELLS = 40
# After a color's last stroke, before the sign is captured to check it: the
# game has to present the frame that carries the stroke, which at its
# slowest observed can be a quarter of a second away.
CONFIRM_SETTLE_SECONDS = 0.45

# Keystrokes into Rust's Size field are held across a frame boundary and
# separated from the next, for the same reason.
KEY_HOLD_SECONDS = 0.03
KEY_GAP_SECONDS = 0.02

# Clearing the Size field from both sides of the caret, then typing a value
# of up to six characters and Enter.
BRUSH_FIELD_KEYSTROKES = 12 + 6 + 1

# Time a stroke costs beyond its scripted holds and delays: the SendInput
# calls, the safety checkpoints between them, the scheduler's slack on every
# sleep slice, and the progress callback.  Measured at 10-12 ms on the
# machine this was written on; runs refine it for the machine they ran on.
DEFAULT_STROKE_OVERHEAD_SECONDS = 0.012
MAX_STROKE_OVERHEAD_SECONDS = 0.25

# The brush measurement before a run: a scout stroke, five probes, and two
# wipes of the sign.  Measured at 18 s on a real sign.
BRUSH_CALIBRATION_SECONDS = 18.0

# How many strokes a run must have painted before its timing is believed,
# and how much of the disagreement between prediction and reality is folded
# into the learned overhead each time.
LEARN_MIN_STROKES = 200
LEARN_BLEND = 0.7


@dataclass(frozen=True, slots=True)
class StrokeTiming:
    """The painter's timing rules, reduced to what they cost in seconds."""

    stroke_speed_pixels_per_second: float
    mouse_down_duration_seconds: float
    delay_between_strokes_seconds: float
    delay_after_hue_seconds: float
    delay_after_saturation_value_seconds: float
    delay_between_colors_seconds: float
    delay_after_brush_seconds: float
    stroke_interpolation_step_pixels: float = 4.0
    overhead_seconds: float = DEFAULT_STROKE_OVERHEAD_SECONDS
    # Mock and dry-run input skips the frame holds entirely.
    real_input: bool = True

    @classmethod
    def from_settings(
        cls,
        settings: "PainterSettings",
        *,
        overhead_seconds: float = DEFAULT_STROKE_OVERHEAD_SECONDS,
        real_input: bool = True,
    ) -> "StrokeTiming":
        return cls(
            stroke_speed_pixels_per_second=settings.stroke_speed_pixels_per_second,
            mouse_down_duration_seconds=settings.mouse_down_duration_seconds,
            delay_between_strokes_seconds=settings.delay_between_strokes_seconds,
            delay_after_hue_seconds=settings.delay_after_hue_seconds,
            delay_after_saturation_value_seconds=(
                settings.delay_after_saturation_value_seconds
            ),
            delay_between_colors_seconds=settings.delay_between_colors_seconds,
            delay_after_brush_seconds=settings.delay_after_brush_seconds,
            stroke_interpolation_step_pixels=settings.stroke_interpolation_step_pixels,
            overhead_seconds=overhead_seconds,
            real_input=real_input,
        )

    def _held(self, seconds: float, floor: float) -> float:
        return max(seconds, floor) if self.real_input else seconds

    def stroke_seconds(
        self, screen_length_pixels: float, texel_pitch_pixels: float | None = None
    ) -> float:
        """One stroke of this on-screen length, start of press to next stroke.

        ``texel_pitch_pixels`` is what the painter paces long drags by; the
        estimate uses the same rule so a plan of long sweeps is not promised
        at a speed the drags are never driven at.
        """

        if screen_length_pixels <= 0:
            press = self._held(self.mouse_down_duration_seconds, MIN_PRESS_SECONDS)
        else:
            pace = stroke_pace(
                screen_length_pixels,
                speed_pixels_per_second=self.stroke_speed_pixels_per_second,
                step_pixels=self.stroke_interpolation_step_pixels,
                texel_pitch_pixels=(
                    texel_pitch_pixels if texel_pitch_pixels is not None else float("nan")
                ),
                real_input=self.real_input,
            )
            press = self._held(pace.move_seconds, MIN_PRESS_SECONDS)
        gap = self._held(self.delay_between_strokes_seconds, STROKE_GAP_FLOOR_SECONDS)
        return press + gap + self.overhead_seconds

    def group_gap_seconds(self) -> float:
        """The pause after a color's last stroke, before the picker."""

        return self._held(self.delay_between_colors_seconds, SETTLE_FLOOR_SECONDS)

    def color_change_seconds(self) -> float:
        """Two held picker clicks and the settle after each."""

        click = self._held(self.mouse_down_duration_seconds, PICKER_CLICK_HOLD_SECONDS)
        return (
            2 * click
            + self._held(self.delay_after_hue_seconds, SETTLE_FLOOR_SECONDS)
            + self._held(self.delay_after_saturation_value_seconds, SETTLE_FLOOR_SECONDS)
            + 2 * self.overhead_seconds
        )

    def brush_change_seconds(self) -> float:
        """Clicking into the Size field, retyping it, and committing."""

        settle = (
            max(self.delay_after_brush_seconds, SETTLE_FLOOR_SECONDS)
            if self.real_input
            else 0.0
        )
        click = self._held(self.mouse_down_duration_seconds, PICKER_CLICK_HOLD_SECONDS)
        keys = (
            BRUSH_FIELD_KEYSTROKES * (KEY_HOLD_SECONDS + KEY_GAP_SECONDS)
            if self.real_input
            else 0.0
        )
        return click + 2 * settle + keys + self.overhead_seconds


@dataclass(frozen=True, slots=True)
class GroupSummary:
    color: RGBColor
    brush_diameter: int
    stroke_count: int
    # (cells in stroke, how many strokes have that many), so a plan with
    # fifty thousand strokes reduces to a few dozen numbers.
    length_counts: tuple[tuple[int, int], ...]


@dataclass(frozen=True, slots=True)
class PlanProfile:
    """What a plan costs in strokes and changes, independent of any timing.

    Built once per plan (it walks every stroke) and then priced in
    microseconds for any timing, which is what lets the estimate follow the
    speed sliders live.
    """

    groups: tuple[GroupSummary, ...]

    @classmethod
    def from_plan(cls, plan: PaintPlan) -> "PlanProfile":
        groups = []
        for group in plan.color_groups:
            counts: dict[int, int] = {}
            for stroke in group.strokes:
                cells = max(1, int(stroke.pixel_count))
                counts[cells] = counts.get(cells, 0) + 1
            groups.append(
                GroupSummary(
                    color=group.color,
                    brush_diameter=max(1, int(group.brush_diameter)),
                    stroke_count=len(group.strokes),
                    length_counts=tuple(sorted(counts.items())),
                )
            )
        return cls(groups=tuple(groups))

    @property
    def stroke_count(self) -> int:
        return sum(group.stroke_count for group in self.groups)

    def seconds(
        self,
        timing: StrokeTiming,
        cell_width_pixels: float,
        *,
        sizing: bool,
        texel_pitch_pixels: float | None = None,
    ) -> float:
        """Painting time for the whole plan, calibration and countdown excluded.

        Long drags are paced by the sign's texel pitch; until a job has
        measured one, a logical cell - never smaller than a texel - stands
        in, which errs toward promising speed rather than time.
        """

        pitch = texel_pitch_pixels if texel_pitch_pixels is not None else cell_width_pixels
        total = 0.0
        previous_color: RGBColor | None = None
        previous_diameter: int | None = None
        for group in self.groups:
            if group.stroke_count == 0:
                continue
            if group.color != previous_color:
                total += timing.color_change_seconds()
                previous_color = group.color
            if sizing and group.brush_diameter != previous_diameter:
                total += timing.brush_change_seconds()
                previous_diameter = group.brush_diameter
            for cells, count in group.length_counts:
                total += count * timing.stroke_seconds(
                    (cells - 1) * cell_width_pixels, pitch
                )
            total += timing.group_gap_seconds()
        return total


def estimate_plan_seconds(
    plan: PaintPlan,
    timing: StrokeTiming,
    cell_width_pixels: float,
    *,
    sizing: bool,
    profile: PlanProfile | None = None,
    texel_pitch_pixels: float | None = None,
) -> float:
    profile = profile if profile is not None else PlanProfile.from_plan(plan)
    return profile.seconds(
        timing, cell_width_pixels, sizing=sizing, texel_pitch_pixels=texel_pitch_pixels
    )


class PlanWorkSchedule:
    """Per-stroke predicted costs, for progress that advances in seconds.

    Progress weighted by cells painted races ahead while the big, long-stroke
    colors go down and crawls through the small ones; weighted by predicted
    seconds it moves at the rate the clock does, so the bar and the time left
    stay honest from the first color to the last.
    """

    __slots__ = ("_group_costs", "_stroke_costs", "total")

    def __init__(
        self,
        plan: PaintPlan,
        timing: StrokeTiming,
        cell_width_pixels: float,
        *,
        sizing: bool,
        texel_pitch_pixels: float | None = None,
    ) -> None:
        pitch = texel_pitch_pixels if texel_pitch_pixels is not None else cell_width_pixels
        group_costs: list[float] = []
        stroke_costs: list[list[float]] = []
        previous_color: RGBColor | None = None
        previous_diameter: int | None = None
        total = 0.0
        for group in plan.color_groups:
            cost = 0.0
            if group.strokes:
                if group.color != previous_color:
                    cost += timing.color_change_seconds()
                    previous_color = group.color
                diameter = max(1, int(group.brush_diameter))
                if sizing and diameter != previous_diameter:
                    cost += timing.brush_change_seconds()
                    previous_diameter = diameter
            cost += timing.group_gap_seconds()
            group_costs.append(cost)
            costs = [
                timing.stroke_seconds(
                    (max(1, int(stroke.pixel_count)) - 1) * cell_width_pixels, pitch
                )
                for stroke in group.strokes
            ]
            stroke_costs.append(costs)
            total += cost + sum(costs)
        self._group_costs = group_costs
        self._stroke_costs = stroke_costs
        self.total = total

    def group_cost(self, group_index: int) -> float:
        """Color and brush changes charged when a group starts."""

        return self._group_costs[group_index]

    def stroke_cost(self, group_index: int, stroke_index: int) -> float:
        return self._stroke_costs[group_index][stroke_index]


# Seconds of predicted work the live estimate trusts the model for before the
# measured pace takes over.  Early in a run a handful of strokes is a noisy
# sample; by this much work the measured pace is worth more than the prior.
PACE_PRIOR_SECONDS = 20.0


def remaining_seconds(
    elapsed: float, completed_work: float, total_work: float
) -> float | None:
    """Time left, from predicted work done and the clock that did it.

    The prediction is scaled by how fast the job is actually going, with a
    prior that pulls the scale toward 1 until enough work has been measured
    to trust the observed pace - so the first estimate is the model's, and
    every later one is the model corrected by this run.
    """

    if total_work <= 0:
        return None
    left = max(0.0, total_work - completed_work)
    if completed_work <= 0:
        return left
    pace = (elapsed + PACE_PRIOR_SECONDS) / (completed_work + PACE_PRIOR_SECONDS)
    return left * pace


@dataclass
class LearnedTiming:
    """The per-stroke overhead this machine has shown, kept between runs."""

    overhead_seconds: float = DEFAULT_STROKE_OVERHEAD_SECONDS
    samples: int = 0
    history: list[dict[str, float]] = field(default_factory=list)

    @classmethod
    def load(cls, path: Path) -> "LearnedTiming":
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except FileNotFoundError:
            return cls()
        except (OSError, ValueError):
            LOGGER.warning("Learned timing at %s is unreadable; starting over", path)
            return cls()
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "LearnedTiming":
        try:
            overhead = float(data.get("overhead_seconds", DEFAULT_STROKE_OVERHEAD_SECONDS))
            samples = int(data.get("samples", 0))
        except (TypeError, ValueError):
            return cls()
        if not 0.0 <= overhead <= MAX_STROKE_OVERHEAD_SECONDS:
            overhead = DEFAULT_STROKE_OVERHEAD_SECONDS
        history = data.get("history")
        return cls(
            overhead_seconds=overhead,
            samples=max(0, samples),
            history=[dict(item) for item in history] if isinstance(history, list) else [],
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "overhead_seconds": round(self.overhead_seconds, 5),
            "samples": self.samples,
            "history": self.history[-20:],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def observe(
        self, *, predicted_seconds: float, actual_seconds: float, strokes: int
    ) -> bool:
        """Fold one run's prediction-versus-clock into the overhead.

        ``predicted_seconds`` must have been computed with the overhead this
        object held at the time, so the residual per stroke is exactly the
        correction the overhead needs.  Returns whether anything was learned.
        """

        if strokes < LEARN_MIN_STROKES or predicted_seconds <= 0 or actual_seconds <= 0:
            return False
        residual = (actual_seconds - predicted_seconds) / strokes
        corrected = self.overhead_seconds + LEARN_BLEND * residual
        corrected = min(MAX_STROKE_OVERHEAD_SECONDS, max(0.0, corrected))
        self.history.append(
            {
                "predicted_seconds": round(predicted_seconds, 1),
                "actual_seconds": round(actual_seconds, 1),
                "strokes": strokes,
                "overhead_seconds": round(corrected, 5),
            }
        )
        self.overhead_seconds = corrected
        self.samples += 1
        return True


@dataclass(frozen=True, slots=True)
class PhaseTiming:
    """How a painting phase's clock compared with its prediction.

    ``checking_seconds`` is the part of the clock spent checking colors as
    they went down and repainting what missed.  The time left is predicted
    from the whole clock, since the job will go on spending it; the
    per-stroke overhead is learned without it, since it is the game's
    dropped presses on this sign and not this machine's cost of a stroke.
    """

    predicted_seconds: float
    actual_seconds: float
    strokes: int
    checking_seconds: float = 0.0


__all__ = [
    "BRUSH_CALIBRATION_SECONDS",
    "DEFAULT_STROKE_OVERHEAD_SECONDS",
    "FRAME_SECONDS",
    "KEY_GAP_SECONDS",
    "KEY_HOLD_SECONDS",
    "LONG_DRAG_MAX_STEP_TEXELS",
    "LONG_DRAG_MAX_TEXELS_PER_SECOND",
    "LearnedTiming",
    "MIN_PRESS_SECONDS",
    "PICKER_CLICK_HOLD_SECONDS",
    "PhaseTiming",
    "PlanProfile",
    "PlanWorkSchedule",
    "SETTLE_FLOOR_SECONDS",
    "SHORT_RUN_TEXELS",
    "STROKE_GAP_FLOOR_SECONDS",
    "StrokePace",
    "StrokeTiming",
    "TIMING_FLOORS",
    "estimate_plan_seconds",
    "fields_below_floor",
    "floored",
    "remaining_seconds",
    "stroke_pace",
]
