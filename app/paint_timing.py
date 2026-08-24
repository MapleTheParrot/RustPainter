"""Predict how long a plan takes to paint, from the painter's own timing rules.

The painter does not spend its time moving the mouse.  Rust samples its
painting UI at about 15 FPS, so every press is held for at least one frame
(:data:`MIN_PRESS_SECONDS`) and every picker click a little longer; at any
speed setting worth using, almost every stroke is shorter than that hold.
Measured on real runs the cost of a stroke is therefore nearly flat -
around 85 ms whether it paints one cell or thirty - and an estimate built
from mouse travel alone lands five times short.

Everything here is derived from the same constants the painter executes
with, so the estimate and the job cannot drift apart.  What is not a
constant is learned from finished runs (:class:`LearnedTiming`): the
per-stroke overhead of input calls, checkpoints and timer slack, what a
capture costs when a color is checked as it goes down, and how much
repainting the checks and the touch-up pass at the end turn out to need -
the last two being the game's dropped presses on this sign, which no
estimate can know before the sign exists.  Each defaults to what was
measured on the machine this was written on.
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

# Rust's paint UI has a line tool: with Shift held through a drag, the game
# fills the straight stroke between the press and the release itself, in its
# own texture space.  A run that long stops being a capped drag and becomes
# a press, one cursor jump and a release, so a full row on an XXL sign goes
# down in a quarter of a second instead of two, and the texels between the
# endpoints are the game's to fill, out of reach of the cursor quantization
# that shifts individual dabs on a DPI-scaled display.  The painter only
# ever uses it after proving it on this sign with a probe stroke, since the
# mechanic is the game's and could change under us.  Runs shorter than this
# many texels keep the glide: below it the drag is about as fast, and the
# drag is the proven path.
SHIFT_LINE_MIN_TEXELS = 32.0
# The Shift key goes down this long before the line's press and stays down
# this long after its release.  The game reads the keyboard and the mouse
# per frame; a modifier that arrives in the same slice as the press, or
# leaves in the slice of the release, can be sampled as absent and the line
# becomes a drag the cursor never travelled - two dabs.
SHIFT_LINE_MODIFIER_LEAD_SECONDS = 0.05

# The press hold IS the painting phase on a detailed sign: tens of thousands
# of dabs, each held for the 70 ms frame floor.  Yet 7,473 presses at holds
# from 8 ms to 160 ms landed every single one (measured live on the largest
# sign), so the floor is caution left over from the 15 FPS theory, not
# physics.  The painter proves per sign whether the caution is needed: one
# batch of dots per candidate hold, longest first, each batch captured and
# counted, stopping at the first batch that dropped a dot.  The hold adopted
# is the shortest clean one whose next-shorter neighbour ALSO landed
# everything - a step of demonstrated margin - and it applies to stationary
# presses only: dabs and the line tool's clicks.  Drag dwells keep the full
# floor, because dropped short drags were a real live failure (708 bare
# cells in one overnight run) and the measurement above covered presses that
# never moved.
PRESS_HOLD_PROBE_CANDIDATES = (0.040, 0.030, 0.024, 0.018)
PRESS_HOLD_PROBE_DOTS = 24
# Below this many strokes the probe's captures cost more than its shorter
# holds could save.
PRESS_HOLD_PROBE_MIN_STROKES = 200

# A lone dab is the stroke that goes missing.  Measured on a finished XXL
# sign: texels painted as a lone dab came out bare 4.4% of the time (7.4%
# in the light areas) against 0.3% for texels swept by a drag - two thirds
# of every hole on the sign, on a sign that is a third lone dabs.  The
# smallest brush there covers about half a texel, so whether one stationary
# stamp takes its texel depends on where within the texel the game samples
# the cursor.  The painter proves the dab per sign: batches of lone-dab
# strokes - through the same aim, hold and extension the artwork will use -
# at these Size numbers, smallest first, each captured and counted, and the
# first Size that lands all but DAB_PROBE_MAX_MISSES of its dots is what the
# job types for its one-cell strokes.  Spill into the neighbours is counted
# and logged: a stamp wide enough never to miss can also be wide enough to
# smear, and the log is where that trade shows.
DAB_PROBE_SIZES = (1.0, 1.25, 1.5, 1.75, 2.0)
DAB_PROBE_DOTS = 96
DAB_PROBE_MAX_MISSES = 1
# Below this many lone dabs in the plan the probe cannot pay for itself.
DAB_PROBE_MIN_DABS = 200

# The gap between one stroke's release and the next press exists so the game
# sees the release: two presses in one frame can read as one held press.
# Its 20 ms floor was set by reasoning, not measurement, and on a dab-heavy
# sign it is a third of every stroke once the hold has been probed down.
# So it is probed the way the hold is: batches of dots at these gaps,
# longest first, at the hold this sign already proved, adopting the shortest
# clean gap whose next-shorter neighbour was clean too.
STROKE_GAP_PROBE_CANDIDATES = (0.020, 0.012, 0.008, 0.005)

# Long drags are capped at LONG_DRAG_MAX_TEXELS_PER_SECOND because the game
# paints between the cursor positions it samples only up to some distance;
# past it a run has holes.  That cap was inferred from one sign.  The painter
# probes it: one drag across half a row per candidate rate, fastest last,
# read back for coverage, adopting the second-fastest clean rate so the one
# used always has a proven step of margin above it.  Only signs whose plan
# has runs at least DRAG_RATE_PROBE_MIN_RUN_TEXELS long bother.
DRAG_RATE_PROBE_TEXELS_PER_SECOND = (250.0, 400.0, 600.0, 900.0)
DRAG_RATE_PROBE_MIN_RUN_TEXELS = 8

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

# How many strokes a run must have painted before its timing is believed.
# Each believed run measures the overhead on its own (its residual per
# stroke on top of the overhead it was predicted with), and the learned
# value is the mean of those measurements weighted by how many strokes
# each run painted - a short run that ran late says little, an overnight
# one says a lot - with the default standing in as a run of
# LEARN_PRIOR_STROKES.  A single run's weight is capped so one enormous
# sign does not hold the figure against every run after it.
LEARN_MIN_STROKES = 200
LEARN_PRIOR_STROKES = 2000
LEARN_MAX_WEIGHT_STROKES = 20000

# Checking each color as it goes down (``Painter._confirm_group``) costs a
# capture per color - the settle before it, the screenshot, reading every
# cell - and then whatever repainting the capture asks for, which is the
# game's dropped presses on this sign and scales with how much was
# painted.  The touch-up pass at the end (``Painter._verify_and_touch_up``)
# is the same shape without the per-color part: a capture or two, and a
# repaint that scales with the sign.  Neither can be known before the sign
# exists, so both are learned from finished runs as averages and the
# defaults below stand in until one has been measured: the capture from the
# settle and sampling the painter does, the fractions from the runs this
# was written against.
CHECK_CAPTURE_SECONDS_DEFAULT = CONFIRM_SETTLE_SECONDS + 0.25
CHECK_REPAINT_FRACTION_DEFAULT = 0.15
TOUCH_UP_FRACTION_DEFAULT = 0.1
# How many colors a run must have checked, and how long its artwork must
# have painted, before its checks and touch-up are believed; the prior
# weights match the stroke prior above in spirit - a few small runs.
LEARN_MIN_CHECKED_COLORS = 3
LEARN_MIN_PAINT_SECONDS = 30.0
LEARN_PRIOR_COLORS = 10
LEARN_MAX_WEIGHT_COLORS = 200
LEARN_PRIOR_PAINT_SECONDS = 300.0
LEARN_MAX_WEIGHT_PAINT_SECONDS = 3600.0
MAX_CHECK_CAPTURE_SECONDS = 10.0
MAX_REPAINT_FRACTION = 3.0


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
    # The press hold this sign's probe proved for stationary presses, when
    # one was measured and is shorter than the configured hold.  Prices dabs
    # and the line tool's presses; drag dwells keep the configured floor.
    dab_press_seconds: float | None = None
    # The gap between strokes this sign's probe proved, and the drag rate it
    # proved, when measured and quicker than the floors.
    gap_seconds: float | None = None
    drag_texels_per_second: float | None = None

    @classmethod
    def from_settings(
        cls,
        settings: "PainterSettings",
        *,
        overhead_seconds: float = DEFAULT_STROKE_OVERHEAD_SECONDS,
        real_input: bool = True,
        dab_press_seconds: float | None = None,
        gap_seconds: float | None = None,
        drag_texels_per_second: float | None = None,
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
            dab_press_seconds=dab_press_seconds,
            gap_seconds=gap_seconds,
            drag_texels_per_second=drag_texels_per_second,
        )

    def _held(self, seconds: float, floor: float) -> float:
        return max(seconds, floor) if self.real_input else seconds

    def stroke_seconds(
        self,
        screen_length_pixels: float,
        texel_pitch_pixels: float | None = None,
        *,
        line_tool: bool = False,
    ) -> float:
        """One stroke of this on-screen length, start of press to next stroke.

        ``texel_pitch_pixels`` is what the painter paces long drags by; the
        estimate uses the same rule so a plan of long sweeps is not promised
        at a speed the drags are never driven at.  ``line_tool`` prices the
        stroke as the Shift line the painter will draw it with: a press held
        a frame, one jump, a frame held at the far end, and the modifier's
        lead and trail - a flat cost however long the run is.
        """

        gap_floor = self._held(self.delay_between_strokes_seconds, STROKE_GAP_FLOOR_SECONDS)
        if self.gap_seconds is not None and self.real_input:
            gap_floor = min(gap_floor, self.gap_seconds)
        stationary = self._held(self.mouse_down_duration_seconds, MIN_PRESS_SECONDS)
        if self.dab_press_seconds is not None and self.real_input:
            stationary = min(stationary, self.dab_press_seconds)
        if line_tool and screen_length_pixels > 0:
            lead = SHIFT_LINE_MODIFIER_LEAD_SECONDS if self.real_input else 0.0
            return 2 * stationary + 2 * lead + gap_floor + 2 * self.overhead_seconds
        if screen_length_pixels <= 0:
            press = stationary
        else:
            pace = stroke_pace(
                screen_length_pixels,
                speed_pixels_per_second=self.stroke_speed_pixels_per_second,
                step_pixels=self.stroke_interpolation_step_pixels,
                texel_pitch_pixels=(
                    texel_pitch_pixels if texel_pitch_pixels is not None else float("nan")
                ),
                real_input=self.real_input,
                max_texels_per_second=(
                    self.drag_texels_per_second
                    if self.drag_texels_per_second is not None
                    else LONG_DRAG_MAX_TEXELS_PER_SECOND
                ),
            )
            press = self._held(pace.move_seconds, MIN_PRESS_SECONDS)
        return press + gap_floor + self.overhead_seconds

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
        line_min_pixels: float | None = None,
    ) -> None:
        """``line_min_pixels`` prices straight runs at least that long as
        Shift-click lines; ``None`` (the tool unproven or off) prices every
        run as the drag it will be.  Priced wrong, a line-heavy run finishes
        far ahead of its prediction and the learned per-stroke overhead is
        dragged negative, souring the estimate for every plan after it."""

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
                    (max(1, int(stroke.pixel_count)) - 1) * cell_width_pixels,
                    pitch,
                    line_tool=(
                        line_min_pixels is not None
                        and (stroke.start_x == stroke.end_x or stroke.start_y == stroke.end_y)
                        and (max(1, int(stroke.pixel_count)) - 1) * cell_width_pixels
                        >= line_min_pixels
                    ),
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


def _weighted_mean(
    samples: list[tuple[float, float]], *, prior: float, prior_weight: float
) -> float:
    """The mean of ``(value, weight)`` samples, with the prior counted in."""

    total = prior * prior_weight
    weight = prior_weight
    for value, sample_weight in samples:
        total += value * sample_weight
        weight += sample_weight
    return total / weight if weight > 0 else prior


@dataclass
class LearnedTiming:
    """What this machine's runs have shown the estimate cannot know upfront.

    The per-stroke overhead of input calls and timer slack, the cost of a
    capture when a color is checked, and how much repainting the checks and
    the touch-up pass turn out to need, as fractions of the painting they
    follow.  Every figure is a weighted mean over the runs on record with
    the default as a prior, recomputed whenever a run is added, so a single
    odd run moves it in proportion to how much that run painted.
    """

    overhead_seconds: float = DEFAULT_STROKE_OVERHEAD_SECONDS
    check_capture_seconds: float = CHECK_CAPTURE_SECONDS_DEFAULT
    check_repaint_fraction: float = CHECK_REPAINT_FRACTION_DEFAULT
    touch_up_fraction: float = TOUCH_UP_FRACTION_DEFAULT
    samples: int = 0
    check_samples: int = 0
    touch_up_samples: int = 0
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
        learned = cls(
            overhead_seconds=overhead,
            samples=max(0, samples),
            history=(
                [dict(item) for item in history if isinstance(item, Mapping)]
                if isinstance(history, list)
                else []
            ),
        )
        learned._restore_measurements()
        learned._recompute()
        return learned

    def _restore_measurements(self) -> None:
        """Give runs recorded before per-run measurements were kept theirs.

        An older entry holds the overhead *after* the run was folded in;
        the overhead it was predicted with is the entry before it (or the
        default), and its own measurement is that plus its residual.
        """

        previous = DEFAULT_STROKE_OVERHEAD_SECONDS
        for entry in self.history:
            try:
                strokes = int(entry.get("strokes", 0))
                predicted = float(entry.get("predicted_seconds", 0.0))
                actual = float(entry.get("actual_seconds", 0.0))
                after = float(entry.get("overhead_seconds", previous))
            except (TypeError, ValueError):
                continue
            if "measured_overhead_seconds" not in entry and strokes > 0:
                measured = previous + (actual - predicted) / strokes
                entry["measured_overhead_seconds"] = round(
                    min(MAX_STROKE_OVERHEAD_SECONDS, max(0.0, measured)), 5
                )
            previous = after

    def _recompute(self) -> None:
        overhead: list[tuple[float, float]] = []
        capture: list[tuple[float, float]] = []
        repaint: list[tuple[float, float]] = []
        touch_up: list[tuple[float, float]] = []
        for entry in self.history:
            try:
                strokes = float(entry.get("strokes", 0))
                measured = entry.get("measured_overhead_seconds")
                if measured is not None and strokes > 0:
                    overhead.append(
                        (float(measured), min(strokes, LEARN_MAX_WEIGHT_STROKES))
                    )
                colors = float(entry.get("colors_checked", 0))
                if colors > 0 and "check_capture_seconds" in entry:
                    capture.append(
                        (
                            float(entry["check_capture_seconds"]) / colors,
                            min(colors, LEARN_MAX_WEIGHT_COLORS),
                        )
                    )
                paint = float(entry.get("actual_seconds", 0.0))
                if paint > 0 and "check_repaint_seconds" in entry:
                    repaint.append(
                        (
                            float(entry["check_repaint_seconds"]) / paint,
                            min(paint, LEARN_MAX_WEIGHT_PAINT_SECONDS),
                        )
                    )
                if paint > 0 and "touch_up_seconds" in entry:
                    touch_up.append(
                        (
                            float(entry["touch_up_seconds"]) / paint,
                            min(paint, LEARN_MAX_WEIGHT_PAINT_SECONDS),
                        )
                    )
            except (TypeError, ValueError):
                continue
        if overhead:
            self.overhead_seconds = min(
                MAX_STROKE_OVERHEAD_SECONDS,
                max(
                    0.0,
                    _weighted_mean(
                        overhead,
                        prior=DEFAULT_STROKE_OVERHEAD_SECONDS,
                        prior_weight=LEARN_PRIOR_STROKES,
                    ),
                ),
            )
        self.check_capture_seconds = min(
            MAX_CHECK_CAPTURE_SECONDS,
            max(
                0.0,
                _weighted_mean(
                    capture,
                    prior=CHECK_CAPTURE_SECONDS_DEFAULT,
                    prior_weight=LEARN_PRIOR_COLORS,
                ),
            ),
        )
        self.check_repaint_fraction = min(
            MAX_REPAINT_FRACTION,
            max(
                0.0,
                _weighted_mean(
                    repaint,
                    prior=CHECK_REPAINT_FRACTION_DEFAULT,
                    prior_weight=LEARN_PRIOR_PAINT_SECONDS,
                ),
            ),
        )
        self.touch_up_fraction = min(
            MAX_REPAINT_FRACTION,
            max(
                0.0,
                _weighted_mean(
                    touch_up,
                    prior=TOUCH_UP_FRACTION_DEFAULT,
                    prior_weight=LEARN_PRIOR_PAINT_SECONDS,
                ),
            ),
        )
        self.check_samples = len(capture)
        self.touch_up_samples = len(touch_up)

    def to_dict(self) -> dict[str, Any]:
        return {
            "overhead_seconds": round(self.overhead_seconds, 5),
            "check_capture_seconds": round(self.check_capture_seconds, 3),
            "check_repaint_fraction": round(self.check_repaint_fraction, 4),
            "touch_up_fraction": round(self.touch_up_fraction, 4),
            "samples": self.samples,
            "check_samples": self.check_samples,
            "touch_up_samples": self.touch_up_samples,
            "history": self.history[-20:],
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    def observe(
        self,
        *,
        predicted_seconds: float,
        actual_seconds: float,
        strokes: int,
        colors_checked: int = 0,
        check_capture_seconds: float = 0.0,
        check_repaint_seconds: float = 0.0,
        touch_up_seconds: float | None = None,
    ) -> bool:
        """Fold one run into the learned figures.

        ``predicted_seconds`` must have been computed with the overhead this
        object held at the time, so the residual per stroke is exactly the
        correction the overhead needs; ``actual_seconds`` is the clock the
        artwork's own strokes took, with the checking left out.  The checks
        are described by how many colors had a capture, the time those
        captures took and the time spent repainting from them; the touch-up
        pass by the time it took in all, given only when it ran to its end.
        Returns whether anything was learned.
        """

        if strokes < LEARN_MIN_STROKES or predicted_seconds <= 0 or actual_seconds <= 0:
            return False
        measured = self.overhead_seconds + (actual_seconds - predicted_seconds) / strokes
        entry: dict[str, float] = {
            "predicted_seconds": round(predicted_seconds, 1),
            "actual_seconds": round(actual_seconds, 1),
            "strokes": strokes,
            "measured_overhead_seconds": round(
                min(MAX_STROKE_OVERHEAD_SECONDS, max(0.0, measured)), 5
            ),
        }
        believed_paint = actual_seconds >= LEARN_MIN_PAINT_SECONDS
        if colors_checked >= LEARN_MIN_CHECKED_COLORS:
            entry["colors_checked"] = int(colors_checked)
            entry["check_capture_seconds"] = round(max(0.0, check_capture_seconds), 2)
            if believed_paint:
                entry["check_repaint_seconds"] = round(max(0.0, check_repaint_seconds), 2)
        if touch_up_seconds is not None and believed_paint:
            entry["touch_up_seconds"] = round(max(0.0, touch_up_seconds), 2)
        self.history.append(entry)
        self.history = self.history[-20:]
        self.samples += 1
        self._recompute()
        # Kept for readers of the file: the overhead after this run.
        entry["overhead_seconds"] = round(self.overhead_seconds, 5)
        return True

    def check_seconds(self, colors_checked: int, paint_seconds: float) -> float:
        """What checking ``colors_checked`` colors over ``paint_seconds`` of painting costs."""

        return (
            max(0, int(colors_checked)) * self.check_capture_seconds
            + max(0.0, float(paint_seconds)) * self.check_repaint_fraction
        )

    def touch_up_seconds(self, paint_seconds: float) -> float:
        """What the touch-up pass after ``paint_seconds`` of painting costs."""

        return max(0.0, float(paint_seconds)) * self.touch_up_fraction


@dataclass(frozen=True, slots=True)
class PhaseTiming:
    """How a painting phase's clock compared with its prediction.

    ``checking_seconds`` is the part of the clock spent checking colors as
    they went down and repainting what missed; of it,
    ``check_capture_seconds`` went on the first capture of each of the
    ``colors_checked`` colors, and the rest on repainting and re-capturing.
    The time left is predicted from the whole clock, since the job will go
    on spending it; the per-stroke overhead is learned without it, since it
    is the game's dropped presses on this sign and not this machine's cost
    of a stroke.
    """

    predicted_seconds: float
    actual_seconds: float
    strokes: int
    checking_seconds: float = 0.0
    colors_checked: int = 0
    check_capture_seconds: float = 0.0


@dataclass(frozen=True, slots=True)
class RunEstimate:
    """A whole run's predicted seconds, part by part."""

    paint: float
    checks: float = 0.0
    touch_up: float = 0.0
    calibration: float = 0.0
    countdown: float = 0.0

    @property
    def total(self) -> float:
        return self.paint + self.checks + self.touch_up + self.calibration + self.countdown


@dataclass(frozen=True, slots=True)
class TouchUpTiming:
    """How long the touch-up pass at the end of a run took, when it ran to its end.

    ``passes`` counts the captures taken; a pass that found nothing to
    repaint still counts, and its few seconds are the honest cost of a
    clean sign.
    """

    seconds: float
    passes: int


__all__ = [
    "BRUSH_CALIBRATION_SECONDS",
    "CHECK_CAPTURE_SECONDS_DEFAULT",
    "CHECK_REPAINT_FRACTION_DEFAULT",
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
    "RunEstimate",
    "SETTLE_FLOOR_SECONDS",
    "SHORT_RUN_TEXELS",
    "STROKE_GAP_FLOOR_SECONDS",
    "StrokePace",
    "StrokeTiming",
    "TIMING_FLOORS",
    "TOUCH_UP_FRACTION_DEFAULT",
    "TouchUpTiming",
    "estimate_plan_seconds",
    "fields_below_floor",
    "floored",
    "remaining_seconds",
    "stroke_pace",
]
