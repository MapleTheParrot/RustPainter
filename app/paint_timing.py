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
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Mapping

from .models import PaintPlan, RGBColor

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .painter import PainterSettings


LOGGER = logging.getLogger("rust_painter.timing")

# Rust has been observed running its painting UI at 15 FPS (a 67 ms frame).
# A press and release inside one frame can be sampled as nothing, so the
# painter keeps every stroke's button down until the press has lasted this
# long.  Slightly over a frame, so a press that starts late in one frame
# still straddles the boundary into the next.
MIN_PRESS_SECONDS = 0.07

# Picker clicks are rare (a handful per color change) and a dropped one
# paints a whole color wrong, so they are held longer still.
PICKER_CLICK_HOLD_SECONDS = 0.09

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
            overhead_seconds=overhead_seconds,
            real_input=real_input,
        )

    def _held(self, seconds: float, floor: float) -> float:
        return max(seconds, floor) if self.real_input else seconds

    def stroke_seconds(self, screen_length_pixels: float) -> float:
        """One stroke of this on-screen length, start of press to next stroke."""

        if screen_length_pixels <= 0:
            press = self._held(self.mouse_down_duration_seconds, MIN_PRESS_SECONDS)
        else:
            press = self._held(
                screen_length_pixels / max(self.stroke_speed_pixels_per_second, 1e-9),
                MIN_PRESS_SECONDS,
            )
        return press + self.delay_between_strokes_seconds + self.overhead_seconds

    def color_change_seconds(self) -> float:
        """Two held picker clicks and the settle after each."""

        click = self._held(self.mouse_down_duration_seconds, PICKER_CLICK_HOLD_SECONDS)
        return (
            2 * click
            + self.delay_after_hue_seconds
            + self.delay_after_saturation_value_seconds
            + 2 * self.overhead_seconds
        )

    def brush_change_seconds(self) -> float:
        """Clicking into the Size field, retyping it, and committing."""

        settle = max(self.delay_after_brush_seconds, 0.05) if self.real_input else 0.0
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
    ) -> float:
        """Painting time for the whole plan, calibration and countdown excluded."""

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
                total += count * timing.stroke_seconds((cells - 1) * cell_width_pixels)
            total += timing.delay_between_colors_seconds
        return total


def estimate_plan_seconds(
    plan: PaintPlan,
    timing: StrokeTiming,
    cell_width_pixels: float,
    *,
    sizing: bool,
    profile: PlanProfile | None = None,
) -> float:
    profile = profile if profile is not None else PlanProfile.from_plan(plan)
    return profile.seconds(timing, cell_width_pixels, sizing=sizing)


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
    ) -> None:
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
            cost += timing.delay_between_colors_seconds
            group_costs.append(cost)
            costs = [
                timing.stroke_seconds(
                    (max(1, int(stroke.pixel_count)) - 1) * cell_width_pixels
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
    """How a painting phase's clock compared with its prediction."""

    predicted_seconds: float
    actual_seconds: float
    strokes: int


__all__ = [
    "BRUSH_CALIBRATION_SECONDS",
    "DEFAULT_STROKE_OVERHEAD_SECONDS",
    "KEY_GAP_SECONDS",
    "KEY_HOLD_SECONDS",
    "LearnedTiming",
    "MIN_PRESS_SECONDS",
    "PICKER_CLICK_HOLD_SECONDS",
    "PhaseTiming",
    "PlanProfile",
    "PlanWorkSchedule",
    "StrokeTiming",
    "estimate_plan_seconds",
    "remaining_seconds",
]
