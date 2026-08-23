"""Record everything one paint job needs to be diagnosed after the fact.

A sign that came out wrong is usually explained hours later from whatever
survived the run, and a folder of canvas frames cannot answer the questions
that decide the diagnosis: what was the plan, what order did the colors go
down in, how wide was the brush the job measured for itself, and what did the
game's UI look like while it painted.  Reconstructing those by hand gets the
answer roughly right and the details wrong - a replanned image whose text sits
eighteen rows off the one that was painted is worse than no reference at all.

So the job writes them down while it runs.  One folder per run holds the exact
planned image, the plan's structure, the measured brush, the settings and
profile as they were at the countdown, a full-screen capture of the game, and a
progress trace that maps any timelapse frame back to the stroke being painted
when it was taken.

Nothing here may interrupt painting.  Every entry point swallows and logs its
own failures, because a missing diagnostic is a nuisance and a paint job lost
four hours in is not.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

LOGGER = logging.getLogger("rust_painter.run_report")

RUN_REPORT_SCHEMA = 1

# Progress arrives many times a second; a trace that fine would be megabytes of
# noise.  One sample per this many seconds still lands several times between
# two timelapse frames at any interval a user would choose.
_SAMPLE_INTERVAL_SECONDS = 2.0


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass(slots=True)
class _Sample:
    elapsed: float
    clock: str
    state: str
    phase: str
    color_index: int
    total_colors: int
    completed_strokes: int
    total_strokes: int
    percent: float
    timelapse_frame: int


class RunReport:
    """One run's diagnostic folder, written incrementally as the job goes.

    The report is created the moment artwork starts going down and finalized
    whatever the outcome - completed, aborted, or failed.  An aborted run is
    the one most worth keeping, since a run nobody stopped rarely needs
    explaining.
    """

    def __init__(self, root: Path | str, *, session_name: str | None = None) -> None:
        stamp = session_name or time.strftime("%Y%m%d-%H%M%S")
        self.directory = Path(root) / stamp
        self._started = time.monotonic()
        self._document: dict[str, Any] = {
            "schemaVersion": RUN_REPORT_SCHEMA,
            "startedAt": _timestamp(),
        }
        self._samples: list[_Sample] = []
        self._last_sample = -_SAMPLE_INTERVAL_SECONDS
        self._closed = False
        try:
            self.directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            LOGGER.exception("Could not create the run report folder")

    # ------------------------------------------------------------- recording

    def record_context(
        self,
        *,
        settings: Mapping[str, Any] | None = None,
        profile: Any = None,
        timelapse_directory: Path | str | None = None,
        image_path: Path | str | None = None,
        dry_run: bool = False,
    ) -> None:
        """Freeze the inputs, which the live settings file will not remember.

        Settings are edited between runs, so reading them afterwards describes
        some later state and quietly misattributes it to this job.
        """

        try:
            self._document["dryRun"] = bool(dry_run)
            if image_path is not None:
                self._document["sourceImage"] = str(image_path)
            if settings is not None:
                self._document["settings"] = json.loads(json.dumps(settings, default=str))
            if profile is not None and hasattr(profile, "to_dict"):
                self._document["profile"] = profile.to_dict()
            if timelapse_directory is not None:
                self._document["timelapseDirectory"] = Path(timelapse_directory).name
        except Exception:
            LOGGER.exception("Could not record the run's context")

    def record_plan(self, plan: Any, processed: Any = None) -> None:
        """Write the plan's structure, and the exact image it reproduces.

        The image is the reference every later comparison needs: replanning it
        from the source afterwards reproduces the pipeline but not the fonts,
        the palette, or the settings of the day.
        """

        try:
            groups = []
            dabs = 0
            for order, group in enumerate(plan.color_groups):
                strokes = group.strokes
                dabs += sum(
                    1 for stroke in strokes if stroke.start_x == stroke.end_x
                )
                groups.append(
                    {
                        "order": order,
                        "color": [int(channel) for channel in group.color],
                        "strokes": len(strokes),
                        "cells": int(group.pixel_count),
                        "brushDiameter": int(getattr(group, "brush_diameter", 1)),
                    }
                )
            self._document["plan"] = {
                "width": int(plan.width),
                "height": int(plan.height),
                "colors": len(plan.color_groups),
                "strokes": int(plan.stroke_count),
                "singleCellStrokes": dabs,
                "paintedCells": int(plan.painted_pixels),
                "unpaintedCells": int(plan.unpainted_pixels),
                "colorGroups": groups,
            }
        except Exception:
            LOGGER.exception("Could not record the plan's structure")

        if processed is None:
            return
        try:
            image = getattr(processed, "image", processed)
            image.convert("RGBA").save(self.directory / "plan.png", format="PNG")
        except Exception:
            LOGGER.exception("Could not save the planned image")

    def record_brush(
        self,
        model: Any,
        *,
        canvas_height: float,
        canvas_width: float,
        plan_width: int,
        plan_height: int,
        texel_grid: Any = None,
    ) -> None:
        """Record the measured brush next to what the plan asked it to do.

        The pair is the whole diagnosis for a blurred sign: a brush that covers
        several logical cells repaints its neighbours on every stroke, and the
        last color to cross a cell is the one that keeps it.  The texel grid,
        when one was measured, says what the sign really resolves and where
        its cells sit - the numbers to hold a downloaded sign texture against.
        """

        try:
            self._document["texelGrid"] = (
                texel_grid.to_dict() if texel_grid is not None else None
            )
            if model is None:
                self._document["brush"] = None
                return
            pitch = min(canvas_width / plan_width, canvas_height / plan_height)
            smallest = float(model.fraction_for_size(1.0)) * canvas_height
            rows = float(model.sign_pixel_rows)
            self._document["brush"] = {
                "model": model.to_dict(),
                "signTextureRows": rows,
                "cellPitchPixels": pitch,
                "smallestBrushPixels": smallest,
                "smallestBrushCells": smallest / pitch if pitch > 0 else None,
                "planRowsOverSignRows": plan_height / rows if rows > 0 else None,
            }
        except Exception:
            LOGGER.exception("Could not record the measured brush")

    def record_confirmation(self, summary: Any) -> None:
        """Record what checking each color as it went down found and did.

        The numbers that say whether the game was dropping presses on this
        sign - how many cells missed on first reading, how many repaints it
        took, how far the press hold had to be raised - which is the first
        question about a sign that came out with holes.
        """

        try:
            if summary is None:
                self._document["confirmation"] = None
            elif hasattr(summary, "to_dict"):
                self._document["confirmation"] = summary.to_dict()
            else:
                self._document["confirmation"] = json.loads(json.dumps(summary, default=str))
        except Exception:
            LOGGER.exception("Could not record the color checks")

    def record_screen(self) -> None:
        """Capture the whole desktop once, as the artwork starts.

        This is the only record of what the game itself was set to - the brush
        shape and opacity, the size field, the selected color, how the sign was
        framed - none of which a canvas-only capture can show.
        """

        try:
            from .screen import capture_region, get_virtual_screen_rect

            capture_region(get_virtual_screen_rect()).save(
                self.directory / "screen_start.png", format="PNG"
            )
        except Exception:
            LOGGER.exception("Could not capture the screen for the run report")

    def record_canvas(self, canvas: Any, name: str) -> None:
        """Capture the sign itself, whatever the run's outcome was."""

        try:
            from .screen import capture_region

            capture_region(canvas).save(self.directory / f"{name}.png", format="PNG")
        except Exception:
            LOGGER.exception("Could not capture the canvas for the run report")

    def sample_progress(self, progress: Any, *, timelapse_frame: int = 0) -> None:
        """Trace where the job was, thinned to a readable rate.

        Carrying the timelapse frame count is what lets a frame that shows
        something wrong be traced back to the color and stroke that were being
        painted when it was taken.
        """

        if self._closed:
            return
        try:
            elapsed = time.monotonic() - self._started
            if elapsed - self._last_sample < _SAMPLE_INTERVAL_SECONDS:
                return
            self._last_sample = elapsed
            self._samples.append(
                _Sample(
                    elapsed=round(elapsed, 2),
                    clock=_timestamp(),
                    state=str(getattr(getattr(progress, "state", ""), "value", "")),
                    phase=str(getattr(progress, "phase", "")),
                    color_index=int(getattr(progress, "color_index", 0)),
                    total_colors=int(getattr(progress, "total_colors", 0)),
                    completed_strokes=int(getattr(progress, "completed_strokes", 0)),
                    total_strokes=int(getattr(progress, "total_strokes", 0)),
                    percent=round(float(getattr(progress, "percent", 0.0)), 3),
                    timelapse_frame=int(timelapse_frame),
                )
            )
        except Exception:
            LOGGER.exception("Could not sample progress for the run report")

    # -------------------------------------------------------------- finishing

    def finish(self, outcome: str, reason: str = "") -> Path | None:
        """Write the report out.  Safe to call more than once."""

        if self._closed:
            return self.directory
        self._closed = True
        try:
            self._document["finishedAt"] = _timestamp()
            self._document["elapsedSeconds"] = round(time.monotonic() - self._started, 2)
            self._document["outcome"] = outcome
            self._document["outcomeReason"] = reason
            self._write_samples()
            path = self.directory / "run.json"
            path.write_text(
                json.dumps(self._document, indent=2, default=str), encoding="utf-8"
            )
            LOGGER.info("Run report written to %s", self.directory)
            return self.directory
        except Exception:
            LOGGER.exception("Could not write the run report")
            return None

    def _write_samples(self) -> None:
        if not self._samples:
            return
        header = (
            "elapsed_seconds,clock,state,phase,color_index,total_colors,"
            "completed_strokes,total_strokes,percent,timelapse_frame\n"
        )
        rows = "".join(
            f"{s.elapsed},{s.clock},{s.state},{s.phase},{s.color_index},"
            f"{s.total_colors},{s.completed_strokes},{s.total_strokes},"
            f"{s.percent},{s.timelapse_frame}\n"
            for s in self._samples
        )
        (self.directory / "progress.csv").write_text(header + rows, encoding="utf-8")


__all__ = ["RUN_REPORT_SCHEMA", "RunReport"]
