"""Where a painting got to, written down so it can be picked up again.

Rust saves the sign while the painting UI is open, so a job interrupted
hours in - a server restart, a kick, a crash - leaves nearly everything it
painted on the sign.  What it loses is its place.  This module keeps that
place: a small JSON record per plan, refreshed every few seconds while the
artwork goes down and stamped with the reason the moment a job stops,
from which a later job can start at the stroke this one was on.

A record is only meaningful for the plan it was written against.  Stroke
numbers index the plan's stroke order, and a plan made from the same image
with a different resolution, palette, or optimisation has a different
order, so every record carries a fingerprint of its plan's strokes and
colours and is offered only to a plan with the same one.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import struct
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .models import PaintPlan

LOGGER = logging.getLogger("rust_painter.resume")

RESUME_RECORD_SCHEMA = 1


def plan_fingerprint(plan: PaintPlan) -> str:
    """A hash of everything that gives a stroke index its meaning.

    The canvas size, and every group's colour, brush diameter, and strokes
    in order.  Two plans with the same fingerprint paint the same strokes
    in the same sequence, so a stroke count from one applies to the other.
    """

    digest = hashlib.sha256()
    digest.update(struct.pack("<II", int(plan.width), int(plan.height)))
    for group in plan.color_groups:
        color = tuple(int(channel) for channel in group.color)
        digest.update(
            struct.pack(
                "<BBBII",
                color[0],
                color[1],
                color[2],
                max(1, int(group.brush_diameter)),
                len(group.strokes),
            )
        )
        if group.strokes:
            coordinates = np.fromiter(
                (
                    value
                    for stroke in group.strokes
                    for value in (stroke.start_x, stroke.start_y, stroke.end_x, stroke.end_y)
                ),
                dtype=np.int32,
                count=len(group.strokes) * 4,
            )
            digest.update(coordinates.tobytes())
    return digest.hexdigest()


def _timestamp() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


@dataclass(frozen=True, slots=True)
class ResumeRecord:
    """One job's place in its plan."""

    fingerprint: str
    total_strokes: int
    completed_strokes: int = 0
    color_index: int = 0
    total_colors: int = 0
    plan_width: int = 0
    plan_height: int = 0
    profile_id: str | None = None
    profile_name: str | None = None
    image_path: str | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    # The painter's state and reason when the record was last written:
    # "running" while the job goes, then "paused", "aborted", "error", or
    # "completed" - with the painter's own words for why.
    state: str = "running"
    reason: str = ""
    interrupted_by_ui_loss: bool = False
    # The screen as it was when a guard paused the job, if one was taken:
    # what tripped the guard, to look at rather than guess.
    screenshot_path: str | None = None
    # A finished job's record is kept for the run's history but is never
    # offered as a place to resume from.
    finished: bool = False
    started_at: str = ""
    updated_at: str = ""

    @property
    def percent(self) -> float:
        if self.total_strokes <= 0:
            return 100.0
        return min(100.0, self.completed_strokes * 100.0 / self.total_strokes)

    @property
    def resumable(self) -> bool:
        return not self.finished and 0 < self.completed_strokes <= self.total_strokes

    def to_dict(self) -> dict[str, Any]:
        document = asdict(self)
        document["schemaVersion"] = RESUME_RECORD_SCHEMA
        return document

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "ResumeRecord":
        known = {name for name in cls.__dataclass_fields__}  # type: ignore[attr-defined]
        fields = {key: value[key] for key in known if key in value}
        fields["fingerprint"] = str(fields.get("fingerprint", ""))
        for name in (
            "total_strokes",
            "completed_strokes",
            "color_index",
            "total_colors",
            "plan_width",
            "plan_height",
        ):
            fields[name] = int(fields.get(name, 0) or 0)
        settings = fields.get("settings")
        fields["settings"] = dict(settings) if isinstance(settings, Mapping) else {}
        for name in ("interrupted_by_ui_loss", "finished"):
            fields[name] = bool(fields.get(name, False))
        return cls(**fields)

    def describe(self) -> str:
        """One line for a status label: where, why, and when."""

        place = (
            f"stroke {self.completed_strokes:,} of {self.total_strokes:,} "
            f"({self.percent:.0f}%)"
        )
        if self.finished:
            return f"finished, {place}"
        if self.state == "running":
            return f"last seen painting at {place}, {self.updated_at}"
        reason = f" - {self.reason}" if self.reason else ""
        return f"{self.state} at {place}{reason}, {self.updated_at}"


class ResumeRecordStore:
    """The records on disk, one file per plan fingerprint."""

    def __init__(self, directory: Path | str) -> None:
        self.directory = Path(directory)

    def path_for(self, fingerprint: str) -> Path:
        return self.directory / f"{fingerprint[:32]}.json"

    def load(self, fingerprint: str) -> ResumeRecord | None:
        path = self.path_for(fingerprint)
        if not path.exists():
            return None
        try:
            with path.open("r", encoding="utf-8") as handle:
                document = json.load(handle)
            record = ResumeRecord.from_dict(document)
        except (OSError, ValueError, TypeError):
            LOGGER.warning("Could not read the resume record %s", path, exc_info=True)
            return None
        return record if record.fingerprint == fingerprint else None

    def save(self, record: ResumeRecord) -> Path:
        """Write the record whole, so a crash mid-write leaves the old one."""

        path = self.path_for(record.fingerprint)
        self.directory.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".json.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(record.to_dict(), handle, indent=2)
        os.replace(temporary, path)
        return path

    def delete(self, fingerprint: str) -> bool:
        """Forget a record: the sign keeps its paint, only the place is lost."""

        path = self.path_for(fingerprint)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError:
            LOGGER.warning("Could not delete the resume record %s", path, exc_info=True)
            return False
        return True

    def records(self) -> list[ResumeRecord]:
        """Every readable record, newest first."""

        found: list[tuple[float, ResumeRecord]] = []
        try:
            paths = list(self.directory.glob("*.json"))
        except OSError:
            return []
        for path in paths:
            try:
                with path.open("r", encoding="utf-8") as handle:
                    record = ResumeRecord.from_dict(json.load(handle))
                found.append((path.stat().st_mtime, record))
            except (OSError, ValueError, TypeError):
                LOGGER.debug("Skipping unreadable resume record %s", path, exc_info=True)
        found.sort(key=lambda item: item[0], reverse=True)
        return [record for _, record in found]

    def latest_resumable(self, *, excluding: Iterable[str] = ()) -> ResumeRecord | None:
        """The most recent record a job could still be resumed from."""

        skip = set(excluding)
        for record in self.records():
            if record.resumable and record.fingerprint not in skip:
                return record
        return None


def record_for_job(
    plan: PaintPlan,
    *,
    profile: Any = None,
    image_path: Path | str | None = None,
    settings: Mapping[str, Any] | None = None,
    completed_strokes: int = 0,
) -> ResumeRecord:
    """A fresh record for a job that is about to paint ``plan``."""

    now = _timestamp()
    return ResumeRecord(
        fingerprint=plan_fingerprint(plan),
        total_strokes=plan.stroke_count,
        completed_strokes=int(completed_strokes),
        total_colors=len(plan.color_groups),
        plan_width=int(plan.width),
        plan_height=int(plan.height),
        profile_id=str(getattr(profile, "id", "") or "") or None,
        profile_name=str(getattr(profile, "name", "") or "") or None,
        image_path=str(image_path) if image_path else None,
        settings=dict(settings) if settings else {},
        started_at=now,
        updated_at=now,
    )


def advanced(
    record: ResumeRecord,
    *,
    completed_strokes: int,
    color_index: int,
    state: str = "running",
    reason: str = "",
    interrupted_by_ui_loss: bool = False,
    finished: bool = False,
    screenshot_path: str | None = None,
) -> ResumeRecord:
    """The record moved on to where the job is now.

    ``interrupted_by_ui_loss`` marks a stop the painter's UI guard called -
    the sign went away - as opposed to a hand on the mouse or a window in
    front; the record is the same either way, the label is not.  A
    screenshot, once attached, stays attached until a new one replaces it.
    """

    return replace(
        record,
        completed_strokes=int(completed_strokes),
        color_index=int(color_index),
        state=state,
        reason=reason,
        interrupted_by_ui_loss=bool(interrupted_by_ui_loss),
        finished=finished,
        screenshot_path=screenshot_path or record.screenshot_path,
        updated_at=_timestamp(),
    )


def plan_prefix_labels(plan: PaintPlan, stroke_count: int) -> np.ndarray:
    """Which group painted each cell after the plan's first ``stroke_count``.

    A ``(height, width)`` array of group numbers, 1-based, with 0 where
    nothing has been painted yet: the sign that far into the plan.  Band
    strokes cover their brush diameter across the stroke - the rows around
    a horizontal one - as the optimizer counted them, and no further along
    it.
    """

    height, width = int(plan.height), int(plan.width)
    labels = np.zeros((height, width), dtype=np.uint32)
    remaining = max(0, int(stroke_count))
    for group_number, group in enumerate(plan.color_groups, start=1):
        if remaining <= 0:
            break
        radius = (max(1, int(group.brush_diameter)) - 1) // 2
        for stroke in group.strokes[:remaining]:
            x0, x1 = sorted((int(stroke.start_x), int(stroke.end_x)))
            y0, y1 = sorted((int(stroke.start_y), int(stroke.end_y)))
            if y0 == y1:
                cells = ((x0, y0 - radius, x1, y1 + radius),)
            elif x0 == x1:
                cells = ((x0 - radius, y0, x1 + radius, y1),)
            else:
                # A diagonal: walk it a cell at a time.
                steps = max(x1 - x0, y1 - y0)
                cells = tuple(
                    (x - radius, y - radius, x + radius, y + radius)
                    for x, y in (
                        (
                            int(round(stroke.start_x + (stroke.end_x - stroke.start_x) * step / steps)),
                            int(round(stroke.start_y + (stroke.end_y - stroke.start_y) * step / steps)),
                        )
                        for step in range(steps + 1)
                    )
                )
            for left, top, right, bottom in cells:
                top, bottom = max(0, top), min(height, bottom + 1)
                left, right = max(0, left), min(width, right + 1)
                if top < bottom and left < right:
                    labels[top:bottom, left:right] = group_number
        remaining -= len(group.strokes)
    return labels


def paint_plan_prefix(base: np.ndarray, plan: PaintPlan, stroke_count: int) -> np.ndarray:
    """``base`` with the plan's first ``stroke_count`` strokes painted on it."""

    image = np.array(base, dtype=np.uint8, copy=True)
    labels = plan_prefix_labels(plan, stroke_count)
    for group_number, group in enumerate(plan.color_groups, start=1):
        painted = labels == group_number
        if painted.any():
            image[painted] = np.asarray(group.color, dtype=np.uint8)
    return image


def stroke_position(plan: PaintPlan, stroke_count: int) -> tuple[int, int]:
    """Which group (1-based) and stroke within it the count lands on.

    The position the next stroke would be painted at: ``(0, 0)`` for a
    plan with nothing to paint, and the last group's end when every stroke
    is behind the count.
    """

    remaining = max(0, int(stroke_count))
    group_index = 0
    for group_index, group in enumerate(plan.color_groups, start=1):
        if remaining < len(group.strokes):
            return group_index, remaining
        remaining -= len(group.strokes)
    last = plan.color_groups[-1] if plan.color_groups else None
    return group_index, (len(last.strokes) if last is not None else 0)


__all__ = [
    "RESUME_RECORD_SCHEMA",
    "ResumeRecord",
    "ResumeRecordStore",
    "advanced",
    "paint_plan_prefix",
    "plan_fingerprint",
    "plan_prefix_labels",
    "record_for_job",
    "stroke_position",
]
