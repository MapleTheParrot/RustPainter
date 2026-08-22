from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.models import ColorGroup, PaintPlan, Stroke
from app.resume_record import (
    ResumeRecord,
    ResumeRecordStore,
    advanced,
    paint_plan_prefix,
    plan_fingerprint,
    plan_prefix_labels,
    record_for_job,
    stroke_position,
)


def _plan(*, shuffle: bool = False, color=(10, 20, 30), diameter: int = 1) -> PaintPlan:
    first = tuple(Stroke(x, 0, x + 1, 0) for x in range(0, 8, 2))
    second = tuple(Stroke(0, y, 3, y) for y in range(1, 4))
    groups = (
        ColorGroup(color, first, 8, brush_diameter=diameter),
        ColorGroup((200, 100, 0), second, 12),
    )
    if shuffle:
        groups = groups[::-1]
    return PaintPlan(8, 4, groups)


def test_a_fingerprint_names_a_stroke_order_and_nothing_else() -> None:
    assert plan_fingerprint(_plan()) == plan_fingerprint(_plan())
    assert plan_fingerprint(_plan()) != plan_fingerprint(_plan(shuffle=True))
    assert plan_fingerprint(_plan()) != plan_fingerprint(_plan(color=(11, 20, 30)))
    assert plan_fingerprint(_plan()) != plan_fingerprint(_plan(diameter=3))
    moved = _plan()
    nudged = PaintPlan(
        moved.width,
        moved.height,
        (
            ColorGroup(
                moved.color_groups[0].color,
                (Stroke(1, 0, 2, 0),) + moved.color_groups[0].strokes[1:],
                8,
            ),
            moved.color_groups[1],
        ),
    )
    assert plan_fingerprint(moved) != plan_fingerprint(nudged)
    # Sixty-four hex digits: a whole SHA-256, so two plans never collide.
    assert len(plan_fingerprint(moved)) == 64


def test_a_record_round_trips_through_the_store(tmp_path: Path) -> None:
    plan = _plan()
    store = ResumeRecordStore(tmp_path / "resume")
    record = record_for_job(
        plan,
        profile=type("P", (), {"id": "abc", "name": "XXL"})(),
        image_path=tmp_path / "cat.png",
        settings={"painting": {"brush_size": 1.0}},
    )
    assert record.fingerprint == plan_fingerprint(plan)
    assert record.total_strokes == plan.stroke_count
    assert record.completed_strokes == 0 and not record.resumable

    moved = advanced(record, completed_strokes=5, color_index=2)
    path = store.save(moved)
    assert path.parent == store.directory
    assert not path.with_suffix(".json.tmp").exists()
    loaded = store.load(record.fingerprint)
    assert loaded is not None
    assert loaded.completed_strokes == 5 and loaded.color_index == 2
    assert loaded.state == "running" and loaded.resumable
    assert loaded.profile_id == "abc" and loaded.profile_name == "XXL"
    assert loaded.image_path == str(tmp_path / "cat.png")
    assert loaded.settings == {"painting": {"brush_size": 1.0}}
    assert loaded.percent == pytest.approx(5 * 100.0 / 7)
    assert "stroke 5 of 7" in loaded.describe()

    stopped = advanced(
        loaded,
        completed_strokes=6,
        color_index=2,
        state="paused",
        reason="painting UI not found - open the sign again and resume",
        interrupted_by_ui_loss=True,
    )
    store.save(stopped)
    again = store.load(record.fingerprint)
    assert again is not None
    assert again.interrupted_by_ui_loss and again.state == "paused"
    assert "painting UI not found" in again.describe()

    done = advanced(again, completed_strokes=7, color_index=2, state="completed", finished=True)
    store.save(done)
    final = store.load(record.fingerprint)
    assert final is not None and final.finished and not final.resumable


def test_the_store_offers_only_the_plan_a_record_was_written_for(tmp_path: Path) -> None:
    store = ResumeRecordStore(tmp_path / "resume")
    assert store.load(plan_fingerprint(_plan())) is None
    assert store.latest_resumable() is None

    first = advanced(record_for_job(_plan()), completed_strokes=3, color_index=1)
    store.save(first)
    other = plan_fingerprint(_plan(shuffle=True))
    assert store.load(other) is None
    # The record is still the latest interrupted job, for the warning.
    latest = store.latest_resumable(excluding=(other,))
    assert latest is not None and latest.fingerprint == first.fingerprint
    assert store.latest_resumable(excluding=(first.fingerprint,)) is None

    # A file that does not parse is skipped rather than trusted.
    (store.directory / "broken.json").write_text("{not json", encoding="utf-8")
    assert [record.fingerprint for record in store.records()] == [first.fingerprint]
    # A file whose content names another plan is not offered under this one.
    impostor = store.path_for(other)
    impostor.write_text(json.dumps(first.to_dict()), encoding="utf-8")
    assert store.load(other) is None


def test_old_records_missing_fields_still_load() -> None:
    record = ResumeRecord.from_dict(
        {"fingerprint": "f" * 64, "total_strokes": "10", "completed_strokes": 4}
    )
    assert record.total_strokes == 10 and record.completed_strokes == 4
    assert record.settings == {} and record.state == "running"
    assert record.resumable


def test_the_prefix_shows_the_sign_that_far_into_the_plan() -> None:
    plan = _plan(diameter=3)
    labels = plan_prefix_labels(plan, 0)
    assert not labels.any()
    # Two strokes of the first group: a three-cell band on row 0, so rows 0
    # and 1 are covered (row -1 is off the sign) across x 0..1 and 2..3.
    labels = plan_prefix_labels(plan, 2)
    assert labels[0, :4].tolist() == [1, 1, 1, 1]
    assert labels[1, :4].tolist() == [1, 1, 1, 1]
    assert not labels[0, 4:].any() and not labels[2:].any()
    # Into the second group: its first row stroke overwrites the band.
    labels = plan_prefix_labels(plan, 5)
    assert labels[1, :4].tolist() == [2, 2, 2, 2]
    assert labels[0, :4].tolist() == [1, 1, 1, 1]
    # Everything, and more than everything, is the whole plan.
    whole = plan_prefix_labels(plan, plan.stroke_count)
    assert np.array_equal(whole, plan_prefix_labels(plan, plan.stroke_count + 50))

    base = np.zeros((4, 8, 3), dtype=np.uint8)
    painted = paint_plan_prefix(base, plan, 5)
    assert tuple(painted[1, 0]) == (200, 100, 0)
    assert tuple(painted[0, 0]) == (10, 20, 30)
    assert tuple(painted[3, 7]) == (0, 0, 0)
    assert not base.any(), "the base is copied, not painted on"


def test_stroke_position_names_the_group_and_stroke_a_count_lands_on() -> None:
    plan = _plan()
    assert stroke_position(plan, 0) == (1, 0)
    assert stroke_position(plan, 3) == (1, 3)
    assert stroke_position(plan, 4) == (2, 0)
    assert stroke_position(plan, 7) == (2, 3)
    assert stroke_position(plan, 99) == (2, 3)
    assert stroke_position(PaintPlan(1, 1, ()), 0) == (0, 0)
