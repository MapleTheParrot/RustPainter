from __future__ import annotations

import json

import numpy as np
from PIL import Image

from app.brush_calibration import BrushSizeModel
from app.models import ColorGroup, PaintPlan, ProcessedImage, Stroke
from app.run_report import RunReport


def _plan() -> PaintPlan:
    return PaintPlan(
        width=4,
        height=2,
        color_groups=(
            ColorGroup(
                color=(200, 30, 30),
                strokes=(Stroke(0, 0, 3, 0),),
                pixel_count=4,
            ),
            ColorGroup(
                color=(255, 0, 255),
                strokes=(Stroke(1, 1, 1, 1), Stroke(3, 1, 3, 1)),
                pixel_count=2,
            ),
        ),
        unpainted_pixels=2,
    )


def _processed() -> ProcessedImage:
    image = Image.new("RGBA", (4, 2), (200, 30, 30, 255))
    return ProcessedImage(image, np.ones((2, 4), dtype=bool), 8)


class _Progress:
    def __init__(self, strokes: int) -> None:
        self.state = type("S", (), {"value": "running"})()
        self.phase = "paint"
        self.color_index = 2
        self.total_colors = 2
        self.completed_strokes = strokes
        self.total_strokes = 3
        self.percent = 100.0 * strokes / 3


def test_the_report_records_the_plan_and_the_image_it_reproduces(tmp_path) -> None:
    report = RunReport(tmp_path, session_name="job-1")
    report.record_plan(_plan(), _processed())
    report.finish("completed")

    document = json.loads((tmp_path / "job-1" / "run.json").read_text())
    plan = document["plan"]
    assert (plan["width"], plan["height"]) == (4, 2)
    assert plan["strokes"] == 3
    # Two of the three strokes are single cells, which is what dominates a
    # run's wall clock and has to be visible without replanning.
    assert plan["singleCellStrokes"] == 2
    # Paint order is the whole basis of overpaint correctness, so groups are
    # recorded in the order they will be painted, not sorted or deduplicated.
    assert [group["color"] for group in plan["colorGroups"]] == [
        [200, 30, 30],
        [255, 0, 255],
    ]
    with Image.open(tmp_path / "job-1" / "plan.png") as saved:
        assert saved.size == (4, 2)


def test_the_report_states_the_brush_against_the_cells_it_must_fit(tmp_path) -> None:
    report = RunReport(tmp_path, session_name="job-2")
    # One Size unit paints a tenth of the sign, so the smallest brush is far
    # wider than a cell of a 100-row plan - the case that blurs a sign.
    model = BrushSizeModel(
        slope=0.1, intercept=0.0, samples=((1.0, 0.1), (2.0, 0.2))
    )
    report.record_brush(
        model,
        canvas_height=1000.0,
        canvas_width=1000.0,
        plan_width=100,
        plan_height=100,
    )
    report.finish("aborted", "user")

    brush = json.loads((tmp_path / "job-2" / "run.json").read_text())["brush"]
    assert brush["cellPitchPixels"] == 10.0
    assert brush["smallestBrushPixels"] == 100.0
    assert brush["smallestBrushCells"] == 10.0
    assert brush["signTextureRows"] == 10.0
    assert brush["planRowsOverSignRows"] == 10.0


def test_progress_samples_map_a_timelapse_frame_back_to_a_stroke(tmp_path) -> None:
    report = RunReport(tmp_path, session_name="job-3")
    report._last_sample = -1e9  # first sample is always taken
    report.sample_progress(_Progress(1), timelapse_frame=7)
    # Thinning drops anything arriving inside the sample interval, so a job
    # emitting progress many times a second cannot fill the disk.
    report.sample_progress(_Progress(2), timelapse_frame=7)
    report.finish("completed")

    rows = (tmp_path / "job-3" / "progress.csv").read_text().strip().splitlines()
    assert rows[0].startswith("elapsed_seconds,")
    assert len(rows) == 2
    assert rows[1].endswith(",7")
    assert ",1,3," in rows[1]


def test_a_failed_capture_never_stops_the_report(tmp_path, monkeypatch) -> None:
    report = RunReport(tmp_path, session_name="job-4")

    class _Broken:
        color_groups = ()

        def __getattr__(self, name):
            raise RuntimeError("plan went away")

    report.record_plan(_Broken())
    report.record_brush(
        None, canvas_height=1.0, canvas_width=1.0, plan_width=1, plan_height=1
    )
    written = report.finish("error", "something broke")

    assert written == tmp_path / "job-4"
    document = json.loads((tmp_path / "job-4" / "run.json").read_text())
    assert document["outcome"] == "error"
    assert document["brush"] is None


def test_finishing_twice_keeps_the_first_result(tmp_path) -> None:
    report = RunReport(tmp_path, session_name="job-5")
    report.finish("aborted", "user")
    report.finish("completed", "second call")

    document = json.loads((tmp_path / "job-5" / "run.json").read_text())
    assert document["outcome"] == "aborted"


def test_the_report_records_what_checking_each_color_found(tmp_path) -> None:
    from app.painter import ConfirmationSummary

    report = RunReport(tmp_path, session_name="job-checks")
    report.record_confirmation(
        ConfirmationSummary(
            colors=12,
            judged=4000,
            missed=1300,
            repainted_strokes=900,
            unrepaired=7,
            rounds=20,
            hold_boost_seconds=0.04,
        )
    )
    report.finish("completed")
    recorded = json.loads((tmp_path / "job-checks" / "run.json").read_text())
    assert recorded["confirmation"] == {
        "colorsChecked": 12,
        "cellsJudged": 4000,
        "cellsMissedFirstCheck": 1300,
        "repaintStrokes": 900,
        "cellsUnrepaired": 7,
        "repaintRounds": 20,
        "pressHoldBoostSeconds": 0.04,
        "skippedReason": "",
    }
