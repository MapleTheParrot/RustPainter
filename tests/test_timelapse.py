from __future__ import annotations

from PIL import Image

from app.models import ScreenRect
from app.timelapse import TimelapseRecorder


REGION = ScreenRect(100, 100, 32, 16)


def test_frames_are_saved_numbered_into_one_session_folder(tmp_path) -> None:
    captured: list[ScreenRect] = []

    def capture(region):
        captured.append(region)
        return Image.new("RGB", (region.width, region.height), (20, 120, 60))

    recorder = TimelapseRecorder(
        tmp_path, REGION, capture=capture, session_name="job-1"
    )
    first = recorder.capture_frame()
    second = recorder.capture_frame()

    assert captured == [REGION, REGION]
    assert recorder.frame_count == 2
    assert first is not None and first.name == "frame_00001.png"
    assert second is not None and second.name == "frame_00002.png"
    assert first.parent == tmp_path / "job-1"
    with Image.open(first) as saved:
        assert saved.size == (REGION.width, REGION.height)


def test_a_failed_capture_is_skipped_not_raised(tmp_path) -> None:
    attempts: list[int] = []

    def capture(region):
        attempts.append(1)
        if len(attempts) == 1:
            raise OSError("screen went away")
        return Image.new("RGB", (region.width, region.height))

    recorder = TimelapseRecorder(
        tmp_path, REGION, capture=capture, session_name="job-2"
    )
    assert recorder.capture_frame() is None
    assert recorder.frame_count == 0

    # The recorder recovers on the next tick and numbering stays gapless.
    path = recorder.capture_frame()
    assert path is not None and path.name == "frame_00001.png"
    assert recorder.frame_count == 1


def test_capture_guard_skips_a_frame_without_touching_the_screen(tmp_path) -> None:
    allowed = False
    captures: list[ScreenRect] = []

    recorder = TimelapseRecorder(
        tmp_path,
        REGION,
        capture=lambda region: captures.append(region) or Image.new("RGB", (32, 16)),
        capture_allowed=lambda: allowed,
        session_name="guarded",
    )

    assert recorder.capture_frame() is None
    assert captures == []
    assert recorder.frame_count == 0

    allowed = True
    assert recorder.capture_frame() is not None
    assert captures == [REGION]
