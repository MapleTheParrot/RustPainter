"""The AVI writer is hand-rolled, so its bytes are checked, not just its size.

A file that opens without error in Pillow proves nothing here: nothing in the
project reads AVI back.  These tests therefore walk the container the way a
player does - header, stream description, frame chunks, index - because the
failure mode of a wrong offset is a video that saves happily and then refuses
to play on the one machine the user wanted to show it on.
"""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from PIL import Image

from app.timelapse_export import (
    AVI_FORMAT,
    GIF_FORMAT,
    MP4_FORMAT,
    ExportCancelled,
    ExportError,
    available_formats,
    export_session,
    format_for,
    format_for_suffix,
    session_frames,
)


def _record(directory: Path, count: int, size: tuple[int, int] = (64, 32)) -> list[Path]:
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(1, count + 1):
        color = (index * 37 % 256, 90, 160)
        Image.new("RGB", size, color).save(directory / f"frame_{index:05d}.png")
    return session_frames(directory)


def _chunks(data: bytes) -> tuple[list[tuple[int, int]], int, int]:
    """Walk the movi list, returning (offset, length) per frame plus its bounds."""

    assert data[212:216] == b"LIST"
    movi_size = struct.unpack_from("<I", data, 216)[0]
    assert data[220:224] == b"movi"
    found: list[tuple[int, int]] = []
    position = 224
    while position < 220 + movi_size:
        assert data[position : position + 4] == b"00dc"
        length = struct.unpack_from("<I", data, position + 4)[0]
        found.append((position - 220, length))
        # Every frame is a whole JPEG, which is what makes the file playable
        # without a codec pack.
        assert data[position + 8 : position + 10] == b"\xff\xd8"
        position += 8 + length + (length % 2)
    return found, 220 + movi_size, position


def test_session_frames_are_listed_in_capture_order(tmp_path: Path) -> None:
    frames = _record(tmp_path / "job", 12)
    assert [path.name for path in frames] == [
        f"frame_{index:05d}.png" for index in range(1, 13)
    ]


def test_an_avi_declares_every_frame_it_contains(tmp_path: Path) -> None:
    frames = _record(tmp_path / "job", 5)
    destination = tmp_path / "out.avi"

    export_session(frames, destination, frame_rate=10, video_format=AVI_FORMAT)
    data = destination.read_bytes()

    assert data[:4] == b"RIFF" and data[8:12] == b"AVI "
    assert struct.unpack_from("<I", data, 4)[0] == len(data) - 8
    micro_seconds, _rate, _pad, flags, total, *_rest = struct.unpack_from(
        "<10I", data, 32
    )
    assert total == len(frames)
    assert micro_seconds == 100_000  # 10 fps
    assert flags & 0x10  # AVIF_HASINDEX, or players will not seek
    assert struct.unpack_from("<4s4s", data, 108) == (b"vids", b"MJPG")
    # dwLength, the fifth of the eight longs that follow the stream header's
    # type, handler, flags, priority, and language fields.
    assert struct.unpack_from("<8I", data, 108 + 16)[4] == len(frames)
    assert struct.unpack_from("<4s", data, 172 + 16)[0] == b"MJPG"


def test_the_avi_index_points_at_the_frames_it_claims(tmp_path: Path) -> None:
    """A wrong offset here is a file that plays as one frozen frame."""

    frames = _record(tmp_path / "job", 7)
    destination = tmp_path / "out.avi"

    export_session(frames, destination, video_format=AVI_FORMAT)
    data = destination.read_bytes()

    written, movi_end, walked_to = _chunks(data)
    assert len(written) == len(frames)
    assert walked_to == movi_end
    assert data[movi_end : movi_end + 4] == b"idx1"
    index_size = struct.unpack_from("<I", data, movi_end + 4)[0]
    assert index_size == len(frames) * 16
    indexed = [
        struct.unpack_from("<4s3I", data, movi_end + 8 + entry * 16)
        for entry in range(len(frames))
    ]
    assert all(identifier == b"00dc" and flags & 0x10 for identifier, flags, _, _ in indexed)
    assert [(offset, length) for _id, _flags, offset, length in indexed] == written
    assert movi_end + 8 + index_size == len(data)


def test_odd_sized_frames_are_trimmed_to_even_dimensions(tmp_path: Path) -> None:
    """Odd dimensions break the chroma subsampling every consumer applies."""

    frames = _record(tmp_path / "job", 3, size=(65, 33))
    destination = tmp_path / "out.avi"

    export_session(frames, destination, video_format=AVI_FORMAT)
    data = destination.read_bytes()

    assert struct.unpack_from("<10I", data, 32)[8:10] == (64, 32)
    assert struct.unpack_from("<I2i", data, 172)[1:] == (64, 32)


def test_a_gif_keeps_one_frame_per_capture(tmp_path: Path) -> None:
    frames = _record(tmp_path / "job", 6)
    destination = tmp_path / "out.gif"

    export_session(frames, destination, frame_rate=20, video_format=GIF_FORMAT)

    with Image.open(destination) as animation:
        assert animation.n_frames == len(frames)
        assert animation.size == (64, 32)
        assert animation.info["duration"] == 50


def test_a_cancelled_export_leaves_no_half_written_video(tmp_path: Path) -> None:
    """A truncated video that looks finished is worse than no video at all."""

    frames = _record(tmp_path / "job", 40)
    destination = tmp_path / "out.avi"
    seen: list[int] = []

    with pytest.raises(ExportCancelled):
        export_session(
            frames,
            destination,
            video_format=AVI_FORMAT,
            on_progress=lambda done, _total: seen.append(done),
            should_cancel=lambda: len(seen) >= 3,
        )

    assert seen == [1, 2, 3]
    assert not destination.exists()


def test_progress_counts_every_frame(tmp_path: Path) -> None:
    frames = _record(tmp_path / "job", 9)
    seen: list[tuple[int, int]] = []

    export_session(
        frames,
        tmp_path / "out.avi",
        video_format=AVI_FORMAT,
        on_progress=lambda done, total: seen.append((done, total)),
    )

    assert seen == [(index, 9) for index in range(1, 10)]


def test_an_empty_recording_is_refused_rather_than_written(tmp_path: Path) -> None:
    destination = tmp_path / "out.avi"
    with pytest.raises(ExportError, match="no frames"):
        export_session([], destination, video_format=AVI_FORMAT)
    assert not destination.exists()


def test_formats_are_resolved_by_key_and_by_suffix() -> None:
    assert format_for("avi") is AVI_FORMAT
    assert format_for_suffix(".GIF") is GIF_FORMAT
    assert format_for_suffix(".mkv") is None
    with pytest.raises(ExportError):
        format_for("webm")


def test_only_formats_this_machine_can_write_are_offered() -> None:
    """MP4 needs ffmpeg; AVI and GIF are written by RustPainter itself."""

    offered = available_formats()
    assert AVI_FORMAT in offered and GIF_FORMAT in offered
    assert (MP4_FORMAT in offered) == (MP4_FORMAT.requirement == "ffmpeg" and _has_ffmpeg())


def _has_ffmpeg() -> bool:
    from app.timelapse_export import find_ffmpeg

    return find_ffmpeg() is not None
