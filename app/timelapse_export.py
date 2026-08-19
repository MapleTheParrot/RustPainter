"""Assemble a recorded timelapse session into one playable video file.

A session on disk is a folder of numbered PNG frames, which is the right
thing to record - lossless, resumable, and impossible to corrupt halfway
through a paint job - and the wrong thing to keep.  This module turns one
into a single file that plays anywhere.

MJPEG AVI is written directly here rather than shelled out to an encoder,
because the alternative is telling a user who just watched their sign get
painted that they need to install ffmpeg first.  The container is a few
fixed-size headers and an index, and every frame is an ordinary JPEG, so the
result plays in Windows' own players and in VLC without a codec pack.  When
ffmpeg *is* on the machine it is offered as well, because MP4 is smaller and
travels better.
"""

from __future__ import annotations

import io
import logging
import os
import shutil
import struct
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Iterable, Sequence

from PIL import Image


LOGGER = logging.getLogger("rust_painter.timelapse")

FRAME_GLOB = "frame_*.png"

# Slow enough that a sign appearing stroke by stroke is watchable, fast enough
# that an hour-long job is over in a minute or two.
DEFAULT_FRAME_RATE = 15

MIN_FRAME_RATE = 1
MAX_FRAME_RATE = 60

# High enough that JPEG artifacts stay well under the sign's own texture noise,
# low enough that a thousand-frame session is not a gigabyte.
_JPEG_QUALITY = 90

ProgressCallback = Callable[[int, int], None]
CancelCallback = Callable[[], bool]


class ExportError(RuntimeError):
    """Raised when a session cannot be turned into a video file."""


class ExportCancelled(ExportError):
    """Raised when the caller asked for the export to stop."""


@dataclass(frozen=True, slots=True)
class VideoFormat:
    """One offered output container."""

    key: str
    label: str
    suffix: str
    # Named so a caller can explain why a format is missing instead of hiding it.
    requirement: str = ""

    @property
    def filter_text(self) -> str:
        return f"{self.label} (*{self.suffix})"


AVI_FORMAT = VideoFormat("avi", "AVI video", ".avi")
MP4_FORMAT = VideoFormat("mp4", "MP4 video", ".mp4", requirement="ffmpeg")
GIF_FORMAT = VideoFormat("gif", "Animated GIF", ".gif")

_ALL_FORMATS = (MP4_FORMAT, AVI_FORMAT, GIF_FORMAT)


def session_frames(directory: Path | str) -> list[Path]:
    """Every frame of one session, in the order it was captured."""

    try:
        return sorted(Path(directory).glob(FRAME_GLOB))
    except OSError:
        LOGGER.exception("Could not list the frames of %s", directory)
        return []


def find_ffmpeg() -> str | None:
    """Locate an ffmpeg binary, preferring an explicitly configured one."""

    override = os.environ.get("RUST_PAINTER_FFMPEG", "").strip()
    if override:
        candidate = Path(override).expanduser()
        if candidate.is_file():
            return str(candidate)
        LOGGER.warning("RUST_PAINTER_FFMPEG does not point at a file: %s", override)
    found = shutil.which("ffmpeg")
    if found:
        return found
    # A frozen build may ship the encoder beside the executable.
    bundled = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
    for name in ("ffmpeg.exe", "ffmpeg"):
        candidate = bundled / name
        if candidate.is_file():
            return str(candidate)
    return None


def available_formats() -> tuple[VideoFormat, ...]:
    """The formats this machine can actually write, best first."""

    always = tuple(item for item in _ALL_FORMATS if not item.requirement)
    return (MP4_FORMAT, *always) if find_ffmpeg() is not None else always


def format_for(key: str) -> VideoFormat:
    for item in _ALL_FORMATS:
        if item.key == key:
            return item
    raise ExportError(f"Unknown video format {key!r}")


def format_for_suffix(suffix: str) -> VideoFormat | None:
    wanted = suffix.lower()
    for item in _ALL_FORMATS:
        if item.suffix == wanted:
            return item
    return None


def export_session(
    frames: Sequence[Path],
    destination: Path | str,
    *,
    frame_rate: int = DEFAULT_FRAME_RATE,
    video_format: VideoFormat | str = AVI_FORMAT,
    on_progress: ProgressCallback | None = None,
    should_cancel: CancelCallback | None = None,
) -> Path:
    """Write ``frames`` to ``destination`` as one video file.

    A partially written file is removed when the export fails or is cancelled,
    so a half-encoded video can never be mistaken for a finished one.
    """

    chosen = (
        video_format
        if isinstance(video_format, VideoFormat)
        else format_for(video_format)
    )
    if not frames:
        raise ExportError("This recording has no frames to export")
    rate = max(MIN_FRAME_RATE, min(MAX_FRAME_RATE, int(frame_rate)))
    target = Path(destination)
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        if chosen.key == "mp4":
            _encode_with_ffmpeg(frames, target, rate, on_progress, should_cancel)
        elif chosen.key == "gif":
            _encode_gif(frames, target, rate, on_progress, should_cancel)
        else:
            _encode_mjpeg_avi(frames, target, rate, on_progress, should_cancel)
    except BaseException:
        _discard(target)
        raise
    LOGGER.info(
        "Exported %d frames at %d fps to %s (%.1f MB)",
        len(frames),
        rate,
        target,
        target.stat().st_size / (1024 * 1024),
    )
    return target


def _discard(path: Path) -> None:
    try:
        path.unlink(missing_ok=True)
    except OSError:
        LOGGER.warning("Could not remove the incomplete export %s", path)


def _check_cancelled(should_cancel: CancelCallback | None) -> None:
    if should_cancel is not None and should_cancel():
        raise ExportCancelled("The export was cancelled")


def _report(on_progress: ProgressCallback | None, done: int, total: int) -> None:
    if on_progress is not None:
        on_progress(done, total)


def _open_frame(path: Path, size: tuple[int, int] | None) -> Image.Image:
    """Load one frame as RGB at the session's frame size.

    A frame is normally exactly the recorded size, or one pixel over on an
    axis that had to be rounded down to an even number.  Anything further off
    means the folder was edited by hand: resizing rather than refusing keeps
    one stray file from costing the user the whole recording.
    """

    with Image.open(path) as opened:
        image = opened.convert("RGB")
    if size is None or image.size == size:
        return image
    width, height = size
    if 0 <= image.width - width <= 1 and 0 <= image.height - height <= 1:
        return image.crop((0, 0, width, height))
    LOGGER.warning("Frame %s is %s, not %s; resizing it", path.name, image.size, size)
    return image.resize(size, Image.Resampling.LANCZOS)


def _frame_size(first: Path) -> tuple[int, int]:
    with Image.open(first) as opened:
        width, height = opened.size
    # Even dimensions keep every consumer happy, including the chroma
    # subsampling an MP4 encoder applies later.
    width -= width % 2
    height -= height % 2
    if width < 2 or height < 2:
        raise ExportError("The recorded frames are too small to make a video")
    return width, height


# ------------------------------------------------------------------ MJPEG AVI

_AVIF_HASINDEX = 0x00000010
_AVIIF_KEYFRAME = 0x00000010

# The 'strl' list holds the stream header and its format; 'hdrl' holds the main
# header plus that list.  Both are fixed size, which is what lets the frame
# count be patched in afterwards instead of buffering the whole video.
_STRL_SIZE = 4 + (8 + 56) + (8 + 40)
_HDRL_SIZE = 4 + (8 + 56) + (8 + _STRL_SIZE)

# Byte offsets of the fields patched once every frame has been written.
_RIFF_SIZE_OFFSET = 4
_AVIH_OFFSET = 32
_STRH_OFFSET = _AVIH_OFFSET + 56 + 8 + 4 + 8
_MOVI_SIZE_OFFSET = _AVIH_OFFSET + 56 + 8 + _STRL_SIZE + 4


def _write_chunk_header(stream: BinaryIO, fourcc: bytes, size: int) -> None:
    stream.write(fourcc)
    stream.write(struct.pack("<I", size))


def _avih(
    width: int, height: int, frame_rate: int, frames: int, buffer_size: int
) -> bytes:
    return struct.pack(
        "<10I16x",
        max(1, round(1_000_000 / frame_rate)),
        min(0xFFFFFFFF, buffer_size * frame_rate),
        0,
        _AVIF_HASINDEX,
        frames,
        0,
        1,
        buffer_size,
        width,
        height,
    )


def _strh(
    width: int, height: int, frame_rate: int, frames: int, buffer_size: int
) -> bytes:
    return struct.pack(
        "<4s4sI2H8I4H",
        b"vids",
        b"MJPG",
        0,
        0,
        0,
        0,
        1,
        frame_rate,
        0,
        frames,
        buffer_size,
        0xFFFFFFFF,
        0,
        0,
        0,
        width,
        height,
    )


def _strf(width: int, height: int) -> bytes:
    return struct.pack(
        "<I2i2H4sI2i2I",
        40,
        width,
        height,
        1,
        24,
        b"MJPG",
        width * height * 3,
        0,
        0,
        0,
        0,
    )


def _encode_mjpeg_avi(
    frames: Sequence[Path],
    destination: Path,
    frame_rate: int,
    on_progress: ProgressCallback | None,
    should_cancel: CancelCallback | None,
) -> None:
    """Write the frames as a Motion JPEG AVI, using no external encoder."""

    width, height = _frame_size(frames[0])
    total = len(frames)
    index: list[tuple[int, int]] = []
    with destination.open("wb") as stream:
        stream.write(b"RIFF")
        stream.write(struct.pack("<I", 0))  # patched below
        stream.write(b"AVI ")
        _write_chunk_header(stream, b"LIST", _HDRL_SIZE)
        stream.write(b"hdrl")
        _write_chunk_header(stream, b"avih", 56)
        stream.write(_avih(width, height, frame_rate, 0, 0))
        _write_chunk_header(stream, b"LIST", _STRL_SIZE)
        stream.write(b"strl")
        _write_chunk_header(stream, b"strh", 56)
        stream.write(_strh(width, height, frame_rate, 0, 0))
        _write_chunk_header(stream, b"strf", 40)
        stream.write(_strf(width, height))
        _write_chunk_header(stream, b"LIST", 0)  # patched below
        # Index offsets are measured from the 'movi' fourcc itself, which is
        # the convention every player in circulation expects.
        movi_origin = stream.tell()
        stream.write(b"movi")

        largest = 0
        for done, path in enumerate(frames, start=1):
            _check_cancelled(should_cancel)
            payload = _jpeg_bytes(_open_frame(path, (width, height)))
            largest = max(largest, len(payload))
            index.append((stream.tell() - movi_origin, len(payload)))
            _write_chunk_header(stream, b"00dc", len(payload))
            stream.write(payload)
            if len(payload) % 2:
                stream.write(b"\x00")
            _report(on_progress, done, total)

        movi_size = stream.tell() - movi_origin
        _write_chunk_header(stream, b"idx1", len(index) * 16)
        for offset, length in index:
            stream.write(
                struct.pack("<4s3I", b"00dc", _AVIIF_KEYFRAME, offset, length)
            )
        file_size = stream.tell()

        stream.seek(_RIFF_SIZE_OFFSET)
        stream.write(struct.pack("<I", file_size - 8))
        stream.seek(_AVIH_OFFSET)
        stream.write(_avih(width, height, frame_rate, total, largest))
        stream.seek(_STRH_OFFSET)
        stream.write(_strh(width, height, frame_rate, total, largest))
        stream.seek(_MOVI_SIZE_OFFSET)
        stream.write(struct.pack("<I", movi_size))


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=_JPEG_QUALITY, subsampling=0)
    return buffer.getvalue()


# ------------------------------------------------------------------------ GIF


def _encode_gif(
    frames: Sequence[Path],
    destination: Path,
    frame_rate: int,
    on_progress: ProgressCallback | None,
    should_cancel: CancelCallback | None,
) -> None:
    size = _frame_size(frames[0])
    total = len(frames)

    def remaining() -> Iterable[Image.Image]:
        for done, path in enumerate(frames[1:], start=2):
            _check_cancelled(should_cancel)
            yield _open_frame(path, size)
            _report(on_progress, done, total)

    _check_cancelled(should_cancel)
    first = _open_frame(frames[0], size)
    _report(on_progress, 1, total)
    first.save(
        destination,
        format="GIF",
        save_all=True,
        append_images=remaining(),
        duration=max(20, round(1000 / frame_rate)),
        loop=0,
        disposal=1,
    )


# --------------------------------------------------------------------- ffmpeg


def _encode_with_ffmpeg(
    frames: Sequence[Path],
    destination: Path,
    frame_rate: int,
    on_progress: ProgressCallback | None,
    should_cancel: CancelCallback | None,
) -> None:
    """Pipe the frames through ffmpeg to get an H.264 MP4."""

    executable = find_ffmpeg()
    if executable is None:
        raise ExportError(
            "MP4 export needs ffmpeg. Install it and put it on PATH, or export "
            "an AVI instead."
        )
    width, height = _frame_size(frames[0])
    total = len(frames)
    command = [
        executable,
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "image2pipe",
        "-framerate",
        str(frame_rate),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "20",
        "-pix_fmt",
        "yuv420p",
        "-vf",
        f"scale={width}:{height}",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    # CREATE_NO_WINDOW; an encoder console flashing over Rust would be its own
    # small disaster during a paint job.
    creation_flags = 0x08000000 if os.name == "nt" else 0
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=creation_flags,
    )
    stdin = process.stdin
    if stdin is None:  # pragma: no cover - Popen always provides the pipe
        process.kill()
        raise ExportError("ffmpeg could not be given the frames")
    try:
        for done, path in enumerate(frames, start=1):
            _check_cancelled(should_cancel)
            stdin.write(path.read_bytes())
            _report(on_progress, done, total)
        stdin.close()
    except ExportCancelled:
        process.kill()
        _drain(process)
        raise
    except (BrokenPipeError, OSError) as exc:
        process.kill()
        detail = _drain(process)
        raise ExportError(f"ffmpeg stopped reading frames: {detail or exc}") from exc
    detail = _drain(process)
    if process.returncode != 0:
        raise ExportError(
            f"ffmpeg failed: {detail or f'exit code {process.returncode}'}"
        )


def _drain(process: "subprocess.Popen[bytes]") -> str:
    try:
        _stdout, stderr = process.communicate(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        return "ffmpeg did not exit"
    return (stderr or b"").decode("utf-8", "replace").strip()


__all__ = [
    "AVI_FORMAT",
    "DEFAULT_FRAME_RATE",
    "ExportCancelled",
    "ExportError",
    "FRAME_GLOB",
    "GIF_FORMAT",
    "MAX_FRAME_RATE",
    "MIN_FRAME_RATE",
    "MP4_FORMAT",
    "VideoFormat",
    "available_formats",
    "export_session",
    "find_ffmpeg",
    "format_for",
    "format_for_suffix",
    "session_frames",
]
