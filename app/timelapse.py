"""Capture periodic frames of the sign while it is painted.

Each paint job gets its own timestamped session folder of numbered PNG
frames, ready to be assembled into a timelapse video with any external
tool. Capturing is deliberately tolerant: a failed frame is logged and
skipped rather than allowed to interrupt the paint job it documents.
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Callable

from .screen import RectangleLike, capture_region

if TYPE_CHECKING:
    from PIL.Image import Image


LOGGER = logging.getLogger("rust_painter.timelapse")


class TimelapseRecorder:
    """Save numbered PNG frames of one screen region into a session folder."""

    def __init__(
        self,
        root: Path | str,
        region: RectangleLike,
        *,
        capture: "Callable[[RectangleLike], Image] | None" = None,
        session_name: str | None = None,
    ) -> None:
        self.region = region
        self._capture = capture
        stamp = session_name or time.strftime("%Y%m%d-%H%M%S")
        self.directory = Path(root) / stamp
        self._lock = threading.Lock()
        self._busy = False
        self._frame_count = 0

    @property
    def frame_count(self) -> int:
        return self._frame_count

    def capture_frame(self) -> Path | None:
        """Capture and save one frame; ``None`` when skipped or failed.

        A capture already in flight makes this a no-op instead of queueing,
        so a slow disk can never pile up captures behind a fast interval.
        """

        with self._lock:
            if self._busy:
                return None
            self._busy = True
        try:
            capturer = self._capture if self._capture is not None else capture_region
            image = capturer(self.region)
            self.directory.mkdir(parents=True, exist_ok=True)
            path = self.directory / f"frame_{self._frame_count + 1:05d}.png"
            image.save(path, format="PNG")
            self._frame_count += 1
            return path
        except Exception:
            LOGGER.exception("Could not capture a timelapse frame")
            return None
        finally:
            with self._lock:
                self._busy = False


__all__ = ["TimelapseRecorder"]
