"""Rust's own export of the painting, read back as texel-exact ground truth.

The paint UI's download button writes the sign's texture to the desktop as
``artists_canvas_<size>_<w>x<h>_<unixtime>.png`` - every texel exactly as the
game holds it, alpha 0 where nothing was ever painted.  Measured against it,
a screenshot of the sign at 1.77 px per texel is nearly blind: on one
finished 1024x512 sign the screenshot-based check found 1,006 wrong texels
where the export showed 80,000.  So whenever the download button is
calibrated, probes and the touch-up pass read the export instead.

The file is the painter's own measurement, not the user's download: the
painter clicks the button, takes the one file that appears, and removes it
from the desktop, so a job that exports fifty times leaves nothing behind.
Files that were on the desktop before the click are never touched.
"""

from __future__ import annotations

import glob
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

LOGGER = logging.getLogger("rust_painter.export")

EXPORT_PATTERN = "artists_canvas_*.png"
# How long the game gets to write the file after the button is clicked, and
# how long the file gets to stop growing once it appears.
EXPORT_WAIT_SECONDS = 12.0
EXPORT_SETTLE_SECONDS = 0.4


def default_export_directory() -> Path:
    """Where Rust writes downloads: the user's desktop."""

    home = Path(os.path.expanduser("~"))
    for candidate in (home / "Desktop", home / "OneDrive" / "Desktop"):
        if candidate.is_dir():
            return candidate
    return home / "Desktop"


@dataclass(frozen=True, slots=True)
class SignExport:
    """One export: RGB per texel and whether each texel was ever painted."""

    rgb: np.ndarray  # (rows, columns, 3) float32
    painted: np.ndarray  # (rows, columns) bool - alpha at 255
    source: str

    @property
    def columns(self) -> int:
        return int(self.rgb.shape[1])

    @property
    def rows(self) -> int:
        return int(self.rgb.shape[0])


def load_export(path: str | Path) -> SignExport:
    image = Image.open(path).convert("RGBA")
    data = np.asarray(image, dtype=np.float32)
    return SignExport(rgb=data[:, :, :3].copy(), painted=data[:, :, 3] >= 250, source=str(path))


class ExportWatcher:
    """Take the one export file the painter's click produces, then remove it.

    ``snapshot`` before the click records what is already on the desktop;
    ``collect`` afterwards waits for a new file, reads it, deletes it, and
    returns the export.  Only files created after the snapshot are ever
    read or removed.
    """

    def __init__(self, directory: str | Path | None = None) -> None:
        self.directory = Path(directory) if directory is not None else default_export_directory()
        self._before: set[str] = set()
        self._started = 0.0

    def snapshot(self) -> None:
        self._before = set(glob.glob(str(self.directory / EXPORT_PATTERN)))
        self._started = time.time()

    def _new_files(self) -> list[str]:
        return [
            path
            for path in glob.glob(str(self.directory / EXPORT_PATTERN))
            if path not in self._before and os.path.getmtime(path) >= self._started - 2.0
        ]

    def collect(
        self,
        *,
        wait_seconds: float = EXPORT_WAIT_SECONDS,
        sleep: Callable[[float], None] = time.sleep,
        keep_copy_in: str | Path | None = None,
    ) -> SignExport | None:
        deadline = time.monotonic() + wait_seconds
        path: str | None = None
        while time.monotonic() < deadline:
            new = self._new_files()
            if new:
                path = max(new, key=os.path.getmtime)
                break
            sleep(0.2)
        if path is None:
            LOGGER.warning(
                "No export appeared in %s within %.0fs of clicking the download button",
                self.directory,
                wait_seconds,
            )
            return None
        # Let the game finish writing: wait until the size stops changing.
        last = -1
        for _ in range(20):
            size = os.path.getsize(path)
            if size == last and size > 0:
                break
            last = size
            sleep(EXPORT_SETTLE_SECONDS / 4)
        try:
            export = load_export(path)
        except Exception:
            LOGGER.exception("The export %s could not be read", path)
            return None
        if keep_copy_in is not None:
            try:
                Path(keep_copy_in).mkdir(parents=True, exist_ok=True)
                target = Path(keep_copy_in) / os.path.basename(path)
                os.replace(path, target)
                export = SignExport(export.rgb, export.painted, str(target))
                return export
            except Exception:
                LOGGER.exception("The export could not be moved into %s", keep_copy_in)
        try:
            os.remove(path)
        except OSError:
            LOGGER.warning("The export %s could not be removed from the desktop", path)
        return export
