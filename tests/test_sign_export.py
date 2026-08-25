from __future__ import annotations

import os
import time
from pathlib import Path

import numpy as np
from PIL import Image

from app.sign_export import ExportWatcher, load_export


def _write_export(directory: Path, stamp: int, painted: bool = True) -> Path:
    image = Image.new("RGBA", (8, 4), (0, 0, 0, 0))
    if painted:
        image.putpixel((2, 1), (30, 60, 220, 255))
    path = directory / f"artists_canvas_XXL_500x250_{stamp}.png"
    image.save(path)
    return path


def test_load_export_reads_texels_and_whether_they_were_painted(tmp_path: Path) -> None:
    path = _write_export(tmp_path, 1)
    export = load_export(path)
    assert (export.columns, export.rows) == (8, 4)
    assert export.painted.sum() == 1 and export.painted[1, 2]
    assert tuple(export.rgb[1, 2]) == (30.0, 60.0, 220.0)


def test_the_watcher_takes_only_the_file_that_appeared_after_its_click(tmp_path: Path) -> None:
    """The user's own downloads are never read or removed; the painter's is."""

    old = _write_export(tmp_path, 100)
    os.utime(old, (time.time() - 3600, time.time() - 3600))
    watcher = ExportWatcher(tmp_path)
    watcher.snapshot()
    produced = _write_export(tmp_path, 200)
    export = watcher.collect(wait_seconds=2.0, sleep=lambda s: None)
    assert export is not None and export.painted[1, 2]
    assert not produced.exists(), "the painter's own export is cleaned off the desktop"
    assert old.exists(), "a file that was there before the click is left alone"


def test_the_watcher_gives_up_quietly_when_nothing_appears(tmp_path: Path) -> None:
    watcher = ExportWatcher(tmp_path)
    watcher.snapshot()
    assert watcher.collect(wait_seconds=0.3, sleep=lambda s: time.sleep(min(s, 0.05))) is None


def test_the_watcher_can_keep_the_export_in_a_run_folder(tmp_path: Path) -> None:
    watcher = ExportWatcher(tmp_path)
    watcher.snapshot()
    _write_export(tmp_path, 300)
    export = watcher.collect(wait_seconds=2.0, sleep=lambda s: None, keep_copy_in=tmp_path / "run")
    assert export is not None
    assert Path(export.source).parent == tmp_path / "run"
    assert not list(tmp_path.glob("artists_canvas_*.png"))
