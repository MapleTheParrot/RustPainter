from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from app.calibration import (
    CalibrationPreviewOverlay,
    _resize_edges,
    _resized_rect,
)
from app.gui.main_window import MainWindow
from app.profiles import Rect


def test_edge_and_corner_resizing_keeps_the_opposite_sides_fixed() -> None:
    original = Rect(100, 200, 80, 40)

    left_edge = _resize_edges(original, 102, 220, 7)
    top_right_corner = _resize_edges(original, 178, 203, 7)

    assert left_edge == (True, False, False, False)
    assert _resized_rect(original, left_edge, 90, 999, 3) == Rect(90, 200, 90, 40)
    assert top_right_corner == (False, True, True, False)
    assert _resized_rect(original, top_right_corner, 195, 185, 3) == Rect(
        100, 185, 95, 55
    )


def test_resizing_cannot_collapse_a_calibrated_box() -> None:
    original = Rect(10, 20, 30, 40)

    resized = _resized_rect(original, (True, False, True, False), 100, 100, 3)

    assert resized == Rect(37, 57, 3, 3)


def test_preview_reports_the_finished_resize_without_requesting_a_redrag() -> None:
    updates: list[tuple[str, Rect]] = []
    overlay = CalibrationPreviewOverlay()
    overlay.set_rectangles([("Canvas", Rect(10, 20, 100, 50))])
    overlay.set_resize_callback(lambda label, rect: updates.append((label, rect)))

    overlay._finish_resize("Canvas", Rect(8, 20, 102, 55))

    assert overlay._entries == [("Canvas", Rect(8, 20, 102, 55))]
    assert updates == [("Canvas", Rect(8, 20, 102, 55))]


def test_finished_overlay_resize_is_saved_to_the_current_profile(
    tmp_path: Path, monkeypatch, qtbot
) -> None:
    monkeypatch.setenv("RUST_PAINTER_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.setenv("RUST_PAINTER_DISABLE_HOTKEYS", "1")
    window = MainWindow()
    qtbot.addWidget(window)
    window.show_calibration_check.setChecked(False)
    profile = window._current_profile
    assert profile is not None
    profile.color_box = Rect(100, 100, 80, 80)
    profile.metadata["ui_reference"] = {"path": "old.png"}
    profile.metadata["color_correction"] = {"old": True}
    window._current_profile = window._profile_store.save(profile)

    replacement = Rect(95, 100, 85, 90)
    window._resize_calibration_rectangle("Color box", replacement)

    stored = window._profile_store.require(profile.id)
    assert stored.color_box == replacement
    assert "ui_reference" not in stored.metadata
    assert "color_correction" not in stored.metadata
    window.close()
