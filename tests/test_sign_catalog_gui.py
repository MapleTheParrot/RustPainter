from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PySide6.QtCore import Qt

from app.gui.main_window import MAX_QUALITY_PRESET, MainWindow, _SignCatalogDialog
from app.models import ScreenRect
from app.sign_catalog import catalog_entry


@pytest.fixture
def window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qtbot):
    monkeypatch.setenv("RUST_PAINTER_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.setenv("RUST_PAINTER_DISABLE_HOTKEYS", "1")
    widget = MainWindow()
    qtbot.addWidget(widget)
    yield widget
    widget.close()


def test_premade_profile_uses_the_game_asset_resolution_before_measurement(
    window: MainWindow,
) -> None:
    entry = catalog_entry("lightup-xxl")
    assert entry is not None
    assert window._current_profile is not None
    window._current_profile.canvas = ScreenRect(10, 10, 1000, 500)

    created = window._create_catalog_profile(entry)

    assert created is not None
    assert window._current_profile is not None
    assert window._current_profile.id == created.id
    assert window._current_profile.metadata["sign_texture_size"] == [1024, 512]
    assert window._sign_resolution_cap() == (1024, 512)
    assert window._sign_resolution_cap_source() == "catalog"
    window.quality_combo.setCurrentText(MAX_QUALITY_PRESET)
    assert (window.logical_width_spin.value(), window.logical_height_spin.value()) == (
        1024,
        512,
    )
    assert not window.profile_combo.itemIcon(window.profile_combo.currentIndex()).isNull()


def test_premade_profile_picker_fuzzy_filters_with_icons(window: MainWindow) -> None:
    dialog = _SignCatalogDialog(window)
    dialog.search_edit.setText("artst canv xxl")

    assert dialog.list.count() > 0
    first = dialog.list.item(0)
    assert first.data(Qt.ItemDataRole.UserRole) == "artist-xxl"
    assert "1024×512" in first.text()
    assert not first.icon().isNull()
    assert dialog.list.currentItem() is first
    assert dialog.list.visualItemRect(first).top() >= 0
    assert dialog.list.verticalScrollBar().value() == dialog.list.verticalScrollBar().minimum()
    assert "border: 2px solid" in dialog.list.styleSheet()
    dialog.close()


def test_fresh_install_starts_with_the_catalog_large_wooden_sign(
    window: MainWindow,
) -> None:
    assert window.profile_combo.count() == 1
    assert window._current_profile is not None
    assert window._current_profile.metadata["sign_catalog_id"] == "wood-large"
    assert window._current_profile.metadata["sign_texture_size"] == [512, 256]


def test_premade_profile_supplies_its_aspect_before_canvas_calibration(
    window: MainWindow,
) -> None:
    entry = catalog_entry("artist-small")
    assert entry is not None

    created = window._create_catalog_profile(entry)

    assert created is not None and created.canvas is None
    window.quality_combo.setCurrentText("Fast")
    assert (window.logical_width_spin.value(), window.logical_height_spin.value()) == (
        96,
        128,
    )
