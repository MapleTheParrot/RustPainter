from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, QMimeData, QPoint, QPointF, Qt, QTimer, QUrl
from PySide6.QtGui import (
    QColor,
    QDragEnterEvent,
    QDropEvent,
    QKeyEvent,
    QKeySequence,
    QMouseEvent,
    QShortcut,
)
from PySide6.QtWidgets import QColorDialog, QGraphicsSceneMouseEvent

import app.gui.main_window as main_window_module
from app.gui.main_window import (
    MAX_QUALITY_PRESET,
    MainWindow,
    PLAN_SETTLE_MS,
    _PendingPaint,
    _TextOverlayOptions,
    _build_plan_prefix_image,
)
from app.color_calibration import ColorCorrectionModel
from app.gui.widgets import ColorButton, CountdownDialog
from app.gui.tutorial import TUTORIAL_STEPS, TUTORIAL_VERSION
from app.setup_detection import DetectedRegion, SetupDetection
from app.input_controller import MockInputController
from app.models import (
    ColorGroup,
    PaintPlan,
    ProcessedImage,
    ScreenRect,
    ScaleMode,
    Stroke,
    TransparencyMode,
)
from app.painter import PaintProgress, Painter, PainterState
from app.brush_calibration import fit_brush_size_model
from app.profiles import DisplayMetadata, Profile, Rect
from app.screen import VirtualScreen


@pytest.fixture
def window(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qtbot):
    monkeypatch.setenv("RUST_PAINTER_DATA_DIR", str(tmp_path / "app-data"))
    monkeypatch.setenv("RUST_PAINTER_DISABLE_HOTKEYS", "1")
    widget = MainWindow()
    qtbot.addWidget(widget)
    yield widget
    widget.close()


def test_image_to_preview_and_plan(window: MainWindow, tmp_path: Path, qtbot) -> None:
    source_path = tmp_path / "source.png"
    source = Image.new("RGBA", (40, 20), (210, 30, 40, 255))
    for x in range(20, 40):
        for y in range(20):
            source.putpixel((x, y), (20, 100, 220, 255))
    source.save(source_path)

    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)

    assert window._plan is not None
    assert (window._plan.width, window._plan.height) == (512, 256)
    assert len(window._plan.color_groups) <= 256
    assert window._plan.stroke_count > 0
    assert not window.paint_preview._source.isNull()
    assert not window.dry_run_check.isChecked()
    assert not window.start_button.isEnabled()


def test_start_calculates_a_deferred_plan_without_preview_click(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    window.dry_run_check.setChecked(True)
    window._original_image = Image.new("RGB", (8, 4), (180, 80, 30))
    window._plan = None
    window._plan_deferred = True
    started: list[bool] = []
    monkeypatch.setattr(window, "_start_processing", lambda: started.append(True))

    window._update_start_availability()
    assert window.start_button.isEnabled()
    window._start_or_resume()

    assert window._start_after_processing is True
    assert started == [True]
    assert not window.start_button.isEnabled()

    window._abort_painting()
    assert window._start_after_processing is False


def test_history_is_an_icon_button_beside_start(window: MainWindow) -> None:
    assert window.sessions_button.text() == ""
    assert not window.sessions_button.icon().isNull()
    assert window.sessions_button.accessibleName() == "Painting history"
    assert window.sessions_button.height() == window.start_button.minimumHeight()


def test_optimization_mode_merges_colors_and_gates_controls(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    # Quality is the default, and it supersedes stroke merging.
    assert window.paint_mode_combo.currentData() == "quality"
    assert not window.merge_combo.isEnabled()
    window._set_combo_data(window.paint_mode_combo, "exact")
    assert window.merge_combo.isEnabled()

    source_path = tmp_path / "flat.png"
    image = Image.new("RGB", (64, 32), (250, 250, 250))
    for x in range(10):
        image.putpixel((x, 0), (252, 250, 250))
    image.save(source_path)
    window._set_combo_data(window.paint_mode_combo, "fast")
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)

    # The near-identical whites collapse into a single paint pass, and with
    # no brush calibration the plan never promises a larger brush.
    assert len(window._plan.color_groups) == 1
    assert all(group.brush_diameter == 1 for group in window._plan.color_groups)
    assert "optimization" in window.processing_label.text()


def test_text_overlay_is_editable_and_included_in_paint_plan(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    source_path = tmp_path / "text-background.png"
    Image.new("RGB", (128, 64), (0, 0, 0)).save(source_path)
    window.quality_combo.setCurrentText("Custom")
    window.logical_width_spin.setValue(128)
    assert window.text_edit.isEnabled()
    assert not window.text_font_combo.isEditable()
    assert window.text_smooth_check.isChecked()
    assert window._text_layers[0].smooth is True
    window.text_edit.setText("RUST")
    window.text_size_spin.setValue(30)
    window.text_color_button.set_color("#FFFFFF", emit=True)
    window.text_bold_check.setChecked(True)
    window.add_text_button.click()
    assert all(layer.smooth for layer in window._text_layers)
    window.text_edit.setText("BIRD")
    window.text_color_button.set_color("#55FF55", emit=True)

    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._processed is not None, timeout=5000)

    assert window.text_options_panel.isEnabled()
    assert window.text_font_combo.currentFont().family()
    assert len(window._text_layers) == 2
    assert len(window.original_preview._items) == 2
    assert window.original_preview._items[1].defaultTextColor().name() == "#55ff55"
    assert window._processed is not None
    old_y = window._text_layers[1].y
    item = window.original_preview._items[1]
    item.setPos(item.pos().x(), item.pos().y() + 3)
    assert window._text_layers[1].y != old_y
    qtbot.waitUntil(lambda: window._processed is not None, timeout=5000)
    item = window.original_preview._items[1]
    double_click = QGraphicsSceneMouseEvent(
        QEvent.Type.GraphicsSceneMouseDoubleClick
    )
    double_click.setButton(Qt.MouseButton.LeftButton)
    double_click.setPos(item.boundingRect().center())
    item.mouseDoubleClickEvent(double_click)
    assert item._editing
    assert item.textInteractionFlags() == Qt.TextInteractionFlag.TextEditorInteraction
    item.setPlainText("EDITED HERE")
    assert window._text_layers[1].text == "EDITED HERE"
    item._resize_center = item.mapToScene(item.boundingRect().center())
    item._apply_font_size(42)
    assert window._text_layers[1].font_size == 42
    assert window.text_size_spin.value() == 42
    qtbot.waitUntil(lambda: window._processed is not None, timeout=5000)
    pixels = np.asarray(window._processed.image.convert("RGB"))
    assert np.any(pixels > 128)
    assert len(window._plan.color_groups) >= 2

    # The Rust preview bakes the text in rather than floating it as live
    # vector items, so the black source must show bright text pixels there.
    simulation = window.paint_preview._source.toImage()
    assert any(
        QColor(simulation.pixel(x, y)).value() > 128
        for x in range(0, simulation.width(), 2)
        for y in range(0, simulation.height(), 2)
    )


def test_text_size_survives_a_change_of_quality_preset(window: MainWindow) -> None:
    """Text keeps its proportions when the painting resolution changes.

    Sizes are stored in logical canvas pixels, so a 40px caption placed under
    "High" used to cover most of a "Very Fast" canvas (and shrink to nothing
    going the other way). The layer now stores the size as a fraction of the
    canvas height, and the pixel size is re-derived per resolution.
    """

    window.quality_combo.setCurrentText("High")
    window.text_edit.setText("RUST")
    window.text_size_spin.setValue(40)
    high_height = window.logical_height_spin.value()
    high_ratio = window._text_layers[0].size_ratio
    assert high_ratio == pytest.approx(40 / high_height)

    window.quality_combo.setCurrentText("Very Fast")
    fast_layer = window._text_layers[0]
    assert fast_layer.size_ratio == pytest.approx(high_ratio)
    assert fast_layer.font_size == pytest.approx(
        round(high_ratio * window.logical_height_spin.value())
    )
    assert fast_layer.font_size < 40
    assert window.text_size_spin.value() == fast_layer.font_size

    window.quality_combo.setCurrentText("High")
    assert window._text_layers[0].font_size == 40
    assert window.text_size_spin.value() == 40


def test_text_editor_maps_the_sign_canvas_onto_the_source_image(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """Layer fractions stay canvas fractions while editing over the source.

    A square image on the default 2:1 canvas keeps only its middle band under
    Fill, so the editor's canvas rectangle must cover exactly that band of the
    displayed pixmap; under Fit the same canvas letterboxes and the rectangle
    must extend past the pixmap's sides instead.
    """

    source_path = tmp_path / "square.png"
    Image.new("RGB", (30, 30), (80, 80, 80)).save(source_path)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._source_preview_size is not None, timeout=5000)
    preview_width, preview_height = window._source_preview_size

    window._set_combo_data(window.scale_mode_combo, ScaleMode.FILL.value)
    geometry = window._source_canvas_geometry()
    assert geometry is not None
    rect, font_scale = geometry
    assert rect.width() == pytest.approx(preview_width)
    assert rect.height() == pytest.approx(preview_height / 2)
    assert rect.top() == pytest.approx(preview_height / 4)
    assert font_scale == pytest.approx(
        rect.height() / window.logical_height_spin.value()
    )

    window._set_combo_data(window.scale_mode_combo, ScaleMode.FIT.value)
    geometry = window._source_canvas_geometry()
    assert geometry is not None
    rect, _ = geometry
    assert rect.left() < 0
    assert rect.right() > preview_width
    assert rect.top() == pytest.approx(0.0, abs=1e-6)


def test_stretch_edits_text_over_the_stretched_result(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """Stretch pre-distorts the backdrop so placement matches the sign.

    A square photo pulled onto a 2:1 canvas keeps none of its own proportions,
    and text dropped over the undistorted original landed nowhere near the
    spot it was aimed at.  The Source tab therefore shows the source already
    stretched, which also makes one uniform text scale the true one.
    """

    source_path = tmp_path / "square.png"
    Image.new("RGB", (30, 30), (80, 80, 80)).save(source_path)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._source_preview_size is not None, timeout=5000)
    decoded = window.original_preview._source.size()

    window._set_combo_data(window.scale_mode_combo, ScaleMode.STRETCH.value)
    logical_width = window.logical_width_spin.value()
    logical_height = window.logical_height_spin.value()
    backdrop = window.original_preview._source.size()
    assert backdrop != decoded
    assert backdrop.width() / backdrop.height() == pytest.approx(
        logical_width / logical_height, rel=0.05
    )

    geometry = window._source_canvas_geometry()
    assert geometry is not None
    rect, font_scale = geometry
    # The whole backdrop is the canvas, and one logical pixel is the same
    # length on both axes - which is what makes a placed caption land right.
    assert rect.width() == pytest.approx(backdrop.width())
    assert rect.height() == pytest.approx(backdrop.height())
    assert font_scale == pytest.approx(rect.width() / logical_width, rel=0.05)
    assert font_scale == pytest.approx(rect.height() / logical_height)

    # Fit never distorts, so it gets the source back at its own shape.
    window._set_combo_data(window.scale_mode_combo, ScaleMode.FIT.value)
    assert window.original_preview._source.size() == decoded


def test_control_z_walks_the_history_while_a_layer_is_being_edited(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """Typing into a layer is exactly when an undo is wanted."""

    _two_layer_window(window, tmp_path, qtbot)
    preview = window.original_preview
    item = preview._items[1]
    double_click = QGraphicsSceneMouseEvent(QEvent.Type.GraphicsSceneMouseDoubleClick)
    double_click.setButton(Qt.MouseButton.LeftButton)
    double_click.setPos(item.boundingRect().center())
    item.mouseDoubleClickEvent(double_click)
    assert preview.is_editing_text

    # A step of another kind so the typing below cannot fold into the one
    # that named this layer in the first place.
    window.text_align_buttons["top"].click()
    item.setPlainText("TYPO")
    assert window._text_layers[1].text == "TYPO"

    preview.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier
        )
    )
    assert window._text_layers[1].text == "SECOND"

    preview.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Y, Qt.KeyboardModifier.ControlModifier
        )
    )
    assert window._text_layers[1].text == "TYPO"


def test_the_window_answers_undo_keys_from_the_side_panel(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """The same keys walk the same history wherever the focus sits.

    Text is as often typed into the side panel as onto the canvas, so the undo
    keys are bound to the window rather than to the one widget that used to
    answer them.
    """

    _two_layer_window(window, tmp_path, qtbot)
    window.text_align_buttons["top"].click()
    window.text_edit.setFocus()
    window.text_edit.setText("RENAMED")
    assert window._text_layers[1].text == "RENAMED"

    shortcuts = {
        shortcut.key().toString(): shortcut
        for shortcut in window.findChildren(QShortcut)
    }
    assert {"Ctrl+Z", "Ctrl+Y", "Ctrl+Shift+Z"} <= set(shortcuts)
    assert all(
        shortcuts[keys].context() == Qt.ShortcutContext.WindowShortcut
        for keys in ("Ctrl+Z", "Ctrl+Y", "Ctrl+Shift+Z")
    )

    shortcuts["Ctrl+Z"].activated.emit()
    assert window._text_layers[1].text == "SECOND"
    assert window.text_edit.text() == "SECOND"

    shortcuts["Ctrl+Y"].activated.emit()
    assert window._text_layers[1].text == "RENAMED"
    shortcuts["Ctrl+Z"].activated.emit()
    shortcuts["Ctrl+Shift+Z"].activated.emit()
    assert window._text_layers[1].text == "RENAMED"


def test_shift_click_adds_a_layer_to_the_selection_and_takes_it_back_out(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """Shift is the multi-select modifier every other editor uses."""

    _two_layer_window(window, tmp_path, qtbot)
    preview = window.original_preview
    preview.select_layers([0], 0)

    second = preview._items[1]
    _click_item(second, Qt.KeyboardModifier.ShiftModifier)
    assert preview.selected_indices() == [0, 1]
    assert window._edit_target_indices() == [0, 1]

    _click_item(second, Qt.KeyboardModifier.ShiftModifier)
    assert preview.selected_indices() == [0]

    # A plain click still means "just this one".
    _click_item(second)
    assert preview.selected_indices() == [1]


def test_delete_key_removes_the_selected_text_layer(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    source_path = tmp_path / "two-layers.png"
    Image.new("RGB", (64, 32), (10, 10, 10)).save(source_path)
    window.text_edit.setText("FIRST")
    window.add_text_button.click()
    window.text_edit.setText("SECOND")
    window.load_image(source_path)
    qtbot.waitUntil(lambda: len(window.original_preview._items) == 2, timeout=5000)

    window.original_preview.select_layer(1)
    assert window.original_preview.selected_index() == 1
    window.original_preview.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
    )

    assert [layer.text for layer in window._text_layers] == ["FIRST"]

    # The last layer is emptied rather than removed, so there is always one to
    # type into, and Backspace does the same job as Delete.
    window.original_preview.select_layer(0)
    window.original_preview.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Backspace, Qt.KeyboardModifier.NoModifier
        )
    )
    assert [layer.text for layer in window._text_layers] == [""]


@pytest.mark.parametrize("key", [Qt.Key.Key_D, Qt.Key.Key_C])
def test_control_d_and_control_c_copy_the_selected_text_layer(
    window: MainWindow, tmp_path: Path, qtbot, key
) -> None:
    source_path = tmp_path / "one-layer.png"
    Image.new("RGB", (64, 32), (10, 10, 10)).save(source_path)
    window.text_edit.setText("FIRST")
    window.load_image(source_path)
    qtbot.waitUntil(lambda: len(window.original_preview._items) == 1, timeout=5000)

    window.original_preview.select_layer(0)
    window.original_preview.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, key, Qt.KeyboardModifier.ControlModifier)
    )

    assert [layer.text for layer in window._text_layers] == ["FIRST", "FIRST"]
    # The copy is selected and offset, so dragging it does not move the original.
    assert window._selected_text_layer == 1
    original, copy = window._text_layers
    assert (copy.x, copy.y) != (original.x, original.y)
    assert copy.font_size == original.font_size


def test_control_c_copies_characters_while_a_layer_is_being_edited(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    source_path = tmp_path / "editing-copy.png"
    Image.new("RGB", (64, 32), (10, 10, 10)).save(source_path)
    window.text_edit.setText("FIRST")
    window.load_image(source_path)
    qtbot.waitUntil(lambda: len(window.original_preview._items) == 1, timeout=5000)

    item = window.original_preview._items[0]
    double_click = QGraphicsSceneMouseEvent(QEvent.Type.GraphicsSceneMouseDoubleClick)
    double_click.setButton(Qt.MouseButton.LeftButton)
    double_click.setPos(item.boundingRect().center())
    item.mouseDoubleClickEvent(double_click)
    assert window.original_preview.is_editing_text

    window.original_preview.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_C, Qt.KeyboardModifier.ControlModifier
        )
    )
    assert len(window._text_layers) == 1


def _two_layer_window(window: MainWindow, tmp_path: Path, qtbot) -> None:
    """Open a plain backdrop carrying two captions, ready to be edited."""

    source_path = tmp_path / "two-captions.png"
    Image.new("RGB", (128, 64), (10, 10, 10)).save(source_path)
    window.text_edit.setText("FIRST")
    window.add_text_button.click()
    window.text_edit.setText("SECOND")
    window.load_image(source_path)
    qtbot.waitUntil(lambda: len(window.original_preview._items) == 2, timeout=5000)


def _drag_item(item, start: QPointF, end: QPointF | None, modifiers=None) -> None:
    """Press, move and release one text item, in scene coordinates.

    ``end`` of ``None`` is a click: the item is pressed and let go without
    ever moving, which is a case Qt settles the selection differently for.
    """

    modifiers = modifiers or Qt.KeyboardModifier.NoModifier
    steps = (
        ((QEvent.Type.GraphicsSceneMousePress, start),)
        if end is None
        else (
            (QEvent.Type.GraphicsSceneMousePress, start),
            (QEvent.Type.GraphicsSceneMouseMove, end),
        )
    ) + ((QEvent.Type.GraphicsSceneMouseRelease, end or start),)
    for event_type, position in steps:
        event = QGraphicsSceneMouseEvent(event_type)
        event.setButton(Qt.MouseButton.LeftButton)
        event.setButtons(Qt.MouseButton.LeftButton)
        event.setScenePos(position)
        event.setPos(item.mapFromScene(position))
        event.setModifiers(modifiers)
        if event_type == QEvent.Type.GraphicsSceneMousePress:
            item.mousePressEvent(event)
        elif event_type == QEvent.Type.GraphicsSceneMouseMove:
            item.mouseMoveEvent(event)
        else:
            item.mouseReleaseEvent(event)


def _click_item(item, modifiers=None) -> None:
    """Press and release one text item without moving it."""

    _drag_item(item, item.mapToScene(item.text_rect().center()), None, modifiers)


def test_selecting_several_layers_edits_them_together(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """Styling reaches every selected layer; the typed text reaches one.

    Two captions that should match are the whole point of a multiple
    selection, but they are still two captions - giving them a shared font
    must not give them a shared string.
    """

    _two_layer_window(window, tmp_path, qtbot)
    window.original_preview.select_layers([0, 1], 1)
    assert window._edit_target_indices() == [0, 1]
    assert "2 layers selected" in window.text_selection_label.text()

    window.text_size_spin.setValue(18)
    window.text_color_button.set_color("#22CCFF", emit=True)
    window.text_bold_check.setChecked(True)

    assert [layer.font_size for layer in window._text_layers] == [18, 18]
    assert [layer.color for layer in window._text_layers] == [(34, 204, 255)] * 2
    assert all(layer.bold for layer in window._text_layers)
    assert [layer.text for layer in window._text_layers] == ["FIRST", "SECOND"]

    window.text_edit.setText("ONLY THIS ONE")
    assert [layer.text for layer in window._text_layers] == ["FIRST", "ONLY THIS ONE"]


def test_dragging_one_of_several_selected_layers_moves_them_all(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    _two_layer_window(window, tmp_path, qtbot)
    window.original_preview.select_layers([0, 1], 1)
    before = [(layer.x, layer.y) for layer in window._text_layers]

    item = window.original_preview._items[1]
    start = item.mapToScene(item.text_rect().center())
    _drag_item(item, start, start + QPointF(0.0, 12.0))

    after = [(layer.x, layer.y) for layer in window._text_layers]
    assert after != before
    # One step for the group, so the layers keep the spacing they had.
    assert after[0][1] - before[0][1] == pytest.approx(after[1][1] - before[1][1])
    assert after[0][0] == pytest.approx(before[0][0])


def test_dragging_text_snaps_onto_the_middle_of_the_sign(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """A near miss lands on the middle, and Alt hands the pixels back."""

    _two_layer_window(window, tmp_path, qtbot)
    preview = window.original_preview
    canvas = preview.canvas_rect()
    near_middle = QPointF(canvas.center().x() + 1.5, canvas.center().y() + 1.5)

    preview.select_layers([0], 0)
    item = preview._items[0]
    _drag_item(item, item.mapToScene(item.text_rect().center()), near_middle)
    assert (window._text_layers[0].x, window._text_layers[0].y) == (0.5, 0.5)

    preview.select_layers([0], 0)
    item = preview._items[0]
    _drag_item(
        item,
        item.mapToScene(item.text_rect().center()),
        near_middle,
        Qt.KeyboardModifier.AltModifier,
    )
    assert window._text_layers[0].x != 0.5
    assert window._text_layers[0].y != 0.5


def test_align_and_spread_place_layers_without_dragging(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    _two_layer_window(window, tmp_path, qtbot)
    window.original_preview.select_layers([0], 0)

    window.text_align_buttons["center"].click()
    window.text_align_buttons["top"].click()
    layer = window._text_layers[0]
    assert layer.x == pytest.approx(0.5)
    # Parked against the edge means touching it, not straddling it.
    assert layer.y == pytest.approx(window._text_layer_extent(layer)[1] / 2)

    window.text_align_buttons["right"].click()
    assert window._text_layers[0].x == pytest.approx(
        1.0 - window._text_layer_extent(window._text_layers[0])[0] / 2
    )

    # Spreading needs a layer in the middle to move, so two is not enough.
    assert not window.text_spread_buttons["down"].isEnabled()
    window.add_text_button.click()
    window.text_edit.setText("THIRD")
    for index, y in ((0, 0.1), (1, 0.9), (2, 0.7)):
        window._text_layers[index] = replace(window._text_layers[index], y=y)
    window.original_preview.select_layers([0, 1, 2], 0)
    assert window.text_spread_buttons["down"].isEnabled()

    window.text_spread_buttons["down"].click()
    assert window._text_layers[2].y == pytest.approx(0.5)


def test_arrow_keys_nudge_every_selected_layer(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    _two_layer_window(window, tmp_path, qtbot)
    window.original_preview.select_layers([0, 1], 0)
    before = [layer.x for layer in window._text_layers]

    for _ in range(2):
        window.original_preview.keyPressEvent(
            QKeyEvent(
                QEvent.Type.KeyPress,
                Qt.Key.Key_Right,
                Qt.KeyboardModifier.NoModifier,
            )
        )

    # Two logical pixels of a 128-wide canvas, for both layers.
    step = 2.0 / window.logical_width_spin.value()
    assert [layer.x for layer in window._text_layers] == [
        pytest.approx(value + step) for value in before
    ]

    window.original_preview.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Up, Qt.KeyboardModifier.ShiftModifier
        )
    )
    assert window._text_layers[0].y == pytest.approx(
        0.5 - 10.0 / window.logical_height_spin.value()
    )


def test_control_a_takes_every_layer_and_escape_lets_them_go(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    _two_layer_window(window, tmp_path, qtbot)
    window.original_preview.select_layers([], 0)

    window.original_preview.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier
        )
    )
    assert window.original_preview.selected_indices() == [0, 1]
    assert window._edit_target_indices() == [0, 1]

    window.original_preview.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Escape, Qt.KeyboardModifier.NoModifier
        )
    )
    assert window.original_preview.selected_indices() == []
    # With nothing selected the panel still has the named layer to edit.
    assert window._edit_target_indices() == [window._selected_text_layer]


def test_delete_and_copy_apply_to_the_whole_selection(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    _two_layer_window(window, tmp_path, qtbot)
    window.original_preview.select_layers([0, 1], 0)

    window.original_preview.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_D, Qt.KeyboardModifier.ControlModifier
        )
    )
    assert [layer.text for layer in window._text_layers] == [
        "FIRST",
        "FIRST",
        "SECOND",
        "SECOND",
    ]

    qtbot.waitUntil(lambda: len(window.original_preview._items) == 4, timeout=5000)
    window.original_preview.select_layers([1, 3], 1)
    window.original_preview.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier
        )
    )
    assert [layer.text for layer in window._text_layers] == ["FIRST", "SECOND"]


def test_a_gradient_and_an_outline_survive_into_the_painted_image() -> None:
    """Both decorations are baked, not just drawn on the editing canvas."""

    backdrop = ProcessedImage(
        Image.new("RGB", (160, 80), (0, 0, 0)),
        np.zeros((80, 160), dtype=bool),
        64,
    )
    gradient = _TextOverlayOptions(
        "RUST",
        "",
        44,
        (255, 0, 0),
        gradient=True,
        gradient_color=(0, 0, 255),
        gradient_direction="vertical",
    )
    baked = main_window_module._apply_text_overlays(backdrop, (gradient,))
    pixels = np.asarray(baked.image.convert("RGB"), dtype=np.int16)
    mask = np.asarray(baked.paint_mask, dtype=bool)
    rows = np.flatnonzero(mask.any(axis=1))
    top = mask.copy()
    top[rows[len(rows) // 2] :] = False
    bottom = mask & ~top
    # Red at the start of the ramp, blue at its end.
    assert pixels[top][:, 0].mean() > pixels[bottom][:, 0].mean()
    assert pixels[bottom][:, 2].mean() > pixels[top][:, 2].mean()

    outlined = _TextOverlayOptions(
        "RUST", "", 44, (255, 255, 255), outline_width=3, outline_color=(0, 255, 0)
    )
    baked = main_window_module._apply_text_overlays(backdrop, (outlined,))
    painted = np.asarray(baked.image.convert("RGB"), dtype=np.int16)[
        np.asarray(baked.paint_mask, dtype=bool)
    ]
    green = painted[:, 1] > 150
    assert np.any(green & (painted[:, 0] < 100) & (painted[:, 2] < 100))
    # An outline reaches past the letters, so it paints more of the sign.
    assert baked.paint_mask.sum() > _apply_plain_text_mask(backdrop)


def _apply_plain_text_mask(backdrop: ProcessedImage) -> int:
    plain = _TextOverlayOptions("RUST", "", 44, (255, 255, 255))
    return int(main_window_module._apply_text_overlays(backdrop, (plain,)).paint_mask.sum())


def test_baked_text_has_no_antialiased_fringe() -> None:
    """Every lettered texel carries the text color, never an edge blend.

    Blends would survive quantization as fattened strokes and filled-in
    counters, so coverage is thresholded before the palette pass.
    """

    backdrop = ProcessedImage(
        Image.new("RGB", (160, 80), (0, 0, 0)),
        np.zeros((80, 160), dtype=bool),
        64,
    )
    plain = _TextOverlayOptions("RUST", "", 44, (255, 255, 255))
    baked = main_window_module._apply_text_overlays(backdrop, (plain,))
    painted = np.asarray(baked.image.convert("RGB"), dtype=np.uint8)[
        np.asarray(baked.paint_mask, dtype=bool)
    ]
    assert painted.size > 0
    assert np.all(painted == 255)


def test_text_can_smooth_its_edges_without_smoothing_the_image() -> None:
    backdrop_array = np.zeros((80, 160, 3), dtype=np.uint8)
    backdrop_array[:, ::2] = (20, 40, 60)
    backdrop_array[:, 1::2] = (180, 200, 220)
    backdrop = ProcessedImage(
        Image.fromarray(backdrop_array, mode="RGB"),
        np.ones((80, 160), dtype=bool),
        256,
    )
    smooth = _TextOverlayOptions(
        "RUST", "", 44, (255, 255, 255), smooth=True
    )

    baked = main_window_module._apply_text_overlays(backdrop, (smooth,))
    result = np.asarray(baked.image.convert("RGB"), dtype=np.uint8)

    # The corners are outside the glyphs, so the alternating hard-edged image
    # pixels must survive byte for byte.
    assert np.array_equal(result[0], backdrop_array[0])
    # Antialiased glyph coverage creates edge colors between backdrop and fill.
    assert np.any((result > backdrop_array) & (result < 255))


def test_gradient_and_outline_round_trip_through_settings(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    _two_layer_window(window, tmp_path, qtbot)
    window.original_preview.select_layers([0, 1], 0)
    window.text_gradient_check.setChecked(True)
    window._set_combo_data(window.text_gradient_direction_combo, "diagonal")
    window.text_gradient_color_button.set_color("#FF0000", emit=True)
    window.text_outline_spin.setValue(2)
    window.text_outline_color_button.set_color("#00FF00", emit=True)
    window.text_smooth_check.setChecked(True)

    document = window._settings_document()
    saved = document["image"]["text_overlay"]["layers"]
    assert [layer["gradient"] for layer in saved] == [True, True]
    assert saved[0]["gradient_direction"] == "diagonal"
    assert saved[0]["gradient_color"] == "#FF0000"
    assert saved[0]["outline_width"] == 2
    assert saved[0]["outline_color"] == "#00FF00"
    assert saved[0]["smooth"] is True

    window._apply_settings(document)
    restored = window._text_layers[0]
    assert restored.gradient is True
    assert restored.gradient_direction == "diagonal"
    assert restored.gradient_color == (255, 0, 0)
    assert restored.outline_width == 2
    assert restored.outline_color == (0, 255, 0)
    assert restored.smooth is True
    # The gradient's own controls follow the checkbox rather than sitting live
    # next to a switched-off gradient.
    window.text_gradient_check.setChecked(False)
    assert not window.text_gradient_color_button.isEnabled()


def test_the_text_canvas_keeps_its_own_undo_history(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    _two_layer_window(window, tmp_path, qtbot)
    window.original_preview.select_layers([0, 1], 0)
    window.text_size_spin.setValue(30)
    window.text_align_buttons["center"].click()
    assert [layer.font_size for layer in window._text_layers] == [30, 30]
    assert [layer.x for layer in window._text_layers] == [0.5, 0.5]

    window.undo_text_button.click()
    assert [layer.x for layer in window._text_layers] != [0.5, 0.5]
    # One step back is one step, so the size change is still standing.
    assert [layer.font_size for layer in window._text_layers] == [30, 30]

    window.undo_text_button.click()
    assert [layer.font_size for layer in window._text_layers] == [24, 24]

    window.redo_text_button.click()
    window.redo_text_button.click()
    assert [layer.font_size for layer in window._text_layers] == [30, 30]
    assert [layer.x for layer in window._text_layers] == [0.5, 0.5]
    assert not window.redo_text_button.isEnabled()

    # Undoing back past the start is a message, not an exception.
    while window.undo_text_button.isEnabled():
        window.undo_text_button.click()
    window._undo_text_edit()
    assert [layer.text for layer in window._text_layers] == [""]


def test_a_new_text_edit_drops_whatever_was_waiting_to_be_redone(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    _two_layer_window(window, tmp_path, qtbot)
    window.text_align_buttons["top"].click()
    window.undo_text_button.click()
    assert window.redo_text_button.isEnabled()

    window.text_align_buttons["bottom"].click()
    assert not window.redo_text_button.isEnabled()
    assert window._text_layers[window._selected_text_layer].y > 0.5


def test_a_whole_drag_takes_a_single_step_back(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """Dragging reports every step it takes, and undoes as one move."""

    _two_layer_window(window, tmp_path, qtbot)
    preview = window.original_preview
    preview.select_layers([1], 1)
    before = (window._text_layers[1].x, window._text_layers[1].y)

    item = preview._items[1]
    start = item.mapToScene(item.text_rect().center())
    _drag_item(item, start, start + QPointF(-40.0, 60.0))
    assert (window._text_layers[1].x, window._text_layers[1].y) != before

    preview.keyPressEvent(
        QKeyEvent(
            QEvent.Type.KeyPress, Qt.Key.Key_Z, Qt.KeyboardModifier.ControlModifier
        )
    )
    assert window._text_layers[1].x == pytest.approx(before[0])
    assert window._text_layers[1].y == pytest.approx(before[1])


def test_undoing_a_text_edit_keeps_the_size_the_current_canvas_asks_for(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """A step taken under one quality preset comes back sized for another.

    Layers hold their size as a fraction of the canvas, so the history has to
    hold the fraction too rather than the pixels it happened to mean.
    """

    _two_layer_window(window, tmp_path, qtbot)
    window.quality_combo.setCurrentText("High")
    window.original_preview.select_layers([0], 0)
    window.text_size_spin.setValue(40)
    ratio = window._text_layers[0].size_ratio

    window.quality_combo.setCurrentText("Very Fast")
    assert window._text_layers[0].font_size < 40
    window.undo_text_button.click()

    restored = window._text_layers[0]
    assert restored.size_ratio != ratio
    assert restored.font_size == window._text_font_size(restored.size_ratio)


def test_background_removal_toggles_its_options_and_shrinks_the_plan(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    source_path = tmp_path / "logo-on-white.png"
    source = Image.new("RGB", (64, 64), (255, 255, 255))
    source.paste(Image.new("RGB", (16, 16), (200, 30, 40)), (24, 24))
    source.save(source_path)

    # The window itself is never shown offscreen, so ask whether the panel is
    # hidden in its own right rather than whether it is on screen.
    assert window.background_removal_panel.isHidden()
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    with_background = window._plan.painted_pixels

    window.remove_background_check.setChecked(True)
    qtbot.waitUntil(
        lambda: window._plan is not None
        and window._plan.painted_pixels < with_background,
        timeout=5000,
    )
    assert not window.background_removal_panel.isHidden()
    assert not window.removal_color_button.isEnabled()

    window._set_combo_data(window.removal_source_combo, "custom")
    assert window.removal_color_button.isEnabled()

    document = window._settings_document()
    assert document["image"]["remove_background"] is True
    assert document["image"]["background_removal_source"] == "custom"
    # The smart matcher is what removal reaches for unless told otherwise.
    assert document["image"]["background_removal_scope"] == "subject"

    window._set_combo_data(window.removal_scope_combo, "connected")
    assert (
        window._settings_document()["image"]["background_removal_scope"] == "connected"
    )


def test_delete_key_edits_text_while_a_layer_is_being_edited(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """Inside the inline editor both keys belong to the text cursor."""

    source_path = tmp_path / "editing.png"
    Image.new("RGB", (64, 32), (10, 10, 10)).save(source_path)
    window.text_edit.setText("FIRST")
    window.add_text_button.click()
    window.text_edit.setText("SECOND")
    window.load_image(source_path)
    qtbot.waitUntil(lambda: len(window.original_preview._items) == 2, timeout=5000)

    item = window.original_preview._items[1]
    double_click = QGraphicsSceneMouseEvent(QEvent.Type.GraphicsSceneMouseDoubleClick)
    double_click.setButton(Qt.MouseButton.LeftButton)
    double_click.setPos(item.boundingRect().center())
    item.mouseDoubleClickEvent(double_click)
    assert window.original_preview.is_editing_text

    window.original_preview.keyPressEvent(
        QKeyEvent(QEvent.Type.KeyPress, Qt.Key.Key_Delete, Qt.KeyboardModifier.NoModifier)
    )
    assert len(window._text_layers) == 2


def _url_drop_event(mime: QMimeData, event_type: QEvent.Type):
    """Build a drag/drop event around mime data the caller keeps alive.

    The event does not own its mime data, so a QMimeData created and dropped
    inside this helper would leave the handler with a dangling pointer.
    """

    dropping = event_type == QEvent.Type.Drop
    factory = QDropEvent if dropping else QDragEnterEvent
    return factory(
        QPointF(10.0, 10.0) if dropping else QPoint(10, 10),
        Qt.DropAction.CopyAction,
        mime,
        Qt.MouseButton.LeftButton,
        Qt.KeyboardModifier.NoModifier,
    )


@pytest.mark.parametrize("preview_name", ["original_preview", "paint_preview"])
def test_previews_accept_dropped_images(
    window: MainWindow, tmp_path: Path, qtbot, preview_name: str
) -> None:
    """Both preview tabs are drop targets themselves.

    A QGraphicsView accepts drag events so its items can see them, so the Rust
    preview tab used to swallow every drop before the window's handler ran.
    """

    source_path = tmp_path / "dropped.png"
    Image.new("RGB", (48, 24), (200, 40, 40)).save(source_path)
    preview = getattr(window, preview_name)
    assert preview.acceptDrops()

    mime = QMimeData()
    mime.setUrls([QUrl.fromLocalFile(str(source_path))])
    enter = _url_drop_event(mime, QEvent.Type.DragEnter)
    preview.dragEnterEvent(enter)
    assert enter.isAccepted()

    preview.dropEvent(_url_drop_event(mime, QEvent.Type.Drop))
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    assert window._image_path == source_path


@pytest.mark.parametrize("preview_name", ["original_preview", "paint_preview"])
def test_clicking_an_empty_preview_opens_the_file_browser(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, qtbot, preview_name: str
) -> None:
    opened: list[str] = []

    def fake_dialog(*args, **kwargs):
        opened.append("browse")
        return "", ""

    monkeypatch.setattr(
        main_window_module.QFileDialog, "getOpenFileName", staticmethod(fake_dialog)
    )
    preview = getattr(window, preview_name)
    viewport = getattr(preview, "viewport", None)
    qtbot.mouseClick(
        viewport() if viewport is not None else preview, Qt.MouseButton.LeftButton
    )

    assert opened == ["browse"]
    assert preview.cursor().shape() == Qt.CursorShape.PointingHandCursor


def test_dithering_uses_a_single_labeled_checkbox(window: MainWindow) -> None:
    assert window.dither_check.text() == "Dithering"
    assert not any(
        label.text() == "Enabled"
        for label in window.findChildren(main_window_module.QLabel)
    )


def test_color_button_previews_changes_before_dialog_confirmation(qtbot) -> None:
    button = ColorButton("#112233", dialog_title="Choose text color")
    qtbot.addWidget(button)
    observed: list[str] = []
    button.colorChanged.connect(lambda color: observed.append(color.name()))

    def change_then_cancel() -> None:
        dialog = button.findChild(QColorDialog)
        assert dialog is not None
        dialog.setCurrentColor(QColor("#55ff55"))
        dialog.reject()

    QTimer.singleShot(0, change_then_cancel)
    button.click()

    assert "#55ff55" in observed
    assert button.color().name() == "#112233"


def test_primary_workspace_separates_daily_flow_from_advanced_settings(
    window: MainWindow,
) -> None:
    def belongs_to(widget, ancestor) -> bool:
        parent = widget
        while parent is not None:
            if parent is ancestor:
                return True
            parent = parent.parentWidget()
        return False

    workspace = window.page_stack.widget(0)
    timelapse = window.page_stack.widget(1)
    settings = window.page_stack.widget(2)
    assert window.page_stack.currentWidget() is workspace
    assert belongs_to(window.browse_button, workspace)
    assert belongs_to(window.profile_combo, workspace)
    assert belongs_to(window.start_button, workspace)
    assert belongs_to(window.scale_mode_combo, workspace)
    assert belongs_to(window.quality_combo, workspace)
    assert belongs_to(window.crop_alignment_combo, workspace)
    assert belongs_to(window.dither_check, workspace)
    assert belongs_to(window.logical_width_spin, workspace)
    assert belongs_to(window.logical_height_spin, workspace)
    assert window.custom_resolution_panel.isHidden()
    assert belongs_to(window.stroke_speed_spin, settings)
    assert belongs_to(window.dry_run_check, settings)
    assert not belongs_to(window.dry_run_check, workspace)
    assert not window.dry_run_check.isChecked()
    assert belongs_to(window.log_view, settings)
    assert belongs_to(window.timelapse_check, timelapse)
    assert belongs_to(window.timelapse_sessions, timelapse)
    assert not belongs_to(window.timelapse_check, settings)
    assert not window.workspace_nav_button.icon().isNull()
    assert not window.timelapse_nav_button.icon().isNull()
    assert not window.settings_nav_button.icon().isNull()

    window.settings_nav_button.click()
    assert window.page_stack.currentWidget() is settings

    window.timelapse_nav_button.click()
    assert window.page_stack.currentWidget() is timelapse

    window.workspace_nav_button.click()
    window.quality_combo.setCurrentText("Custom")
    assert not window.custom_resolution_panel.isHidden()
    assert window.logical_width_spin.isEnabled()
    assert window.logical_height_spin.isEnabled()


def test_first_install_shows_replayable_getting_started_guide(
    window: MainWindow, qtbot
) -> None:
    assert window._first_run_tutorial_pending is True

    window.show()
    qtbot.waitUntil(lambda: window._tutorial_dialog is not None)
    dialog = window._tutorial_dialog
    assert dialog is not None
    assert dialog.isVisible()
    assert window._settings_document()["ui"]["tutorial_version_seen"] == TUTORIAL_VERSION
    assert window._settings_store.path.is_file()
    assert "Adaptive Palette" in " ".join(step.body for step in TUTORIAL_STEPS)

    dialog.accept()
    qtbot.waitUntil(lambda: window._tutorial_dialog is None)
    window.show_tutorial_button.click()
    qtbot.waitUntil(lambda: window._tutorial_dialog is not None)
    replay = window._tutorial_dialog
    assert replay is not None and replay.isVisible()
    replay.accept()


def test_existing_install_does_not_interrupt_with_tutorial(
    window: MainWindow, qtbot
) -> None:
    window._show_first_run_tutorial()
    dialog = window._tutorial_dialog
    assert dialog is not None
    dialog.accept()

    reopened = MainWindow()
    qtbot.addWidget(reopened)
    assert reopened._first_run_tutorial_pending is False
    reopened.show()
    qtbot.wait(10)
    assert reopened._tutorial_dialog is None
    reopened.close()


def test_new_user_path_starts_simple_and_presets_drive_expert_controls(
    window: MainWindow,
) -> None:
    """The first screen is outcome-led, while every detailed control remains."""

    assert window.experience_combo.currentText() == "Best quality"
    assert window.customize_image_panel.isHidden()
    assert window.optional_setup_panel.isHidden()
    assert window.required_setup_panel.isHidden()
    assert window.resume_panel.isHidden()

    window.experience_combo.setCurrentText("Faster")
    assert window.quality_combo.currentText() == "Fast"
    assert window.paint_mode_combo.currentData() == "fast"
    assert window.color_count_combo.currentData() == 64
    assert window.speed_preset_combo.currentText() == "Fast"

    # Editing one of the detailed choices never gets overwritten; the plain
    # selector honestly names the combination Custom instead.
    window.dither_check.setChecked(True)
    assert window.experience_combo.currentText() == "Custom"

    window.customize_image_button.click()
    assert not window.customize_image_panel.isHidden()
    assert window.quality_combo.isVisibleTo(window)


def test_required_setup_summary_names_progress_in_plain_language(
    window: MainWindow,
) -> None:
    profile = window._current_profile
    assert profile is not None
    profile.canvas = None
    profile.color_box = None
    profile.hue_bar = None
    window._refresh_profile_ui()
    assert window.setup_state_label.text() == "3 required areas remaining"
    assert "canvas" in window.setup_hint_label.text()

    profile.canvas = ScreenRect(10, 10, 400, 200)
    profile.color_box = ScreenRect(600, 100, 120, 120)
    profile.hue_bar = ScreenRect(730, 100, 20, 120)
    window._refresh_profile_ui()
    assert window.setup_state_label.text() == "Rust setup complete"
    assert window.setup_summary.property("state") == "ready"


def test_detected_setup_populates_required_and_common_optional_regions(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(window, "_update_calibration_overlay", lambda: None)
    regions = {
        "canvas": DetectedRegion(ScreenRect(100, 100, 800, 400), 0.86, "test"),
        "color_box": DetectedRegion(ScreenRect(1000, 300, 160, 160), 0.95, "test"),
        "hue_bar": DetectedRegion(ScreenRect(1164, 300, 26, 160), 0.97, "test"),
        "brush_size_box": DetectedRegion(ScreenRect(1200, 160, 40, 24), 0.68, "test"),
        "clear_button": DetectedRegion(ScreenRect(20, 20, 50, 50), 0.66, "test"),
        "save_button": DetectedRegion(ScreenRect(1000, 650, 240, 30), 0.62, "test"),
        "download_button": DetectedRegion(ScreenRect(90, 20, 45, 50), 0.64, "test"),
    }

    window._save_detected_setup(SetupDetection(regions))

    profile = window._current_profile
    assert profile is not None
    assert profile.canvas == regions["canvas"].rect
    assert profile.color_box == regions["color_box"].rect
    assert profile.hue_bar == regions["hue_bar"].rect
    assert profile.brush_size_box == regions["brush_size_box"].rect
    assert profile.clear_button == regions["clear_button"].rect
    assert profile.metadata["auto_setup_confidence"]["canvas"] == 0.86
    assert window.setup_state_label.text() == "Rust setup complete"


def test_automatic_brush_sizing_marks_its_calibration_as_required(
    window: MainWindow,
) -> None:
    window._current_profile.brush_size_box = None
    window._current_profile.clear_button = None
    window.apply_brush_check.setChecked(False)
    window._refresh_profile_ui()
    assert window.brush_size_box_status._value.text() == "Optional"
    assert window.clear_button_status._value.text() == "Optional"

    # Sizing measures the brush on the sign every run, so the control that
    # wipes the measurement off is as required as the Size field itself.
    window.apply_brush_check.setChecked(True)
    assert window.brush_size_box_status._value.text() == "Needed"
    assert window.clear_button_status._value.text() == "Needed"


def test_brush_model_status_says_what_is_still_missing(window: MainWindow) -> None:
    window.apply_brush_check.setChecked(True)
    window._current_profile.brush_size_box = None
    window._current_profile.clear_button = None
    window._current_profile.metadata.pop("brush_size_model", None)
    window._refresh_profile_ui()
    assert "Size value box and clear button" in window.brush_model_status.text()

    window._current_profile.brush_size_box = ScreenRect(220, 120, 60, 24)
    window._refresh_profile_ui()
    assert "clear button calibrated" in window.brush_model_status.text()

    window._current_profile.clear_button = ScreenRect(300, 120, 24, 24)
    window._refresh_profile_ui()
    assert "measures this sign's brush" in window.brush_model_status.text()

    window._current_profile.metadata["brush_size_model"] = fit_brush_size_model(
        [(size, size / 128.0) for size in (60, 30, 12)]
    ).to_dict()
    window._refresh_profile_ui()
    # A 2:1 rectangle and a ~128-unit brush count name the 256x128 sign
    # outright - the number a user can sanity-check against the sign.
    assert "256×128-texel texture, by Rust's sign data" in window.brush_model_status.text()

    window.apply_brush_check.setChecked(False)
    assert "Automatic brush sizing is off" in window.brush_model_status.text()


def test_quality_presets_cap_at_the_signs_measured_resolution(
    window: MainWindow,
) -> None:
    """A plan finer than the sign's texture is held at what the sign resolves.

    Rust's smallest brush covers a full texture pixel, so extra logical rows
    never land on the sign - they only make neighbouring strokes overpaint
    each other and blur fine detail the preview promised.  Once a job has
    measured the sign, the planner must stay inside that ceiling.
    """

    assert window._current_profile is not None
    window._current_profile.canvas = ScreenRect(10, 10, 200, 100)
    window._current_profile.metadata["brush_size_model"] = fit_brush_size_model(
        [(size, size / 128.0) for size in (60, 30, 12)]
    ).to_dict()
    window._refresh_profile_ui()

    # The preset wants 512×256; the sign resolves 128 rows.
    window.quality_combo.setCurrentText("Very High")
    assert window.logical_width_spin.value() == 256
    assert window.logical_height_spin.value() == 128

    # Custom asks are held to the same physics.
    window.quality_combo.setCurrentText("Custom")
    window.logical_width_spin.setValue(512)
    assert window.logical_width_spin.value() == 256
    assert window.logical_height_spin.value() == 128

    # Requests below the ceiling pass through untouched.
    window.quality_combo.setCurrentText("Very Fast")
    assert window.logical_width_spin.value() == 64
    assert window.logical_height_spin.value() == 32


def test_max_quality_plans_one_cell_per_measured_texel(
    window: MainWindow,
) -> None:
    """Max has no fixed long edge: it asks the sign what it holds.

    The measured count carries noise - 130-ish rows on a 128-row sign - and
    the preset must land on the canonical texture size, or every cell fights
    its neighbour over a texel that is not there.
    """

    assert window._current_profile is not None
    window._current_profile.canvas = ScreenRect(10, 10, 200, 100)
    window._current_profile.metadata["brush_size_model"] = fit_brush_size_model(
        [(size, size / 130.5) for size in (60, 30, 12)]
    ).to_dict()
    window._refresh_profile_ui()

    window.quality_combo.setCurrentText("Max")
    assert window.logical_width_spin.value() == 256
    assert window.logical_height_spin.value() == 128


def test_presets_the_sign_cannot_hold_are_greyed_out(
    window: MainWindow,
) -> None:
    """Turning the quality up on a small sign used to do nothing, silently.

    High and Very High ask for more cells than a 320x240 sign has, so they
    were held at 320x240 and painted exactly what Max paints - a setting
    that looks finer and is not.  They are offered as unavailable instead,
    and the panel says why.
    """

    from PySide6.QtCore import Qt

    from app.texel_grid import TexelGridModel

    def enabled(preset: str) -> bool:
        index = window.quality_combo.findText(preset)
        flags = window.quality_combo.itemData(index, Qt.ItemDataRole.UserRole - 1)
        return flags is None or bool(flags & Qt.ItemFlag.ItemIsEnabled)

    assert window._current_profile is not None
    window._current_profile.canvas = ScreenRect(10, 10, 1299, 1085)
    window.quality_combo.setCurrentText("High")
    # Nothing measured yet: every preset is still worth offering.
    assert enabled("High") and enabled("Very High")

    window._current_profile.metadata["texel_grid"] = TexelGridModel(
        columns=320, rows=240, pitch_x=4.06, pitch_y=4.51, origin_x=10.0, origin_y=10.0
    ).to_dict()
    window._refresh_profile_ui()

    # The two that cannot be delivered are greyed out, and the selection
    # moves off the one that was chosen - to the same size, honestly named.
    assert not enabled("High")
    assert not enabled("Very High")
    assert enabled("Balanced") and enabled(MAX_QUALITY_PRESET)
    assert window.quality_combo.currentText() == MAX_QUALITY_PRESET
    assert (window.logical_width_spin.value(), window.logical_height_spin.value()) == (
        320,
        240,
    )
    # Picking one from a saved setting lands on Max just the same.
    window.quality_combo.setCurrentText("Very High")
    assert window.quality_combo.currentText() == MAX_QUALITY_PRESET

    # The panel says why the greyed entries are greyed.
    assert window.resolution_cap_panel.isVisibleTo(window)
    text = window.resolution_cap_label.text()
    assert "320×240" in text
    assert "High and Very High" in text
    assert "greyed out" in text
    assert "measured" in window.resolution_cap_panel.toolTip()
    # The unavailable entries carry the reason too.
    index = window.quality_combo.findText("High")
    assert "320×240" in window.quality_combo.itemData(index, Qt.ItemDataRole.ToolTipRole)

    # A preset the sign can hold is quiet and selectable.
    window.quality_combo.setCurrentText("Fast")
    assert not window.resolution_cap_panel.isVisibleTo(window)

    # Losing the measurement restores every preset.
    window._current_profile.metadata.pop("texel_grid")
    window._refresh_profile_ui()
    assert enabled("High") and enabled("Very High")
    window.quality_combo.setCurrentText("Very High")
    assert window.quality_combo.currentText() == "Very High"
    assert not window.resolution_cap_panel.isVisibleTo(window)


def test_a_measured_texel_grid_outranks_the_brush_inference(
    window: MainWindow,
) -> None:
    """A grid counted on the sign is the ceiling; the brush only guesses it.

    The brush measurement here reads ~315 Size units, which Rust's sign table
    makes a 512x256 texture on this 2:1 canvas.  The grid says the sign
    is really 300x150 - a
    size no table lists - and Max has to plan exactly that, because a cell
    planned on a texel that is not there fights its neighbour for one that is.
    """

    from app.texel_grid import TexelGridModel

    assert window._current_profile is not None
    window._current_profile.canvas = ScreenRect(10, 10, 900, 450)
    window._current_profile.metadata["brush_size_model"] = fit_brush_size_model(
        [(size, size / 315.0) for size in (60, 30, 12)]
    ).to_dict()
    window._current_profile.metadata["texel_grid"] = TexelGridModel(
        columns=300, rows=150, pitch_x=3.0, pitch_y=3.0, origin_x=10.0, origin_y=10.0
    ).to_dict()
    window._refresh_profile_ui()

    window.quality_combo.setCurrentText("Max")
    assert window.logical_width_spin.value() == 300
    assert window.logical_height_spin.value() == 150

    # A re-framed sign of another shape forgets the grid with the brush; the
    # brush's ~315 units are about 252 texels, which on a 2:1 sign is 512x256.
    window._current_profile.metadata.pop("texel_grid")
    window._refresh_profile_ui()
    window.quality_combo.setCurrentText("Balanced")
    window.quality_combo.setCurrentText("Max")
    assert window.logical_height_spin.value() == 256
    assert window.logical_width_spin.value() == 512


def test_max_quality_without_a_measurement_plans_the_screen_grid(
    window: MainWindow,
) -> None:
    """Max works from the first paint, before any job has measured the sign.

    Unmeasured, it plans one logical cell per screen pixel of the calibrated
    canvas - the finest grid the mouse can address, which brush size 1 paints
    without losing detail.  A measurement snaps the same selection onto the
    sign's true texel grid, and losing the measurement drops it back to the
    screen grid instead of abandoning the preset.
    """

    assert window._current_profile is not None
    window._current_profile.canvas = ScreenRect(10, 10, 200, 100)
    window._current_profile.metadata.pop("brush_size_model", None)
    window._refresh_profile_ui()

    index = window.quality_combo.findText("Max")
    assert index >= 0
    assert window.quality_combo.model().item(index).isEnabled()

    window.quality_combo.setCurrentText("Max")
    assert window.logical_width_spin.value() == 200
    assert window.logical_height_spin.value() == 100

    # Every real path that changes the model or canvas - profile switch,
    # canvas recalibration, a finished job storing its measurement - refreshes
    # the profile UI and re-derives the quality dimensions, so the test walks
    # the same pair.
    window._current_profile.metadata["brush_size_model"] = fit_brush_size_model(
        [(size, size / 128.0) for size in (60, 30, 12)]
    ).to_dict()
    window._refresh_profile_ui()
    window._update_quality_dimensions()
    assert window.quality_combo.currentText() == "Max"
    assert window.logical_width_spin.value() == 256
    assert window.logical_height_spin.value() == 128

    window._current_profile.metadata.pop("brush_size_model", None)
    window._refresh_profile_ui()
    window._update_quality_dimensions()
    assert window.quality_combo.currentText() == "Max"
    assert window.logical_width_spin.value() == 200
    assert window.logical_height_spin.value() == 100


def test_plan_summary_announces_a_capped_resolution(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    assert window._current_profile is not None
    window._current_profile.canvas = ScreenRect(10, 10, 200, 100)
    window._current_profile.metadata["brush_size_model"] = fit_brush_size_model(
        [(size, size / 128.0) for size in (60, 30, 12)]
    ).to_dict()
    window._refresh_profile_ui()
    # Very High is greyed out on this sign, so the way to ask for more than
    # it holds is a custom resolution.
    window.quality_combo.setCurrentText("Custom")
    window.logical_width_spin.setValue(512)

    source_path = tmp_path / "source.png"
    Image.new("RGB", (64, 32), (210, 30, 40)).save(source_path)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)

    # The plan, and therefore the Rust preview, stays inside the ceiling and
    # says so next to the stroke counts instead of silently swapping sizes.
    assert (window._plan.width, window._plan.height) == (256, 128)
    assert "capped at 256×128" in window.processing_label.text()

    # Max is not capped - it asked for the ceiling - but the summary still
    # names the size, which is what a finished plan is read for.
    window.quality_combo.setCurrentText(MAX_QUALITY_PRESET)
    qtbot.waitUntil(
        lambda: "full resolution" in window.processing_label.text(), timeout=5000
    )
    assert "256×128, this sign's full resolution" in window.processing_label.text()


def test_dry_run_completes_without_sendinput(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    source_path = tmp_path / "small.png"
    Image.new("RGB", (16, 8), (50, 180, 90)).save(source_path)
    window.quality_combo.setCurrentText("Custom")
    window.logical_width_spin.setValue(16)
    window.logical_height_spin.setValue(8)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)

    window.dry_run_check.setChecked(True)
    window.countdown_spin.setValue(0)
    window._start_or_resume()
    assert window._countdown is not None
    window._countdown._tick()
    qtbot.waitUntil(
        lambda: window._painter is not None
        and getattr(window._painter.state, "value", "") == "completed",
        timeout=5000,
    )
    qtbot.waitUntil(lambda: window.paint_progress.value() == 1000, timeout=2000)

    assert window._painter.input.is_dry_run
    assert window.paint_progress.value() == 1000
    assert window._painter.progress.completed_strokes == window._plan.stroke_count


def test_cancelled_countdown_cannot_fire_later(qtbot) -> None:
    callbacks: list[str] = []
    dialog = CountdownDialog(1, lambda: callbacks.append("started"))
    qtbot.addWidget(dialog)
    dialog.show()
    dialog.reject()

    qtbot.wait(1200)
    assert callbacks == []
    assert not dialog._timer.isActive()


def test_custom_resolution_and_alpha_fill_follow_canvas(window: MainWindow) -> None:
    window._current_profile.canvas = ScreenRect(100, 100, 300, 100)
    window._refresh_profile_ui()
    window.quality_combo.setCurrentText("Custom")
    window.logical_width_spin.setValue(300)

    assert window.logical_height_spin.value() == 100

    window._set_combo_data(window.scale_mode_combo, ScaleMode.STRETCH.value)
    window._set_combo_data(
        window.transparency_combo, TransparencyMode.USE_BACKGROUND.value
    )
    window._set_combo_data(window.background_combo, "custom")
    assert window.background_combo.isEnabled()
    assert window.background_color_button.isEnabled()


def test_live_debug_stroke_is_immediately_abortable(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, qtbot
) -> None:
    class GatedDebugInput(MockInputController):
        emits_real_input = True

        def __init__(self) -> None:
            super().__init__(initial_position=(500, 400))
            self.mouse_went_down = threading.Event()

        def mouse_down(self, button="left") -> None:
            super().mouse_down(button)
            self.mouse_went_down.set()

    controller = GatedDebugInput()
    monkeypatch.setattr(
        "app.input_controller.create_system_input_controller", lambda: controller
    )
    profile = window._current_profile
    profile.canvas = ScreenRect(100, 100, 400, 80)
    profile.color_box = ScreenRect(600, 100, 100, 100)
    profile.hue_bar = ScreenRect(720, 100, 12, 100)
    window.dry_run_check.setChecked(False)
    window._hotkeys_ready = True
    window._hotkeys = type("LiveHotkeys", (), {"running": True})()
    window.focus_guard_check.setChecked(False)
    window.stroke_speed_spin.setValue(10.0)
    window._pending_start_cancelled = False
    window._debug_abort_event.clear()

    window._execute_debug_action("test_stroke")
    assert controller.mouse_went_down.wait(1.0)
    window._hotkey_abort_immediate()
    event_count = len(controller.events)
    qtbot.waitUntil(lambda: not window._debug_running, timeout=2000)

    assert not controller.held_buttons
    assert not any(
        event.kind in {"move", "mouse_down"}
        for event in controller.events[event_count:]
    )


def test_real_debug_action_requires_live_abort_hotkey(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    window.dry_run_check.setChecked(False)
    window._hotkeys_ready = False
    window._current_profile.canvas = ScreenRect(100, 100, 400, 80)
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        "app.gui.main_window.QMessageBox.critical",
        lambda _parent, title, message, *_args: shown.append((title, message)),
    )

    window._run_debug_action("canvas_center")

    assert shown and shown[0][0] == "Emergency hotkey unavailable"
    assert not window._debug_running
    assert window._countdown is None


def test_hotkey_runtime_error_disables_real_start(window: MainWindow) -> None:
    class DeadHotkeys:
        running = False

    window._hotkeys = DeadHotkeys()
    window._hotkeys_ready = True
    window.dry_run_check.setChecked(False)

    window._on_hotkey_error("message loop stopped")

    assert window._hotkeys_ready is False
    assert not window.start_button.isEnabled()


def test_hotkey_thread_failure_immediately_aborts_published_painter(
    window: MainWindow,
) -> None:
    reasons: list[str] = []

    class ActivePainter:
        def abort(self, reason: str) -> bool:
            reasons.append(reason)
            return True

    window._painter = ActivePainter()
    window._hotkeys_ready = True
    hotkey = threading.Thread(
        target=window._hotkey_failure_immediate,
        args=(RuntimeError("message loop failed"),),
    )
    hotkey.start()
    hotkey.join(1.0)

    assert not hotkey.is_alive()
    assert not window._hotkeys_ready
    assert window._pending_start_cancelled
    assert reasons == ["global emergency hotkey"]
    window._painter = None


def test_dead_hotkey_at_countdown_completion_blocks_real_input(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = PaintPlan(
        1,
        1,
        (ColorGroup((100, 150, 200), (Stroke(0, 0, 0, 0),), 1),),
    )
    window._pending_paint = _PendingPaint(
        plan=plan,
        profile=window._current_profile,
        settings=window._settings_document(),
        dry_run=False,
    )
    window._pending_start_cancelled = False
    window._hotkeys_ready = True
    window._hotkeys = type("DeadHotkeys", (), {"running": False})()
    shown: list[str] = []
    monkeypatch.setattr(
        "app.gui.main_window.QMessageBox.critical",
        lambda _parent, _title, message, *_args: shown.append(message),
    )

    window._begin_paint_after_countdown()

    assert shown and "stopped during the countdown" in shown[0]
    assert window._painter is None


def test_f10_at_gui_countdown_boundary_cancels_pending_start(
    window: MainWindow, qtbot
) -> None:
    plan = PaintPlan(
        1,
        1,
        (ColorGroup((100, 150, 200), (Stroke(0, 0, 0, 0),), 1),),
    )
    window._plan = plan
    window.dry_run_check.setChecked(True)
    window.countdown_spin.setValue(0)
    window._start_or_resume()
    dialog = window._countdown
    assert dialog is not None

    # Run the immediate half exactly as the Win32 hotkey thread does, while
    # deliberately leaving its queued Qt cleanup pending until the countdown
    # reaches zero.
    hotkey = threading.Thread(target=window._hotkey_abort_immediate)
    hotkey.start()
    hotkey.join(1.0)
    assert not hotkey.is_alive()
    assert window._pending_start_cancelled
    dialog._tick()
    qtbot.wait(20)

    assert window._pending_paint is None
    assert window._painter is None


def test_f10_after_ready_painter_publish_prevents_worker_start(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, qtbot
) -> None:
    plan = PaintPlan(
        1,
        1,
        (ColorGroup((100, 150, 200), (Stroke(0, 0, 0, 0),), 1),),
    )
    entered_start = threading.Event()
    allow_start = threading.Event()

    class GatedPainter(Painter):
        def start(self, *args, **kwargs):
            entered_start.set()
            assert allow_start.wait(1.0)
            return super().start(*args, **kwargs)

    monkeypatch.setattr("app.painter.Painter", GatedPainter)
    settings = window._settings_document()
    window._pending_paint = _PendingPaint(
        plan=plan,
        profile=window._execution_profile(True, plan),
        settings=settings,
        dry_run=True,
    )
    window._pending_start_cancelled = False

    def abort_at_start_gate() -> None:
        assert entered_start.wait(1.0)
        window._hotkey_abort_immediate()
        allow_start.set()

    hotkey = threading.Thread(target=abort_at_start_gate)
    hotkey.start()
    window._begin_paint_after_countdown()
    hotkey.join(1.0)
    qtbot.wait(20)

    assert not hotkey.is_alive()
    assert window._painter is not None
    assert window._painter.state is PainterState.ABORTED
    assert window._painter.input.events == []
    assert not window._painter.is_alive


def test_failed_debug_release_stays_locked_until_abort_retry(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    class RetryReleaseController:
        def __init__(self) -> None:
            self.calls = 0

        def release_all(self) -> None:
            self.calls += 1
            if self.calls == 1:
                raise OSError("temporary release failure")

    controller = RetryReleaseController()
    window._debug_controller = controller
    window._debug_thread = None
    window._debug_running = True
    monkeypatch.setattr(
        "app.gui.main_window.QMessageBox.warning", lambda *_args, **_kwargs: None
    )

    window._on_debug_finished("completed", "test move")

    assert window._debug_controller is controller
    assert window._debug_running
    assert window.abort_button.isEnabled()

    window._abort_painting()
    assert controller.calls == 2
    assert window._debug_controller is None
    assert not window._debug_running


def test_stale_same_generation_running_callback_cannot_hide_abort(
    window: MainWindow,
) -> None:
    class AbortedPainter:
        state = PainterState.ABORTED

    window._painter = AbortedPainter()
    window._paint_generation = 7
    window.state_badge.setText("STOPPED")

    window._on_paint_state(7, PainterState.RUNNING, "late worker callback")

    assert window.state_badge.text() == "STOPPED"


def test_previous_held_button_blocks_painter_replacement(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = PaintPlan(
        1,
        1,
        (ColorGroup((100, 150, 200), (Stroke(0, 0, 0, 0),), 1),),
    )

    class StuckInput:
        held_buttons = frozenset({"left"})

        def release_all(self) -> None:
            raise OSError("button-up rejected")

    class OldPainter:
        input = StuckInput()

        def shutdown(self, timeout: float) -> None:
            del timeout

    old_painter = OldPainter()
    window._painter = old_painter
    window._pending_paint = _PendingPaint(
        plan=plan,
        profile=window._execution_profile(True, plan),
        settings=window._settings_document(),
        dry_run=True,
    )
    window._pending_start_cancelled = False
    errors: list[str] = []
    monkeypatch.setattr(
        window, "_on_paint_error", lambda _generation, message: errors.append(message)
    )

    window._begin_paint_after_countdown()

    assert window._painter is old_painter
    assert errors and "button-up rejected" in errors[0]


def test_previous_live_worker_blocks_painter_replacement(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    plan = PaintPlan(
        1,
        1,
        (ColorGroup((100, 150, 200), (Stroke(0, 0, 0, 0),), 1),),
    )

    class ReleasedInput:
        held_buttons = frozenset()

        def release_all(self) -> None:
            return None

    class OldPainter:
        input = ReleasedInput()
        is_alive = True

        def shutdown(self, timeout: float) -> None:
            del timeout

    old_painter = OldPainter()
    window._painter = old_painter
    window._pending_paint = _PendingPaint(
        plan=plan,
        profile=window._execution_profile(True, plan),
        settings=window._settings_document(),
        dry_run=True,
    )
    window._pending_start_cancelled = False
    errors: list[str] = []
    monkeypatch.setattr(
        window, "_on_paint_error", lambda _generation, message: errors.append(message)
    )

    window._begin_paint_after_countdown()

    assert window._painter is old_painter
    assert errors and "did not stop in time" in errors[0]


def test_latest_async_image_import_wins_when_an_older_decode_is_running(
    window: MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot,
) -> None:
    first = tmp_path / "slow-red.png"
    second = tmp_path / "latest-blue.png"
    Image.new("RGB", (20, 10), (230, 20, 20)).save(first)
    Image.new("RGB", (24, 12), (20, 40, 230)).save(second)
    first_started = threading.Event()
    original_open = Image.open

    def delayed_open(path, *args, **kwargs):
        if Path(path) == first:
            first_started.set()
            time.sleep(0.15)
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(main_window_module.Image, "open", delayed_open)

    window.load_image(first)
    qtbot.waitUntil(first_started.is_set, timeout=2000)
    window.load_image(second)
    qtbot.waitUntil(
        lambda: window._image_path == second and window._plan is not None,
        timeout=5000,
    )

    assert window._original_image is not None
    assert window._original_image.size == (24, 12)
    assert window._original_image.getpixel((0, 0)) == (20, 40, 230, 255)


def test_processing_and_preview_generation_run_off_the_gui_thread(
    window: MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot,
) -> None:
    path = tmp_path / "thread-check.png"
    Image.new("RGB", (32, 16), (30, 170, 80)).save(path)
    gui_thread = threading.get_ident()
    processing_threads: list[int] = []
    simulation_threads: list[int] = []
    original_process = main_window_module.process_image
    original_simulation = main_window_module._build_simulation_image

    def observed_process(*args, **kwargs):
        processing_threads.append(threading.get_ident())
        return original_process(*args, **kwargs)

    def observed_simulation(*args, **kwargs):
        simulation_threads.append(threading.get_ident())
        return original_simulation(*args, **kwargs)

    monkeypatch.setattr(main_window_module, "process_image", observed_process)
    monkeypatch.setattr(
        main_window_module,
        "_build_simulation_image",
        observed_simulation,
    )

    window.load_image(path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)

    assert processing_threads and all(value != gui_thread for value in processing_threads)
    assert simulation_threads and all(value != gui_thread for value in simulation_threads)


def test_preview_renders_artwork_through_the_measured_sign_response() -> None:
    # A sign measured at 60% brightness: a mid gray survives the correction
    # round trip untouched, but white is out of reach, and the preview has to
    # show the color the material will really produce rather than promise one.
    model = ColorCorrectionModel(
        forward_matrix=(
            (0.6, 0.0, 0.0, 0.0),
            (0.0, 0.6, 0.0, 0.0),
            (0.0, 0.0, 0.6, 0.0),
        ),
        fit_rmse=0.01,
        sample_count=32,
        captured_at="2026-01-01T00:00:00+00:00",
    )
    image = Image.new("RGBA", (2, 1))
    image.putdata([(120, 120, 120, 255), (255, 255, 255, 255)])
    processed = ProcessedImage(image, np.ones((1, 2), dtype=bool), 2)

    plain = main_window_module._build_simulation_image(processed)
    corrected = main_window_module._build_simulation_image(processed, model)

    assert plain.getpixel((0, 0)) == (120, 120, 120)
    assert plain.getpixel((1, 0)) == (255, 255, 255)
    assert corrected.getpixel((0, 0)) == (120, 120, 120)
    assert corrected.getpixel((1, 0))[0] == pytest.approx(153, abs=2)


def test_preview_ignores_the_correction_while_the_chart_is_loaded(
    window: MainWindow, tmp_path: Path
) -> None:
    profile = window._current_profile
    profile.metadata["color_correction"] = {
        "forwardMatrix": [[0.6, 0, 0, 0], [0, 0.6, 0, 0], [0, 0, 0.6, 0]],
        "fitRmse": 0.01,
        "sampleCount": 32,
        "capturedAt": "2026-01-01T00:00:00+00:00",
    }
    assert window._color_correction_model() is not None

    # The chart measures the raw material response and must not be shown, or
    # painted, through an earlier measurement of itself.
    chart = tmp_path / "chart.png"
    Image.new("RGB", (8, 4), (10, 20, 30)).save(chart)
    window._color_chart_profile_id = profile.id
    window._color_chart_path = chart
    window._image_path = chart

    assert window._color_correction_model() is None


def test_custom_resolution_tracks_both_canvas_axes(window: MainWindow) -> None:
    assert window._current_profile is not None
    window._current_profile.canvas = ScreenRect(10, 20, 300, 100)
    window.quality_combo.setCurrentText("Custom")

    window.logical_width_spin.setValue(300)
    assert window.logical_height_spin.value() == 100

    window.logical_height_spin.setValue(75)
    assert window.logical_width_spin.value() == 225


def test_mouse_wheel_cannot_change_setting_controls(window: MainWindow) -> None:
    class IgnoredWheel:
        def __init__(self) -> None:
            self.ignored = False

        def ignore(self) -> None:
            self.ignored = True

    controls = (
        window.quality_combo,
        window.color_count_combo,
        window.logical_width_spin,
        window.stroke_speed_spin,
        window.profile_combo,
        window.text_font_combo,
    )
    for control in controls:
        before = (
            control.currentIndex()
            if hasattr(control, "currentIndex")
            else control.value()
        )
        event = IgnoredWheel()
        control.wheelEvent(event)
        after = (
            control.currentIndex()
            if hasattr(control, "currentIndex")
            else control.value()
        )
        assert event.ignored
        assert after == before


def test_prepare_color_chart_configures_raw_chart_job(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, qtbot
) -> None:
    profile = window._current_profile
    profile.canvas = ScreenRect(10, 10, 800, 400)
    profile.color_box = ScreenRect(820, 10, 120, 120)
    profile.hue_bar = ScreenRect(950, 10, 20, 120)
    profile.metadata["color_correction"] = {
        "schemaVersion": 1,
        "forwardMatrix": [
            [0.8, 0.0, 0.0, 0.0],
            [0.0, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.0],
        ],
        "fitRmse": 0.01,
        "sampleCount": 32,
        "capturedAt": "now",
    }
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda *_args, **_kwargs: main_window_module.QMessageBox.StandardButton.Yes,
    )

    window._prepare_color_chart()
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)

    assert window._image_path == window._color_chart_path
    assert window.scale_mode_combo.currentData() == ScaleMode.STRETCH.value
    assert window.quality_combo.currentText() == "Very Fast"
    assert window.color_count_combo.currentData() == 32
    assert not window.dither_check.isChecked()

    window.dry_run_check.setChecked(True)
    window.countdown_spin.setValue(0)
    window._start_or_resume()
    assert window._pending_paint is not None
    assert "color_correction" not in window._pending_paint.profile.metadata
    window._countdown.reject()


def test_measure_painted_chart_saves_profile_correction(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, qtbot
) -> None:
    profile = window._current_profile
    profile.canvas = ScreenRect(10, 10, 800, 400)
    profile.color_box = ScreenRect(820, 10, 120, 120)
    profile.hue_bar = ScreenRect(950, 10, 20, 120)
    window.focus_guard_check.setChecked(False)
    monkeypatch.setattr(
        "app.screen.get_virtual_screen",
        lambda: VirtualScreen(0, 0, 1200, 800),
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda *_args, **_kwargs: main_window_module.QMessageBox.StandardButton.Yes,
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_a, **_k: 0)
    monkeypatch.setattr(main_window_module.QMessageBox, "critical", lambda *_a, **_k: 0)
    window._prepare_color_chart()
    qtbot.waitUntil(lambda: window._processed is not None, timeout=5000)

    command = window._processed.image.resize((800, 400), Image.Resampling.NEAREST)
    array = np.asarray(command.convert("RGB"), dtype=np.float64) / 255.0
    rendered = np.clip(array * np.asarray((0.72, 0.78, 0.68)) + 0.05, 0.0, 1.0)
    capture = Image.fromarray(np.rint(rendered * 255).astype(np.uint8), "RGB")
    monkeypatch.setattr("app.screen.capture_region", lambda _rect: capture)

    window._do_measure_color_chart()

    correction = window._current_profile.metadata.get("color_correction")
    assert isinstance(correction, dict)
    assert correction["sampleCount"] == 32
    assert correction["fitRmse"] < 0.02


def test_transparent_background_choice_is_visible_and_persisted(
    window: MainWindow,
) -> None:
    window._set_combo_data(
        window.transparency_combo,
        TransparencyMode.USE_BACKGROUND.value,
    )
    assert window.background_combo.currentData() == "white"

    # Re-selecting an impossible combination is normalized immediately rather
    # than being displayed as unpainted while processing silently uses white.
    window._set_combo_data(window.background_combo, "unpainted")
    assert window.background_combo.currentData() == "white"
    document = window._settings_document()
    assert document["image"]["transparent_pixels"] == "use_background"
    assert document["image"]["background_mode"] == "white"


def test_alpha_fill_is_off_by_default_and_reaches_the_rust_preview(
    window: MainWindow,
) -> None:
    """The Rust preview must not show a fill the settings say is off.

    Painting has no transparency, so a soft edge either becomes solid
    background or is left alone; the checkbox is what decides, and it starts
    out leaving it alone.
    """

    assert not window.alpha_fill_check.isChecked()
    assert window._processing_options().alpha_fill is False
    assert window._settings_document()["image"]["alpha_fill"] is False

    window.alpha_fill_check.setChecked(True)
    assert window._processing_options().alpha_fill is True
    assert window._settings_document()["image"]["alpha_fill"] is True


def test_gui_persists_complete_brush_settings_and_future_keys(
    window: MainWindow,
) -> None:
    window._settings["painting"]["future_tuning_value"] = 321
    window.brush_delay_spin.setValue(275)
    window.apply_brush_check.setChecked(True)

    saved = window._settings_store.save(window._settings_document())

    assert saved["painting"]["brush_size"] == 0.0
    assert saved["painting"]["delay_after_brush_seconds"] == pytest.approx(0.275)
    assert saved["painting"]["apply_brush_size"] is True
    assert saved["painting"]["brush_direction"] == "low_to_high"
    assert saved["painting"]["future_tuning_value"] == 321


def test_size_value_box_is_required_only_when_brush_application_is_enabled(
    window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = Profile.new(
        "No size box",
        canvas=ScreenRect(10, 10, 200, 100),
        color_box=ScreenRect(220, 10, 100, 100),
        hue_bar=ScreenRect(325, 10, 10, 100),
    )
    monkeypatch.setattr(
        "app.screen.get_virtual_screen",
        lambda: VirtualScreen(0, 0, 800, 600),
    )

    window._validate_profile_on_virtual_screen(profile, apply_brush_size=False)
    with pytest.raises(ValueError, match="numeric Size field"):
        window._validate_profile_on_virtual_screen(profile, apply_brush_size=True)

    # Sizing measures the brush on the sign, so it also needs the control that
    # wipes the measurement off before the artwork goes down.
    profile.brush_size_box = ScreenRect(220, 120, 60, 24)
    with pytest.raises(ValueError, match="clear button"):
        window._validate_profile_on_virtual_screen(profile, apply_brush_size=True)

    profile.clear_button = ScreenRect(300, 120, 24, 24)
    window._validate_profile_on_virtual_screen(profile, apply_brush_size=True)


def _profile_with_full_calibration(display: DisplayMetadata) -> Profile:
    profile = Profile.new(
        "GUI calibration test",
        canvas=ScreenRect(10, 10, 200, 100),
        color_box=ScreenRect(220, 10, 100, 100),
        hue_bar=ScreenRect(325, 10, 10, 100),
        brush_size_box=ScreenRect(220, 120, 60, 24),
        display=display,
        metadata={"ui_reference": {"path": "old.png"}},
    )
    profile.metadata["brush_size_model"] = fit_brush_size_model(
        [(size, size / 128.0) for size in (60, 30, 12)]
    ).to_dict()
    return profile


def test_partial_recalibration_preserves_other_regions_on_same_display(
    window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display = DisplayMetadata(
        virtual_screen=ScreenRect(0, 0, 800, 600),
        coordinate_space="logical",
    )
    profile = window._profile_store.save(_profile_with_full_calibration(display))
    window._current_profile = profile
    new_canvas = ScreenRect(20, 30, 240, 120)
    monkeypatch.setattr(main_window_module, "select_screen_rect", lambda *_a, **_k: new_canvas)
    monkeypatch.setattr(main_window_module, "capture_display_metadata", lambda: display)

    window._begin_calibration("canvas", "canvas")

    assert window._current_profile.canvas == new_canvas
    assert window._current_profile.color_box == profile.color_box
    assert window._current_profile.hue_bar == profile.hue_bar
    assert "ui_reference" in window._current_profile.metadata
    # Re-framing the same sign leaves its shape alone, and the brush model is a
    # fraction of the sign, so it has to survive standing somewhere else.
    assert "brush_size_model" in window._current_profile.metadata


def test_recalibrating_onto_a_differently_shaped_sign_drops_the_brush_model(
    window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display = DisplayMetadata(
        virtual_screen=ScreenRect(0, 0, 800, 600),
        coordinate_space="logical",
    )
    profile = window._profile_store.save(_profile_with_full_calibration(display))
    window._current_profile = profile
    # 200x100 was 2:1; a square sign is a different sign, whose texture
    # resolution the old measurement says nothing about.
    square_canvas = ScreenRect(20, 30, 160, 160)
    monkeypatch.setattr(
        main_window_module, "select_screen_rect", lambda *_a, **_k: square_canvas
    )
    monkeypatch.setattr(main_window_module, "capture_display_metadata", lambda: display)

    window._begin_calibration("canvas", "canvas")

    assert window._current_profile.canvas == square_canvas
    assert "brush_size_model" not in window._current_profile.metadata


def test_recalibration_after_display_change_invalidates_stale_regions(
    window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    old_display = DisplayMetadata(
        virtual_screen=ScreenRect(0, 0, 800, 600),
        coordinate_space="logical",
    )
    new_display = DisplayMetadata(
        virtual_screen=ScreenRect(-200, 0, 1000, 600),
        coordinate_space="logical",
    )
    profile = window._profile_store.save(_profile_with_full_calibration(old_display))
    window._current_profile = profile
    new_canvas = ScreenRect(-180, 20, 240, 120)
    monkeypatch.setattr(main_window_module, "select_screen_rect", lambda *_a, **_k: new_canvas)
    monkeypatch.setattr(main_window_module, "capture_display_metadata", lambda: new_display)
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_a, **_k: 0)

    window._begin_calibration("canvas", "canvas")

    assert window._current_profile.canvas == new_canvas
    assert window._current_profile.color_box is None
    assert window._current_profile.hue_bar is None
    assert window._current_profile.brush_size_box is None
    assert "ui_reference" not in window._current_profile.metadata


def test_failed_calibration_save_does_not_mutate_live_profile(
    window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    display = DisplayMetadata(
        virtual_screen=ScreenRect(0, 0, 800, 600),
        coordinate_space="logical",
    )
    profile = window._profile_store.save(_profile_with_full_calibration(display))
    window._current_profile = profile
    old_document = profile.to_dict()
    monkeypatch.setattr(
        main_window_module,
        "select_screen_rect",
        lambda *_a, **_k: ScreenRect(40, 50, 260, 130),
    )
    monkeypatch.setattr(main_window_module, "capture_display_metadata", lambda: display)
    monkeypatch.setattr(
        window._profile_store,
        "save",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "critical", lambda *_a, **_k: 0)

    window._begin_calibration("canvas", "canvas")

    assert window._current_profile.to_dict() == old_document


def test_picker_mapping_is_fixed_for_current_rust_ui(window: MainWindow) -> None:
    profile = window._current_profile
    assert profile is not None
    assert profile.hue_direction == "bottom_to_top"
    assert profile.saturation_direction == "left_low"
    assert profile.value_direction == "top_bright"
    assert not hasattr(window, "hue_direction_combo")


def test_failed_reference_profile_save_does_not_mutate_live_metadata(
    window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = window._current_profile
    assert profile is not None
    profile.color_box = ScreenRect(220, 10, 100, 100)
    profile.hue_bar = ScreenRect(325, 10, 10, 100)
    original_metadata = dict(profile.metadata)
    monkeypatch.setattr("app.screen.save_reference", lambda *_a, **_k: None)
    monkeypatch.setattr(
        window._profile_store,
        "save",
        lambda *_a, **_k: (_ for _ in ()).throw(OSError("disk full")),
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *_a, **_k: 0)

    window._do_capture_reference()

    assert window._current_profile.metadata == original_metadata


def test_main_window_owns_only_one_countdown_and_close_cancels_it(
    window: MainWindow,
    qtbot,
) -> None:
    callbacks: list[str] = []
    assert window._launch_countdown(1, lambda: callbacks.append("first"), hint="test")
    assert not window._launch_countdown(1, lambda: callbacks.append("second"), hint="test")

    window.close()
    qtbot.wait(1100)

    assert callbacks == []
    assert window._countdown is None


def test_close_ignores_an_image_decode_that_finishes_during_shutdown(
    window: MainWindow,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    qtbot,
) -> None:
    path = tmp_path / "close-during-load.png"
    Image.new("RGB", (20, 10), (100, 120, 140)).save(path)
    decode_started = threading.Event()
    original_open = Image.open

    def delayed_open(source, *args, **kwargs):
        decode_started.set()
        time.sleep(0.1)
        return original_open(source, *args, **kwargs)

    monkeypatch.setattr(main_window_module.Image, "open", delayed_open)
    window.load_image(path)
    qtbot.waitUntil(decode_started.is_set, timeout=2000)

    window.close()
    qtbot.wait(20)

    assert window._closing
    assert window._original_image is None
    assert window._plan is None


def test_running_job_swaps_the_plan_panel_for_large_progress(
    window: MainWindow,
) -> None:
    from types import SimpleNamespace

    assert window.plan_progress_stack.currentIndex() == 0
    assert not window.progress_frame.isHidden()

    generation = window._paint_generation
    window._on_paint_state(generation, SimpleNamespace(value="running"), "started")
    assert window.plan_progress_stack.currentIndex() == 1
    assert window.progress_frame.isHidden()
    assert window.active_progress_title.text() == "PAINTING"

    window._on_paint_state(generation, SimpleNamespace(value="paused"), "user")
    assert window.plan_progress_stack.currentIndex() == 1
    assert window.active_progress_title.text() == "PAUSED"

    window._on_paint_state(generation, SimpleNamespace(value="completed"), "done")
    assert window.plan_progress_stack.currentIndex() == 0
    assert not window.progress_frame.isHidden()


def test_progress_updates_fill_the_large_readout(window: MainWindow) -> None:
    from types import SimpleNamespace

    progress = SimpleNamespace(
        state=SimpleNamespace(value="running"),
        message="Painting",
        percent=42.5,
        color_index=2,
        total_colors=5,
        completed_strokes=425,
        total_strokes=1000,
        estimated_remaining_seconds=95.0,
        elapsed_seconds=70.0,
    )
    window._on_paint_progress(window._paint_generation, progress)

    assert window.active_percent_label.text() == "42%"
    assert "remaining" in window.active_remaining_label.text()
    assert "1m 35s" in window.active_remaining_label.text()
    assert "425" in window.active_detail_label.text()
    assert "elapsed" in window.active_detail_label.text()
    assert window.active_paint_progress.value() == 425


def test_calibration_has_its_own_progress_bar_and_eta(window: MainWindow) -> None:
    from types import SimpleNamespace

    generation = window._paint_generation
    window._paint_job_snapshot = SimpleNamespace(
        profile=SimpleNamespace(metadata={}),
        settings={"painting": {"reuse_calibration": False}},
    )
    calibrating = PaintProgress(
        PainterState.RUNNING,
        0,
        1,
        0,
        1,
        0,
        1,
        0.0,
        9.0,
        None,
        "Measuring brush",
        "calibrate",
    )

    window._calibration_started_elapsed = 0.0
    window._on_paint_progress(generation, calibrating)

    assert window.active_phase_progress.currentWidget() is window.active_calibration_progress
    assert 450 <= window.active_calibration_progress.value() <= 550
    assert window.active_progress_title.text() == "CALIBRATING BRUSH"
    assert "remaining" in window.active_remaining_label.text()

    painting = replace(calibrating, phase="paint", percent=10.0)
    window._on_paint_progress(generation, painting)
    assert window.active_phase_progress.currentWidget() is window.active_paint_progress


def test_start_hotkey_toggles_a_running_job_to_paused(window: MainWindow) -> None:
    from types import SimpleNamespace

    paused: list[str] = []
    window._painter = SimpleNamespace(
        state=SimpleNamespace(value="running"),
        pause=lambda reason: paused.append(reason),
    )

    window._hotkey_toggle_immediate()

    assert paused == ["global start/pause hotkey"]


def test_plan_recalculation_shows_pending_feedback(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    source_path = tmp_path / "spin.png"
    Image.new("RGB", (32, 16), (10, 120, 60)).save(source_path)

    window.load_image(source_path)
    assert window.processing_spinner.is_spinning
    assert window.analysis_time.value_label.text() == "…"
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    assert not window.processing_spinner.is_spinning
    assert window.analysis_time.value_label.text() != "…"

    # Changing a setting that invalidates the plan restarts the feedback.  The
    # numbers read as pending at once; the spinner waits for the settle delay
    # to hand the work to a worker, so a run of quick changes spins nothing.
    window._set_combo_data(window.paint_mode_combo, "fast")
    assert window._plan_pending
    assert not window.processing_spinner.is_spinning
    assert window.analysis_time.value_label.text() == "…"
    window._process_timer.stop()
    window._start_processing()
    assert window.processing_spinner.is_spinning
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    assert not window.processing_spinner.is_spinning
    assert window.analysis_time.value_label.text() != "…"


def test_ready_preview_is_shown_while_its_plan_is_still_building(window: MainWindow) -> None:
    preview = Image.new("RGB", (12, 6), (30, 140, 90))
    window._process_serial = 17

    window._on_processing_preview_ready(
        main_window_module._ProcessPreview(17, preview)
    )

    assert "preview ready" in window.processing_label.text().lower()
    assert not window.paint_preview._source.isNull()


def test_plan_builder_reports_its_current_background_stage(window: MainWindow) -> None:
    window._process_serial = 18

    window._on_processing_stage(18, "Building paint strokes…")

    assert window.processing_label.text() == "Building paint strokes…"


def test_the_estimate_is_one_figure_that_learns_the_checks_and_touch_up(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """The checks and the touch-up are in the number, not tacked on as
    words; what each costs is in the tooltip; and a finished run refines
    them and refreshes the figure."""

    from types import SimpleNamespace

    from app.paint_timing import LearnedTiming, PhaseTiming, TouchUpTiming

    _load_small_plan(window, tmp_path, qtbot)
    plan = window._plan
    assert plan is not None
    window.confirm_strokes_check.setChecked(True)
    window.verify_passes_spin.setValue(1)
    window._refresh_statistics()
    shown = window.analysis_time.value_label.text()
    assert "+" not in shown and "check" not in shown and "touch" not in shown
    tip = window.analysis_time.toolTip()
    assert "Color checks" in tip and "Touch-up" in tip
    assert "guess" in tip  # nothing measured yet
    estimate = window._estimate(plan)
    assert estimate.checks > 0 and estimate.touch_up > 0
    assert estimate.total == pytest.approx(
        estimate.paint + estimate.checks + estimate.touch_up + estimate.calibration + estimate.countdown
    )
    # Turned off, neither is priced.
    window.confirm_strokes_check.setChecked(False)
    window.verify_passes_spin.setValue(0)
    without = window._estimate(plan)
    assert without.checks == 0 and without.touch_up == 0
    assert without.total < estimate.total
    window.confirm_strokes_check.setChecked(True)
    window.verify_passes_spin.setValue(1)


    # A run that painted 600 s of artwork, checked 20 colors at a second
    # a capture with a minute of repainting, and touched up in 90 s.
    before = window._learned_timing
    assert before.check_samples == 0 and before.touch_up_samples == 0

    class _Painter:
        input = SimpleNamespace(emits_real_input=True)
        paint_phase_timing = PhaseTiming(
            predicted_seconds=600.0,
            actual_seconds=680.0,
            strokes=5000,
            checking_seconds=80.0,
            colors_checked=20,
            check_capture_seconds=20.0,
        )
        touch_up_timing = TouchUpTiming(seconds=90.0, passes=2)

    window._painter = _Painter()  # type: ignore[assignment]
    window._learn_timing()
    window._painter = None
    learned = window._learned_timing
    assert learned.check_samples == 1 and learned.touch_up_samples == 1
    assert learned.touch_up_fraction > LearnedTiming().touch_up_fraction
    assert learned.check_capture_seconds > LearnedTiming().check_capture_seconds
    # Written down for the next session ...
    saved = LearnedTiming.load(window._timing_path())
    assert saved.touch_up_samples == 1
    assert saved.history[-1]["touch_up_seconds"] == 90.0
    assert saved.history[-1]["check_repaint_seconds"] == 60.0
    # ... and already in the figure on the screen.
    assert "from 1 run" in window.analysis_time.toolTip()
    assert window._estimate(plan).touch_up > estimate.touch_up


def test_clicking_the_estimate_shows_its_breakdown(
    window: MainWindow,
    tmp_path: Path,
    qtbot,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _load_small_plan(window, tmp_path, qtbot)
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, text: shown.append((title, text)),
    )

    qtbot.mouseClick(window.analysis_time, Qt.MouseButton.LeftButton)

    assert shown
    assert shown[0][0] == "Estimated time breakdown"
    assert "Estimated total:" in shown[0][1]
    assert "Painting:" in shown[0][1]


def test_typing_a_caption_waits_for_a_pause_before_replanning(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """Every keystroke invalidates the plan, so none of them may announce one.

    Recalculating between characters throws the work away on the next one and
    leaves the busy overlay standing for as long as the sentence takes to
    write, which is what made typing feel like the application was struggling.
    """

    source_path = tmp_path / "typed.png"
    Image.new("RGB", (32, 16), (30, 30, 30)).save(source_path)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)

    for index in range(1, len("HELLO") + 1):
        window.text_edit.setText("HELLO"[:index])
        assert window._plan_pending
        # Nothing on screen claims to be working, and the settle delay is
        # pushed back out to its full length by every further character.
        assert not window.plan_busy.is_pending
        assert not window.processing_spinner.is_spinning
        assert window._process_timer.remainingTime() > PLAN_SETTLE_MS

    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    assert not window._plan_pending
    assert window._text_layers[0].text == "HELLO"


def test_edits_on_the_source_tab_defer_replanning_until_the_preview_is_shown(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """A recalculation due mid-edit would drop the busy overlay on the editor.

    So while the Source tab is in front it is held instead of run, however
    long the editing takes; flipping to the Rust preview - the only place the
    result is visible - runs the one recalculation that matters.
    """

    source_path = tmp_path / "deferred.png"
    Image.new("RGB", (32, 16), (30, 30, 30)).save(source_path)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    # The first plan of an import fronts the preview itself; editing means
    # going back to the Source tab.
    assert window.preview_tabs.currentIndex() == 1
    window.preview_tabs.setCurrentIndex(0)

    window.text_edit.setText("HELLO")
    window._process_timer.stop()
    window._start_processing()
    # The settle delay ran out mid-edit, and still nothing recalculates: the
    # work is held, the numbers read as pending, no overlay goes up.
    assert window._plan_deferred
    assert window._plan_pending
    assert window._plan is None
    assert not window.processing_spinner.is_spinning

    window.preview_tabs.setCurrentIndex(1)
    assert not window._plan_deferred
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    assert window._text_layers[0].text == "HELLO"


def test_the_rust_preview_can_be_saved_as_an_image(
    window: MainWindow, tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The promise has to leave the app to be held against the result.

    It is written at one pixel per logical cell - the grid the plan is
    expressed in - and unpainted cells stay empty rather than picking up the
    checkerboard the preview draws them over, so a capture scaled to match can
    simply be subtracted from it.
    """

    source_path = tmp_path / "exported.png"
    image = Image.new("RGBA", (32, 16), (0, 0, 0, 0))
    for x in range(16):
        for y in range(16):
            image.putpixel((x, y), (220, 40, 60, 255))
    image.save(source_path)
    window.quality_combo.setCurrentText("Custom")
    window.logical_width_spin.setValue(32)
    window.logical_height_spin.setValue(16)

    assert not window.export_preview_button.isEnabled()
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._processed is not None, timeout=5000)
    assert window.export_preview_button.isEnabled()

    destination = tmp_path / "out" / "sign.png"
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_a, **_k: (str(destination), ""),
    )
    window.export_preview_button.click()

    assert destination.exists()
    with Image.open(destination) as saved:
        saved.load()
        assert saved.size == window._processed.image.size
        assert saved.mode == "RGBA"
        # Painted where the artwork was, and nothing at all where it was not.
        assert saved.getpixel((0, 0))[3] == 255
        assert saved.getpixel((31, 0))[3] == 0
    assert window._settings["ui"]["last_preview_export_directory"] == str(
        destination.parent
    )


def test_saving_the_preview_to_a_flat_format_keeps_the_checker(
    window: MainWindow, tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A format with no alpha has to put something under the bare cells."""

    source_path = tmp_path / "flat.png"
    Image.new("RGB", (32, 16), (60, 160, 90)).save(source_path)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._processed is not None, timeout=5000)

    destination = tmp_path / "sign.bmp"
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        lambda *_a, **_k: (str(destination), ""),
    )
    window.export_preview_button.click()

    with Image.open(destination) as saved:
        saved.load()
        assert saved.mode == "RGB"
        assert saved.size == window._processed.image.size


def _drag_view(view, start: QPointF, end: QPointF) -> None:
    """Press, move and release on a view's viewport, in scene coordinates."""

    for event_type, position in (
        (QEvent.Type.MouseButtonPress, start),
        (QEvent.Type.MouseMove, end),
        (QEvent.Type.MouseButtonRelease, end),
    ):
        local = QPointF(view.mapFromScene(position))
        event = QMouseEvent(
            event_type,
            local,
            local,
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        if event_type == QEvent.Type.MouseButtonPress:
            view.mousePressEvent(event)
        elif event_type == QEvent.Type.MouseMove:
            view.mouseMoveEvent(event)
        else:
            view.mouseReleaseEvent(event)


def test_dragging_the_source_image_reframes_the_fill_crop(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """Fill keeps a band of the image, and the drag says which band.

    The five named anchors cannot express "a little above centre", which is
    almost always where the subject of a photo is, so a drag writes its own
    centring, marks the alignment Custom, and moves the dashed frame with the
    pointer rather than waiting for the plan to catch up.
    """

    source_path = tmp_path / "tall.png"
    Image.new("RGB", (60, 60), (90, 60, 140)).save(source_path)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)

    window._set_combo_data(window.scale_mode_combo, ScaleMode.FILL.value)
    preview = window.original_preview
    assert preview.can_pan_crop()
    centred = preview.canvas_rect().top()

    start = preview.canvas_rect().center()
    _drag_view(preview, start, start - QPointF(0.0, 1000.0))

    assert window.crop_alignment_combo.currentData() == "custom"
    assert window._crop_focus == (0.5, 0.0)
    assert preview.canvas_rect().top() == pytest.approx(0.0, abs=1e-6)
    assert centred > 0.0
    assert window._processing_options().crop_focus == (0.5, 0.0)
    assert window._settings_document()["image"]["crop_focus"] == [0.5, 0.0]

    # Picking a named anchor again is how the hand-framed crop is given back.
    window._set_combo_data(window.crop_alignment_combo, "center")
    assert window._crop_focus is None
    assert preview.canvas_rect().top() == pytest.approx(centred)
    assert window._settings_document()["image"]["crop_focus"] is None


def test_only_a_fill_crop_with_room_to_move_takes_the_drag(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """Fit has nothing to reframe, so the drag stays a rubber band there."""

    source_path = tmp_path / "square.png"
    Image.new("RGB", (40, 40), (30, 120, 90)).save(source_path)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)

    window._set_combo_data(window.scale_mode_combo, ScaleMode.FIT.value)
    assert not window.original_preview.can_pan_crop()

    window._set_combo_data(window.scale_mode_combo, ScaleMode.STRETCH.value)
    assert not window.original_preview.can_pan_crop()

    window._set_combo_data(window.scale_mode_combo, ScaleMode.FILL.value)
    assert window.original_preview.can_pan_crop()


def test_a_hand_framed_crop_survives_a_restart(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    window._set_combo_data(window.scale_mode_combo, ScaleMode.FILL.value)
    window._on_crop_focus_dragged(0.25, 0.75)
    saved = window._settings_document()
    assert saved["image"]["crop_focus"] == [0.25, 0.75]

    window._apply_settings(saved)
    assert window._crop_focus == (0.25, 0.75)
    assert window.crop_alignment_combo.currentData() == "custom"


def test_editing_the_rust_preview_points_at_the_source_tab(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """A click on the read-only preview is answered, not swallowed.

    Both tabs show the same artwork, so trying to drag text on the wrong one
    is a reasonable mistake; it earns a sentence and a one-click way over to
    the tab that does take the edit, rather than a dialog or silence.
    """

    source_path = tmp_path / "flat.png"
    Image.new("RGB", (32, 16), (200, 120, 40)).save(source_path)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)

    window.preview_tabs.setCurrentIndex(1)
    # isHidden rather than isVisible: the window itself is never shown here,
    # and a child of a hidden parent is never visible however it was asked.
    assert window.preview_notice.isHidden()

    window.paint_preview.mousePressEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonPress,
            QPointF(40.0, 40.0),
            QPointF(40.0, 40.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert not window.preview_notice.isHidden()
    assert "Source tab" in window.preview_notice._message.text()

    window.preview_notice._trigger()
    assert window.preview_tabs.currentIndex() == 0
    assert window.preview_notice.isHidden()

    # Insisting with a double-click needs no notice to read: it just goes.
    window.preview_tabs.setCurrentIndex(1)
    window.paint_preview.mouseDoubleClickEvent(
        QMouseEvent(
            QEvent.Type.MouseButtonDblClick,
            QPointF(40.0, 40.0),
            QPointF(40.0, 40.0),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
    )
    assert window.preview_tabs.currentIndex() == 0


def test_a_plan_already_computed_comes_straight_back(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """Returning to a preset already planned costs nothing to show again.

    A Max-quality plan can take a minute, and stepping away to compare and
    back again used to pay that minute twice.  The finished plan is kept, so
    the second visit is the same object, immediately, with no spinner.
    """

    source_path = tmp_path / "cached.png"
    Image.new("RGB", (48, 24), (40, 160, 210)).save(source_path)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    original = window._plan

    window.quality_combo.setCurrentText("Very Fast")
    qtbot.waitUntil(lambda: window._plan not in (None, original), timeout=5000)
    fast = window._plan

    window.quality_combo.setCurrentText(MAX_QUALITY_PRESET)
    # No wait: a cached plan is applied inside the settings change itself.
    assert window._plan is original
    assert not window.processing_spinner.is_spinning
    assert not window.plan_busy.is_pending

    window.quality_combo.setCurrentText("Very Fast")
    assert window._plan is fast


def test_a_plan_overtaken_before_it_landed_is_still_filed_right(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """A result that arrived too late to show is still a result worth keeping.

    It also must not be filed under whatever was asked for last: switching
    away mid-recalculation and back again has to bring back the plan for the
    settings on screen, not the one the abandoned worker was computing.
    """

    source_path = tmp_path / "overtaken.png"
    image = Image.new("RGB", (64, 32), (250, 250, 250))
    for x in range(20):
        image.putpixel((x, 0), (251, 249, 250))
    image.save(source_path)
    window._set_combo_data(window.paint_mode_combo, "exact")
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    exact = window._plan

    # Ask for another mode and start planning it, then go back before it can
    # land, so its result arrives with the cached exact plan already on screen.
    window._set_combo_data(window.paint_mode_combo, "fast")
    window._start_processing()
    window._set_combo_data(window.paint_mode_combo, "exact")
    assert window._plan is exact

    qtbot.waitUntil(lambda: len(window._plan_cache) == 2, timeout=5000)
    assert window._plan is exact

    window._set_combo_data(window.paint_mode_combo, "fast")
    assert not window.processing_spinner.is_spinning
    assert window._plan is not None
    assert window._plan is not exact


def test_the_plan_cache_belongs_to_one_image_and_one_budget(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    first = tmp_path / "first.png"
    Image.new("RGB", (32, 16), (10, 90, 40)).save(first)
    window.load_image(first)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    assert len(window._plan_cache) == 1

    second = tmp_path / "second.png"
    Image.new("RGB", (32, 16), (200, 30, 30)).save(second)
    window.load_image(second)
    # The plans held belonged to the image that was replaced.
    assert not window._plan_cache
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)

    # Nothing accumulates without bound: the oldest plans are let go first.
    only = window._plan_cache[next(iter(window._plan_cache))]
    for index in range(main_window_module.PLAN_CACHE_ENTRIES + 3):
        window._remember_plan((index,), only)
    assert len(window._plan_cache) == main_window_module.PLAN_CACHE_ENTRIES

    # And a single plan too large for the whole budget is still kept, because
    # the plan on screen is the one in the cache.
    huge = replace(only, plan=replace(only.plan, width=4096, height=4096))
    window._remember_plan(("huge",), huge)
    assert window._plan_cache_cost(huge) > main_window_module.PLAN_CACHE_BYTES
    assert list(window._plan_cache) == [("huge",)]


def test_recalculating_covers_the_preview_it_is_recalculating(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """The small spinner is easy to miss; the overlay is not.

    It also names the settings being planned for, because the question a slow
    recalculation raises is "what is it doing?", not just "is it busy?".
    """

    source_path = tmp_path / "busy.png"
    Image.new("RGB", (32, 16), (120, 60, 200)).save(source_path)
    window.load_image(source_path)
    assert window.plan_busy.is_pending
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    assert not window.plan_busy.is_pending

    # The overlay belongs to work that is really running, so it goes up when
    # the settle delay expires rather than the moment a control moves.
    window._set_combo_data(window.paint_mode_combo, "fast")
    assert not window.plan_busy.is_pending
    window._process_timer.stop()
    window._start_processing()
    assert window.plan_busy.is_pending
    summary = window._plan_summary()
    assert "Max quality" in summary
    assert "Fast optimization" in summary
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    assert not window.plan_busy.is_pending


def test_paused_jobs_still_show_the_calibration_outlines(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A pause is exactly when the boxes are worth looking at.

    Nothing is being clicked while a job is held, the outlines take no input,
    and what the user is usually checking is whether Rust still lines up with
    the calibration before letting the job carry on.
    """

    from types import SimpleNamespace

    class _FakeOverlay:
        def __init__(self) -> None:
            self.visible = False
            self.entries: list = []

        def set_rectangles(self, entries) -> None:
            self.entries = list(entries)

        def set_status(self, status) -> None:
            self.status = status

        def show_overlay(self) -> None:
            self.visible = True

        def hide(self) -> None:
            self.visible = False

        def isVisible(self) -> bool:  # noqa: N802 - mirrors the QWidget API
            return self.visible

    monkeypatch.setattr(main_window_module, "CalibrationPreviewOverlay", _FakeOverlay)
    window.show_calibration_check.setChecked(True)
    window.show_status_check.setChecked(False)
    window._current_profile.canvas = ScreenRect(0, 0, 200, 100)
    window._update_calibration_overlay()
    assert window._calibration_preview.isVisible()

    window._painter = SimpleNamespace(
        state=PainterState.RUNNING, is_active=True, is_alive=True
    )
    window._update_calibration_overlay()
    assert not window._calibration_preview.isVisible()

    window._painter.state = PainterState.PAUSED
    window._update_calibration_overlay()
    assert window._calibration_preview.isVisible()

    window._painter.state = PainterState.RUNNING
    window._update_calibration_overlay()
    assert not window._calibration_preview.isVisible()


def test_the_status_label_follows_the_job_in_the_corner(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With Show status on, the job's state is written on the sign's monitor.

    It says IDLE with no job at all - the switch is what puts it up, not a
    running job - PAINTING while strokes go down, with the boxes kept off
    the screen as they are for any running job, PAUSED through a pause, and
    the job's last word for a moment after it ends before falling back to
    IDLE.  Both switches are on by default and both are saved.
    """

    from types import SimpleNamespace

    class _FakeOverlay:
        def __init__(self) -> None:
            self.visible = False
            self.entries: list = []
            self.status = None

        def set_rectangles(self, entries) -> None:
            self.entries = list(entries)

        def set_status(self, status) -> None:
            self.status = status

        def show_overlay(self) -> None:
            self.visible = True

        def hide(self) -> None:
            self.visible = False

        def isVisible(self) -> bool:  # noqa: N802 - mirrors the QWidget API
            return self.visible

    monkeypatch.setattr(main_window_module, "CalibrationPreviewOverlay", _FakeOverlay)
    assert window.show_calibration_check.isChecked()
    assert window.show_status_check.isChecked()
    canvas = ScreenRect(0, 0, 200, 100)
    window._current_profile.canvas = canvas
    window._update_calibration_overlay()
    overlay = window._calibration_preview
    assert overlay.isVisible()
    assert overlay.status == ("IDLE", canvas), "the label is up with no job running"

    window._painter = SimpleNamespace(
        state=PainterState.RUNNING, is_active=True, is_alive=True
    )
    window._update_calibration_overlay()
    assert overlay.isVisible()
    assert overlay.status == ("PAINTING", canvas)
    assert overlay.entries == []

    window._painter.state = PainterState.PAUSED
    window._update_calibration_overlay()
    assert overlay.status == ("PAUSED", canvas)
    assert overlay.entries, "the boxes come back while paused"

    # Stopped: the word stays up for a moment, then the sign is left alone.
    window._painter.state = PainterState.ABORTED
    window._painter.is_active = False
    window._status_overlay_linger.start()
    window._update_calibration_overlay()
    assert overlay.status == ("ABORTED", canvas)
    window._status_overlay_linger.stop()
    window._update_calibration_overlay()
    assert overlay.status == ("IDLE", canvas), "and back to IDLE once it is read"
    assert overlay.isVisible(), "the boxes are back once the job is over"
    window._painter = None

    window.show_status_check.setChecked(False)
    window.show_calibration_check.setChecked(False)
    assert not overlay.isVisible()
    window._save_settings()
    ui = window._settings_document()["ui"]
    assert ui["show_status_overlay"] is False
    assert ui["show_calibration_overlay"] is False


def test_sharpen_choice_reaches_the_plan_and_survives_a_restart(
    window: MainWindow,
) -> None:
    assert window.sharpen_combo.currentData() == "light"
    assert window._processing_options().sharpen.value == "light"

    window._set_combo_data(window.sharpen_combo, "strong")
    assert window._processing_options().sharpen.value == "strong"
    assert window._settings_document()["image"]["sharpen"] == "strong"

    settings = window._settings_document()
    settings["image"]["sharpen"] = "off"
    window._apply_settings(settings)
    assert window.sharpen_combo.currentData() == "off"
    assert window._processing_options().sharpen.value == "off"


def test_rust_preview_scaling_is_smooth_by_default_and_switchable(
    window: MainWindow,
) -> None:
    """Filtered scaling is the honest guess at the sign; blocky is opt-in.

    The switch has to reach the label itself, not just a setting, and has to
    come back the same way the window was left.
    """

    assert window.smooth_preview_check.isChecked()
    assert window.paint_preview.is_smooth()
    assert window._settings_document()["ui"]["smooth_rust_preview"] is True

    window.smooth_preview_check.setChecked(False)
    assert not window.paint_preview.is_smooth()
    assert window._settings_document()["ui"]["smooth_rust_preview"] is False

    settings = window._settings_document()
    settings["ui"]["smooth_rust_preview"] = True
    window._apply_settings(settings)
    assert window.smooth_preview_check.isChecked()
    assert window.paint_preview.is_smooth()


def test_the_timelapse_speed_slider_says_what_it_costs(
    window: MainWindow,
) -> None:
    """A slider asks "how fast", not "how many frames per second".

    The readout keeps the frame rate, because that is what the saved file is
    written at, and adds the thing the number actually buys: how much of the
    paint job one second of video covers.
    """

    window.timelapse_interval_spin.setValue(10)
    window.timelapse_speed_slider.setValue(15)
    assert window.timelapse_speed_label.text() == "15 fps  •  2m 30s per second"

    window.timelapse_speed_slider.setValue(30)
    assert window.timelapse_speed_label.text() == "30 fps  •  5m 00s per second"

    window.timelapse_interval_spin.setValue(5)
    assert window.timelapse_speed_label.text() == "30 fps  •  2m 30s per second"


def test_rust_on_another_monitor_offers_and_applies_a_move(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.screen as screen_module
    from app.screen import ForegroundWindowInfo

    profile = window._current_profile
    assert profile is not None
    profile.canvas = ScreenRect(100, 100, 800, 400)
    profile.color_box = ScreenRect(1000, 100, 200, 200)
    profile.hue_bar = ScreenRect(1250, 100, 20, 200)
    profile.metadata["ui_reference"] = {"path": "stale.png", "rect": {"left": 0, "top": 0, "width": 1, "height": 1}}
    window._current_profile = window._profile_store.save(profile)

    calibrated_monitor = ScreenRect(0, 0, 1920, 1080)
    rust_monitor = ScreenRect(1920, 0, 1920, 1080)
    monkeypatch.setattr(
        screen_module,
        "find_window_matching",
        lambda **_kwargs: ForegroundWindowInfo(hwnd=42, title="Rust", process_id=7),
    )
    monkeypatch.setattr(
        screen_module, "window_monitor_rect", lambda _hwnd: rust_monitor
    )
    monkeypatch.setattr(
        screen_module, "monitor_rect_at", lambda _x, _y: calibrated_monitor
    )

    window._check_rust_monitor()
    assert window.rust_monitor_label.isVisible() or not window.rust_monitor_label.isHidden()
    assert not window.move_to_rust_button.isHidden()

    window._move_calibration_to_rust_monitor()

    moved = window._current_profile
    assert moved.canvas == ScreenRect(2020, 100, 800, 400)
    assert moved.color_box == ScreenRect(2920, 100, 200, 200)
    assert moved.hue_bar == ScreenRect(3170, 100, 20, 200)
    assert "ui_reference" not in moved.metadata
    assert window.move_to_rust_button.isHidden()

    # Once Rust and the boxes share a monitor, the prompt goes away.
    monkeypatch.setattr(
        screen_module, "monitor_rect_at", lambda _x, _y: rust_monitor
    )
    window._check_rust_monitor()
    assert window.rust_monitor_label.isHidden()


def test_timelapse_records_only_real_running_jobs(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, qtbot
) -> None:
    from types import SimpleNamespace

    import app.timelapse as timelapse_module

    frames: list[object] = []
    monkeypatch.setattr(
        timelapse_module,
        "capture_region",
        lambda region: frames.append(region)
        or Image.new("RGB", (region.width, region.height)),
    )

    window.timelapse_check.setChecked(True)
    window.timelapse_interval_spin.setValue(1)
    profile = window._current_profile
    profile.canvas = ScreenRect(0, 0, 64, 32)

    # A dry run never records.
    window._painter = SimpleNamespace(
        input=SimpleNamespace(emits_real_input=False),
        state=SimpleNamespace(value="running"),
    )
    window._maybe_start_timelapse()
    assert window._timelapse_recorder is None
    assert not window._timelapse_timer.isActive()

    # A real job records from the moment it runs.
    window._painter = SimpleNamespace(
        input=SimpleNamespace(emits_real_input=True),
        state=SimpleNamespace(value="running"),
    )
    window._maybe_start_timelapse()
    recorder = window._timelapse_recorder
    assert recorder is not None
    assert window._timelapse_timer.isActive()
    qtbot.waitUntil(lambda: recorder.frame_count >= 1, timeout=3000)

    # Finishing stops the timer and detaches the recorder.
    window._finish_timelapse(final=True)
    assert window._timelapse_recorder is None
    assert not window._timelapse_timer.isActive()
    qtbot.waitUntil(lambda: recorder.frame_count >= 2, timeout=3000)


def test_recording_waits_for_the_artwork_and_does_not_restart_at_the_end(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A run opens with brush-calibration strokes that get wiped off again.

    Recording those would put the throwaway probes at the head of every video,
    so the recorder waits for the painting phase - and the job's *final*
    progress update is a painting one too, which must not open a fresh
    recording the moment the finished one closes.
    """

    from types import SimpleNamespace

    import app.timelapse as timelapse_module

    monkeypatch.setattr(
        timelapse_module,
        "capture_region",
        lambda region: Image.new("RGB", (region.width, region.height)),
    )
    window.timelapse_check.setChecked(True)
    window._current_profile.canvas = ScreenRect(0, 0, 64, 32)
    window._painter = SimpleNamespace(
        input=SimpleNamespace(emits_real_input=True),
        state=PainterState.RUNNING,
    )

    def progress(phase: str, state: PainterState) -> PaintProgress:
        return PaintProgress(
            state, 0, 1, 0, 1, 0, 1, 0.0, 0.0, None, "", phase
        )

    window._on_paint_progress(
        window._paint_generation, progress("calibrate", PainterState.RUNNING)
    )
    assert window._timelapse_recorder is None

    window._on_paint_progress(
        window._paint_generation, progress("paint", PainterState.RUNNING)
    )
    assert window._timelapse_recorder is not None

    window._finish_timelapse(final=False)
    assert window._timelapse_recorder is None

    # The completion update still reports the painting phase.
    window._painter.state = PainterState.COMPLETED
    window._on_paint_progress(
        window._paint_generation, progress("paint", PainterState.COMPLETED)
    )
    assert window._timelapse_recorder is None
    window._painter = None


def test_timelapse_settings_persist(window: MainWindow) -> None:
    window.timelapse_check.setChecked(True)
    window.timelapse_interval_spin.setValue(25)
    window._set_combo_data(window.timelapse_sort_combo, "largest")

    document = window._settings_document()

    assert document["timelapse"]["enabled"] is True
    assert document["timelapse"]["interval_seconds"] == 25
    assert document["timelapse"]["sort_order"] == "largest"


def test_timelapse_storage_total_and_size_sorting(window: MainWindow) -> None:
    root = window._timelapse_root()
    small = root / "20260818-100000"
    large = root / "20260818-090000"
    small.mkdir(parents=True)
    large.mkdir(parents=True)
    (small / "frame_00001.png").write_bytes(b"s" * 100)
    (large / "frame_00001.png").write_bytes(b"l" * 1500)

    window._set_combo_data(window.timelapse_sort_combo, "largest")
    window._refresh_timelapse_sessions()
    assert window._selected_session_paths() == []
    assert window.timelapse_sessions.item(0).data(Qt.ItemDataRole.UserRole) == str(large)
    assert window.timelapse_total_storage_label.text() == "Total storage: 1.6 KB"

    window._set_combo_data(window.timelapse_sort_combo, "smallest")
    assert window.timelapse_sessions.item(0).data(Qt.ItemDataRole.UserRole) == str(small)


def test_timelapse_page_lists_recordings_and_deletes_a_selected_one(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    from PySide6.QtWidgets import QMessageBox

    root = window._timelapse_root()
    for name, frames in (("20260818-100000", 3), ("20260818-090000", 1)):
        session = root / name
        session.mkdir(parents=True)
        for index in range(1, frames + 1):
            Image.new("RGB", (8, 4)).save(session / f"frame_{index:05d}.png")

    window.timelapse_nav_button.click()

    assert window.page_stack.currentIndex() == 1
    labels = [
        window.timelapse_sessions.item(row).text()
        for row in range(window.timelapse_sessions.count())
    ]
    # Newest first, each with its own frame count.
    assert labels[0].startswith("20260818-100000")
    assert "3 frames" in labels[0]
    assert labels[1].startswith("20260818-090000")
    assert "1 frame" in labels[1] and "1 frames" not in labels[1]

    # Nothing is selected yet, so the per-session actions stay disabled.
    assert not window.open_session_button.isEnabled()
    assert not window.delete_session_button.isEnabled()

    window.timelapse_sessions.setCurrentRow(1)
    assert window.delete_session_button.isEnabled()
    assert window._selected_session_path() == root / "20260818-090000"

    monkeypatch.setattr(
        QMessageBox, "question", lambda *_a, **_k: QMessageBox.StandardButton.Yes
    )
    window._delete_selected_sessions()

    assert not (root / "20260818-090000").exists()
    assert (root / "20260818-100000").exists()
    assert window.timelapse_sessions.count() == 1


def test_timelapse_page_reports_recording_status(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    from types import SimpleNamespace

    import app.timelapse as timelapse_module

    monkeypatch.setattr(
        timelapse_module,
        "capture_region",
        lambda region: Image.new("RGB", (region.width, region.height)),
    )
    assert window.timelapse_status_badge.text() == "Not recording"

    window.timelapse_check.setChecked(True)
    window._current_profile.canvas = ScreenRect(0, 0, 64, 32)
    window._painter = SimpleNamespace(
        input=SimpleNamespace(emits_real_input=True),
        state=SimpleNamespace(value="running"),
    )
    window._maybe_start_timelapse()

    assert window.timelapse_status_badge.text().startswith("Recording")

    # A paused job keeps the session but stops adding frames.
    window._painter.state = SimpleNamespace(value="paused")
    window._capture_timelapse_frame()
    assert window.timelapse_status_badge.text().startswith("Paused")

    window._finish_timelapse(final=False)
    assert window.timelapse_status_badge.text() == "Not recording"


def test_disabled_start_button_names_the_blocker(window: MainWindow) -> None:
    # A greyed-out START with no explanation reads as a broken app.  Both common
    # blockers are user-fixable, so each has to say so.
    window.dry_run_check.setChecked(False)

    assert not window.start_button.isEnabled()
    assert "Load an image" in window.start_button.toolTip()

    window._plan = object()
    window._current_profile = Profile(id="p", name="Sign", canvas=Rect(0, 0, 100, 100))
    window._update_start_availability()

    tooltip = window.start_button.toolTip()
    assert not window.start_button.isEnabled()
    assert "color box" in tooltip and "hue bar" in tooltip
    assert "canvas" not in tooltip


def test_start_names_the_rectangles_automatic_sizing_still_needs(
    window: MainWindow,
) -> None:
    """Sizing now measures on the sign, so its calibration is a start blocker."""

    window.dry_run_check.setChecked(False)
    window.apply_brush_check.setChecked(True)
    window._plan = PaintPlan(4, 4, (ColorGroup((10, 20, 30), (Stroke(0, 0, 3, 0),), 1),))
    window._current_profile = Profile(
        id="p",
        name="Sign",
        canvas=Rect(0, 0, 100, 100),
        color_box=Rect(120, 0, 40, 40),
        hue_bar=Rect(170, 0, 10, 40),
    )
    window._update_start_availability()

    tooltip = window.start_button.toolTip()
    assert not window.start_button.isEnabled()
    assert "Size value box and clear button" in tooltip

    window._current_profile.brush_size_box = Rect(200, 0, 40, 20)
    window._current_profile.clear_button = Rect(250, 0, 20, 20)
    window._update_start_availability()
    assert "Size value box" not in window.start_button.toolTip()


def _recorded_session(window: MainWindow, name: str, frames: int) -> Path:
    directory = window._timelapse_root() / name
    directory.mkdir(parents=True, exist_ok=True)
    for index in range(1, frames + 1):
        Image.new("RGB", (48, 24), (index * 30 % 256, 70, 150)).save(
            directory / f"frame_{index:05d}.png"
        )
    return directory


def _session_row(window: MainWindow, directory: Path) -> int:
    for row in range(window.timelapse_sessions.count()):
        item = window.timelapse_sessions.item(row)
        if item.data(Qt.ItemDataRole.UserRole) == str(directory):
            return row
    raise AssertionError(f"{directory} was not listed")


def _select_session(window: MainWindow, *directories: Path) -> None:
    """Pick exactly these recordings, the way clicking them would."""

    window._refresh_timelapse_sessions()
    window.timelapse_sessions.clearSelection()
    for directory in directories:
        item = window.timelapse_sessions.item(_session_row(window, directory))
        window.timelapse_sessions.setCurrentItem(item)
        item.setSelected(True)


def test_watching_and_saving_need_a_recording_with_frames(window: MainWindow) -> None:
    empty = window._timelapse_root() / "20260101-000000"
    empty.mkdir(parents=True, exist_ok=True)

    _select_session(window, empty)
    # An empty folder can still be opened and deleted, but there is nothing in
    # it to play or encode.
    assert not window.play_session_button.isEnabled()
    assert not window.export_session_button.isEnabled()
    assert window.delete_session_button.isEnabled()

    recorded = _recorded_session(window, "20260101-000100", 4)
    _select_session(window, recorded)
    assert window.play_session_button.isEnabled()
    assert window.export_session_button.isEnabled()


def test_a_run_of_recordings_is_selected_and_deleted_together(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Clearing out old recordings is a batch job, so it asks once.

    Shift-clicking down a list is how a run of them gets picked everywhere
    else, and Delete is the key that removes what is picked.
    """

    sessions = [
        _recorded_session(window, f"2026010{index}-000000", index)
        for index in range(1, 5)
    ]
    window._refresh_timelapse_sessions()
    # Newest first, so the run from the top down covers the three newest.
    window.timelapse_sessions.setCurrentRow(0)
    window.timelapse_sessions.item(2).setSelected(True)
    window.timelapse_sessions.item(1).setSelected(True)

    assert len(window._selected_session_paths()) == 3
    # Three recordings cannot be watched or encoded at once, but they can
    # certainly be deleted at once.
    assert not window.play_session_button.isEnabled()
    assert not window.export_session_button.isEnabled()
    assert not window.open_session_button.isEnabled()
    assert window.delete_session_button.isEnabled()
    assert "3 recordings selected" in window.timelapse_selection_label.text()
    assert window._selected_session_path() is None

    asked: list[str] = []

    def _confirm(_parent, _title, text, *_args, **_kwargs):
        asked.append(text)
        return main_window_module.QMessageBox.StandardButton.Yes

    monkeypatch.setattr(main_window_module.QMessageBox, "question", _confirm)
    window._delete_selected_sessions()

    assert len(asked) == 1
    assert "3 recordings" in asked[0]
    assert [session.exists() for session in sessions] == [True, False, False, False]
    assert window.timelapse_sessions.count() == 1
    assert window.timelapse_selection_label.text() == ""


def test_the_delete_key_removes_the_selected_recordings(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = _recorded_session(window, "20260101-000000", 2)
    _select_session(window, session)

    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *_a, **_k: main_window_module.QMessageBox.StandardButton.Yes,
    )
    assert window.delete_session_shortcut.key() == QKeySequence.StandardKey.Delete
    window.delete_session_shortcut.activated.emit()

    assert not session.exists()


def test_a_recording_still_being_written_survives_a_batch_delete(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The live session is dropped from the batch, not allowed to fail it."""

    from types import SimpleNamespace

    finished = _recorded_session(window, "20260101-000000", 2)
    live = _recorded_session(window, "20260101-000100", 1)
    window._timelapse_recorder = SimpleNamespace(directory=live)
    _select_session(window, live, finished)

    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        lambda *_a, **_k: main_window_module.QMessageBox.StandardButton.Yes,
    )
    window._delete_selected_sessions()

    assert live.exists()
    assert not finished.exists()


def test_the_player_steps_scrubs_and_stops_at_the_last_frame(
    window: MainWindow, qtbot
) -> None:
    from app.gui.timelapse_player import TimelapsePlayer
    from app.timelapse_export import session_frames

    directory = _recorded_session(window, "20260101-000200", 5)
    player = TimelapsePlayer("20260101-000200", session_frames(directory), frame_rate=30)
    qtbot.addWidget(player)

    assert player.position_slider.maximum() == 4
    assert "1 of 5" in player.counter_label.text()

    player.step(2)
    assert player.position_slider.value() == 2
    assert "3 of 5" in player.counter_label.text()

    # Scrubbing drives the frame, and the frame drives the counter.
    player.position_slider.setValue(4)
    assert "5 of 5" in player.counter_label.text()

    player.play()
    assert player.is_playing
    # Play on the final frame means "watch it again" rather than sit still.
    assert player.position_slider.value() == 0

    player.pause()
    assert not player.is_playing
    player.close()


def test_saving_a_recording_writes_a_video_the_user_chose(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qtbot
) -> None:
    directory = _recorded_session(window, "20260101-000300", 6)
    destination = tmp_path / "sign.avi"
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(destination), "")),
    )
    opened: list[Path] = []
    monkeypatch.setattr(
        MainWindow, "_open_in_file_manager", staticmethod(opened.append)
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        staticmethod(
            lambda *args, **kwargs: main_window_module.QMessageBox.StandardButton.Close
        ),
    )

    _select_session(window, directory)
    window.timelapse_speed_slider.setValue(12)
    window._set_combo_data(window.timelapse_format_combo, "avi")
    window._export_selected_session()

    qtbot.waitUntil(lambda: window._timelapse_export is None, timeout=10000)
    assert destination.is_file()
    assert destination.read_bytes()[:4] == b"RIFF"
    # The frames themselves are never touched by an export.
    assert len(list(directory.glob("frame_*.png"))) == 6
    assert not opened


def test_a_failed_export_says_so_and_leaves_the_button_usable(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qtbot
) -> None:
    directory = _recorded_session(window, "20260101-000400", 3)
    monkeypatch.setattr(
        main_window_module.QFileDialog,
        "getSaveFileName",
        staticmethod(lambda *args, **kwargs: (str(tmp_path / "sign.avi"), "")),
    )

    def explode(*_args, **_kwargs):
        raise RuntimeError("the disk went away")

    monkeypatch.setattr(main_window_module, "export_session", explode)
    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        staticmethod(lambda _parent, _title, message, *a, **k: warnings.append(message)),
    )

    _select_session(window, directory)
    window._export_selected_session()

    qtbot.waitUntil(lambda: bool(warnings), timeout=10000)
    assert "the disk went away" in warnings[0]
    assert window._timelapse_export is None
    assert window.export_session_button.isEnabled()
    assert not window.timelapse_export_progress.isVisible()


def test_the_playback_speed_and_format_survive_a_restart(window: MainWindow) -> None:
    window.timelapse_speed_slider.setValue(24)
    window._set_combo_data(window.timelapse_format_combo, "gif")

    saved = window._settings_document()

    assert saved["timelapse"]["playback_frame_rate"] == 24
    assert saved["timelapse"]["export_format"] == "gif"


def test_a_run_report_records_the_plan_and_survives_an_abort(
    window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, qtbot
) -> None:
    """An aborted run is the one that needs explaining, so it still reports."""

    source_path = tmp_path / "report-source.png"
    Image.new("RGBA", (32, 16), (210, 30, 40, 255)).save(source_path)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)

    from types import SimpleNamespace

    from app import run_report as run_report_module

    # The real ones photograph the desktop; the wiring under test is which
    # files appear and when, not Pillow's ability to grab a screen.
    monkeypatch.setattr(
        run_report_module.RunReport, "record_screen", lambda self: None
    )
    monkeypatch.setattr(
        run_report_module.RunReport,
        "record_canvas",
        lambda self, canvas, name: None,
    )
    window._current_profile.canvas = ScreenRect(0, 0, 64, 32)
    window._painter = SimpleNamespace(
        input=SimpleNamespace(emits_real_input=True),
        state=PainterState.RUNNING,
        measured_brush_size_model=None,
    )

    def progress(phase: str, strokes: int) -> PaintProgress:
        return PaintProgress(
            PainterState.RUNNING, 1, 1, strokes, 4, strokes, 4,
            25.0 * strokes, 0.0, None, "", phase,
        )

    # Calibration strokes are not the artwork, so nothing opens yet.
    window._on_paint_progress(window._paint_generation, progress("calibrate", 0))
    assert window._run_report is None

    window._on_paint_progress(window._paint_generation, progress("paint", 1))
    report = window._run_report
    assert report is not None
    directory = report.directory

    window._painter.state = PainterState.ABORTED
    window._on_paint_state(
        window._paint_generation, SimpleNamespace(value="aborted"), "user"
    )
    assert window._run_report is None
    qtbot.waitUntil(lambda: (directory / "run.json").exists(), timeout=5000)

    document = json.loads((directory / "run.json").read_text())
    assert document["outcome"] == "aborted"
    assert document["outcomeReason"] == "user"
    assert document["plan"]["width"] == window._plan.width
    assert document["plan"]["strokes"] == window._plan.stroke_count
    # The settings and profile of the day, not whatever is on disk later.
    assert document["settings"]["image"]["paint_mode"]
    assert (directory / "plan.png").exists()
    assert (directory / "progress.csv").exists()
    window._painter = None


def test_merge_box_says_it_is_automatic_outside_exact_mode_and_keeps_the_choice(
    window: MainWindow,
) -> None:
    from app.gui.main_window import MERGE_MODE_OPTIMIZER

    # Quality paint mode: the box is disabled and says why, rather than
    # showing a greyed-out choice that reads as merging being off.
    assert window.paint_mode_combo.currentData() == "quality"
    assert not window.merge_combo.isEnabled()
    assert window.merge_combo.currentData() == MERGE_MODE_OPTIMIZER
    assert "optimizer" in window.merge_combo.currentText()
    # The user's real choice is what gets saved and planned with.
    assert window._settings_document()["painting"]["stroke_merge_mode"] == "balanced"
    assert window._current_overpaint_gap() == 6

    window._set_combo_data(window.paint_mode_combo, "exact")
    assert window.merge_combo.isEnabled()
    assert window.merge_combo.currentData() == "balanced"
    window._set_combo_data(window.merge_combo, "maximum")
    assert window._settings_document()["painting"]["stroke_merge_mode"] == "maximum"
    assert window._current_overpaint_gap() is None

    # Leaving Exact hides the choice behind the caption and keeps it.
    window._set_combo_data(window.paint_mode_combo, "fast")
    assert window.merge_combo.currentData() == MERGE_MODE_OPTIMIZER
    assert window._settings_document()["painting"]["stroke_merge_mode"] == "maximum"
    window._set_combo_data(window.paint_mode_combo, "exact")
    assert window.merge_combo.currentData() == "maximum"


def test_time_estimate_prices_the_frame_hold_per_stroke(window: MainWindow) -> None:
    """Fifty dabs are fifty held presses, whatever the speed slider says."""

    from app.models import ColorGroup, PaintPlan, Stroke
    from app.paint_timing import MIN_PRESS_SECONDS

    strokes = tuple(Stroke(x, 0, x, 0) for x in range(0, 100, 2))
    plan = PaintPlan(100, 1, (ColorGroup((0, 0, 0), strokes, len(strokes)),))

    window.stroke_speed_spin.setValue(window.stroke_speed_spin.maximum())
    fast = window._estimate_seconds(plan)
    window.stroke_speed_spin.setValue(window.stroke_speed_spin.minimum())
    slow = window._estimate_seconds(plan)
    # Dabs do not move, so speed changes nothing; each still costs its hold.
    assert fast == pytest.approx(slow)
    assert fast >= plan.stroke_count * MIN_PRESS_SECONDS


def test_speed_presets_sit_on_the_frame_floors_and_old_profiles_still_match(
    window: MainWindow,
) -> None:
    """The holds and settles cannot go under a game frame, so the presets
    start there and a profile saved with the old, lower values - which the
    painter always ran at the floor anyway - is still recognised."""

    from app.gui.main_window import SPEED_FLOORS_MS, SPEED_PRESETS
    from app.paint_timing import TIMING_FLOORS

    # Turbo is the floors everywhere.
    turbo = SPEED_PRESETS["Turbo"]
    for key, floor in SPEED_FLOORS_MS.items():
        if key in turbo:
            assert turbo[key] == floor, key
    # The spinboxes stop at the floors, in the painter's own numbers.
    assert window.dot_duration_spin.minimum() == round(
        TIMING_FLOORS["mouse_down_duration_seconds"] * 1000
    )
    assert window.stroke_delay_spin.minimum() == round(
        TIMING_FLOORS["delay_between_strokes_seconds"] * 1000
    )
    window.dot_duration_spin.setValue(1)
    assert window.dot_duration_spin.value() == window.dot_duration_spin.minimum()

    # A profile saved before the floors: Standard with a 28 ms hold and an
    # 18 ms gap.
    document = window._settings_document()
    document["painting"].update(
        {
            "stroke_speed_pixels_per_second": 700.0,
            "mouse_down_duration_seconds": 0.028,
            "delay_after_hue_seconds": 0.09,
            "delay_after_saturation_value_seconds": 0.09,
            "delay_between_strokes_seconds": 0.018,
            "delay_between_colors_seconds": 0.12,
            "stroke_interpolation_step_pixels": 4.0,
        }
    )
    window._apply_settings(document)
    assert window.speed_preset_combo.currentText() == "Standard"
    # And the old Turbo, every value under its floor, is still Turbo.
    document["painting"].update(
        {
            "stroke_speed_pixels_per_second": 2200.0,
            "mouse_down_duration_seconds": 0.012,
            "delay_after_hue_seconds": 0.045,
            "delay_after_saturation_value_seconds": 0.045,
            "delay_between_strokes_seconds": 0.005,
            "delay_between_colors_seconds": 0.05,
            "stroke_interpolation_step_pixels": 8.0,
        }
    )
    window._apply_settings(document)
    assert window.speed_preset_combo.currentText() == "Turbo"
    assert window.dot_duration_spin.value() == SPEED_FLOORS_MS["dot_ms"]

    window.speed_preset_combo.setCurrentText("Relaxed")
    assert window.dot_duration_spin.value() == SPEED_PRESETS["Relaxed"]["dot_ms"]
    assert window._detect_speed_preset() == "Relaxed"


def test_text_controls_follow_the_source_tab(window: MainWindow) -> None:
    """Text is edited on the Source tab, so its controls live there too.

    On the Rust preview the layers are baked in and untouchable, and a panel
    of controls for them would only invite the read-only notice.
    """

    # isHidden rather than isVisible: the window itself is never shown here.
    assert not window.text_section.isHidden()
    window.preview_tabs.setCurrentIndex(1)
    assert window.text_section.isHidden()
    window.preview_tabs.setCurrentIndex(0)
    assert not window.text_section.isHidden()


def test_a_paused_job_leaves_the_timing_controls_live_and_retunes_on_resume(
    window: MainWindow,
) -> None:
    """A pause is when a hold that looked too short gets lengthened.

    The timing and the guards stay editable through a pause and reach the
    painter when the job resumes; everything that shaped the job - the image,
    the plan, the calibration - stays locked until it is over.
    """

    from types import SimpleNamespace

    class _PausedPainter:
        state = PainterState.PAUSED
        is_active = True
        is_alive = True
        input = SimpleNamespace(emits_real_input=True)

        def __init__(self) -> None:
            self.retuned: list = []
            self.resumed = 0

        def retune(self, settings) -> bool:
            self.retuned.append(settings)
            return True

        def resume(self) -> bool:
            self.resumed += 1
            return True

    painter = _PausedPainter()
    window._painter = painter
    window._update_start_availability()

    assert window.speed_preset_combo.isEnabled()
    assert window.stroke_delay_spin.isEnabled()
    assert window.verify_passes_spin.isEnabled()
    assert not window.quality_combo.isEnabled()
    assert not window.pixel_spacing_spin.isEnabled()
    assert not window.browse_button.isEnabled()
    assert not window.apply_brush_check.isEnabled()
    assert not window.abort_hotkey_combo.isEnabled()

    window.stroke_delay_spin.setValue(250)
    window._start_or_resume()

    assert painter.resumed == 1
    assert len(painter.retuned) == 1
    settings = painter.retuned[0]
    assert settings.delay_between_strokes_seconds == pytest.approx(0.25)
    # The worker's own countdown never runs again after a resume.
    assert settings.countdown_seconds == 0

    # Once the job is over, the ordinary lock lifts everything together.
    window._painter = None
    window._update_start_availability()
    assert window.quality_combo.isEnabled()
    assert window.stroke_delay_spin.isEnabled()


def test_anti_afk_needs_the_save_button_and_reaches_the_painter(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """The break leaves the painting UI through Save, so Save must be known.

    With the switch on, the Save button turns from optional to needed and an
    uncalibrated one blocks Start with a reason; the switch and its interval
    are saved, and reach the painter in seconds.
    """

    assert window.anti_afk_check.isChecked()
    window._current_profile.save_button = None
    window.anti_afk_check.setChecked(False)
    window._refresh_profile_ui()
    assert window.save_button_status._value.text() == "Optional"

    window.anti_afk_check.setChecked(True)
    window.anti_afk_interval_spin.setValue(12)
    assert window.save_button_status._value.text() == "Needed"

    source_path = tmp_path / "flat.png"
    Image.new("RGB", (32, 16), (200, 120, 40)).save(source_path)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    profile = window._current_profile
    profile.canvas = ScreenRect(10, 10, 200, 100)
    profile.color_box = ScreenRect(300, 10, 100, 100)
    profile.hue_bar = ScreenRect(420, 10, 12, 100)
    window.apply_brush_check.setChecked(False)
    window.dry_run_check.setChecked(True)
    window._update_start_availability()
    # A dry run never opens the painting UI, so it does not need Save.
    assert window.start_button.isEnabled()

    window.dry_run_check.setChecked(False)
    window._hotkeys_ready = True
    window._hotkeys = type("LiveHotkeys", (), {"running": True})()
    window._update_start_availability()
    assert not window.start_button.isEnabled()
    assert "Save button" in window.start_button.toolTip()
    with pytest.raises(ValueError, match="Save button"):
        window._validate_profile_on_virtual_screen(anti_afk=True)

    profile.save_button = ScreenRect(300, 130, 90, 30)
    window._refresh_profile_ui()
    qtbot.waitUntil(
        lambda: window._plan is not None
        and not window._plan_pending
        and not window._plan_processing,
        timeout=5000,
    )
    assert window.save_button_status._value.text() == "Ready"
    assert window.start_button.isEnabled()

    document = window._settings_document()
    assert document["safety"]["anti_afk_enabled"] is True
    assert document["safety"]["anti_afk_interval_minutes"] == 12
    settings = window._painter_settings(document, dry_run=False)
    assert settings.anti_afk_enabled is True
    assert settings.anti_afk_interval_seconds == 12 * 60
    assert window._painter_settings(document, dry_run=True).anti_afk_enabled is False

    # Off again, the button is optional and Start no longer asks for it.
    window.anti_afk_check.setChecked(False)
    profile.save_button = None
    window._refresh_profile_ui()
    assert window.save_button_status._value.text() == "Optional"
    assert window.start_button.isEnabled()


def test_the_resolution_cap_comes_from_rusts_sign_table_when_the_grid_is_unread(
    window: MainWindow,
) -> None:
    """Live: the brush read ~649 rows on the 1024x512 XXL canvas (its grid
    probe failed at two pixels per texel), the snap made that 640, the cap
    came out 1278x640 and Max planned a quarter more cells than the sign
    shows.  The brush count plus the rectangle's shape name the sign's
    table entry, which is exact; and typing that entry's width derives its
    height rather than the rectangle's rounding of it."""

    assert window._current_profile is not None
    window._current_profile.canvas = ScreenRect(69, 200, 2079, 1041)
    window._current_profile.metadata["brush_size_model"] = fit_brush_size_model(
        [(size, size / 649.0) for size in (24, 5.25, 2.5, 1.25, 1)]
    ).to_dict()
    window._refresh_profile_ui()

    assert window._sign_resolution_cap() == (1024, 512)
    window.quality_combo.setCurrentText("Max")
    assert (window.logical_width_spin.value(), window.logical_height_spin.value()) == (1024, 512)
    assert "Rust's own sign data" in window.resolution_cap_panel.toolTip()

    window.quality_combo.setCurrentText("Custom")
    window.logical_width_spin.setValue(1024)
    assert window.logical_height_spin.value() == 512
    assert "1024×512" in window.resolution_cap_label.text()
    # A size that is nobody's texture is left as the rectangle derives it.
    window.logical_width_spin.setValue(500)
    assert window.logical_height_spin.value() == 250


def test_the_resolution_boxes_apply_on_enter_not_per_keystroke(window: MainWindow) -> None:
    """Typing "1024" used to be rewritten after "10": the derived height
    clamped, both boxes were written back, and the user's next digits landed
    on "16" - ending at the cap, never at 1024."""

    from PySide6.QtCore import Qt
    from PySide6.QtTest import QTest

    assert window._current_profile is not None
    window._current_profile.canvas = ScreenRect(69, 200, 2079, 1041)
    window._refresh_profile_ui()
    window.quality_combo.setCurrentText("Custom")
    spin = window.logical_width_spin
    spin.setFocus()
    spin.selectAll()
    for char in "1024":
        QTest.keyClick(spin, char)
    assert spin.text() == "1024"
    QTest.keyClick(spin, Qt.Key_Return)
    assert (spin.value(), window.logical_height_spin.value()) == (1024, 512)


def test_a_timelapse_can_be_switched_on_during_a_pause(
    window: MainWindow, monkeypatch: pytest.MonkeyPatch, qtbot
) -> None:
    """The wish to keep a run arrives once it is plainly going well.

    The timelapse controls stay live through a pause, and a recording that
    was never started opens as the job resumes - on the artwork as it is
    now, never on the probe strokes of a job still measuring its brush.
    The same live controls change a running recording's pace or stop it.
    """

    from types import SimpleNamespace

    import app.timelapse as timelapse_module

    monkeypatch.setattr(
        timelapse_module,
        "capture_region",
        lambda region: Image.new("RGB", (region.width, region.height)),
    )
    window._current_profile.canvas = ScreenRect(0, 0, 64, 32)

    class _PausedPainter:
        state = PainterState.PAUSED
        is_active = True
        is_alive = True
        input = SimpleNamespace(emits_real_input=True)

        def __init__(self, phase: str) -> None:
            self.progress = PaintProgress(
                PainterState.PAUSED, 0, 1, 0, 1, 0, 1, 0.0, 0.0, None, "", phase
            )
            self.resumed = 0

        def retune(self, settings) -> bool:
            return True

        def resume(self) -> bool:
            self.resumed += 1
            return True

    window.timelapse_check.setChecked(False)
    painter = _PausedPainter("paint")
    window._painter = painter
    window._update_start_availability()
    assert window.timelapse_check.isEnabled()
    assert window.timelapse_interval_spin.isEnabled()
    assert window.timelapse_final_check.isEnabled()
    assert not window.quality_combo.isEnabled()

    # Left off, a resume records nothing.
    window._start_or_resume()
    assert painter.resumed == 1
    assert window._timelapse_recorder is None

    # Switched on during the pause, the resume starts the recording.
    window.timelapse_check.setChecked(True)
    window.timelapse_interval_spin.setValue(7)
    window._start_or_resume()
    assert painter.resumed == 2
    recorder = window._timelapse_recorder
    assert recorder is not None
    assert window._timelapse_timer.isActive()
    assert window._timelapse_timer.interval() == 7000

    # A slower pace set during the next pause reaches the running timer.
    window.timelapse_interval_spin.setValue(9)
    window._start_or_resume()
    assert window._timelapse_recorder is recorder
    assert window._timelapse_timer.interval() == 9000

    # And switching it off stops the recording without ending the job.
    window.timelapse_check.setChecked(False)
    window._start_or_resume()
    assert painter.resumed == 4
    assert window._timelapse_recorder is None
    assert not window._timelapse_timer.isActive()

    # A job paused while still measuring its brush waits for the artwork:
    # the painting-phase progress update opens the recording, as always.
    window.timelapse_check.setChecked(True)
    calibrating = _PausedPainter("calibrate")
    window._painter = calibrating
    window._start_or_resume()
    assert calibrating.resumed == 1
    assert window._timelapse_recorder is None
    calibrating.state = PainterState.RUNNING
    window._on_paint_progress(
        window._paint_generation,
        PaintProgress(PainterState.RUNNING, 0, 1, 0, 1, 0, 1, 0.0, 0.0, None, "", "paint"),
    )
    assert window._timelapse_recorder is not None
    window._finish_timelapse(final=False)
    window._painter = None
    window._update_start_availability()


def test_the_ui_guard_is_a_live_safety_setting_that_reaches_the_painter(
    window: MainWindow,
) -> None:
    """On by default, saved with the safety settings, editable in a pause."""

    from types import SimpleNamespace

    assert window.ui_guard_check.isChecked()
    document = window._settings_document()
    assert document["safety"]["ui_guard_enabled"] is True
    assert window._painter_settings(document, False).ui_guard_enabled is True

    window.ui_guard_check.setChecked(False)
    document = window._settings_document()
    assert document["safety"]["ui_guard_enabled"] is False
    assert window._painter_settings(document, False).ui_guard_enabled is False

    class _PausedPainter:
        state = PainterState.PAUSED
        is_active = True
        is_alive = True
        input = SimpleNamespace(emits_real_input=True)

        def __init__(self) -> None:
            self.retuned: list = []

        def retune(self, settings) -> bool:
            self.retuned.append(settings)
            return True

        def resume(self) -> bool:
            return True

    painter = _PausedPainter()
    window._painter = painter
    window._update_start_availability()
    assert window.ui_guard_check.isEnabled()
    window.ui_guard_check.setChecked(True)
    window._start_or_resume()
    assert painter.retuned and painter.retuned[0].ui_guard_enabled is True
    window._painter = None
    window._update_start_availability()


# --------------------------------------------------------------- resume from here


def _load_small_plan(window: MainWindow, tmp_path: Path, qtbot) -> None:
    source_path = tmp_path / "two-tone.png"
    source = Image.new("RGB", (16, 8), (50, 180, 90))
    for x in range(8, 16):
        for y in range(8):
            source.putpixel((x, y), (200, 40, 40))
    source.save(source_path)
    window.quality_combo.setCurrentText("Custom")
    window.logical_width_spin.setValue(16)
    window.logical_height_spin.setValue(8)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)


def test_the_resume_slider_offers_a_record_only_to_the_plan_it_was_written_for(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """A stroke number means something only in its own plan's order.

    A record for this plan puts the slider at its stroke and ticks Resume;
    a record for another plan is named as such and not offered; no record
    leaves the slider at zero for the user to set by eye.
    """

    from app.resume_record import advanced, plan_fingerprint, record_for_job

    _load_small_plan(window, tmp_path, qtbot)
    plan = window._plan
    assert plan is not None and plan.stroke_count >= 4
    store = window._resume_store

    # No record: nothing offered, but the control is there to set by hand.
    assert window.resume_panel.isEnabled()
    assert not window.resume_check.isChecked()
    assert window.resume_slider.maximum() == plan.stroke_count
    assert window.resume_slider.value() == 0
    assert "No record for this plan" in window.resume_notice.text()
    assert window.start_button.text().startswith("START PAINTING")

    # A record for a different plan is warned about, not offered.
    other = PaintPlan(4, 4, (ColorGroup((1, 2, 3), (Stroke(0, 0, 3, 0), Stroke(0, 1, 3, 1)), 8),))
    store.save(advanced(record_for_job(other, image_path="elsewhere.png"), completed_strokes=1, color_index=1))
    window._refresh_resume_offer()
    assert not window.resume_check.isChecked()
    assert window.resume_slider.value() == 0
    assert "planned differently" in window.resume_notice.text()
    assert "elsewhere.png" in window.resume_notice.text()

    # A record for this very plan is offered at its stroke.
    mine = advanced(
        record_for_job(plan),
        completed_strokes=3,
        color_index=1,
        state="paused",
        reason="painting UI not found - open the sign again and resume",
        interrupted_by_ui_loss=True,
    )
    store.save(mine)
    assert mine.fingerprint == plan_fingerprint(plan)
    window._refresh_resume_offer()
    assert window.resume_check.isChecked()
    assert window.resume_slider.value() == 3
    assert "the sign went away" in window.resume_notice.text()
    assert window._resume_start_stroke() == 3
    assert window.start_button.text().startswith("RESUME FROM STROKE 3")
    assert "3 of" in window.resume_position_label.text()

    # A finished record is history, not an offer.
    store.save(advanced(mine, completed_strokes=plan.stroke_count, color_index=1, state="completed", finished=True))
    window._refresh_resume_offer()
    assert not window.resume_check.isChecked()
    assert window.resume_slider.value() == 0


def test_the_resume_slider_previews_the_first_strokes_and_starts_the_job_there(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    _load_small_plan(window, tmp_path, qtbot)
    plan = window._plan
    assert plan is not None
    whole = window.paint_preview._source.toImage()

    window.resume_check.setChecked(True)
    window.resume_slider.setValue(plan.stroke_count // 2)
    qtbot.waitUntil(lambda: not window._resume_preview_timer.isActive(), timeout=2000)
    partial = window.paint_preview._source.toImage()
    assert partial.size() == whole.size()
    assert partial != whole, "the preview should show only the first strokes"
    assert window.start_button.text().startswith(
        f"RESUME FROM STROKE {plan.stroke_count // 2:,}"
    )

    # Unticked, the whole picture is back.
    window.resume_check.setChecked(False)
    qtbot.waitUntil(lambda: not window._resume_preview_timer.isActive(), timeout=2000)
    assert window.paint_preview._source.toImage() == whole
    assert window.start_button.text().startswith("START PAINTING")

    # Ticked again, the job starts at the slider's stroke and paints the rest.
    window.resume_check.setChecked(True)
    offset = plan.stroke_count // 2
    window.dry_run_check.setChecked(True)
    window.countdown_spin.setValue(0)
    window._start_or_resume()
    assert window._countdown is not None
    assert window._pending_paint is not None
    assert window._pending_paint.start_stroke == offset
    # The control locks with the rest of the job's shape.
    assert not window.resume_panel.isEnabled()
    window._countdown._tick()
    qtbot.waitUntil(
        lambda: window._painter is not None
        and getattr(window._painter.state, "value", "") == "completed",
        timeout=5000,
    )
    assert window._painter._job.start_stroke == offset
    assert window._painter.progress.completed_strokes == plan.stroke_count
    qtbot.waitUntil(lambda: window.resume_panel.isEnabled(), timeout=2000)


def test_resume_preview_replays_early_color_instead_of_revealing_final_color() -> None:
    plan = PaintPlan(
        2,
        1,
        (
            ColorGroup((180, 30, 20), (Stroke(0, 0, 1, 0),), 2),
            ColorGroup((20, 40, 190), (Stroke(0, 0, 0, 0),), 1),
        ),
    )

    after_first = _build_plan_prefix_image(plan, 1)
    finished = _build_plan_prefix_image(plan, 2)

    assert after_first.getpixel((0, 0)) == (180, 30, 20)
    assert after_first.getpixel((1, 0)) == (180, 30, 20)
    assert finished.getpixel((0, 0)) == (20, 40, 190)
    assert finished.getpixel((1, 0)) == (180, 30, 20)


def test_resume_records_follow_a_real_job_and_stamp_why_it_stopped(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """Every few seconds while painting, at once when it stops, finished at the end."""

    from types import SimpleNamespace

    from app.resume_record import plan_fingerprint

    _load_small_plan(window, tmp_path, qtbot)
    plan = window._plan
    assert plan is not None
    store = window._resume_store
    fingerprint = plan_fingerprint(plan)

    class _Painter:
        state = PainterState.RUNNING
        input = SimpleNamespace(emits_real_input=True)
        progress = PaintProgress(PainterState.RUNNING, 0, 1, 0, 1, 0, 1, 0.0, 0.0, None)
        measured_brush_size_model = None
        measured_texel_grid = None
        paint_phase_timing = None

    painter = _Painter()
    window._painter = painter
    generation = window._paint_generation
    pending = _PendingPaint(
        plan=plan,
        profile=window._current_profile,
        settings=window._settings_document(),
        dry_run=False,
        start_stroke=2,
    )
    window._open_resume_record(pending)
    assert window._resume_record is not None
    assert window._resume_record.completed_strokes == 2
    # Nothing on disk until the artwork is going down.
    assert store.load(fingerprint) is None

    def progress(completed: int, phase: str, state: PainterState = PainterState.RUNNING) -> PaintProgress:
        return PaintProgress(state, 1, 2, completed, plan.stroke_count, completed, plan.stroke_count, 0.0, 0.0, None, "", phase)

    window._on_paint_progress(generation, progress(2, "calibrate"))
    assert store.load(fingerprint) is None
    window._on_paint_progress(generation, progress(5, "paint"))
    first = store.load(fingerprint)
    assert first is not None and first.completed_strokes == 5 and first.color_index == 1
    assert first.state == "running" and first.resumable
    # Throttled: the next update within the interval is held in memory ...
    window._on_paint_progress(generation, progress(6, "paint"))
    assert store.load(fingerprint).completed_strokes == 5
    # ... and written the moment the job stops, with the painter's reason.
    painter.state = PainterState.PAUSED
    window._on_paint_state(
        generation,
        PainterState.PAUSED,
        "painting UI not found - open the sign again and resume",
    )
    stopped = store.load(fingerprint)
    assert stopped is not None
    assert stopped.completed_strokes == 6
    assert stopped.state == "paused" and stopped.interrupted_by_ui_loss
    assert "painting UI not found" in stopped.reason
    # While the job is paused the offer is not refreshed under it; the
    # record is what a restart of the app would find.

    # The touch-up phase means the artwork is complete.
    painter.state = PainterState.RUNNING
    window._on_paint_progress(generation, progress(1, "verify"))
    assert store.load(fingerprint).completed_strokes == plan.stroke_count

    painter.state = PainterState.COMPLETED
    window._on_paint_state(generation, PainterState.COMPLETED, "")
    done = store.load(fingerprint)
    assert done is not None and done.finished and done.state == "completed"
    assert window._resume_record is None
    # And the panel no longer offers it.
    assert not window.resume_check.isChecked()
    window._painter = None


def test_an_automatic_pause_screenshots_the_screen_and_shows_it(
    window: MainWindow, tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """What tripped a guard is on the screen for a moment; the app keeps it.

    A pause the painter called on its own comes with a screenshot of the
    whole screen, offered from the progress panel while the job is paused
    and from the resume control afterwards; a pause the user asked for
    does not.
    """

    from types import SimpleNamespace

    import app.screen as screen_module
    from app.gui.widgets import ScreenshotViewer
    from app.resume_record import plan_fingerprint

    _load_small_plan(window, tmp_path, qtbot)
    plan = window._plan
    assert plan is not None
    captured: list[object] = []
    monkeypatch.setattr(
        screen_module, "get_virtual_screen_rect", lambda: ScreenRect(0, 0, 48, 24)
    )
    monkeypatch.setattr(
        screen_module,
        "capture_region",
        lambda rect: captured.append(rect)
        or Image.new("RGB", (rect.width, rect.height), (200, 30, 30)),
    )

    class _Painter:
        state = PainterState.RUNNING
        input = SimpleNamespace(emits_real_input=True, release_all=lambda: None)
        progress = PaintProgress(PainterState.RUNNING, 0, 1, 0, 1, 0, 1, 0.0, 0.0, None)
        measured_brush_size_model = None
        measured_texel_grid = None
        paint_phase_timing = None

        def shutdown(self, timeout: float = 0.0) -> None:
            pass

    painter = _Painter()
    window._painter = painter
    generation = window._paint_generation
    window._open_resume_record(
        _PendingPaint(
            plan=plan,
            profile=window._current_profile,
            settings=window._settings_document(),
            dry_run=False,
        )
    )
    window._on_paint_progress(
        generation,
        PaintProgress(PainterState.RUNNING, 1, 2, 3, plan.stroke_count, 3, plan.stroke_count, 0.0, 0.0, None, "", "paint"),
    )

    # The run report takes its own picture of the screen as the artwork
    # starts, on a worker; let it land before counting the pause's.
    qtbot.waitUntil(lambda: len(captured) == 1, timeout=5000)
    captured.clear()

    # The user's own pause needs no explaining.
    painter.state = PainterState.PAUSED
    window._on_paint_state(generation, PainterState.PAUSED, "user hotkey/button")
    qtbot.wait(50)
    assert captured == []
    assert window.pause_screenshot_button.isHidden()

    painter.state = PainterState.RUNNING
    window._on_paint_state(generation, PainterState.RUNNING, "resumed")
    painter.state = PainterState.PAUSED
    reason = "painting UI not found - open the sign again and resume"
    window._on_paint_state(generation, PainterState.PAUSED, reason)
    qtbot.waitUntil(lambda: window._pause_screenshot is not None, timeout=5000)
    assert captured == [ScreenRect(0, 0, 48, 24)]
    path, shown_reason, taken_at = window._pause_screenshot
    assert path.exists() and path.suffix == ".png"
    assert path.parent == window._pause_screenshot_directory()
    assert "painting-ui-not-found" in path.name
    assert shown_reason == reason and taken_at
    assert not window.pause_screenshot_button.isHidden()
    assert Image.open(path).size == (48, 24)

    # The record carries the screenshot for a later session.
    record = window._resume_store.load(plan_fingerprint(plan))
    assert record is not None
    assert record.screenshot_path == str(path)
    assert record.interrupted_by_ui_loss and record.completed_strokes == 3

    window._show_pause_screenshot()
    assert len(window._screenshot_viewers) == 1
    viewer = window._screenshot_viewers[0]
    assert isinstance(viewer, ScreenshotViewer)
    assert not viewer._pixmap.isNull()
    viewer.close()

    # Resumed, the button goes; the file stays.
    painter.state = PainterState.RUNNING
    window._on_paint_state(generation, PainterState.RUNNING, "resumed")
    assert window.pause_screenshot_button.isHidden()
    assert path.exists()

    # The job is abandoned; the next session's resume offer shows it.
    painter.state = PainterState.ABORTED
    window._on_paint_state(generation, PainterState.ABORTED, "emergency stop")
    window._painter = None
    window._refresh_resume_offer()
    assert window.resume_check.isChecked()
    assert not window.resume_screenshot_button.isHidden()
    window._show_offered_screenshot()
    qtbot.waitUntil(lambda: len(window._screenshot_viewers) == 1, timeout=2000)
    window._screenshot_viewers[0].close()


def test_a_paused_job_lends_out_the_resume_slider_as_a_viewfinder(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """While paused, the slider shows the picture as far as any stroke.

    It lands on the stroke the job stopped at, so the sign can be checked
    against what it should show by now, and moving it changes only the
    preview: the tick that would change where a job starts stays locked,
    and the resume goes ahead from where the job stopped.  Once the job
    is running again the slider and the preview are given back.
    """

    from types import SimpleNamespace

    _load_small_plan(window, tmp_path, qtbot)
    plan = window._plan
    assert plan is not None
    qtbot.waitUntil(lambda: not window._resume_preview_timer.isActive(), timeout=2000)
    whole = window.paint_preview._source.toImage()
    stopped_at = plan.stroke_count // 2

    class _PausedPainter:
        is_active = True
        is_alive = True
        input = SimpleNamespace(emits_real_input=True)

        def __init__(self) -> None:
            self.state = PainterState.PAUSED
            self.progress = SimpleNamespace(
                completed_strokes=stopped_at, elapsed_seconds=12.0
            )
            self.resumed = 0

        def seconds_until_anti_afk(self):
            return None

        def retune(self, settings) -> bool:
            return True

        def resume(self) -> bool:
            self.resumed += 1
            self.state = PainterState.RUNNING
            return True

    painter = _PausedPainter()
    window._painter = painter
    window._on_paint_state(window._paint_generation, PainterState.PAUSED, "user")

    assert window.resume_slider.isEnabled()
    assert window.resume_slider.value() == stopped_at
    assert not window.resume_check.isEnabled()
    assert not window.resume_check.isChecked()
    assert "Paused" in window.resume_notice.text()
    qtbot.waitUntil(lambda: not window._resume_preview_timer.isActive(), timeout=2000)
    at_stop = window.paint_preview._source.toImage()
    assert at_stop != whole, "the preview should stop at the paused stroke"

    window.resume_slider.setValue(2)
    qtbot.waitUntil(lambda: not window._resume_preview_timer.isActive(), timeout=2000)
    earlier = window.paint_preview._source.toImage()
    assert earlier != at_stop and earlier != whole
    assert window.start_button.text().startswith("RESUME PAINTING")

    window._start_or_resume()
    assert painter.resumed == 1
    window._on_paint_state(window._paint_generation, PainterState.RUNNING, "resumed")
    assert not window.resume_slider.isEnabled()
    assert window.resume_slider.value() == 0
    assert "Paused" not in window.resume_notice.text()
    qtbot.waitUntil(lambda: not window._resume_preview_timer.isActive(), timeout=2000)
    assert window.paint_preview._source.toImage() == whole


def test_the_progress_readout_counts_down_to_the_next_anti_afk_break(
    window: MainWindow,
) -> None:
    """Next to the elapsed time sits when the job will next save and jump."""

    from types import SimpleNamespace

    class _Painter:
        state = PainterState.RUNNING
        is_active = True
        is_alive = True
        input = SimpleNamespace(emits_real_input=True)
        until = 95.0
        progress = SimpleNamespace(elapsed_seconds=61.0)

        def seconds_until_anti_afk(self):
            return self.until

    painter = _Painter()
    window._painter = painter
    window._active_detail = "Stroke 3 / 9"
    window._refresh_active_detail()
    assert window.active_detail_label.text() == (
        "Stroke 3 / 9  •  1m 01s elapsed  •  anti-AFK in 1m 35s"
    )
    painter.until = 0.0
    window._refresh_active_detail()
    assert window.active_detail_label.text().endswith("anti-AFK break due now")
    painter.until = None
    window._refresh_active_detail()
    assert window.active_detail_label.text() == "Stroke 3 / 9  •  1m 01s elapsed"


# --------------------------------------------------------------- paint sessions


def test_opening_a_session_restores_its_image_and_settings_and_arms_the_offer(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    """A long sign set aside for a smaller one can be picked up again.

    Opening a saved session reloads its image and the picture settings its
    plan was made with; the rebuilt plan matches the record's fingerprint,
    so the resume offer arms itself at the recorded stroke.
    """

    from app.resume_record import advanced, plan_fingerprint, record_for_job

    # The long sign: painted as far as stroke 3, then set aside.
    _load_small_plan(window, tmp_path, qtbot)
    long_path = window._image_path
    record = advanced(
        record_for_job(
            window._plan,
            image_path=long_path,
            settings=window._settings_document(),
        ),
        completed_strokes=3,
        color_index=1,
        state="aborted",
        reason="switched paint sessions",
    )
    window._resume_store.save(record)

    # The smaller sign, planned at a different resolution.
    small_path = tmp_path / "small.png"
    Image.new("RGB", (8, 8), (30, 60, 200)).save(small_path)
    window.logical_width_spin.setValue(8)
    window.logical_height_spin.setValue(8)
    window.load_image(small_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    assert plan_fingerprint(window._plan) != record.fingerprint
    assert not window.resume_check.isChecked()

    # Back to the long sign through its session.
    window._open_paint_session(record)
    qtbot.waitUntil(
        lambda: window._plan_fingerprint == record.fingerprint, timeout=5000
    )
    assert window._image_path == long_path
    assert window.logical_width_spin.value() == 16
    assert window.resume_check.isChecked()
    assert window.resume_slider.value() == 3
    assert window.start_button.text().startswith("RESUME FROM STROKE 3")


def test_the_sessions_dialog_lists_deletes_and_protects_the_active_job(
    window: MainWindow, tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    import app.gui.sessions as sessions_module
    from app.gui.sessions import SessionListDialog
    from app.resume_record import advanced, record_for_job

    _load_small_plan(window, tmp_path, qtbot)
    store = window._resume_store
    mine = advanced(
        record_for_job(window._plan, image_path=window._image_path, settings={}),
        completed_strokes=2,
        color_index=1,
        state="paused",
    )
    other_plan = PaintPlan(4, 4, (ColorGroup((1, 2, 3), (Stroke(0, 0, 3, 0),), 8),))
    other = advanced(
        record_for_job(other_plan, image_path="elsewhere.png"),
        completed_strokes=1,
        color_index=1,
        state="aborted",
    )
    store.save(mine)
    store.save(other)

    dialog = SessionListDialog(
        store,
        store.records(),
        current_fingerprint=mine.fingerprint,
        active_fingerprint=mine.fingerprint,
        parent=window,
    )
    qtbot.addWidget(dialog)
    assert dialog.list.count() == 2

    def _select(fingerprint: str) -> None:
        dialog.list.clearSelection()
        for row in range(dialog.list.count()):
            item = dialog.list.item(row)
            item.setSelected(
                item.data(Qt.ItemDataRole.UserRole).fingerprint == fingerprint
            )

    # The job painting right now can be neither opened nor deleted from here.
    _select(mine.fingerprint)
    assert not dialog.delete_button.isEnabled()
    assert not dialog.open_button.isEnabled()

    # The other session deletes after a confirmation, and only from disk once.
    _select(other.fingerprint)
    assert dialog.delete_button.isEnabled()
    monkeypatch.setattr(
        sessions_module.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: sessions_module.QMessageBox.StandardButton.Yes),
    )
    dialog._delete_selected()
    assert dialog.list.count() == 1
    assert store.load(other.fingerprint) is None
    assert store.load(mine.fingerprint) is not None

    # With no job running, opening the remaining session reports it back.
    idle = SessionListDialog(
        store,
        store.records(),
        current_fingerprint=mine.fingerprint,
        active_fingerprint=None,
        parent=window,
    )
    qtbot.addWidget(idle)
    idle.list.setCurrentRow(0)
    assert idle.open_button.isEnabled()
    idle._open_selected()
    assert idle.chosen is not None
    assert idle.chosen.fingerprint == mine.fingerprint


def test_switching_sessions_stops_a_paused_job_and_keeps_its_place(
    window: MainWindow, tmp_path: Path, qtbot, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opening another session mid-job asks first, then stops with its reason."""

    from types import SimpleNamespace

    from app.resume_record import advanced, record_for_job

    _load_small_plan(window, tmp_path, qtbot)
    small_path = tmp_path / "small.png"
    Image.new("RGB", (8, 8), (30, 60, 200)).save(small_path)
    other_plan = PaintPlan(4, 4, (ColorGroup((1, 2, 3), (Stroke(0, 0, 3, 0),), 8),))
    record = advanced(
        record_for_job(other_plan, image_path=small_path, settings={}),
        completed_strokes=1,
        color_index=1,
        state="paused",
    )

    class _PausedPainter:
        state = PainterState.PAUSED
        is_active = True
        is_alive = True
        input = SimpleNamespace(emits_real_input=True)

        def __init__(self) -> None:
            self.aborted: list[str] = []

        def abort(self, reason: str = "") -> None:
            self.aborted.append(reason)

    painter = _PausedPainter()
    window._painter = painter

    # Declined, nothing moves.
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: main_window_module.QMessageBox.StandardButton.No),
    )
    window._open_paint_session(record)
    assert painter.aborted == []
    assert window._image_path != small_path

    # Confirmed, the job is stopped with its own reason and the image follows.
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "question",
        staticmethod(lambda *a, **k: main_window_module.QMessageBox.StandardButton.Yes),
    )
    window._open_paint_session(record)
    assert painter.aborted == ["switched paint sessions"]
    window._painter = None
    qtbot.waitUntil(lambda: window._image_path == small_path, timeout=5000)


