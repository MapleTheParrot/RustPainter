from __future__ import annotations

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
from PySide6.QtGui import QColor, QDragEnterEvent, QDropEvent, QKeyEvent, QShortcut
from PySide6.QtWidgets import QColorDialog, QGraphicsSceneMouseEvent

import app.gui.main_window as main_window_module
from app.gui.main_window import (
    MainWindow,
    _PendingPaint,
    _TextOverlayOptions,
)
from app.color_calibration import ColorCorrectionModel
from app.gui.widgets import ColorButton, CountdownDialog
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
    assert (window._plan.width, window._plan.height) == (256, 128)
    assert len(window._plan.color_groups) <= 32
    assert window._plan.stroke_count > 0
    assert not window.paint_preview._source.isNull()
    assert not window.dry_run_check.isChecked()
    assert not window.start_button.isEnabled()


def test_optimization_mode_merges_colors_and_gates_controls(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    # Balanced is the recommended default, and it supersedes stroke merging.
    assert window.paint_mode_combo.currentData() == "balanced"
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
    window.text_edit.setText("RUST")
    window.text_size_spin.setValue(30)
    window.text_color_button.set_color("#FFFFFF", emit=True)
    window.text_bold_check.setChecked(True)
    window.add_text_button.click()
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

    document = window._settings_document()
    saved = document["image"]["text_overlay"]["layers"]
    assert [layer["gradient"] for layer in saved] == [True, True]
    assert saved[0]["gradient_direction"] == "diagonal"
    assert saved[0]["gradient_color"] == "#FF0000"
    assert saved[0]["outline_width"] == 2
    assert saved[0]["outline_color"] == "#00FF00"

    window._apply_settings(document)
    restored = window._text_layers[0]
    assert restored.gradient is True
    assert restored.gradient_direction == "diagonal"
    assert restored.gradient_color == (255, 0, 0)
    assert restored.outline_width == 2
    assert restored.outline_color == (0, 255, 0)
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
    assert document["image"]["background_removal_scope"] == "connected"


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
    # The derived row count is the number a user can sanity-check against the
    # sign they are actually looking at.
    assert "128 rows" in window.brush_model_status.text()

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


def test_plan_summary_announces_a_capped_resolution(
    window: MainWindow, tmp_path: Path, qtbot
) -> None:
    assert window._current_profile is not None
    window._current_profile.canvas = ScreenRect(10, 10, 200, 100)
    window._current_profile.metadata["brush_size_model"] = fit_brush_size_model(
        [(size, size / 128.0) for size in (60, 30, 12)]
    ).to_dict()
    window._refresh_profile_ui()
    window.quality_combo.setCurrentText("Very High")

    source_path = tmp_path / "source.png"
    Image.new("RGB", (64, 32), (210, 30, 40)).save(source_path)
    window.load_image(source_path)
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)

    # The plan, and therefore the Rust preview, stays inside the ceiling and
    # says so next to the stroke counts instead of silently swapping sizes.
    assert (window._plan.width, window._plan.height) == (256, 128)
    assert "capped at 256×128" in window.processing_label.text()


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
    window.corner_abort_check.setChecked(False)
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
    window.state_badge.setText("ABORTED")

    window._on_paint_state(7, PainterState.RUNNING, "late worker callback")

    assert window.state_badge.text() == "ABORTED"


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

    # Changing a setting that invalidates the plan restarts the feedback.
    window._set_combo_data(window.paint_mode_combo, "fast")
    assert window.processing_spinner.is_spinning
    assert window.analysis_time.value_label.text() == "…"
    qtbot.waitUntil(lambda: window._plan is not None, timeout=5000)
    assert not window.processing_spinner.is_spinning
    assert window.analysis_time.value_label.text() != "…"


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

    document = window._settings_document()

    assert document["timelapse"]["enabled"] is True
    assert document["timelapse"]["interval_seconds"] == 25


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
    window._delete_selected_session()

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


def _select_session(window: MainWindow, directory: Path) -> None:
    window._refresh_timelapse_sessions()
    for row in range(window.timelapse_sessions.count()):
        item = window.timelapse_sessions.item(row)
        if item.data(Qt.ItemDataRole.UserRole) == str(directory):
            window.timelapse_sessions.setCurrentItem(item)
            return
    raise AssertionError(f"{directory} was not listed")


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
    window.timelapse_fps_spin.setValue(12)
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
    window.timelapse_fps_spin.setValue(24)
    window._set_combo_data(window.timelapse_format_combo, "gif")

    saved = window._settings_document()

    assert saved["timelapse"]["playback_frame_rate"] == 24
    assert saved["timelapse"]["export_format"] == "gif"
