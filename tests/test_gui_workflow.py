from __future__ import annotations

import os
import threading
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
import numpy as np
from PIL import Image
from PySide6.QtCore import QEvent, Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import QColorDialog, QGraphicsSceneMouseEvent

import app.gui.main_window as main_window_module
from app.gui.main_window import MainWindow, _PendingPaint
from app.gui.widgets import ColorButton, CountdownDialog
from app.input_controller import MockInputController
from app.models import ColorGroup, PaintPlan, ScreenRect, ScaleMode, Stroke, TransparencyMode
from app.painter import Painter, PainterState
from app.profiles import DisplayMetadata, Profile
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
    assert len(window.paint_preview._items) == 2
    assert window.paint_preview._items[1].defaultTextColor().name() == "#55ff55"
    assert window._processed is not None
    old_y = window._text_layers[1].y
    item = window.paint_preview._items[1]
    item.setPos(item.pos().x(), item.pos().y() + 3)
    assert window._text_layers[1].y != old_y
    qtbot.waitUntil(lambda: window._processed is not None, timeout=5000)
    item = window.paint_preview._items[1]
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
    settings = window.page_stack.widget(1)
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
    assert not window.workspace_nav_button.icon().isNull()
    assert not window.settings_nav_button.icon().isNull()

    window.settings_nav_button.click()
    assert window.page_stack.currentWidget() is settings

    window.workspace_nav_button.click()
    window.quality_combo.setCurrentText("Custom")
    assert not window.custom_resolution_panel.isHidden()
    assert window.logical_width_spin.isEnabled()
    assert window.logical_height_spin.isEnabled()


def test_automatic_brush_sizing_marks_its_calibrations_as_required(
    window: MainWindow,
) -> None:
    window._current_profile.brush_slider = None
    window._current_profile.brush_preview = None
    window.apply_brush_check.setChecked(False)
    window._refresh_profile_ui()
    assert window.brush_slider_status._value.text() == "Optional"
    assert window.brush_preview_status._value.text() == "Optional"

    window.apply_brush_check.setChecked(True)
    assert window.brush_slider_status._value.text() == "Needed"
    assert window.brush_preview_status._value.text() == "Needed"


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


def test_brush_slider_is_required_only_when_brush_application_is_enabled(
    window: MainWindow,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = Profile.new(
        "No brush slider",
        canvas=ScreenRect(10, 10, 200, 100),
        color_box=ScreenRect(220, 10, 100, 100),
        hue_bar=ScreenRect(325, 10, 10, 100),
    )
    monkeypatch.setattr(
        "app.screen.get_virtual_screen",
        lambda: VirtualScreen(0, 0, 800, 600),
    )

    window._validate_profile_on_virtual_screen(profile, apply_brush_size=False)
    with pytest.raises(ValueError, match="Size slider"):
        window._validate_profile_on_virtual_screen(profile, apply_brush_size=True)

    profile.brush_slider = ScreenRect(220, 120, 115, 12)
    with pytest.raises(ValueError, match="brush-preview"):
        window._validate_profile_on_virtual_screen(profile, apply_brush_size=True)


def _profile_with_full_calibration(display: DisplayMetadata) -> Profile:
    return Profile.new(
        "GUI calibration test",
        canvas=ScreenRect(10, 10, 200, 100),
        color_box=ScreenRect(220, 10, 100, 100),
        hue_bar=ScreenRect(325, 10, 10, 100),
        brush_slider=ScreenRect(220, 120, 115, 12),
        brush_preview=ScreenRect(350, 120, 80, 80),
        display=display,
        metadata={"ui_reference": {"path": "old.png"}},
    )


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
    assert window._current_profile.brush_slider is None
    assert window._current_profile.brush_preview is None
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
