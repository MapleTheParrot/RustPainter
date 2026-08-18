"""Exercise the macOS backends on a real Mac (CI or a developer machine).

These run only on darwin and are the main defence against porting bugs for
maintainers who do not own a Mac.  A headless CI runner has no Accessibility
or Screen Recording grant, so anything gated behind those permissions is
allowed to *fail closed* -- but the failure must be the specific, actionable
one the code promises, not a crash or a wrong-shaped result.

What this still cannot prove: that strokes land correctly inside Rust. Only a
human on a Mac with the game open can confirm that.
"""

from __future__ import annotations

import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin", reason="macOS backends require darwin"
)


def _require_display() -> None:
    """Skip when the runner exposes no usable display.

    Everything below reads or drives the window server. A CI image without an
    attached display cannot exercise it, and that is an environment limit
    rather than a defect worth failing the build over.
    """

    from app.screen import get_virtual_screen

    try:
        screen = get_virtual_screen()
    except OSError as exc:
        pytest.skip(f"No usable display on this runner: {exc}")
    if screen.width <= 0 or screen.height <= 0:
        pytest.skip("Display reports an empty rectangle")


def test_pyobjc_frameworks_are_installed() -> None:
    import AppKit  # noqa: F401
    import Quartz  # noqa: F401


def test_virtual_screen_is_sane() -> None:
    _require_display()
    from app.screen import get_virtual_screen

    screen = get_virtual_screen()
    assert screen.width > 0 and screen.height > 0
    assert screen.monitor_count >= 1
    assert screen.right > screen.left and screen.bottom > screen.top


def test_cursor_position_is_reported_as_integers() -> None:
    _require_display()
    from app.screen import get_cursor_position

    x, y = get_cursor_position()
    assert isinstance(x, int) and isinstance(y, int)


def test_controller_constructs_and_reports_position() -> None:
    _require_display()
    from app.input_controller import QuartzInputController, create_system_input_controller

    controller = create_system_input_controller()
    assert isinstance(controller, QuartzInputController)
    x, y = controller.get_cursor_position()
    assert isinstance(x, int) and isinstance(y, int)


def test_move_mouse_rejects_targets_outside_the_desktop() -> None:
    _require_display()
    from app.input_controller import create_system_input_controller
    from app.screen import get_virtual_screen

    controller = create_system_input_controller()
    screen = get_virtual_screen()
    with pytest.raises(ValueError):
        controller.move_mouse(screen.right + 500, screen.top)
    with pytest.raises(ValueError):
        controller.move_mouse(screen.left, screen.bottom + 500)


def test_quartz_event_calls_use_correct_api_signatures() -> None:
    """Post real events to validate every Quartz call we make.

    Without the Accessibility grant these are silently dropped by the window
    server, so nothing moves on a CI runner -- but pyobjc still raises on a
    wrong argument type or arity, which is exactly the porting mistake worth
    catching.
    """
    _require_display()

    from app.input_controller import MouseButton, create_system_input_controller
    from app.screen import get_virtual_screen

    controller = create_system_input_controller()
    screen = get_virtual_screen()
    target_x = screen.left + screen.width // 2
    target_y = screen.top + screen.height // 2

    controller.move_mouse(target_x, target_y)
    assert controller.get_cursor_position() is not None

    controller.mouse_down(MouseButton.LEFT)
    assert MouseButton.LEFT in controller.held_buttons
    # Movement while held must take the Dragged path.
    controller.move_mouse(target_x + 5, target_y + 5)
    controller.mouse_up(MouseButton.LEFT)
    assert MouseButton.LEFT not in controller.held_buttons

    controller.mouse_down(MouseButton.RIGHT)
    controller.release_all()
    assert not controller.held_buttons

    controller.press_key("F8", hold_seconds=0.0)


def test_capture_region_matches_the_requested_rectangle() -> None:
    """Retina displays capture at pixel scale; we must return point size."""
    _require_display()

    from app.models import ScreenRect
    from app.screen import capture_region, get_virtual_screen

    screen = get_virtual_screen()
    rect = ScreenRect(screen.left + 10, screen.top + 10, 120, 80)
    try:
        image = capture_region(rect)
    except OSError as exc:
        # Screen Recording is not granted on a headless runner. The message
        # must tell a user how to fix it.
        assert "Screen Recording" in str(exc)
        pytest.skip("Screen Recording permission unavailable in this environment")
    assert image.size == (rect.width, rect.height)
    assert image.mode == "RGB"


def test_foreground_window_is_absent_or_well_formed() -> None:
    from app.screen import get_foreground_window

    info = get_foreground_window()
    if info is None:
        pytest.skip("No frontmost application in this environment")
    assert isinstance(info.title, str)
    assert info.process_id is None or info.process_id > 0
    assert isinstance(info.executable_name, str)


def test_hotkeys_either_register_or_fail_closed_with_guidance() -> None:
    """The abort hotkey is a safety control: it must never claim false health."""

    from app.hotkeys import GlobalHotkeyManager

    manager = GlobalHotkeyManager()
    assert manager.available is True
    try:
        started = manager.start(timeout=5.0)
        if started:
            assert manager.running is True
        else:
            assert manager.running is False
            error = manager.startup_error
            assert error is not None
            assert "Accessibility" in str(error)
    finally:
        manager.stop(timeout=5.0)
    assert manager.running is False
