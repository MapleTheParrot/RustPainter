from __future__ import annotations

import threading

import pytest

import os

from app.input_controller import (
    MouseButton,
    QuartzInputController,
    SendInputController,
    create_system_input_controller,
    mac_virtual_key_code,
)
from app.screen import VirtualScreen


def test_failed_mouse_up_remains_tracked_for_retry() -> None:
    controller = object.__new__(SendInputController)
    controller._lock = threading.RLock()
    controller._held_buttons = {MouseButton.LEFT}

    def fail(_flags: int, **_coordinates: int) -> None:
        raise OSError("simulated SendInput failure")

    controller._mouse_event = fail
    with pytest.raises(OSError):
        controller.mouse_up(MouseButton.LEFT)
    assert controller.held_buttons == frozenset({MouseButton.LEFT})

    controller._mouse_event = lambda _flags, **_coordinates: None
    controller.mouse_up(MouseButton.LEFT)
    assert not controller.held_buttons


def test_absolute_move_rejects_targets_outside_virtual_desktop() -> None:
    controller = object.__new__(SendInputController)
    controller._lock = threading.RLock()
    controller._get_virtual_screen = lambda: VirtualScreen(-100, -50, 200, 100)
    calls: list[tuple[int, int, int]] = []
    controller._mouse_event = lambda flags, *, x=0, y=0: calls.append((flags, x, y))

    with pytest.raises(ValueError, match="outside the virtual desktop"):
        controller.move_mouse(100, 0)  # exclusive right edge
    assert calls == []

    controller.move_mouse(-100, -50)
    controller.move_mouse(99, 49)
    assert calls[0][1:] == (0, 0)
    assert calls[1][1:] == (65535, 65535)


def test_mac_virtual_key_codes_cover_hotkeys_letters_and_digits() -> None:
    assert mac_virtual_key_code("F8") == 0x64
    assert mac_virtual_key_code("F9") == 0x65
    assert mac_virtual_key_code("F10") == 0x6D
    assert mac_virtual_key_code("a") == 0x00
    assert mac_virtual_key_code("Z") == 0x06
    assert mac_virtual_key_code("0") == 0x1D
    assert mac_virtual_key_code("SPACE") == 0x31
    assert mac_virtual_key_code(0x64) == 0x64


def test_mac_virtual_key_code_rejects_unknown_input() -> None:
    import pytest

    with pytest.raises(ValueError):
        mac_virtual_key_code("F25")
    with pytest.raises(ValueError):
        mac_virtual_key_code("NOSUCHKEY")
    with pytest.raises(ValueError):
        mac_virtual_key_code(300)


def test_create_system_input_controller_picks_platform_backend() -> None:
    import sys

    import pytest

    if os.name == "nt":
        assert isinstance(create_system_input_controller(), SendInputController)
        with pytest.raises(OSError):
            QuartzInputController()
    elif sys.platform == "darwin":
        assert isinstance(create_system_input_controller(), QuartzInputController)
        with pytest.raises(OSError):
            SendInputController()
    else:
        with pytest.raises(OSError):
            create_system_input_controller()
