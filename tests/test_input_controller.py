from __future__ import annotations

import os
import threading

import pytest

from app.input_controller import (
    MockInputController,
    MouseButton,
    SendInputController,
    create_system_input_controller,
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


def test_press_key_sends_the_scan_code_games_listen_for() -> None:
    """A game reads the hardware scan code, not the virtual key.

    Rust ignored a Space or E sent as a bare virtual key; the same events
    with the layout's scan code and the scan-code flag register as real
    presses.  Arrow and Delete keys additionally carry the extended flag.
    """

    controller = object.__new__(SendInputController)
    controller._lock = threading.RLock()
    scan_codes = {0x20: 0x39, ord("E"): 0x12, 0x25: 0x4B}
    controller._map_virtual_key = lambda vk, _kind: scan_codes.get(vk, 0)
    sent: list[tuple[int, int, int]] = []
    controller._send = lambda native: sent.append(
        (native.ki.wVk, native.ki.wScan, native.ki.dwFlags)
    )

    controller.press_key("SPACE", hold_seconds=0)
    controller.press_key("E", hold_seconds=0)
    controller.press_key("LEFT", hold_seconds=0)

    scancode, keyup, extended = 0x0008, 0x0002, 0x0001
    assert sent == [
        (0x20, 0x39, scancode),
        (0x20, 0x39, scancode | keyup),
        (ord("E"), 0x12, scancode),
        (ord("E"), 0x12, scancode | keyup),
        (0x25, 0x4B, scancode | extended),
        (0x25, 0x4B, scancode | extended | keyup),
    ]


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
    # win32k recovers the pixel by truncation: pixel = (dx * extent) >> 16.
    # The encoding aims at the center of each pixel's slice, so every corner
    # must recover exactly - the old inclusive-endpoint formula missed a
    # scatter of pixels by one on wide desktops.
    for (_, dx, dy), (pixel_x, pixel_y) in zip(calls, ((0, 0), (199, 99))):
        assert (dx * 200) >> 16 == pixel_x
        assert (dy * 100) >> 16 == pixel_y


def test_create_system_input_controller_picks_platform_backend() -> None:
    if os.name == "nt":
        assert isinstance(create_system_input_controller(), SendInputController)
    else:
        with pytest.raises(OSError):
            create_system_input_controller()


def test_a_held_modifier_key_is_tracked_and_released_with_everything_else() -> None:
    """Shift stays down between key_down and key_up, and release_all lifts it.

    The line tool holds Shift across a click; a modifier left stuck after an
    interrupted stroke would turn every later click into a line, so the held
    key must be part of what release_all cleans up.
    """

    controller = object.__new__(SendInputController)
    controller._lock = threading.RLock()
    controller._held_buttons = set()
    controller._held_keys = []
    controller._map_virtual_key = lambda _vk, _kind: 0x2A
    sent: list[tuple[int, int]] = []
    controller._send = lambda native: sent.append((native.ki.wVk, native.ki.dwFlags))

    controller.key_down("SHIFT")
    controller.key_down("SHIFT")  # already held: no second press
    assert [vk for vk, _flags in sent] == [0x10]
    controller.release_all()
    keyup = 0x0002
    assert (sent[-1][0], sent[-1][1] & keyup) == (0x10, keyup)
    assert controller._held_keys == []
    controller.key_up("SHIFT")  # not held: nothing to send
    assert len(sent) == 2


def test_mock_controller_records_held_keys_and_releases_them() -> None:
    controller = MockInputController()
    controller.key_down("shift")
    controller.key_down("shift")
    assert controller.held_keys == ("SHIFT",)
    controller.mouse_down()
    controller.release_all()
    assert controller.held_keys == ()
    assert [event.kind for event in controller.events] == [
        "key_down",
        "mouse_down",
        "mouse_up",
        "key_up",
    ]
