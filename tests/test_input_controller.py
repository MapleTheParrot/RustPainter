from __future__ import annotations

import threading

import pytest

from app.input_controller import MouseButton, SendInputController
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
