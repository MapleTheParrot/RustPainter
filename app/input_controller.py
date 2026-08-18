"""Small, auditable input backends for painting with normal Windows input.

``SendInputController`` is the only class in this project that emits real input.
The painter depends on the ``InputController`` protocol instead, which keeps dry
runs and unit tests completely isolated from the user's mouse.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Protocol, runtime_checkable


LOGGER = logging.getLogger("rust_painter.input")


class MouseButton(str, Enum):
    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass(frozen=True, slots=True)
class InputEvent:
    """One deterministic event recorded by a mock or dry-run controller."""

    kind: str
    x: int | None = None
    y: int | None = None
    value: str | int | None = None


@runtime_checkable
class InputController(Protocol):
    """The GUI-independent input surface used by :class:`app.painter.Painter`."""

    is_dry_run: bool
    emits_real_input: bool
    skip_timing: bool

    def move_mouse(self, x: float, y: float) -> None: ...

    def mouse_down(self, button: MouseButton | str = MouseButton.LEFT) -> None: ...

    def mouse_up(self, button: MouseButton | str = MouseButton.LEFT) -> None: ...

    def click(
        self,
        x: float,
        y: float,
        *,
        button: MouseButton | str = MouseButton.LEFT,
        hold_seconds: float = 0.01,
    ) -> None: ...

    def drag(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        duration_seconds: float = 0.1,
        step_pixels: float = 4.0,
        button: MouseButton | str = MouseButton.LEFT,
        should_continue: Callable[[], bool] | None = None,
    ) -> bool: ...

    def press_key(self, key: int | str, *, hold_seconds: float = 0.01) -> None: ...

    def release_all(self) -> None: ...

    def get_cursor_position(self) -> tuple[int, int]: ...


def _coerce_button(button: MouseButton | str) -> MouseButton:
    try:
        return button if isinstance(button, MouseButton) else MouseButton(str(button).lower())
    except ValueError as exc:
        raise ValueError(f"Unsupported mouse button: {button!r}") from exc


class BaseInputController:
    """Convenience operations shared by the real and mock implementations."""

    is_dry_run = False
    emits_real_input = True
    skip_timing = False

    def click(
        self,
        x: float,
        y: float,
        *,
        button: MouseButton | str = MouseButton.LEFT,
        hold_seconds: float = 0.01,
    ) -> None:
        if hold_seconds < 0:
            raise ValueError("hold_seconds cannot be negative")
        self.move_mouse(x, y)
        self.mouse_down(button)
        try:
            if hold_seconds:
                time.sleep(hold_seconds)
        finally:
            self.mouse_up(button)

    def drag(
        self,
        start: tuple[float, float],
        end: tuple[float, float],
        *,
        duration_seconds: float = 0.1,
        step_pixels: float = 4.0,
        button: MouseButton | str = MouseButton.LEFT,
        should_continue: Callable[[], bool] | None = None,
    ) -> bool:
        if duration_seconds < 0:
            raise ValueError("duration_seconds cannot be negative")
        if step_pixels <= 0:
            raise ValueError("step_pixels must be positive")

        start_x, start_y = start
        end_x, end_y = end
        distance = math.hypot(end_x - start_x, end_y - start_y)
        steps = max(1, int(math.ceil(distance / step_pixels)))
        delay = duration_seconds / steps if duration_seconds else 0.0
        self.move_mouse(start_x, start_y)
        self.mouse_down(button)
        completed = False
        try:
            for index in range(1, steps + 1):
                if should_continue is not None and not should_continue():
                    return False
                ratio = index / steps
                self.move_mouse(
                    start_x + (end_x - start_x) * ratio,
                    start_y + (end_y - start_y) * ratio,
                )
                if delay:
                    time.sleep(delay)
            completed = True
            return True
        finally:
            self.mouse_up(button)
            if not completed:
                LOGGER.debug("Drag interrupted and mouse button released")


# Virtual-key names accepted by press_key.  Letters and digits are handled
# directly, while function keys are calculated below.
_NAMED_VIRTUAL_KEYS: dict[str, int] = {
    "BACKSPACE": 0x08,
    "TAB": 0x09,
    "ENTER": 0x0D,
    "SHIFT": 0x10,
    "CTRL": 0x11,
    "CONTROL": 0x11,
    "ALT": 0x12,
    "ESC": 0x1B,
    "ESCAPE": 0x1B,
    "SPACE": 0x20,
    "LEFT": 0x25,
    "UP": 0x26,
    "RIGHT": 0x27,
    "DOWN": 0x28,
    "DELETE": 0x2E,
}


def virtual_key_code(key: int | str) -> int:
    """Convert a compact key name such as ``F8`` or ``A`` to a Windows VK."""

    if isinstance(key, int):
        if not 0 <= key <= 0xFF:
            raise ValueError("A virtual-key code must be in the range 0..255")
        return key
    name = str(key).strip().upper()
    if len(name) == 1 and name.isascii() and name.isalnum():
        return ord(name)
    if name.startswith("F") and name[1:].isdigit():
        number = int(name[1:])
        if 1 <= number <= 24:
            return 0x70 + number - 1
    try:
        return _NAMED_VIRTUAL_KEYS[name]
    except KeyError as exc:
        raise ValueError(f"Unsupported key name: {key!r}") from exc


if os.name == "nt":
    from ctypes import wintypes

    _ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong

    class _MOUSEINPUT(ctypes.Structure):
        _fields_ = (
            ("dx", wintypes.LONG),
            ("dy", wintypes.LONG),
            ("mouseData", wintypes.DWORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", _ULONG_PTR),
        )

    class _KEYBDINPUT(ctypes.Structure):
        _fields_ = (
            ("wVk", wintypes.WORD),
            ("wScan", wintypes.WORD),
            ("dwFlags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", _ULONG_PTR),
        )

    class _HARDWAREINPUT(ctypes.Structure):
        _fields_ = (
            ("uMsg", wintypes.DWORD),
            ("wParamL", wintypes.WORD),
            ("wParamH", wintypes.WORD),
        )

    class _INPUTUNION(ctypes.Union):
        _fields_ = (("mi", _MOUSEINPUT), ("ki", _KEYBDINPUT), ("hi", _HARDWAREINPUT))

    class _INPUT(ctypes.Structure):
        _anonymous_ = ("data",)
        _fields_ = (("type", wintypes.DWORD), ("data", _INPUTUNION))

    class _POINT(ctypes.Structure):
        _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))


class SendInputController(BaseInputController):
    """Emit mouse and keyboard input through the Windows ``SendInput`` API.

    Absolute coordinates are normalized across the complete virtual desktop, so
    calibrated coordinates may be negative on monitors left of the primary one.
    """

    _INPUT_MOUSE = 0
    _INPUT_KEYBOARD = 1
    _KEYEVENTF_KEYUP = 0x0002
    _MOUSEEVENTF_MOVE = 0x0001
    _MOUSEEVENTF_LEFTDOWN = 0x0002
    _MOUSEEVENTF_LEFTUP = 0x0004
    _MOUSEEVENTF_RIGHTDOWN = 0x0008
    _MOUSEEVENTF_RIGHTUP = 0x0010
    _MOUSEEVENTF_MIDDLEDOWN = 0x0020
    _MOUSEEVENTF_MIDDLEUP = 0x0040
    _MOUSEEVENTF_VIRTUALDESK = 0x4000
    _MOUSEEVENTF_ABSOLUTE = 0x8000

    _BUTTON_FLAGS = {
        MouseButton.LEFT: (_MOUSEEVENTF_LEFTDOWN, _MOUSEEVENTF_LEFTUP),
        MouseButton.RIGHT: (_MOUSEEVENTF_RIGHTDOWN, _MOUSEEVENTF_RIGHTUP),
        MouseButton.MIDDLE: (_MOUSEEVENTF_MIDDLEDOWN, _MOUSEEVENTF_MIDDLEUP),
    }

    def __init__(self) -> None:
        if os.name != "nt":
            raise OSError("SendInputController is available only on Windows")
        from .screen import get_virtual_screen, set_dpi_awareness

        set_dpi_awareness()
        self._get_virtual_screen = get_virtual_screen
        self._user32 = ctypes.WinDLL("user32", use_last_error=True)
        self._send_input = self._user32.SendInput
        self._send_input.argtypes = (wintypes.UINT, ctypes.POINTER(_INPUT), ctypes.c_int)
        self._send_input.restype = wintypes.UINT
        self._get_cursor_pos = self._user32.GetCursorPos
        self._get_cursor_pos.argtypes = (ctypes.POINTER(_POINT),)
        self._get_cursor_pos.restype = wintypes.BOOL
        self._lock = threading.RLock()
        self._held_buttons: set[MouseButton] = set()

    @property
    def held_buttons(self) -> frozenset[MouseButton]:
        with self._lock:
            return frozenset(self._held_buttons)

    def _send(self, native_input: "_INPUT") -> None:
        sent = self._send_input(1, ctypes.byref(native_input), ctypes.sizeof(_INPUT))
        if sent != 1:
            error = ctypes.get_last_error()
            raise ctypes.WinError(error or 1)

    def _mouse_event(self, flags: int, *, x: int = 0, y: int = 0) -> None:
        native_input = _INPUT(type=self._INPUT_MOUSE)
        native_input.mi = _MOUSEINPUT(
            dx=x,
            dy=y,
            mouseData=0,
            dwFlags=flags,
            time=0,
            dwExtraInfo=0,
        )
        self._send(native_input)

    def move_mouse(self, x: float, y: float) -> None:
        virtual = self._get_virtual_screen()
        clamped_x = int(round(x))
        clamped_y = int(round(y))
        if not (
            virtual.left <= clamped_x < virtual.right
            and virtual.top <= clamped_y < virtual.bottom
        ):
            raise ValueError(
                f"Input target ({clamped_x}, {clamped_y}) is outside the virtual desktop "
                f"({virtual.left}, {virtual.top}, {virtual.width}, {virtual.height})"
            )
        # Windows maps the inclusive range 0..65535 to the inclusive desktop
        # endpoints.  width-1/height-1 is therefore important at the far edge.
        denominator_x = max(1, virtual.width - 1)
        denominator_y = max(1, virtual.height - 1)
        normalized_x = round((clamped_x - virtual.left) * 65535 / denominator_x)
        normalized_y = round((clamped_y - virtual.top) * 65535 / denominator_y)
        flags = (
            self._MOUSEEVENTF_MOVE
            | self._MOUSEEVENTF_ABSOLUTE
            | self._MOUSEEVENTF_VIRTUALDESK
        )
        with self._lock:
            self._mouse_event(flags, x=normalized_x, y=normalized_y)

    def mouse_down(self, button: MouseButton | str = MouseButton.LEFT) -> None:
        resolved = _coerce_button(button)
        with self._lock:
            if resolved in self._held_buttons:
                return
            self._mouse_event(self._BUTTON_FLAGS[resolved][0])
            self._held_buttons.add(resolved)

    def mouse_up(self, button: MouseButton | str = MouseButton.LEFT) -> None:
        resolved = _coerce_button(button)
        with self._lock:
            if resolved not in self._held_buttons:
                return
            # Retain tracking when SendInput reports failure so release_all()
            # can retry. A duplicate successful button-up is harmless; losing
            # the only record of a potentially held button is not.
            self._mouse_event(self._BUTTON_FLAGS[resolved][1])
            self._held_buttons.discard(resolved)

    def press_key(self, key: int | str, *, hold_seconds: float = 0.01) -> None:
        if hold_seconds < 0:
            raise ValueError("hold_seconds cannot be negative")
        vk = virtual_key_code(key)
        key_down = _INPUT(type=self._INPUT_KEYBOARD)
        key_down.ki = _KEYBDINPUT(vk, 0, 0, 0, 0)
        key_up = _INPUT(type=self._INPUT_KEYBOARD)
        key_up.ki = _KEYBDINPUT(vk, 0, self._KEYEVENTF_KEYUP, 0, 0)
        with self._lock:
            self._send(key_down)
        try:
            if hold_seconds:
                time.sleep(hold_seconds)
        finally:
            with self._lock:
                self._send(key_up)

    def release_all(self) -> None:
        """Release every mouse button this controller believes it pressed."""

        first_error: BaseException | None = None
        for button in tuple(self.held_buttons):
            try:
                self.mouse_up(button)
            except BaseException as exc:  # continue releasing the other buttons
                first_error = first_error or exc
                LOGGER.exception("Could not release %s mouse button", button.value)
        if first_error is not None:
            raise first_error

    def get_cursor_position(self) -> tuple[int, int]:
        point = _POINT()
        if not self._get_cursor_pos(ctypes.byref(point)):
            raise ctypes.WinError(ctypes.get_last_error() or 1)
        return int(point.x), int(point.y)


class MockInputController(BaseInputController):
    """In-memory input controller used by tests and plan visualizers."""

    is_dry_run = True
    emits_real_input = False
    skip_timing = False

    def __init__(
        self,
        *,
        initial_position: tuple[int, int] = (0, 0),
        operation_delay: float = 0.0,
        record_events: bool = True,
    ) -> None:
        if operation_delay < 0:
            raise ValueError("operation_delay cannot be negative")
        self.events: list[InputEvent] = []
        self._position = (int(initial_position[0]), int(initial_position[1]))
        self._held_buttons: set[MouseButton] = set()
        self._operation_delay = operation_delay
        self._record_events = record_events
        self._lock = threading.RLock()

    @property
    def actions(self) -> list[InputEvent]:
        return self.events

    @property
    def held_buttons(self) -> frozenset[MouseButton]:
        with self._lock:
            return frozenset(self._held_buttons)

    def _delay(self) -> None:
        if self._operation_delay:
            time.sleep(self._operation_delay)

    def move_mouse(self, x: float, y: float) -> None:
        position = (int(round(x)), int(round(y)))
        with self._lock:
            self._position = position
            if self._record_events:
                self.events.append(InputEvent("move", *position))
        self._delay()

    def mouse_down(self, button: MouseButton | str = MouseButton.LEFT) -> None:
        resolved = _coerce_button(button)
        with self._lock:
            if resolved in self._held_buttons:
                return
            self._held_buttons.add(resolved)
            if self._record_events:
                self.events.append(InputEvent("mouse_down", value=resolved.value))
        self._delay()

    def mouse_up(self, button: MouseButton | str = MouseButton.LEFT) -> None:
        resolved = _coerce_button(button)
        with self._lock:
            if resolved not in self._held_buttons:
                return
            self._held_buttons.discard(resolved)
            if self._record_events:
                self.events.append(InputEvent("mouse_up", value=resolved.value))
        self._delay()

    def press_key(self, key: int | str, *, hold_seconds: float = 0.01) -> None:
        if hold_seconds < 0:
            raise ValueError("hold_seconds cannot be negative")
        value = key if isinstance(key, int) else str(key).upper()
        with self._lock:
            if self._record_events:
                self.events.append(InputEvent("key_down", value=value))
        if hold_seconds:
            time.sleep(hold_seconds)
        with self._lock:
            if self._record_events:
                self.events.append(InputEvent("key_up", value=value))
        self._delay()

    def release_all(self) -> None:
        for button in tuple(self.held_buttons):
            self.mouse_up(button)

    def get_cursor_position(self) -> tuple[int, int]:
        with self._lock:
            return self._position

    def clear(self) -> None:
        with self._lock:
            self.events.clear()


class DryRunInputController(MockInputController):
    """A mock backend that optionally logs its plan-level input actions."""

    skip_timing = True

    def __init__(self, *, detailed_logging: bool = False) -> None:
        # A large high-quality plan can contain millions of interpolation moves;
        # progress callbacks already visualize it, so dry runs do not retain an
        # unbounded event list. Tests use MockInputController's default instead.
        super().__init__(record_events=False)
        self.detailed_logging = detailed_logging

    def move_mouse(self, x: float, y: float) -> None:
        super().move_mouse(x, y)
        if self.detailed_logging:
            LOGGER.debug("Dry run: move mouse to (%d, %d)", round(x), round(y))

    def mouse_down(self, button: MouseButton | str = MouseButton.LEFT) -> None:
        super().mouse_down(button)
        if self.detailed_logging:
            LOGGER.debug("Dry run: %s mouse down", _coerce_button(button).value)

    def mouse_up(self, button: MouseButton | str = MouseButton.LEFT) -> None:
        was_held = _coerce_button(button) in self.held_buttons
        super().mouse_up(button)
        if self.detailed_logging and was_held:
            LOGGER.debug("Dry run: %s mouse up", _coerce_button(button).value)


# Short aliases make dependency injection at the GUI boundary pleasant.
WindowsInputController = SendInputController
MockInput = MockInputController


__all__ = [
    "DryRunInputController",
    "InputController",
    "InputEvent",
    "MockInput",
    "MockInputController",
    "MouseButton",
    "SendInputController",
    "WindowsInputController",
    "virtual_key_code",
]
