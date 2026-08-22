"""Small, auditable input backends for painting with ordinary OS input.

``SendInputController`` (Windows) and ``QuartzInputController`` (macOS) are the
only classes in this project that emit real input.
The painter depends on the ``InputController`` protocol instead, which keeps dry
runs and unit tests completely isolated from the user's mouse.
"""

from __future__ import annotations

import ctypes
import logging
import math
import os
import sys
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


# ANSI-layout macOS virtual keycodes (HIToolbox Events.h kVK_* values).
_MAC_LETTER_KEYS = {
    "A": 0x00, "B": 0x0B, "C": 0x08, "D": 0x02, "E": 0x0E, "F": 0x03,
    "G": 0x05, "H": 0x04, "I": 0x22, "J": 0x26, "K": 0x28, "L": 0x25,
    "M": 0x2E, "N": 0x2D, "O": 0x1F, "P": 0x23, "Q": 0x0C, "R": 0x0F,
    "S": 0x01, "T": 0x11, "U": 0x20, "V": 0x09, "W": 0x0D, "X": 0x07,
    "Y": 0x10, "Z": 0x06,
    "0": 0x1D, "1": 0x12, "2": 0x13, "3": 0x14, "4": 0x15, "5": 0x17,
    "6": 0x16, "7": 0x1A, "8": 0x1C, "9": 0x19,
}
_MAC_FUNCTION_KEYS = {
    1: 0x7A, 2: 0x78, 3: 0x63, 4: 0x76, 5: 0x60, 6: 0x61, 7: 0x62,
    8: 0x64, 9: 0x65, 10: 0x6D, 11: 0x67, 12: 0x6F, 13: 0x69, 14: 0x6B,
    15: 0x71, 16: 0x6A, 17: 0x40, 18: 0x4F, 19: 0x50, 20: 0x5A,
}
_MAC_NAMED_KEYS = {
    "BACKSPACE": 0x33, "TAB": 0x30, "ENTER": 0x24, "SHIFT": 0x38,
    "CTRL": 0x3B, "CONTROL": 0x3B, "ALT": 0x3A, "ESC": 0x35,
    "ESCAPE": 0x35, "SPACE": 0x31, "LEFT": 0x7B, "UP": 0x7E,
    "RIGHT": 0x7C, "DOWN": 0x7D, "DELETE": 0x75,
}


def mac_virtual_key_code(key: int | str) -> int:
    """Convert a compact key name such as ``F8`` or ``A`` to a macOS keycode.

    Integers pass through so callers may supply a raw kVK_* value directly;
    they are NOT Windows VK codes and the two numbering schemes differ.
    """

    if isinstance(key, int):
        if not 0 <= key <= 0x7F:
            raise ValueError("A macOS virtual keycode must be in the range 0..127")
        return key
    name = str(key).strip().upper()
    if name in _MAC_LETTER_KEYS:
        return _MAC_LETTER_KEYS[name]
    if name.startswith("F") and name[1:].isdigit():
        number = int(name[1:])
        if number in _MAC_FUNCTION_KEYS:
            return _MAC_FUNCTION_KEYS[number]
    try:
        return _MAC_NAMED_KEYS[name]
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
    _KEYEVENTF_EXTENDEDKEY = 0x0001
    _KEYEVENTF_KEYUP = 0x0002
    _KEYEVENTF_SCANCODE = 0x0008
    _MAPVK_VK_TO_VSC = 0
    # Virtual keys whose scan code carries the 0xE0 prefix on a PC keyboard.
    _EXTENDED_VIRTUAL_KEYS = frozenset({0x25, 0x26, 0x27, 0x28, 0x2E})
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
        self._map_virtual_key = self._user32.MapVirtualKeyW
        self._map_virtual_key.argtypes = (wintypes.UINT, wintypes.UINT)
        self._map_virtual_key.restype = wintypes.UINT
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
        # win32k recovers the pixel by truncation - x = (dx * width) >> 16 - so
        # the safe encoding aims at the *center* of the target pixel's slice of
        # the 0..65535 range.  The older inclusive-endpoint formula
        # (round(x * 65535 / (width - 1))) landed one pixel short for a scatter
        # of positions on every desktop wider than 65536/width-ish pixels.
        normalized_x = min(
            65535,
            int((clamped_x - virtual.left + 0.5) * 65536 / max(1, virtual.width)),
        )
        normalized_y = min(
            65535,
            int((clamped_y - virtual.top + 0.5) * 65536 / max(1, virtual.height)),
        )
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

    def _key_input(self, vk: int, *, up: bool) -> "_INPUT":
        """Build a key event that a game will see as a real key.

        A keyboard event carrying only a virtual key reaches ordinary windows
        and text boxes through the message queue, but a game reading the
        keyboard through raw input or DirectInput looks at the hardware scan
        code and sees nothing.  Rust takes digits typed into its Size box
        either way, yet ignored Space and E sent this way.  The scan code is
        looked up from the layout and sent alongside the virtual key, with the
        extended-key flag for the keys whose scan code carries it.
        """

        scan = self._map_virtual_key(vk, self._MAPVK_VK_TO_VSC)
        flags = self._KEYEVENTF_KEYUP if up else 0
        if scan:
            flags |= self._KEYEVENTF_SCANCODE
            if vk in self._EXTENDED_VIRTUAL_KEYS:
                flags |= self._KEYEVENTF_EXTENDEDKEY
        native_input = _INPUT(type=self._INPUT_KEYBOARD)
        native_input.ki = _KEYBDINPUT(vk, scan, flags, 0, 0)
        return native_input

    def press_key(self, key: int | str, *, hold_seconds: float = 0.01) -> None:
        if hold_seconds < 0:
            raise ValueError("hold_seconds cannot be negative")
        vk = virtual_key_code(key)
        key_down = self._key_input(vk, up=False)
        key_up = self._key_input(vk, up=True)
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


class QuartzInputController(BaseInputController):
    """Emit mouse and keyboard input through macOS Quartz ``CGEventPost``.

    Coordinates are global display points with the origin at the top-left of
    the primary display, matching Qt's global coordinate space on macOS, so
    calibrated rectangles and synthesized input agree without conversion.
    Posting events requires the Accessibility permission (System Settings >
    Privacy & Security > Accessibility).
    """

    def __init__(self) -> None:
        if sys.platform != "darwin":
            raise OSError("QuartzInputController is available only on macOS")
        import Quartz  # pyobjc-framework-Quartz

        from .screen import get_virtual_screen

        self._quartz = Quartz
        self._get_virtual_screen = get_virtual_screen
        self._buttons = {
            MouseButton.LEFT: (
                Quartz.kCGMouseButtonLeft,
                Quartz.kCGEventLeftMouseDown,
                Quartz.kCGEventLeftMouseUp,
                Quartz.kCGEventLeftMouseDragged,
            ),
            MouseButton.RIGHT: (
                Quartz.kCGMouseButtonRight,
                Quartz.kCGEventRightMouseDown,
                Quartz.kCGEventRightMouseUp,
                Quartz.kCGEventRightMouseDragged,
            ),
            MouseButton.MIDDLE: (
                Quartz.kCGMouseButtonCenter,
                Quartz.kCGEventOtherMouseDown,
                Quartz.kCGEventOtherMouseUp,
                Quartz.kCGEventOtherMouseDragged,
            ),
        }
        self._lock = threading.RLock()
        self._held_buttons: set[MouseButton] = set()
        self._position = self.get_cursor_position()

    @property
    def held_buttons(self) -> frozenset[MouseButton]:
        with self._lock:
            return frozenset(self._held_buttons)

    def _post_mouse(self, event_type: int, x: float, y: float, cg_button: int) -> None:
        event = self._quartz.CGEventCreateMouseEvent(None, event_type, (x, y), cg_button)
        if event is None:
            raise OSError("Quartz refused to create a mouse event")
        self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, event)

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
        with self._lock:
            if self._held_buttons:
                # While a button is down macOS expects Dragged events; plain
                # MouseMoved during a hold is ignored by many applications.
                button = next(iter(self._held_buttons))
                cg_button, _down, _up, dragged = self._buttons[button]
                self._post_mouse(dragged, clamped_x, clamped_y, cg_button)
            else:
                self._post_mouse(
                    self._quartz.kCGEventMouseMoved,
                    clamped_x,
                    clamped_y,
                    self._quartz.kCGMouseButtonLeft,
                )
            self._position = (clamped_x, clamped_y)

    def mouse_down(self, button: MouseButton | str = MouseButton.LEFT) -> None:
        resolved = _coerce_button(button)
        with self._lock:
            if resolved in self._held_buttons:
                return
            cg_button, down, _up, _dragged = self._buttons[resolved]
            x, y = self._position
            self._post_mouse(down, x, y, cg_button)
            self._held_buttons.add(resolved)

    def mouse_up(self, button: MouseButton | str = MouseButton.LEFT) -> None:
        resolved = _coerce_button(button)
        with self._lock:
            if resolved not in self._held_buttons:
                return
            # Mirror SendInputController: keep tracking on failure so
            # release_all() can retry rather than losing the held state.
            cg_button, _down, up, _dragged = self._buttons[resolved]
            x, y = self._position
            self._post_mouse(up, x, y, cg_button)
            self._held_buttons.discard(resolved)

    def press_key(self, key: int | str, *, hold_seconds: float = 0.01) -> None:
        if hold_seconds < 0:
            raise ValueError("hold_seconds cannot be negative")
        keycode = mac_virtual_key_code(key)
        with self._lock:
            key_down = self._quartz.CGEventCreateKeyboardEvent(None, keycode, True)
            self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, key_down)
        try:
            if hold_seconds:
                time.sleep(hold_seconds)
        finally:
            with self._lock:
                key_up = self._quartz.CGEventCreateKeyboardEvent(None, keycode, False)
                self._quartz.CGEventPost(self._quartz.kCGHIDEventTap, key_up)

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
        event = self._quartz.CGEventCreate(None)
        location = self._quartz.CGEventGetLocation(event)
        return int(round(location.x)), int(round(location.y))


def create_system_input_controller() -> BaseInputController:
    """Return the real input backend for the current operating system."""

    if os.name == "nt":
        return SendInputController()
    if sys.platform == "darwin":
        return QuartzInputController()
    raise OSError("Real input synthesis is supported only on Windows and macOS")


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
    "QuartzInputController",
    "SendInputController",
    "create_system_input_controller",
    "mac_virtual_key_code",
    "WindowsInputController",
    "virtual_key_code",
]
