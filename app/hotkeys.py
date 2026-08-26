"""Configurable global hotkeys via ``RegisterHotKey`` on Windows."""

from __future__ import annotations

import ctypes
import logging
import os
import threading
from dataclasses import dataclass
from typing import Callable, Mapping

from .input_controller import virtual_key_code


LOGGER = logging.getLogger("rust_painter.hotkeys")


_MODIFIER_VALUES = {
    "ALT": 0x0001,
    "CTRL": 0x0002,
    "CONTROL": 0x0002,
    "SHIFT": 0x0004,
    "WIN": 0x0008,
    "WINDOWS": 0x0008,
}
MOD_NOREPEAT = 0x4000
WM_HOTKEY = 0x0312
WM_QUIT = 0x0012


class HotkeyRegistrationError(RuntimeError):
    pass


# RegisterHotKey reports a bare code that tells the user nothing.  1409 is by
# far the common one: some other running program already owns the combination
# system-wide, and Windows gives it to whoever asked first.
_WINDOWS_HOTKEY_ERRORS = {
    1409: "another running program already owns it",
    87: "Windows rejected the key combination",
}


def _registration_failure_detail(error_code: int) -> str:
    reason = _WINDOWS_HOTKEY_ERRORS.get(error_code)
    if reason is None:
        return f"Windows error {error_code or 'unknown'}"
    return f"{reason} (Windows error {error_code})"


@dataclass(frozen=True, slots=True)
class HotkeySpec:
    key: int | str
    modifiers: tuple[str, ...] = ()

    @classmethod
    def parse(cls, value: "HotkeySpec | int | str") -> "HotkeySpec":
        if isinstance(value, cls):
            return value
        if isinstance(value, int):
            return cls(value)
        parts = [part.strip() for part in str(value).split("+") if part.strip()]
        if not parts:
            raise ValueError("Hotkey cannot be empty")
        return cls(parts[-1], tuple(part.upper() for part in parts[:-1]))

    @property
    def virtual_key(self) -> int:
        return virtual_key_code(self.key)

    @property
    def modifier_mask(self) -> int:
        result = MOD_NOREPEAT
        for modifier in self.modifiers:
            try:
                result |= _MODIFIER_VALUES[modifier.upper()]
            except KeyError as exc:
                raise ValueError(f"Unsupported hotkey modifier: {modifier!r}") from exc
        return result

    def __str__(self) -> str:
        pieces = [*self.modifiers, str(self.key).upper()]
        return "+".join(pieces)


# Compact laptop keyboards put F5-F12 behind an Fn key that the keyboard
# firmware consumes, so those presses never reach RegisterHotKey and the app
# looks unusable.  Offer modifier combos that every keyboard can produce.
# Ctrl+Alt+letter avoids both Rust's bindings and the common Windows shortcuts.
FUNCTION_KEY_CHOICES: tuple[str, ...] = tuple(f"F{number}" for number in range(5, 13))
MODIFIER_KEY_CHOICES: tuple[str, ...] = tuple(
    f"CTRL+ALT+{letter}" for letter in ("S", "P", "X", "B", "N", "M")
)
SUPPORTED_HOTKEY_CHOICES: tuple[str, ...] = FUNCTION_KEY_CHOICES + MODIFIER_KEY_CHOICES


def normalize_hotkey(value: object) -> str:
    """Return the canonical spelling used by settings and the chooser."""

    return "+".join(part.strip() for part in str(value).strip().upper().split("+"))


def is_supported_hotkey(value: object) -> bool:
    return normalize_hotkey(value) in SUPPORTED_HOTKEY_CHOICES


@dataclass(frozen=True, slots=True)
class HotkeyBindings:
    start_resume: HotkeySpec | int | str = "F8"
    abort: HotkeySpec | int | str = "F10"

    def normalized(self) -> "HotkeyBindings":
        return HotkeyBindings(
            HotkeySpec.parse(self.start_resume),
            HotkeySpec.parse(self.abort),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, int | str | HotkeySpec]) -> "HotkeyBindings":
        return cls(
            start_resume=values.get("start_resume", "F8"),
            abort=values.get("abort", "F10"),
        )


if os.name == "nt":
    from ctypes import wintypes

    class _POINT(ctypes.Structure):
        _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))

    class _MSG(ctypes.Structure):
        _fields_ = (
            ("hwnd", wintypes.HWND),
            ("message", wintypes.UINT),
            ("wParam", wintypes.WPARAM),
            ("lParam", wintypes.LPARAM),
            ("time", wintypes.DWORD),
            ("pt", _POINT),
            ("lPrivate", wintypes.DWORD),
        )


class GlobalHotkeyManager:
    """Own a small Win32 message-loop thread for start/pause and abort hotkeys.

    Callbacks run on the hotkey thread. Qt clients should emit a signal (or use
    another queued bridge) instead of touching widgets directly.
    """

    _IDS = {"start_resume": 0xB100, "abort": 0xB102}

    def __init__(
        self,
        on_start_resume: Callable[[], None] | None = None,
        on_abort: Callable[[], None] | None = None,
        *,
        bindings: HotkeyBindings | Mapping[str, int | str | HotkeySpec] | None = None,
        on_error: Callable[[BaseException], None] | None = None,
    ) -> None:
        if bindings is None:
            resolved_bindings = HotkeyBindings()
        elif isinstance(bindings, HotkeyBindings):
            resolved_bindings = bindings
        else:
            resolved_bindings = HotkeyBindings.from_mapping(bindings)
        self.bindings = resolved_bindings.normalized()
        self._callbacks: dict[str, Callable[[], None] | None] = {
            "start_resume": on_start_resume,
            "abort": on_abort,
        }
        self._on_error = on_error
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._started = threading.Event()
        self._stop_requested = threading.Event()
        self._running = False
        self._startup_error: BaseException | None = None
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return os.name == "nt"

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def startup_error(self) -> BaseException | None:
        return self._startup_error

    def set_callbacks(
        self,
        *,
        on_start_resume: Callable[[], None] | None = None,
        on_abort: Callable[[], None] | None = None,
    ) -> None:
        with self._lock:
            self._callbacks.update(
                start_resume=on_start_resume,
                abort=on_abort,
            )

    def start(self, timeout: float = 2.0) -> bool:
        """Register hotkeys; return ``False`` on non-Windows or a conflict."""

        if not self.available:
            LOGGER.info(
                "Global hotkeys are unavailable on this platform; continuing without them"
            )
            return False
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return self._running
            self._started.clear()
            self._stop_requested.clear()
            self._startup_error = None
            self._thread = threading.Thread(
                target=self._message_loop,
                name="RustPainterHotkeys",
                daemon=True,
            )
            self._thread.start()
        if not self._started.wait(timeout):
            error = HotkeyRegistrationError("Timed out starting the global hotkey thread")
            self._startup_error = error
            self._report_error(error)
            return False
        return self.running

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_requested.set()
        with self._lock:
            thread = self._thread
            thread_id = self._thread_id
        if os.name == "nt" and thread_id is not None:
            user32 = ctypes.WinDLL("user32", use_last_error=True)
            post = user32.PostThreadMessageW
            post.argtypes = (wintypes.DWORD, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM)
            post.restype = wintypes.BOOL
            post(thread_id, WM_QUIT, 0, 0)
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)
            if thread.is_alive():
                error = HotkeyRegistrationError(
                    "Timed out stopping the global hotkey message loop"
                )
                self._startup_error = error
                # Fail closed even if Windows has not yet acknowledged WM_QUIT.
                # A caller must not treat an unresponsive fail-safe thread as a
                # healthy emergency-abort binding.
                with self._lock:
                    self._running = False
                self._report_error(error)

    close = stop

    def _report_error(self, error: BaseException) -> None:
        LOGGER.error("Global hotkey error: %s", error)
        if self._on_error is not None:
            try:
                self._on_error(error)
            except Exception:
                LOGGER.exception("Hotkey error callback failed")

    def _binding_items(self) -> tuple[tuple[str, HotkeySpec], ...]:
        normalized = self.bindings.normalized()
        return (
            ("start_resume", normalized.start_resume),
            ("abort", normalized.abort),
        )  # type: ignore[return-value]

    def _message_loop(self) -> None:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        register = user32.RegisterHotKey
        register.argtypes = (wintypes.HWND, ctypes.c_int, wintypes.UINT, wintypes.UINT)
        register.restype = wintypes.BOOL
        unregister = user32.UnregisterHotKey
        unregister.argtypes = (wintypes.HWND, ctypes.c_int)
        unregister.restype = wintypes.BOOL
        get_message = user32.GetMessageW
        get_message.argtypes = (ctypes.POINTER(_MSG), wintypes.HWND, wintypes.UINT, wintypes.UINT)
        get_message.restype = ctypes.c_int
        current_thread_id = kernel32.GetCurrentThreadId
        current_thread_id.argtypes = ()
        current_thread_id.restype = wintypes.DWORD

        registered_ids: list[int] = []
        by_id = {identifier: name for name, identifier in self._IDS.items()}
        try:
            self._thread_id = int(current_thread_id())
            for name, spec in self._binding_items():
                identifier = self._IDS[name]
                if not register(None, identifier, spec.modifier_mask, spec.virtual_key):
                    error_code = ctypes.get_last_error()
                    raise HotkeyRegistrationError(
                        f"Could not register {name} hotkey {spec}: "
                        f"{_registration_failure_detail(error_code)}. "
                        "Choose a different hotkey in Preferences."
                    )
                registered_ids.append(identifier)
            with self._lock:
                self._running = True
            LOGGER.info(
                "Global hotkeys registered: start/pause=%s abort=%s",
                self.bindings.start_resume,
                self.bindings.abort,
            )
            self._started.set()

            message = _MSG()
            while not self._stop_requested.is_set():
                result = get_message(ctypes.byref(message), None, 0, 0)
                if result == 0:
                    break
                if result == -1:
                    raise ctypes.WinError(ctypes.get_last_error() or 1)
                if message.message != WM_HOTKEY:
                    continue
                name = by_id.get(int(message.wParam))
                if name is None:
                    continue
                with self._lock:
                    callback = self._callbacks.get(name)
                if callback is not None:
                    try:
                        callback()
                    except Exception as exc:
                        LOGGER.exception("%s hotkey callback failed", name)
                        # A failed callback makes the emergency control path
                        # untrustworthy. Tear down every registration and make
                        # real-input clients fail closed.
                        raise HotkeyRegistrationError(
                            f"{name} hotkey callback failed"
                        ) from exc
        except BaseException as exc:
            self._startup_error = exc
            # Publish the failed state before notifying clients.  In
            # particular, the GUI uses ``running`` to fail closed when the
            # emergency-abort registration/message loop dies at runtime.
            with self._lock:
                self._running = False
            self._report_error(exc)
            self._started.set()
        finally:
            unexpected_exit = (
                not self._stop_requested.is_set() and self._startup_error is None
            )
            for identifier in registered_ids:
                unregister(None, identifier)
            with self._lock:
                self._running = False
                self._thread_id = None
            self._started.set()
            if unexpected_exit:
                error = HotkeyRegistrationError(
                    "Global hotkey message loop exited unexpectedly"
                )
                self._startup_error = error
                self._report_error(error)

    def __enter__(self) -> "GlobalHotkeyManager":
        self.start()
        return self

    def __exit__(self, *_exc_info: object) -> None:
        self.stop()


HotkeyManager = GlobalHotkeyManager
DEFAULT_HOTKEYS = HotkeyBindings()


__all__ = [
    "DEFAULT_HOTKEYS",
    "GlobalHotkeyManager",
    "HotkeyBindings",
    "HotkeyManager",
    "HotkeyRegistrationError",
    "HotkeySpec",
]
