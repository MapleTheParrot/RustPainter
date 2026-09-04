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
_CANONICAL_MODIFIERS = {
    "CONTROL": "CTRL",
    "WINDOWS": "WIN",
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
        parts = [part.strip().upper() for part in str(value).split("+") if part.strip()]
        if not parts:
            raise ValueError("Hotkey cannot be empty")
        modifiers = tuple(_CANONICAL_MODIFIERS.get(part, part) for part in parts[:-1])
        if len(set(modifiers)) != len(modifiers):
            raise ValueError("A hotkey cannot repeat a modifier")
        if any(modifier not in _MODIFIER_VALUES for modifier in modifiers):
            unsupported = next(
                modifier for modifier in modifiers if modifier not in _MODIFIER_VALUES
            )
            raise ValueError(f"Unsupported hotkey modifier: {unsupported!r}")
        key = _CANONICAL_MODIFIERS.get(parts[-1], parts[-1])
        if key in _MODIFIER_VALUES:
            raise ValueError("A hotkey must include a non-modifier key")
        spec = cls(key, modifiers)
        # Validate the key name before a bad value reaches the hotkey thread.
        spec.virtual_key
        return spec

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


def normalize_hotkey(value: object) -> str:
    """Return the canonical spelling used by settings and the recorder."""

    return str(HotkeySpec.parse(str(value).strip()))


def is_supported_hotkey(value: object) -> bool:
    """Return whether a hotkey can be represented by ``RegisterHotKey``."""

    try:
        spec = HotkeySpec.parse(str(value).strip())
        return 0 < spec.virtual_key < 0xFF
    except (TypeError, ValueError):
        return False


@dataclass(frozen=True, slots=True)
class HotkeyBindings:
    start_resume: HotkeySpec | int | str = "CTRL+ALT+S"
    abort: HotkeySpec | int | str = "CTRL+ALT+X"
    anti_afk: HotkeySpec | int | str = "CTRL+ALT+K"

    def normalized(self) -> "HotkeyBindings":
        return HotkeyBindings(
            HotkeySpec.parse(self.start_resume),
            HotkeySpec.parse(self.abort),
            HotkeySpec.parse(self.anti_afk),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, int | str | HotkeySpec]) -> "HotkeyBindings":
        return cls(
            start_resume=values.get("start_resume", "CTRL+ALT+S"),
            abort=values.get("abort", "CTRL+ALT+X"),
            anti_afk=values.get("anti_afk", "CTRL+ALT+K"),
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

    class _KBDLLHOOKSTRUCT(ctypes.Structure):
        _fields_ = (
            ("vkCode", wintypes.DWORD),
            ("scanCode", wintypes.DWORD),
            ("flags", wintypes.DWORD),
            ("time", wintypes.DWORD),
            ("dwExtraInfo", ctypes.c_size_t),
        )


class GlobalHotkeyManager:
    """Own a small Win32 message-loop thread for start/pause and abort hotkeys.

    Callbacks run on the hotkey thread. Qt clients should emit a signal (or use
    another queued bridge) instead of touching widgets directly.
    """

    _IDS = {"start_resume": 0xB100, "abort": 0xB102, "anti_afk": 0xB103}

    def __init__(
        self,
        on_start_resume: Callable[[], None] | None = None,
        on_abort: Callable[[], None] | None = None,
        on_anti_afk: Callable[[], None] | None = None,
        on_movement: Callable[[], None] | None = None,
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
            "anti_afk": on_anti_afk,
        }
        self._on_movement = on_movement
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
        on_anti_afk: Callable[[], None] | None = None,
    ) -> None:
        with self._lock:
            self._callbacks.update(
                start_resume=on_start_resume,
                abort=on_abort,
                anti_afk=on_anti_afk,
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
            ("anti_afk", normalized.anti_afk),
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

        hook = None
        hook_proc = None
        unhook = None
        # RegisterHotKey deliberately consumes its combination; W/A/S/D must
        # remain ordinary Rust movement input, so observe them with a passive
        # low-level hook instead.  It never returns a non-zero value.
        if hasattr(user32, "SetWindowsHookExW") and hasattr(user32, "CallNextHookEx"):
            set_hook = user32.SetWindowsHookExW
            set_hook.argtypes = (ctypes.c_int, ctypes.c_void_p, wintypes.HINSTANCE, wintypes.DWORD)
            set_hook.restype = wintypes.HHOOK
            call_next = user32.CallNextHookEx
            call_next.argtypes = (wintypes.HHOOK, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM)
            call_next.restype = ctypes.c_long
            unhook = getattr(user32, "UnhookWindowsHookEx", None)
            if unhook is not None:
                unhook.argtypes = (wintypes.HHOOK,)
                unhook.restype = wintypes.BOOL
            hook_callback_type = ctypes.WINFUNCTYPE(
                ctypes.c_long, ctypes.c_int, wintypes.WPARAM, wintypes.LPARAM
            )

            @hook_callback_type
            def movement_hook(code, w_param, l_param):
                if code >= 0 and int(w_param) in (0x0100, 0x0104):  # key down / system key down
                    event = ctypes.cast(l_param, ctypes.POINTER(_KBDLLHOOKSTRUCT)).contents
                    if event.vkCode in (0x57, 0x41, 0x53, 0x44):  # W, A, S, D
                        callback = self._on_movement
                        if callback is not None:
                            try:
                                callback()
                            except Exception:
                                LOGGER.exception("Movement callback failed")
                return call_next(hook, code, w_param, l_param)

            hook_proc = movement_hook  # Keep the ctypes callback alive.
            hook = set_hook(13, hook_proc, None, 0)  # WH_KEYBOARD_LL

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
                "Global hotkeys registered: start/pause=%s abort=%s anti-AFK=%s",
                self.bindings.start_resume,
                self.bindings.abort,
                self.bindings.anti_afk,
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
            if hook and unhook is not None:
                unhook(hook)
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
