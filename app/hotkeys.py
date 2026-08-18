"""Configurable global hotkeys: ``RegisterHotKey`` on Windows, a Quartz
listen-only event tap on macOS (requires the Accessibility permission)."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
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


@dataclass(frozen=True, slots=True)
class HotkeyBindings:
    start_resume: HotkeySpec | int | str = "F8"
    pause: HotkeySpec | int | str = "F9"
    abort: HotkeySpec | int | str = "F10"

    def normalized(self) -> "HotkeyBindings":
        return HotkeyBindings(
            HotkeySpec.parse(self.start_resume),
            HotkeySpec.parse(self.pause),
            HotkeySpec.parse(self.abort),
        )

    @classmethod
    def from_mapping(cls, values: Mapping[str, int | str | HotkeySpec]) -> "HotkeyBindings":
        return cls(
            start_resume=values.get("start_resume", "F8"),
            pause=values.get("pause", "F9"),
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
    """Own a small Win32 message-loop thread for start/pause/abort hotkeys.

    Callbacks run on the hotkey thread. Qt clients should emit a signal (or use
    another queued bridge) instead of touching widgets directly.
    """

    _IDS = {"start_resume": 0xB100, "pause": 0xB101, "abort": 0xB102}

    def __init__(
        self,
        on_start_resume: Callable[[], None] | None = None,
        on_pause: Callable[[], None] | None = None,
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
            "pause": on_pause,
            "abort": on_abort,
        }
        self._on_error = on_error
        self._thread: threading.Thread | None = None
        self._thread_id: int | None = None
        self._darwin_runloop: object | None = None
        self._started = threading.Event()
        self._stop_requested = threading.Event()
        self._running = False
        self._startup_error: BaseException | None = None
        self._lock = threading.RLock()

    @property
    def available(self) -> bool:
        return os.name == "nt" or sys.platform == "darwin"

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
        on_pause: Callable[[], None] | None = None,
        on_abort: Callable[[], None] | None = None,
    ) -> None:
        with self._lock:
            self._callbacks.update(
                start_resume=on_start_resume,
                pause=on_pause,
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
        if sys.platform == "darwin":
            with self._lock:
                runloop = self._darwin_runloop
            if runloop is not None:
                import Quartz

                Quartz.CFRunLoopStop(runloop)
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
            ("pause", normalized.pause),
            ("abort", normalized.abort),
        )  # type: ignore[return-value]

    def _message_loop(self) -> None:
        if sys.platform == "darwin":
            self._message_loop_darwin()
            return
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
                        f"Could not register {name} hotkey {spec} "
                        f"(Windows error {error_code or 'unknown'})"
                    )
                registered_ids.append(identifier)
            with self._lock:
                self._running = True
            LOGGER.info(
                "Global hotkeys registered: start/resume=%s pause=%s abort=%s",
                self.bindings.start_resume,
                self.bindings.pause,
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

    _DARWIN_MODIFIER_MASKS = {
        "ALT": 0x00080000,      # kCGEventFlagMaskAlternate (Option)
        "CTRL": 0x00040000,     # kCGEventFlagMaskControl
        "CONTROL": 0x00040000,
        "SHIFT": 0x00020000,    # kCGEventFlagMaskShift
        "WIN": 0x00100000,      # kCGEventFlagMaskCommand
        "WINDOWS": 0x00100000,
    }
    _DARWIN_ALL_MODIFIERS = 0x00080000 | 0x00040000 | 0x00020000 | 0x00100000

    def _darwin_bindings(self) -> tuple[tuple[str, int, int], ...]:
        """Resolve (name, mac keycode, required modifier flags) triples."""

        from .input_controller import mac_virtual_key_code

        resolved = []
        for name, spec in self._binding_items():
            mask = 0
            for modifier in spec.modifiers:
                try:
                    mask |= self._DARWIN_MODIFIER_MASKS[modifier.upper()]
                except KeyError as exc:
                    raise ValueError(
                        f"Unsupported hotkey modifier: {modifier!r}"
                    ) from exc
            resolved.append((name, mac_virtual_key_code(spec.key), mask))
        return tuple(resolved)

    def _message_loop_darwin(self) -> None:
        """Watch key-down events with a listen-only Quartz event tap.

        Mirrors the Windows loop's contract: publish ``running`` before
        signalling ``_started``, fail closed when a callback raises, and
        report an unexpected exit.  Creating the tap fails (returns ``None``)
        until the user grants this app the Accessibility permission.
        """

        import Quartz

        bindings = self._darwin_bindings()
        failure: dict[str, BaseException] = {}
        state: dict[str, object] = {"tap": None}

        def handle_event(_proxy: object, event_type: int, event: object, _refcon: object) -> object:
            if event_type in (
                Quartz.kCGEventTapDisabledByTimeout,
                Quartz.kCGEventTapDisabledByUserInput,
            ):
                tap = state["tap"]
                if tap is not None:
                    Quartz.CGEventTapEnable(tap, True)
                return event
            if event_type != Quartz.kCGEventKeyDown:
                return event
            if Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventAutorepeat
            ):
                return event  # parity with Windows MOD_NOREPEAT
            keycode = Quartz.CGEventGetIntegerValueField(
                event, Quartz.kCGKeyboardEventKeycode
            )
            flags = Quartz.CGEventGetFlags(event) & self._DARWIN_ALL_MODIFIERS
            for name, wanted_code, wanted_mask in bindings:
                if keycode != wanted_code or flags != wanted_mask:
                    continue
                with self._lock:
                    callback = self._callbacks.get(name)
                if callback is not None:
                    try:
                        callback()
                    except Exception as exc:
                        LOGGER.exception("%s hotkey callback failed", name)
                        # A failed callback makes the emergency control path
                        # untrustworthy; tear down and fail closed.
                        failure["error"] = HotkeyRegistrationError(
                            f"{name} hotkey callback failed"
                        )
                        failure["error"].__cause__ = exc
                        Quartz.CFRunLoopStop(Quartz.CFRunLoopGetCurrent())
                break
            return event

        try:
            tap = Quartz.CGEventTapCreate(
                Quartz.kCGSessionEventTap,
                Quartz.kCGHeadInsertEventTap,
                Quartz.kCGEventTapOptionListenOnly,
                Quartz.CGEventMaskBit(Quartz.kCGEventKeyDown),
                handle_event,
                None,
            )
            if tap is None:
                raise HotkeyRegistrationError(
                    "Could not create the macOS key event tap. Grant this app "
                    "the Accessibility permission under System Settings > "
                    "Privacy & Security > Accessibility, then restart it."
                )
            state["tap"] = tap
            source = Quartz.CFMachPortCreateRunLoopSource(None, tap, 0)
            runloop = Quartz.CFRunLoopGetCurrent()
            Quartz.CFRunLoopAddSource(runloop, source, Quartz.kCFRunLoopCommonModes)
            Quartz.CGEventTapEnable(tap, True)
            with self._lock:
                self._darwin_runloop = runloop
                self._running = True
            LOGGER.info(
                "Global hotkeys registered: start/resume=%s pause=%s abort=%s",
                self.bindings.start_resume,
                self.bindings.pause,
                self.bindings.abort,
            )
            self._started.set()
            Quartz.CFRunLoopRun()
            if "error" in failure:
                raise failure["error"]
        except BaseException as exc:
            self._startup_error = exc
            with self._lock:
                self._running = False
            self._report_error(exc)
            self._started.set()
        finally:
            unexpected_exit = (
                not self._stop_requested.is_set() and self._startup_error is None
            )
            tap = state["tap"]
            if tap is not None:
                Quartz.CGEventTapEnable(tap, False)
            with self._lock:
                self._running = False
                self._darwin_runloop = None
            self._started.set()
            if unexpected_exit:
                error = HotkeyRegistrationError(
                    "Global hotkey event tap exited unexpectedly"
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
