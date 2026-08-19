from __future__ import annotations

import os
from typing import Any, Callable

import pytest

from app.hotkeys import (
    MOD_NOREPEAT,
    GlobalHotkeyManager,
    HotkeyRegistrationError,
    HotkeySpec,
)
from app.input_controller import mac_virtual_key_code

# These two exercise the Win32 message loop by patching ctypes.WinDLL, which
# only exists on Windows. The darwin event-tap path is covered separately by
# tests/test_macos_backends.py.
windows_only = pytest.mark.skipif(
    os.name != "nt", reason="Win32 hotkey message loop requires Windows"
)


class _NativeFunction:
    def __init__(self, implementation: Callable[..., Any]) -> None:
        object.__setattr__(self, "implementation", implementation)

    def __call__(self, *args: Any) -> Any:
        return self.implementation(*args)


@windows_only
def test_unexpected_message_loop_exit_is_reported_after_running_is_cleared(
    monkeypatch,
) -> None:
    states_seen_by_error_callback: list[bool] = []
    fake_user32 = type(
        "FakeUser32",
        (),
        {
            "RegisterHotKey": _NativeFunction(lambda *_args: 1),
            "UnregisterHotKey": _NativeFunction(lambda *_args: 1),
            "GetMessageW": _NativeFunction(lambda *_args: 0),
        },
    )()
    fake_kernel32 = type(
        "FakeKernel32",
        (),
        {"GetCurrentThreadId": _NativeFunction(lambda: 1234)},
    )()

    def fake_windll(name: str, **_kwargs: Any) -> Any:
        return fake_user32 if name == "user32" else fake_kernel32

    monkeypatch.setattr("app.hotkeys.ctypes.WinDLL", fake_windll)
    manager: GlobalHotkeyManager
    manager = GlobalHotkeyManager(
        on_error=lambda _error: states_seen_by_error_callback.append(manager.running)
    )

    manager._message_loop()

    assert manager.running is False
    assert isinstance(manager.startup_error, HotkeyRegistrationError)
    assert "exited unexpectedly" in str(manager.startup_error)
    assert states_seen_by_error_callback == [False]


@windows_only
def test_callback_failure_tears_down_hotkeys(monkeypatch) -> None:
    messages = [0x0312, 0]

    def get_message(message_pointer, *_args: Any) -> int:
        result = messages.pop(0)
        if result:
            message_pointer._obj.message = 0x0312
            message_pointer._obj.wParam = GlobalHotkeyManager._IDS["abort"]
        return result

    fake_user32 = type(
        "FakeUser32",
        (),
        {
            "RegisterHotKey": _NativeFunction(lambda *_args: 1),
            "UnregisterHotKey": _NativeFunction(lambda *_args: 1),
            "GetMessageW": _NativeFunction(get_message),
        },
    )()
    fake_kernel32 = type(
        "FakeKernel32",
        (),
        {"GetCurrentThreadId": _NativeFunction(lambda: 1234)},
    )()
    monkeypatch.setattr(
        "app.hotkeys.ctypes.WinDLL",
        lambda name, **_kwargs: fake_user32 if name == "user32" else fake_kernel32,
    )
    errors: list[BaseException] = []
    manager = GlobalHotkeyManager(
        on_abort=lambda: (_ for _ in ()).throw(RuntimeError("broken abort callback")),
        on_error=errors.append,
    )

    manager._message_loop()

    assert manager.running is False
    assert isinstance(manager.startup_error, HotkeyRegistrationError)
    assert "abort hotkey callback failed" in str(manager.startup_error)
    assert len(errors) == 1


def test_stop_timeout_marks_emergency_hotkeys_unhealthy() -> None:
    class HungThread:
        def is_alive(self) -> bool:
            return True

        def join(self, _timeout: float) -> None:
            return None

    errors: list[BaseException] = []
    manager = GlobalHotkeyManager(on_error=errors.append)
    manager._thread = HungThread()  # type: ignore[assignment]
    manager._thread_id = None
    manager._running = True

    manager.stop(timeout=0.0)

    assert manager.running is False
    assert isinstance(manager.startup_error, HotkeyRegistrationError)
    assert "Timed out stopping" in str(manager.startup_error)
    assert errors == [manager.startup_error]


def test_darwin_bindings_resolve_mac_keycodes_and_modifiers() -> None:
    manager = GlobalHotkeyManager(
        bindings={"start_resume": "F8", "pause": "CTRL+SHIFT+F9", "abort": "F10"}
    )
    resolved = dict(
        (name, (keycode, mask)) for name, keycode, mask in manager._darwin_bindings()
    )
    assert resolved["start_resume"] == (0x64, 0)
    assert resolved["abort"] == (0x6D, 0)
    keycode, mask = resolved["pause"]
    assert keycode == 0x65
    assert mask == (0x00040000 | 0x00020000)  # Control | Shift


def test_every_offered_hotkey_choice_resolves_on_both_platforms() -> None:
    # The chooser, the settings validator, and both key-code tables must agree,
    # or a selectable entry would fail to register at runtime.
    from app.hotkeys import SUPPORTED_HOTKEY_CHOICES, is_supported_hotkey

    assert len(set(SUPPORTED_HOTKEY_CHOICES)) == len(SUPPORTED_HOTKEY_CHOICES)
    assert any("+" in choice for choice in SUPPORTED_HOTKEY_CHOICES)
    for choice in SUPPORTED_HOTKEY_CHOICES:
        assert is_supported_hotkey(choice.lower())
        spec = HotkeySpec.parse(choice)
        assert str(spec) == choice
        assert spec.virtual_key > 0
        assert spec.modifier_mask >= MOD_NOREPEAT
        assert mac_virtual_key_code(spec.key) >= 0
