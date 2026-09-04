from __future__ import annotations

import os
from typing import Any, Callable

import pytest

from app.hotkeys import (
    DEFAULT_HOTKEYS,
    MOD_NOREPEAT,
    GlobalHotkeyManager,
    HotkeyRegistrationError,
    HotkeySpec,
)
# These two exercise the Win32 message loop by patching ctypes.WinDLL, which
# only exists on Windows.
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


def test_default_hotkeys_do_not_require_function_keys() -> None:
    assert DEFAULT_HOTKEYS.start_resume == "CTRL+ALT+S"
    assert DEFAULT_HOTKEYS.abort == "CTRL+ALT+X"
    assert DEFAULT_HOTKEYS.anti_afk == "CTRL+ALT+K"


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("ctrl+alt+q", "CTRL+ALT+Q"),
        ("shift+home", "SHIFT+HOME"),
        ("F24", "F24"),
        ("ctrl+semicolon", "CTRL+SEMICOLON"),
        ("alt+vk_ab", "ALT+VK_AB"),
    ],
)
def test_recordable_hotkeys_resolve(value: str, expected: str) -> None:
    from app.hotkeys import is_supported_hotkey, normalize_hotkey

    assert is_supported_hotkey(value)
    assert normalize_hotkey(value) == expected
    spec = HotkeySpec.parse(value)
    assert spec.virtual_key > 0
    assert spec.modifier_mask >= MOD_NOREPEAT


@pytest.mark.parametrize("value", ["", "CTRL", "CTRL+CTRL+S", "CTRL+MOUSE1", "VK_00"])
def test_invalid_hotkeys_are_rejected(value: str) -> None:
    from app.hotkeys import is_supported_hotkey

    assert not is_supported_hotkey(value)


def test_registration_failure_explains_an_already_owned_hotkey() -> None:
    # RegisterHotKey's 1409 means some other program claimed the key first.
    # The bare code told users nothing about what to do next.
    from app.hotkeys import _registration_failure_detail

    detail = _registration_failure_detail(1409)

    assert "another running program already owns it" in detail
    assert "1409" in detail
    assert "Windows error 4242" == _registration_failure_detail(4242)
