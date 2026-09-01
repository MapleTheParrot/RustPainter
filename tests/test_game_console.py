from __future__ import annotations

import pytest

from app.game_console import (
    apply_console_commands,
    format_ui_scale,
    ui_scale_command,
    validate_console_key,
    validate_ui_scale,
)
from app.input_controller import MockInputController


def test_ui_scale_command_is_stable_and_validated() -> None:
    assert format_ui_scale(0.5) == "0.5"
    assert format_ui_scale(1.0) == "1"
    assert ui_scale_command(0.75) == "graphics.uiscale 0.75"
    for value in (0.49, 1.01, float("nan"), True, "not a number"):
        with pytest.raises(ValueError):
            validate_ui_scale(value)


def test_console_key_accepts_safe_function_key_chords() -> None:
    assert str(validate_console_key("ctrl+f1")) == "CTRL+F1"
    with pytest.raises(ValueError):
        validate_console_key("WIN+F1")


def test_console_commands_open_type_enter_and_close() -> None:
    controller = MockInputController()

    typed = apply_console_commands(
        controller,
        ("graphics.uiscale 0.5",),
        console_key="CTRL+F1",
    )

    assert typed == ("graphics.uiscale 0.5",)
    actions = [(event.kind, event.value) for event in controller.events]
    assert actions[:4] == [
        ("key_down", "CTRL"),
        ("key_down", "F1"),
        ("key_up", "F1"),
        ("key_up", "CTRL"),
    ]
    typed_text = "".join(
        str(value) for kind, value in actions if kind == "type_text"
    )
    assert typed_text == "graphics.uiscale 0.5"
    assert actions[-6:] == [
        ("key_down", "ENTER"),
        ("key_up", "ENTER"),
        ("key_down", "CTRL"),
        ("key_down", "F1"),
        ("key_up", "F1"),
        ("key_up", "CTRL"),
    ]


def test_console_is_closed_when_a_checkpoint_stops_the_pass() -> None:
    controller = MockInputController()
    calls = 0

    def checkpoint() -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("stop")

    with pytest.raises(RuntimeError, match="stop"):
        apply_console_commands(controller, ("graphics.uiscale 0.5",), checkpoint=checkpoint)

    f1_down = [
        event for event in controller.events if event.kind == "key_down" and event.value == "F1"
    ]
    assert len(f1_down) == 2


def test_abort_still_closes_console_when_rust_remains_foreground() -> None:
    controller = MockInputController()
    calls = 0

    def abort_checkpoint() -> None:
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise RuntimeError("abort")

    with pytest.raises(RuntimeError, match="abort"):
        apply_console_commands(
            controller,
            ("graphics.uiscale 0.5",),
            checkpoint=abort_checkpoint,
            close_checkpoint=lambda: None,
        )

    f1_down = [
        event for event in controller.events if event.kind == "key_down" and event.value == "F1"
    ]
    assert len(f1_down) == 2
