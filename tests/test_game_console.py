from __future__ import annotations

import pytest

from app.game_console import (
    CONSOLE_OPEN_SECONDS,
    COMMAND_GAP_SECONDS,
    PaintConVars,
    apply_convars,
    format_convar_value,
    normalize_console_key,
    validate_console_key,
)
from app.input_controller import MockInputController


def test_convar_values_are_spelled_the_way_rusts_console_shows_them() -> None:
    assert format_convar_value(False) == "False"
    assert format_convar_value(True) == "True"
    assert format_convar_value(3) == "3"
    assert format_convar_value(1.0) == "1"
    assert format_convar_value(0.01) == "0.01"
    assert format_convar_value(0.25) == "0.25"
    assert format_convar_value(100.0) == "100"
    with pytest.raises(ValueError):
        format_convar_value(float("nan"))


def test_default_convars_pin_everything_painting_assumes() -> None:
    """Opacity, tool and side are pinned outright; brush and spacing follow settings."""

    assert PaintConVars().commands() == (
        "paint.brushopacity 1",
        "paint.selectedtool 0",
        "paint.selectedbrush 3",
        "paint.leftsided False",
        "paint.brushspacing 0.01",
        "paint.maxbrushsize 100",
    )
    custom = PaintConVars(selected_brush=0, brush_spacing=0.25)
    assert "paint.selectedbrush 0" in custom.commands()
    assert "paint.brushspacing 0.25" in custom.commands()


@pytest.mark.parametrize(
    "values",
    [
        {"selected_brush": -1},
        {"selected_brush": 99},
        {"brush_spacing": 1.5},
        {"brush_spacing": float("inf")},
        {"brush_opacity": 0.0},
        {"max_brush_size": 0.5},
    ],
)
def test_convars_reject_values_rust_would_not_take(values: dict) -> None:
    with pytest.raises(ValueError):
        PaintConVars(**values)


def test_console_key_accepts_chords_and_rejects_the_windows_key() -> None:
    assert normalize_console_key(" ctrl + f1 ") == "CTRL+F1"
    spec = validate_console_key("ctrl+f1")
    assert spec.key == "F1" and spec.modifiers == ("CTRL",)
    assert validate_console_key("F1").modifiers == ()
    with pytest.raises(ValueError):
        validate_console_key("")
    with pytest.raises(ValueError):
        validate_console_key("WIN+F1")
    with pytest.raises(ValueError):
        validate_console_key("CTRL+NOSUCHKEY")


def test_apply_convars_opens_types_and_closes_the_console() -> None:
    controller = MockInputController()
    commands = PaintConVars().commands()

    typed = apply_convars(controller, commands, console_key="CTRL+F1")

    assert typed == commands
    values = [(event.kind, event.value) for event in controller.events]
    # The chord: Ctrl held around F1, released afterwards.
    assert values[:3] == [("key_down", "CTRL"), ("key_down", "F1"), ("key_up", "F1")]
    assert values[3] == ("key_up", "CTRL")
    body = values[4:-4]
    assert body == [
        item
        for command in commands
        for item in (
            ("type_text", command),
            ("key_down", "ENTER"),
            ("key_up", "ENTER"),
        )
    ]
    # Closed with the same chord, and nothing left held.
    assert values[-4:] == values[:4]
    assert not controller.held_keys
    assert not controller.held_buttons


def test_apply_convars_waits_only_for_real_input() -> None:
    waits: list[float] = []
    controller = MockInputController()
    apply_convars(controller, ("paint.brushopacity 1",), sleep=waits.append)
    assert waits == []

    class RealLooking(MockInputController):
        emits_real_input = True

    waits.clear()
    apply_convars(RealLooking(), ("a", "b"), sleep=waits.append)
    assert waits == [CONSOLE_OPEN_SECONDS, COMMAND_GAP_SECONDS, COMMAND_GAP_SECONDS]


def test_apply_convars_stops_at_the_checkpoint_and_still_closes_the_console() -> None:
    class Stop(RuntimeError):
        pass

    controller = MockInputController()
    calls = {"count": 0}

    def checkpoint() -> None:
        calls["count"] += 1
        if calls["count"] == 3:  # before the second command
            raise Stop

    with pytest.raises(Stop):
        apply_convars(controller, ("one", "two", "three"), checkpoint=checkpoint)

    typed = [event.value for event in controller.events if event.kind == "type_text"]
    assert typed == ["one"]
    # The console chord was pressed twice: once to open, once on the way out.
    f1_presses = [
        event for event in controller.events if event.kind == "key_down" and event.value == "F1"
    ]
    assert len(f1_presses) == 2
    assert not controller.held_keys
