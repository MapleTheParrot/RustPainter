"""Safe, narrowly scoped access to Rust's in-game console.

RustPainter only uses this for client settings that materially improve a paint
session.  Commands are typed as Unicode so keyboard layout does not change
punctuation, and callers provide a checkpoint that enforces the foreground and
emergency-stop guards between every input step.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Sequence

from .hotkeys import HotkeySpec
from .input_controller import InputController, virtual_key_code
from .paint_timing import KEY_GAP_SECONDS, KEY_HOLD_SECONDS


DEFAULT_CONSOLE_KEY = "F1"
CONSOLE_KEY_CHOICES: tuple[str, ...] = ("F1", "CTRL+F1", "SHIFT+F1", "ALT+F1")
_CONSOLE_MODIFIERS = frozenset({"CTRL", "CONTROL", "SHIFT", "ALT"})
CONSOLE_OPEN_SECONDS = 0.6
COMMAND_GAP_SECONDS = 0.25


def normalize_console_key(value: object) -> str:
    return "+".join(
        part.strip().upper() for part in str(value).strip().split("+") if part.strip()
    )


def validate_console_key(value: object) -> HotkeySpec:
    text = normalize_console_key(value)
    if not text:
        raise ValueError("The console key cannot be empty")
    spec = HotkeySpec.parse(text)
    for modifier in spec.modifiers:
        if modifier not in _CONSOLE_MODIFIERS:
            raise ValueError(f"Unsupported console key modifier: {modifier}")
    virtual_key_code(spec.key)
    return spec


def validate_ui_scale(value: object) -> float:
    if isinstance(value, bool):
        raise ValueError("Rust UI scale must be numeric")
    try:
        scale = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("Rust UI scale must be numeric") from exc
    if not math.isfinite(scale) or not 0.5 <= scale <= 1.0:
        raise ValueError("Rust UI scale must be between 0.5 and 1.0")
    return scale


def format_ui_scale(value: object) -> str:
    scale = validate_ui_scale(value)
    return f"{scale:.2f}".rstrip("0").rstrip(".")


def ui_scale_command(value: object) -> str:
    return f"graphics.uiscale {format_ui_scale(value)}"


def _press_chord(
    controller: InputController, spec: HotkeySpec, *, hold_seconds: float
) -> None:
    held: list[str] = []
    try:
        for modifier in spec.modifiers:
            controller.key_down(modifier)
            held.append(modifier)
        controller.press_key(spec.key, hold_seconds=hold_seconds)
    finally:
        for modifier in reversed(held):
            controller.key_up(modifier)


def apply_console_commands(
    controller: InputController,
    commands: Sequence[str],
    *,
    console_key: HotkeySpec | str = DEFAULT_CONSOLE_KEY,
    checkpoint: Callable[[], None] | None = None,
    close_checkpoint: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, ...]:
    """Open Rust's console, type commands, and close it even on failure."""

    spec = validate_console_key(console_key)
    real = bool(getattr(controller, "emits_real_input", True))
    hold = KEY_HOLD_SECONDS if real else 0.0
    gap = KEY_GAP_SECONDS if real else 0.0

    def check() -> None:
        if checkpoint is not None:
            checkpoint()

    def wait(seconds: float) -> None:
        if real and seconds > 0:
            sleep(seconds)

    check()
    _press_chord(controller, spec, hold_seconds=hold)
    wait(CONSOLE_OPEN_SECONDS)
    typed: list[str] = []
    try:
        for command in commands:
            for character in command:
                check()
                controller.type_text(character, hold_seconds=hold, gap_seconds=gap)
            check()
            controller.press_key("ENTER", hold_seconds=hold)
            typed.append(command)
            wait(COMMAND_GAP_SECONDS)
    finally:
        # Focus may have changed during the command. Never send even the
        # closing F1 chord into another application; the resulting error tells
        # the user the Rust console may still be open.
        if close_checkpoint is not None:
            close_checkpoint()
        else:
            check()
        _press_chord(controller, spec, hold_seconds=hold)
    return tuple(typed)


def set_ui_scale(
    controller: InputController,
    value: object,
    *,
    console_key: HotkeySpec | str = DEFAULT_CONSOLE_KEY,
    checkpoint: Callable[[], None] | None = None,
    close_checkpoint: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> str:
    command = ui_scale_command(value)
    apply_console_commands(
        controller,
        (command,),
        console_key=console_key,
        checkpoint=checkpoint,
        close_checkpoint=close_checkpoint,
        sleep=sleep,
    )
    return command


__all__ = [
    "CONSOLE_KEY_CHOICES",
    "DEFAULT_CONSOLE_KEY",
    "apply_console_commands",
    "format_ui_scale",
    "normalize_console_key",
    "set_ui_scale",
    "ui_scale_command",
    "validate_console_key",
    "validate_ui_scale",
]
