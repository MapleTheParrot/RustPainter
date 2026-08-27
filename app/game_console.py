"""Pin Rust's painting ConVars through the in-game console before a job.

The painting UI keeps several settings the app never sees and cannot read
back: the brush opacity, which brush and tool are selected, how densely a
drag is stamped, whether the UI sits on the left or the right of the screen,
and how far the Size field goes.  Every one of them is silently assumed by
the rest of the app - the brush model was measured on one brush at full
opacity, the calibrated rectangles are where the right-handed UI draws them,
the planner trusts the Size field to reach 100 - and a value left over from
someone's last session breaks that assumption without a single error.  A
half-opacity brush paints every colour wrong and the export audit then
marks the whole sign as mismatched.

Rust exposes all of them as ``paint.*`` console variables, so instead of
hoping they are right, they are typed into the console once, before the
sign is opened::

    paint.brushopacity 1
    paint.selectedtool 0
    paint.selectedbrush 3
    paint.leftsided False
    paint.brushspacing 0.01
    paint.maxbrushsize 100

This runs from the app's own Settings page, with the same foreground guard
as every other real-input action: nothing is typed unless the Rust window
is in front, and the whole thing stops the moment it is not.
"""

from __future__ import annotations

import logging
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from .brush_calibration import BRUSH_SIZE_MAX
from .hotkeys import HotkeySpec
from .input_controller import InputController, virtual_key_code
from .paint_timing import KEY_GAP_SECONDS, KEY_HOLD_SECONDS

LOGGER = logging.getLogger("rust_painter.console")

# Rust opens its console on F1.  Compact laptop keyboards that route the
# function row through Fn need a modifier to deliver the press, which is why
# the chord is a setting rather than a constant.
DEFAULT_CONSOLE_KEY = "F1"
CONSOLE_KEY_CHOICES: tuple[str, ...] = ("F1", "CTRL+F1", "SHIFT+F1", "ALT+F1")
_CONSOLE_MODIFIERS = frozenset({"CTRL", "CONTROL", "SHIFT", "ALT"})

# Rust's brush spacing runs 0..1 as a fraction of the brush diameter.  The
# app's drag speeds were tuned with stamps a hundredth of a brush apart, so
# that is what a fresh install pins; the game's own default is 0.25.
DEFAULT_BRUSH_SPACING = 0.01
# The brush index every profile shipped with this app was measured on.
DEFAULT_SELECTED_BRUSH = 3
MAX_SELECTED_BRUSH = 15

# How long the console gets to open before the first command is typed, and
# how long each command gets to be applied before the next.  The console is
# an ordinary UI panel and appears within a frame; the waits are generous
# because a command typed into the game world lands on the hotbar.
CONSOLE_OPEN_SECONDS = 0.6
COMMAND_GAP_SECONDS = 0.25
CHARACTER_GAP_SECONDS = KEY_GAP_SECONDS


def normalize_console_key(value: object) -> str:
    """The canonical spelling of a console chord, e.g. ``CTRL+F1``."""

    return "+".join(
        part.strip().upper() for part in str(value).strip().split("+") if part.strip()
    )


def validate_console_key(value: object) -> HotkeySpec:
    """Parse a console chord, rejecting anything the app cannot press.

    Modifiers are limited to Ctrl, Shift and Alt: the Windows key opens the
    Start menu on release, which is the last thing to do over a game.
    """

    text = normalize_console_key(value)
    if not text:
        raise ValueError("The console key cannot be empty")
    spec = HotkeySpec.parse(text)
    for modifier in spec.modifiers:
        if modifier not in _CONSOLE_MODIFIERS:
            raise ValueError(f"Unsupported console key modifier: {modifier}")
    virtual_key_code(spec.key)
    return spec


def format_convar_value(value: object) -> str:
    """Spell a value the way Rust's console shows it: ``False``, ``1``, ``0.25``."""

    if isinstance(value, bool):
        return "True" if value else "False"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("ConVar values must be finite")
        text = f"{value:.4f}".rstrip("0").rstrip(".")
        return text or "0"
    return str(value)


@dataclass(frozen=True, slots=True)
class PaintConVars:
    """The painting ConVars a job depends on, and the values it needs."""

    selected_brush: int = DEFAULT_SELECTED_BRUSH
    brush_spacing: float = DEFAULT_BRUSH_SPACING
    brush_opacity: float = 1.0
    selected_tool: int = 0
    left_sided: bool = False
    max_brush_size: float = BRUSH_SIZE_MAX

    def __post_init__(self) -> None:
        if isinstance(self.selected_brush, bool) or not (
            0 <= int(self.selected_brush) <= MAX_SELECTED_BRUSH
        ):
            raise ValueError(f"selected_brush must be between 0 and {MAX_SELECTED_BRUSH}")
        if not math.isfinite(self.brush_spacing) or not 0.0 <= self.brush_spacing <= 1.0:
            raise ValueError("brush_spacing must be between 0 and 1")
        if not math.isfinite(self.brush_opacity) or not 0.0 < self.brush_opacity <= 1.0:
            raise ValueError("brush_opacity must be between 0 (exclusive) and 1")
        if isinstance(self.selected_tool, bool) or int(self.selected_tool) < 0:
            raise ValueError("selected_tool must be a non-negative index")
        if not math.isfinite(self.max_brush_size) or self.max_brush_size < 1.0:
            raise ValueError("max_brush_size must be at least 1")

    def commands(self) -> tuple[str, ...]:
        """The console lines, in the order they are typed.

        Opacity and tool go first: they are the two whose wrong value ruins
        paint outright, so if the pass is interrupted they are the ones most
        likely to have landed.
        """

        return (
            f"paint.brushopacity {format_convar_value(float(self.brush_opacity))}",
            f"paint.selectedtool {format_convar_value(int(self.selected_tool))}",
            f"paint.selectedbrush {format_convar_value(int(self.selected_brush))}",
            f"paint.leftsided {format_convar_value(bool(self.left_sided))}",
            f"paint.brushspacing {format_convar_value(float(self.brush_spacing))}",
            f"paint.maxbrushsize {format_convar_value(float(self.max_brush_size))}",
        )


def _press_chord(
    controller: InputController, spec: HotkeySpec, *, hold_seconds: float
) -> None:
    """Press a key with its modifiers held, releasing them however it ends."""

    held: list[str] = []
    try:
        for modifier in spec.modifiers:
            controller.key_down(modifier)
            held.append(modifier)
        controller.press_key(spec.key, hold_seconds=hold_seconds)
    finally:
        for modifier in reversed(held):
            controller.key_up(modifier)


def apply_convars(
    controller: InputController,
    commands: Sequence[str],
    *,
    console_key: HotkeySpec | str = DEFAULT_CONSOLE_KEY,
    checkpoint: Callable[[], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> tuple[str, ...]:
    """Open the console, type each command, and close the console again.

    ``checkpoint`` runs before every keystroke that matters and may raise to
    stop the pass; the console is still closed on the way out so the game is
    left the way it was found.  Returns the commands that were typed in full.
    """

    spec = validate_console_key(console_key)
    real = bool(getattr(controller, "emits_real_input", True))
    hold = KEY_HOLD_SECONDS if real else 0.0
    gap = CHARACTER_GAP_SECONDS if real else 0.0

    def wait(seconds: float) -> None:
        if real and seconds > 0:
            sleep(seconds)

    def check() -> None:
        if checkpoint is not None:
            checkpoint()

    check()
    LOGGER.info("Opening Rust's console with %s", spec)
    _press_chord(controller, spec, hold_seconds=hold)
    wait(CONSOLE_OPEN_SECONDS)
    typed: list[str] = []
    try:
        for command in commands:
            check()
            controller.type_text(command, hold_seconds=hold, gap_seconds=gap)
            controller.press_key("ENTER", hold_seconds=hold)
            typed.append(command)
            LOGGER.info("Console: %s", command)
            wait(COMMAND_GAP_SECONDS)
    finally:
        # The same chord toggles the console shut.  Leaving it open would
        # swallow the user's next keystrokes, and if it never opened at all
        # this opens it, which shows plainly that the key is wrong.
        try:
            _press_chord(controller, spec, hold_seconds=hold)
        except Exception:  # the pass is already ending; report the first error
            LOGGER.exception("Could not close Rust's console")
    return tuple(typed)


__all__ = [
    "CONSOLE_KEY_CHOICES",
    "CONSOLE_OPEN_SECONDS",
    "COMMAND_GAP_SECONDS",
    "DEFAULT_BRUSH_SPACING",
    "DEFAULT_CONSOLE_KEY",
    "DEFAULT_SELECTED_BRUSH",
    "MAX_SELECTED_BRUSH",
    "PaintConVars",
    "apply_convars",
    "format_convar_value",
    "normalize_console_key",
    "validate_console_key",
]
