"""Layered abort for scripts that synthesize input into Rust.

Every tool in this directory drives the real mouse and keyboard, so each one
needs more than one way to be stopped by a person who is watching it go wrong.
``Guard.check()`` is called before every click, keystroke, and stroke, and any
single trigger ends the run immediately:

* Escape held down
* Rust stops being the focused window (so stray input cannot land elsewhere)
* the mouse moves somewhere the script did not put it - a hand on the mouse
* a wall-clock deadline, so nothing can run away unattended

The mouse rule is the reason every helper reports where it left the cursor:
comparing the real position against the last commanded one is what separates a
person grabbing the mouse from the script's own movement.
"""

from __future__ import annotations

import ctypes
import time

from app.screen import ForegroundRequirement, foreground_window_matches


RUST = ForegroundRequirement(title_contains="Rust", executable="RustClient.exe")

VK_ESCAPE = 0x1B

# Windows reports a click's own travel a few pixels late, so a tolerance below
# this reads the script's own cursor as a person's hand.
DRIFT_TOLERANCE_PIXELS = 60


class Aborted(SystemExit):
    """Raised the moment any stop condition trips."""


class Guard:
    """Stop conditions shared by every tool that emits real input."""

    def __init__(self, controller, *, budget_seconds: float = 90.0) -> None:
        self.input = controller
        self.deadline = time.monotonic() + budget_seconds
        self.budget_seconds = budget_seconds
        self._commanded: tuple[int, int] | None = None

    def commanded(self, x: float, y: float) -> None:
        """Record where the script just put the cursor."""

        self._commanded = (int(round(x)), int(round(y)))

    def check(self) -> None:
        if ctypes.windll.user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
            raise Aborted("Stopped: Escape")
        if time.monotonic() > self.deadline:
            raise Aborted(f"Stopped: exceeded its {self.budget_seconds:.0f}s budget")
        if not foreground_window_matches(RUST):
            raise Aborted("Stopped: Rust is no longer the focused window")
        if self._commanded is not None:
            actual = self.input.get_cursor_position()
            drift = max(
                abs(actual[0] - self._commanded[0]), abs(actual[1] - self._commanded[1])
            )
            if drift > DRIFT_TOLERANCE_PIXELS:
                raise Aborted(
                    f"Stopped: the mouse moved {drift}px from where the script left it"
                )

    # ------------------------------------------------------- guarded input

    def click(self, x: float, y: float, *, settle: float = 0.12) -> None:
        self.check()
        self.input.click(round(x), round(y))
        self.commanded(x, y)
        time.sleep(settle)

    def press(self, key: str) -> None:
        self.check()
        self.input.press_key(key)

    def drag(self, start, end, *, duration_seconds: float = 0.35) -> None:
        self.check()
        self.input.drag(start, end, duration_seconds=duration_seconds, step_pixels=3.0)
        self.commanded(*end)
        time.sleep(0.12)

    def park(self, point, *, settle: float = 0.45) -> None:
        """Move the cursor somewhere harmless and let the frame settle."""

        self.check()
        self.input.move_mouse(*point)
        self.commanded(*point)
        time.sleep(settle)


def countdown(seconds: int, message: str = "starting") -> None:
    """Give a person time to focus Rust, honouring Escape throughout."""

    for remaining in range(seconds, 0, -1):
        print(f"  {message} in {remaining}...", flush=True)
        time.sleep(1.0)
        if ctypes.windll.user32.GetAsyncKeyState(VK_ESCAPE) & 0x8000:
            raise Aborted("Stopped: Escape")


__all__ = ["Aborted", "Guard", "RUST", "countdown"]
