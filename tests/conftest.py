"""Shared pytest safety guards.

GUI tests must explicitly decide how to answer modal message boxes.  Letting an
unexpected box open in a headless run blocks pytest forever, which also hides
the test that needs to be updated.
"""

from __future__ import annotations

import pytest
from PySide6.QtWidgets import QMessageBox


@pytest.fixture(autouse=True)
def reject_unexpected_message_boxes(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fail immediately when a test opens a QMessageBox it did not mock."""

    def unexpected(method: str):
        def fail(*args, **_kwargs):
            title = args[1] if len(args) > 1 else "<unknown title>"
            message = args[2] if len(args) > 2 else "<unknown message>"
            pytest.fail(
                f"Unexpected QMessageBox.{method}({title!r}, {message!r}); "
                "mock this dialog explicitly in the test",
                pytrace=False,
            )

        return fail

    for method in ("critical", "information", "question", "warning"):
        monkeypatch.setattr(QMessageBox, method, unexpected(method))
