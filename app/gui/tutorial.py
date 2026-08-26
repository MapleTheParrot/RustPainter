"""Small, replayable getting-started guide for first-time users.

The copy and optional artwork live in data-only ``TutorialStep`` objects so a
future screenshot can be added without rebuilding the dialog.  Drop the named
PNG into ``assets/ui`` and the corresponding step will display it.
"""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .assets import asset_path


TUTORIAL_VERSION = 1


@dataclass(frozen=True, slots=True)
class TutorialStep:
    title: str
    body: str
    image_asset: str | None = None


TUTORIAL_STEPS: tuple[TutorialStep, ...] = (
    TutorialStep(
        "Open the sign in Rust",
        "Open the sign's painting screen. Rust initially shows basic colour "
        "swatches, so toggle Adaptive Palette until you can see the large "
        "gradient colour box and the vertical rainbow hue bar.",
        "tutorial-adaptive-palette",
    ),
    TutorialStep(
        "Prepare RustPainter",
        "Under Prepare Rust, set the canvas, colour box, and hue bar. Keep the "
        "painting screen open and stationary while you mark them. RustPainter "
        "remembers these areas for the sign profile.",
        "tutorial-prepare-rust",
    ),
    TutorialStep(
        "Add your artwork",
        "Choose an image or drag one into RustPainter, then check the Rust "
        "preview. The defaults are ready for a first paint.",
        "tutorial-add-artwork",
    ),
    TutorialStep(
        "Start with a quick test",
        "Try a small image first. Click Paint, switch back to Rust during the "
        "countdown, and keep F9 ready to pause or F10 to stop.",
        "tutorial-start-painting",
    ),
)


class GettingStartedDialog(QDialog):
    """A concise first-run guide that can also be opened from Preferences."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Getting started with RustPainter")
        self.setModal(True)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, True)
        self.setMinimumSize(600, 540)
        self.resize(680, 650)

        outer = QVBoxLayout(self)
        outer.setContentsMargins(18, 18, 18, 14)
        outer.setSpacing(12)

        title = QLabel("Paint your first Rust sign")
        title.setObjectName("pageTitle")
        intro = QLabel(
            "RustPainter only needs a few things before it can place your artwork."
        )
        intro.setObjectName("muted")
        intro.setWordWrap(True)
        outer.addWidget(title)
        outer.addWidget(intro)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        content = QWidget()
        steps_layout = QVBoxLayout(content)
        steps_layout.setContentsMargins(0, 0, 8, 0)
        steps_layout.setSpacing(10)

        for number, step in enumerate(TUTORIAL_STEPS, 1):
            steps_layout.addWidget(self._step_card(number, step))
        steps_layout.addStretch(1)
        scroll.setWidget(content)
        outer.addWidget(scroll, 1)

        reminder = QLabel(
            "You can open this guide again from Preferences → Show tutorial again."
        )
        reminder.setObjectName("muted")
        reminder.setWordWrap(True)
        close_button = QPushButton("Got it")
        close_button.setDefault(True)
        close_button.clicked.connect(self.accept)
        footer = QHBoxLayout()
        footer.addWidget(reminder, 1)
        footer.addWidget(close_button)
        outer.addLayout(footer)

    @staticmethod
    def _step_card(number: int, step: TutorialStep) -> QFrame:
        card = QFrame()
        card.setObjectName("detailsPanel")
        layout = QHBoxLayout(card)
        layout.setContentsMargins(14, 12, 14, 12)
        layout.setSpacing(12)

        badge = QLabel(str(number))
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        badge.setFixedSize(30, 30)
        badge.setStyleSheet(
            "border-radius: 15px; background: #d76524; color: #fff4e7; "
            "font-weight: 800;"
        )
        layout.addWidget(badge, 0, Qt.AlignmentFlag.AlignTop)

        copy_layout = QVBoxLayout()
        copy_layout.setSpacing(4)
        heading = QLabel(step.title)
        heading.setObjectName("sectionTitle")
        body = QLabel(step.body)
        body.setWordWrap(True)
        copy_layout.addWidget(heading)
        copy_layout.addWidget(body)

        # Image names are assigned now even though the files do not exist yet.
        # Adding a PNG with the matching name makes it appear automatically.
        if step.image_asset:
            path = asset_path(step.image_asset)
            if path.is_file():
                image = QLabel()
                image.setAlignment(Qt.AlignmentFlag.AlignCenter)
                pixmap = QPixmap(str(path))
                if not pixmap.isNull():
                    image.setPixmap(
                        pixmap.scaledToWidth(
                            520, Qt.TransformationMode.SmoothTransformation
                        )
                    )
                    copy_layout.addWidget(image)

        layout.addLayout(copy_layout, 1)
        return card


__all__ = [
    "GettingStartedDialog",
    "TUTORIAL_STEPS",
    "TUTORIAL_VERSION",
    "TutorialStep",
]
