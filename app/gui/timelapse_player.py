"""Watch a recorded timelapse session without leaving the application.

Frames are decoded on demand rather than held in memory: a long paint job can
leave thousands of full-canvas PNGs behind, and loading them all to press play
would cost more RAM than the rest of the application put together.  A small
cache of already-scaled pixmaps covers the frames a viewer keeps scrubbing
over, and is dropped whenever the window is resized because every entry in it
is scaled to the old size.
"""

from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Sequence

from PySide6.QtCore import Qt, QTimer, Slot
from PySide6.QtGui import QCloseEvent, QKeyEvent, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from app.timelapse_export import (
    DEFAULT_FRAME_RATE,
    MAX_FRAME_RATE,
    MIN_FRAME_RATE,
)

from .widgets import NoWheelSpinBox


LOGGER = logging.getLogger("rust_painter.timelapse")

# Enough that scrubbing back and forth over a moment of interest is instant,
# small enough that the cache stays a fraction of one decoded frame's cost.
_CACHE_FRAMES = 48


class TimelapsePlayer(QDialog):
    """Play one session's PNG frames back as video, with scrubbing."""

    def __init__(
        self,
        name: str,
        frames: Sequence[Path],
        parent: QWidget | None = None,
        *,
        frame_rate: int = DEFAULT_FRAME_RATE,
    ) -> None:
        super().__init__(parent)
        self._frames = list(frames)
        if not self._frames:
            raise ValueError("A timelapse player needs at least one frame")
        self._index = 0
        self._cache: OrderedDict[int, QPixmap] = OrderedDict()
        self._scaled_to = (0, 0)

        self.setWindowTitle(f"Timelapse — {name}")
        self.setModal(False)
        self.setMinimumSize(520, 400)
        self.resize(900, 640)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setMinimumSize(320, 240)
        self.view.setObjectName("panel")
        # Without this the label demands the pixmap's own size and the dialog
        # grows to the canvas resolution the moment a frame is shown.
        self.view.setScaledContents(False)
        layout.addWidget(self.view, 1)

        self.position_slider = QSlider(Qt.Orientation.Horizontal)
        self.position_slider.setRange(0, len(self._frames) - 1)
        self.position_slider.setSingleStep(1)
        self.position_slider.setPageStep(max(1, len(self._frames) // 20))
        self.position_slider.valueChanged.connect(self._seek)
        layout.addWidget(self.position_slider)

        controls = QHBoxLayout()
        controls.setSpacing(8)
        self.play_button = QPushButton("Play")
        self.play_button.setObjectName("accent")
        self.play_button.setMinimumWidth(96)
        self.play_button.clicked.connect(self.toggle_playback)
        self.restart_button = QPushButton("Restart")
        self.restart_button.clicked.connect(self.restart)
        self.speed_spin = NoWheelSpinBox()
        self.speed_spin.setRange(MIN_FRAME_RATE, MAX_FRAME_RATE)
        self.speed_spin.setValue(
            max(MIN_FRAME_RATE, min(MAX_FRAME_RATE, int(frame_rate)))
        )
        self.speed_spin.setSuffix(" fps")
        self.speed_spin.setToolTip(
            "Playback speed. This is also the frame rate the video export "
            "suggests, so a comfortable speed here is a comfortable video."
        )
        self.speed_spin.valueChanged.connect(self._apply_speed)
        self.counter_label = QLabel()
        self.counter_label.setObjectName("muted")
        controls.addWidget(self.play_button)
        controls.addWidget(self.restart_button)
        controls.addStretch(1)
        controls.addWidget(QLabel("Speed"))
        controls.addWidget(self.speed_spin)
        controls.addStretch(1)
        controls.addWidget(self.counter_label)
        layout.addLayout(controls)

        hint = QLabel("Space plays and pauses; the arrow keys step one frame.")
        hint.setObjectName("muted")
        layout.addWidget(hint)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)
        self._apply_speed()
        self._show_frame(0)

    # ------------------------------------------------------------- properties

    @property
    def frame_rate(self) -> int:
        return int(self.speed_spin.value())

    @property
    def is_playing(self) -> bool:
        return self._timer.isActive()

    # ---------------------------------------------------------------- control

    @Slot()
    def toggle_playback(self) -> None:
        if self.is_playing:
            self.pause()
        else:
            self.play()

    @Slot()
    def play(self) -> None:
        # Pressing play on the last frame reads as "watch it again", not as a
        # request to sit on a frozen final frame.
        if self._index >= len(self._frames) - 1:
            self._show_frame(0)
        self._timer.start()
        self.play_button.setText("Pause")

    @Slot()
    def pause(self) -> None:
        self._timer.stop()
        self.play_button.setText("Play")

    @Slot()
    def restart(self) -> None:
        self._show_frame(0)

    def step(self, delta: int) -> None:
        self.pause()
        self._show_frame(self._index + delta)

    @Slot()
    def _apply_speed(self) -> None:
        self._timer.setInterval(max(16, round(1000 / self.frame_rate)))

    @Slot(int)
    def _seek(self, index: int) -> None:
        if index != self._index:
            self._show_frame(index)

    @Slot()
    def _advance(self) -> None:
        if self._index >= len(self._frames) - 1:
            self.pause()
            return
        self._show_frame(self._index + 1)

    # ----------------------------------------------------------------- render

    def _show_frame(self, index: int) -> None:
        index = max(0, min(len(self._frames) - 1, index))
        self._index = index
        if self.position_slider.value() != index:
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(index)
            self.position_slider.blockSignals(False)
        pixmap = self._pixmap(index)
        if pixmap is not None:
            self.view.setPixmap(pixmap)
        self.counter_label.setText(f"Frame {index + 1} of {len(self._frames)}")

    def _pixmap(self, index: int) -> QPixmap | None:
        target = (max(1, self.view.width()), max(1, self.view.height()))
        if target != self._scaled_to:
            self._cache.clear()
            self._scaled_to = target
        cached = self._cache.get(index)
        if cached is not None:
            self._cache.move_to_end(index)
            return cached
        source = QPixmap(str(self._frames[index]))
        if source.isNull():
            LOGGER.warning("Could not decode timelapse frame %s", self._frames[index])
            return None
        scaled = source.scaled(
            target[0],
            target[1],
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self._cache[index] = scaled
        while len(self._cache) > _CACHE_FRAMES:
            self._cache.popitem(last=False)
        return scaled

    # -------------------------------------------------------------- Qt events

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API
        super().resizeEvent(event)
        self._show_frame(self._index)

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API
        key = event.key()
        if key == Qt.Key.Key_Space:
            self.toggle_playback()
            event.accept()
            return
        if key in (Qt.Key.Key_Left, Qt.Key.Key_Right):
            self.step(-1 if key == Qt.Key.Key_Left else 1)
            event.accept()
            return
        if key == Qt.Key.Key_Home:
            self.step(-len(self._frames))
            event.accept()
            return
        if key == Qt.Key.Key_End:
            self.step(len(self._frames))
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event: QCloseEvent) -> None:  # noqa: N802 - Qt API
        self.pause()
        self._cache.clear()
        super().closeEvent(event)


__all__ = ["TimelapsePlayer"]
