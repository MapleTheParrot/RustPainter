"""Review window for automatically detected Rust painting regions."""

from __future__ import annotations

from PIL import Image, ImageQt
from PySide6.QtCore import QRect, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFrame,
    QLabel,
    QVBoxLayout,
    QWidget,
)

from app.models import ScreenRect
from app.setup_detection import SetupDetection


_LABELS = {
    "canvas": "Canvas",
    "color_box": "Colour box",
    "hue_bar": "Hue bar",
    "brush_size_box": "Size value",
    "clear_button": "Clear",
    "save_button": "Save",
    "download_button": "Download",
}


class SetupReviewDialog(QDialog):
    """Show proposed rectangles before they replace the saved calibration."""

    def __init__(
        self,
        capture: Image.Image,
        screen: ScreenRect,
        detection: SetupDetection,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Review detected Rust setup")
        self.setMinimumSize(760, 620)
        self.resize(940, 760)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 14)
        layout.setSpacing(10)
        title = QLabel("Check the detected areas")
        title.setObjectName("pageTitle")
        missing = detection.missing_required
        detail = (
            "RustPainter found the painting UI. Save these areas, then use any "
            "individual Set area button if an outline needs fine-tuning."
            if not missing
            else "RustPainter found part of the painting UI. The missing "
            + ", ".join(_LABELS.get(name, name) for name in missing)
            + " can still be set manually."
        )
        note = QLabel(detail)
        note.setObjectName("muted")
        note.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(note)

        preview = QLabel()
        preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        preview.setFrameShape(QFrame.Shape.StyledPanel)
        preview.setMinimumHeight(360)
        preview.setPixmap(self._annotated(capture, screen, detection))
        layout.addWidget(preview, 1)

        confidence_lines = []
        for name, region in detection.regions.items():
            level = "high" if region.confidence >= 0.85 else "check"
            confidence_lines.append(
                f"{_LABELS.get(name, name)}: {level} confidence — {region.method}"
            )
        confidence = QLabel("\n".join(confidence_lines))
        confidence.setWordWrap(True)
        confidence.setObjectName("muted")
        layout.addWidget(confidence)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        save = buttons.button(QDialogButtonBox.StandardButton.Save)
        save.setText("Use detected areas")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _annotated(
        capture: Image.Image, screen: ScreenRect, detection: SetupDetection
    ) -> QPixmap:
        pixmap = QPixmap.fromImage(ImageQt.ImageQt(capture.convert("RGB")))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        font = painter.font()
        font.setBold(True)
        font.setPointSize(max(9, round(min(capture.size) / 95)))
        painter.setFont(font)
        for name, region in detection.regions.items():
            color = QColor("#58c878" if region.confidence >= 0.85 else "#e0a34b")
            pen = QPen(color, max(2, round(min(capture.size) / 400)))
            painter.setPen(pen)
            rect = region.rect
            local = QRect(
                rect.left - screen.left,
                rect.top - screen.top,
                rect.width,
                rect.height,
            )
            painter.drawRect(local)
            painter.fillRect(
                QRect(local.left(), max(0, local.top() - font.pointSize() - 8), 150, font.pointSize() + 8),
                QColor(18, 18, 18, 205),
            )
            painter.drawText(
                local.left() + 4,
                max(font.pointSize() + 2, local.top() - 5),
                _LABELS.get(name, name),
            )
        painter.end()
        return pixmap.scaled(
            880,
            500,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )


__all__ = ["SetupReviewDialog"]
