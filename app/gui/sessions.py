"""The paint sessions a job has been through, offered back as a list.

Every real painting run files a resume record - where it got to, why it
stopped, the image and settings it was planned from.  This dialog is those
records as history: a long sign can be set aside for a smaller one and
picked up later by opening its session, which reloads its image and the
picture settings its plan was made with; the resume offer then arms itself
the moment the regenerated plan matches the record.  Sessions that are
done with can be deleted here - the sign keeps its paint, only the saved
place is lost.
"""

from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QSize, Qt, Slot
from PySide6.QtGui import QImageReader, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from app.resume_record import ResumeRecord, ResumeRecordStore

LOGGER = logging.getLogger("rust_painter.sessions")

_THUMBNAIL_SIZE = QSize(96, 54)


def _thumbnail(image_path: str | None) -> QPixmap | None:
    """The session's image at list size, or nothing if it cannot be read."""

    if not image_path or not Path(image_path).exists():
        return None
    try:
        reader = QImageReader(image_path)
        reader.setAutoTransform(True)
        size = reader.size()
        if size.isValid() and size.width() > 0 and size.height() > 0:
            scaled = size.scaled(_THUMBNAIL_SIZE, Qt.AspectRatioMode.KeepAspectRatio)
            reader.setScaledSize(scaled)
        image = reader.read()
        if image.isNull():
            return None
        return QPixmap.fromImage(image)
    except Exception:
        LOGGER.debug("Could not read a session thumbnail from %s", image_path, exc_info=True)
        return None


class SessionListDialog(QDialog):
    """Pick a past paint session to open, or delete the ones done with."""

    def __init__(
        self,
        store: ResumeRecordStore,
        records: list[ResumeRecord],
        *,
        current_fingerprint: str | None = None,
        active_fingerprint: str | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._store = store
        self._active_fingerprint = active_fingerprint
        self._current_fingerprint = current_fingerprint
        # The record the user chose to open, read by the caller after exec().
        self.chosen: ResumeRecord | None = None

        self.setWindowTitle("Paint sessions")
        self.setModal(True)
        self.setMinimumSize(520, 320)
        self.resize(640, 460)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        hint = QLabel(
            "Every real painting run keeps its place here.  Open a session to "
            "reload its image and picture settings; once its plan is rebuilt, "
            "the resume offer picks up at the recorded stroke.  Deleting a "
            "session only forgets the saved place - the sign keeps its paint."
        )
        hint.setWordWrap(True)
        hint.setObjectName("muted")
        layout.addWidget(hint)

        self.list = QListWidget()
        self.list.setSelectionMode(QListWidget.SelectionMode.ExtendedSelection)
        self.list.itemSelectionChanged.connect(self._refresh_buttons)
        self.list.itemDoubleClicked.connect(self._open_item)
        layout.addWidget(self.list, 1)

        self.empty_label = QLabel(
            "No paint sessions yet.  Start a real painting run (not a dry "
            "run) and its place will be kept here."
        )
        self.empty_label.setWordWrap(True)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_label.setObjectName("muted")
        layout.addWidget(self.empty_label)

        buttons = QHBoxLayout()
        self.delete_button = QPushButton("Delete")
        self.delete_button.setObjectName("danger")
        self.delete_button.clicked.connect(self._delete_selected)
        buttons.addWidget(self.delete_button)
        buttons.addStretch(1)
        self.open_button = QPushButton("Open session")
        self.open_button.setObjectName("accent")
        self.open_button.setDefault(True)
        self.open_button.clicked.connect(self._open_selected)
        close_button = QPushButton("Close")
        close_button.clicked.connect(self.reject)
        buttons.addWidget(self.open_button)
        buttons.addWidget(close_button)
        layout.addLayout(buttons)

        for record in records:
            self._add_record(record)
        if self.list.count():
            self.list.setCurrentRow(0)
        self.empty_label.setVisible(not self.list.count())
        self.list.setVisible(bool(self.list.count()))
        self._refresh_buttons()

    # ------------------------------------------------------------------ rows

    def _add_record(self, record: ResumeRecord) -> None:
        item = QListWidgetItem()
        item.setData(Qt.ItemDataRole.UserRole, record)
        self.list.addItem(item)
        self.list.setItemWidget(item, self._row_widget(record))
        item.setSizeHint(self.list.itemWidget(item).sizeHint())

    def _row_widget(self, record: ResumeRecord) -> QWidget:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(8, 6, 8, 6)
        row_layout.setSpacing(10)

        art = QLabel()
        art.setFixedSize(_THUMBNAIL_SIZE)
        art.setAlignment(Qt.AlignmentFlag.AlignCenter)
        pixmap = _thumbnail(record.image_path)
        if pixmap is not None:
            art.setPixmap(pixmap)
        else:
            art.setText("no\nimage")
            art.setObjectName("muted")
        row_layout.addWidget(art)

        lines = QVBoxLayout()
        lines.setSpacing(2)
        name = Path(record.image_path).name if record.image_path else "(image not recorded)"
        badge = ""
        if record.fingerprint == self._active_fingerprint:
            badge = "  •  painting now"
        elif record.fingerprint == self._current_fingerprint:
            badge = "  •  on screen now"
        title = QLabel(f"{name}{badge}")
        title.setStyleSheet("font-weight: 600;")
        lines.addWidget(title)
        status = QLabel(record.describe())
        status.setObjectName("muted")
        status.setWordWrap(True)
        lines.addWidget(status)
        details: list[str] = []
        if record.plan_width and record.plan_height:
            details.append(f"{record.plan_width}×{record.plan_height}")
        if record.profile_name:
            details.append(record.profile_name)
        if record.started_at:
            details.append(f"started {record.started_at}")
        if details:
            detail = QLabel("  •  ".join(details))
            detail.setObjectName("muted")
            lines.addWidget(detail)
        row_layout.addLayout(lines, 1)
        return row

    def _selected_records(self) -> list[ResumeRecord]:
        return [
            item.data(Qt.ItemDataRole.UserRole)
            for item in self.list.selectedItems()
        ]

    # --------------------------------------------------------------- actions

    @Slot()
    def _refresh_buttons(self) -> None:
        selected = self._selected_records()
        # The job painting right now is already on screen, and it rewrites
        # its record as it goes: opening it goes nowhere, and deleting it
        # would only have the record reappear seconds later.
        includes_active = any(
            record.fingerprint == self._active_fingerprint for record in selected
        )
        self.open_button.setEnabled(len(selected) == 1 and not includes_active)
        self.delete_button.setEnabled(bool(selected) and not includes_active)
        tip = (
            "The session painting right now cannot be opened or deleted - "
            "stop the job first."
            if includes_active
            else ""
        )
        self.open_button.setToolTip(tip)
        self.delete_button.setToolTip(tip)

    @Slot()
    def _open_selected(self) -> None:
        selected = self._selected_records()
        if len(selected) == 1 and selected[0].fingerprint != self._active_fingerprint:
            self.chosen = selected[0]
            self.accept()

    @Slot(QListWidgetItem)
    def _open_item(self, item: QListWidgetItem) -> None:
        record = item.data(Qt.ItemDataRole.UserRole)
        if record is not None and record.fingerprint != self._active_fingerprint:
            self.chosen = record
            self.accept()

    @Slot()
    def _delete_selected(self) -> None:
        records = [
            record for record in self._selected_records()
            if record.fingerprint != self._active_fingerprint
        ]
        if not records:
            return
        count = len(records)
        what = "this session" if count == 1 else f"these {count} sessions"
        if (
            QMessageBox.question(
                self,
                "Delete paint sessions",
                f"Delete {what}?  The sign keeps its paint - only the saved "
                "place to resume from is lost.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            != QMessageBox.StandardButton.Yes
        ):
            return
        doomed = {record.fingerprint for record in records}
        for fingerprint in doomed:
            self._store.delete(fingerprint)
        for row in range(self.list.count() - 1, -1, -1):
            item = self.list.item(row)
            record = item.data(Qt.ItemDataRole.UserRole)
            if record is not None and record.fingerprint in doomed:
                self.list.takeItem(row)
        self.empty_label.setVisible(not self.list.count())
        self.list.setVisible(bool(self.list.count()))
        self._refresh_buttons()


__all__ = ["SessionListDialog"]
