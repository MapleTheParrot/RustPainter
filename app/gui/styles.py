"""Application palette and stylesheet.

The theme is built around the baked artwork in ``assets/ui``. Every surface in
the window is one of four seamless grain tiles rather than a flat colour — base
behind the window, panel behind cards, raised for anything that sits above a
panel (buttons, tabs, popups, metric cards) and inset for anything recessed into
one (inputs, previews, progress grooves, the log) — plus a wide worn plate
behind the header, a rust tile inside the progress bar, and two rounded rust
fills used as border-images on the primary and destructive buttons.
"""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from .assets import asset_url


ACCENT = "#ff9336"
ACCENT_SOFT = "#c9761f"
ACCENT_DEEP = "#a43c08"
ACCENT_HOVER = "#c9541a"
ACCENT_PRESSED = "#7d2c05"
DANGER = "#c0392b"
DANGER_HOVER = "#e0503d"
BACKGROUND = "#121110"
SURFACE = "#17130f"
PANEL = "#171412"
PANEL_RAISED = "#241d18"
BORDER = "#3a2617"
BORDER_LIGHT = "#5b3a22"
TEXT = "#ece2d8"
MUTED = "#9c8b7c"
WARNING = "#e6ae5f"
SUCCESS = "#a1d073"

# Text drawn on top of the rust button fills.
ON_ACCENT = "#fff1e2"

_BASE_TEXTURE = asset_url("surface-base")
_PANEL_TEXTURE = asset_url("surface-panel")
_RAISED_TEXTURE = asset_url("surface-raised")
_INSET_TEXTURE = asset_url("surface-inset")
_HEADER_TEXTURE = asset_url("surface-header")
_ACCENT_FILL = asset_url("fill-accent")
_DANGER_FILL = asset_url("fill-danger")
_PROGRESS_FILL = asset_url("fill-progress")


# Qt only tiles a background image through the shorthand form, and every rule
# below repeats the colour so a missing asset still degrades to the flat theme.
def _tile(color: str, texture: str) -> str:
    return f"background: {color} url({texture}) repeat;"


_BASE = _tile(BACKGROUND, _BASE_TEXTURE)
_PANEL_SURFACE = _tile(PANEL, _PANEL_TEXTURE)
_RAISED = _tile("#221c17", _RAISED_TEXTURE)
_INSET = _tile("#0e0c0b", _INSET_TEXTURE)
_POPUP = _tile(PANEL_RAISED, _RAISED_TEXTURE)
# The warm plate under a selected tab or the checked navigation button.
_SELECTED = _tile("#2d1c10", _RAISED_TEXTURE)
_PROGRESS = _tile(ACCENT_HOVER, _PROGRESS_FILL)


STYLE_SHEET = f"""
QWidget {{
    color: {TEXT};
    {_BASE}
    font-family: "Segoe UI Variable Text", "Segoe UI";
    font-size: 9.5pt;
}}
QMainWindow, QDialog {{ {_BASE} }}
QFrame#appHeader {{
    border: 1px solid {BORDER};
    border-radius: 12px;
    border-image: url({_HEADER_TEXTURE}) 0 0 0 0 stretch stretch;
}}
QFrame#appHeader > QWidget {{ background: transparent; }}
QLabel#appTitle {{
    color: {TEXT};
    font-size: 15pt;
    font-weight: 700;
    letter-spacing: 0.5px;
}}
QLabel#appMark {{
    background: transparent;
    border: 1px solid {BORDER_LIGHT};
    border-radius: 8px;
}}
QFrame#panel, QGroupBox {{
    {_PANEL_SURFACE}
    border: 1px solid {BORDER};
    border-radius: 11px;
}}
QFrame#inlinePanel {{
    {_INSET}
    border: 1px solid {BORDER};
    border-radius: 8px;
}}
/* Floating over a preview rather than sitting in the layout, so both of
   these carry their own plate and a rust edge that reads as "on top". */
QFrame#inlineNotice {{
    {_POPUP}
    border: 1px solid {ACCENT_SOFT};
    border-radius: 9px;
}}
QLabel#inlineNoticeText {{ color: {TEXT}; font-weight: 600; }}
QFrame#busyCard {{
    {_POPUP}
    border: 1px solid {ACCENT_SOFT};
    border-radius: 11px;
}}
QLabel#busyTitle {{ color: {TEXT}; font-size: 11pt; font-weight: 700; }}
QGroupBox {{
    margin-top: 12px;
    padding: 15px 12px 12px 12px;
    font-weight: 700;
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {ACCENT};
    font-size: 9pt;
    letter-spacing: 1.2px;
}}
QGroupBox::indicator {{
    width: 14px;
    height: 14px;
    border: 1px solid {BORDER_LIGHT};
    border-radius: 4px;
    background: {SURFACE};
}}
QGroupBox::indicator:checked {{ background: {ACCENT_DEEP}; border-color: {ACCENT}; }}
QLabel {{ background: transparent; }}
QLabel#pageTitle {{
    font-size: 13pt;
    font-weight: 700;
    color: {ACCENT};
    letter-spacing: 1.5px;
}}
QLabel#sectionTitle {{
    font-size: 10pt;
    font-weight: 700;
    color: {ACCENT_SOFT};
    letter-spacing: 0.6px;
}}
QLabel#stepBadge {{
    color: {ACCENT};
    background: #2a170c;
    border: 1px solid {BORDER_LIGHT};
    border-radius: 5px;
    font-weight: 800;
    font-size: 9pt;
}}
QLabel#panelTitle {{
    color: {TEXT};
    font-size: 10pt;
    font-weight: 700;
    letter-spacing: 1.4px;
}}
QFrame#metricCard {{
    {_RAISED}
    border: 1px solid {BORDER};
    border-radius: 9px;
}}
QLabel#muted, QLabel.muted {{ color: {MUTED}; }}
QLabel#metricValue {{ font-size: 13pt; font-weight: 700; color: {TEXT}; }}
QLabel#preview, QGraphicsView#preview {{
    {_INSET}
    border: 1px dashed {BORDER_LIGHT};
    border-radius: 10px;
    color: #7a6c60;
}}
QLabel#preview[dropActive="true"], QGraphicsView#preview[dropActive="true"] {{
    border: 1px dashed {ACCENT};
}}
QGraphicsView#preview {{ padding: 0; }}
QPushButton {{
    {_RAISED}
    border: 1px solid {BORDER_LIGHT};
    border-radius: 8px;
    padding: 7px 12px;
    min-height: 20px;
}}
QPushButton:hover {{ background-color: #2e2620; border-color: {ACCENT_SOFT}; }}
QPushButton:pressed {{ background-color: #191410; }}
QPushButton:focus {{ border-color: {ACCENT_SOFT}; }}
QPushButton:disabled {{ color: #6b5f55; background-color: #1a1613; border-color: #2a221c; }}
QPushButton#accent, QPushButton#danger {{
    color: {ON_ACCENT};
    border-width: 12px;
    border-radius: 0;
    padding: 4px 14px;
    font-weight: 800;
    letter-spacing: 0.8px;
}}
QPushButton#accent {{ border-image: url({_ACCENT_FILL}) 12 12 12 12 stretch stretch; }}
QPushButton#accent:hover {{ color: #ffffff; }}
QPushButton#accent:pressed {{ color: #f0cdae; }}
QPushButton#accent:disabled {{ color: #8c6a52; }}
QPushButton#danger {{ border-image: url({_DANGER_FILL}) 12 12 12 12 stretch stretch; }}
QPushButton#danger:hover {{ color: #ffffff; }}
QPushButton#danger:disabled {{ color: #8a6058; }}
QPushButton#flat, QPushButton#iconButton {{
    background-color: transparent;
    border-color: transparent;
    padding: 5px;
}}
QPushButton#flat:hover, QPushButton#iconButton:hover {{
    background-color: #2a1d14;
    border-color: {BORDER};
}}
QPushButton#compactButton {{ padding: 4px 8px; min-height: 18px; }}
QPushButton#navButton {{
    color: {MUTED};
    background-color: transparent;
    border-color: transparent;
    padding: 7px 12px;
    font-weight: 600;
}}
QPushButton#navButton:hover {{ color: {TEXT}; background-color: #241a12; }}
QPushButton#navButton:checked {{
    color: #ffd2a5;
    {_SELECTED}
    border-color: {BORDER_LIGHT};
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QKeySequenceEdit {{
    {_INSET}
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 5px 8px;
    min-height: 22px;
    selection-background-color: {ACCENT_DEEP};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT_SOFT};
}}
QLineEdit:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled, QComboBox:disabled {{
    color: #6b5f55;
    background-color: #161311;
    border-color: #2a221c;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button,
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    /* Left unstyled these draw a flat native block over the inset plate. */
    background: transparent;
    border: 0;
    subcontrol-origin: padding;
    width: 17px;
    height: 11px;
}}
QSpinBox::up-button, QDoubleSpinBox::up-button {{ subcontrol-position: top right; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{ subcontrol-position: bottom right; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: #2a1d14;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow,
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
    width: 0;
    height: 0;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
}}
QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ border-bottom: 5px solid {ACCENT_SOFT}; }}
QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ border-top: 5px solid {ACCENT_SOFT}; }}
QSpinBox::up-arrow:disabled, QDoubleSpinBox::up-arrow:disabled {{
    border-bottom-color: #4a3d33;
}}
QSpinBox::down-arrow:disabled, QDoubleSpinBox::down-arrow:disabled {{
    border-top-color: #4a3d33;
}}
QComboBox::drop-down {{ border: 0; width: 26px; }}
QComboBox::down-arrow {{
    width: 0; height: 0;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 6px solid {ACCENT_SOFT};
    margin-right: 8px;
}}
QComboBox QAbstractItemView {{
    {_POPUP}
    border: 1px solid {BORDER_LIGHT};
    border-radius: 6px;
    padding: 3px;
    selection-background-color: {ACCENT_DEEP};
}}
QCheckBox, QRadioButton {{ spacing: 8px; background: transparent; }}
QCheckBox::indicator, QRadioButton::indicator {{ width: 17px; height: 17px; }}
QCheckBox::indicator {{
    border: 1px solid {BORDER_LIGHT};
    border-radius: 5px;
    background: #0e0c0b;
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT}; }}
QCheckBox::indicator:checked {{ background: {ACCENT_DEEP}; border-color: {ACCENT}; }}
QCheckBox::indicator:disabled {{ background: #1a1613; border-color: #2a221c; }}
QTabWidget::pane {{
    border: 1px solid {BORDER};
    {_PANEL_SURFACE}
    border-radius: 10px;
    top: -1px;
}}
QTabBar {{ background: transparent; }}
QTabBar::tab {{
    background: transparent;
    color: {MUTED};
    padding: 7px 14px;
    border: 1px solid transparent;
    border-top-left-radius: 7px;
    border-top-right-radius: 7px;
    margin-right: 2px;
    font-weight: 600;
}}
QTabBar::tab:hover {{ color: {TEXT}; background-color: #241a12; }}
QTabBar::tab:selected {{
    {_SELECTED}
    color: #ffc793;
    border-color: {BORDER};
    border-bottom-color: #2d1c10;
}}
QProgressBar {{
    {_INSET}
    border: 1px solid {BORDER};
    border-radius: 7px;
    text-align: center;
    min-height: 20px;
    color: {TEXT};
}}
QProgressBar::chunk {{
    {_PROGRESS}
    border-radius: 6px;
}}
QSlider::groove:horizontal {{ height: 5px; background: #0e0c0b; border-radius: 2px; }}
QSlider::handle:horizontal {{
    width: 16px; margin: -6px 0; background: {ACCENT_HOVER}; border-radius: 8px;
}}
QScrollArea {{ border: 0; background: transparent; }}
QScrollArea > QWidget > QWidget {{ background: transparent; }}
QScrollBar:vertical {{ width: 10px; background: transparent; margin: 2px; }}
QScrollBar::handle:vertical {{ background: #4a3627; min-height: 30px; border-radius: 5px; }}
QScrollBar::handle:vertical:hover {{ background: {ACCENT_PRESSED}; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 10px; background: transparent; margin: 2px; }}
QScrollBar::handle:horizontal {{ background: #4a3627; min-width: 30px; border-radius: 5px; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
QSplitter::handle {{ background: transparent; width: 6px; }}
QPlainTextEdit {{
    {_INSET}
    border: 1px solid {BORDER};
    border-radius: 7px;
    color: #c8bcb0;
    font-family: "Cascadia Mono", Consolas;
    font-size: 8.5pt;
}}
QStatusBar {{
    {_BASE}
    color: {MUTED};
    border-top: 1px solid {BORDER};
}}
QStatusBar::item {{ border: 0; }}
QToolTip {{
    color: {TEXT};
    {_POPUP}
    border: 1px solid {BORDER_LIGHT};
    border-radius: 5px;
    padding: 5px 8px;
}}
QMessageBox {{ {_PANEL_SURFACE} }}
"""


# One badge style per painter state keeps the header consistent with the theme.
_BADGE_COLORS: dict[str, tuple[str, str, str]] = {
    "idle": (SUCCESS, "#1c2413", "#3f5a2a"),
    "ready": (SUCCESS, "#1c2413", "#3f5a2a"),
    "countdown": ("#ffcf8f", "#2c1d09", "#6b4318"),
    "running": ("#d9f2ad", "#25330f", "#4a6a26"),
    "paused": ("#ffcf8f", "#2c1d09", "#6b4318"),
    "completed": ("#d9f2ad", "#25330f", "#4a6a26"),
    "error": ("#ffb9ae", "#33130e", "#7b2a20"),
    "aborted": ("#ffb9ae", "#2d120e", "#6b2419"),
}


def state_badge_style(state: str) -> str:
    """Stylesheet for the header state badge frame and the labels inside it."""

    foreground, background, border = _BADGE_COLORS.get(
        state, _BADGE_COLORS["idle"]
    )
    return (
        f"QFrame#stateBadge {{ background: {background}; "
        f"border: 1px solid {border}; border-radius: 12px; }}"
        f"QFrame#stateBadge QLabel {{ color: {foreground}; background: transparent; "
        "border: 0; font-weight: 700; letter-spacing: 0.8px; }"
    )


def badge_foreground(state: str) -> str:
    """The badge text colour, so its icon can be tinted to match."""

    return _BADGE_COLORS.get(state, _BADGE_COLORS["idle"])[0]


def apply_theme(application: QApplication) -> None:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(BACKGROUND))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Base, QColor("#0e0c0b"))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(PANEL_RAISED))
    palette.setColor(QPalette.ColorRole.Text, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Button, QColor("#221c17"))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.ColorRole.Highlight, QColor(ACCENT_DEEP))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor("#7a6c60"))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(PANEL_RAISED))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(TEXT))
    application.setPalette(palette)
    application.setStyleSheet(STYLE_SHEET)
