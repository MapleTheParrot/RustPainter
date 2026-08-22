"""RustPainter application entry point."""

from __future__ import annotations

import ctypes
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path


def _bundled_resource(name: str) -> Path:
    """Return a resource path in source and PyInstaller builds."""

    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root / name


def _enable_physical_screen_coordinates() -> None:
    """Opt into DPI awareness before Qt creates any native windows."""

    if os.name != "nt":
        return
    try:
        # PER_MONITOR_AWARE_V2; the negative pseudo-handle is intentional.
        if ctypes.windll.user32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4)):
            return
    except (AttributeError, OSError):
        pass
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


def _data_directory() -> Path:
    override = os.environ.get("RUST_PAINTER_DATA_DIR")
    if override:
        directory = Path(override).expanduser()
        directory.mkdir(parents=True, exist_ok=True)
        return directory
    root = os.environ.get("LOCALAPPDATA")
    directory = Path(root) / "RustPainter" if root else Path.cwd() / "data"
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def _configure_logging() -> Path:
    log_directory = _data_directory() / "logs"
    log_directory.mkdir(parents=True, exist_ok=True)
    path = log_directory / "rust_painter.log"
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    file_handler = RotatingFileHandler(
        path, maxBytes=2 * 1024 * 1024, backupCount=3, encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, stream_handler],
        force=True,
    )
    return path


def main() -> int:
    _enable_physical_screen_coordinates()
    log_path = _configure_logging()

    from PySide6.QtCore import QCoreApplication, Qt
    from PySide6.QtGui import QIcon
    from PySide6.QtWidgets import QApplication, QMessageBox

    from app.gui.main_window import MainWindow
    from app.gui.styles import apply_theme

    QCoreApplication.setOrganizationName("RustPainter")
    QCoreApplication.setApplicationName("RustPainter")
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_DontUseNativeMenuBar, True)
    application = QApplication(sys.argv)
    application.setApplicationDisplayName("RustPainter")
    icon_path = _bundled_resource("RustPainterIcon.png")
    if icon_path.is_file():
        application.setWindowIcon(QIcon(str(icon_path)))
    application.setQuitOnLastWindowClosed(True)
    apply_theme(application)

    def report_uncaught(error_type, error, traceback) -> None:
        logging.getLogger("rust_painter").critical(
            "Unhandled exception", exc_info=(error_type, error, traceback)
        )
        QMessageBox.critical(
            None,
            "RustPainter error",
            f"{error}\n\nDetails were written to:\n{log_path}",
        )

    sys.excepthook = report_uncaught
    window = MainWindow()
    if len(sys.argv) > 1:
        candidate = Path(sys.argv[1])
        if candidate.is_file():
            window.load_image(candidate)
    window.show()
    return application.exec()


if __name__ == "__main__":
    raise SystemExit(main())
