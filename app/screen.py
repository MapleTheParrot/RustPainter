"""Screen, foreground-window, and lightweight reference utilities.

Coordinates are Win32 physical pixels throughout."""

from __future__ import annotations

import ctypes
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .models import ScreenRect

if TYPE_CHECKING:
    from PIL.Image import Image


LOGGER = logging.getLogger("rust_painter.screen")


def _executable_basename(value: str) -> str:
    """Return the file name from either a Windows or a POSIX path.

    ``Path(value).name`` only understands the host separator, so a POSIX-style
    path typed into the expected-process setting would otherwise be treated as
    one long file name and never match.
    """

    if not value:
        return ""
    return re.split(r"[\\/]", value)[-1]


class RectangleLike(Protocol):
    left: int
    top: int
    width: int
    height: int


@dataclass(frozen=True, slots=True)
class VirtualScreen:
    left: int
    top: int
    width: int
    height: int
    monitor_count: int = 1

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def rect(self) -> ScreenRect:
        return ScreenRect(self.left, self.top, self.width, self.height)


@dataclass(frozen=True, slots=True)
class ForegroundWindowInfo:
    hwnd: int
    title: str
    process_id: int | None = None
    executable: str | None = None

    @property
    def executable_name(self) -> str:
        return _executable_basename(self.executable) if self.executable else ""


@dataclass(frozen=True, slots=True)
class ForegroundRequirement:
    """Any populated fields must match before automation may continue."""

    title_contains: str | None = None
    executable: str | None = None


@dataclass(frozen=True, slots=True)
class ReferenceComparison:
    similarity: float
    passed: bool
    mean_absolute_error: float
    reason: str = ""


_DPI_AWARENESS_ATTEMPTED = False
_DPI_AWARENESS_RESULT = False


def set_dpi_awareness() -> bool:
    """Request physical, per-monitor-v2 coordinates as early as practical.

    Calling this after Qt has created windows can be rejected by Windows.  That
    is harmless if Qt already selected an equivalent DPI mode, so this function
    remains idempotent and reports whether one of the APIs succeeded.
    """

    global _DPI_AWARENESS_ATTEMPTED, _DPI_AWARENESS_RESULT
    if _DPI_AWARENESS_ATTEMPTED:
        return _DPI_AWARENESS_RESULT
    _DPI_AWARENESS_ATTEMPTED = True
    if os.name != "nt":
        return False

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        setter = getattr(user32, "SetProcessDpiAwarenessContext", None)
        if setter is not None:
            setter.argtypes = (ctypes.c_void_p,)
            setter.restype = ctypes.c_bool
            # DPI_AWARENESS_CONTEXT_PER_MONITOR_AWARE_V2 is the signed pseudo
            # handle -4.  c_void_p preserves it correctly on 32- and 64-bit.
            context = ctypes.c_void_p(ctypes.c_ssize_t(-4).value)
            if setter(context):
                _DPI_AWARENESS_RESULT = True
                return True
    except OSError:
        LOGGER.debug("Per-monitor-v2 DPI awareness API unavailable", exc_info=True)

    try:
        shcore = ctypes.WinDLL("shcore", use_last_error=True)
        setter = shcore.SetProcessDpiAwareness
        setter.argtypes = (ctypes.c_int,)
        setter.restype = ctypes.c_long
        # PROCESS_PER_MONITOR_DPI_AWARE. S_OK (0) means this call set it.
        if setter(2) == 0:
            _DPI_AWARENESS_RESULT = True
            return True
    except OSError:
        LOGGER.debug("Shcore DPI awareness API unavailable", exc_info=True)

    try:
        user32 = ctypes.WinDLL("user32", use_last_error=True)
        setter = user32.SetProcessDPIAware
        setter.argtypes = ()
        setter.restype = ctypes.c_bool
        _DPI_AWARENESS_RESULT = bool(setter())
    except OSError:
        LOGGER.debug("Legacy DPI awareness API unavailable", exc_info=True)
    return _DPI_AWARENESS_RESULT


def get_virtual_screen() -> VirtualScreen:
    """Return the physical bounding rectangle of all Windows monitors."""

    if os.name != "nt":
        # This deterministic fallback supports dry runs and unit tests. Real
        # capture/input methods still state clearly when a supported OS is
        # required.
        return VirtualScreen(0, 0, 1920, 1080, 1)
    set_dpi_awareness()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    get_metric = user32.GetSystemMetrics
    get_metric.argtypes = (ctypes.c_int,)
    get_metric.restype = ctypes.c_int
    left = get_metric(76)  # SM_XVIRTUALSCREEN
    top = get_metric(77)  # SM_YVIRTUALSCREEN
    width = get_metric(78)  # SM_CXVIRTUALSCREEN
    height = get_metric(79)  # SM_CYVIRTUALSCREEN
    monitor_count = max(1, get_metric(80))  # SM_CMONITORS
    if width <= 0 or height <= 0:
        raise OSError("Windows returned an invalid virtual-screen rectangle")
    return VirtualScreen(left, top, width, height, monitor_count)


def get_virtual_screen_rect() -> ScreenRect:
    return get_virtual_screen().rect


def get_cursor_position() -> tuple[int, int]:
    if os.name != "nt":
        raise OSError("Global cursor position is available only on Windows")
    from ctypes import wintypes

    class POINT(ctypes.Structure):
        _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))

    point = POINT()
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    function = user32.GetCursorPos
    function.argtypes = (ctypes.POINTER(POINT),)
    function.restype = wintypes.BOOL
    if not function(ctypes.byref(point)):
        raise ctypes.WinError(ctypes.get_last_error() or 1)
    return int(point.x), int(point.y)


def _window_title(user32: object, hwnd: int) -> str:
    length_function = user32.GetWindowTextLengthW  # type: ignore[attr-defined]
    length_function.argtypes = (ctypes.c_void_p,)
    length_function.restype = ctypes.c_int
    text_function = user32.GetWindowTextW  # type: ignore[attr-defined]
    text_function.argtypes = (ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int)
    text_function.restype = ctypes.c_int
    length = max(0, length_function(hwnd))
    buffer = ctypes.create_unicode_buffer(length + 1)
    text_function(hwnd, buffer, len(buffer))
    return buffer.value


def _process_path(process_id: int) -> str | None:
    from ctypes import wintypes

    PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    open_process = kernel32.OpenProcess
    open_process.argtypes = (wintypes.DWORD, wintypes.BOOL, wintypes.DWORD)
    open_process.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = (wintypes.HANDLE,)
    close_handle.restype = wintypes.BOOL
    query_path = kernel32.QueryFullProcessImageNameW
    query_path.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.c_wchar_p,
        ctypes.POINTER(wintypes.DWORD),
    )
    query_path.restype = wintypes.BOOL

    handle = open_process(PROCESS_QUERY_LIMITED_INFORMATION, False, process_id)
    if not handle:
        return None
    try:
        capacity = wintypes.DWORD(32768)
        buffer = ctypes.create_unicode_buffer(capacity.value)
        if not query_path(handle, 0, buffer, ctypes.byref(capacity)):
            return None
        return buffer.value
    finally:
        close_handle(handle)


def get_foreground_window() -> ForegroundWindowInfo | None:
    """Describe the active top-level window, including executable when allowed."""

    if os.name != "nt":
        return None
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    foreground = user32.GetForegroundWindow
    foreground.argtypes = ()
    foreground.restype = wintypes.HWND
    hwnd = foreground()
    if not hwnd:
        return None
    pid = wintypes.DWORD()
    get_pid = user32.GetWindowThreadProcessId
    get_pid.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    get_pid.restype = wintypes.DWORD
    get_pid(hwnd, ctypes.byref(pid))
    process_id = int(pid.value) if pid.value else None
    executable = _process_path(process_id) if process_id is not None else None
    return ForegroundWindowInfo(
        hwnd=int(hwnd),
        title=_window_title(user32, hwnd),
        process_id=process_id,
        executable=executable,
    )


def find_window_matching(
    *,
    title_contains: str | None = None,
    executable: str | None = None,
    exclude_current_process: bool = True,
) -> ForegroundWindowInfo | None:
    """Find any visible top-level window matching the populated requirements.

    Unlike :func:`get_foreground_window`, the window does not have to be
    focused, so the app can locate the game while it is itself in front.
    Windows only; other platforms return ``None``.
    """

    if os.name != "nt" or (not title_contains and not executable):
        return None
    from ctypes import wintypes

    user32 = ctypes.WinDLL("user32", use_last_error=True)
    is_visible = user32.IsWindowVisible
    is_visible.argtypes = (wintypes.HWND,)
    is_visible.restype = wintypes.BOOL
    get_pid = user32.GetWindowThreadProcessId
    get_pid.argtypes = (wintypes.HWND, ctypes.POINTER(wintypes.DWORD))
    get_pid.restype = wintypes.DWORD

    expected_name = _executable_basename(executable or "").casefold()
    found: list[ForegroundWindowInfo] = []
    callback_type = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)

    def visit(hwnd: int, _lparam: int) -> bool:
        if not is_visible(hwnd):
            return True
        title = _window_title(user32, hwnd)
        if not title:
            return True
        if title_contains and title_contains.casefold() not in title.casefold():
            return True
        pid = wintypes.DWORD()
        get_pid(hwnd, ctypes.byref(pid))
        process_id = int(pid.value) if pid.value else None
        if exclude_current_process and process_id == os.getpid():
            return True
        path = _process_path(process_id) if process_id is not None else None
        if expected_name:
            current_name = _executable_basename(path or "").casefold()
            if not current_name or current_name != expected_name:
                return True
        found.append(
            ForegroundWindowInfo(
                hwnd=int(hwnd),
                title=title,
                process_id=process_id,
                executable=path,
            )
        )
        return False  # first match wins; stop enumerating

    enumerate_windows = user32.EnumWindows
    enumerate_windows.argtypes = (callback_type, wintypes.LPARAM)
    enumerate_windows.restype = wintypes.BOOL
    try:
        # Stopping the enumeration early makes EnumWindows report failure;
        # a hit in ``found`` is the actual result either way.
        enumerate_windows(callback_type(visit), 0)
    except OSError:
        LOGGER.warning("Could not enumerate top-level windows", exc_info=True)
        return None
    return found[0] if found else None


def _monitor_rect_from_handle(user32: ctypes.WinDLL, monitor: int) -> ScreenRect | None:
    from ctypes import wintypes

    class MonitorInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    if not monitor:
        return None
    info = MonitorInfo()
    info.cbSize = ctypes.sizeof(info)
    get_info = user32.GetMonitorInfoW
    get_info.argtypes = (wintypes.HANDLE, ctypes.POINTER(MonitorInfo))
    get_info.restype = wintypes.BOOL
    if not get_info(monitor, ctypes.byref(info)):
        return None
    bounds = info.rcMonitor
    width = int(bounds.right - bounds.left)
    height = int(bounds.bottom - bounds.top)
    if width <= 0 or height <= 0:
        return None
    return ScreenRect(int(bounds.left), int(bounds.top), width, height)


def window_monitor_rect(hwnd: int) -> ScreenRect | None:
    """The physical bounds of the monitor showing most of ``hwnd``."""

    if os.name != "nt" or not hwnd:
        return None
    from ctypes import wintypes

    MONITOR_DEFAULTTONEAREST = 2
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    from_window = user32.MonitorFromWindow
    from_window.argtypes = (wintypes.HWND, wintypes.DWORD)
    from_window.restype = wintypes.HANDLE
    try:
        monitor = from_window(hwnd, MONITOR_DEFAULTTONEAREST)
        return _monitor_rect_from_handle(user32, monitor)
    except OSError:
        LOGGER.warning("Could not resolve a window's monitor", exc_info=True)
        return None


def monitor_rect_at(x: int, y: int) -> ScreenRect | None:
    """The physical bounds of the monitor nearest the given desktop point."""

    if os.name != "nt":
        return None
    from ctypes import wintypes

    class POINT(ctypes.Structure):
        _fields_ = (("x", wintypes.LONG), ("y", wintypes.LONG))

    MONITOR_DEFAULTTONEAREST = 2
    user32 = ctypes.WinDLL("user32", use_last_error=True)
    from_point = user32.MonitorFromPoint
    from_point.argtypes = (POINT, wintypes.DWORD)
    from_point.restype = wintypes.HANDLE
    try:
        monitor = from_point(POINT(int(x), int(y)), MONITOR_DEFAULTTONEAREST)
        return _monitor_rect_from_handle(user32, monitor)
    except OSError:
        LOGGER.warning("Could not resolve a point's monitor", exc_info=True)
        return None


def map_rect_between_monitors(
    rect: ScreenRect, source: ScreenRect, target: ScreenRect
) -> ScreenRect:
    """Reproject a rectangle from one monitor onto another.

    Position and size scale with the monitor, so a calibration made on a
    1080p display lands proportionally on a 1440p one. Same-sized monitors
    reduce to a plain translation.
    """

    scale_x = target.width / source.width
    scale_y = target.height / source.height
    return ScreenRect(
        round(target.left + (rect.left - source.left) * scale_x),
        round(target.top + (rect.top - source.top) * scale_y),
        max(1, round(rect.width * scale_x)),
        max(1, round(rect.height * scale_y)),
    )


def foreground_window_matches(
    requirement: ForegroundRequirement | None = None,
    *,
    title_contains: str | None = None,
    executable: str | None = None,
    info: ForegroundWindowInfo | None = None,
    exclude_current_process: bool = True,
) -> bool:
    """Check populated title/executable requirements case-insensitively."""

    if requirement is not None:
        title_contains = requirement.title_contains
        executable = requirement.executable
    if not title_contains and not executable:
        return True
    current = info if info is not None else get_foreground_window()
    if current is None:
        return False
    # A broad title such as "Rust" also matches this application's own title,
    # "RustPainter". Never treat our process as a safe automation target.
    if exclude_current_process and current.process_id == os.getpid():
        return False
    if title_contains and title_contains.casefold() not in current.title.casefold():
        return False
    if executable:
        expected_name = _executable_basename(executable).casefold()
        current_name = current.executable_name.casefold()
        if not current_name or expected_name != current_name:
            return False
    return True


def _bbox(rect: RectangleLike) -> tuple[int, int, int, int]:
    if rect.width <= 0 or rect.height <= 0:
        raise ValueError("Capture rectangle must have positive width and height")
    return rect.left, rect.top, rect.left + rect.width, rect.top + rect.height


def capture_region(rect: RectangleLike) -> "Image":
    """Capture a calibrated physical-screen rectangle.

    On Windows the rectangle is copied straight from the screen with GDI.
    Pillow's grab copies the whole virtual desktop and crops afterwards,
    which on a two-monitor desktop costs around a hundred milliseconds per
    call however small the rectangle - too slow for anything that watches
    the screen while strokes are going down.  Pillow stays as the fallback.
    """

    set_dpi_awareness()
    left, top, right, bottom = _bbox(rect)
    if os.name == "nt":
        try:
            return _capture_region_gdi(left, top, right - left, bottom - top)
        except Exception:
            LOGGER.debug("GDI capture failed; falling back to Pillow", exc_info=True)
    from PIL import ImageGrab

    try:
        return ImageGrab.grab(bbox=(left, top, right, bottom), all_screens=True).convert("RGB")
    except TypeError:
        # Older Pillow/non-Windows implementations may not accept all_screens.
        return ImageGrab.grab(bbox=(left, top, right, bottom)).convert("RGB")


_SRCCOPY = 0x00CC0020
_DIB_RGB_COLORS = 0
_BI_RGB = 0


class _BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [
        ("biSize", ctypes.c_uint32),
        ("biWidth", ctypes.c_int32),
        ("biHeight", ctypes.c_int32),
        ("biPlanes", ctypes.c_uint16),
        ("biBitCount", ctypes.c_uint16),
        ("biCompression", ctypes.c_uint32),
        ("biSizeImage", ctypes.c_uint32),
        ("biXPelsPerMeter", ctypes.c_int32),
        ("biYPelsPerMeter", ctypes.c_int32),
        ("biClrUsed", ctypes.c_uint32),
        ("biClrImportant", ctypes.c_uint32),
    ]


class _BITMAPINFO(ctypes.Structure):
    _fields_ = [("bmiHeader", _BITMAPINFOHEADER), ("bmiColors", ctypes.c_uint32 * 3)]


def _capture_region_gdi(left: int, top: int, width: int, height: int) -> "Image":
    """BitBlt one rectangle of the virtual desktop into a Pillow image."""

    from PIL import Image as PillowImage

    user32 = ctypes.windll.user32
    gdi32 = ctypes.windll.gdi32
    for name, restype in (
        ("GetDC", ctypes.c_void_p),
        ("ReleaseDC", ctypes.c_int),
    ):
        getattr(user32, name).restype = restype
    user32.GetDC.argtypes = [ctypes.c_void_p]
    user32.ReleaseDC.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    gdi32.CreateCompatibleDC.restype = ctypes.c_void_p
    gdi32.CreateCompatibleDC.argtypes = [ctypes.c_void_p]
    gdi32.CreateCompatibleBitmap.restype = ctypes.c_void_p
    gdi32.CreateCompatibleBitmap.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    gdi32.SelectObject.restype = ctypes.c_void_p
    gdi32.SelectObject.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    gdi32.BitBlt.restype = ctypes.c_int
    gdi32.BitBlt.argtypes = [
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
        ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_uint32,
    ]
    gdi32.GetDIBits.restype = ctypes.c_int
    gdi32.GetDIBits.argtypes = [
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint, ctypes.c_uint,
        ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint,
    ]
    gdi32.DeleteObject.argtypes = [ctypes.c_void_p]
    gdi32.DeleteDC.argtypes = [ctypes.c_void_p]

    screen = user32.GetDC(None)
    if not screen:
        raise OSError("GetDC failed")
    memory = None
    bitmap = None
    try:
        memory = gdi32.CreateCompatibleDC(screen)
        bitmap = gdi32.CreateCompatibleBitmap(screen, width, height)
        if not memory or not bitmap:
            raise OSError("Could not create a capture bitmap")
        previous = gdi32.SelectObject(memory, bitmap)
        try:
            # Plain SRCCOPY, as Pillow's grab does it, so what this sees is
            # what every capture before it saw.
            if not gdi32.BitBlt(memory, 0, 0, width, height, screen, left, top, _SRCCOPY):
                raise OSError("BitBlt failed")
        finally:
            gdi32.SelectObject(memory, previous)
        info = _BITMAPINFO()
        info.bmiHeader.biSize = ctypes.sizeof(_BITMAPINFOHEADER)
        info.bmiHeader.biWidth = width
        # Negative height: rows top-down, so the buffer is the image as is.
        info.bmiHeader.biHeight = -height
        info.bmiHeader.biPlanes = 1
        info.bmiHeader.biBitCount = 32
        info.bmiHeader.biCompression = _BI_RGB
        buffer = ctypes.create_string_buffer(width * height * 4)
        rows = gdi32.GetDIBits(
            memory, bitmap, 0, height, buffer, ctypes.byref(info), _DIB_RGB_COLORS
        )
        if rows != height:
            raise OSError("GetDIBits failed")
        return PillowImage.frombuffer(
            "RGB", (width, height), buffer.raw, "raw", "BGRX", 0, 1
        ).copy()
    finally:
        if bitmap:
            gdi32.DeleteObject(bitmap)
        if memory:
            gdi32.DeleteDC(memory)
        user32.ReleaseDC(None, screen)


def save_reference(rect: RectangleLike, path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    capture_region(rect).save(target, format="PNG")
    return target


def compare_images(
    reference: "Image | str | Path",
    current: "Image",
    *,
    minimum_similarity: float = 0.85,
) -> ReferenceComparison:
    """Compare images by normalized mean absolute RGB error.

    This deliberately catches only obvious layout/profile mistakes. It is not a
    template matcher and can be affected by animation, lighting, or overlays.
    """

    from PIL import Image as PillowImage
    from PIL import ImageChops, ImageStat

    if not 0.0 <= minimum_similarity <= 1.0:
        raise ValueError("minimum_similarity must be in the range [0, 1]")
    if isinstance(reference, (str, Path)):
        with PillowImage.open(reference) as opened:
            reference_image = opened.convert("RGB")
    else:
        reference_image = reference.convert("RGB")
    current_image = current.convert("RGB")
    if reference_image.size != current_image.size:
        return ReferenceComparison(0.0, False, 1.0, "image dimensions differ")
    difference = ImageChops.difference(reference_image, current_image)
    means = ImageStat.Stat(difference).mean
    error = sum(means[:3]) / (3.0 * 255.0)
    similarity = min(max(1.0 - error, 0.0), 1.0)
    return ReferenceComparison(
        similarity=similarity,
        passed=similarity >= minimum_similarity,
        mean_absolute_error=error,
    )


def compare_region_to_reference(
    rect: RectangleLike,
    reference_path: str | Path,
    *,
    minimum_similarity: float = 0.85,
) -> ReferenceComparison:
    return compare_images(
        reference_path,
        capture_region(rect),
        minimum_similarity=minimum_similarity,
    )


__all__ = [
    "ForegroundRequirement",
    "ForegroundWindowInfo",
    "ReferenceComparison",
    "VirtualScreen",
    "capture_region",
    "compare_images",
    "compare_region_to_reference",
    "find_window_matching",
    "foreground_window_matches",
    "get_cursor_position",
    "get_foreground_window",
    "get_virtual_screen",
    "get_virtual_screen_rect",
    "map_rect_between_monitors",
    "monitor_rect_at",
    "save_reference",
    "set_dpi_awareness",
    "window_monitor_rect",
]
