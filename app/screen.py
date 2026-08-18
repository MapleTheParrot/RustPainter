"""Screen, foreground-window, and lightweight reference utilities.

Windows uses Win32 physical coordinates; macOS uses global display points
(the Quartz/Qt coordinate space).  Both share the same public surface."""

from __future__ import annotations

import ctypes
import logging
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .models import ScreenRect

if TYPE_CHECKING:
    from PIL.Image import Image


LOGGER = logging.getLogger("rust_painter.screen")


def _executable_basename(value: str) -> str:
    """Return the file name from either a Windows or a POSIX path.

    ``Path(value).name`` only understands the host separator, so a Windows
    path read on macOS -- or typed into the expected-process setting by a user
    on either OS -- would otherwise be treated as one long file name and never
    match.
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


def _darwin_virtual_screen() -> VirtualScreen:
    """Union of active display bounds in global points, top-left origin."""

    import Quartz

    error, display_ids, count = Quartz.CGGetActiveDisplayList(16, None, None)
    if error or not count:
        raise OSError(f"Could not enumerate macOS displays (error {error})")
    left = top = None
    right = bottom = None
    for display_id in display_ids[:count]:
        bounds = Quartz.CGDisplayBounds(display_id)
        display_left = int(bounds.origin.x)
        display_top = int(bounds.origin.y)
        display_right = display_left + int(bounds.size.width)
        display_bottom = display_top + int(bounds.size.height)
        left = display_left if left is None else min(left, display_left)
        top = display_top if top is None else min(top, display_top)
        right = display_right if right is None else max(right, display_right)
        bottom = display_bottom if bottom is None else max(bottom, display_bottom)
    return VirtualScreen(left, top, right - left, bottom - top, int(count))


def get_virtual_screen() -> VirtualScreen:
    """Return the physical bounding rectangle of all Windows monitors."""

    if sys.platform == "darwin":
        return _darwin_virtual_screen()
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
    if sys.platform == "darwin":
        import Quartz

        location = Quartz.CGEventGetLocation(Quartz.CGEventCreate(None))
        return int(round(location.x)), int(round(location.y))
    if os.name != "nt":
        raise OSError("Global cursor position is available only on Windows and macOS")
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


def _darwin_foreground_window() -> ForegroundWindowInfo | None:
    """Describe the frontmost application via AppKit/Quartz.

    The focused window's title is only visible with the Screen Recording
    permission; without it the application name doubles as the title, which
    still supports name-based foreground matching.
    """

    from AppKit import NSWorkspace

    application = NSWorkspace.sharedWorkspace().frontmostApplication()
    if application is None:
        return None
    process_id = int(application.processIdentifier())
    name = str(application.localizedName() or "")
    executable_url = application.executableURL()
    executable = str(executable_url.path()) if executable_url is not None else None
    title = name
    try:
        import Quartz

        windows = Quartz.CGWindowListCopyWindowInfo(
            Quartz.kCGWindowListOptionOnScreenOnly
            | Quartz.kCGWindowListExcludeDesktopElements,
            Quartz.kCGNullWindowID,
        )
        for window in windows or ():
            if int(window.get("kCGWindowOwnerPID", -1)) != process_id:
                continue
            if int(window.get("kCGWindowLayer", 0)) != 0:
                continue
            title = str(window.get("kCGWindowName") or name)
            break
    except Exception:
        LOGGER.debug("Could not read the frontmost window title", exc_info=True)
    return ForegroundWindowInfo(
        hwnd=0,
        title=title,
        process_id=process_id,
        executable=executable,
    )


def get_foreground_window() -> ForegroundWindowInfo | None:
    """Describe the active top-level window, including executable when allowed."""

    if sys.platform == "darwin":
        return _darwin_foreground_window()
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


def _darwin_capture_region(rect: RectangleLike) -> "Image":
    """Capture a global-point rectangle on macOS, Retina-normalized.

    ``CGWindowListCreateImage`` accepts the same global point coordinates the
    rest of the app uses and captures across displays.  On Retina screens the
    result arrives at pixel scale, so it is resized back to point dimensions
    to preserve the "capture size == rectangle size" invariant relied on by
    brush and color calibration.  Requires the Screen Recording permission.
    """

    import Quartz
    from PIL import Image as PillowImage

    left, top, right, bottom = _bbox(rect)
    cg_rect = Quartz.CGRectMake(left, top, right - left, bottom - top)
    image = Quartz.CGWindowListCreateImage(
        cg_rect,
        Quartz.kCGWindowListOptionOnScreenOnly,
        Quartz.kCGNullWindowID,
        Quartz.kCGWindowImageDefault,
    )
    if image is None:
        raise OSError(
            "Could not capture the screen. Grant this app the Screen Recording "
            "permission under System Settings > Privacy & Security, then "
            "restart it."
        )
    width = Quartz.CGImageGetWidth(image)
    height = Quartz.CGImageGetHeight(image)
    bytes_per_row = Quartz.CGImageGetBytesPerRow(image)
    data = Quartz.CGDataProviderCopyData(Quartz.CGImageGetDataProvider(image))
    captured = PillowImage.frombuffer(
        "RGBA", (width, height), bytes(data), "raw", "BGRA", bytes_per_row, 1
    ).convert("RGB")
    expected = (rect.width, rect.height)
    if captured.size != expected:
        captured = captured.resize(expected, PillowImage.Resampling.LANCZOS)
    return captured


def capture_region(rect: RectangleLike) -> "Image":
    """Capture a calibrated physical-screen rectangle using Pillow."""

    if sys.platform == "darwin":
        return _darwin_capture_region(rect)
    from PIL import ImageGrab

    set_dpi_awareness()
    try:
        return ImageGrab.grab(bbox=_bbox(rect), all_screens=True).convert("RGB")
    except TypeError:
        # Older Pillow/non-Windows implementations may not accept all_screens.
        return ImageGrab.grab(bbox=_bbox(rect)).convert("RGB")


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
    "foreground_window_matches",
    "get_cursor_position",
    "get_foreground_window",
    "get_virtual_screen",
    "get_virtual_screen_rect",
    "save_reference",
    "set_dpi_awareness",
]
