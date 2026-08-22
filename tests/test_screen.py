from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from PIL import Image

from app.screen import (
    ForegroundWindowInfo,
    compare_images,
    foreground_window_matches,
)


def test_foreground_match_checks_each_configured_identity() -> None:
    rust = ForegroundWindowInfo(
        hwnd=1,
        title="Rust",
        process_id=123,
        executable=r"C:\Games\Rust\RustClient.exe",
    )
    assert foreground_window_matches(title_contains="rust", info=rust)
    assert foreground_window_matches(executable="RustClient.exe", info=rust)
    assert foreground_window_matches(
        title_contains="Rust", executable=r"D:\Elsewhere\RustClient.exe", info=rust
    )
    assert not foreground_window_matches(title_contains="Notepad", info=rust)
    assert not foreground_window_matches(executable="Rust.exe", info=rust)
    own_window = ForegroundWindowInfo(
        hwnd=2,
        title="RustPainter",
        process_id=os.getpid(),
        executable="RustPainter.exe",
    )
    assert not foreground_window_matches(title_contains="Rust", info=own_window)


def test_reference_comparison_is_coarse_and_deterministic() -> None:
    reference = Image.new("RGB", (8, 8), (100, 120, 140))
    identical = compare_images(reference, reference.copy())
    different = compare_images(reference, Image.new("RGB", (8, 8), (255, 255, 255)))
    wrong_size = compare_images(reference, Image.new("RGB", (7, 8), (100, 120, 140)))

    assert identical.passed and identical.similarity == 1.0
    assert not different.passed and different.similarity < 0.85
    assert not wrong_size.passed and wrong_size.reason == "image dimensions differ"

def test_executable_name_is_parsed_regardless_of_path_separator() -> None:
    """Path().name only understands the host separator.

    A POSIX-style executable path typed into the expected-process setting
    must still reduce to its file name, otherwise the foreground guard
    silently never matches.
    """

    from app.screen import _executable_basename

    assert _executable_basename(r"C:\Games\Rust\RustClient.exe") == "RustClient.exe"
    assert _executable_basename("/opt/rust/Rust") == "Rust"
    assert _executable_basename("RustClient.exe") == "RustClient.exe"
    assert _executable_basename("") == ""

    windows_style = ForegroundWindowInfo(
        hwnd=1,
        title="Rust",
        process_id=123,
        executable=r"C:\Games\Rust\RustClient.exe",
    )
    assert windows_style.executable_name == "RustClient.exe"
    posix_style = ForegroundWindowInfo(
        hwnd=1,
        title="Rust",
        process_id=123,
        executable="/opt/rust/Rust",
    )
    assert posix_style.executable_name == "Rust"
    assert foreground_window_matches(executable="Rust", info=posix_style)


def test_rect_mapping_translates_between_same_size_monitors() -> None:
    from app.models import ScreenRect
    from app.screen import map_rect_between_monitors

    source = ScreenRect(0, 0, 1920, 1080)
    target = ScreenRect(1920, 0, 1920, 1080)
    rect = ScreenRect(100, 200, 640, 320)

    moved = map_rect_between_monitors(rect, source, target)

    assert moved == ScreenRect(2020, 200, 640, 320)


def test_rect_mapping_scales_between_different_resolutions() -> None:
    from app.models import ScreenRect
    from app.screen import map_rect_between_monitors

    source = ScreenRect(0, 0, 1920, 1080)
    target = ScreenRect(-2560, -100, 2560, 1440)
    rect = ScreenRect(192, 108, 960, 540)

    moved = map_rect_between_monitors(rect, source, target)

    assert moved == ScreenRect(-2560 + 256, -100 + 144, 1280, 720)


@pytest.mark.skipif(os.name != "nt", reason="GDI capture is Windows-only")
def test_gdi_capture_matches_pillow_pixel_for_pixel() -> None:
    """The fast path must see exactly what Pillow's grab saw.

    Everything that reads the screen - the verifier, the grid probe, the UI
    guard - goes through ``capture_region``, so a capture that differed from
    the old one by a channel order or a row flip would quietly break them
    all.  The desktop is live, so a region is compared only when two Pillow
    grabs of it agree with each other.
    """

    from PIL import ImageGrab

    from app.models import ScreenRect
    from app.screen import _capture_region_gdi, capture_region

    import numpy as np

    def pixels(image: Image.Image) -> np.ndarray:
        return np.asarray(image.convert("RGB"), dtype=np.int16)

    # The middle of the primary screen: the corners hold clocks and blinking
    # cursors, and a live region cannot be compared across grabs.
    rect = ScreenRect(1100, 600, 96, 64)
    bbox = (rect.left, rect.top, rect.right, rect.bottom)
    for _attempt in range(8):
        try:
            before = ImageGrab.grab(bbox=bbox, all_screens=True)
            ours = _capture_region_gdi(rect.left, rect.top, rect.width, rect.height)
            after = ImageGrab.grab(bbox=bbox, all_screens=True)
        except OSError as exc:  # a session with no desktop to read
            pytest.skip(f"no screen to capture: {exc}")
        assert ours.mode == "RGB" and ours.size == (rect.width, rect.height)
        if np.array_equal(pixels(before), pixels(after)):
            break
    else:
        pytest.skip("the screen kept changing between grabs")
    assert np.array_equal(pixels(ours), pixels(before))
    # And the public function takes the fast path to the same answer.
    assert capture_region(rect).size == (rect.width, rect.height)


def test_capture_region_refuses_an_empty_rectangle() -> None:
    from app.screen import capture_region

    with pytest.raises(ValueError):
        capture_region(SimpleNamespace(left=0, top=0, width=0, height=10))
