from __future__ import annotations

import os

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

    A Windows-style executable path read (or configured) on macOS must still
    reduce to its file name, otherwise the foreground guard silently never
    matches. Regression test for the macOS CI failure.
    """

    from app.screen import _executable_basename

    assert _executable_basename(r"C:\Games\Rust\RustClient.exe") == "RustClient.exe"
    assert _executable_basename("/Applications/Rust.app/Contents/MacOS/Rust") == "Rust"
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
        executable="/Applications/Rust.app/Contents/MacOS/Rust",
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
