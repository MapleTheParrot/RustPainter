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
