from __future__ import annotations

import colorsys
import functools
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw

from app.models import ScreenRect
from app.ui_guard import (
    PaintingUiGuard,
    WatchedRegion,
    looks_like_hue_bar,
    region_signature,
    signature_similarity,
)


def rainbow(width: int, height: int, *, reverse: bool = False) -> Image.Image:
    """A vertical hue bar, fully saturated, like Rust's."""

    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    for y in range(height):
        hue = y / height
        if reverse:
            hue = 1.0 - hue
        pixels[y, :] = [int(c * 255) for c in colorsys.hsv_to_rgb(hue, 1.0, 1.0)]
    return Image.fromarray(pixels)


@functools.lru_cache(maxsize=None)
def sv_box(size: int, hue: float) -> Image.Image:
    """A saturation/value square for one hue, white top-left, black bottom."""

    pure = np.array(colorsys.hsv_to_rgb(hue, 1.0, 1.0), dtype=np.float32)
    saturation = np.linspace(0.0, 1.0, size, dtype=np.float32)[None, :, None]
    value = np.linspace(1.0, 0.0, size, dtype=np.float32)[:, None, None]
    pixels = value * (1.0 - saturation + saturation * pure[None, None, :])
    return Image.fromarray((pixels * 255).astype(np.uint8))


def button(width: int, height: int, text: str, fill=(88, 128, 40)) -> Image.Image:
    image = Image.new("RGB", (width, height), fill)
    ImageDraw.Draw(image).text((6, 4), text, fill=(230, 230, 230))
    return image


def world(width: int, height: int, seed: int = 7) -> Image.Image:
    """Something that is not the painting UI: a noisy game world."""

    rng = np.random.default_rng(seed)
    return Image.fromarray(rng.integers(0, 255, (height, width, 3), dtype=np.uint8))


class FakeScreen:
    """Captures the UI when it is open, and the world when it is not."""

    def __init__(self, target) -> None:
        self.target = target
        self.open = True
        self.hue = 0.7
        self.captures = 0

    def __call__(self, rect) -> Image.Image:
        self.captures += 1
        size = (rect.width, rect.height)
        if not self.open:
            return world(*size, seed=self.captures)
        if rect == self.target.hue_bar:
            return rainbow(*size)
        if rect == self.target.color_box:
            return sv_box(rect.width, self.hue).resize(size)
        if rect == getattr(self.target, "save_button", None):
            return button(rect.width, rect.height, "SAVE")
        if rect == getattr(self.target, "clear_button", None):
            return button(rect.width, rect.height, "X", fill=(170, 50, 40))
        return Image.new("RGB", size, (21, 21, 12))


def _target(**extra):
    fields = {
        "canvas": ScreenRect(100, 100, 400, 80),
        "color_box": ScreenRect(600, 100, 100, 100),
        "hue_bar": ScreenRect(720, 100, 12, 100),
        "clear_button": None,
        "save_button": None,
    }
    fields.update(extra)
    return SimpleNamespace(**fields)


def test_the_real_hue_bar_is_recognised_any_way_up() -> None:
    assert looks_like_hue_bar(rainbow(12, 100))
    assert looks_like_hue_bar(rainbow(12, 100, reverse=True))
    assert looks_like_hue_bar(rainbow(12, 100).transpose(Image.Transpose.ROTATE_90))
    # Dimmer, as behind a tint, and with a marker drawn across it.
    dim = Image.fromarray((np.asarray(rainbow(12, 100)) * 0.6).astype(np.uint8))
    assert looks_like_hue_bar(dim)
    marked = rainbow(12, 100)
    ImageDraw.Draw(marked).rectangle((0, 40, 11, 44), fill=(255, 255, 255))
    assert looks_like_hue_bar(marked)


def test_things_that_are_not_a_hue_bar_are_refused() -> None:
    assert not looks_like_hue_bar(Image.new("RGB", (12, 100), (21, 21, 12)))
    assert not looks_like_hue_bar(world(12, 100))
    # Saturated and colourful, but the hues are in no order.
    shuffled = np.asarray(rainbow(12, 100)).copy()
    np.random.default_rng(3).shuffle(shuffled, axis=0)
    assert not looks_like_hue_bar(Image.fromarray(shuffled))
    # A single saturated colour covers no spectrum.
    assert not looks_like_hue_bar(Image.new("RGB", (12, 100), (255, 0, 0)))


def test_the_colour_box_is_recognised_whatever_hue_is_picked() -> None:
    reference = region_signature(sv_box(64, 0.0), hue_invariant=True)
    for hue in (0.2, 0.5, 0.8):
        current = region_signature(sv_box(64, hue), hue_invariant=True)
        assert signature_similarity(reference, current) > 0.95
    # Compared on colour it would not be.
    plain = region_signature(sv_box(64, 0.0))
    assert signature_similarity(plain, region_signature(sv_box(64, 0.5))) < 0.9


def test_a_widget_survives_a_highlight_but_not_its_absence() -> None:
    save = button(120, 30, "SAVE")
    reference = region_signature(save)
    brighter = Image.fromarray(
        np.clip(np.asarray(save, dtype=np.float32) * 1.3, 0, 255).astype(np.uint8)
    )
    assert signature_similarity(reference, region_signature(brighter)) > 0.95
    assert signature_similarity(reference, region_signature(world(120, 30))) < 0.3
    assert signature_similarity(reference, region_signature(Image.new("RGB", (120, 30)))) == 0.0


def test_a_flat_widget_falls_back_to_pixel_error() -> None:
    flat = region_signature(Image.new("RGB", (40, 20), (30, 30, 30)))
    assert signature_similarity(flat, flat) == pytest.approx(1.0)
    assert signature_similarity(flat, region_signature(Image.new("RGB", (40, 20), (230, 230, 230)))) < 0.3


def test_the_guard_watches_what_is_calibrated_and_skips_the_size_field() -> None:
    target = _target(save_button=ScreenRect(600, 300, 100, 30), brush_size_box=ScreenRect(800, 100, 40, 20))
    guard = PaintingUiGuard.for_target(target)
    assert guard is not None
    assert [region.name for region in guard.regions] == ["color_box", "hue_bar", "save_button"]
    assert PaintingUiGuard.for_target(SimpleNamespace()) is None


def test_arming_needs_the_hue_bar_on_the_screen() -> None:
    target = _target()
    screen = FakeScreen(target)
    guard = PaintingUiGuard.for_target(target)
    screen.open = False
    assert not guard.arm(screen)
    assert not guard.armed
    with pytest.raises(RuntimeError):
        guard.check(screen)
    screen.open = True
    assert guard.arm(screen)
    assert guard.armed


def test_the_ui_is_present_until_most_of_it_is_gone() -> None:
    target = _target(
        clear_button=ScreenRect(40, 40, 30, 30), save_button=ScreenRect(600, 300, 100, 30)
    )
    screen = FakeScreen(target)
    guard = PaintingUiGuard.for_target(target)
    assert guard.arm(screen)
    # A different colour picked: still the UI.
    screen.hue = 0.1
    verdict = guard.check(screen)
    assert verdict.present and not verdict.missing, verdict.describe()
    # One widget hidden behind something: still the UI.
    one_hidden = FakeScreen(target)
    original = one_hidden.__call__

    def hide_save(rect):
        if rect == target.save_button:
            return world(rect.width, rect.height)
        return original(rect)

    verdict = guard.check(hide_save)
    assert verdict.present and verdict.missing == ("save_button",), verdict.describe()
    # The UI closed: everything is gone.
    screen.open = False
    verdict = guard.check(screen)
    assert not verdict.present
    assert set(verdict.missing) == {"color_box", "hue_bar", "clear_button", "save_button"}


def test_with_two_widgets_both_must_go_before_the_ui_counts_as_gone() -> None:
    target = _target()
    screen = FakeScreen(target)
    guard = PaintingUiGuard.for_target(target)
    assert guard.arm(screen)
    original = screen.__call__

    def hide_hue_bar(rect):
        if rect == target.hue_bar:
            return world(rect.width, rect.height)
        return original(rect)

    assert guard.check(hide_hue_bar).present
    screen.open = False
    assert not guard.check(screen).present


def test_a_guard_needs_something_to_watch() -> None:
    with pytest.raises(ValueError):
        PaintingUiGuard([])
    region = WatchedRegion("save_button", ScreenRect(0, 0, 10, 10))
    assert PaintingUiGuard([region]).regions == (region,)
