from __future__ import annotations

from PIL import Image, ImageDraw, ImageFont

from app.digit_reader import read_number


def _number_image(value: int, *, inverted: bool = False) -> Image.Image:
    background, foreground = ((238, 238, 238), (24, 24, 24)) if inverted else (
        (16, 22, 28),
        (240, 180, 40),
    )
    image = Image.new("RGB", (110, 54), background)
    ImageDraw.Draw(image).text(
        (55, 27),
        str(value),
        font=ImageFont.load_default(size=42),
        fill=foreground,
        anchor="mm",
    )
    return image


def test_reads_bright_coloured_hud_digits() -> None:
    assert read_number(_number_image(49)) == 49


def test_reads_dark_digits_and_rejects_an_empty_crop() -> None:
    assert read_number(_number_image(100, inverted=True)) == 100
    assert read_number(Image.new("RGB", (80, 40), (20, 20, 20))) is None
