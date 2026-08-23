from __future__ import annotations

from PIL import Image, ImageDraw

from app.color_mapping import map_rgb_to_picker, picker_points_to_rgb
from app.color_swatch import (
    LOCATOR_COLOR,
    MATCH_TOLERANCE,
    SwatchReading,
    locate_swatch,
    read_swatch,
    search_region,
)
from app.models import ScreenRect

HUE_BAR = ScreenRect(2364, 762, 52, 313)
COLOR_BOX = ScreenRect(2049, 761, 313, 315)
PANEL = (21, 21, 12)


def _panel(selected, *, block: ScreenRect | None = None, hex_text: bool = True):
    """A capture function drawing Rust's panel: the hue bar, the selected
    color's block right of it, and the hex readout beneath the block."""

    if block is None:
        block = ScreenRect(HUE_BAR.left + HUE_BAR.width + 2, HUE_BAR.top, 90, 236)

    def capture(rect: ScreenRect) -> Image.Image:
        image = Image.new("RGB", (rect.width, rect.height), PANEL)
        draw = ImageDraw.Draw(image)

        def box(r: ScreenRect):
            return (
                r.left - rect.left,
                r.top - rect.top,
                r.left + r.width - 1 - rect.left,
                r.top + r.height - 1 - rect.top,
            )

        for i in range(HUE_BAR.height):
            draw.line(
                (
                    HUE_BAR.left - rect.left,
                    HUE_BAR.top + i - rect.top,
                    HUE_BAR.left + HUE_BAR.width - 1 - rect.left,
                    HUE_BAR.top + i - rect.top,
                ),
                fill=(255, (i * 7) % 256, (i * 3) % 256),
            )
        draw.rectangle(box(block), fill=selected)
        if hex_text:
            text_box = ScreenRect(block.left, block.top + block.height + 4, block.width, 60)
            draw.rectangle(box(text_box), fill=(35, 35, 35))
            draw.text((block.left - rect.left + 10, text_box.top - rect.top + 20), "#F0C21E", fill=(240, 240, 240))
        return image

    return capture


def test_the_block_is_found_right_of_the_hue_bar_and_read_inset_from_its_edges() -> None:
    swatch = locate_swatch(_panel(LOCATOR_COLOR), HUE_BAR, LOCATOR_COLOR)
    assert swatch is not None
    block = ScreenRect(HUE_BAR.left + HUE_BAR.width + 2, HUE_BAR.top, 90, 236)
    assert block.left < swatch.left < block.left + 20
    assert block.top < swatch.top < block.top + 50
    assert swatch.left + swatch.width < block.left + block.width
    assert swatch.top + swatch.height < block.top + block.height
    reading = read_swatch(_panel(LOCATOR_COLOR), swatch)
    assert reading.color == LOCATOR_COLOR
    assert reading.spread == 0.0


def test_the_block_is_not_found_when_the_panel_is_flat_or_shows_another_color() -> None:
    def flat(rect: ScreenRect) -> Image.Image:
        return Image.new("RGB", (rect.width, rect.height), PANEL)

    assert locate_swatch(flat, HUE_BAR, LOCATOR_COLOR) is None
    # The block is there but in the wrong color: the locator clicks did not
    # take, and a block of the previous color must not be adopted.
    assert locate_swatch(_panel((30, 60, 220)), HUE_BAR, LOCATOR_COLOR) is None


def test_a_narrow_block_is_rejected() -> None:
    sliver = ScreenRect(HUE_BAR.left + HUE_BAR.width + 2, HUE_BAR.top, 12, 236)
    assert locate_swatch(_panel(LOCATOR_COLOR, block=sliver), HUE_BAR, LOCATOR_COLOR) is None


def test_the_search_region_hugs_the_hue_bar() -> None:
    region = search_region(HUE_BAR)
    assert region.left == HUE_BAR.left + HUE_BAR.width
    assert region.top < HUE_BAR.top
    assert region.top + region.height > HUE_BAR.top + HUE_BAR.height
    assert region.width >= 3 * HUE_BAR.width


def test_a_reading_matches_within_the_picker_quantisation_and_not_beyond() -> None:
    reading = SwatchReading((229, 53, 40), 0.0)
    assert reading.matches((230, 40, 40))
    assert reading.hex == "#E53528"
    # The previous group's near-twin on a fine palette is still told apart
    # once it is more than the tolerance away.
    assert not reading.matches((229, 53, 40 + int(MATCH_TOLERANCE) + 2))
    # A block with something drawn over it is not believed either way.
    assert not SwatchReading((230, 40, 40), 30.0).matches((230, 40, 40))


def test_reading_the_block_ignores_the_cursor_and_text_under_it() -> None:
    swatch = locate_swatch(_panel((240, 194, 30)), HUE_BAR, (240, 194, 30))
    assert swatch is not None
    reading = read_swatch(_panel((240, 194, 30)), swatch)
    assert reading.color == (240, 194, 30)


def test_picker_points_round_trip_and_follow_the_click_inset() -> None:
    directions = dict(
        hue_direction="bottom_to_top",
        saturation_direction="left_low",
        value_direction="top_bright",
    )
    for color in ((255, 0, 0), (230, 40, 40), (0, 0, 0), (255, 255, 255), (40, 90, 230)):
        points = map_rgb_to_picker(color, HUE_BAR, COLOR_BOX, **directions)
        back = picker_points_to_rgb(points.hue, points.saturation_value, HUE_BAR, COLOR_BOX, **directions)
        assert max(abs(a - b) for a, b in zip(back, color)) <= 1, (color, back)
    # A click pulled 2% inside the hue bar's end selects a hue 7 degrees
    # off pure red: that shade, not pure red, is what the panel shows.
    points = map_rgb_to_picker((255, 0, 0), HUE_BAR, COLOR_BOX, **directions)
    inset_hue = (points.hue[0], points.hue[1] - HUE_BAR.height * 0.02)
    shown = picker_points_to_rgb(inset_hue, points.saturation_value, HUE_BAR, COLOR_BOX, **directions)
    assert shown[0] == 255 and shown[2] == 0 and 20 <= shown[1] <= 40, shown
