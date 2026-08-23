"""RGB/HSV conversion and calibrated Rust color-picker coordinates."""

from __future__ import annotations

import colorsys
from dataclasses import dataclass
from typing import Iterable

from .coordinates import RectangleLike, ScreenPoint, normalized_point
from .models import (
    HSVColor,
    HueDirection,
    RGBColor,
    SaturationDirection,
    ValueDirection,
)


@dataclass(frozen=True, slots=True)
class PickerCoordinates:
    hue: ScreenPoint
    saturation_value: ScreenPoint
    hsv: HSVColor

    @property
    def hue_point(self) -> ScreenPoint:
        return self.hue

    @property
    def sv_point(self) -> ScreenPoint:
        return self.saturation_value

    @property
    def color_point(self) -> ScreenPoint:
        return self.saturation_value

    def __iter__(self):
        yield self.hue
        yield self.saturation_value


def _coerce_rgb(
    red_or_rgb: RGBColor | Iterable[int], green: int | None, blue: int | None
) -> RGBColor:
    if green is None and blue is None:
        values = tuple(red_or_rgb)  # type: ignore[arg-type]
        if len(values) != 3:
            raise ValueError("RGB requires exactly three channels")
        red, green_value, blue_value = values
    elif green is not None and blue is not None and isinstance(red_or_rgb, (int, float)):
        red, green_value, blue_value = red_or_rgb, green, blue
    else:
        raise TypeError("Pass either an RGB tuple or three channel values")

    channels = (red, green_value, blue_value)
    if any(not 0 <= channel <= 255 for channel in channels):
        raise ValueError("RGB channels must be in the range 0..255")
    return int(red), int(green_value), int(blue_value)


def rgb_to_hsv(
    red_or_rgb: RGBColor | Iterable[int] | int,
    green: int | None = None,
    blue: int | None = None,
) -> HSVColor:
    """Convert 8-bit RGB to hue degrees and normalized saturation/value."""

    red, green_value, blue_value = _coerce_rgb(red_or_rgb, green, blue)
    hue, saturation, value = colorsys.rgb_to_hsv(
        red / 255.0, green_value / 255.0, blue_value / 255.0
    )
    return HSVColor(hue * 360.0, saturation, value)


def _enum_key(value: object) -> str:
    raw = value.value if hasattr(value, "value") else value
    return str(raw).strip().lower().replace("-", "_").replace(" ", "_")


def _hue_direction(value: HueDirection | str) -> HueDirection:
    key = _enum_key(value)
    aliases = {
        "top_to_bottom": HueDirection.TOP_TO_BOTTOM,
        "top_bottom": HueDirection.TOP_TO_BOTTOM,
        "bottom_to_top": HueDirection.BOTTOM_TO_TOP,
        "bottom_top": HueDirection.BOTTOM_TO_TOP,
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"Unknown hue direction: {value!r}") from exc


def _saturation_direction(value: SaturationDirection | str) -> SaturationDirection:
    key = _enum_key(value)
    aliases = {
        "left_low": SaturationDirection.LEFT_LOW,
        "left_low_right_high": SaturationDirection.LEFT_LOW,
        "left_to_right": SaturationDirection.LEFT_LOW,
        "right_high": SaturationDirection.LEFT_LOW,
        "left_high": SaturationDirection.LEFT_HIGH,
        "left_high_right_low": SaturationDirection.LEFT_HIGH,
        "right_to_left": SaturationDirection.LEFT_HIGH,
        "right_low": SaturationDirection.LEFT_HIGH,
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"Unknown saturation direction: {value!r}") from exc


def _value_direction(value: ValueDirection | str) -> ValueDirection:
    key = _enum_key(value)
    aliases = {
        "top_bright": ValueDirection.TOP_BRIGHT,
        "top_bright_bottom_dark": ValueDirection.TOP_BRIGHT,
        "top_to_bottom": ValueDirection.TOP_BRIGHT,
        "top_dark": ValueDirection.TOP_DARK,
        "top_dark_bottom_bright": ValueDirection.TOP_DARK,
        "bottom_to_top": ValueDirection.TOP_DARK,
    }
    try:
        return aliases[key]
    except KeyError as exc:
        raise ValueError(f"Unknown value direction: {value!r}") from exc


def map_hue_to_screen(
    hue_degrees: float,
    hue_bar: RectangleLike,
    direction: HueDirection | str = HueDirection.TOP_TO_BOTTOM,
) -> ScreenPoint:
    """Map a hue to the horizontal center of a calibrated vertical hue strip."""

    hue_normalized = (float(hue_degrees) % 360.0) / 360.0
    if _hue_direction(direction) is HueDirection.BOTTOM_TO_TOP:
        hue_normalized = 1.0 - hue_normalized
    return normalized_point(hue_bar, 0.5, hue_normalized)


def map_sv_to_screen(
    saturation: float,
    value: float,
    color_box: RectangleLike,
    saturation_direction: SaturationDirection | str = SaturationDirection.LEFT_LOW,
    value_direction: ValueDirection | str = ValueDirection.TOP_BRIGHT,
) -> ScreenPoint:
    """Map normalized saturation/value to a safe point in the color box."""

    saturation = min(max(float(saturation), 0.0), 1.0)
    value = min(max(float(value), 0.0), 1.0)
    x_normalized = saturation
    if _saturation_direction(saturation_direction) is SaturationDirection.LEFT_HIGH:
        x_normalized = 1.0 - x_normalized

    y_normalized = 1.0 - value
    if _value_direction(value_direction) is ValueDirection.TOP_DARK:
        y_normalized = value
    return normalized_point(color_box, x_normalized, y_normalized)


def hsv_to_picker_coordinates(
    hsv: HSVColor | tuple[float, float, float],
    hue_bar: RectangleLike,
    color_box: RectangleLike,
    *,
    hue_direction: HueDirection | str = HueDirection.TOP_TO_BOTTOM,
    saturation_direction: SaturationDirection | str = SaturationDirection.LEFT_LOW,
    value_direction: ValueDirection | str = ValueDirection.TOP_BRIGHT,
) -> PickerCoordinates:
    hsv_color = HSVColor(*hsv)
    return PickerCoordinates(
        hue=map_hue_to_screen(hsv_color.hue, hue_bar, hue_direction),
        saturation_value=map_sv_to_screen(
            hsv_color.saturation,
            hsv_color.value,
            color_box,
            saturation_direction,
            value_direction,
        ),
        hsv=hsv_color,
    )


def rgb_to_picker_coordinates(
    rgb: RGBColor | Iterable[int],
    hue_bar: RectangleLike,
    color_box: RectangleLike,
    *,
    hue_direction: HueDirection | str = HueDirection.TOP_TO_BOTTOM,
    saturation_direction: SaturationDirection | str = SaturationDirection.LEFT_LOW,
    value_direction: ValueDirection | str = ValueDirection.TOP_BRIGHT,
) -> PickerCoordinates:
    return hsv_to_picker_coordinates(
        rgb_to_hsv(rgb),
        hue_bar,
        color_box,
        hue_direction=hue_direction,
        saturation_direction=saturation_direction,
        value_direction=value_direction,
    )


def picker_points_to_rgb(
    hue_point: ScreenPoint,
    sv_point: ScreenPoint,
    hue_bar: RectangleLike,
    color_box: RectangleLike,
    *,
    hue_direction: HueDirection | str = HueDirection.TOP_TO_BOTTOM,
    saturation_direction: SaturationDirection | str = SaturationDirection.LEFT_LOW,
    value_direction: ValueDirection | str = ValueDirection.TOP_BRIGHT,
) -> RGBColor:
    """The color the picker selects when clicked at these two points.

    The inverse of ``rgb_to_picker_coordinates``: a click pulled inside a
    widget's edge, or clamped to it, selects a slightly different color from
    the one asked for, and this is what the panel will show for it.
    """

    def fraction(position: float, start: float, length: float) -> float:
        span = max(length - 1.0, 0.0)
        if span <= 0.0:
            return 0.0
        return min(max((float(position) - start) / span, 0.0), 1.0)

    hue_normalized = fraction(hue_point[1], hue_bar.top, hue_bar.height)
    if _hue_direction(hue_direction) is HueDirection.BOTTOM_TO_TOP:
        hue_normalized = 1.0 - hue_normalized
    saturation = fraction(sv_point[0], color_box.left, color_box.width)
    if _saturation_direction(saturation_direction) is SaturationDirection.LEFT_HIGH:
        saturation = 1.0 - saturation
    vertical = fraction(sv_point[1], color_box.top, color_box.height)
    value = vertical if _value_direction(value_direction) is ValueDirection.TOP_DARK else 1.0 - vertical
    red, green, blue = colorsys.hsv_to_rgb(hue_normalized % 1.0, saturation, value)
    return int(round(red * 255.0)), int(round(green * 255.0)), int(round(blue * 255.0))


# Friendly alias for painter code.
map_rgb_to_picker = rgb_to_picker_coordinates


def inset_click_point(
    point: ScreenPoint, rect: RectangleLike, margin_pixels: float
) -> tuple[int, int]:
    """A picker click target held ``margin_pixels`` inside ``rect``, rounded.

    The picker widgets ignore clicks on a couple of their outermost pixels
    (measured live: the hue bar's bottom two, nothing at its top, nothing at
    the S/V corner), so the painter clicks the exact computed point first
    and only pulls inward when the panel's read-back says the click was
    swallowed.  Fractional-of-the-widget insets are how every hue above
    352 degrees was lost on the murica sign: 2% of a 313 px bar is 6.3 px,
    which on a wrapping 360-degree axis is 7.2 degrees of red at EACH end.
    """

    margin_x = min(float(margin_pixels), rect.width * 0.10)
    margin_y = min(float(margin_pixels), rect.height * 0.10)
    x = min(max(point[0], rect.left + margin_x), rect.left + rect.width - 1 - margin_x)
    y = min(max(point[1], rect.top + margin_y), rect.top + rect.height - 1 - margin_y)
    return int(round(x)), int(round(y))


def picker_click_plan(
    rgb: RGBColor | Iterable[int],
    hue_bar: RectangleLike,
    color_box: RectangleLike,
    *,
    hue_direction: HueDirection | str = HueDirection.TOP_TO_BOTTOM,
    saturation_direction: SaturationDirection | str = SaturationDirection.LEFT_LOW,
    value_direction: ValueDirection | str = ValueDirection.TOP_BRIGHT,
    margin_pixels: float = 0.0,
) -> tuple[tuple[int, int], tuple[int, int], RGBColor]:
    """Click points for ``rgb`` at one edge inset, and the color they select.

    Returns ``(hue_point, sv_point, expected)``: the rounded click targets
    and the color the picker will actually show for them - the single source
    both the painter (to click and verify) and the planner/preview (to
    promise only reachable colors) share.
    """

    coordinates = rgb_to_picker_coordinates(
        rgb,
        hue_bar,
        color_box,
        hue_direction=hue_direction,
        saturation_direction=saturation_direction,
        value_direction=value_direction,
    )
    hue_point = inset_click_point(coordinates.hue, hue_bar, margin_pixels)
    sv_point = inset_click_point(coordinates.saturation_value, color_box, margin_pixels)
    expected = picker_points_to_rgb(
        hue_point,
        sv_point,
        hue_bar,
        color_box,
        hue_direction=hue_direction,
        saturation_direction=saturation_direction,
        value_direction=value_direction,
    )
    return hue_point, sv_point, expected


def reachable_color(
    rgb: RGBColor | Iterable[int],
    hue_bar: RectangleLike,
    color_box: RectangleLike,
    *,
    hue_direction: HueDirection | str = HueDirection.TOP_TO_BOTTOM,
    saturation_direction: SaturationDirection | str = SaturationDirection.LEFT_LOW,
    value_direction: ValueDirection | str = ValueDirection.TOP_BRIGHT,
) -> RGBColor:
    """The nearest color the picker can actually select for ``rgb``.

    What ``picker_click_plan`` expects at the painter's first-attempt inset:
    quantized to the widgets' pixel rasters.  Running the plan's palette
    through this is what makes the preview honest about color.
    """

    return picker_click_plan(
        rgb,
        hue_bar,
        color_box,
        hue_direction=hue_direction,
        saturation_direction=saturation_direction,
        value_direction=value_direction,
    )[2]


__all__ = [
    "HueDirection",
    "PickerCoordinates",
    "SaturationDirection",
    "ValueDirection",
    "hsv_to_picker_coordinates",
    "inset_click_point",
    "map_hue_to_screen",
    "map_rgb_to_picker",
    "map_sv_to_screen",
    "picker_click_plan",
    "picker_points_to_rgb",
    "reachable_color",
    "rgb_to_hsv",
    "rgb_to_picker_coordinates",
]
