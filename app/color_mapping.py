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


# Friendly alias for painter code.
map_rgb_to_picker = rgb_to_picker_coordinates


__all__ = [
    "HueDirection",
    "PickerCoordinates",
    "SaturationDirection",
    "ValueDirection",
    "hsv_to_picker_coordinates",
    "map_hue_to_screen",
    "map_rgb_to_picker",
    "map_sv_to_screen",
    "rgb_to_hsv",
    "rgb_to_picker_coordinates",
]
