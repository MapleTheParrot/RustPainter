"""Central application defaults and atomic JSON settings persistence."""

from __future__ import annotations

import json
import math
import os
import re
import threading
import uuid
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any


# Rust's Windows client runs as RustClient.exe. On other platforms the
# executable name differs, so a Windows name there can never match and would
# make the foreground guard pause the moment the user focuses the game. The
# window-title check governs instead, and the field stays editable for anyone
# who wants the stricter two-part guard.
DEFAULT_EXPECTED_PROCESS_NAME = "RustClient.exe" if os.name == "nt" else ""

SETTINGS_SCHEMA_VERSION = 1
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "settings.json"


QUALITY_PRESETS: dict[str, dict[str, int]] = {
    "very_fast": {"logical_width": 64, "color_count": 16},
    "fast": {"logical_width": 128, "color_count": 24},
    "balanced": {"logical_width": 256, "color_count": 32},
    "high": {"logical_width": 384, "color_count": 64},
    "very_high": {"logical_width": 512, "color_count": 96},
}

SUPPORTED_COLOR_COUNTS = {8, 16, 24, 32, 48, 64, 96, 128, 256}
_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


# Nested sections keep ownership clear while still producing ordinary JSON and
# dictionaries that are easy for Qt widgets and worker threads to consume.
DEFAULT_SETTINGS: dict[str, Any] = {
    "schema_version": SETTINGS_SCHEMA_VERSION,
    "image": {
        "scale_mode": "fit",
        "crop_alignment": "center",
        "quality_preset": "balanced",
        "logical_width": 256,
        "logical_height": 128,
        "color_count": 32,
        "dithering": False,
        "background_mode": "unpainted",
        "background_color": "#FFFFFF",
        "transparent_pixels": "leave_unpainted",
        "text_overlay": {
            "layers": [
                {
                    "text": "",
                    "font_family": "",
                    "font_size": 24,
                    # Font height as a fraction of the logical canvas height.
                    # The GUI keeps this fixed and re-derives font_size, so text
                    # stays the same size when the quality preset changes.
                    "size_ratio": 0.1875,
                    "color": "#FFFFFF",
                    "x": 0.5,
                    "y": 0.5,
                    "bold": False,
                    "italic": False,
                }
            ],
        },
    },
    "painting": {
        "brush_size": 0.15,
        "apply_brush_size": False,
        "brush_direction": "low_to_high",
        "logical_pixel_spacing": 1.0,
        "stroke_speed_pixels_per_second": 700.0,
        "mouse_down_duration_seconds": 0.028,
        "delay_after_hue_seconds": 0.09,
        "delay_after_saturation_value_seconds": 0.09,
        "delay_after_brush_seconds": 0.06,
        "delay_between_strokes_seconds": 0.018,
        "delay_between_colors_seconds": 0.12,
        "stroke_interpolation_step_pixels": 4.0,
        "stroke_merge_mode": "balanced",
    },
    "hotkeys": {
        "start_resume": "F8",
        "pause": "F9",
        "abort": "F10",
    },
    "safety": {
        "countdown_seconds": 3,
        "corner_abort_enabled": True,
        "corner_abort_margin_pixels": 3,
        "require_rust_foreground": True,
        "expected_process_name": DEFAULT_EXPECTED_PROCESS_NAME,
        "expected_window_title_contains": "Rust",
        "verify_calibrated_ui": False,
    },
    "execution": {
        "dry_run": False,
        "debug_mouse_logging": False,
    },
    "ui": {
        "selected_profile_id": None,
        "last_image_path": None,
        "window_geometry": None,
        "show_calibration_overlay": False,
    },
}


class SettingsError(ValueError):
    """Raised when settings cannot be decoded, validated, or written."""


_MISSING = object()


def default_settings() -> dict[str, Any]:
    """Return an independent copy so callers cannot mutate global defaults."""

    return deepcopy(DEFAULT_SETTINGS)


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """Merge recursively, retaining unknown keys for forward compatibility."""

    result = deepcopy(dict(base))
    for key, value in override.items():
        existing = result.get(key)
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            result[key] = _deep_merge(existing, value)
        else:
            result[key] = deepcopy(value)
    return result


def _validate(settings: Mapping[str, Any]) -> None:
    image = settings.get("image")
    painting = settings.get("painting")
    hotkeys = settings.get("hotkeys")
    safety = settings.get("safety")
    execution = settings.get("execution")
    ui = settings.get("ui")
    for label, section in (
        ("image", image),
        ("painting", painting),
        ("hotkeys", hotkeys),
        ("safety", safety),
        ("execution", execution),
        ("ui", ui),
    ):
        if not isinstance(section, Mapping):
            raise SettingsError(f"Settings section {label!r} must be a JSON object")

    assert isinstance(image, Mapping)
    if not isinstance(image.get("scale_mode"), str) or image.get("scale_mode") not in {
        "fit",
        "fill",
        "stretch",
    }:
        raise SettingsError("image.scale_mode must be fit, fill, or stretch")
    if not isinstance(image.get("crop_alignment"), str) or image.get(
        "crop_alignment"
    ) not in {"center", "top", "bottom", "left", "right"}:
        raise SettingsError("image.crop_alignment is invalid")
    if not isinstance(image.get("quality_preset"), str) or image.get(
        "quality_preset"
    ) not in {*QUALITY_PRESETS, "custom"}:
        raise SettingsError("image.quality_preset is invalid")
    for key in ("logical_width", "logical_height"):
        value = image.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or not 8 <= value <= 2048:
            raise SettingsError(f"image.{key} must be an integer from 8 to 2048")
    color_count = image.get("color_count")
    if (
        isinstance(color_count, bool)
        or not isinstance(color_count, int)
        or color_count not in SUPPORTED_COLOR_COUNTS
    ):
        raise SettingsError(
            "image.color_count must be one of "
            + ", ".join(str(value) for value in sorted(SUPPORTED_COLOR_COUNTS))
        )
    if not isinstance(image.get("dithering"), bool):
        raise SettingsError("image.dithering must be true or false")
    if not isinstance(image.get("background_mode"), str) or image.get(
        "background_mode"
    ) not in {"unpainted", "white", "black", "custom"}:
        raise SettingsError("image.background_mode is invalid")
    background_color = image.get("background_color")
    if not isinstance(background_color, str) or _HEX_COLOR.fullmatch(background_color) is None:
        raise SettingsError("image.background_color must use #RRGGBB format")
    if not isinstance(image.get("transparent_pixels"), str) or image.get(
        "transparent_pixels"
    ) not in {"leave_unpainted", "use_background"}:
        raise SettingsError("image.transparent_pixels is invalid")
    text_overlay = image.get("text_overlay")
    if not isinstance(text_overlay, Mapping):
        raise SettingsError("image.text_overlay must be a JSON object")
    text_layers = text_overlay.get("layers")
    if not isinstance(text_layers, list) or not 1 <= len(text_layers) <= 20:
        raise SettingsError("image.text_overlay.layers must contain 1 to 20 text layers")
    for index, layer in enumerate(text_layers):
        label = f"image.text_overlay.layers[{index}]"
        if not isinstance(layer, Mapping):
            raise SettingsError(f"{label} must be a JSON object")
        text = layer.get("text")
        if not isinstance(text, str) or len(text) > 500:
            raise SettingsError(f"{label}.text must contain at most 500 characters")
        font_family = layer.get("font_family")
        if not isinstance(font_family, str) or len(font_family) > 200:
            raise SettingsError(f"{label}.font_family must be text")
        font_size = layer.get("font_size")
        if (
            isinstance(font_size, bool)
            or not isinstance(font_size, int)
            or not 4 <= font_size <= 256
        ):
            raise SettingsError(f"{label}.font_size must be an integer from 4 to 256")
        size_ratio = layer.get("size_ratio")
        if size_ratio is not None and (
            isinstance(size_ratio, bool)
            or not isinstance(size_ratio, (int, float))
            or not math.isfinite(float(size_ratio))
            or not 0.0 < float(size_ratio) <= 32.0
        ):
            raise SettingsError(f"{label}.size_ratio must be between 0 and 32")
        text_color = layer.get("color")
        if not isinstance(text_color, str) or _HEX_COLOR.fullmatch(text_color) is None:
            raise SettingsError(f"{label}.color must use #RRGGBB format")
        for coordinate in ("x", "y"):
            value = layer.get(coordinate)
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(float(value))
                or not 0.0 <= float(value) <= 1.0
            ):
                raise SettingsError(f"{label}.{coordinate} must be between 0 and 1")
        for key in ("bold", "italic"):
            if not isinstance(layer.get(key), bool):
                raise SettingsError(f"{label}.{key} must be true or false")

    assert isinstance(painting, Mapping)
    nonnegative = {
        "brush_size",
        "mouse_down_duration_seconds",
        "delay_after_hue_seconds",
        "delay_after_saturation_value_seconds",
        "delay_after_brush_seconds",
        "delay_between_strokes_seconds",
        "delay_between_colors_seconds",
    }
    positive = {
        "logical_pixel_spacing",
        "stroke_speed_pixels_per_second",
        "stroke_interpolation_step_pixels",
    }
    for key in nonnegative | positive:
        value = painting.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise SettingsError(f"painting.{key} must be numeric")
        if not math.isfinite(float(value)):
            raise SettingsError(f"painting.{key} must be finite")
        if key in positive and value <= 0:
            raise SettingsError(f"painting.{key} must be positive")
        if key in nonnegative and value < 0:
            raise SettingsError(f"painting.{key} cannot be negative")
    if float(painting["brush_size"]) > 1:
        raise SettingsError("painting.brush_size must be between 0 and 1")
    if not isinstance(painting.get("apply_brush_size"), bool):
        raise SettingsError("painting.apply_brush_size must be true or false")
    if not isinstance(painting.get("brush_direction"), str) or painting.get(
        "brush_direction"
    ) not in {"low_to_high", "high_to_low"}:
        raise SettingsError(
            "painting.brush_direction must be low_to_high or high_to_low"
        )
    merge_mode = painting.get("stroke_merge_mode", "balanced")
    if not isinstance(merge_mode, str) or merge_mode not in {
        "off",
        "balanced",
        "maximum",
    }:
        raise SettingsError(
            "painting.stroke_merge_mode must be off, balanced, or maximum"
        )

    assert isinstance(hotkeys, Mapping)
    allowed_hotkeys = {f"F{number}" for number in range(5, 13)}
    for key in ("start_resume", "pause", "abort"):
        if not isinstance(hotkeys.get(key), str) or not str(hotkeys[key]).strip():
            raise SettingsError(f"hotkeys.{key} must be a non-empty string")
        if str(hotkeys[key]).strip().upper() not in allowed_hotkeys:
            raise SettingsError(f"hotkeys.{key} must be one of F5 through F12")
    normalized_hotkeys = {
        str(hotkeys[key]).strip().upper()
        for key in ("start_resume", "pause", "abort")
    }
    if len(normalized_hotkeys) != 3:
        raise SettingsError("Start, pause, and abort hotkeys must be different")

    assert isinstance(safety, Mapping)
    countdown = safety.get("countdown_seconds")
    if isinstance(countdown, bool) or not isinstance(countdown, int) or countdown < 0:
        raise SettingsError("safety.countdown_seconds must be a non-negative integer")
    for key in (
        "corner_abort_enabled",
        "require_rust_foreground",
        "verify_calibrated_ui",
    ):
        if not isinstance(safety.get(key), bool):
            raise SettingsError(f"safety.{key} must be true or false")
    for key in ("expected_process_name", "expected_window_title_contains"):
        if safety.get(key) is not None and not isinstance(safety.get(key), str):
            raise SettingsError(f"safety.{key} must be a string or null")
    corner_margin = safety.get("corner_abort_margin_pixels")
    if (
        isinstance(corner_margin, bool)
        or not isinstance(corner_margin, int)
        or corner_margin < 0
    ):
        raise SettingsError(
            "safety.corner_abort_margin_pixels must be a non-negative integer"
        )
    for key in (
        "corner_abort_minimum_distance_pixels",
        "focus_check_interval_seconds",
        "safety_poll_interval_seconds",
    ):
        if key not in safety:
            continue
        value = safety[key]
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or value < 0
        ):
            raise SettingsError(f"safety.{key} must be a finite non-negative number")
        if key != "corner_abort_minimum_distance_pixels" and value == 0:
            raise SettingsError(f"safety.{key} must be positive")

    assert isinstance(execution, Mapping)
    for key in ("dry_run", "debug_mouse_logging"):
        if not isinstance(execution.get(key), bool):
            raise SettingsError(f"execution.{key} must be true or false")

    assert isinstance(ui, Mapping)
    for key in ("selected_profile_id", "last_image_path"):
        if ui.get(key) is not None and not isinstance(ui.get(key), str):
            raise SettingsError(f"ui.{key} must be a string or null")


class SettingsStore:
    """Read and update settings while always supplying newly added defaults."""

    def __init__(
        self,
        path: str | Path = DEFAULT_SETTINGS_PATH,
        *,
        defaults: Mapping[str, Any] | None = None,
    ) -> None:
        self.path = Path(path).expanduser()
        self.defaults = deepcopy(dict(DEFAULT_SETTINGS if defaults is None else defaults))
        self._lock = threading.RLock()

    def load(self) -> dict[str, Any]:
        with self._lock:
            if not self.path.exists():
                result = deepcopy(self.defaults)
                _validate(result)
                return result
            try:
                with self.path.open("r", encoding="utf-8-sig") as handle:
                    raw = json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise SettingsError(f"Could not read settings {self.path}: {exc}") from exc
            if not isinstance(raw, Mapping):
                raise SettingsError("Settings file must contain a JSON object")
            # Accept either spelling used by settings/profile schema documents.
            if "schemaVersion" in raw and "schema_version" not in raw:
                raw = dict(raw)
                raw["schema_version"] = raw.pop("schemaVersion")
            result = _deep_merge(self.defaults, raw)
            result["schema_version"] = SETTINGS_SCHEMA_VERSION
            _validate(result)
            return result

    def save(self, settings: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(settings, Mapping):
            raise SettingsError("Settings must be a mapping")
        merged = _deep_merge(self.defaults, settings)
        merged["schema_version"] = SETTINGS_SCHEMA_VERSION
        _validate(merged)
        with self._lock:
            self._write(merged)
        return deepcopy(merged)

    def update(self, patch: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(patch, Mapping):
            raise SettingsError("Settings patch must be a mapping")
        with self._lock:
            return self.save(_deep_merge(self.load(), patch))

    def get(self, dotted_key: str, default: Any = _MISSING) -> Any:
        current: Any = self.load()
        for part in dotted_key.split("."):
            if not isinstance(current, Mapping) or part not in current:
                if default is _MISSING:
                    raise KeyError(dotted_key)
                return deepcopy(default)
            current = current[part]
        return deepcopy(current)

    def set(self, dotted_key: str, value: Any) -> dict[str, Any]:
        parts = [part for part in dotted_key.split(".") if part]
        if not parts:
            raise SettingsError("Settings key must not be empty")
        patch: dict[str, Any] = {}
        cursor = patch
        for part in parts[:-1]:
            nested: dict[str, Any] = {}
            cursor[part] = nested
            cursor = nested
        cursor[parts[-1]] = value
        return self.update(patch)

    def reset(self, *, persist: bool = True) -> dict[str, Any]:
        values = deepcopy(self.defaults)
        values["schema_version"] = SETTINGS_SCHEMA_VERSION
        _validate(values)
        if persist:
            with self._lock:
                self._write(values)
        return values

    def _write(self, settings: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(f".{self.path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("x", encoding="utf-8", newline="\n") as handle:
                json.dump(
                    settings,
                    handle,
                    indent=2,
                    ensure_ascii=False,
                    sort_keys=False,
                    allow_nan=False,
                )
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, self.path)
        except (OSError, TypeError, ValueError) as exc:
            raise SettingsError(f"Could not write settings {self.path}: {exc}") from exc
        finally:
            try:
                temporary.unlink(missing_ok=True)
            except OSError:
                pass


__all__ = [
    "DEFAULT_SETTINGS",
    "DEFAULT_SETTINGS_PATH",
    "QUALITY_PRESETS",
    "SETTINGS_SCHEMA_VERSION",
    "SettingsError",
    "SettingsStore",
    "default_settings",
]
