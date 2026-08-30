"""Central application defaults and atomic JSON settings persistence."""

from __future__ import annotations

import json
import logging
import math
import os
import re
import threading
import uuid
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any

from .game_console import MAX_SELECTED_BRUSH, validate_console_key
from .hotkeys import (
    SUPPORTED_HOTKEY_CHOICES,
    is_supported_hotkey,
    normalize_hotkey,
)
from .models import PaintMode, SharpenMode


# Rust's client runs as RustClient.exe. The field stays editable for anyone
# running a differently named build.
DEFAULT_EXPECTED_PROCESS_NAME = "RustClient.exe"

# 2: stroke merging defaults on.  Merging never changes the painted result,
#    only how many strokes it takes, so a saved "off" from before that was
#    understood is lifted to "balanced" once.
SETTINGS_SCHEMA_VERSION = 2
DEFAULT_SETTINGS_PATH = Path(__file__).resolve().parent.parent / "data" / "settings.json"


QUALITY_PRESETS: dict[str, dict[str, int]] = {
    "very_fast": {"logical_width": 64, "color_count": 16},
    "fast": {"logical_width": 128, "color_count": 24},
    "balanced": {"logical_width": 256, "color_count": 32},
    "high": {"logical_width": 384, "color_count": 64},
    "very_high": {"logical_width": 512, "color_count": 96},
}

# 0 is "Unlimited": no palette cap at all - every distinct color the picker
# can reach paints separately, at a picker trip per color.  Counts past 256
# are cut by the app's own median cut; Pillow's quantizers stop at 256.
SUPPORTED_COLOR_COUNTS = {0, 8, 16, 24, 32, 48, 64, 96, 128, 256, 384, 512, 768, 1024}

# New installs start at 256: the richest palette whose picker-trip cost
# stays a rounding error of the paint time.  Fewer colors is a deliberate
# trade for speed; more (or Unlimited) is a deliberate trade the other way,
# whose cost the time estimate prices - neither is quietly chosen for
# someone who never opens the setting.
DEFAULT_COLOR_COUNT = 256

# Containers a recorded timelapse can be saved as.  Kept here rather than
# imported from the exporter so validating a settings file never has to
# load an image library.
SUPPORTED_VIDEO_FORMATS = {"mp4", "avi", "gif"}

_HEX_COLOR = re.compile(r"^#[0-9A-Fa-f]{6}$")


# Nested sections keep ownership clear while still producing ordinary JSON and
# dictionaries that are easy for Qt widgets and worker threads to consume.
DEFAULT_SETTINGS: dict[str, Any] = {
    "schema_version": SETTINGS_SCHEMA_VERSION,
    "image": {
        "scale_mode": "fit",
        "crop_alignment": "center",
        # Where Fill anchors the region it keeps, as [x, y] fractions of the
        # margin it gives away.  ``null`` follows crop_alignment; dragging the
        # crop on the Source tab stores the exact pair it was left at.
        "crop_focus": None,
        # New installs begin at the sign's highest practical detail. The GUI
        # adapts Max to the calibrated sign rather than promising pixels it
        # cannot display.
        "quality_preset": "max",
        # How boldly planning may simplify the image to paint faster; "exact"
        # reproduces the raw quantized image with the classic pipeline.
        "paint_mode": "quality",
        "logical_width": 256,
        "logical_height": 128,
        "color_count": DEFAULT_COLOR_COUNT,
        "dithering": False,
        # Put back the edge contrast the game's texture filtering softens;
        # "light" suits nearly everything, "strong" is for line art.
        "sharpen": SharpenMode.LIGHT.value,
        "background_mode": "unpainted",
        "background_color": "#FFFFFF",
        "transparent_pixels": "leave_unpainted",
        # Whether a partly transparent pixel is mixed into the background
        # color.  Off by default: painting has no alpha, and blending a soft
        # edge into the backdrop rings a cut-out subject with a halo of
        # background the artwork never had.
        "alpha_fill": False,
        "remove_background": False,
        # "auto" reads the key color off the artwork edges; "custom" uses
        # background_removal_color as typed.
        "background_removal_source": "auto",
        "background_removal_color": "#FFFFFF",
        "background_removal_tolerance": 12,
        # "subject" is the smart matcher: several key colors read off a band
        # around the artwork, grown from a strict match through a looser one.
        "background_removal_scope": "subject",
        "text_overlay": {
            "layers": [
                {
                    "text": "",
                    "font_family": "",
                    "font_size": 24,
                    # Font height as a fraction of the logical canvas height.
                    # The GUI keeps this fixed and re-derives font_size, so text
                    # stays the same size when the quality preset changes.
                    "size_ratio": 0.09375,
                    "color": "#FFFFFF",
                    "x": 0.5,
                    "y": 0.5,
                    "bold": False,
                    "italic": False,
                    # Blend the glyph edge colors without filtering the image
                    # underneath. This is the default for newly created text;
                    # saved layers can still opt into the crisp mode.
                    "smooth": True,
                    # A gradient runs from "color" into "gradient_color" over
                    # the text's own box; an outline rings every letter with
                    # that many logical pixels of "outline_color".
                    "gradient": False,
                    "gradient_direction": "vertical",
                    "gradient_color": "#FF9336",
                    "outline_width": 0,
                    "outline_color": "#000000",
                }
            ],
        },
    },
    "painting": {
        "brush_size": 0.15,
        "apply_brush_size": False,
        # Reuse a saved brush/grid measurement when a passive edge check proves
        # the same sign is still aligned. Any doubt falls back to full probes.
        "reuse_calibration": True,
        # Measure the sign's texel grid at the start of each job (needs brush
        # sizing).  An escape hatch, not a feature switch: a sign the probe
        # cannot read already falls back to the brush-derived grid on its own.
        "measure_texel_grid": True,
        "brush_direction": "low_to_high",
        "logical_pixel_spacing": 1.0,
        "stroke_speed_pixels_per_second": 700.0,
        # The Standard preset.  Holds and settles sit on the game's frame
        # floors (app/paint_timing.py); anything lower is run at the floor.
        "mouse_down_duration_seconds": 0.07,
        "delay_after_hue_seconds": 0.09,
        "delay_after_saturation_value_seconds": 0.09,
        "delay_after_brush_seconds": 0.07,
        "delay_between_strokes_seconds": 0.02,
        "delay_between_colors_seconds": 0.12,
        "stroke_interpolation_step_pixels": 4.0,
        "stroke_merge_mode": "balanced",
        # After painting, audit the canvas and repaint cells that stayed bare
        # or took the wrong color up to this many times. Every repair is audited,
        # including the final one. Two repairs by default; zero disables it.
        "verify_passes": 2,
        # Check each color as it goes down: capture the sign after its
        # strokes, repaint the cells that did not take, and capture again -
        # up to this many rounds per color.  Off: the game does not drop
        # presses, and the check misread painted cells as missing.
        "confirm_strokes": False,
        "confirm_max_rounds": 4,
        # Read the selected color back off the panel after every pick and
        # click again when it did not take.  A swallowed picker click paints
        # a whole color group in the previous group's color - 43 of 240
        # groups on one 1024x512 sign - and this is what stops it.
        "verify_color_picks": True,
    },
    "timelapse": {
        # Capture the calibrated canvas region while painting, one PNG frame
        # per interval, into a per-job session folder under data/timelapse.
        "enabled": False,
        "interval_seconds": 10,
        "capture_final_frame": True,
        # Speed used both to play a recording back in the app and to encode it
        # into a video file, so what the user watched is what they save.
        "playback_frame_rate": 15,
        "export_format": "avi",
        # The recordings page opens newest-first; users can instead put the
        # largest folders at either end for storage cleanup.
        "sort_order": "recent",
    },
    "hotkeys": {
        "start_resume": "F8",
        "abort": "F10",
    },
    # Settings that live inside Rust and that painting silently depends on.
    # They are typed into the game's console from Settings > Rust; see
    # app/game_console.py for what each one protects against.
    "game": {
        # The key that opens Rust's console: F1, or a modifier chord on a
        # laptop whose function row hides behind Fn.
        "console_key": "F1",
        # Rust's brush index (0 is the first).  Whatever brush the profile
        # was measured on; the app was developed on brush 3.
        "selected_brush": 3,
        # Stamp spacing as a fraction of the brush diameter (0..1).  The
        # drag speeds were tuned at 0.01; Rust's own default is 0.25.
        "brush_spacing": 0.01,
    },
    "safety": {
        "countdown_seconds": 3,
        "pause_on_mouse_move": True,
        "mouse_move_pause_threshold_pixels": 24,
        "require_rust_foreground": True,
        "auto_resume_on_focus_return": True,
        "auto_resume_focus_retry_seconds": 10,
        "expected_process_name": DEFAULT_EXPECTED_PROCESS_NAME,
        "expected_window_title_contains": "Rust",
        "verify_calibrated_ui": False,
        # Every so often, save the sign, jump, and reopen it, so a server
        # that kicks idle players sees one moving.
        "anti_afk_enabled": True,
        "anti_afk_interval_minutes": 30,
        # Pause when the painting UI's calibrated widgets leave the screen:
        # a kick, a server restart, or the sign closed by hand.
        "ui_guard_enabled": True,
    },
    "execution": {
        "dry_run": False,
        "debug_mouse_logging": False,
    },
    "ui": {
        "selected_profile_id": None,
        "last_image_path": None,
        "window_geometry": None,
        "show_calibration_overlay": True,
        "show_status_overlay": True,
        "smooth_rust_preview": True,
        # Zero until the first-run guide has been displayed. Keeping a version
        # instead of a boolean leaves room for a deliberately revised guide.
        "tutorial_version_seen": 0,
    },
}


class SettingsError(ValueError):
    """Raised when settings cannot be decoded, validated, or written."""


_MISSING = object()


def default_settings() -> dict[str, Any]:
    """Return an independent copy so callers cannot mutate global defaults."""

    return deepcopy(DEFAULT_SETTINGS)


def _migrate(settings: dict[str, Any], stored_version: Any) -> None:
    """Bring a settings document saved by an older schema up to date."""

    try:
        version = int(stored_version)
    except (TypeError, ValueError):
        version = 1
    painting = settings.get("painting")
    if version < 2 and isinstance(painting, dict):
        if painting.get("stroke_merge_mode") == "off":
            painting["stroke_merge_mode"] = "balanced"
            logging.getLogger("rust_painter.settings").info(
                "Stroke merging was off in the saved settings; it is now "
                "balanced, which paints the same picture in fewer strokes"
            )


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
    timelapse = settings.get("timelapse")
    hotkeys = settings.get("hotkeys")
    game = settings.get("game")
    safety = settings.get("safety")
    execution = settings.get("execution")
    ui = settings.get("ui")
    for label, section in (
        ("image", image),
        ("painting", painting),
        ("timelapse", timelapse),
        ("hotkeys", hotkeys),
        ("game", game),
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
    focus = image.get("crop_focus")
    if focus is not None:
        if not isinstance(focus, Sequence) or isinstance(focus, (str, bytes)):
            raise SettingsError("image.crop_focus must be null or [x, y]")
        if len(focus) != 2 or any(
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not math.isfinite(float(value))
            or not 0.0 <= float(value) <= 1.0
            for value in focus
        ):
            raise SettingsError("image.crop_focus must be two fractions from 0 to 1")
    # "max" and "custom" are GUI-computed presets: the sign measurement or the
    # spin boxes decide the dimensions, so they own no entry in the preset
    # table.  Rejecting them here silently discarded every settings save made
    # while one of them was selected.
    if not isinstance(image.get("quality_preset"), str) or image.get(
        "quality_preset"
    ) not in {*QUALITY_PRESETS, "max", "custom"}:
        raise SettingsError("image.quality_preset is invalid")
    paint_modes = {mode.value for mode in PaintMode}
    if not isinstance(image.get("paint_mode"), str) or image.get(
        "paint_mode"
    ) not in paint_modes:
        raise SettingsError(
            "image.paint_mode must be one of: " + ", ".join(sorted(paint_modes))
        )
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
    if image.get("sharpen") not in {mode.value for mode in SharpenMode}:
        raise SettingsError("image.sharpen must be off, light, or strong")
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
    if not isinstance(image.get("alpha_fill"), bool):
        raise SettingsError("image.alpha_fill must be true or false")
    if not isinstance(image.get("remove_background"), bool):
        raise SettingsError("image.remove_background must be true or false")
    if not isinstance(image.get("background_removal_source"), str) or image.get(
        "background_removal_source"
    ) not in {"auto", "custom"}:
        raise SettingsError("image.background_removal_source must be auto or custom")
    removal_color = image.get("background_removal_color")
    if not isinstance(removal_color, str) or _HEX_COLOR.fullmatch(removal_color) is None:
        raise SettingsError(
            "image.background_removal_color must use #RRGGBB format"
        )
    tolerance = image.get("background_removal_tolerance")
    if (
        isinstance(tolerance, bool)
        or not isinstance(tolerance, (int, float))
        or not math.isfinite(float(tolerance))
        or not 0 <= float(tolerance) <= 100
    ):
        raise SettingsError(
            "image.background_removal_tolerance must be between 0 and 100"
        )
    if not isinstance(image.get("background_removal_scope"), str) or image.get(
        "background_removal_scope"
    ) not in {"subject", "connected", "everywhere"}:
        raise SettingsError(
            "image.background_removal_scope must be subject, connected, or everywhere"
        )
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
        # Colors written before gradients and outlines existed are absent
        # rather than wrong, so only the keys a document does carry are read.
        for key in ("color", "gradient_color", "outline_color"):
            text_color = layer.get(key, "#FFFFFF" if key == "color" else None)
            if text_color is None:
                continue
            if (
                not isinstance(text_color, str)
                or _HEX_COLOR.fullmatch(text_color) is None
            ):
                raise SettingsError(f"{label}.{key} must use #RRGGBB format")
        direction = layer.get("gradient_direction")
        if direction is not None and direction not in {
            "vertical",
            "horizontal",
            "diagonal",
        }:
            raise SettingsError(
                f"{label}.gradient_direction must be vertical, horizontal, "
                "or diagonal"
            )
        outline_width = layer.get("outline_width")
        if outline_width is not None and (
            isinstance(outline_width, bool)
            or not isinstance(outline_width, int)
            or not 0 <= outline_width <= 16
        ):
            raise SettingsError(
                f"{label}.outline_width must be an integer from 0 to 16"
            )
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
        if not isinstance(layer.get("smooth", False), bool):
            raise SettingsError(f"{label}.smooth must be true or false")
        if not isinstance(layer.get("gradient", False), bool):
            raise SettingsError(f"{label}.gradient must be true or false")

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
    if not isinstance(painting.get("reuse_calibration"), bool):
        raise SettingsError("painting.reuse_calibration must be true or false")
    if not isinstance(painting.get("measure_texel_grid"), bool):
        raise SettingsError("painting.measure_texel_grid must be true or false")
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
    verify_passes = painting.get("verify_passes", 2)
    if (
        isinstance(verify_passes, bool)
        or not isinstance(verify_passes, int)
        or not 0 <= verify_passes <= 5
    ):
        raise SettingsError("painting.verify_passes must be an integer from 0 to 5")
    if not isinstance(painting.get("confirm_strokes", False), bool):
        raise SettingsError("painting.confirm_strokes must be true or false")
    confirm_rounds = painting.get("confirm_max_rounds", 4)
    if (
        isinstance(confirm_rounds, bool)
        or not isinstance(confirm_rounds, int)
        or not 1 <= confirm_rounds <= 8
    ):
        raise SettingsError("painting.confirm_max_rounds must be an integer from 1 to 8")
    if not isinstance(painting.get("verify_color_picks", True), bool):
        raise SettingsError("painting.verify_color_picks must be true or false")

    assert isinstance(timelapse, Mapping)
    for key in ("enabled", "capture_final_frame"):
        if not isinstance(timelapse.get(key), bool):
            raise SettingsError(f"timelapse.{key} must be true or false")
    interval = timelapse.get("interval_seconds")
    if (
        isinstance(interval, bool)
        or not isinstance(interval, (int, float))
        or not math.isfinite(float(interval))
        or not 1 <= float(interval) <= 3600
    ):
        raise SettingsError(
            "timelapse.interval_seconds must be between 1 and 3600 seconds"
        )
    frame_rate = timelapse.get("playback_frame_rate", 15)
    if (
        isinstance(frame_rate, bool)
        or not isinstance(frame_rate, int)
        or not 1 <= frame_rate <= 60
    ):
        raise SettingsError(
            "timelapse.playback_frame_rate must be an integer from 1 to 60"
        )
    if timelapse.get("export_format", "avi") not in SUPPORTED_VIDEO_FORMATS:
        raise SettingsError(
            "timelapse.export_format must be one of: "
            + ", ".join(sorted(SUPPORTED_VIDEO_FORMATS))
        )
    if timelapse.get("sort_order", "recent") not in {
        "recent",
        "largest",
        "smallest",
    }:
        raise SettingsError(
            "timelapse.sort_order must be recent, largest, or smallest"
        )

    assert isinstance(hotkeys, Mapping)
    for key in ("start_resume", "abort"):
        if not isinstance(hotkeys.get(key), str) or not str(hotkeys[key]).strip():
            raise SettingsError(f"hotkeys.{key} must be a non-empty string")
        if not is_supported_hotkey(hotkeys[key]):
            raise SettingsError(
                f"hotkeys.{key} must be one of: " + ", ".join(SUPPORTED_HOTKEY_CHOICES)
            )
    normalized_hotkeys = {
        normalize_hotkey(hotkeys[key]) for key in ("start_resume", "abort")
    }
    if len(normalized_hotkeys) != 2:
        raise SettingsError("Start/pause and abort hotkeys must be different")

    assert isinstance(safety, Mapping)
    countdown = safety.get("countdown_seconds")
    if isinstance(countdown, bool) or not isinstance(countdown, int) or countdown < 0:
        raise SettingsError("safety.countdown_seconds must be a non-negative integer")
    for key in (
        "pause_on_mouse_move",
        "require_rust_foreground",
        "auto_resume_on_focus_return",
        "verify_calibrated_ui",
        "anti_afk_enabled",
        "ui_guard_enabled",
    ):
        if not isinstance(safety.get(key), bool):
            raise SettingsError(f"safety.{key} must be true or false")
    afk_interval = safety.get("anti_afk_interval_minutes")
    if isinstance(afk_interval, bool) or not isinstance(afk_interval, int) or afk_interval < 1:
        raise SettingsError("safety.anti_afk_interval_minutes must be a positive integer")
    for key in ("expected_process_name", "expected_window_title_contains"):
        if safety.get(key) is not None and not isinstance(safety.get(key), str):
            raise SettingsError(f"safety.{key} must be a string or null")
    may_be_zero = {"mouse_move_tolerance_pixels"}
    for key in (
        "focus_check_interval_seconds",
        "auto_resume_focus_retry_seconds",
        "safety_poll_interval_seconds",
        "mouse_move_pause_threshold_pixels",
        "mouse_move_tolerance_pixels",
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
        if key not in may_be_zero and value == 0:
            raise SettingsError(f"safety.{key} must be positive")

    assert isinstance(game, Mapping)
    console_key = game.get("console_key")
    if not isinstance(console_key, str) or not console_key.strip():
        raise SettingsError("game.console_key must be a non-empty string")
    try:
        validate_console_key(console_key)
    except ValueError as exc:
        raise SettingsError(f"game.console_key is not a key the app can press: {exc}") from exc
    selected_brush = game.get("selected_brush")
    if (
        isinstance(selected_brush, bool)
        or not isinstance(selected_brush, int)
        or not 0 <= selected_brush <= MAX_SELECTED_BRUSH
    ):
        raise SettingsError(f"game.selected_brush must be an integer from 0 to {MAX_SELECTED_BRUSH}")
    spacing = game.get("brush_spacing")
    if (
        isinstance(spacing, bool)
        or not isinstance(spacing, (int, float))
        or not math.isfinite(float(spacing))
        or not 0.0 <= float(spacing) <= 1.0
    ):
        raise SettingsError("game.brush_spacing must be a number from 0 to 1")

    assert isinstance(execution, Mapping)
    for key in ("dry_run", "debug_mouse_logging"):
        if not isinstance(execution.get(key), bool):
            raise SettingsError(f"execution.{key} must be true or false")

    assert isinstance(ui, Mapping)
    for key in ("selected_profile_id", "last_image_path"):
        if ui.get(key) is not None and not isinstance(ui.get(key), str):
            raise SettingsError(f"ui.{key} must be a string or null")
    tutorial_version = ui.get("tutorial_version_seen")
    if (
        isinstance(tutorial_version, bool)
        or not isinstance(tutorial_version, int)
        or tutorial_version < 0
    ):
        raise SettingsError("ui.tutorial_version_seen must be a non-negative integer")


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
            _migrate(result, raw.get("schema_version"))
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
    "DEFAULT_COLOR_COUNT",
    "DEFAULT_SETTINGS",
    "DEFAULT_SETTINGS_PATH",
    "QUALITY_PRESETS",
    "SETTINGS_SCHEMA_VERSION",
    "SUPPORTED_VIDEO_FORMATS",
    "SettingsError",
    "SettingsStore",
    "default_settings",
]
