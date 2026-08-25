"""Persistent calibration profiles.

Profiles deliberately contain only calibration/display facts.  Image processing,
timing, hotkeys, and other application preferences live in :mod:`app.settings`.
The JSON reader accepts the camelCase form used by early prototypes as well as
the canonical snake_case form, while the writer emits one stable schema.
"""

from __future__ import annotations

import json
import math
import os
import threading
import uuid
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from .models import (
    HueDirection,
    SaturationDirection,
    ScreenRect,
    ValueDirection,
)


PROFILE_SCHEMA_VERSION = 1
DEFAULT_PROFILE_STORE_PATH = (
    Path(__file__).resolve().parent.parent / "data" / "profiles" / "profiles.json"
)

# A short alias is convenient for GUI/calibration callers and remains the exact
# same runtime type as the core coordinate model.
Rect = ScreenRect


class ProfileError(ValueError):
    """Base error for invalid profile data or operations."""


class ProfileDataError(ProfileError):
    """Raised when a profile JSON document cannot be decoded or validated."""


class ProfileNotFoundError(ProfileError):
    """Raised when a requested profile does not exist."""


class DuplicateProfileNameError(ProfileError):
    """Raised when two profiles would have the same display name."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _mapping(value: object, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ProfileDataError(f"{label} must be a JSON object")
    return value


def _first(mapping: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    for key in keys:
        if key in mapping:
            return mapping[key]
    return default


def _nonempty_text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ProfileDataError(f"{label} must be a non-empty string")
    return value.strip()


def _optional_float(value: object, label: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        raise ProfileDataError(f"{label} must be a number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ProfileDataError(f"{label} must be a number") from exc
    if not math.isfinite(result):
        raise ProfileDataError(f"{label} must be finite")
    return result


def _rect_from(value: object, label: str, *, optional: bool = False) -> Rect | None:
    if value is None and optional:
        return None
    try:
        return Rect.from_dict(dict(_mapping(value, label)))
    except (KeyError, TypeError, ValueError) as exc:
        raise ProfileDataError(f"Invalid {label}: {exc}") from exc


def _rect_dict(value: Rect | None) -> dict[str, int] | None:
    return None if value is None else value.to_dict()


def _enum_value(value: object, enum_type: type, aliases: Mapping[str, str], label: str) -> str:
    raw = value.value if hasattr(value, "value") else value
    key = str(raw).strip().lower().replace("-", "_").replace(" ", "_")
    key = aliases.get(key, key)
    try:
        return enum_type(key).value
    except ValueError as exc:
        choices = ", ".join(item.value for item in enum_type)
        raise ProfileDataError(f"{label} must be one of: {choices}") from exc


_HUE_ALIASES = {
    "top_bottom": "top_to_bottom",
    "bottom_top": "bottom_to_top",
}
_SATURATION_ALIASES = {
    "left_to_right": "left_low",
    "left_low_right_high": "left_low",
    "right_high": "left_low",
    "right_to_left": "left_high",
    "left_high_right_low": "left_high",
    "right_low": "left_high",
}
_VALUE_ALIASES = {
    "top_to_bottom": "top_bright",
    "top_bright_bottom_dark": "top_bright",
    "bottom_to_top": "top_dark",
    "top_dark_bottom_bright": "top_dark",
}


@dataclass(frozen=True, slots=True)
class MonitorMetadata:
    """Display facts recorded at calibration time, in physical coordinates."""

    name: str
    rect: Rect
    available_rect: Rect | None = None
    logical_rect: Rect | None = None
    device_pixel_ratio: float = 1.0
    logical_dpi_x: float | None = None
    logical_dpi_y: float | None = None
    physical_dpi_x: float | None = None
    physical_dpi_y: float | None = None
    primary: bool = False

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ProfileDataError("Monitor name must not be empty")
        # Qt pads some EDID-derived screen names ("ATNA40CU05-0 ").  Deserialization
        # strips names, so an unstripped captured name would never compare equal to
        # its own saved copy and every calibration would look like a layout change.
        if self.name != self.name.strip():
            object.__setattr__(self, "name", self.name.strip())
        if not math.isfinite(self.device_pixel_ratio) or self.device_pixel_ratio <= 0:
            raise ProfileDataError("Monitor device_pixel_ratio must be positive")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "rect": self.rect.to_dict(),
            "availableRect": _rect_dict(self.available_rect),
            "logicalRect": _rect_dict(self.logical_rect),
            "devicePixelRatio": self.device_pixel_ratio,
            "logicalDpiX": self.logical_dpi_x,
            "logicalDpiY": self.logical_dpi_y,
            "physicalDpiX": self.physical_dpi_x,
            "physicalDpiY": self.physical_dpi_y,
            "primary": self.primary,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "MonitorMetadata":
        value = _mapping(value, "monitor")
        rect_value = _first(value, "rect", "geometry")
        return cls(
            name=_nonempty_text(value.get("name", "Unknown display"), "monitor name"),
            rect=_rect_from(rect_value, "monitor rectangle"),  # type: ignore[arg-type]
            available_rect=_rect_from(
                _first(value, "availableRect", "available_rect"),
                "monitor available rectangle",
                optional=True,
            ),
            logical_rect=_rect_from(
                _first(value, "logicalRect", "logical_rect"),
                "monitor logical rectangle",
                optional=True,
            ),
            device_pixel_ratio=(
                1.0
                if _first(value, "devicePixelRatio", "device_pixel_ratio") is None
                else _optional_float(
                    _first(value, "devicePixelRatio", "device_pixel_ratio"),
                    "monitor device pixel ratio",
                )
            ),  # type: ignore[arg-type]
            logical_dpi_x=_optional_float(
                _first(value, "logicalDpiX", "logical_dpi_x"), "monitor logical DPI X"
            ),
            logical_dpi_y=_optional_float(
                _first(value, "logicalDpiY", "logical_dpi_y"), "monitor logical DPI Y"
            ),
            physical_dpi_x=_optional_float(
                _first(value, "physicalDpiX", "physical_dpi_x"), "monitor physical DPI X"
            ),
            physical_dpi_y=_optional_float(
                _first(value, "physicalDpiY", "physical_dpi_y"), "monitor physical DPI Y"
            ),
            primary=bool(value.get("primary", False)),
        )


@dataclass(frozen=True, slots=True)
class DisplayMetadata:
    """The monitor layout against which a profile was calibrated."""

    monitors: tuple[MonitorMetadata, ...] = ()
    virtual_screen: Rect | None = None
    captured_at: str = field(default_factory=_utc_now)
    coordinate_space: str = "physical"

    def __post_init__(self) -> None:
        if self.coordinate_space not in {"physical", "logical", "unknown"}:
            raise ProfileDataError("Unknown display coordinate space")

    @property
    def summary(self) -> str:
        count = len(self.monitors)
        noun = "display" if count == 1 else "displays"
        if self.virtual_screen is None:
            return f"{count} {noun}"
        rect = self.virtual_screen
        return f"{count} {noun}; {rect.width} x {rect.height} at ({rect.left}, {rect.top})"

    def differences(self, current: "DisplayMetadata", *, dpr_tolerance: float = 0.02) -> list[str]:
        """Return user-facing reasons why two display configurations differ."""

        changes: list[str] = []
        if self.coordinate_space != current.coordinate_space:
            changes.append("display coordinate mode changed")
        if self.virtual_screen != current.virtual_screen:
            changes.append("virtual desktop bounds changed")
        if len(self.monitors) != len(current.monitors):
            changes.append("display count changed")

        expected = {monitor.name.casefold(): monitor for monitor in self.monitors}
        actual = {monitor.name.casefold(): monitor for monitor in current.monitors}
        if set(expected) != set(actual):
            changes.append("connected displays changed")
        for name in sorted(set(expected) & set(actual)):
            before, after = expected[name], actual[name]
            if before.rect != after.rect:
                changes.append(f"{before.name} bounds changed")
            if abs(before.device_pixel_ratio - after.device_pixel_ratio) > dpr_tolerance:
                changes.append(f"{before.name} scaling changed")
        return changes

    def is_compatible(self, current: "DisplayMetadata", *, dpr_tolerance: float = 0.02) -> bool:
        return not self.differences(current, dpr_tolerance=dpr_tolerance)

    def to_dict(self) -> dict[str, Any]:
        return {
            "capturedAt": self.captured_at,
            "coordinateSpace": self.coordinate_space,
            "virtualScreen": _rect_dict(self.virtual_screen),
            "monitors": [monitor.to_dict() for monitor in self.monitors],
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "DisplayMetadata":
        value = _mapping(value, "display metadata")
        monitor_values = value.get("monitors", [])
        if not isinstance(monitor_values, Sequence) or isinstance(monitor_values, (str, bytes)):
            raise ProfileDataError("display monitors must be a JSON array")
        return cls(
            monitors=tuple(MonitorMetadata.from_dict(item) for item in monitor_values),
            virtual_screen=_rect_from(
                _first(value, "virtualScreen", "virtual_screen"),
                "virtual screen",
                optional=True,
            ),
            captured_at=str(_first(value, "capturedAt", "captured_at", default=_utc_now())),
            coordinate_space=str(
                _first(value, "coordinateSpace", "coordinate_space", default="physical")
            ),
        )


@dataclass(slots=True)
class CalibrationProfile:
    """Named rectangles and color-picker orientation for one Rust sign/UI layout."""

    id: str
    name: str
    canvas: Rect | None = None
    color_box: Rect | None = None
    hue_bar: Rect | None = None
    brush_size_box: Rect | None = None
    # Rust's "clear the sign" control.  Painting wipes the sign with it after
    # measuring the brush, so the probe strokes never end up under the artwork.
    clear_button: Rect | None = None
    # Rust's "Save changes" control.  The anti-AFK break clicks it to leave
    # the painting UI, jumps, and reopens the sign with a click.
    save_button: Rect | None = None
    # Optional HUD number crops.  They are read while the painting UI is closed
    # for an anti-AFK break, when Rust's survival HUD is visible again.
    hunger: Rect | None = None
    thirst: Rect | None = None
    # Rust's download control, which writes the sign's texture to the desktop
    # texel for texel.  With it calibrated the probes and the touch-up pass
    # read the sign exactly instead of through a screenshot.
    download_button: Rect | None = None
    hue_direction: str = HueDirection.BOTTOM_TO_TOP.value
    saturation_direction: str = SaturationDirection.LEFT_LOW.value
    value_direction: str = ValueDirection.TOP_BRIGHT.value
    display: DisplayMetadata | None = None
    created_at: str = field(default_factory=_utc_now)
    updated_at: str = field(default_factory=_utc_now)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.id = _nonempty_text(self.id, "profile id")
        self.name = _nonempty_text(self.name, "profile name")
        # Rust's current sign-painter layout has one fixed orientation.  Older
        # profiles retain their serialized values for compatibility, but the
        # application deliberately normalizes them so a stray option cannot
        # invert every painted color.
        self.hue_direction = HueDirection.BOTTOM_TO_TOP.value
        self.saturation_direction = SaturationDirection.LEFT_LOW.value
        self.value_direction = ValueDirection.TOP_BRIGHT.value
        if not isinstance(self.metadata, dict):
            raise ProfileDataError("profile metadata must be a JSON object")

    @classmethod
    def new(cls, name: str = "New Profile", **values: Any) -> "CalibrationProfile":
        return cls(id=uuid.uuid4().hex, name=name, **values)

    @property
    def profile_id(self) -> str:
        return self.id

    @property
    def display_metadata(self) -> DisplayMetadata | None:
        return self.display

    @display_metadata.setter
    def display_metadata(self, value: DisplayMetadata | None) -> None:
        self.display = value

    @property
    def is_ready(self) -> bool:
        return self.canvas is not None and self.color_box is not None and self.hue_bar is not None

    @property
    def is_calibrated(self) -> bool:
        return self.is_ready

    @property
    def calibration_status(self) -> dict[str, bool]:
        return {
            "canvas": self.canvas is not None,
            "color_box": self.color_box is not None,
            "hue_bar": self.hue_bar is not None,
            "brush_size_box": self.brush_size_box is not None,
            "clear_button": self.clear_button is not None,
            "save_button": self.save_button is not None,
            "hunger": self.hunger is not None,
            "thirst": self.thirst is not None,
            "download_button": self.download_button is not None,
        }

    @property
    def canvas_summary(self) -> str:
        if self.canvas is None:
            return "Not calibrated"
        return (
            f"{self.canvas.width} x {self.canvas.height} "
            f"({self.canvas.aspect_ratio:.4f}:1)"
        )

    @property
    def display_label(self) -> str:
        if self.canvas is None:
            return self.name
        return f"{self.name} - {self.canvas.width} x {self.canvas.height}"

    def touch(self) -> None:
        self.updated_at = _utc_now()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schemaVersion": PROFILE_SCHEMA_VERSION,
            "id": self.id,
            "name": self.name,
            "canvas": _rect_dict(self.canvas),
            "colorBox": _rect_dict(self.color_box),
            "hueBar": _rect_dict(self.hue_bar),
            "brushSizeBox": _rect_dict(self.brush_size_box),
            "clearButton": _rect_dict(self.clear_button),
            "saveButton": _rect_dict(self.save_button),
            "hunger": _rect_dict(self.hunger),
            "thirst": _rect_dict(self.thirst),
            "downloadButton": _rect_dict(self.download_button),
            "pickerDirections": {
                "hue": self.hue_direction,
                "saturation": self.saturation_direction,
                "value": self.value_direction,
            },
            "display": None if self.display is None else self.display.to_dict(),
            "createdAt": self.created_at,
            "updatedAt": self.updated_at,
            "metadata": deepcopy(self.metadata),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "CalibrationProfile":
        value = _mapping(value, "profile")
        directions_value = _first(value, "pickerDirections", "picker_directions", default={})
        directions = _mapping(directions_value or {}, "picker directions")
        display_value = _first(value, "display", "displayMetadata", "display_metadata")
        metadata = value.get("metadata", {})
        return cls(
            id=str(value.get("id") or value.get("profile_id") or uuid.uuid4().hex),
            name=_nonempty_text(value.get("name"), "profile name"),
            canvas=_rect_from(value.get("canvas"), "canvas", optional=True),
            color_box=_rect_from(
                _first(value, "colorBox", "color_box"), "color box", optional=True
            ),
            hue_bar=_rect_from(_first(value, "hueBar", "hue_bar"), "hue bar", optional=True),
            brush_size_box=_rect_from(
                _first(value, "brushSizeBox", "brush_size_box"),
                "brush size box",
                optional=True,
            ),
            clear_button=_rect_from(
                _first(value, "clearButton", "clear_button"),
                "clear button",
                optional=True,
            ),
            save_button=_rect_from(
                _first(value, "saveButton", "save_button"),
                "save button",
                optional=True,
            ),
            hunger=_rect_from(value.get("hunger"), "hunger", optional=True),
            thirst=_rect_from(value.get("thirst"), "thirst", optional=True),
            download_button=_rect_from(
                _first(value, "downloadButton", "download_button"),
                "download button",
                optional=True,
            ),
            hue_direction=_first(
                directions,
                "hue",
                "hueDirection",
                "hue_direction",
                default=_first(value, "hueDirection", "hue_direction", default="top_to_bottom"),
            ),
            saturation_direction=_first(
                directions,
                "saturation",
                "saturationDirection",
                "saturation_direction",
                default=_first(
                    value, "saturationDirection", "saturation_direction", default="left_low"
                ),
            ),
            value_direction=_first(
                directions,
                "value",
                "valueDirection",
                "value_direction",
                default=_first(value, "valueDirection", "value_direction", default="top_bright"),
            ),
            display=None
            if display_value is None
            else DisplayMetadata.from_dict(_mapping(display_value, "display metadata")),
            created_at=str(_first(value, "createdAt", "created_at", default=_utc_now())),
            updated_at=str(_first(value, "updatedAt", "updated_at", default=_utc_now())),
            metadata=deepcopy(dict(_mapping(metadata, "profile metadata"))),
        )


# The shorter name reads naturally throughout the GUI.
Profile = CalibrationProfile


class ProfileStore:
    """Atomic JSON CRUD for calibration profiles.

    ``path`` may be a JSON filename or a directory.  A directory is resolved to
    ``profiles.json`` so callers can simply pass ``data/profiles``.
    """

    def __init__(self, path: str | Path = DEFAULT_PROFILE_STORE_PATH) -> None:
        candidate = Path(path).expanduser()
        self.path = candidate if candidate.suffix.lower() == ".json" else candidate / "profiles.json"
        self._lock = threading.RLock()

    def list_profiles(self) -> list[Profile]:
        with self._lock:
            profiles, _ = self._load()
            return profiles

    # Friendly aliases for GUI code and older prototypes.
    load = list_profiles
    all = list_profiles

    def get(self, profile_id_or_name: str) -> Profile | None:
        lookup = str(profile_id_or_name).strip()
        with self._lock:
            profiles, _ = self._load()
        for profile in profiles:
            if profile.id == lookup or profile.name.casefold() == lookup.casefold():
                return profile
        return None

    def require(self, profile_id_or_name: str) -> Profile:
        profile = self.get(profile_id_or_name)
        if profile is None:
            raise ProfileNotFoundError(f"Profile not found: {profile_id_or_name}")
        return profile

    def get_default(self) -> Profile | None:
        with self._lock:
            profiles, default_id = self._load()
        if not profiles:
            return None
        return next((profile for profile in profiles if profile.id == default_id), profiles[0])

    default_profile = get_default

    def ensure_default_profile(self, name: str = "Default") -> Profile:
        existing = self.get_default()
        if existing is not None:
            return existing
        return self.create(name, make_default=True)

    def create(self, name: str, *, make_default: bool | None = None, **values: Any) -> Profile:
        profile = Profile.new(name, **values)
        return self.save(profile, make_default=make_default)

    def save(self, profile: Profile, *, make_default: bool | None = None) -> Profile:
        # Round-tripping here validates nested metadata and detaches caller-owned
        # mutable dictionaries before persistence.
        candidate = Profile.from_dict(profile.to_dict())
        candidate.updated_at = _utc_now()
        with self._lock:
            profiles, default_id = self._load()
            duplicate = next(
                (
                    item
                    for item in profiles
                    if item.id != candidate.id and item.name.casefold() == candidate.name.casefold()
                ),
                None,
            )
            if duplicate is not None:
                raise DuplicateProfileNameError(
                    f"A profile named {candidate.name!r} already exists"
                )

            index = next((i for i, item in enumerate(profiles) if item.id == candidate.id), None)
            if index is None:
                profiles.append(candidate)
            else:
                profiles[index] = candidate

            if make_default is True or (default_id is None and profiles):
                default_id = candidate.id
            elif make_default is False and default_id == candidate.id:
                default_id = next((item.id for item in profiles if item.id != candidate.id), None)
            self._write(profiles, default_id)
        return Profile.from_dict(candidate.to_dict())

    def rename(self, profile_id_or_name: str, new_name: str) -> Profile:
        profile = self.require(profile_id_or_name)
        profile.name = _nonempty_text(new_name, "profile name")
        return self.save(profile)

    def delete(self, profile_id_or_name: str) -> bool:
        lookup = str(profile_id_or_name).strip()
        with self._lock:
            profiles, default_id = self._load()
            remaining = [
                item
                for item in profiles
                if item.id != lookup and item.name.casefold() != lookup.casefold()
            ]
            if len(remaining) == len(profiles):
                return False
            if default_id not in {item.id for item in remaining}:
                default_id = remaining[0].id if remaining else None
            self._write(remaining, default_id)
        return True

    def set_default(self, profile_id_or_name: str | None) -> Profile | None:
        with self._lock:
            profiles, _ = self._load()
            if profile_id_or_name is None:
                self._write(profiles, None)
                return None
            lookup = str(profile_id_or_name).strip()
            selected = next(
                (
                    item
                    for item in profiles
                    if item.id == lookup or item.name.casefold() == lookup.casefold()
                ),
                None,
            )
            if selected is None:
                raise ProfileNotFoundError(f"Profile not found: {profile_id_or_name}")
            self._write(profiles, selected.id)
            return Profile.from_dict(selected.to_dict())

    def import_profile(self, path: str | Path, *, make_default: bool | None = None) -> Profile:
        try:
            with Path(path).open("r", encoding="utf-8-sig") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileDataError(f"Could not read profile {path}: {exc}") from exc
        return self.save(Profile.from_dict(_mapping(value, "profile")), make_default=make_default)

    @staticmethod
    def export_profile(profile: Profile, path: str | Path) -> None:
        destination = Path(path)
        _atomic_json_write(destination, profile.to_dict())

    def _load(self) -> tuple[list[Profile], str | None]:
        if not self.path.exists():
            return [], None
        try:
            with self.path.open("r", encoding="utf-8-sig") as handle:
                raw = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ProfileDataError(f"Could not read profile store {self.path}: {exc}") from exc

        if isinstance(raw, list):
            values, default_id = raw, None
        elif isinstance(raw, Mapping) and "profiles" in raw:
            values = raw["profiles"]
            default_id = _first(raw, "defaultProfileId", "default_profile_id")
        elif isinstance(raw, Mapping) and "name" in raw:
            values, default_id = [raw], raw.get("id")
        else:
            raise ProfileDataError("Profile store must contain a profile array")
        if not isinstance(values, list):
            raise ProfileDataError("Profile store 'profiles' value must be an array")

        profiles = [Profile.from_dict(_mapping(value, "profile")) for value in values]
        ids: set[str] = set()
        names: set[str] = set()
        for profile in profiles:
            if profile.id in ids:
                raise ProfileDataError(f"Duplicate profile id: {profile.id}")
            folded = profile.name.casefold()
            if folded in names:
                raise ProfileDataError(f"Duplicate profile name: {profile.name}")
            ids.add(profile.id)
            names.add(folded)
        return profiles, None if default_id is None else str(default_id)

    def _write(self, profiles: Sequence[Profile], default_id: str | None) -> None:
        document = {
            "schemaVersion": PROFILE_SCHEMA_VERSION,
            "defaultProfileId": default_id,
            "profiles": [profile.to_dict() for profile in profiles],
        }
        _atomic_json_write(self.path, document)


def _atomic_json_write(path: Path, value: object) -> None:
    """Write JSON without leaving a half-written settings/profile file."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with temporary.open("x", encoding="utf-8", newline="\n") as handle:
            json.dump(
                value,
                handle,
                indent=2,
                ensure_ascii=False,
                sort_keys=False,
                allow_nan=False,
            )
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except (OSError, TypeError, ValueError) as exc:
        raise ProfileDataError(f"Could not write JSON file {path}: {exc}") from exc
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError:
            pass


__all__ = [
    "CalibrationProfile",
    "DEFAULT_PROFILE_STORE_PATH",
    "DisplayMetadata",
    "DuplicateProfileNameError",
    "MonitorMetadata",
    "PROFILE_SCHEMA_VERSION",
    "Profile",
    "ProfileDataError",
    "ProfileError",
    "ProfileNotFoundError",
    "ProfileStore",
    "Rect",
]
