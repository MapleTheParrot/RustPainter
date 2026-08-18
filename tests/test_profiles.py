from __future__ import annotations

import json

import pytest

from app.profiles import (
    DisplayMetadata,
    DuplicateProfileNameError,
    MonitorMetadata,
    Profile,
    ProfileDataError,
    ProfileStore,
    Rect,
)


def _complete_profile() -> Profile:
    display = DisplayMetadata(
        monitors=(
            MonitorMetadata(
                name=r"\\.\DISPLAY1",
                rect=Rect(0, 0, 2560, 1440),
                available_rect=Rect(0, 0, 2560, 1400),
                logical_rect=Rect(0, 0, 2048, 1152),
                device_pixel_ratio=1.25,
                logical_dpi_x=96.0,
                logical_dpi_y=96.0,
                physical_dpi_x=109.0,
                physical_dpi_y=109.0,
                primary=True,
            ),
            MonitorMetadata(
                name=r"\\.\DISPLAY2",
                rect=Rect(-1920, 120, 1920, 1080),
                device_pixel_ratio=1.0,
            ),
        ),
        virtual_screen=Rect(-1920, 0, 4480, 1440),
    )
    return Profile(
        id="wooden-sign",
        name="Large Wooden Sign",
        canvas=Rect(-1500, 250, 1048, 526),
        color_box=Rect(1680, 610, 254, 252),
        hue_bar=Rect(1937, 610, 40, 252),
        brush_slider=Rect(1760, 392, 265, 34),
        brush_preview=Rect(1400, 970, 160, 160),
        square_shape_button=Rect(1700, 340, 36, 36),
        circle_shape_button=Rect(1744, 340, 36, 36),
        hue_direction="bottom_to_top",
        saturation_direction="left_high",
        value_direction="top_dark",
        display=display,
        created_at="2026-08-17T00:00:00+00:00",
        updated_at="2026-08-17T00:01:00+00:00",
        metadata={"notes": "borderless, UI scale 1.0"},
    )


def test_profile_serialization_round_trip_preserves_complete_calibration() -> None:
    original = _complete_profile()

    encoded = json.dumps(original.to_dict())
    restored = Profile.from_dict(json.loads(encoded))

    assert restored == original
    assert restored.canvas is not None
    assert restored.canvas.right == -452
    assert restored.canvas.aspect_ratio == pytest.approx(1048 / 526)
    assert restored.is_ready
    assert restored.calibration_status == {
        "canvas": True,
        "color_box": True,
        "hue_bar": True,
        "brush_slider": True,
        "brush_preview": True,
        "square_shape_button": True,
        "circle_shape_button": True,
    }
    assert restored.display is not None
    assert restored.display.virtual_screen == Rect(-1920, 0, 4480, 1440)


def test_profile_deserializer_accepts_legacy_names_and_rect_edges() -> None:
    value = {
        "name": "Legacy",
        "canvas": {"x": -10, "y": 20, "right": 90, "bottom": 70},
        "color_box": {"left": 100, "top": 200, "width": 300, "height": 250},
        "hue_bar": {"left": 410, "top": 200, "width": 25, "height": 250},
        "brush_slider": None,
        "hue_direction": "top_bottom",
        "saturation_direction": "left_low_right_high",
        "value_direction": "top_bright_bottom_dark",
    }

    profile = Profile.from_dict(value)

    assert profile.id
    assert profile.canvas == Rect(-10, 20, 100, 50)
    assert profile.color_box == Rect(100, 200, 300, 250)
    assert profile.hue_direction == "bottom_to_top"
    assert profile.saturation_direction == "left_low"
    assert profile.value_direction == "top_bright"
    assert profile.brush_slider is None
    assert profile.brush_preview is None
    # Documents written before the optional shape buttons existed stay valid.
    assert profile.square_shape_button is None
    assert profile.circle_shape_button is None


def test_profile_store_crud_and_default_survive_reload(tmp_path) -> None:
    path = tmp_path / "profiles"
    store = ProfileStore(path)

    first = store.create("Wooden Sign", canvas=Rect(10, 20, 800, 400))
    second = store.create("Banner", make_default=True)
    renamed = store.rename(first.id, "Picture Frame")
    renamed.color_box = Rect(1000, 400, 300, 300)
    store.save(renamed)

    reloaded = ProfileStore(path)
    assert [item.name for item in reloaded.list_profiles()] == ["Picture Frame", "Banner"]
    assert reloaded.get_default() == second
    assert reloaded.require("picture frame").color_box == Rect(1000, 400, 300, 300)

    assert reloaded.delete(second.id)
    assert reloaded.get_default() is not None
    assert reloaded.get_default().id == first.id
    assert not reloaded.delete("does-not-exist")

    document = json.loads((path / "profiles.json").read_text(encoding="utf-8"))
    assert document["schemaVersion"] == 1
    assert document["defaultProfileId"] == first.id


def test_profile_store_rejects_duplicate_names_and_corrupt_json(tmp_path) -> None:
    path = tmp_path / "profiles.json"
    store = ProfileStore(path)
    store.create("Custom 1")
    with pytest.raises(DuplicateProfileNameError):
        store.create(" custom 1 ")

    path.write_text("{not valid JSON", encoding="utf-8")
    with pytest.raises(ProfileDataError, match="Could not read profile store"):
        store.list_profiles()


def test_display_metadata_reports_layout_and_scaling_changes() -> None:
    original = _complete_profile().display
    assert original is not None
    changed = DisplayMetadata(
        monitors=(
            MonitorMetadata(
                name=r"\\.\DISPLAY1",
                rect=Rect(0, 0, 1920, 1080),
                device_pixel_ratio=1.0,
                primary=True,
            ),
        ),
        virtual_screen=Rect(0, 0, 1920, 1080),
    )

    differences = original.differences(changed)

    assert "virtual desktop bounds changed" in differences
    assert "display count changed" in differences
    assert "connected displays changed" in differences
    assert not original.is_compatible(changed)


def test_display_metadata_warns_when_coordinate_mode_changes() -> None:
    bounds = Rect(0, 0, 1920, 1080)
    logical = DisplayMetadata(virtual_screen=bounds, coordinate_space="logical")
    physical = DisplayMetadata(virtual_screen=bounds, coordinate_space="physical")

    assert "display coordinate mode changed" in logical.differences(physical)
