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
        brush_size_box=Rect(1760, 392, 62, 26),
        clear_button=Rect(1690, 392, 30, 30),
        save_button=Rect(1840, 900, 120, 36),
        hunger=Rect(2260, 1290, 55, 28),
        thirst=Rect(2260, 1320, 55, 28),
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
        "brush_size_box": True,
        "clear_button": True,
        "save_button": True,
        "hunger": True,
        "thirst": True,
        "download_button": False,
    }
    assert restored.display is not None
    assert restored.display.virtual_screen == Rect(-1920, 0, 4480, 1440)


def test_profile_deserializer_accepts_legacy_names_and_rect_edges() -> None:
    value = {
        "name": "Legacy",
        "canvas": {"x": -10, "y": 20, "right": 90, "bottom": 70},
        "color_box": {"left": 100, "top": 200, "width": 300, "height": 250},
        "hue_bar": {"left": 410, "top": 200, "width": 25, "height": 250},
        "brush_size_box": None,
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
    assert profile.brush_size_box is None


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
    assert reloaded.get_default() is not None
    assert reloaded.get_default().id == second.id
    assert reloaded.require("picture frame").color_box == Rect(1000, 400, 300, 300)

    assert reloaded.delete(second.id)
    assert reloaded.get_default() is not None
    assert reloaded.get_default().id == first.id
    assert not reloaded.delete("does-not-exist")

    document = json.loads((path / "profiles.json").read_text(encoding="utf-8"))
    assert document["schemaVersion"] == 1
    assert document["defaultProfileId"] == first.id


def test_fixed_ui_calibrations_are_shared_without_sharing_the_sign(tmp_path) -> None:
    store = ProfileStore(tmp_path / "profiles")
    first = store.create(
        "Wooden Sign",
        canvas=Rect(10, 20, 800, 400),
        color_box=Rect(1000, 400, 300, 300),
        hue_bar=Rect(1310, 400, 25, 300),
        brush_size_box=Rect(900, 250, 60, 24),
        clear_button=Rect(850, 250, 30, 30),
        hunger=Rect(1800, 900, 50, 24),
        thirst=Rect(1800, 930, 50, 24),
        download_button=Rect(1400, 250, 30, 30),
        save_button=Rect(1500, 850, 120, 36),
    )

    second = store.create("Banner", canvas=Rect(20, 30, 600, 200))

    assert second.canvas == Rect(20, 30, 600, 200)
    assert second.save_button is None
    for field in (
        "color_box",
        "hue_bar",
        "brush_size_box",
        "clear_button",
        "hunger",
        "thirst",
        "download_button",
    ):
        assert getattr(second, field) == getattr(first, field)

    second.hue_bar = Rect(1320, 410, 30, 310)
    store.save(second)
    assert store.require(first.id).hue_bar == second.hue_bar


def test_legacy_profiles_inherit_the_newest_available_shared_rectangles(tmp_path) -> None:
    path = tmp_path / "profiles.json"
    older = Profile(id="older", name="Older", color_box=Rect(1, 2, 3, 4))
    older.updated_at = "2026-01-01T00:00:00+00:00"
    newer = Profile(id="newer", name="Newer", hue_bar=Rect(5, 6, 7, 8))
    newer.updated_at = "2026-02-01T00:00:00+00:00"
    path.write_text(
        json.dumps({"profiles": [older.to_dict(), newer.to_dict()]}),
        encoding="utf-8",
    )

    profiles = ProfileStore(path).list_profiles()

    assert [profile.color_box for profile in profiles] == [Rect(1, 2, 3, 4)] * 2
    assert [profile.hue_bar for profile in profiles] == [Rect(5, 6, 7, 8)] * 2


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


def test_padded_screen_name_survives_a_save_and_reload_without_looking_changed() -> None:
    # Qt reports some EDID-derived names with trailing whitespace, e.g. the
    # laptop panel "ATNA40CU05-0 ".  Deserialization strips names, so an
    # unstripped captured name used to mismatch its own saved copy and make
    # every recalibration report "connected displays changed".
    captured = DisplayMetadata(
        monitors=(
            MonitorMetadata(name="ATNA40CU05-0 ", rect=Rect(0, 0, 2880, 1800), primary=True),
            MonitorMetadata(name="CQ32G4", rect=Rect(149, -1440, 2560, 1440)),
        ),
        virtual_screen=Rect(0, -1440, 2880, 3240),
    )

    reloaded = DisplayMetadata.from_dict(json.loads(json.dumps(captured.to_dict())))

    assert captured.monitors[0].name == "ATNA40CU05-0"
    assert reloaded.differences(captured) == []
    assert reloaded.is_compatible(captured)
