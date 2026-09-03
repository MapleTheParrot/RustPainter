from __future__ import annotations

import json

import pytest

from app.settings import (
    DEFAULT_COLOR_COUNT,
    SETTINGS_SCHEMA_VERSION,
    SettingsError,
    SettingsStore,
    default_settings,
)


def _text_layer(**overrides):
    return {
        "text": "",
        "font_family": "",
        "font_size": 24,
        "size_ratio": 0.1875,
        "color": "#FFFFFF",
        "x": 0.5,
        "y": 0.5,
        "bold": False,
        "italic": False,
        **overrides,
    }


def test_settings_store_deep_merges_new_defaults_and_persists_updates(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps(
            {
                "image": {"color_count": 64},
                "painting": {"brush_size": 0.0},
                "future_section": {"keep_me": True},
            }
        ),
        encoding="utf-8",
    )
    store = SettingsStore(path)

    loaded = store.load()

    assert loaded["image"]["color_count"] == 64
    assert loaded["image"]["scale_mode"] == "fit"
    assert loaded["painting"]["brush_size"] == 0.0
    assert loaded["future_section"] == {"keep_me": True}

    updated = store.set("safety.countdown_seconds", 5)
    assert updated["safety"]["countdown_seconds"] == 5
    assert SettingsStore(path).get("safety.countdown_seconds") == 5


def test_sharpen_defaults_to_light_and_rejects_unknown_modes(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"image": {"color_count": 64}}), encoding="utf-8")
    assert SettingsStore(path).load()["image"]["sharpen"] == "light"

    path.write_text(json.dumps({"image": {"sharpen": "extreme"}}), encoding="utf-8")
    with pytest.raises(SettingsError):
        SettingsStore(path).load()


def test_default_settings_returns_independent_documents() -> None:
    first = default_settings()
    second = default_settings()
    first["image"]["color_count"] = 8
    assert second["image"]["color_count"] == DEFAULT_COLOR_COUNT
    assert second["execution"]["dry_run"] is False
    assert second["ui"]["tutorial_version_seen"] == 0
    assert second["painting"]["reuse_calibration"] is True
    assert second["painting"]["apply_brush_size"] is True
    assert second["ui"]["show_calibration_overlay"] is True
    assert second["ui"]["smooth_rust_preview"] is True
    assert second["safety"]["anti_afk_enabled"] is True
    assert second["safety"]["anti_afk_interval_minutes"] == 30
    assert second["timelapse"]["enabled"] is False
    assert second["painting"]["stroke_speed_pixels_per_second"] == 2200.0
    assert second["painting"]["mouse_down_duration_seconds"] == pytest.approx(0.07)
    assert second["painting"]["delay_after_hue_seconds"] == pytest.approx(0.07)
    assert second["painting"]["delay_after_saturation_value_seconds"] == pytest.approx(0.07)
    assert second["painting"]["delay_after_brush_seconds"] == pytest.approx(0.07)
    assert second["painting"]["delay_between_strokes_seconds"] == pytest.approx(0.02)
    assert second["painting"]["delay_between_colors_seconds"] == pytest.approx(0.07)
    assert second["painting"]["stroke_interpolation_step_pixels"] == pytest.approx(8.0)
    assert second["image"]["text_overlay"] == {
        "layers": [
            {
                "text": "",
                "font_family": "",
                "font_size": 24,
                "size_ratio": 0.09375,
                "color": "#FFFFFF",
                "x": 0.5,
                "y": 0.5,
                "bold": False,
                "italic": False,
                "smooth": True,
                "gradient": False,
                "gradient_direction": "vertical",
                "gradient_color": "#FF9336",
                "outline_width": 0,
                "outline_color": "#000000",
            }
        ]
    }


def test_settings_store_surfaces_corruption_and_invalid_values(tmp_path) -> None:
    path = tmp_path / "settings.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(SettingsError, match="JSON object"):
        SettingsStore(path).load()

    path.unlink()
    with pytest.raises(SettingsError, match="color_count"):
        SettingsStore(path).save({"image": {"color_count": 999}})

    with pytest.raises(SettingsError, match="execution"):
        SettingsStore(path).save({"execution": []})

    with pytest.raises(SettingsError, match="hotkeys"):
        SettingsStore(path).save(
            {"hotkeys": {"start_resume": "F8", "pause": "F9", "abort": "F8"}}
        )

    with pytest.raises(SettingsError, match="tutorial_version_seen"):
        SettingsStore(path).save({"ui": {"tutorial_version_seen": -1}})


def test_painting_delay_and_unknown_keys_survive_round_trip(tmp_path) -> None:
    path = tmp_path / "settings.json"
    document = default_settings()
    document["painting"]["delay_after_brush_seconds"] = 0.27
    document["painting"]["future_tuning_value"] = 123

    saved = SettingsStore(path).save(document)
    loaded = SettingsStore(path).load()

    assert saved["painting"]["delay_after_brush_seconds"] == pytest.approx(0.27)
    assert loaded["painting"]["future_tuning_value"] == 123


@pytest.mark.parametrize(
    ("patch", "message"),
    [
        ({"image": {"logical_width": 4096}}, "logical_width"),
        ({"image": {"scale_mode": []}}, "scale_mode"),
        ({"image": {"color_count": 7}}, "color_count"),
        ({"image": {"color_count": []}}, "color_count"),
        ({"image": {"dithering": "yes"}}, "dithering"),
        ({"image": {"background_mode": "purple"}}, "background_mode"),
        ({"image": {"background_color": "white"}}, "background_color"),
        ({"image": {"transparent_pixels": "discard"}}, "transparent_pixels"),
        ({"image": {"alpha_fill": "yes"}}, "alpha_fill"),
        ({"image": {"remove_background": "yes"}}, "remove_background"),
        ({"image": {"background_removal_source": "guess"}}, "background_removal_source"),
        ({"image": {"background_removal_color": "white"}}, "background_removal_color"),
        (
            {"image": {"background_removal_tolerance": 140}},
            "background_removal_tolerance",
        ),
        ({"image": {"background_removal_scope": "some"}}, "background_removal_scope"),
        (
            {"image": {"text_overlay": {"layers": [_text_layer(font_size=2)]}}},
            "font_size",
        ),
        (
            {"image": {"text_overlay": {"layers": [_text_layer(size_ratio=0.0)]}}},
            "size_ratio",
        ),
        (
            {"image": {"text_overlay": {"layers": [_text_layer(color="white")]}}},
            "color",
        ),
        (
            {"image": {"text_overlay": {"layers": [_text_layer(x=2.0)]}}},
            "layers\\[0\\].x",
        ),
        ({"painting": {"brush_size": 1.01}}, "brush_size"),
        (
            {"painting": {"stroke_speed_pixels_per_second": float("nan")}},
            "stroke_speed",
        ),
        ({"hotkeys": {"start_resume": "Ctrl+F8"}}, "start_resume"),
        ({"painting": {"confirm_strokes": "yes"}}, "confirm_strokes"),
        ({"painting": {"confirm_max_rounds": 0}}, "confirm_max_rounds"),
        ({"painting": {"confirm_max_rounds": True}}, "confirm_max_rounds"),
        ({"painting": {"reuse_calibration": "yes"}}, "reuse_calibration"),
    ],
)
def test_settings_reject_values_the_gui_cannot_represent(
    tmp_path, patch, message
) -> None:
    with pytest.raises(SettingsError, match=message):
        SettingsStore(tmp_path / "settings.json").save(patch)


def test_expected_process_defaults_to_the_rust_client() -> None:
    from app.settings import DEFAULT_SETTINGS

    safety = DEFAULT_SETTINGS["safety"]
    assert safety["expected_window_title_contains"] == "Rust"
    assert safety["expected_process_name"] == "RustClient.exe"
    assert safety["anti_afk_enabled"] is True


def test_session_ui_scale_defaults_are_safe_for_existing_profiles() -> None:
    from app.settings import DEFAULT_SETTINGS

    game = DEFAULT_SETTINGS["game"]
    assert game == {
        "manage_ui_scale": False,
        "painting_ui_scale": 0.5,
        "normal_ui_scale": 1.0,
        "console_key": "F1",
    }


@pytest.mark.parametrize(
    "patch",
    [
        {"game": {"manage_ui_scale": "yes"}},
        {"game": {"painting_ui_scale": 0.49}},
        {"game": {"normal_ui_scale": 1.01}},
        {"game": {"console_key": "WIN+F1"}},
    ],
)
def test_settings_reject_invalid_session_ui_scale_values(tmp_path, patch) -> None:
    with pytest.raises(SettingsError, match="game"):
        SettingsStore(tmp_path / "settings.json").save(patch)


def test_settings_accept_modifier_hotkeys_for_keyboards_without_a_usable_fn_key(
    tmp_path,
) -> None:
    # Compact laptop keyboards swallow F5-F12 behind Fn, so the app was
    # unusable there while only function keys were allowed.
    store = SettingsStore(tmp_path / "settings.json")

    saved = store.save(
        {
            "hotkeys": {
                "start_resume": "CTRL+ALT+S",
                "pause": "CTRL+ALT+P",
                "abort": "CTRL+ALT+X",
            }
        }
    )

    assert saved["hotkeys"]["abort"] == "CTRL+ALT+X"
    assert store.load()["hotkeys"]["start_resume"] == "CTRL+ALT+S"


def test_settings_still_reject_a_modifier_hotkey_the_chooser_cannot_offer(
    tmp_path,
) -> None:
    with pytest.raises(SettingsError, match="start_resume"):
        SettingsStore(tmp_path / "settings.json").save(
            {"hotkeys": {"start_resume": "CTRL+ALT+Q"}}
        )


def test_gui_computed_quality_presets_are_saveable(tmp_path) -> None:
    """"max" and "custom" come from the GUI, not the preset table.

    Their dimensions are computed (from the sign measurement and the spin
    boxes respectively), so they own no table entry - but rejecting them
    silently discarded every settings save made while one was selected.
    """

    store = SettingsStore(tmp_path / "settings.json")
    for preset in ("max", "custom", "balanced"):
        saved = store.save({"image": {"quality_preset": preset}})
        assert saved["image"]["quality_preset"] == preset

    with pytest.raises(SettingsError, match="quality_preset"):
        store.save({"image": {"quality_preset": "ultra"}})


def test_stroke_merging_switched_off_under_the_old_schema_is_lifted_to_balanced(
    tmp_path,
) -> None:
    """Merging never changes the picture, only how many strokes it takes, so
    an "off" saved before that was understood is turned on once."""

    path = tmp_path / "settings.json"
    path.write_text(
        json.dumps({"schema_version": 1, "painting": {"stroke_merge_mode": "off"}}),
        encoding="utf-8",
    )
    loaded = SettingsStore(path).load()
    assert loaded["painting"]["stroke_merge_mode"] == "balanced"
    assert loaded["schema_version"] == SETTINGS_SCHEMA_VERSION

    # A document that never said: the same lift (it predates the version key).
    path.write_text(json.dumps({"painting": {"stroke_merge_mode": "off"}}), encoding="utf-8")
    assert SettingsStore(path).load()["painting"]["stroke_merge_mode"] == "balanced"

    # Chosen under the current schema, "off" is the user's and stays.
    path.write_text(
        json.dumps(
            {"schema_version": SETTINGS_SCHEMA_VERSION, "painting": {"stroke_merge_mode": "off"}}
        ),
        encoding="utf-8",
    )
    assert SettingsStore(path).load()["painting"]["stroke_merge_mode"] == "off"


def test_checking_each_color_is_off_by_default_and_reading_picks_back_is_on() -> None:
    from app.painter import PainterSettings
    from app.settings import default_settings

    defaults = default_settings()
    assert defaults["painting"]["confirm_strokes"] is False
    assert defaults["painting"]["confirm_max_rounds"] == 4
    assert defaults["painting"]["verify_color_picks"] is True
    settings = PainterSettings.from_mapping(defaults)
    assert settings.confirm_strokes is False and settings.confirm_max_rounds == 4
    assert settings.verify_color_picks is True
    assert settings.auto_resume_on_focus_return is True
    assert settings.auto_resume_focus_retry_seconds == 10.0
    assert "verify_color_picks" in PainterSettings.RETUNABLE_FIELDS
    assert "auto_resume_on_focus_return" in PainterSettings.RETUNABLE_FIELDS
    # Both can be changed from a paused job.
    assert "confirm_strokes" in PainterSettings.RETUNABLE_FIELDS
    assert "confirm_max_rounds" in PainterSettings.RETUNABLE_FIELDS
    off = PainterSettings.from_mapping(
        {"painting": {"confirm_strokes": False, "confirm_max_rounds": 2}}
    )
    assert off.confirm_strokes is False and off.confirm_max_rounds == 2


