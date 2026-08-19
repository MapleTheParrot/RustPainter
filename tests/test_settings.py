from __future__ import annotations

import json

import pytest

from app.settings import (
    DEFAULT_COLOR_COUNT,
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


def test_default_settings_returns_independent_documents() -> None:
    first = default_settings()
    second = default_settings()
    first["image"]["color_count"] = 8
    assert second["image"]["color_count"] == DEFAULT_COLOR_COUNT
    assert second["execution"]["dry_run"] is False
    assert second["image"]["text_overlay"] == {
        "layers": [
            {
                "text": "",
                "font_family": "",
                "font_size": 24,
                "size_ratio": 0.1875,
                "color": "#FFFFFF",
                "x": 0.5,
                "y": 0.5,
                "bold": False,
                "italic": False,
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
        ({"safety": {"corner_abort_margin_pixels": -1}}, "corner_abort_margin"),
        ({"hotkeys": {"start_resume": "Ctrl+F8"}}, "start_resume"),
    ],
)
def test_settings_reject_values_the_gui_cannot_represent(
    tmp_path, patch, message
) -> None:
    with pytest.raises(SettingsError, match=message):
        SettingsStore(tmp_path / "settings.json").save(patch)


def test_expected_process_default_is_platform_appropriate() -> None:
    """A Windows executable name can never match on another platform.

    Shipping "RustClient.exe" as the default on macOS made the foreground
    guard pause the instant the user focused the game, which is the whole
    point of the countdown. Regression test for that report.
    """

    import os

    from app.settings import DEFAULT_SETTINGS

    safety = DEFAULT_SETTINGS["safety"]
    assert safety["expected_window_title_contains"] == "Rust"
    if os.name == "nt":
        assert safety["expected_process_name"] == "RustClient.exe"
    else:
        assert safety["expected_process_name"] == ""


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
