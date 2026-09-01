from __future__ import annotations

import json
from pathlib import Path

from app.sign_catalog import SIGN_CATALOG, catalog_icon_path, search_catalog


ROOT = Path(__file__).resolve().parent.parent


def test_catalog_sizes_all_come_from_the_extracted_rust_prefabs() -> None:
    rows = json.loads((ROOT / "tools" / "sign_sizes.json").read_text(encoding="utf-8"))
    extracted_sizes = {(int(row["width"]), int(row["height"])) for row in rows}

    assert len({entry.id for entry in SIGN_CATALOG}) == len(SIGN_CATALOG)
    assert all(entry.texture_size in extracted_sizes for entry in SIGN_CATALOG)


def test_every_premade_profile_has_a_bundled_recognition_icon() -> None:
    missing = [entry.name for entry in SIGN_CATALOG if not catalog_icon_path(entry).is_file()]
    assert missing == []


def test_fuzzy_search_handles_typos_and_size_tokens() -> None:
    assert search_catalog("xxl light frame")[0].name == "Light-Up Frame XXL"
    assert search_catalog("artst canv xxl")[0].name == "Artist Canvas XXL"
    assert search_catalog("ornte standng")[0].name == "Ornate Frame Standing"
