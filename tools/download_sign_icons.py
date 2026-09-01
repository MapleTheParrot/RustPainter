"""Download Rust's official item icons for the premade profile catalog.

Facepunch publishes the same ItemDefinition icons used by the game at a
stable URL keyed by item shortname. Keeping resized copies in ``assets/signs``
avoids opening Rust's roughly 29 GB of texture bundles at app startup.
"""

from __future__ import annotations

import io
import sys
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.sign_catalog import SIGN_CATALOG  # noqa: E402


SOURCES = (
    "https://files.facepunch.com/rust/item/{shortname}_512.png",
    # Some DLC entries are present in the current game bundles but their old
    # Facepunch CDN path returns 404. RustLabs mirrors the in-game item icon.
    "https://rustlabs.com/img/items180/{shortname}.png",
)


def main() -> int:
    destination = ROOT / "assets" / "signs"
    destination.mkdir(parents=True, exist_ok=True)
    failed: list[str] = []
    for shortname in sorted({entry.item_shortname for entry in SIGN_CATALOG}):
        path = destination / f"{shortname}.png"
        try:
            payload = None
            last_error: Exception | None = None
            for template in SOURCES:
                try:
                    request = urllib.request.Request(
                        template.format(shortname=shortname),
                        headers={"User-Agent": "RustPainter asset updater"},
                    )
                    with urllib.request.urlopen(
                        request, timeout=30
                    ) as response:
                        payload = response.read()
                    break
                except (OSError, urllib.error.URLError) as exc:
                    last_error = exc
            if payload is None:
                raise OSError(str(last_error or "no icon source answered"))
            source = Image.open(io.BytesIO(payload)).convert("RGBA")
            source.thumbnail((128, 128), Image.Resampling.LANCZOS)
            source.save(path, "PNG", optimize=True)
            print(f"saved {path.relative_to(ROOT)}")
        except (OSError, urllib.error.URLError) as exc:
            failed.append(shortname)
            print(f"could not fetch {shortname}: {exc}", file=sys.stderr)
    if failed:
        print(f"Missing {len(failed)} icon(s): {', '.join(failed)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
