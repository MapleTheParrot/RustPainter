"""Read the sign texture sizes out of Rust's asset bundles, for development.

The app never needs this: at run time it measures the texel grid on the sign
in front of it.  This is the cross-check - the list of texture sizes Rust's
prefabs actually declare this month, to hold a measurement against and to
keep the canonical-size table honest after a wipe.

It reads files on disk and nothing else; it neither touches the running game
nor needs its managed assemblies (the client is IL2CPP and has none).  Rust's
bundles embed their type trees, so every MonoBehaviour's fields can be read
by name straight from the bundle.  The sign sizes live on the deployable
prefabs, which are in ``content.bundle`` - several gigabytes that UnityPy
decompresses whole, so run that with the game closed and about twice the
bundle's size free in memory.

    pip install UnityPy
    python tools/dump_sign_sizes.py --smoke          # list sign items (fast)
    python tools/dump_sign_sizes.py                  # content.bundle, prints sizes
    python tools/dump_sign_sizes.py --json sizes.json

Field names are not assumed: any MonoBehaviour whose tree holds keys that look
like a texture width and height is reported, under the prefab it sits on, so
a renamed component still shows up rather than silently vanishing.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Iterator

DEFAULT_RUST = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Rust")
BUNDLES = Path("Bundles") / "shared"

# What a prefab has to be called to be worth listing; every paintable in the
# game so far has one of these in its name.
PAINTABLE_NAME = re.compile(
    r"sign|frame|banner|canvas|photo|neon|spinner|easel|poster", re.IGNORECASE
)
WIDTH_KEY = re.compile(r"texture.?width|textureSize.*x$|^width$", re.IGNORECASE)
HEIGHT_KEY = re.compile(r"texture.?height|textureSize.*y$|^height$", re.IGNORECASE)


def _load_unitypy():
    try:
        import UnityPy  # type: ignore
    except ImportError:
        sys.exit(
            "UnityPy is not installed.  It is a development-only dependency:\n"
            "    pip install UnityPy"
        )
    return UnityPy


def _walk(tree: Any, path: str = "") -> Iterator[tuple[str, Any]]:
    """Every (dotted key, value) in a type tree, depth first."""

    if isinstance(tree, dict):
        for key, value in tree.items():
            yield from _walk(value, f"{path}.{key}" if path else str(key))
    elif isinstance(tree, list):
        for index, value in enumerate(tree):
            yield from _walk(value, f"{path}[{index}]")
    else:
        yield path, tree


def _texture_sizes(tree: dict[str, Any]) -> list[tuple[str, int, int]]:
    """``(where, width, height)`` for every width/height pair in a tree.

    Pairs are matched by the path they share up to the final key, so an
    array of paintable sources yields one entry per element.
    """

    widths: dict[str, int] = {}
    heights: dict[str, int] = {}
    for path, value in _walk(tree):
        if not isinstance(value, int) or isinstance(value, bool):
            continue
        parent, _, key = path.rpartition(".")
        if WIDTH_KEY.search(key):
            widths[parent] = value
        elif HEIGHT_KEY.search(key):
            heights[parent] = value
    return [
        (parent or "<root>", widths[parent], heights[parent])
        for parent in widths
        if parent in heights and widths[parent] > 0 and heights[parent] > 0
    ]


def scan(bundle: Path, *, name_filter: re.Pattern[str] = PAINTABLE_NAME) -> list[dict[str, Any]]:
    """Texture sizes declared by every paintable prefab in ``bundle``."""

    UnityPy = _load_unitypy()
    env = UnityPy.load(str(bundle))
    found: list[dict[str, Any]] = []
    for obj in env.objects:
        if obj.type.name != "MonoBehaviour":
            continue
        try:
            tree = obj.read_typetree()
        except Exception:
            continue
        game_object = tree.get("m_GameObject")
        prefab = ""
        try:
            if game_object is not None and getattr(game_object, "path_id", 0):
                prefab = game_object.read().m_Name
        except Exception:
            prefab = ""
        if prefab and not name_filter.search(prefab):
            continue
        sizes = _texture_sizes(tree)
        if not sizes:
            continue
        script = ""
        try:
            script = tree["m_Script"].read().m_ClassName
        except Exception:
            pass
        for where, width, height in sizes:
            found.append(
                {
                    "prefab": prefab or "?",
                    "component": script,
                    "field": where,
                    "width": width,
                    "height": height,
                }
            )
    found.sort(key=lambda row: (row["prefab"], row["field"]))
    return found


def smoke(bundle: Path) -> list[str]:
    """The sign item names in the small preload bundle - proves the pipeline."""

    UnityPy = _load_unitypy()
    env = UnityPy.load(str(bundle))
    names = sorted(
        {
            obj.read().m_Name
            for obj in env.objects
            if obj.type.name == "GameObject" and PAINTABLE_NAME.search(obj.read().m_Name)
        }
    )
    return names


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rust", type=Path, default=DEFAULT_RUST, help="Rust install folder")
    parser.add_argument(
        "--bundle",
        type=Path,
        default=None,
        help="bundle to scan (default: Bundles/shared/content.bundle)",
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the table as JSON")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="only list sign items from items.preload.bundle (fast, little memory)",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        bundle = args.bundle or args.rust / BUNDLES / "items.preload.bundle"
        for name in smoke(bundle):
            print(name)
        return 0

    bundle = args.bundle or args.rust / BUNDLES / "content.bundle"
    if not bundle.exists():
        sys.exit(f"No bundle at {bundle}")
    size_gb = bundle.stat().st_size / 1e9
    if size_gb > 1.0:
        print(
            f"{bundle.name} is {size_gb:.1f} GB and is decompressed whole; close the "
            "game and make sure roughly twice that is free in memory.",
            file=sys.stderr,
        )
    rows = scan(bundle)
    if not rows:
        print("No texture sizes found.", file=sys.stderr)
        return 1
    width = max(len(row["prefab"]) for row in rows)
    for row in rows:
        print(
            f"{row['prefab']:<{width}}  {row['width']:>5} x {row['height']:<5}  "
            f"{row['component']}  {row['field']}"
        )
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONUTF8", "1")
    raise SystemExit(main())
