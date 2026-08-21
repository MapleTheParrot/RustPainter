"""Read the sign texture sizes out of Rust's asset bundles, for development.

The app never needs this: at run time it measures the texel grid on the sign
in front of it.  This is the cross-check - the list of texture sizes Rust's
prefabs actually declare this month, to hold a measurement against and to
keep the fallback table of texture sizes honest after a wipe.

It reads files on disk and nothing else; it neither touches the running game
nor needs its managed assemblies (the client is IL2CPP and has none).  Rust's
bundles embed their type trees, so every MonoBehaviour's fields can be read
by name straight from the bundle, and UnityPy reads the bundles lazily
enough that even the multi-gigabyte ones open in seconds.

What is read: every ``MeshPaintableSource`` component - the thing a
``Signage`` entity's ``paintableSources`` point at - with its ``texWidth`` and
``texHeight``, under the name of the object it sits on.  The deployable
signs live in ``assetscenes.bundle`` (Facepunch bakes prefabs into asset
scenes), a handful in ``content.bundle``; all the non-texture bundles are
scanned so nothing is missed when they move again.

    pip install UnityPy
    python tools/dump_sign_sizes.py --smoke              # list sign items (fast)
    python tools/dump_sign_sizes.py                      # print the table
    python tools/dump_sign_sizes.py --json tools/sign_sizes.json

The committed ``tools/sign_sizes.json`` is the last run's output.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

DEFAULT_RUST = Path(r"C:\Program Files (x86)\Steam\steamapps\common\Rust")
BUNDLES = Path("Bundles") / "shared"
# Texture bundles hold no prefabs and are the bulk of the install.
SKIPPED_BUNDLES = re.compile(r"^(textures\.|audio\.)", re.IGNORECASE)

# The serialized fields of MeshPaintableSource that size the texture.
WIDTH_FIELD = "texWidth"
HEIGHT_FIELD = "texHeight"

# What a paintable item is called in items.preload.bundle.
PAINTABLE_NAME = re.compile(
    r"sign|frame|banner|canvas|photo|neon|spinner|easel|poster|window\.paintable",
    re.IGNORECASE,
)


def _load_unitypy():
    try:
        import UnityPy  # type: ignore
    except ImportError:
        sys.exit(
            "UnityPy is not installed.  It is a development-only dependency:\n"
            "    pip install UnityPy"
        )
    return UnityPy


def scan_bundle(bundle: Path) -> list[dict[str, Any]]:
    """Every paintable source in ``bundle``: object name, width, height.

    Objects are looked up by path id from the type trees rather than through
    UnityPy's parsed object layer, which has crashed on some of Rust's
    MonoBehaviours; the trees are plain dictionaries and never have.
    """

    UnityPy = _load_unitypy()
    env = UnityPy.load(str(bundle))
    by_path = {(obj.assets_file, obj.path_id): obj for obj in env.objects}

    def owner_name(obj: Any, tree: dict[str, Any]) -> str:
        reference = tree.get("m_GameObject") or {}
        owner = by_path.get((obj.assets_file, reference.get("m_PathID")))
        if owner is None:
            return "?"
        try:
            return str(owner.read_typetree().get("m_Name", "?"))
        except Exception:
            return "?"

    found: list[dict[str, Any]] = []
    for obj in env.objects:
        if obj.type.name != "MonoBehaviour":
            continue
        try:
            tree = obj.read_typetree()
        except Exception:
            continue
        if WIDTH_FIELD not in tree or HEIGHT_FIELD not in tree:
            continue
        width, height = tree[WIDTH_FIELD], tree[HEIGHT_FIELD]
        if not isinstance(width, int) or not isinstance(height, int):
            continue
        if width <= 0 or height <= 0:
            continue
        found.append(
            {
                "object": owner_name(obj, tree),
                "width": width,
                "height": height,
                "bundle": bundle.name,
            }
        )
    return found


def scan_all(rust: Path) -> list[dict[str, Any]]:
    """Distinct paintable sources across every non-texture bundle."""

    rows: dict[tuple[str, int, int], dict[str, Any]] = {}
    for bundle in sorted((rust / BUNDLES).glob("*.bundle")):
        if SKIPPED_BUNDLES.match(bundle.name):
            continue
        print(f"scanning {bundle.name} ...", file=sys.stderr)
        for row in scan_bundle(bundle):
            key = (row["object"], row["width"], row["height"])
            rows.setdefault(key, row)
    return sorted(rows.values(), key=lambda row: (row["object"].lower(), row["width"], row["height"]))


def smoke(bundle: Path) -> list[str]:
    """The paintable item names in the small preload bundle - proves the pipeline."""

    UnityPy = _load_unitypy()
    env = UnityPy.load(str(bundle))
    names = set()
    for obj in env.objects:
        if obj.type.name != "GameObject":
            continue
        try:
            name = str(obj.read_typetree().get("m_Name", ""))
        except Exception:
            continue
        if PAINTABLE_NAME.search(name):
            names.add(name)
    return sorted(names)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--rust", type=Path, default=DEFAULT_RUST, help="Rust install folder")
    parser.add_argument(
        "--bundle", type=Path, default=None, help="scan one bundle instead of all of them"
    )
    parser.add_argument("--json", type=Path, default=None, help="also write the table as JSON")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="only list paintable items from items.preload.bundle (fast)",
    )
    args = parser.parse_args(argv)

    if args.smoke:
        bundle = args.bundle or args.rust / BUNDLES / "items.preload.bundle"
        for name in smoke(bundle):
            print(name)
        return 0

    if args.bundle is not None:
        rows = scan_bundle(args.bundle)
    else:
        if not (args.rust / BUNDLES).is_dir():
            sys.exit(f"No bundles under {args.rust / BUNDLES}")
        rows = scan_all(args.rust)
    if not rows:
        print("No paintable sources found.", file=sys.stderr)
        return 1
    width = max(len(row["object"]) for row in rows)
    for row in rows:
        print(f"{row['object']:<{width}}  {row['width']:>5} x {row['height']:<5}  {row['bundle']}")
    sizes = sorted({(row["width"], row["height"]) for row in rows})
    print(f"\n{len(rows)} sources, {len(sizes)} distinct sizes:", file=sys.stderr)
    print("  " + ", ".join(f"{w}x{h}" for w, h in sizes), file=sys.stderr)
    if args.json:
        args.json.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
        print(f"Wrote {args.json}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
