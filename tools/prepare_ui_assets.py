"""Bake the raw UI art into the small, app-ready PNGs shipped in assets/ui.

Run this after replacing anything in the source art folder:

    python tools/prepare_ui_assets.py "path/to/RustPainter Assets"

The raw art is 1254px-square glow renders and multi-megabyte textures, while
the desktop UI only ever draws them at icon sizes or as stretched surfaces.
The bake step trims them down, keys the flat black backdrop out of the opaque
icon renders, and recolours the shared grain textures into the four surface
tiles and the three rust fills that the stylesheet references. The wear overlay
is composited into the header plate here rather than shipped on its own, since
Qt cannot stack two backgrounds on one widget.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_ROOT = REPO_ROOT / "assets" / "ui"

# Source name -> shipped name. Icons keep a flat vocabulary so the stylesheet
# and the icon helper can address them without knowing the source numbering.
ICONS = {
    "I2_workspace-icon.png": "workspace",
    "I3_settings-icon.png": "settings",
    "I4_choose-image-icon.png": "choose-image",
    "I5_preview-placeholder.png": "preview-placeholder",
    "I6_drag-drop-upload.png": "drag-drop",
    "I8_crop-alignment.png": "crop",
    "I9_sliders-quality.png": "sliders",
    "I10_edit-pencil.png": "pencil",
    "I11_delete-trash-icon.png": "trash",
    "I12_calibration-target.png": "target",
    "I13_ready-check.png": "check",
    "I14_play-start.png": "play",
    "I15_pause.png": "pause",
    "I16_abort-stop.png": "abort",
    "I19_resolution-canvas.png": "resolution",
    "I20_colors-palette.png": "palette",
    "I21_strokes-brush.png": "brush",
    "I22_estimated-time-clock.png": "clock",
    "I23_idle-status.png": "status",
}

ICON_SIZE = 192

# Edge length of the seamless surface tiles. Mirroring makes them symmetric, so
# a large tile is what keeps the repeat from reading as a lattice behind the
# window-sized panels.
TILE_SIZE = 512

# Ramp endpoints used to recolour a grayscale grain into a themed surface. The
# ramp is applied to a contrast-normalised copy of the source (see _ramp), so
# these two colours are the darkest and lightest pixels the surface can show.
BACKGROUND_RAMP = ((11, 10, 9), (36, 31, 26))
PANEL_RAMP = ((15, 13, 11), (52, 43, 36))
PANEL_RAISED_RAMP = ((26, 21, 18), (68, 55, 45))
INSET_RAMP = ((8, 7, 6), (28, 24, 20))
HEADER_RAMP = ((12, 10, 9), (62, 42, 28))
# The button fills were already reading as rust before the surfaces were
# normalised, so their ramps are pulled in to keep the same weight on screen.
ACCENT_RAMP = ((70, 24, 3), (176, 78, 21))
DANGER_RAMP = ((56, 9, 6), (138, 33, 24))


def _key_black_to_alpha(image: Image.Image) -> Image.Image:
    """Turn the flat black backdrop of an opaque icon render into alpha.

    The renders sit on pure black but their subjects include dark grey metal,
    so the ramp only fades the bottom of the range: anything above a low
    luminance stays fully opaque and keeps its body.
    """

    rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
    luminance = rgb.max(axis=2)
    alpha = np.clip((luminance - 6.0) / 26.0, 0.0, 1.0)
    return Image.fromarray(
        np.dstack([rgb, alpha * 255.0]).astype(np.uint8), "RGBA"
    )


def _trim_and_pad(image: Image.Image, size: int) -> Image.Image:
    """Centre the icon subject in a square canvas so every icon reads alike."""

    bbox = image.getchannel("A").point(lambda value: 255 if value > 8 else 0).getbbox()
    if bbox:
        image = image.crop(bbox)
    inner = int(size * 0.92)
    image.thumbnail((inner, inner), Image.LANCZOS)
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(image, ((size - image.width) // 2, (size - image.height) // 2), image)
    return canvas


def _ramp(
    image: Image.Image,
    dark: tuple[int, int, int],
    light: tuple[int, int, int],
) -> Image.Image:
    """Map a texture's luminance onto a two-point colour ramp.

    The source renders are dark: their luminance sits inside a narrow band near
    the bottom of the range, so mapping it straight onto a ramp used only a
    sliver of it and produced surfaces that were flat to the eye. Normalising
    the band to the full ramp first is what makes the grain visible, and the
    percentiles keep a few stray specks from stealing the whole range.
    """

    gray = np.asarray(image.convert("L"), dtype=np.float32)
    low_value, high_value = np.percentile(gray, (2.0, 98.0))
    normalized = np.clip(
        (gray - low_value) / max(float(high_value - low_value), 1.0), 0.0, 1.0
    )
    low = np.asarray(dark, dtype=np.float32)
    high = np.asarray(light, dtype=np.float32)
    rgb = low + normalized[..., None] * (high - low)
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def _mirror_tile(image: Image.Image, size: int) -> Image.Image:
    """Build a seamless tile by mirroring a quadrant, so repeats show no grid."""

    half = size // 2
    quadrant = image.resize((half, half), Image.LANCZOS)
    tile = Image.new(image.mode, (size, size))
    tile.paste(quadrant, (0, 0))
    tile.paste(quadrant.transpose(Image.FLIP_LEFT_RIGHT), (half, 0))
    tile.paste(quadrant.transpose(Image.FLIP_TOP_BOTTOM), (0, half))
    tile.paste(quadrant.transpose(Image.ROTATE_180), (half, half))
    return tile


def _mirror_strip(image: Image.Image, size: tuple[int, int]) -> Image.Image:
    """Mirror a texture horizontally into a wide strip that tiles seamlessly.

    Used for the progress fill, which only ever repeats sideways: a strip wider
    than the bar keeps the grain from reading as a row of beads.
    """

    width, height = size
    half = image.resize((width // 2, height), Image.LANCZOS)
    strip = Image.new(image.mode, size)
    strip.paste(half, (0, 0))
    strip.paste(half.transpose(Image.FLIP_LEFT_RIGHT), (width // 2, 0))
    return strip


def _fade_alpha(image: Image.Image, strength: float) -> Image.Image:
    """Scale an overlay's alpha so it wears a surface instead of hiding it."""

    faded = image.copy()
    faded.putalpha(image.getchannel("A").point(lambda value: int(value * strength)))
    return faded


def _rounded_fill(
    texture: Image.Image,
    ramp: tuple[tuple[int, int, int], tuple[int, int, int]],
    *,
    size: tuple[int, int] = (256, 64),
    radius: int = 10,
    edge: tuple[int, int, int] = (255, 158, 74),
) -> Image.Image:
    """Bake a rounded, grainy button fill for a border-image slice.

    Qt ignores border-radius under a border-image, so the corner rounding and
    the lit top edge are painted into the texture itself. Slicing this at 12px
    keeps the corners crisp at any button size.
    """

    body = _ramp(texture.resize(size, Image.LANCZOS), *ramp)
    # A soft vertical light gradient reads as a bevelled, worn metal face.
    shade = np.linspace(1.22, 0.78, size[1], dtype=np.float32)[:, None, None]
    lit = np.clip(np.asarray(body, dtype=np.float32) * shade, 0, 255).astype(np.uint8)
    fill = Image.fromarray(lit, "RGB").convert("RGBA")

    mask = Image.new("L", size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1), radius=radius, fill=255
    )
    fill.putalpha(mask)

    outline = Image.new("RGBA", size, (0, 0, 0, 0))
    ImageDraw.Draw(outline).rounded_rectangle(
        (0, 0, size[0] - 1, size[1] - 1), radius=radius, outline=(*edge, 150), width=2
    )
    return Image.alpha_composite(fill, outline)


def main(source_root: Path) -> int:
    assets = source_root / "UIAssets"
    if not assets.is_dir():
        print(f"No UIAssets folder under {source_root}", file=sys.stderr)
        return 1
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    for source_name, output_name in ICONS.items():
        image = Image.open(assets / source_name)
        image = (
            image.convert("RGBA")
            if image.mode == "RGBA"
            else _key_black_to_alpha(image)
        )
        _trim_and_pad(image, ICON_SIZE).save(
            OUTPUT_ROOT / f"{output_name}.png", optimize=True
        )

    base = Image.open(assets / "T1_base-app-background.png").convert("RGB")
    grunge = Image.open(assets / "T4_subtle-grunge-overlay.png").convert("RGBA")
    base = Image.alpha_composite(
        base.convert("RGBA"), grunge.resize(base.size, Image.LANCZOS)
    ).convert("RGB")
    _mirror_tile(_ramp(base, *BACKGROUND_RAMP), TILE_SIZE).save(
        OUTPUT_ROOT / "surface-base.png", optimize=True
    )

    panel = Image.open(assets / "T2_panel-card-surface.png").convert("RGB")
    # The source is a rounded plate on a backdrop; only its flat centre tiles
    # cleanly, so crop that before mirroring or the plate edge becomes a grid.
    width, height = panel.size
    panel = panel.crop(
        (width // 4, height // 4, width * 3 // 4, height * 3 // 4)
    )
    _mirror_tile(_ramp(panel, *PANEL_RAMP), TILE_SIZE).save(
        OUTPUT_ROOT / "surface-panel.png", optimize=True
    )

    # Buttons, cards, and popups sit above the panels; inputs and previews sit
    # below them. Both reuse the panel plate at different ramps so every surface
    # in the window carries the same grain.
    _mirror_tile(_ramp(panel, *PANEL_RAISED_RAMP), TILE_SIZE).save(
        OUTPUT_ROOT / "surface-raised.png", optimize=True
    )
    _mirror_tile(_ramp(panel, *INSET_RAMP), TILE_SIZE).save(
        OUTPUT_ROOT / "surface-inset.png", optimize=True
    )

    header = Image.open(assets / "T3_header-topbar-texture.png").convert("RGB")
    wear = Image.open(assets / "T5_rust-edge-wear-overlay.png").convert("RGBA")
    header_size = (1024, 341)
    # The header is the one surface drawn as a single stretched image rather than
    # a tile, so the edge wear can be baked into it without repeating.
    header_plate = _ramp(header.resize(header_size, Image.LANCZOS), *HEADER_RAMP)
    header_plate = Image.alpha_composite(
        header_plate.convert("RGBA"),
        _fade_alpha(
            wear.resize(header_size, Image.LANCZOS).filter(
                ImageFilter.GaussianBlur(0.4)
            ),
            0.45,
        ),
    ).convert("RGB")
    header_plate.save(OUTPUT_ROOT / "surface-header.png", optimize=True)

    grain = Image.open(assets / "T8_red-danger-texture.png").convert("RGB")
    _rounded_fill(grain, ACCENT_RAMP).save(
        OUTPUT_ROOT / "fill-accent.png", optimize=True
    )
    _rounded_fill(grain, DANGER_RAMP, edge=(255, 108, 90)).save(
        OUTPUT_ROOT / "fill-danger.png", optimize=True
    )
    # The progress chunk keeps its rounded corners from the stylesheet, so its
    # fill is a plain tile rather than a sliced button plate. It is taller than
    # any progress bar, so only the horizontal repeat has to be seamless.
    _mirror_strip(_ramp(grain, *ACCENT_RAMP), (512, 64)).save(
        OUTPUT_ROOT / "fill-progress.png", optimize=True
    )

    print(f"Wrote {len(list(OUTPUT_ROOT.glob('*.png')))} files to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    default_root = Path.home() / "Downloads" / "RustPainter Assets"
    raise SystemExit(main(Path(sys.argv[1]) if len(sys.argv) > 1 else default_root))
