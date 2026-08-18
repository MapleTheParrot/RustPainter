"""Image fitting, cropping, transparency handling, and color quantization."""

from __future__ import annotations

from math import ceil, floor
from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image, ImageOps

from .models import (
    CropAlignment,
    ImageProcessOptions,
    ProcessedImage,
    RGBColor,
    ScaleMode,
    TransparencyMode,
)


ImageSource: TypeAlias = str | Path | Image.Image

try:
    _LANCZOS = Image.Resampling.LANCZOS
    _DITHER_NONE = Image.Dither.NONE
    _DITHER_FS = Image.Dither.FLOYDSTEINBERG
    _MEDIANCUT = Image.Quantize.MEDIANCUT
except AttributeError:  # Pillow < 9.1
    _LANCZOS = Image.LANCZOS
    _DITHER_NONE = Image.NONE
    _DITHER_FS = Image.FLOYDSTEINBERG
    _MEDIANCUT = Image.MEDIANCUT


def _validate_size(size: tuple[int, int], label: str) -> None:
    if len(size) != 2 or size[0] <= 0 or size[1] <= 0:
        raise ValueError(f"{label} dimensions must be positive")


def _validate_color(color: RGBColor | None, label: str) -> None:
    if color is None:
        return
    if len(color) != 3 or any(not 0 <= int(channel) <= 255 for channel in color):
        raise ValueError(f"{label} must contain three channels in the range 0..255")


def _enum_key(value: object) -> str:
    raw = value.value if hasattr(value, "value") else value
    return str(raw).strip().lower().replace("-", "_").replace(" ", "_")


def _scale_mode(value: ScaleMode | str) -> ScaleMode:
    aliases = {
        "fit": ScaleMode.FIT,
        "fill": ScaleMode.FILL,
        "fill_crop": ScaleMode.FILL,
        "crop": ScaleMode.FILL,
        "stretch": ScaleMode.STRETCH,
    }
    try:
        return aliases[_enum_key(value).replace("/", "_")]
    except KeyError as exc:
        raise ValueError(f"Unknown scale mode: {value!r}") from exc


def _alignment(value: CropAlignment | str) -> CropAlignment:
    try:
        return CropAlignment(_enum_key(value))
    except ValueError as exc:
        raise ValueError(f"Unknown crop alignment: {value!r}") from exc


def _transparency_mode(value: TransparencyMode | str) -> TransparencyMode:
    aliases = {
        "leave_unpainted": TransparencyMode.LEAVE_UNPAINTED,
        "unpainted": TransparencyMode.LEAVE_UNPAINTED,
        "leave": TransparencyMode.LEAVE_UNPAINTED,
        "use_background": TransparencyMode.USE_BACKGROUND,
        "background": TransparencyMode.USE_BACKGROUND,
    }
    try:
        return aliases[_enum_key(value)]
    except KeyError as exc:
        raise ValueError(f"Unknown transparency mode: {value!r}") from exc


def load_image(source: ImageSource) -> Image.Image:
    """Load a source as a detached, EXIF-oriented RGBA image."""

    if isinstance(source, Image.Image):
        image = source.copy()
    else:
        with Image.open(source) as opened:
            image = opened.copy()
    return ImageOps.exif_transpose(image).convert("RGBA")


def calculate_fit_size(
    source_size: tuple[int, int], target_size: tuple[int, int]
) -> tuple[int, int]:
    """Largest aspect-preserving integer size wholly inside the target."""

    _validate_size(source_size, "Source")
    _validate_size(target_size, "Target")
    source_width, source_height = source_size
    target_width, target_height = target_size
    scale = min(target_width / source_width, target_height / source_height)
    return (
        min(target_width, max(1, floor(source_width * scale + 1e-9))),
        min(target_height, max(1, floor(source_height * scale + 1e-9))),
    )


def calculate_fill_size(
    source_size: tuple[int, int], target_size: tuple[int, int]
) -> tuple[int, int]:
    """Smallest aspect-preserving integer size that covers the target."""

    _validate_size(source_size, "Source")
    _validate_size(target_size, "Target")
    source_width, source_height = source_size
    target_width, target_height = target_size
    scale = max(target_width / source_width, target_height / source_height)
    return (
        max(target_width, ceil(source_width * scale - 1e-9)),
        max(target_height, ceil(source_height * scale - 1e-9)),
    )


def calculate_scaled_size(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    mode: ScaleMode | str,
) -> tuple[int, int]:
    resolved = _scale_mode(mode)
    if resolved is ScaleMode.FIT:
        return calculate_fit_size(source_size, target_size)
    if resolved is ScaleMode.FILL:
        return calculate_fill_size(source_size, target_size)
    _validate_size(source_size, "Source")
    _validate_size(target_size, "Target")
    return target_size


def _scale_layer(
    source: Image.Image,
    target_size: tuple[int, int],
    mode: ScaleMode,
    alignment: CropAlignment,
) -> tuple[Image.Image, np.ndarray]:
    """Return an RGBA layer plus a mask for the source's rectangular footprint."""

    target_width, target_height = target_size
    if mode is ScaleMode.STRETCH:
        return (
            source.resize(target_size, _LANCZOS),
            np.ones((target_height, target_width), dtype=np.bool_),
        )

    if mode is ScaleMode.FILL:
        centering = {
            CropAlignment.CENTER: (0.5, 0.5),
            CropAlignment.TOP: (0.5, 0.0),
            CropAlignment.BOTTOM: (0.5, 1.0),
            CropAlignment.LEFT: (0.0, 0.5),
            CropAlignment.RIGHT: (1.0, 0.5),
        }[alignment]
        # ImageOps.fit supplies a source crop box directly to resize.  Unlike
        # resizing a very wide/tall image before cropping, this cannot create a
        # huge intermediate allocation for an extreme source aspect ratio.
        return (
            ImageOps.fit(source, target_size, method=_LANCZOS, centering=centering),
            np.ones((target_height, target_width), dtype=np.bool_),
        )

    resized_size = calculate_fit_size(source.size, target_size)
    resized = source.resize(resized_size, _LANCZOS)
    layer = Image.new("RGBA", target_size, (0, 0, 0, 0))
    paste_x = (target_width - resized_size[0]) // 2
    paste_y = (target_height - resized_size[1]) // 2
    # No mask argument: retaining the resized alpha is important for deciding
    # later whether a source pixel should be left unpainted.
    layer.paste(resized, (paste_x, paste_y))
    footprint = np.zeros((target_height, target_width), dtype=np.bool_)
    footprint[
        paste_y : paste_y + resized_size[1],
        paste_x : paste_x + resized_size[0],
    ] = True
    return layer, footprint


def scale_image(
    source: ImageSource,
    target_size: tuple[int, int],
    mode: ScaleMode | str = ScaleMode.FIT,
    *,
    alignment: CropAlignment | str = CropAlignment.CENTER,
    background_color: RGBColor | None = None,
    transparency_mode: TransparencyMode | str = TransparencyMode.LEAVE_UNPAINTED,
    transparent_fill_color: RGBColor | None = None,
    alpha_threshold: int = 0,
) -> tuple[Image.Image, np.ndarray]:
    """Scale to a logical canvas and return opaque simulation plus paint mask.

    ``background_color=None`` makes Fit letterbox pixels unpainted.  Fully
    transparent source pixels are independently controlled by
    ``transparency_mode``.  If they should use a background but neither color is
    supplied, white is used as a safe visible default.
    """

    _validate_size(target_size, "Target")
    _validate_color(background_color, "Background color")
    _validate_color(transparent_fill_color, "Transparent fill color")
    if not 0 <= alpha_threshold <= 255:
        raise ValueError("Alpha threshold must be between 0 and 255")

    resolved_mode = _scale_mode(mode)
    resolved_alignment = _alignment(alignment)
    resolved_transparency = _transparency_mode(transparency_mode)
    rgba_source = load_image(source)
    layer, footprint = _scale_layer(
        rgba_source, target_size, resolved_mode, resolved_alignment
    )

    layer_array = np.asarray(layer, dtype=np.uint8)
    source_rgb = layer_array[:, :, :3]
    source_alpha = layer_array[:, :, 3]
    height, width = source_alpha.shape
    background = background_color
    alpha_base = transparent_fill_color or background or (255, 255, 255)

    result_rgb = np.empty((height, width, 3), dtype=np.uint8)
    result_rgb[:] = background or (0, 0, 0)
    paint_mask = (~footprint) & (background is not None)

    visible_source = footprint & (source_alpha > alpha_threshold)
    hidden_source = footprint & ~visible_source

    # Painting has no alpha control.  Resolve partial source alpha now against
    # the selected background (or white when the underlying sign is unknown).
    if np.any(visible_source):
        opacity = source_alpha[visible_source].astype(np.float32)[:, None] / 255.0
        foreground = source_rgb[visible_source].astype(np.float32)
        base = np.asarray(alpha_base, dtype=np.float32)[None, :]
        blended = np.rint(foreground * opacity + base * (1.0 - opacity))
        result_rgb[visible_source] = blended.astype(np.uint8)
        paint_mask[visible_source] = True

    if resolved_transparency is TransparencyMode.USE_BACKGROUND:
        result_rgb[hidden_source] = np.asarray(alpha_base, dtype=np.uint8)
        paint_mask[hidden_source] = True
    else:
        # RGB here is only for predictable previews; alpha/mask remain zero.
        result_rgb[hidden_source] = np.asarray(background or (0, 0, 0), dtype=np.uint8)
        paint_mask[hidden_source] = False

    result_alpha = np.where(paint_mask, 255, 0).astype(np.uint8)
    result = Image.fromarray(
        np.dstack((result_rgb, result_alpha)), mode="RGBA"
    )
    return result, paint_mask


def quantize_image(
    image: Image.Image,
    color_count: int,
    *,
    dither: bool = False,
    paint_mask: np.ndarray | None = None,
) -> Image.Image:
    """Quantize painted RGB pixels while keeping unpainted pixels transparent."""

    if not 1 <= color_count <= 256:
        raise ValueError("Color count must be between 1 and 256")
    rgba = image.convert("RGBA")
    array = np.asarray(rgba, dtype=np.uint8)
    rgb = array[:, :, :3]
    if paint_mask is None:
        mask = array[:, :, 3] > 0
    else:
        mask = np.asarray(paint_mask, dtype=np.bool_)
        if mask.shape != (rgba.height, rgba.width):
            raise ValueError("Paint mask dimensions must match the image")

    painted_colors = rgb[mask]
    if painted_colors.size == 0:
        empty = np.zeros_like(array)
        return Image.fromarray(empty, mode="RGBA")

    unique_count = len(np.unique(painted_colors, axis=0))
    if unique_count <= color_count:
        mapped_rgb = rgb.copy()
    else:
        # Build the palette from painted pixels only.  Otherwise transparent Fit
        # padding can consume a requested palette entry (usually as black).
        samples = Image.fromarray(painted_colors.reshape(1, -1, 3), mode="RGB")
        palette = samples.quantize(
            colors=color_count, method=_MEDIANCUT, dither=_DITHER_NONE
        )
        mapped_rgb = np.asarray(
            Image.fromarray(rgb, mode="RGB")
            .quantize(
                palette=palette,
                dither=_DITHER_FS if dither else _DITHER_NONE,
            )
            .convert("RGB"),
            dtype=np.uint8,
        ).copy()

    output = np.dstack((mapped_rgb, np.where(mask, 255, 0).astype(np.uint8)))
    return Image.fromarray(output, mode="RGBA")


def process_image(
    source: ImageSource,
    options: ImageProcessOptions | None = None,
    **option_overrides: object,
) -> ProcessedImage:
    """Create the exact logical image and mask consumed by paint planning.

    Keyword options are accepted as a convenience when a GUI does not already
    have an :class:`ImageProcessOptions` instance.
    """

    if options is None:
        options = ImageProcessOptions(**option_overrides)  # type: ignore[arg-type]
    elif option_overrides:
        raise TypeError("Pass an options object or keyword options, not both")

    scaled, paint_mask = scale_image(
        source,
        (options.logical_width, options.logical_height),
        options.scale_mode,
        alignment=options.crop_alignment,
        background_color=options.background_color,
        transparency_mode=options.transparency_mode,
        transparent_fill_color=options.transparent_fill_color,
        alpha_threshold=options.alpha_threshold,
    )
    quantized = quantize_image(
        scaled,
        options.color_count,
        dither=options.dither,
        paint_mask=paint_mask,
    )
    return ProcessedImage(quantized, paint_mask.copy(), options.color_count)


# Useful naming aliases for callers and tests.
fit_size = calculate_fit_size
fill_size = calculate_fill_size
resize_image = scale_image


__all__ = [
    "ImageSource",
    "calculate_fill_size",
    "calculate_fit_size",
    "calculate_scaled_size",
    "fill_size",
    "fit_size",
    "load_image",
    "process_image",
    "quantize_image",
    "resize_image",
    "scale_image",
]
