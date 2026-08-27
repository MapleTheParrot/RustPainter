"""Image fitting, cropping, transparency handling, and color quantization."""

from __future__ import annotations

from math import ceil, floor, sqrt
from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image, ImageOps

from .models import (
    BackgroundRemovalScope,
    CropAlignment,
    ImageProcessOptions,
    ProcessedImage,
    RGBColor,
    ScaleMode,
    SharpenMode,
    TransparencyMode,
)


ImageSource: TypeAlias = str | Path | Image.Image

# The longest possible distance between two RGB colors, which turns a
# background tolerance percentage into a concrete color distance.
_MAX_RGB_DISTANCE = sqrt(3.0) * 255.0

# Alpha above this counts as opaque when alpha fill is off, so a pixel that is
# more there than not gets painted and the rest of the soft edge is dropped.
OPAQUE_ALPHA_CUTOFF = 127

# How the smart scope differs from matching one flat color, in three numbers.
#
# A real backdrop is rarely one color: a studio sweep is a gradient, a photo
# has a vignette, and a JPEG has ringing along every edge.  Averaging all of
# that into a single key matches the middle of the range and neither end, so
# several keys are read off instead and every pixel is measured against the
# nearest of them.
_SUBJECT_KEY_COLORS = 4

# The keys are voted for from a band rather than the one-pixel ring, because a
# single ring of a noisy backdrop is a small and unrepresentative sample.
_SUBJECT_BORDER_DEPTH = 3

# Seeds have to match a key strictly; what a seed then spreads into only has
# to be plausible.  That is what carries a fill from one end of a gradient to
# the other without the tolerance having to be wide enough to reach the
# subject from the start - but it is also how a fill leaks, so the growth band
# stays narrow enough that an ordinary subject sits outside it.  There is a
# cliff rather than a slope here: widen it far enough to reach the subject and
# the fill does not take a little more of the picture, it takes all of it.
_SUBJECT_GROWTH = 1.5

# The last pixel or two before the background is a blend of the two, so it
# matches neither and gets painted as a halo around the subject.  Those pixels
# are allowed a much looser match, precisely because being within two pixels
# of removed background is itself most of the evidence.
_FRINGE_GROWTH = 3.0
_FRINGE_PASSES = 2

# Unsharp-mask strength per sharpen mode: how much of the difference between
# a cell and its blurred surroundings is added back.  The game's bilinear
# magnification of the sign is about a one-texel blur, so the mask uses a
# one-cell radius, and Light at 0.6 restores roughly the contrast that blur
# costs a line without ringing.  Strong goes past that for line art that
# should bite, at the price of a faint light halo beside dark lines.
SHARPEN_AMOUNTS: dict[SharpenMode, float] = {
    SharpenMode.OFF: 0.0,
    SharpenMode.LIGHT: 0.6,
    SharpenMode.STRONG: 1.2,
}
_SHARPEN_RADIUS = 1.0

# Where each Fill alignment anchors the kept region, as ``ImageOps.fit``
# centering fractions.  Shared with the GUI so its canvas overlay and the
# resampler can never disagree about which part of the source survives.
CROP_CENTERING: dict[CropAlignment, tuple[float, float]] = {
    CropAlignment.CENTER: (0.5, 0.5),
    CropAlignment.TOP: (0.5, 0.0),
    CropAlignment.BOTTOM: (0.5, 1.0),
    CropAlignment.LEFT: (0.0, 0.5),
    CropAlignment.RIGHT: (1.0, 0.5),
}


def crop_centering(
    alignment: CropAlignment | str = CropAlignment.CENTER,
    focus: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """The centering fractions Fill should keep, named or dragged.

    A crop dragged on the source image lands between the five named
    alignments, so it carries its own pair and simply outranks the name.  Both
    the resampler and the GUI's canvas overlay ask this one function, which is
    what stops the dashed rectangle from drifting off the region that is
    actually kept.
    """

    if focus is not None:
        x, y = focus
        return (min(max(float(x), 0.0), 1.0), min(max(float(y), 0.0), 1.0))
    return CROP_CENTERING[_alignment(alignment)]


try:
    _LANCZOS = Image.Resampling.LANCZOS
    _NEAREST = Image.Resampling.NEAREST
    _DITHER_NONE = Image.Dither.NONE
    _DITHER_FS = Image.Dither.FLOYDSTEINBERG
    _MAXCOVERAGE = Image.Quantize.MAXCOVERAGE
except AttributeError:  # Pillow < 9.1
    _LANCZOS = Image.LANCZOS
    _NEAREST = Image.NEAREST
    _DITHER_NONE = Image.NONE
    _DITHER_FS = Image.FLOYDSTEINBERG
    _MAXCOVERAGE = Image.MAXCOVERAGE


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


def _sharpen_mode(value: SharpenMode | str) -> SharpenMode:
    try:
        return SharpenMode(_enum_key(value))
    except ValueError as exc:
        raise ValueError(f"Unknown sharpen mode: {value!r}") from exc


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


# sRGB transfer function.  Resampling has to happen in linear light: averaging
# gamma-encoded values weighs perceptual codes instead of photons, which lands a
# heavy downscale of a detailed photo about twenty RGB levels dark.
def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    return np.where(
        values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4
    ).astype(np.float32)


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    clamped = np.clip(values, 0.0, 1.0)
    return np.where(
        clamped <= 0.0031308,
        clamped * 12.92,
        1.055 * clamped ** (1.0 / 2.4) - 0.055,
    ).astype(np.float32)


# Decoding always starts from an 8-bit channel, so the transfer function is a
# 256-entry lookup instead of a fractional power over every pixel of a photo.
_SRGB_TO_LINEAR = _srgb_to_linear(np.arange(256, dtype=np.float32) / 255.0)


def fill_crop_box(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    centering: tuple[float, float],
) -> tuple[float, float, float, float]:
    """The source rectangle Fill keeps, matching ``ImageOps.fit`` without bleed.

    Handing this box straight to the resampler is what stops an extreme source
    aspect ratio from allocating a huge intermediate image, which is why Fill
    does not simply crop and then resize.  The GUI also reads it to know where
    the sign canvas sits on the unscaled source image.
    """

    source_width, source_height = source_size
    target_ratio = target_size[0] / target_size[1]
    if source_width / source_height >= target_ratio:
        crop_width, crop_height = target_ratio * source_height, float(source_height)
    else:
        crop_width, crop_height = float(source_width), source_width / target_ratio
    left = (source_width - crop_width) * centering[0]
    top = (source_height - crop_height) * centering[1]
    return (left, top, left + crop_width, top + crop_height)


def _resample(
    source: Image.Image,
    size: tuple[int, int],
    box: tuple[float, float, float, float] | None = None,
) -> Image.Image:
    """Resize RGBA in linear light so brightness survives a heavy downscale.

    Resizing the gamma-encoded image directly is what makes a shrunken photo
    come out muddy: every output pixel of a 12x reduction averages roughly two
    hundred source pixels, and averaging sRGB codes systematically underweights
    the bright ones.  Decoding first and re-encoding after keeps the logical
    image as bright as the original, so the sign inherits the right exposure.
    Alpha is already linear and is resampled untouched.
    """

    array = np.asarray(source.convert("RGBA"), dtype=np.uint8)
    # A pure upscale averages nothing, so linear light buys nothing and
    # Lanczos actively hurts: blowing a 10x8 pixel image up to a 320x240 sign
    # must yield crisp blocks, not a blur of colors the source never held -
    # interpolation invents nothing an upscale could reveal.
    source_width = (box[2] - box[0]) if box is not None else source.width
    source_height = (box[3] - box[1]) if box is not None else source.height
    scaling_up = size[0] >= source_width and size[1] >= source_height
    filter_choice = _NEAREST if scaling_up else _LANCZOS
    resized = []
    # One channel at a time: a phone photo is already fifty megabytes as bytes,
    # and holding all four as float at once would quadruple that for nothing.
    for channel in range(4):
        if channel < 3:
            plane = _SRGB_TO_LINEAR[array[:, :, channel]]
        else:
            plane = np.ascontiguousarray(array[:, :, channel], dtype=np.float32) / 255.0
        resized.append(
            np.asarray(
                Image.fromarray(plane, mode="F").resize(size, filter_choice, box=box),
                dtype=np.float32,
            )
        )
    # Lanczos undershoots at hard edges; clipping before re-encoding keeps the
    # negative lobe out of the fractional power.
    rgb = _linear_to_srgb(np.dstack(resized[:3])) * 255.0
    alpha = np.clip(resized[3], 0.0, 1.0) * 255.0
    return Image.fromarray(
        np.rint(np.dstack((rgb, alpha))).astype(np.uint8), mode="RGBA"
    )


def _scale_layer(
    source: Image.Image,
    target_size: tuple[int, int],
    mode: ScaleMode,
    centering: tuple[float, float],
) -> tuple[Image.Image, np.ndarray]:
    """Return an RGBA layer plus a mask for the source's rectangular footprint."""

    target_width, target_height = target_size
    if mode is ScaleMode.STRETCH:
        return (
            _resample(source, target_size),
            np.ones((target_height, target_width), dtype=np.bool_),
        )

    if mode is ScaleMode.FILL:
        return (
            _resample(
                source,
                target_size,
                box=fill_crop_box(source.size, target_size, centering),
            ),
            np.ones((target_height, target_width), dtype=np.bool_),
        )

    resized_size = calculate_fit_size(source.size, target_size)
    resized = _resample(source, resized_size)
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
    focus: tuple[float, float] | None = None,
    background_color: RGBColor | None = None,
    transparency_mode: TransparencyMode | str = TransparencyMode.LEAVE_UNPAINTED,
    transparent_fill_color: RGBColor | None = None,
    alpha_threshold: int = 0,
    alpha_fill: bool = False,
) -> tuple[Image.Image, np.ndarray]:
    """Scale to a logical canvas and return opaque simulation plus paint mask.

    ``background_color=None`` makes Fit letterbox pixels unpainted.  Fully
    transparent source pixels are independently controlled by
    ``transparency_mode``.  If they should use a background but neither color is
    supplied, white is used as a safe visible default.  ``focus`` overrides
    ``alignment`` with the exact centering a dragged crop was left at.

    ``alpha_fill`` decides what happens to the pixels in between - the soft
    edge of a cut-out subject, an anti-aliased logo.  On, they are mixed into
    the background color the way a compositor would; off, only the mostly
    opaque ones are painted, in their own colors.
    """

    _validate_size(target_size, "Target")
    _validate_color(background_color, "Background color")
    _validate_color(transparent_fill_color, "Transparent fill color")
    if not 0 <= alpha_threshold <= 255:
        raise ValueError("Alpha threshold must be between 0 and 255")

    resolved_mode = _scale_mode(mode)
    resolved_centering = crop_centering(alignment, focus)
    resolved_transparency = _transparency_mode(transparency_mode)
    rgba_source = load_image(source)
    layer, footprint = _scale_layer(
        rgba_source, target_size, resolved_mode, resolved_centering
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

    # Painting has no alpha control, so every partly transparent pixel has to
    # become one opaque color or nothing at all.  Mixing it into the
    # background is right when the sign really will carry that background, and
    # wrong otherwise: the soft edge of a cut-out subject then paints a ring of
    # half-background the artwork never had, which is the halo people see
    # around something they asked to be left unpainted.  So alpha fill is a
    # choice, and with it off a pixel is painted only once it is mostly opaque
    # and is painted in its own color.
    cutoff = alpha_threshold if alpha_fill else max(alpha_threshold, OPAQUE_ALPHA_CUTOFF)
    visible_source = footprint & (source_alpha > cutoff)
    hidden_source = footprint & ~visible_source

    if np.any(visible_source):
        if alpha_fill:
            opacity = source_alpha[visible_source].astype(np.float32)[:, None] / 255.0
            foreground = source_rgb[visible_source].astype(np.float32)
            base = np.asarray(alpha_base, dtype=np.float32)[None, :]
            blended = np.rint(foreground * opacity + base * (1.0 - opacity))
            result_rgb[visible_source] = blended.astype(np.uint8)
        else:
            result_rgb[visible_source] = source_rgb[visible_source]
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


def _removal_scope(value: BackgroundRemovalScope | str) -> BackgroundRemovalScope:
    aliases = {
        "subject": BackgroundRemovalScope.SUBJECT,
        "smart": BackgroundRemovalScope.SUBJECT,
        "connected": BackgroundRemovalScope.CONNECTED,
        "edges": BackgroundRemovalScope.CONNECTED,
        "touching": BackgroundRemovalScope.CONNECTED,
        "everywhere": BackgroundRemovalScope.EVERYWHERE,
        "anywhere": BackgroundRemovalScope.EVERYWHERE,
        "all": BackgroundRemovalScope.EVERYWHERE,
    }
    try:
        return aliases[_enum_key(value)]
    except KeyError as exc:
        raise ValueError(f"Unknown background removal scope: {value!r}") from exc


def _resolved_mask(image: Image.Image, paint_mask: np.ndarray | None) -> np.ndarray:
    if paint_mask is None:
        return np.asarray(image.convert("RGBA"), dtype=np.uint8)[:, :, 3] > 0
    mask = np.asarray(paint_mask, dtype=np.bool_)
    if mask.shape != (image.height, image.width):
        raise ValueError("Paint mask dimensions must match the image")
    return mask


def _painted_border(paint_mask: np.ndarray, depth: int = 1) -> np.ndarray:
    """A ring ``depth`` pixels deep around the painted area's bounding box.

    Fit leaves unpainted bars around the artwork, so the canvas edge is not
    always where the background starts.  Working from the painted bounding box
    means one ring serves letterboxed, cropped, and stretched layouts alike.

    A deeper band is a larger and steadier sample of a noisy backdrop, so it
    is what the smart scope votes on; it is held to a quarter of the artwork
    on each side so that it can never swallow what it is meant to ring.
    """

    border = np.zeros(paint_mask.shape, dtype=np.bool_)
    rows = np.flatnonzero(np.any(paint_mask, axis=1))
    columns = np.flatnonzero(np.any(paint_mask, axis=0))
    if rows.size == 0 or columns.size == 0:
        return border
    top, bottom = int(rows[0]), int(rows[-1])
    left, right = int(columns[0]), int(columns[-1])
    depth = max(
        1,
        min(int(depth), (bottom - top) // 4 + 1, (right - left) // 4 + 1),
    )
    border[top : top + depth, left : right + 1] = True
    border[bottom - depth + 1 : bottom + 1, left : right + 1] = True
    border[top : bottom + 1, left : left + depth] = True
    border[top : bottom + 1, right - depth + 1 : right + 1] = True
    return border & paint_mask


def detect_background_colors(
    image: Image.Image,
    paint_mask: np.ndarray | None = None,
    *,
    limit: int = 1,
    depth: int = 1,
    share: float = 0.05,
) -> list[RGBColor]:
    """Guess the background from the colors ringing the painted artwork.

    Edge pixels are bucketed coarsely before voting so a noisy or JPEG-blurred
    backdrop still lands in one bucket; each winning bucket then reports the
    average of its real colors rather than a quantized stand-in.

    ``limit`` is how many colors may be returned, most popular first.  One is
    right for a flat backdrop, but a gradient or a vignette is genuinely
    several colors and averaging them produces one that matches the middle of
    the range and neither end.  A runner-up has to hold ``share`` of the
    sampled band to count, so a subject clipping the edge does not become a
    background color in its own right.
    """

    mask = _resolved_mask(image, paint_mask)
    border = _painted_border(mask, depth)
    samples = np.asarray(image.convert("RGB"), dtype=np.uint8)[border]
    if samples.size == 0:
        return []
    buckets = samples >> 4
    packed = (
        (buckets[:, 0].astype(np.int32) << 8)
        | (buckets[:, 1].astype(np.int32) << 4)
        | buckets[:, 2].astype(np.int32)
    )
    values, counts = np.unique(packed, return_counts=True)
    ranked = np.argsort(counts)[::-1][: max(1, int(limit))]
    floor = max(1, int(float(share) * samples.shape[0]))
    colors: list[RGBColor] = []
    for index in ranked:
        if colors and int(counts[index]) < floor:
            break
        average = samples[packed == values[index]].mean(axis=0)
        red, green, blue = (int(round(float(channel))) for channel in average)
        colors.append((red, green, blue))
    return colors


def detect_background_color(
    image: Image.Image, paint_mask: np.ndarray | None = None
) -> RGBColor | None:
    """The single most popular color ringing the painted artwork."""

    colors = detect_background_colors(image, paint_mask)
    return colors[0] if colors else None


def _fill_runs(similar: np.ndarray, seeded: np.ndarray) -> np.ndarray:
    """Grow seeds across every horizontal run of ``similar`` that they touch.

    Filling whole runs at once is what keeps the flood fill affordable: a plain
    one-pixel dilation needs a pass for every pixel of travel, while alternating
    row and column runs crosses an open background in a handful of passes.
    """

    height, width = similar.shape
    flat = np.ascontiguousarray(similar).reshape(-1)
    previous = np.empty_like(flat)
    previous[0] = False
    previous[1:] = flat[:-1]
    row_start = np.zeros(flat.size, dtype=np.bool_)
    row_start[::width] = True
    starts = flat & (row_start | ~previous)
    run_count = int(starts.sum())
    if run_count == 0:
        return seeded
    run_index = np.cumsum(starts) - 1
    touched = np.zeros(run_count, dtype=np.bool_)
    seeds_flat = np.ascontiguousarray(seeded).reshape(-1) & flat
    touched[run_index[seeds_flat]] = True
    filled = flat & touched[np.clip(run_index, 0, run_count - 1)]
    return filled.reshape(height, width) | seeded


def _connected_region(similar: np.ndarray, seeds: np.ndarray) -> np.ndarray:
    """Every ``similar`` pixel reachable from a seed along rows and columns."""

    filled = similar & seeds
    if not filled.any():
        return filled
    transposed = np.ascontiguousarray(similar.T)
    previous = -1
    # Row/column passes settle an open background almost immediately; the cap
    # only stops a pathological maze-shaped region from spinning forever.
    for _ in range(256):
        count = int(filled.sum())
        if count == previous:
            break
        previous = count
        filled = _fill_runs(similar, filled)
        filled = _fill_runs(transposed, np.ascontiguousarray(filled.T)).T
    return np.ascontiguousarray(filled)


def _grown(mask: np.ndarray) -> np.ndarray:
    """The mask plus every pixel sharing an edge with it."""

    grown = mask.copy()
    grown[1:, :] |= mask[:-1, :]
    grown[:-1, :] |= mask[1:, :]
    grown[:, 1:] |= mask[:, :-1]
    grown[:, :-1] |= mask[:, 1:]
    return grown


def _neighbour_min(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """The smallest value among each pixel's four neighbours, ``valid`` only.

    A pixel with no valid neighbour at all reports infinity, which is the
    honest answer and happens to be the one that lets an isolated speck go.
    """

    filled = np.where(valid, values, np.inf).astype(np.float32)
    best = np.full(values.shape, np.inf, dtype=np.float32)
    best[1:, :] = np.minimum(best[1:, :], filled[:-1, :])
    best[:-1, :] = np.minimum(best[:-1, :], filled[1:, :])
    best[:, 1:] = np.minimum(best[:, 1:], filled[:, :-1])
    best[:, :-1] = np.minimum(best[:, :-1], filled[:, 1:])
    return best


def _key_distance(rgb: np.ndarray, keys: list[RGBColor]) -> np.ndarray:
    """How far every pixel is from the nearest of the background colors."""

    nearest: np.ndarray | None = None
    for key in keys:
        difference = (rgb - np.asarray(key, dtype=np.int16)).astype(np.float32)
        distance = np.sqrt(np.einsum("ijk,ijk->ij", difference, difference))
        nearest = distance if nearest is None else np.minimum(nearest, distance)
    assert nearest is not None  # keys is never empty here
    return nearest


def _subject_background(
    mask: np.ndarray, distance: np.ndarray, limit: float
) -> np.ndarray:
    """Separate backdrop from subject by growing a strict match into a loose one.

    One tolerance has to be two different things at once: tight enough not to
    reach the subject, and wide enough to cover a backdrop that is a gradient,
    a vignette, or a field of JPEG ringing.  It cannot be both, which is why a
    flat match either leaves a mottled backdrop half painted or eats into the
    artwork.

    So the tolerance is only used to decide where the background certainly is.
    From those seeds the region spreads through anything merely plausible, and
    only through pixels it can actually reach from outside the artwork - which
    is what keeps an enclosed pocket of the same color, the hole in an O or a
    white eye, painted.  A last two pixels of much looser growth follow the
    boundary itself, where the halo lives: a pixel that is a blend of subject
    and backdrop matches neither, so it is judged against what it is attached
    to instead of against the tolerance.  A blend sits partway along a ramp
    into the background, so it is closer to the background than the artwork
    behind it is; the outer pixels of a genuinely pale subject are no closer
    than the rest of that subject, and stay.
    """

    strong = mask & (distance <= limit)
    weak = mask & (distance <= min(limit * _SUBJECT_GROWTH, _MAX_RGB_DISTANCE))
    seeds = strong & _painted_border(mask, _SUBJECT_BORDER_DEPTH)
    removed = _connected_region(weak, seeds)
    if not removed.any():
        return removed
    fringe = (
        mask
        & ~removed
        & (distance <= min(limit * _FRINGE_GROWTH, _MAX_RGB_DISTANCE))
    )
    for _ in range(_FRINGE_PASSES):
        inward = _neighbour_min(distance, mask & ~removed)
        touching = fringe & ~removed & _grown(removed) & (distance < inward)
        if not touching.any():
            break
        removed = removed | touching
    return removed


def background_mask(
    image: Image.Image,
    paint_mask: np.ndarray | None = None,
    *,
    color: RGBColor | None = None,
    tolerance: float = 12.0,
    scope: BackgroundRemovalScope | str = BackgroundRemovalScope.SUBJECT,
) -> np.ndarray:
    """Return the painted pixels that count as background for ``color``.

    ``tolerance`` is a percentage of the longest possible RGB distance, so 0
    matches a single exact color and 100 matches everything.  ``color=None``
    reads the key color off the artwork edges - several of them under the
    smart scope, which measures each pixel against whichever is nearest.
    """

    _validate_color(color, "Background removal color")
    if not 0.0 <= float(tolerance) <= 100.0:
        raise ValueError("Background removal tolerance must be between 0 and 100")
    resolved_scope = _removal_scope(scope)
    mask = _resolved_mask(image, paint_mask)
    empty = np.zeros(mask.shape, dtype=np.bool_)
    if not mask.any():
        return empty
    smart = resolved_scope is BackgroundRemovalScope.SUBJECT
    if color is not None:
        keys = [color]
    elif smart:
        keys = detect_background_colors(
            image,
            mask,
            limit=_SUBJECT_KEY_COLORS,
            depth=_SUBJECT_BORDER_DEPTH,
        )
    else:
        detected = detect_background_color(image, mask)
        keys = [] if detected is None else [detected]
    if not keys:
        return empty

    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    distance = _key_distance(rgb, keys)
    limit = float(tolerance) / 100.0 * _MAX_RGB_DISTANCE
    if smart:
        return _subject_background(mask, distance, limit)
    similar = mask & (distance <= limit)
    if resolved_scope is BackgroundRemovalScope.EVERYWHERE:
        return similar
    return _connected_region(similar, _painted_border(mask))


def remove_background(
    image: Image.Image,
    paint_mask: np.ndarray | None = None,
    *,
    color: RGBColor | None = None,
    tolerance: float = 12.0,
    scope: BackgroundRemovalScope | str = BackgroundRemovalScope.SUBJECT,
) -> tuple[Image.Image, np.ndarray]:
    """Drop background pixels from the paint mask so Rust never paints them."""

    mask = _resolved_mask(image, paint_mask)
    removed = background_mask(
        image, mask, color=color, tolerance=tolerance, scope=scope
    )
    if not removed.any():
        return image, mask
    remaining = mask & ~removed
    array = np.asarray(image.convert("RGBA"), dtype=np.uint8).copy()
    array[:, :, 3] = np.where(remaining, array[:, :, 3], 0)
    return Image.fromarray(array, mode="RGBA"), remaining


# Hue is meaningless for a near-neutral color: five levels of channel spread on
# a near-white pixel swing it by nearly 180 degrees.  Median cut happily spends
# several palette slots on such pixels, and each one then gets a fully saturated
# hue click that only a click one or two pixels from the edge of the saturation
# box pulls back, so any calibration slack there paints visible pastel speckle.
# Four percent is about ten levels of spread on white - invisible on screen, and
# well under any deliberate pastel.
_NEUTRAL_SATURATION_LIMIT = 0.04


def _snap_near_neutrals(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Flatten painted colors too faint to carry a hue onto the gray axis."""

    painted = rgb[mask]
    if painted.size == 0:
        return rgb
    high = painted.max(axis=1).astype(np.int16)
    low = painted.min(axis=1).astype(np.int16)
    # Saturation the way the picker defines it, rearranged so a black pixel does
    # not divide by zero.
    faint = (high - low) <= np.rint(high * _NEUTRAL_SATURATION_LIMIT)
    if not faint.any():
        return rgb
    # The picker's value axis is the largest channel, so the gray it would hand
    # back at zero saturation is that channel repeated.
    painted = painted.copy()
    painted[faint] = high[faint, None].astype(np.uint8)
    snapped = rgb.copy()
    snapped[mask] = painted
    return snapped


def _median_cut_colors(painted_colors: np.ndarray, color_count: int) -> np.ndarray:
    """Reduce ``painted_colors`` (N, 3) to at most ``color_count`` colors.

    Classic median cut for the budgets Pillow cannot take: the bucket with
    the widest channel spread splits at that channel's median until the
    budget is reached or every bucket is a single color, and each pixel
    takes its bucket's median.  Returns the mapped colors in input order.
    """

    import heapq

    values = painted_colors.reshape(-1, 3).astype(np.int16)

    def bucket(indices: np.ndarray, order: int) -> tuple[int, int, int, np.ndarray]:
        sub = values[indices]
        extent = sub.max(axis=0) - sub.min(axis=0)
        channel = int(extent.argmax())
        return (-int(extent[channel]), order, channel, indices)

    counter = 0
    heap = [bucket(np.arange(len(values)), counter)]
    while len(heap) < color_count:
        negative_extent, _, channel, indices = heapq.heappop(heap)
        if negative_extent >= 0:
            heapq.heappush(heap, (negative_extent, counter + 1, channel, indices))
            break
        channel_values = values[indices, channel]
        median = np.median(channel_values)
        left = indices[channel_values <= median]
        right = indices[channel_values > median]
        if len(right) == 0:
            left = indices[channel_values < median]
            right = indices[channel_values >= median]
        counter += 2
        heapq.heappush(heap, bucket(left, counter - 1))
        heapq.heappush(heap, bucket(right, counter))
    mapped = np.empty_like(values, dtype=np.uint8)
    for _, _, _, indices in heap:
        mapped[indices] = np.rint(np.median(values[indices], axis=0)).astype(np.uint8)
    return mapped


def quantize_image(
    image: Image.Image,
    color_count: int,
    *,
    dither: bool = False,
    paint_mask: np.ndarray | None = None,
) -> Image.Image:
    """Quantize painted RGB pixels while keeping unpainted pixels transparent.

    ``color_count`` 0 lifts the palette cap entirely: every distinct color
    survives to the plan (the near-neutral snap and the picker snap still
    collapse what Rust cannot tell apart).  Painting time then scales with
    the image's unique colors - each is its own picker trip - which the
    time estimate prices honestly.
    """

    if color_count != 0 and not 1 <= color_count <= 4096:
        raise ValueError("Color count must be 0 (unlimited) or between 1 and 4096")
    rgba = image.convert("RGBA")
    array = np.asarray(rgba, dtype=np.uint8)
    if paint_mask is None:
        mask = array[:, :, 3] > 0
    else:
        mask = np.asarray(paint_mask, dtype=np.bool_)
        if mask.shape != (rgba.height, rgba.width):
            raise ValueError("Paint mask dimensions must match the image")
    # Snapping before the palette is built stops a handful of indistinguishable
    # near-whites from each consuming a palette slot the artwork could use.
    rgb = _snap_near_neutrals(array[:, :, :3], mask)

    painted_colors = rgb[mask]
    if painted_colors.size == 0:
        empty = np.zeros_like(array)
        return Image.fromarray(empty, mode="RGBA")

    unique_count = len(np.unique(painted_colors, axis=0))
    if color_count == 0 or unique_count <= color_count:
        mapped_rgb = rgb.copy()
    elif color_count > 256:
        # Pillow's quantizers stop at 256 palette entries; larger budgets
        # are cut here.  Error-diffusion dithering needs the palette-image
        # machinery, so it does not apply past 256 - at these budgets the
        # bands it would hide are already a few levels apart.
        mapped_rgb = rgb.copy()
        mapped_rgb[mask] = _median_cut_colors(painted_colors, color_count)
    else:
        # Build the palette from painted pixels only.  Otherwise transparent Fit
        # padding can consume a requested palette entry (usually as black).
        samples = Image.fromarray(painted_colors.reshape(1, -1, 3), mode="RGB")
        # Median cut spends nearly all of its entries where the most pixels
        # are.  That is disastrous for artwork with one dominant family: a
        # mostly orange image can lose every small magenta or cyan accent even
        # with a 256-color budget.  Maximum coverage chooses representatives
        # across the occupied RGB gamut, protecting visually distinct minority
        # hues while still respecting the same hard palette limit.
        palette = samples.quantize(
            colors=color_count, method=_MAXCOVERAGE, dither=_DITHER_NONE
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

    # Median cut can reintroduce a faint tint by averaging a mixed bucket, so the
    # palette it produced gets the same treatment.
    output = np.dstack(
        (
            _snap_near_neutrals(mapped_rgb, mask),
            np.where(mask, 255, 0).astype(np.uint8),
        )
    )
    return Image.fromarray(output, mode="RGBA")


def is_reduction(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    mode: ScaleMode | str,
    centering: tuple[float, float] = (0.5, 0.5),
) -> bool:
    """Whether scaling the source onto the canvas throws detail away.

    Mirrors the resampler's own choice between Lanczos and nearest: a source
    that fits inside the canvas along both axes is enlarged pixel for pixel
    and has nothing to sharpen, while one reduced along either axis has lost
    contrast to the averaging.  Fill judges the cropped region rather than
    the whole source, because that is all that gets resampled.
    """

    resolved = _scale_mode(mode)
    if resolved is ScaleMode.FILL:
        box = fill_crop_box(source_size, target_size, centering)
        source_width, source_height = box[2] - box[0], box[3] - box[1]
        size = target_size
    elif resolved is ScaleMode.FIT:
        source_width, source_height = source_size
        size = calculate_fit_size(source_size, target_size)
    else:
        source_width, source_height = source_size
        size = target_size
    return not (size[0] >= source_width and size[1] >= source_height)


def _gaussian_blur(plane: np.ndarray, sigma: float) -> np.ndarray:
    """Separable Gaussian blur of a 2-D float plane with reflected edges."""

    reach = max(1, int(ceil(3.0 * sigma)))
    taps = np.exp(-0.5 * (np.arange(-reach, reach + 1, dtype=np.float32) / sigma) ** 2)
    taps /= taps.sum()
    padded = np.pad(plane, reach, mode="reflect")
    rows = np.zeros((plane.shape[0] + 2 * reach, plane.shape[1]), dtype=np.float32)
    for offset, tap in enumerate(taps):
        rows += tap * padded[:, offset : offset + plane.shape[1]]
    out = np.zeros(plane.shape, dtype=np.float32)
    for offset, tap in enumerate(taps):
        out += tap * rows[offset : offset + plane.shape[0], :]
    return out


def sharpen_image(
    image: Image.Image,
    paint_mask: np.ndarray | None,
    amount: float,
    radius: float = _SHARPEN_RADIUS,
) -> Image.Image:
    """Unsharp-mask the painted cells of a logical image.

    Only painted cells are changed, and only painted cells contribute to the
    blur each is compared against: the surroundings are averaged with the
    mask as a weight, so a subject cut out of its background is sharpened
    against itself rather than against the color that used to be behind it,
    which would otherwise ring every edge of it with a halo of that color.
    """

    if amount <= 0:
        return image
    rgba = image.convert("RGBA")
    array = np.asarray(rgba, dtype=np.uint8)
    if paint_mask is None:
        mask = array[:, :, 3] > 0
    else:
        mask = np.asarray(paint_mask, dtype=np.bool_)
        if mask.shape != (rgba.height, rgba.width):
            raise ValueError("Paint mask dimensions must match the image")
    if not mask.any():
        return rgba
    weight = mask.astype(np.float32)
    blurred_weight = _gaussian_blur(weight, radius)
    rgb = array[:, :, :3].astype(np.float32)
    sharpened = rgb.copy()
    with np.errstate(divide="ignore", invalid="ignore"):
        for channel in range(3):
            local = _gaussian_blur(rgb[:, :, channel] * weight, radius) / blurred_weight
            local = np.where(blurred_weight > 0, local, rgb[:, :, channel])
            sharpened[:, :, channel] = rgb[:, :, channel] + amount * (
                rgb[:, :, channel] - local
            )
    output = array.copy()
    output[mask, :3] = np.clip(np.rint(sharpened[mask]), 0, 255).astype(np.uint8)
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

    target_size = (options.logical_width, options.logical_height)
    # Loaded once here so the reduction check below does not decode a large
    # photo a second time; ``scale_image`` takes the image as it is.
    loaded = load_image(source)
    scaled, paint_mask = scale_image(
        loaded,
        target_size,
        options.scale_mode,
        alignment=options.crop_alignment,
        focus=options.crop_focus,
        background_color=options.background_color,
        transparency_mode=options.transparency_mode,
        transparent_fill_color=options.transparent_fill_color,
        alpha_threshold=options.alpha_threshold,
        alpha_fill=options.alpha_fill,
    )
    if options.remove_background:
        scaled, paint_mask = remove_background(
            scaled,
            paint_mask,
            color=options.background_removal_color,
            tolerance=options.background_removal_tolerance,
            scope=options.background_removal_scope,
        )
    # After background removal, so a cut-out subject is sharpened against
    # itself and not against the backdrop that was just taken away.
    sharpen_amount = SHARPEN_AMOUNTS[_sharpen_mode(options.sharpen)]
    if sharpen_amount > 0 and is_reduction(
        loaded.size,
        target_size,
        options.scale_mode,
        crop_centering(options.crop_alignment, options.crop_focus),
    ):
        scaled = sharpen_image(scaled, paint_mask, sharpen_amount)
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
    "CROP_CENTERING",
    "ImageSource",
    "OPAQUE_ALPHA_CUTOFF",
    "background_mask",
    "calculate_fill_size",
    "calculate_fit_size",
    "calculate_scaled_size",
    "crop_centering",
    "detect_background_color",
    "detect_background_colors",
    "fill_crop_box",
    "fill_size",
    "fit_size",
    "load_image",
    "SHARPEN_AMOUNTS",
    "is_reduction",
    "process_image",
    "quantize_image",
    "remove_background",
    "resize_image",
    "scale_image",
    "sharpen_image",
]
