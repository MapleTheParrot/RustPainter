"""Human-facing checks for whether an import has enough detail to paint."""

from __future__ import annotations

from dataclasses import dataclass
from math import floor

from .image_processing import calculate_fit_size, fill_crop_box
from .models import ScaleMode


@dataclass(frozen=True, slots=True)
class ImageReadiness:
    """The source area that reaches the sign and its scale at the target."""

    source_size: tuple[int, int]
    used_source_size: tuple[int, int]
    painted_size: tuple[int, int]
    enlargement: float
    recommended_size: tuple[int, int]

    @property
    def needs_warning(self) -> bool:
        # A few percent of resampling is normal rounding.  Calling that out
        # would train people to ignore this notice.
        return self.enlargement > 1.05


def assess_image_readiness(
    source_size: tuple[int, int],
    target_size: tuple[int, int],
    scale_mode: ScaleMode,
) -> ImageReadiness:
    """Measure useful source pixels after Fit/Fill, without decoding pixels.

    This deliberately measures geometry rather than file size: a 4K screenshot
    with a narrow Fill crop may have far fewer useful pixels than its metadata
    suggests.  Upscaling is retained for a predictable result, but the caller
    can offer a one-click target that does not pretend to create detail.
    """

    source_width, source_height = source_size
    target_width, target_height = target_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Image and target sizes must be positive")

    if scale_mode is ScaleMode.FIT:
        painted_width, painted_height = calculate_fit_size(source_size, target_size)
        used_width, used_height = source_width, source_height
    elif scale_mode is ScaleMode.FILL:
        left, top, right, bottom = fill_crop_box(source_size, target_size, (0.5, 0.5))
        used_width, used_height = right - left, bottom - top
        painted_width, painted_height = target_size
    else:
        used_width, used_height = source_width, source_height
        painted_width, painted_height = target_size

    enlargement = max(painted_width / used_width, painted_height / used_height)
    # Both pairs have the same aspect ratio for Fit and Fill.  Scaling the
    # target down by the enlargement is therefore the largest no-upscale grid.
    recommended = (
        max(8, min(target_width, floor(target_width / max(1.0, enlargement)))),
        max(8, min(target_height, floor(target_height / max(1.0, enlargement)))),
    )
    return ImageReadiness(
        source_size=source_size,
        used_source_size=(round(used_width), round(used_height)),
        painted_size=(painted_width, painted_height),
        enlargement=enlargement,
        recommended_size=recommended,
    )
