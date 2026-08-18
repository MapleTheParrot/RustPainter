"""Small shared data models used by image processing and painting.

Screen rectangles use an exclusive right/bottom edge.  Logical stroke endpoints,
on the other hand, are inclusive pixel indices.  Keeping those two conventions
explicit avoids the most common off-by-one errors when a plan is executed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot
from typing import TYPE_CHECKING, Any, Iterator, NamedTuple, TypeAlias

if TYPE_CHECKING:
    import numpy as np
    from PIL import Image


RGBColor: TypeAlias = tuple[int, int, int]


class ScaleMode(str, Enum):
    FIT = "fit"
    FILL = "fill"
    STRETCH = "stretch"


class PaintMode(str, Enum):
    """How aggressively planning may trade fidelity for painting speed.

    ``EXACT`` preserves the raw quantized image and the classic row-by-row
    plan.  The other modes run the optimizer pipeline with progressively
    looser perceptual tolerances and progressively bolder brush work.
    """

    EXACT = "exact"
    QUALITY = "quality"
    BALANCED = "balanced"
    FAST = "fast"


class BrushShape(str, Enum):
    """The two Rust brush shapes planning understands."""

    SQUARE = "square"
    CIRCLE = "circle"


class CropAlignment(str, Enum):
    CENTER = "center"
    TOP = "top"
    BOTTOM = "bottom"
    LEFT = "left"
    RIGHT = "right"


class TransparencyMode(str, Enum):
    LEAVE_UNPAINTED = "leave_unpainted"
    USE_BACKGROUND = "use_background"


class BackgroundRemovalScope(str, Enum):
    """How far a background match is allowed to reach into the image."""

    CONNECTED = "connected"
    EVERYWHERE = "everywhere"


class HueDirection(str, Enum):
    TOP_TO_BOTTOM = "top_to_bottom"
    BOTTOM_TO_TOP = "bottom_to_top"


class ValueDirection(str, Enum):
    TOP_BRIGHT = "top_bright"
    TOP_DARK = "top_dark"


class SaturationDirection(str, Enum):
    LEFT_LOW = "left_low"
    LEFT_HIGH = "left_high"


class HSVColor(NamedTuple):
    """HSV color with hue in degrees and saturation/value in ``[0, 1]``."""

    hue: float
    saturation: float
    value: float

    @property
    def h(self) -> float:
        return self.hue

    @property
    def s(self) -> float:
        return self.saturation

    @property
    def v(self) -> float:
        return self.value


@dataclass(frozen=True, slots=True)
class ScreenRect:
    """A physical-screen rectangle whose right and bottom edges are exclusive."""

    left: int
    top: int
    width: int
    height: int

    def __post_init__(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ValueError("A screen rectangle must have positive width and height")

    @property
    def x(self) -> int:
        return self.left

    @property
    def y(self) -> int:
        return self.top

    @property
    def right(self) -> int:
        return self.left + self.width

    @property
    def bottom(self) -> int:
        return self.top + self.height

    @property
    def center(self) -> tuple[float, float]:
        return (self.left + self.width / 2.0, self.top + self.height / 2.0)

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height

    def contains(self, x: float, y: float) -> bool:
        return self.left <= x < self.right and self.top <= y < self.bottom

    def clamp(self, x: float, y: float) -> tuple[float, float]:
        """Clamp to valid integer mouse-coordinate bounds inside the rectangle."""

        return (
            min(max(x, float(self.left)), float(self.right - 1)),
            min(max(y, float(self.top)), float(self.bottom - 1)),
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "left": self.left,
            "top": self.top,
            "width": self.width,
            "height": self.height,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScreenRect":
        left = int(value.get("left", value.get("x", 0)))
        top = int(value.get("top", value.get("y", 0)))
        if "width" in value and "height" in value:
            width, height = int(value["width"]), int(value["height"])
        elif "right" in value and "bottom" in value:
            width = int(value["right"]) - left
            height = int(value["bottom"]) - top
        else:
            raise ValueError("Rectangle needs width/height or right/bottom")
        return cls(left, top, width, height)

    @classmethod
    def from_points(
        cls, start: tuple[int, int], end: tuple[int, int]
    ) -> "ScreenRect":
        """Build from two exclusive-edge coordinates in either drag direction."""

        left, right = sorted((start[0], end[0]))
        top, bottom = sorted((start[1], end[1]))
        return cls(left, top, right - left, bottom - top)


# A concise alias is useful in calibration/profile code.
Rect = ScreenRect


@dataclass(slots=True)
class ImageProcessOptions:
    logical_width: int
    logical_height: int
    scale_mode: ScaleMode | str = ScaleMode.FIT
    crop_alignment: CropAlignment | str = CropAlignment.CENTER
    color_count: int = 32
    dither: bool = False
    background_color: RGBColor | None = None
    transparency_mode: TransparencyMode | str = TransparencyMode.LEAVE_UNPAINTED
    transparent_fill_color: RGBColor | None = None
    alpha_threshold: int = 0
    remove_background: bool = False
    # ``None`` asks the processor to read the key color off the artwork edges.
    background_removal_color: RGBColor | None = None
    background_removal_tolerance: float = 12.0
    background_removal_scope: BackgroundRemovalScope | str = (
        BackgroundRemovalScope.CONNECTED
    )

    def __post_init__(self) -> None:
        if self.logical_width <= 0 or self.logical_height <= 0:
            raise ValueError("Logical image dimensions must be positive")
        if not 1 <= self.color_count <= 256:
            raise ValueError("Color count must be between 1 and 256")
        if not 0 <= self.alpha_threshold <= 255:
            raise ValueError("Alpha threshold must be between 0 and 255")
        if not 0.0 <= float(self.background_removal_tolerance) <= 100.0:
            raise ValueError("Background removal tolerance must be between 0 and 100")


@dataclass(slots=True)
class ProcessedImage:
    """A quantized logical image and the pixels that should actually be painted."""

    image: "Image.Image"
    paint_mask: "np.ndarray[Any, Any]"
    requested_colors: int

    @property
    def width(self) -> int:
        return self.image.width

    @property
    def height(self) -> int:
        return self.image.height

    @property
    def size(self) -> tuple[int, int]:
        return self.image.size

    @property
    def painted_pixel_count(self) -> int:
        return int(self.paint_mask.sum())


@dataclass(frozen=True, slots=True)
class Stroke:
    """A logical-pixel stroke with inclusive endpoints."""

    start_x: int
    start_y: int
    end_x: int
    end_y: int

    def __post_init__(self) -> None:
        if min(self.start_x, self.start_y, self.end_x, self.end_y) < 0:
            raise ValueError("Logical stroke coordinates cannot be negative")

    @property
    def is_horizontal(self) -> bool:
        return self.start_y == self.end_y

    @property
    def y(self) -> int:
        if not self.is_horizontal:
            raise AttributeError("A non-horizontal stroke has no single y value")
        return self.start_y

    @property
    def pixel_count(self) -> int:
        return max(abs(self.end_x - self.start_x), abs(self.end_y - self.start_y)) + 1

    @property
    def logical_length(self) -> float:
        return hypot(self.end_x - self.start_x, self.end_y - self.start_y)


@dataclass(frozen=True, slots=True)
class ColorGroup:
    """Strokes sharing one color, and optionally one brush size and shape.

    ``brush_diameter`` is in logical cells.  The defaults describe every plan
    the classic pipeline produces, so existing plans keep their meaning: one
    cell per stroke, whatever brush shape Rust currently has selected.
    """

    color: RGBColor
    strokes: tuple[Stroke, ...]
    pixel_count: int
    brush_diameter: int = 1
    brush_shape: str | None = None

    @property
    def rgb(self) -> RGBColor:
        return self.color


@dataclass(frozen=True, slots=True)
class PaintStatistics:
    logical_width: int
    logical_height: int
    unique_colors: int
    stroke_count: int
    painted_pixels: int
    unpainted_pixels: int
    estimated_mouse_travel: float = 0.0
    estimated_seconds: float = 0.0

    @property
    def total_pixels(self) -> int:
        return self.painted_pixels + self.unpainted_pixels


@dataclass(frozen=True, slots=True)
class PaintPlan:
    width: int
    height: int
    color_groups: tuple[ColorGroup, ...]
    unpainted_pixels: int = 0
    _statistics: PaintStatistics | None = field(default=None, repr=False, compare=False)

    @property
    def groups(self) -> tuple[ColorGroup, ...]:
        return self.color_groups

    @property
    def logical_width(self) -> int:
        return self.width

    @property
    def logical_height(self) -> int:
        return self.height

    @property
    def stroke_count(self) -> int:
        return sum(len(group.strokes) for group in self.color_groups)

    @property
    def painted_pixels(self) -> int:
        return sum(group.pixel_count for group in self.color_groups)

    @property
    def statistics(self) -> PaintStatistics:
        if self._statistics is not None:
            return self._statistics
        return PaintStatistics(
            logical_width=self.width,
            logical_height=self.height,
            unique_colors=len(self.color_groups),
            stroke_count=self.stroke_count,
            painted_pixels=self.painted_pixels,
            unpainted_pixels=self.unpainted_pixels,
        )

    def iter_strokes(self) -> Iterator[tuple[RGBColor, Stroke]]:
        for group in self.color_groups:
            for stroke in group.strokes:
                yield group.color, stroke
