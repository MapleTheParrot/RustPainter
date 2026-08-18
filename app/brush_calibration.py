"""Measure Rust's on-screen brush preview for automatic size matching."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from PIL.Image import Image


@dataclass(frozen=True, slots=True)
class BrushFootprint:
    """Detected brush bounds inside a calibrated preview capture."""

    left: int
    top: int
    width: int
    height: int
    confidence: float

    @property
    def diameter(self) -> float:
        """Conservative diameter used to match a logical paint cell."""

        return float(max(self.width, self.height))


def measure_brush_footprint(image: "Image") -> BrushFootprint:
    """Find the centered colored brush shape on Rust's gray preview tile.

    The calibrated region should contain only the gray preview tile.  Its edge
    pixels provide a robust background estimate; the connected foreground
    component nearest the tile center is treated as the brush footprint.
    """

    pixels = np.asarray(image.convert("RGB"), dtype=np.float32)
    if pixels.ndim != 3 or pixels.shape[2] < 3:
        raise ValueError("Brush preview capture must be an RGB image")
    height, width = pixels.shape[:2]
    if width < 8 or height < 8:
        raise ValueError("Brush preview calibration is too small")

    border_width = max(1, min(width, height) // 12)
    border_mask = np.zeros((height, width), dtype=np.bool_)
    border_mask[:border_width, :] = True
    border_mask[-border_width:, :] = True
    border_mask[:, :border_width] = True
    border_mask[:, -border_width:] = True
    border_pixels = pixels[border_mask]
    background = np.median(border_pixels, axis=0)
    distances = np.linalg.norm(pixels - background, axis=2)
    border_distances = distances[border_mask]
    # Rust's preview background has a subtle texture.  Keep its ordinary noise
    # out while still accepting dark, white, and saturated brush colors.
    threshold = max(24.0, float(np.percentile(border_distances, 98)) * 2.25)
    foreground = distances >= threshold

    components: list[tuple[int, int, int, int, int, float]] = []
    visited = np.zeros_like(foreground)
    center_x = (width - 1) / 2.0
    center_y = (height - 1) / 2.0
    for start_y, start_x in np.argwhere(foreground):
        y = int(start_y)
        x = int(start_x)
        if visited[y, x]:
            continue
        queue: deque[tuple[int, int]] = deque([(x, y)])
        visited[y, x] = True
        min_x = max_x = x
        min_y = max_y = y
        area = 0
        while queue:
            current_x, current_y = queue.popleft()
            area += 1
            min_x = min(min_x, current_x)
            max_x = max(max_x, current_x)
            min_y = min(min_y, current_y)
            max_y = max(max_y, current_y)
            for offset_x, offset_y in (
                (-1, -1), (0, -1), (1, -1),
                (-1, 0),             (1, 0),
                (-1, 1),  (0, 1),  (1, 1),
            ):
                next_x = current_x + offset_x
                next_y = current_y + offset_y
                if (
                    0 <= next_x < width
                    and 0 <= next_y < height
                    and foreground[next_y, next_x]
                    and not visited[next_y, next_x]
                ):
                    visited[next_y, next_x] = True
                    queue.append((next_x, next_y))
        component_x = (min_x + max_x) / 2.0
        component_y = (min_y + max_y) / 2.0
        distance_to_center = (component_x - center_x) ** 2 + (
            component_y - center_y
        ) ** 2
        components.append((min_x, min_y, max_x, max_y, area, distance_to_center))

    minimum_area = max(4, round(width * height * 0.0005))
    viable = [component for component in components if component[4] >= minimum_area]
    if not viable:
        raise ValueError(
            "No brush shape was detected. Recalibrate only the gray preview tile "
            "and use a solid circle or square brush."
        )
    min_x, min_y, max_x, max_y, area, distance_to_center = min(
        viable,
        key=lambda component: (component[5], -component[4]),
    )
    detected_width = max_x - min_x + 1
    detected_height = max_y - min_y + 1
    if distance_to_center > (min(width, height) * 0.2) ** 2:
        raise ValueError(
            "The detected shape is not centered in the brush preview. "
            "Recalibrate the gray preview tile."
        )
    fill_ratio = area / float(detected_width * detected_height)
    confidence = min(1.0, max(0.0, fill_ratio))
    return BrushFootprint(
        left=min_x,
        top=min_y,
        width=detected_width,
        height=detected_height,
        confidence=confidence,
    )


__all__ = ["BrushFootprint", "measure_brush_footprint"]
