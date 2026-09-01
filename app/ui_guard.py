"""Notice when Rust's painting UI is no longer on the screen.

A painting job runs for hours, and in those hours the server can restart,
the player can be kicked, or the sign can simply be closed.  Nothing in the
existing guards notices: the Rust window is still the foreground window,
and the cursor goes exactly where the painter puts it - onto a main menu,
or into the game world, where every stroke is a click on something that is
not a sign.

The calibration already says where the painting UI's fixed furniture is:
the colour box, the hue bar, the Clear (trash) button, the Save button.
Those widgets look the same from the first stroke to the last, so a copy of
each taken as the job starts is a fingerprint of "the painting UI is open",
and a capture that no longer resembles it is the UI having gone.

Resemblance is measured as correlation rather than pixel error.  Pixel
error saturates: a dark button against a dark wall scores well on it even
though nothing of the button is there.  Correlation asks whether the
*structure* of the widget - its text, its icon, its gradient - is still in
place, and is indifferent to the brightness changes a hover highlight or a
lighting tweak brings.  The colour box is compared on saturation and value
only, because its hue is whatever colour the painter last picked.

The hue bar doubles as the proof that the fingerprint was taken of the
painting UI at all: a strip of fully saturated colour whose hue sweeps the
whole spectrum in order is not something the game world, a menu, or a
desktop produces by accident.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Callable, Sequence

import numpy as np

from .models import ScreenRect

if TYPE_CHECKING:
    from PIL.Image import Image


CaptureFunction = Callable[[ScreenRect], Any]

# A widget's fingerprint is taken at this many pixels along its longer side.
# Small enough that a pixel or two of capture jitter, the picker's marker,
# and the text in the Size field do not register; large enough that a
# button's label and icon still do.
_SIGNATURE_LIMIT = 48

# Below this correlation with its fingerprint a widget is missing.  The
# widgets themselves score above 0.9 through a whole job; anything else on
# the screen in their place scores around zero either way.
DEFAULT_MINIMUM_CORRELATION = 0.5

# A widget whose fingerprint has no structure at all - a flat panel - cannot
# be correlated with anything, so it is compared by mean pixel error instead.
_FLAT_SIGNATURE_VARIANCE = 1e-4
_FLAT_MINIMUM_SIMILARITY = 0.85

# What a capture must show to be accepted as the hue bar.
_HUE_BAR_SATURATED_FRACTION = 0.5
_HUE_BAR_BINS = 12
_HUE_BAR_BINS_COVERED = 8
_HUE_BAR_MONOTONIC_FRACTION = 0.8


@dataclass(frozen=True, slots=True)
class WatchedRegion:
    """One piece of the painting UI the guard keeps an eye on."""

    name: str
    rect: ScreenRect
    # Compared on saturation and value only: the colour box shows the hue
    # the painter last picked, which changes with every colour group.
    hue_invariant: bool = False
    # Must look like the hue bar for a fingerprint taken from it to count.
    is_hue_bar: bool = False


@dataclass(frozen=True, slots=True)
class UiCheck:
    """The verdict of one look at the screen."""

    present: bool
    scores: dict[str, float] = field(default_factory=dict)
    missing: tuple[str, ...] = ()

    def describe(self) -> str:
        if not self.scores:
            return "nothing watched"
        return ", ".join(f"{name} {score:.2f}" for name, score in self.scores.items())


def _downsampled(image: "Image") -> "Image":
    width, height = image.size
    factor = max(1, int(np.ceil(max(width, height) / _SIGNATURE_LIMIT)))
    return image.reduce(factor) if factor > 1 else image


def region_signature(image: "Image", *, hue_invariant: bool = False) -> np.ndarray:
    """The feature vector a widget is recognised by.

    RGB of a downsampled copy, or saturation and value for a widget whose
    hue is expected to change.
    """

    pixels = np.asarray(_downsampled(image.convert("RGB")), dtype=np.float32) / 255.0
    if not hue_invariant:
        return pixels.ravel()
    brightest = pixels.max(axis=2)
    darkest = pixels.min(axis=2)
    saturation = np.where(
        brightest > 0.0, (brightest - darkest) / np.maximum(brightest, 1e-6), 0.0
    )
    return np.concatenate([saturation.ravel(), brightest.ravel()]).astype(np.float32)


def signature_similarity(reference: np.ndarray, current: np.ndarray) -> float:
    """How much of ``reference``'s structure ``current`` still has, 0 to 1.

    Pearson correlation clipped at zero; a flat reference, which has no
    structure to correlate, falls back to one minus the mean pixel error.
    """

    if reference.shape != current.shape:
        return 0.0
    centred_reference = reference - reference.mean()
    centred_current = current - current.mean()
    reference_energy = float((centred_reference * centred_reference).sum())
    if reference_energy / max(1, reference.size) < _FLAT_SIGNATURE_VARIANCE:
        return float(max(0.0, 1.0 - np.abs(reference - current).mean()))
    current_energy = float((centred_current * centred_current).sum())
    if current_energy <= 0.0:
        return 0.0
    correlation = float((centred_reference * centred_current).sum()) / np.sqrt(
        reference_energy * current_energy
    )
    return float(min(1.0, max(0.0, correlation)))


def looks_like_hue_bar(image: "Image", *, reduced: bool = False) -> bool:
    """Whether ``image`` is a strip of saturated colour sweeping the spectrum.

    Most of it must be saturated, the hues in it must cover most of the
    colour wheel, and they must run in order along the strip: three things
    a photograph or a menu screen in the same place very rarely manage at
    once, and the real bar manages with room to spare. ``reduced`` permits
    the small amount of bilinear colour blending introduced when a full
    monitor capture is downsampled only for setup detection; the live safety
    guard deliberately keeps the stricter default.
    """

    from PIL import Image as PillowImage

    working = image.convert("RGB")
    if working.size[1] < working.size[0]:
        working = working.transpose(PillowImage.Transpose.ROTATE_90)
    hsv = np.asarray(working.convert("HSV"), dtype=np.float32) / 255.0
    hue, saturation, value = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    saturated = (saturation > 0.5) & (value > 0.3)
    saturated_fraction = 0.35 if reduced else _HUE_BAR_SATURATED_FRACTION
    bins = 8 if reduced else _HUE_BAR_BINS
    bins_covered = 5 if reduced else _HUE_BAR_BINS_COVERED
    monotonic_fraction = 0.65 if reduced else _HUE_BAR_MONOTONIC_FRACTION
    if float(saturated.mean()) < saturated_fraction:
        return False
    histogram = np.histogram(hue[saturated], bins=bins, range=(0.0, 1.0))[0]
    covered = int((histogram / max(1, int(saturated.sum())) >= 0.01).sum())
    if covered < bins_covered:
        return False
    angle = hue * 2.0 * np.pi
    row_x = (np.cos(angle) * saturated).sum(axis=1)
    row_y = (np.sin(angle) * saturated).sum(axis=1)
    populated = saturated.sum(axis=1) > 0
    row_hue = (np.arctan2(row_y[populated], row_x[populated]) / (2.0 * np.pi)) % 1.0
    if row_hue.size < (3 if reduced else 4):
        return False
    steps = (np.diff(row_hue) + 0.5) % 1.0 - 0.5
    steps = steps[np.abs(steps) > 1e-3]
    if steps.size == 0:
        return False
    ordered = max(float((steps > 0).mean()), float((steps < 0).mean()))
    return ordered >= monotonic_fraction


class PaintingUiGuard:
    """Fingerprint the painting UI's fixed widgets and recognise them later."""

    def __init__(
        self,
        regions: Sequence[WatchedRegion],
        *,
        minimum_correlation: float = DEFAULT_MINIMUM_CORRELATION,
    ) -> None:
        if not regions:
            raise ValueError("The UI guard needs at least one region to watch")
        self._regions = tuple(regions)
        self._minimum = float(minimum_correlation)
        self._signatures: dict[str, np.ndarray] | None = None

    @classmethod
    def for_target(cls, target: Any) -> "PaintingUiGuard | None":
        """The guard for a painting target, or None if it has nothing to watch.

        The colour box and hue bar are always calibrated; the Clear and
        Save buttons join when they are.  The Size field is left out: the
        painter types into it, so its contents are expected to change.
        """

        regions: list[WatchedRegion] = []
        for name, hue_invariant, is_hue_bar in (
            ("color_box", True, False),
            ("hue_bar", False, True),
            ("clear_button", False, False),
            ("save_button", False, False),
        ):
            rect = getattr(target, name, None)
            if rect is None:
                continue
            try:
                screen_rect = ScreenRect(
                    int(rect.left), int(rect.top), int(rect.width), int(rect.height)
                )
            except (TypeError, ValueError, AttributeError):
                continue
            regions.append(
                WatchedRegion(
                    name, screen_rect, hue_invariant=hue_invariant, is_hue_bar=is_hue_bar
                )
            )
        return cls(regions) if regions else None

    @property
    def regions(self) -> tuple[WatchedRegion, ...]:
        return self._regions

    @property
    def armed(self) -> bool:
        return self._signatures is not None

    def arm(self, capture: CaptureFunction) -> bool:
        """Take the fingerprint from the screen as it is now.

        Returns False, and stays unarmed, when the capture does not show the
        painting UI - judged by the hue bar, when one is watched.  A guard
        armed on the game world would do the opposite of its job: pause the
        moment the sign is opened.
        """

        captures = {region.name: capture(region.rect) for region in self._regions}
        for region in self._regions:
            if region.is_hue_bar and not looks_like_hue_bar(captures[region.name]):
                self._signatures = None
                return False
        self._signatures = {
            region.name: region_signature(
                captures[region.name], hue_invariant=region.hue_invariant
            )
            for region in self._regions
        }
        return True

    def disarm(self) -> None:
        self._signatures = None

    def check(self, capture: CaptureFunction) -> UiCheck:
        """Look at the screen and say whether the painting UI is still there.

        The UI counts as gone when more of its widgets are missing than are
        present.  One widget can be hidden by a tooltip or a highlight; the
        lot of them disappear together only when the UI does.
        """

        signatures = self._signatures
        if signatures is None:
            raise RuntimeError("The UI guard has not been armed")
        scores: dict[str, float] = {}
        missing: list[str] = []
        for region in self._regions:
            current = region_signature(
                capture(region.rect), hue_invariant=region.hue_invariant
            )
            score = signature_similarity(signatures[region.name], current)
            scores[region.name] = score
            if score < self._minimum:
                missing.append(region.name)
        present = len(missing) <= len(self._regions) - len(missing)
        return UiCheck(present=present, scores=scores, missing=tuple(missing))


__all__ = [
    "DEFAULT_MINIMUM_CORRELATION",
    "PaintingUiGuard",
    "UiCheck",
    "WatchedRegion",
    "looks_like_hue_bar",
    "region_signature",
    "signature_similarity",
]
