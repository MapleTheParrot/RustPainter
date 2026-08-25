from __future__ import annotations

import unittest

import numpy as np
from PIL import Image

from app.color_mapping import (
    map_hue_to_screen,
    map_sv_to_screen,
    rgb_to_hsv,
    rgb_to_picker_coordinates,
)
from app.coordinates import logical_pixel_center, screen_to_logical_pixel
from app.image_processing import (
    background_mask,
    detect_background_colors,
    calculate_fill_size,
    calculate_fit_size,
    detect_background_color,
    is_reduction,
    process_image,
    quantize_image,
    remove_background,
    scale_image,
    sharpen_image,
)
from app.models import (
    BackgroundRemovalScope,
    ImageProcessOptions,
    ScreenRect,
    SharpenMode,
)
from app.paint_plan import generate_paint_plan, group_horizontal_runs


class AspectTransformTests(unittest.TestCase):
    def test_aspect_size_calculations(self) -> None:
        self.assertEqual(calculate_fit_size((400, 200), (100, 100)), (100, 50))
        self.assertEqual(calculate_fill_size((400, 200), (100, 100)), (200, 100))
        self.assertEqual(calculate_fit_size((200, 400), (101, 50)), (25, 50))

    def test_fit_centers_image_and_leaves_bars_unpainted(self) -> None:
        source = Image.new("RGBA", (4, 2), (220, 10, 20, 255))
        result, mask = scale_image(source, (4, 4), "fit")

        self.assertEqual(result.size, (4, 4))
        np.testing.assert_array_equal(mask[0], [False] * 4)
        np.testing.assert_array_equal(mask[1:3], np.ones((2, 4), dtype=bool))
        np.testing.assert_array_equal(mask[3], [False] * 4)
        self.assertEqual(result.getpixel((2, 1)), (220, 10, 20, 255))

    def test_fit_can_paint_letterbox_background(self) -> None:
        source = Image.new("RGBA", (4, 2), (255, 0, 0, 255))
        result, mask = scale_image(
            source, (4, 4), "fit", background_color=(255, 255, 255)
        )
        self.assertTrue(mask.all())
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255, 255))

    def test_fill_crop_alignment(self) -> None:
        source = Image.new("RGBA", (4, 1))
        source.putdata(
            [
                (255, 0, 0, 255),
                (0, 255, 0, 255),
                (0, 0, 255, 255),
                (255, 255, 0, 255),
            ]
        )

        center, _ = scale_image(source, (2, 1), "fill", alignment="center")
        left, _ = scale_image(source, (2, 1), "fill", alignment="left")
        right, _ = scale_image(source, (2, 1), "fill", alignment="right")
        pixels = lambda image: [image.getpixel((x, 0)) for x in range(2)]
        self.assertEqual(pixels(center), [(0, 255, 0, 255), (0, 0, 255, 255)])
        self.assertEqual(pixels(left), [(255, 0, 0, 255), (0, 255, 0, 255)])
        self.assertEqual(pixels(right), [(0, 0, 255, 255), (255, 255, 0, 255)])

    def test_downscale_preserves_brightness_in_linear_light(self) -> None:
        # Half black, half white is half the light, which sRGB encodes as 188.
        # Averaging the gamma-encoded codes instead would land near 128 and take
        # the whole sign with it.
        checker = (np.indices((64, 64)).sum(axis=0) % 2 * 255).astype(np.uint8)
        source = Image.fromarray(checker).convert("RGBA")

        result, _ = scale_image(source, (8, 8), "stretch")

        levels = np.asarray(result.convert("RGB"), dtype=np.int16)
        self.assertTrue(np.all(np.abs(levels - 188) <= 2), levels.min())

    def test_stretch_uses_exact_logical_size(self) -> None:
        source = Image.new("RGBA", (5, 2), (20, 40, 60, 255))
        result, mask = scale_image(source, (3, 7), "stretch")
        self.assertEqual(result.size, (3, 7))
        self.assertTrue(mask.all())
        self.assertEqual(result.getpixel((1, 4)), (20, 40, 60, 255))


class TransparencyAndQuantizationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.source = Image.new("RGBA", (2, 1))
        self.source.putdata([(200, 20, 30, 255), (10, 200, 30, 0)])

    def test_transparent_source_pixel_can_remain_unpainted(self) -> None:
        result, mask = scale_image(
            self.source, (2, 1), "stretch", transparency_mode="leave_unpainted"
        )
        np.testing.assert_array_equal(mask, [[True, False]])
        self.assertEqual(result.getpixel((1, 0))[3], 0)

    def test_transparent_source_pixel_can_use_background(self) -> None:
        result, mask = scale_image(
            self.source,
            (2, 1),
            "stretch",
            transparency_mode="use_background",
            transparent_fill_color=(7, 8, 9),
        )
        self.assertTrue(mask.all())
        self.assertEqual(result.getpixel((1, 0)), (7, 8, 9, 255))

    def test_a_soft_edge_is_dropped_rather_than_painted_as_background(self) -> None:
        """Half-transparent pixels are what an unwanted halo is made of.

        Blending them into the background paints a ring of near-background
        around a cut-out subject, which is the opposite of what "leave
        transparent pixels unpainted" was asked for.
        """

        source = Image.new("RGBA", (3, 1))
        source.putdata([(200, 20, 30, 255), (200, 20, 30, 90), (10, 200, 30, 0)])
        result, mask = scale_image(
            source, (3, 1), "stretch", transparent_fill_color=(255, 255, 255)
        )
        np.testing.assert_array_equal(mask, [[True, False, False]])
        # The opaque pixel keeps its own color rather than a blended one.
        self.assertEqual(result.getpixel((0, 0)), (200, 20, 30, 255))
        self.assertEqual(result.getpixel((1, 0))[3], 0)

    def test_alpha_fill_blends_a_soft_edge_into_the_background(self) -> None:
        source = Image.new("RGBA", (2, 1))
        source.putdata([(200, 20, 30, 255), (0, 0, 0, 128)])
        result, mask = scale_image(
            source,
            (2, 1),
            "stretch",
            alpha_fill=True,
            transparent_fill_color=(255, 255, 255),
        )
        self.assertTrue(mask.all())
        red, green, blue, alpha = result.getpixel((1, 0))
        self.assertEqual(alpha, 255)
        # 128/255 of black over white, so every channel lands mid-grey.
        self.assertEqual((red, green, blue), (127, 127, 127))

    def test_alpha_fill_is_off_by_default(self) -> None:
        self.assertFalse(ImageProcessOptions(logical_width=8, logical_height=8).alpha_fill)

    def test_quantization_limits_painted_colors_and_preserves_mask(self) -> None:
        image = Image.new("RGBA", (5, 1))
        image.putdata(
            [
                (255, 0, 0, 255),
                (0, 255, 0, 255),
                (0, 0, 255, 255),
                (255, 255, 0, 255),
                (123, 45, 67, 0),
            ]
        )
        mask = np.array([[True, True, True, True, False]])
        result = quantize_image(image, 2, paint_mask=mask)
        array = np.asarray(result)
        painted = array[:, :, :3][mask]
        self.assertLessEqual(len(np.unique(painted, axis=0)), 2)
        self.assertEqual(result.getpixel((4, 0))[3], 0)


    def test_faint_tints_snap_to_gray_and_deliberate_pastels_survive(self) -> None:
        # A near-white pixel's hue is noise, and the picker would commit to it.
        image = Image.new("RGBA", (3, 1))
        image.putdata(
            [(250, 252, 255, 255), (252, 253, 252, 255), (210, 225, 250, 255)]
        )

        result = quantize_image(image, 16)

        self.assertEqual(result.getpixel((0, 0))[:3], (255, 255, 255))
        self.assertEqual(result.getpixel((1, 0))[:3], (253, 253, 253))
        self.assertEqual(result.getpixel((2, 0))[:3], (210, 225, 250))

    def test_snapping_keeps_palette_slots_for_the_artwork(self) -> None:
        # Four indistinguishable near-whites must not each buy a hued entry.
        image = Image.new("RGBA", (6, 1))
        image.putdata(
            [
                (252, 253, 252, 255),
                (254, 252, 251, 255),
                (252, 252, 253, 255),
                (251, 254, 253, 255),
                (250, 190, 44, 255),
                (120, 60, 220, 255),
            ]
        )

        result = quantize_image(image, 4)

        painted = [result.getpixel((x, 0))[:3] for x in range(6)]
        for gray in painted[:4]:
            self.assertEqual(len(set(gray)), 1, gray)
        # Every palette entry the near-whites did not need stays with the art.
        self.assertEqual(painted[4], (250, 190, 44))
        self.assertEqual(painted[5], (120, 60, 220))

    def test_quantization_keeps_a_rare_distinct_hue(self) -> None:
        """A dominant warm gamut must not erase a small magenta accent."""

        height = width = 64
        rows, columns = np.indices((height, width))
        pixels = np.empty((height, width, 4), dtype=np.uint8)
        pixels[:, :, 0] = 180 + columns * 75 // (width - 1)
        pixels[:, :, 1] = 20 + rows * 140 // (height - 1)
        pixels[:, :, 2] = columns * 40 // (width - 1)
        pixels[:, :, 3] = 255
        magenta = (228, 19, 173)
        pixels[0, 0, :3] = magenta

        result = quantize_image(Image.fromarray(pixels, mode="RGBA"), 16)
        red, green, blue = result.getpixel((0, 0))[:3]

        # Median cut mapped this to (236, 26, 30), visually deleting the hue.
        self.assertGreater(blue, green + 80)
        self.assertLessEqual(max(abs(red - magenta[0]), abs(blue - magenta[2])), 5)


class CoordinateTests(unittest.TestCase):
    def test_logical_pixels_map_to_cell_centers_on_negative_monitor(self) -> None:
        canvas = ScreenRect(-200, 100, 200, 100)
        self.assertEqual(logical_pixel_center(0, 0, 2, 2, canvas), (-150.0, 125.0))
        self.assertEqual(logical_pixel_center(1, 1, 2, 2, canvas), (-50.0, 175.0))
        self.assertEqual(screen_to_logical_pixel(-150, 125, 2, 2, canvas), (0, 0))

    def test_exclusive_right_edge_is_not_in_canvas(self) -> None:
        canvas = ScreenRect(-100, -50, 20, 10)
        self.assertTrue(canvas.contains(-81, -41))
        self.assertFalse(canvas.contains(-80, -41))


class ColorMappingTests(unittest.TestCase):
    def test_rgb_to_hsv_primary_colors(self) -> None:
        red = rgb_to_hsv((255, 0, 0))
        green = rgb_to_hsv(0, 255, 0)
        blue = rgb_to_hsv((0, 0, 255))
        self.assertAlmostEqual(red.hue, 0.0)
        self.assertAlmostEqual(green.hue, 120.0)
        self.assertAlmostEqual(blue.hue, 240.0)
        self.assertEqual((red.saturation, red.value), (1.0, 1.0))

    def test_hue_orientation_is_configurable(self) -> None:
        hue_bar = ScreenRect(100, 200, 11, 101)
        top_down = map_hue_to_screen(120, hue_bar, "top_to_bottom")
        bottom_up = map_hue_to_screen(120, hue_bar, "bottom_to_top")
        self.assertEqual(top_down[0], 105.0)
        self.assertAlmostEqual(top_down[1], 200 + 100 / 3)
        self.assertAlmostEqual(bottom_up[1], 200 + 200 / 3)

    def test_saturation_and_value_orientations(self) -> None:
        color_box = ScreenRect(-100, 50, 101, 201)
        normal = map_sv_to_screen(0.25, 0.75, color_box)
        reversed_axes = map_sv_to_screen(
            0.25,
            0.75,
            color_box,
            saturation_direction="right_to_left",
            value_direction="bottom_to_top",
        )
        self.assertEqual(normal, (-75.0, 100.0))
        self.assertEqual(reversed_axes, (-25.0, 200.0))

    def test_combined_rgb_picker_mapping_stays_inside_rectangles(self) -> None:
        hue_bar = ScreenRect(-20, 10, 3, 20)
        color_box = ScreenRect(-200, -100, 100, 50)
        points = rgb_to_picker_coordinates((255, 0, 255), hue_bar, color_box)
        self.assertTrue(hue_bar.contains(*points.hue))
        self.assertTrue(color_box.contains(*points.saturation_value))


class PaintPlanTests(unittest.TestCase):
    def setUp(self) -> None:
        red = (255, 0, 0, 255)
        green = (0, 255, 0, 255)
        blue = (0, 0, 255, 255)
        clear = (99, 88, 77, 0)
        self.image = np.array(
            [
                [red, red, green, green],
                [red, clear, green, green],
                [blue, blue, blue, red],
            ],
            dtype=np.uint8,
        )

    def test_horizontal_runs_are_maximal_and_transparency_splits_them(self) -> None:
        groups = group_horizontal_runs(self.image)
        red_runs = groups[(255, 0, 0)]
        self.assertEqual(
            [(s.start_x, s.start_y, s.end_x, s.end_y) for s in red_runs],
            [(0, 0, 1, 0), (0, 1, 0, 1), (3, 2, 3, 2)],
        )
        self.assertEqual(len(groups[(0, 255, 0)]), 2)
        self.assertEqual(len(groups[(0, 0, 255)]), 1)

    def test_plan_counts_and_statistics(self) -> None:
        plan = generate_paint_plan(self.image)
        self.assertEqual((plan.width, plan.height), (4, 3))
        self.assertEqual(len(plan.color_groups), 3)
        self.assertEqual(plan.stroke_count, 6)
        self.assertEqual(plan.painted_pixels, 11)
        self.assertEqual(plan.unpainted_pixels, 1)
        self.assertEqual(plan.statistics.stroke_count, 6)
        self.assertGreater(plan.statistics.estimated_mouse_travel, 0)
        self.assertGreater(plan.statistics.estimated_seconds, 0)

    def test_overpaint_merging_crosses_later_colors_only(self) -> None:
        red = (255, 0, 0, 255)
        green = (0, 255, 0, 255)
        clear = (0, 0, 0, 0)
        # Row 0: red pixels split by one green pixel (later color: mergeable).
        # Row 1: red pixels split by an unpainted pixel (never mergeable).
        image = np.array(
            [
                [red, green, red, red],
                [red, clear, red, red],
            ],
            dtype=np.uint8,
        )
        merged = generate_paint_plan(image, color_order="frequency", overpaint_gap=None)
        exact = generate_paint_plan(image, color_order="frequency", overpaint_gap=0)

        self.assertEqual(exact.stroke_count, 5)
        self.assertEqual(merged.stroke_count, 4)
        red_group = merged.color_groups[0]
        self.assertEqual(red_group.color, (255, 0, 0))
        self.assertEqual(
            [(s.start_x, s.start_y, s.end_x, s.end_y) for s in red_group.strokes],
            [(0, 0, 3, 0), (0, 1, 0, 1), (2, 1, 3, 1)],
        )
        # True pixel counts are preserved even though the merged stroke is longer.
        self.assertEqual(red_group.pixel_count, 6)
        self.assertEqual(merged.painted_pixels, exact.painted_pixels)

    def test_overpaint_gap_limit_is_respected(self) -> None:
        red = (255, 0, 0, 255)
        green = (0, 255, 0, 255)
        row = [red, red, green, green, green, red, red]
        image = np.array([row], dtype=np.uint8)
        limited = generate_paint_plan(image, color_order="frequency", overpaint_gap=2)
        unlimited = generate_paint_plan(image, color_order="frequency", overpaint_gap=None)
        red_limited = next(g for g in limited.color_groups if g.color == (255, 0, 0))
        red_unlimited = next(g for g in unlimited.color_groups if g.color == (255, 0, 0))
        self.assertEqual(len(red_limited.strokes), 2)
        self.assertEqual(len(red_unlimited.strokes), 1)

    def test_merged_plan_never_crosses_earlier_colors(self) -> None:
        red = (255, 0, 0, 255)
        green = (0, 255, 0, 255)
        # Green is less frequent, so it paints after red and must not paint
        # across red's pixels.
        image = np.array(
            [[green, red, red, red, green, red, green]], dtype=np.uint8
        )
        plan = generate_paint_plan(image, color_order="frequency", overpaint_gap=None)
        self.assertEqual(plan.color_groups[0].color, (255, 0, 0))
        green_group = plan.color_groups[1]
        self.assertEqual(len(green_group.strokes), 3)


class BackgroundRemovalTests(unittest.TestCase):
    """A ring of one color around a subject that encloses the same color."""

    def setUp(self) -> None:
        size = 32
        pixels = np.full((size, size, 4), 255, dtype=np.uint8)
        rows, columns = np.mgrid[0:size, 0:size]
        radius = (rows - 16) ** 2 + (columns - 16) ** 2
        pixels[radius < 100] = (200, 30, 40, 255)
        pixels[radius < 9] = (255, 255, 255, 255)
        self.image = Image.fromarray(pixels, mode="RGBA")
        self.subject = radius < 100
        self.hole = radius < 9

    def test_edge_color_is_detected_from_the_border_ring(self) -> None:
        self.assertEqual(detect_background_color(self.image), (255, 255, 255))

    def test_connected_removal_keeps_an_enclosed_pocket_painted(self) -> None:
        removed = background_mask(self.image, tolerance=5, scope="connected")
        self.assertFalse(bool(removed[16, 16]))
        self.assertFalse(bool(removed[16, 8]))
        self.assertTrue(bool(removed[0, 0]))

    def test_everywhere_removal_also_drops_enclosed_pockets(self) -> None:
        removed = background_mask(self.image, tolerance=5, scope="everywhere")
        self.assertTrue(bool(removed[16, 16]))
        self.assertFalse(bool(removed[16, 8]))

    def test_removal_updates_both_the_mask_and_the_alpha_channel(self) -> None:
        stripped, mask = remove_background(self.image, tolerance=5)
        expected = int(self.subject.sum())
        self.assertEqual(int(mask.sum()), expected)
        self.assertEqual(stripped.getpixel((16, 16))[3], 255)
        self.assertEqual(stripped.getpixel((0, 0))[3], 0)

    def test_an_explicit_color_only_removes_what_it_matches(self) -> None:
        untouched = background_mask(self.image, color=(0, 0, 255), tolerance=0)
        self.assertEqual(int(untouched.sum()), 0)
        subject = background_mask(
            self.image, color=(200, 30, 40), tolerance=2, scope="everywhere"
        )
        self.assertEqual(int(subject.sum()), int((self.subject & ~self.hole).sum()))

    def _processed(self, **overrides: object):
        options = ImageProcessOptions(
            logical_width=32,
            logical_height=32,
            scale_mode="stretch",
            color_count=8,
            remove_background=True,
            background_removal_tolerance=5.0,
            **overrides,  # type: ignore[arg-type]
        )
        return process_image(self.image, options)

    def test_processing_paints_the_subject_and_nothing_around_it(self) -> None:
        processed = self._processed()
        self.assertEqual(processed.painted_pixel_count, int(self.subject.sum()))
        self.assertEqual(processed.image.getpixel((0, 0))[3], 0)
        self.assertEqual(processed.image.getpixel((16, 8))[3], 255)

    def test_a_fully_removed_background_frees_its_palette_entry(self) -> None:
        processed = self._processed(background_removal_scope="everywhere")
        painted = np.asarray(processed.image.convert("RGB"))[processed.paint_mask]
        colors = {tuple(int(channel) for channel in color) for color in painted}
        self.assertNotIn((255, 255, 255), colors)

    def test_letterbox_bars_do_not_hide_the_artwork_edge(self) -> None:
        """Fit leaves unpainted bars, so the ring must follow the artwork."""

        source = Image.new("RGBA", (32, 8), (255, 255, 255, 255))
        source.paste(Image.new("RGBA", (8, 4), (10, 120, 200, 255)), (12, 2))
        scaled, mask = scale_image(source, (32, 32), "fit")
        self.assertEqual(detect_background_color(scaled, mask), (255, 255, 255))
        _, remaining = remove_background(scaled, mask, tolerance=6)
        self.assertLess(int(remaining.sum()), int(mask.sum()))
        self.assertGreater(int(remaining.sum()), 0)


class SmartBackgroundRemovalTests(unittest.TestCase):
    """Backdrops that are not one flat color, which is most of them."""

    @staticmethod
    def _gradient(subject: tuple[int, int, int] = (30, 90, 200)) -> Image.Image:
        """A studio sweep: the backdrop ramps from one end to the other."""

        height, width = 64, 96
        pixels = np.full((height, width, 4), 255, dtype=np.uint8)
        ramp = np.linspace(205, 255, width).astype(np.uint8)
        for channel in range(3):
            pixels[:, :, channel] = ramp[None, :]
        pixels[20:44, 30:66, :3] = subject
        return Image.fromarray(pixels, mode="RGBA")

    def test_several_key_colors_are_read_off_a_gradient(self) -> None:
        image = self._gradient()
        colors = detect_background_colors(image, limit=4, depth=3)
        self.assertGreater(len(colors), 1)
        # One average of the whole ramp would match its middle and neither
        # end; the keys spread across it instead.
        greys = sorted(color[0] for color in colors)
        self.assertGreater(greys[-1] - greys[0], 20)

    def test_a_gradient_backdrop_comes_away_whole(self) -> None:
        image = self._gradient()
        _, mask = scale_image(image, image.size, "stretch")
        backdrop = int(mask.sum()) - 24 * 36

        flat = background_mask(image, tolerance=12, scope="connected")
        smart = background_mask(image, tolerance=12, scope="subject")

        # One flat key cannot span the ramp, so it leaves part of it painted.
        self.assertLess(int(flat.sum()), backdrop)
        self.assertEqual(int(smart.sum()), backdrop)

    def test_a_pale_subject_is_not_shaved_as_if_it_were_a_halo(self) -> None:
        """A pale subject is close to a white backdrop without blending into it.

        The halo pass is deliberately loose, so what stops it eating two
        pixels off every pale subject is not the tolerance: it is that those
        pixels are no nearer the backdrop than the subject behind them, which
        a genuine blend always is.
        """

        height, width = 64, 96
        pixels = np.full((height, width, 4), 255, dtype=np.uint8)
        pixels[20:44, 30:66, :3] = (200, 200, 200)
        image = Image.fromarray(pixels, mode="RGBA")

        removed = background_mask(image, tolerance=12, scope="subject")
        self.assertEqual(int(removed.sum()), height * width - 24 * 36)

    def test_a_soft_edge_does_not_survive_as_a_halo(self) -> None:
        """A blend of subject and backdrop matches neither, and rings the subject."""

        height, width = 64, 64
        rows, columns = np.mgrid[0:height, 0:width]
        radius = np.sqrt((rows - 32.0) ** 2 + (columns - 32.0) ** 2)
        coverage = np.clip((20.0 - radius) / 3.0, 0.0, 1.0)
        pixels = np.full((height, width, 4), 255, dtype=np.uint8)
        for channel, value in enumerate((20, 40, 60)):
            pixels[:, :, channel] = (
                value * coverage + 255 * (1.0 - coverage)
            ).astype(np.uint8)
        image = Image.fromarray(pixels, mode="RGBA")

        flat = background_mask(image, tolerance=6, scope="connected")
        smart = background_mask(image, tolerance=6, scope="subject")

        # The ring of half-blended pixels the flat match leaves behind is
        # exactly what shows up on a sign as an outline around the subject.
        halo = ~flat & smart
        self.assertGreater(int(halo.sum()), 0)
        # It goes without the solid middle of the disc going with it.
        self.assertFalse(bool(smart[32, 32]))
        self.assertTrue(bool(smart[0, 0]))

    def test_the_smart_scope_still_keeps_enclosed_pockets_painted(self) -> None:
        size = 48
        rows, columns = np.mgrid[0:size, 0:size]
        radius = (rows - 24) ** 2 + (columns - 24) ** 2
        pixels = np.full((size, size, 4), 255, dtype=np.uint8)
        pixels[radius < 225] = (200, 30, 40, 255)
        pixels[radius < 25] = (255, 255, 255, 255)
        image = Image.fromarray(pixels, mode="RGBA")

        removed = background_mask(image, tolerance=5, scope="subject")
        self.assertFalse(bool(removed[24, 24]))
        self.assertTrue(bool(removed[0, 0]))

    def test_the_smart_scope_is_what_removal_reaches_for_by_default(self) -> None:
        options = ImageProcessOptions(logical_width=8, logical_height=8)
        self.assertEqual(
            options.background_removal_scope, BackgroundRemovalScope.SUBJECT
        )


class UpscaleCrispnessTests(unittest.TestCase):
    def test_upscaling_pixel_art_yields_blocks_not_blur(self) -> None:
        """Blowing a tiny image up to a large sign must not invent colors.

        A 10x8 sign texture downloaded from the game and re-imported at Max
        quality came out as a smear: Lanczos interpolated hundreds of blend
        colors between the original pixels.  An upscale reveals nothing new,
        so it must reproduce the source pixels as crisp blocks.
        """

        from PIL import Image

        rng = np.random.default_rng(7)
        tiny = Image.fromarray(
            np.dstack(
                [
                    rng.integers(0, 255, (8, 10, 3), dtype=np.uint8),
                    np.full((8, 10, 1), 255, dtype=np.uint8),
                ]
            ),
            "RGBA",
        )
        scaled, _mask = scale_image(tiny, (320, 240), "stretch")
        colors = np.unique(
            np.asarray(scaled.convert("RGB")).reshape(-1, 3), axis=0
        )
        self.assertLessEqual(len(colors), 80)
        # Downscales must keep the smooth linear-light path: a gradient
        # reduced 4x should still produce colors between the source's.
        gradient = Image.fromarray(
            np.dstack(
                [
                    np.tile(np.arange(0, 240, dtype=np.uint8), (64, 1)),
                    np.zeros((64, 240), dtype=np.uint8),
                    np.zeros((64, 240), dtype=np.uint8),
                    np.full((64, 240), 255, dtype=np.uint8),
                ]
            ),
            "RGBA",
        )
        reduced, _mask = scale_image(gradient, (60, 16), "stretch")
        reds = np.unique(np.asarray(reduced.convert("RGB"))[:, :, 0])
        self.assertGreater(len(reds), 30)


class SharpenTests(unittest.TestCase):
    """Sharpening restores edge contrast where a downscale softened it."""

    @staticmethod
    def _line_on_field() -> tuple["Image.Image", np.ndarray]:
        from PIL import Image

        field = np.full((20, 30, 4), (230, 200, 200, 255), dtype=np.uint8)
        field[9, 4:26] = (120, 100, 100, 255)  # a softened line
        field[10, 4:26] = (60, 50, 50, 255)  # its core
        mask = np.ones((20, 30), dtype=np.bool_)
        mask[:, 26:] = False  # letterboxed margin stays unpainted
        field[~mask] = (0, 0, 0, 0)
        return Image.fromarray(field, "RGBA"), mask

    def test_sharpening_deepens_a_line_and_leaves_the_margin_alone(self) -> None:
        image, mask = self._line_on_field()
        sharpened = np.asarray(sharpen_image(image, mask, 0.6))
        before = np.asarray(image)
        # The line's core gets darker, the field beside it a touch lighter:
        # more contrast across the edge than the downscale left.
        self.assertLess(int(sharpened[10, 12, 0]), int(before[10, 12, 0]))
        self.assertGreaterEqual(int(sharpened[7, 12, 0]), int(before[7, 12, 0]))
        # Flat field far from any edge is untouched.
        self.assertTrue((sharpened[2, 10] == before[2, 10]).all())
        # Unpainted cells are neither changed nor allowed to darken the
        # painted cells beside them: the last painted column of flat field
        # stays the field color.
        self.assertTrue((sharpened[:, 26:] == 0).all())
        self.assertTrue((sharpened[2, 25] == before[2, 25]).all())

    def test_zero_amount_is_the_identity(self) -> None:
        image, mask = self._line_on_field()
        self.assertIs(sharpen_image(image, mask, 0.0), image)

    def test_reduction_is_judged_on_what_is_resampled(self) -> None:
        self.assertTrue(is_reduction((640, 480), (320, 240), "stretch"))
        self.assertFalse(is_reduction((100, 50), (320, 240), "fit"))
        # Fill crops a 300x300 region and enlarges it: nothing was lost.
        self.assertFalse(is_reduction((300, 600), (320, 320), "fill"))
        self.assertTrue(is_reduction((640, 480), (320, 320), "fill"))

    def test_process_image_sharpens_reductions_only(self) -> None:
        from PIL import Image

        rng = np.random.default_rng(3)
        large = Image.fromarray(
            np.dstack(
                [
                    rng.integers(0, 255, (96, 128, 3), dtype=np.uint8),
                    np.full((96, 128, 1), 255, dtype=np.uint8),
                ]
            ),
            "RGBA",
        )
        plain = process_image(
            large, logical_width=32, logical_height=24, color_count=256,
            sharpen=SharpenMode.OFF,
        )
        light = process_image(
            large, logical_width=32, logical_height=24, color_count=256,
            sharpen=SharpenMode.LIGHT,
        )
        self.assertFalse(
            np.array_equal(np.asarray(plain.image), np.asarray(light.image))
        )
        tiny = large.resize((8, 6))
        plain_up = process_image(
            tiny, logical_width=32, logical_height=24, color_count=256,
            sharpen=SharpenMode.OFF,
        )
        light_up = process_image(
            tiny, logical_width=32, logical_height=24, color_count=256,
            sharpen=SharpenMode.LIGHT,
        )
        self.assertTrue(
            np.array_equal(np.asarray(plain_up.image), np.asarray(light_up.image))
        )


if __name__ == "__main__":
    unittest.main()
