from app.image_readiness import assess_image_readiness
from app.models import ScaleMode


def test_fit_only_measures_the_part_of_target_the_image_paints() -> None:
    readiness = assess_image_readiness((200, 100), (100, 100), ScaleMode.FIT)

    assert readiness.painted_size == (100, 50)
    assert readiness.enlargement == 0.5
    assert not readiness.needs_warning


def test_fill_accounts_for_source_pixels_lost_to_the_crop() -> None:
    readiness = assess_image_readiness((400, 100), (200, 200), ScaleMode.FILL)

    assert readiness.used_source_size == (100, 100)
    assert readiness.enlargement == 2.0
    assert readiness.recommended_size == (100, 100)
    assert readiness.needs_warning


def test_small_source_recommends_a_target_that_does_not_invent_detail() -> None:
    readiness = assess_image_readiness((160, 90), (640, 360), ScaleMode.STRETCH)

    assert readiness.enlargement == 4.0
    assert readiness.recommended_size == (160, 90)
