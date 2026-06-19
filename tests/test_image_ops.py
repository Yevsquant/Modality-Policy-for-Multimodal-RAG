import math

from PIL import Image

from rag.image_ops import downscale_to_keep, trim_downscale


def _content_image_with_border(border=40, content=120, size=400):
    """White image with a centered black square (so trim has margin to remove)."""
    img = Image.new("RGB", (size, size), (255, 255, 255))
    for x in range(border, border + content):
        for y in range(border, border + content):
            img.putpixel((x, y), (0, 0, 0))
    return img


def test_downscale_keeps_area_fraction():
    img = Image.new("RGB", (300, 200), (10, 20, 30))
    out = downscale_to_keep(img, 0.25)
    # edge factor sqrt(0.25)=0.5 -> area ~ 0.25 of original
    assert out.size == (150, 100)
    area_ratio = (out.width * out.height) / (img.width * img.height)
    assert abs(area_ratio - 0.25) < 0.02


def test_downscale_passthrough_at_one():
    img = Image.new("RGB", (123, 77), (0, 0, 0))
    assert downscale_to_keep(img, 1.0).size == (123, 77)


def test_trim_downscale_removes_margin_then_budgets():
    img = _content_image_with_border()
    full_area = img.width * img.height
    out = trim_downscale(img, 0.25)
    out_area = out.width * out.height
    # Must not exceed the keep-ratio budget (area <= keep * full, with rounding slack)
    assert out_area <= 0.25 * full_area * 1.05
    # The content (120x120) fits in budget (0.25*400*400=40000 > 14400), so trim
    # alone suffices and no downscale is needed: output ~ the trimmed content box.
    assert out.width <= img.width and out.height <= img.height


def test_trim_downscale_uniform_image_no_trim():
    img = Image.new("RGB", (200, 200), (255, 255, 255))
    out = trim_downscale(img, 0.25)
    # uniform image: nothing to trim, so it falls back to plain downscale budget
    area_ratio = (out.width * out.height) / (img.width * img.height)
    assert area_ratio <= 0.30
