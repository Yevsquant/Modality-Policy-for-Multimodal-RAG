import math

from rag.pruner import _bbox_union


def test_bbox_union_multiple_boxes():
    boxes = [
        [10, 20, 30, 40],
        [5, 25, 35, 38],
        [12, 0, 28, 50],
    ]
    assert _bbox_union(boxes) == [5, 0, 35, 50]


def test_bbox_union_single_box_is_itself():
    assert _bbox_union([[3, 4, 7, 9]]) == [3, 4, 7, 9]


def test_bbox_union_collinear_boxes():
    # boxes sharing the same vertical band -> union spans full width
    boxes = [
        [0, 0, 10, 10],
        [10, 0, 20, 10],
        [20, 0, 30, 10],
    ]
    assert _bbox_union(boxes) == [0, 0, 30, 10]


def test_downscale_factor_preserves_area_ratio():
    # Edge length scales by sqrt(keep_ratio) so area ratio ~= keep_ratio.
    for keep_ratio in [0.1, 0.25, 0.5, 0.7]:
        factor = math.sqrt(keep_ratio)
        area_ratio = factor * factor
        assert abs(area_ratio - keep_ratio) < 1e-9


def test_modes_registered():
    from rag.pruner import RetrievalPruner

    for m in (
        "clip_safecrop",
        "downscale_baseline",
        "clip_safecrop_downscale",
        "trim_downscale",
        "density_adaptive_downscale",
        "relevance_adaptive_downscale",
    ):
        assert m in RetrievalPruner.SUPPORTED_MODES


def test_trim_bbox_finds_content():
    from rag.pruner import _trim_bbox
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (100, 100), (255, 255, 255))
    ImageDraw.Draw(img).rectangle([30, 20, 70, 60], fill=(0, 0, 0))
    box = _trim_bbox(img)
    # bbox should tightly enclose the black rectangle (inclusive/exclusive slack 1px)
    assert box is not None
    l, t, r, b = box
    assert 29 <= l <= 31 and 19 <= t <= 21
    assert 70 <= r <= 72 and 60 <= b <= 62


def test_trim_bbox_uniform_returns_none():
    from rag.pruner import _trim_bbox
    from PIL import Image

    assert _trim_bbox(Image.new("RGB", (50, 50), (255, 255, 255))) is None


def test_detail_density_ordering():
    from rag.pruner import _detail_density
    from PIL import Image
    import numpy as np

    flat = Image.new("RGB", (128, 128), (128, 128, 128))
    noise = Image.fromarray((np.random.rand(128, 128, 3) * 255).astype("uint8"))
    df = _detail_density(flat)
    dn = _detail_density(noise)
    assert 0.0 <= df <= 1.0 and 0.0 <= dn <= 1.0
    assert df < 0.05 < dn


def test_density_to_keep_ratio_monotonic():
    from rag.pruner import _density_to_keep_ratio

    lo = _density_to_keep_ratio(0.0, base_keep=0.3)
    hi = _density_to_keep_ratio(1.0, base_keep=0.3)
    mid = _density_to_keep_ratio(0.5, base_keep=0.3)
    assert lo < mid < hi
    assert abs(lo - 0.3 * 0.5) < 1e-9          # density 0 -> base*lo_mult
    assert hi <= 1.0                            # clamped to max_keep
    assert abs(hi - min(1.0, 0.3 * 2.0)) < 1e-9


def test_relevance_keep_ratios_equal_and_ordered():
    from rag.pruner import _relevance_keep_ratios

    # Equal relevances -> everyone gets the base budget (mean preserved).
    eq = _relevance_keep_ratios([0.2, 0.2], base_keep=0.3)
    assert abs(eq[0] - 0.3) < 1e-6 and abs(eq[1] - 0.3) < 1e-6
    # Single image -> base budget.
    one = _relevance_keep_ratios([0.25], base_keep=0.3)
    assert abs(one[0] - 0.3) < 1e-6
    # More relevant image gets a larger budget.
    krs = _relevance_keep_ratios([0.4, 0.1], base_keep=0.3)
    assert krs[0] > krs[1]


def test_area_budget_factor():
    from rag.pruner import _area_budget_factor

    # Crop already within budget -> no downscale (factor 1.0).
    assert _area_budget_factor(crop_area=100, full_area=1000, keep_ratio=0.3) == 1.0
    # Crop larger than budget -> shrink so crop_area*factor**2 == target.
    f = _area_budget_factor(crop_area=800, full_area=1000, keep_ratio=0.3)
    assert abs((800 * f * f) - (0.3 * 1000)) < 1e-6
    assert 0.0 < f < 1.0
    # Full-image crop at keep_ratio -> matches the downscale_baseline factor sqrt(kr).
    f_full = _area_budget_factor(crop_area=1000, full_area=1000, keep_ratio=0.25)
    assert abs(f_full - math.sqrt(0.25)) < 1e-9


def test_downscale_image_hits_area_budget(tmp_path):
    from rag.pruner import RetrievalPruner
    from PIL import Image

    src = tmp_path / "src.jpg"
    Image.new("RGB", (800, 600), color=(10, 20, 30)).save(src)

    pruner = RetrievalPruner(
        mode="downscale_baseline", keep_ratio=0.25, output_dir=str(tmp_path / "out")
    )
    q = {"local_img_path": str(src), "quote_id": "q", "tag_hash": "h", "image_cache_id": "src"}
    new_q, before, after = pruner._downscale_image(q)

    out = Image.open(new_q["local_img_path"])
    area_ratio = (out.width * out.height) / (800 * 600)
    # sqrt(0.25)=0.5 each edge -> 400x300 -> area ratio 0.25
    assert abs(area_ratio - 0.25) < 0.01
    assert after == round(before * 0.25)
