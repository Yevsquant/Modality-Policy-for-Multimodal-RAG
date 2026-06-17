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

    assert "clip_safecrop" in RetrievalPruner.SUPPORTED_MODES
    assert "downscale_baseline" in RetrievalPruner.SUPPORTED_MODES


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
