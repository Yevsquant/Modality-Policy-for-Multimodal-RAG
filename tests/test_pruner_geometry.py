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
