import numpy as np

from rag.foveated_crop import crop_box_to_budget, heatmap_to_bbox, make_foveated_image


def test_bbox_localizes_single_hot_patch():
    # 4x4 grid, one hot patch at (row=1, col=2); percentile high -> only it is active.
    rel = np.zeros(16)
    rel[1 * 4 + 2] = 10.0
    x0, y0, x1, y1 = heatmap_to_bbox(rel, (4, 4), percentile=90.0, margin=0.0)
    # patch (1,2) spans x in [2/4, 3/4], y in [1/4, 2/4]
    assert abs(x0 - 0.5) < 1e-9 and abs(x1 - 0.75) < 1e-9
    assert abs(y0 - 0.25) < 1e-9 and abs(y1 - 0.5) < 1e-9


def test_bbox_margin_expands_and_clamps():
    rel = np.zeros(16)
    rel[1 * 4 + 1] = 5.0  # patch (1,1): x in [0.25,0.5], y in [0.25,0.5]
    x0, y0, x1, y1 = heatmap_to_bbox(rel, (4, 4), percentile=90.0, margin=0.5)
    # box width 0.25; margin 0.5*0.25=0.125 each side -> [0.125, 0.625]
    assert abs(x0 - 0.125) < 1e-9 and abs(x1 - 0.625) < 1e-9
    # all within [0,1]
    assert 0.0 <= x0 and x1 <= 1.0 and 0.0 <= y0 and y1 <= 1.0


def test_bbox_constant_map_falls_back_to_peak_not_degenerate():
    rel = np.full(9, 0.3)
    box = heatmap_to_bbox(rel, (3, 3), percentile=99.0, margin=0.0)
    x0, y0, x1, y1 = box
    assert x1 > x0 and y1 > y0  # non-degenerate


def test_crop_box_within_budget_no_downscale():
    # crop covers full 280x280 image (100 patches @28px) but budget is generous.
    box, (out_w, out_h) = crop_box_to_budget(
        280, 280, (0.0, 0.0, 1.0, 1.0), budget_tokens=200, patch_px=28)
    assert box == (0, 0, 280, 280)
    assert (out_w, out_h) == (280, 280)  # 100 tokens <= 200 budget -> no resize


def test_crop_box_over_budget_downscales_to_target():
    # full image crop, 280x280 = 100 tokens, budget 25 tokens -> area must shrink 4x.
    box, (out_w, out_h) = crop_box_to_budget(
        280, 280, (0.0, 0.0, 1.0, 1.0), budget_tokens=25, patch_px=28)
    tokens = (out_w * out_h) / (28 * 28)
    assert tokens <= 25 + 1e-6
    # edge factor sqrt(25/100)=0.5 -> ~140x140
    assert abs(out_w - 140) <= 1 and abs(out_h - 140) <= 1


def test_crop_box_clamps_to_image():
    box, _ = crop_box_to_budget(
        100, 100, (-0.5, -0.5, 2.0, 2.0), budget_tokens=1000, patch_px=28)
    assert box == (0, 0, 100, 100)


def test_make_foveated_image_crops_hot_region():
    from PIL import Image

    img = Image.new("RGB", (400, 400), (0, 0, 0))
    rel = np.zeros(16)
    rel[0] = 9.0  # top-left patch hot
    out = make_foveated_image(img, rel, (4, 4), budget_tokens=10000,
                              percentile=90.0, margin=0.0)
    # top-left patch (0,0) -> x,y in [0,0.25] -> 100x100 crop, under budget -> kept
    assert out.size == (100, 100)
