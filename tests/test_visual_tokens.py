from rag.metrics import aggregate_summary


def _row(before, after):
    return {
        "metrics": {},
        "timing": {"retrieval_sec": 0, "ttft_sec": 0.1, "generation_sec": 0, "total_sec": 0},
        "visual_tokens": {"before": before, "after": after},
    }


def test_aggregate_summary_token_reduction():
    s = aggregate_summary([_row(100, 40)])
    assert abs(s["avg_visual_tokens_before"] - 100) < 1e-9
    assert abs(s["avg_visual_tokens_after"] - 40) < 1e-9
    assert abs(s["avg_visual_tokens_reduction_pct"] - 60) < 1e-9


def test_aggregate_summary_token_reduction_multi_row():
    s = aggregate_summary([_row(100, 40), _row(200, 100)])
    # before avg 150, after avg 70 -> reduction 1 - 140/300 = 53.333...%
    assert abs(s["avg_visual_tokens_before"] - 150) < 1e-9
    assert abs(s["avg_visual_tokens_after"] - 70) < 1e-9
    assert abs(s["avg_visual_tokens_reduction_pct"] - (100 * (1 - 140 / 300))) < 1e-9


def test_aggregate_summary_no_token_fields():
    rows = [{
        "metrics": {},
        "timing": {"retrieval_sec": 0, "ttft_sec": 0.1, "generation_sec": 0, "total_sec": 0},
    }]
    s = aggregate_summary(rows)
    assert "avg_visual_tokens_reduction_pct" not in s


def test_visual_token_counter_count_math():
    # Exercise count() math with a fake image_processor (no model download).
    from rag.visual_token_counter import VisualTokenCounter
    from PIL import Image

    class _FakeGridThw:
        def __init__(self, vals):
            self._vals = vals

        def tolist(self):
            return self._vals

    class _FakeImageProcessor:
        merge_size = 2

        def __call__(self, images, return_tensors=None):
            return {"image_grid_thw": [_FakeGridThw([1, 16, 16])]}

    counter = VisualTokenCounter.__new__(VisualTokenCounter)
    counter.image_processor = _FakeImageProcessor()
    counter.merge_size = 2

    img = Image.new("RGB", (32, 32))
    # 1 * 16 * 16 // (2**2) = 64
    assert counter.count(img) == 64
