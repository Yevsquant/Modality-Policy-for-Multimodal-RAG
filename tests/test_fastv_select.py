import torch

from rag.fastv import select_keep_indices


def test_keeps_all_text_and_top_image():
    image_positions = torch.tensor([3, 4, 5, 6])
    attn = torch.tensor([0.1, 0.9, 0.2, 0.8])  # top-2 are positions 4 and 6
    keep = select_keep_indices(attn, image_positions, seq_len=10, keep_ratio=0.5)
    assert keep.tolist() == [0, 1, 2, 4, 6, 7, 8, 9]


def test_keep_ratio_one_keeps_everything():
    image_positions = torch.tensor([2, 3])
    attn = torch.tensor([0.5, 0.5])
    keep = select_keep_indices(attn, image_positions, seq_len=5, keep_ratio=1.0)
    assert keep.tolist() == [0, 1, 2, 3, 4]


def test_sorted_and_at_least_one_image_kept():
    image_positions = torch.tensor([1, 2, 3, 4, 5, 6])
    attn = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6])
    keep = select_keep_indices(attn, image_positions, seq_len=8, keep_ratio=0.01)
    assert keep.tolist() == sorted(keep.tolist())
    # at least one image token survives, and it is the highest-attention one (pos 6)
    assert 6 in keep.tolist()
    kept_imgs = [i for i in keep.tolist() if i in image_positions.tolist()]
    assert len(kept_imgs) == 1
