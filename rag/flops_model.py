"""Transparent forward-pass FLOPs model for Qwen2-VL-7B, used to re-score the
FastV-vs-input-downscale comparison on an honest *total compute* axis (Angle 1).

Why this matters: the Phase-3/4 comparison matched "visual tokens" on the *deep-layer*
count (post-prune). But FastV keeps the FULL image through the vision encoder and the
first K LLM layers, pruning only afterwards — so its token saving is deep-LLM-only.
Input-side downscaling feeds fewer tokens to *everything* (encoder + all layers). This
model counts both the vision-encoder cost (which FastV pays in full) and the per-layer
LLM cost with the FastV early/deep split, so the two methods can be compared fairly.

Multiply-accumulate counted as 2 FLOPs. Architecture constants are Qwen2-VL-7B's actual
config (hidden 3584, 28 layers, GQA 28/4 heads, head_dim 128, d_ff 18944; ViT depth 32,
dim 1280, 16 heads, mlp_ratio 4, 2x2 spatial merge, full attention). The vision tower
uses *full* attention over all 14px patches (Qwen2-VL, pre-2.5), so its attention term
is O(V^2) in the number of patches.
"""
from __future__ import annotations

from typing import Optional, Tuple

LLM = dict(d=3584, n_layers=28, n_h=28, n_kv=4, hd=128, d_ff=18944)
VIT = dict(depth=32, d=1280, n_h=16, hd=80, d_ff=5120, merge=2)
TEXT_TOKENS = 56  # median V*Bench MC prompt length (no image placeholders)


def llm_layer_flops(L: int, c: dict = LLM) -> float:
    """FLOPs for one decoder layer on a length-L prefill (no KV cache)."""
    d, n_h, n_kv, hd, d_ff = c["d"], c["n_h"], c["n_kv"], c["hd"], c["d_ff"]
    proj = 2 * L * d * d + 2 * (2 * L * d * n_kv * hd) + 2 * L * d * d  # Q + (K,V) + O
    attn = 2 * (2 * n_h * L * L * hd)                                   # QK^T + AV
    mlp = 3 * (2 * L * d * d_ff)                                        # gate, up, down
    return float(proj + attn + mlp)


def vit_flops(n_img_tokens: int, c: dict = VIT) -> float:
    """FLOPs for the vision tower processing an image that yields `n_img_tokens` merged
    LLM tokens (i.e. n_img_tokens * merge^2 raw patches, full attention)."""
    V = n_img_tokens * c["merge"] ** 2
    d, n_h, hd, d_ff, depth = c["d"], c["n_h"], c["hd"], c["d_ff"], c["depth"]
    proj = 4 * (2 * V * d * d)            # Q,K,V,O
    attn = 2 * (2 * n_h * V * V * hd)     # QK^T + AV
    mlp = 2 * (2 * V * d * d_ff)          # up, down
    return float(depth * (proj + attn + mlp))


def condition_flops(
    img_before: int,
    img_after: int,
    prune_layer: Optional[int],
    text_tokens: int = TEXT_TOKENS,
    n_layers: int = LLM["n_layers"],
) -> Tuple[float, float, float]:
    """Total forward FLOPs for one condition. Returns (total, vision, llm).

    `prune_layer=None` means no in-model pruning (img_after is ignored / equals
    img_before). FastV runs layers [0, prune_layer) on the full sequence (text +
    img_before), then layers [prune_layer, n_layers) on the pruned sequence
    (text + img_after). The vision tower always processes the full input image."""
    vision = vit_flops(img_before)
    full_L = text_tokens + img_before
    if prune_layer is None:
        llm = n_layers * llm_layer_flops(full_L)
    else:
        pruned_L = text_tokens + img_after
        llm = (prune_layer * llm_layer_flops(full_L)
               + (n_layers - prune_layer) * llm_layer_flops(pruned_L))
    return vision + llm, vision, llm
