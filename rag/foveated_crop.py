"""Angle 2 — query-conditioned coarse-to-fine foveated CROP.

FastV beats *uniform input downscaling* at a matched visual-token budget on V*Bench
because it is spatially selective at full resolution (Phase 3/4:
[[fastv-composition-negative-interaction]], [[vstar-is-the-stress-substrate-trim-doesnt-transfer]]).
This module tests whether an **input-side** crop can do the same thing *before* the
model: localize the query-relevant region, crop it from the full-resolution image, and
(if it still exceeds the budget) downscale only the crop. The periphery is discarded at
the input, so the relevant region keeps the highest resolution the budget allows
(foveation), instead of blurring everything uniformly.

Two localizers (cheap -> stronger):
  - `clip_localize`: per-patch CLIP cosine to the question text (reuse Phase-4
    `SpatialClipFeaturizer._patch_relevance`) -> relevance heatmap. Nearly free, but
    CLIP is a weak localizer for small-object visual search.
  - `c2f_localize` (coarse-to-fine): one cheap LOW-RESOLUTION pass through the 7B,
    read the last-query-token attention to image tokens at a shallow layer (reuse the
    same signal FastV uses), map the high-attention tokens back to image coordinates
    via `image_grid_thw`, take their bbox -> crop the FULL-res image. Mirrors how FastV
    gets its signal, but at a fraction of the cost (low-res locate -> full-res crop).

Geometry (`heatmap_to_bbox`, `crop_box_to_budget`) is pure/numpy so it is unit-tested
without a GPU. The crop is scored by feeding the cropped PIL to
`rag.fastv.FastVQwen2VL.answer_mc(image, q, input_keep=1.0, fastv_layer=None)`; the crop
itself reduces the token count, so no in-model pruning is applied.
"""
from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Pure geometry (unit-tested without a model).
# ---------------------------------------------------------------------------

def heatmap_to_bbox(
    rel: np.ndarray,
    grid_hw: Tuple[int, int],
    percentile: float = 80.0,
    margin: float = 0.10,
) -> Tuple[float, float, float, float]:
    """Bounding box (in normalized [0,1] coords) of the high-relevance region.

    `rel` is a flat per-patch relevance map laid out row-major on `grid_hw=(rows,cols)`.
    Patches with relevance above the `percentile`-th percentile are "active"; the bbox
    is the tight box around them, expanded by `margin` (fraction of width/height) and
    clamped to [0,1]. Always returns a non-degenerate box (falls back to the single
    peak patch, then to the full image) so the crop is well-defined.

    Returns (x0, y0, x1, y1) as fractions of image width/height.
    """
    rows, cols = grid_hw
    grid = np.asarray(rel, dtype=np.float64).reshape(rows, cols)

    thresh = np.percentile(grid, percentile)
    active = grid > thresh
    if not active.any():  # degenerate (e.g. constant map) -> use the peak patch
        pr, pc = np.unravel_index(int(np.argmax(grid)), grid.shape)
        active = np.zeros_like(grid, dtype=bool)
        active[pr, pc] = True

    ys, xs = np.where(active)
    # Patch (r,c) spans [c/cols, (c+1)/cols] x [r/rows, (r+1)/rows] in normalized coords.
    x0 = xs.min() / cols
    x1 = (xs.max() + 1) / cols
    y0 = ys.min() / rows
    y1 = (ys.max() + 1) / rows

    # Expand by margin (relative to the current box size) and clamp.
    bw, bh = x1 - x0, y1 - y0
    x0 = max(0.0, x0 - margin * bw)
    x1 = min(1.0, x1 + margin * bw)
    y0 = max(0.0, y0 - margin * bh)
    y1 = min(1.0, y1 + margin * bh)
    return (float(x0), float(y0), float(x1), float(y1))


def crop_box_to_budget(
    full_w: int,
    full_h: int,
    bbox: Tuple[float, float, float, float],
    budget_tokens: int,
    patch_px: int = 28,
) -> Tuple[Tuple[int, int, int, int], Tuple[int, int]]:
    """Pixel crop box + target output size for a token budget.

    `bbox` is normalized (x0,y0,x1,y1). The crop is taken from the full-res image; if it
    still exceeds `budget_tokens` (one token per `patch_px**2`-pixel cell, matching
    Qwen2-VL where tokens ~= pixels/28^2), it is downscaled to land on the budget. The
    output size is never upscaled beyond the crop's own resolution.

    Returns ((x0,y0,x1,y1) pixel crop box, (out_w,out_h) resize target).
    """
    px0 = int(np.floor(bbox[0] * full_w))
    py0 = int(np.floor(bbox[1] * full_h))
    px1 = int(np.ceil(bbox[2] * full_w))
    py1 = int(np.ceil(bbox[3] * full_h))
    px0 = max(0, min(px0, full_w - 1))
    py0 = max(0, min(py0, full_h - 1))
    px1 = max(px0 + 1, min(px1, full_w))
    py1 = max(py0 + 1, min(py1, full_h))

    crop_w, crop_h = px1 - px0, py1 - py0
    budget_px = budget_tokens * patch_px * patch_px
    crop_area = crop_w * crop_h
    if crop_area > budget_px:
        f = (budget_px / crop_area) ** 0.5
        out_w = max(1, int(crop_w * f))
        out_h = max(1, int(crop_h * f))
    else:
        out_w, out_h = crop_w, crop_h
    return (px0, py0, px1, py1), (out_w, out_h)


def make_foveated_image(
    image: Image.Image,
    rel: np.ndarray,
    grid_hw: Tuple[int, int],
    budget_tokens: int,
    percentile: float = 80.0,
    margin: float = 0.10,
) -> Image.Image:
    """Crop the high-relevance region from `image` and downscale it to the budget."""
    bbox = heatmap_to_bbox(rel, grid_hw, percentile=percentile, margin=margin)
    (px0, py0, px1, py1), (out_w, out_h) = crop_box_to_budget(
        image.width, image.height, bbox, budget_tokens
    )
    crop = image.crop((px0, py0, px1, py1))
    if (out_w, out_h) != (crop.width, crop.height):
        crop = crop.resize((out_w, out_h))
    return crop


# ---------------------------------------------------------------------------
# Localizers.
# ---------------------------------------------------------------------------

class ClipLocalizer:
    """CLIP patch-relevance localizer (cheap). Wraps the Phase-4 spatial featurizer."""

    def __init__(self, device: Optional[str] = None):
        from rag.budget_features import SpatialClipFeaturizer

        self.feat = SpatialClipFeaturizer(device=device)
        self.grid_hw = self.feat.GRID  # (7, 7)

    def relevance(self, image: Image.Image, question: str) -> np.ndarray:
        """Flat per-patch relevance map (one CLIP forward)."""
        return self.feat._patch_relevance(image, question)


@torch.no_grad()
def c2f_relevance(
    fv,
    image: Image.Image,
    question: str,
    layer: int = 3,
    coarse_pixels: int = 256 * 28 * 28,
) -> Tuple[np.ndarray, Tuple[int, int]]:
    """Coarse-to-fine relevance map from a cheap LOW-RES 7B attention pass.

    `fv` is a `rag.fastv.FastVQwen2VL`. The image is fit to ~`coarse_pixels` (default
    ~256 tokens) and run through the model with eager attention; the last query token's
    attention to each image token at `layer` is read and reshaped onto the merged image
    grid `(grid_h//merge, grid_w//merge)`. Returns (flat relevance map, grid_hw).
    """
    from rag.fastv import _fit_area

    small = _fit_area(image, coarse_pixels)
    messages = [{"role": "user", "content": [
        {"type": "image", "image": small}, {"type": "text", "text": question}]}]
    text = fv.processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True)
    inputs = fv.processor(text=[text], images=[small], return_tensors="pt").to(fv.device)

    input_ids = inputs["input_ids"][0]
    image_positions = (input_ids == fv.image_token_id).nonzero(as_tuple=True)[0]

    merge = int(fv.processor.image_processor.merge_size)
    grid_t, grid_h, grid_w = inputs["image_grid_thw"][0].tolist()
    rows, cols = grid_h // merge, grid_w // merge

    fv._set_attn("eager")
    out = fv.model(**inputs, output_attentions=True, use_cache=False)
    attn = out.attentions[layer - 1][0]          # [heads, L, L]
    recv = attn.mean(0)[-1]                       # attn from last token, [L]
    rel = recv[image_positions].float().cpu().numpy()  # [rows*cols] (t=1)
    fv._set_attn("sdpa")
    return rel, (rows, cols)
