"""Standalone, in-memory image transforms for the single-image VQA stress test.

These mirror the `downscale_baseline` and `trim_downscale` pruning modes in
`rag/pruner.py`, but operate on a PIL image directly (no RAG q-dict, no disk
artifact) so they can be applied to arbitrary VQA datasets. The trim geometry is
reused verbatim from the pruner module to keep the two paths consistent.
"""
from __future__ import annotations

import math

from PIL import Image

from rag.pruner import _area_budget_factor, _trim_bbox


def downscale_to_keep(image: Image.Image, keep_ratio: float) -> Image.Image:
    """Uniform downscale so visual tokens land at ~keep_ratio of the full image.

    Visual tokens scale with pixel area, so a linear edge factor of
    sqrt(keep_ratio) targets keep_ratio of the tokens. keep_ratio >= 1 is a
    passthrough (never upscale)."""
    if keep_ratio >= 1.0:
        return image
    factor = math.sqrt(max(1e-6, keep_ratio))
    return image.resize(
        (max(1, int(image.width * factor)), max(1, int(image.height * factor)))
    )


def trim_downscale(image: Image.Image, keep_ratio: float) -> Image.Image:
    """Trim near-uniform borders (content-preserving), then downscale the trimmed
    image to the keep_ratio token budget. Removing blank margins first lets the
    budget land on actual content at higher effective resolution."""
    full_area = image.width * image.height
    box = _trim_bbox(image)
    trimmed = image
    if box is not None:
        cand = image.crop(box)
        # Only accept a trim that removes margin but not (almost) everything.
        if 0.1 * full_area < cand.width * cand.height < 0.995 * full_area:
            trimmed = cand
    factor = _area_budget_factor(trimmed.width * trimmed.height, full_area, keep_ratio)
    if factor < 1.0:
        trimmed = trimmed.resize(
            (max(1, int(trimmed.width * factor)), max(1, int(trimmed.height * factor)))
        )
    return trimmed


def apply_transform(image: Image.Image, mode: str, keep_ratio: float) -> Image.Image:
    if mode == "downscale":
        return downscale_to_keep(image, keep_ratio)
    if mode == "trim_downscale":
        return trim_downscale(image, keep_ratio)
    raise ValueError(f"unknown transform mode: {mode}")
