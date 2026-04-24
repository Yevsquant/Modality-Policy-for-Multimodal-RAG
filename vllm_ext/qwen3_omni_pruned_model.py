from __future__ import annotations

from typing import Any

import torch

from vllm.model_executor.models.interfaces import SupportsMultiModal

try:
    from vllm_omni.model_executor.models.qwen3_omni import (
        Qwen3OmniForConditionalGeneration,
    )
except ImportError:
    from vllm.model_executor.models.qwen3_omni_moe_thinker import (
        Qwen3OmniMoeThinkerForConditionalGeneration as Qwen3OmniForConditionalGeneration,
    )

from .visual_pruning_policy import VisualTokenPruningPolicy
from .visual_pruning_config import VisualPruningConfig


def _coerce_image_items(raw_image_items: Any) -> list[dict[str, Any] | None]:
    if raw_image_items is None:
        return []
    if isinstance(raw_image_items, list):
        return list(raw_image_items)
    if isinstance(raw_image_items, tuple):
        return list(raw_image_items)
    return []


def _maybe_get_policy(mm_kwargs: dict[str, Any]) -> VisualTokenPruningPolicy | None:
    # Fallback to direct config instantiation if engine stripped the kwargs
    config = VisualPruningConfig()
    
    is_enabled = mm_kwargs.get("visual_pruning_enabled", config.enabled)
    if not is_enabled:
        return None
    return VisualTokenPruningPolicy(
        keep_ratio=float(mm_kwargs.get("visual_pruning_keep_ratio", config.keep_ratio)),
        min_keep=int(mm_kwargs.get("visual_pruning_min_keep", config.min_keep)),
        policy=str(mm_kwargs.get("visual_pruning_policy", config.policy)),
        keep_cls_token=bool(mm_kwargs.get("visual_pruning_keep_cls_token", config.keep_cls_token)),
        normalize_scores=bool(mm_kwargs.get("visual_pruning_normalize_scores", config.normalize_scores)),
    )


def _apply_embedding_pruning(
    embeddings: list[torch.Tensor] | tuple[torch.Tensor, ...],
    image_items: list[dict[str, Any] | None],
    mm_kwargs: dict[str, Any],
) -> tuple[list[torch.Tensor], list[dict[str, Any] | None]]:
    policy = _maybe_get_policy(mm_kwargs)
    if policy is None:
        return list(embeddings), image_items

    hard_prune = bool(mm_kwargs.get("visual_pruning_hard_prune", True))
    fallback_to_soft_mask = bool(mm_kwargs.get("visual_pruning_fallback_to_soft_mask", False))

    pruned: list[torch.Tensor] = []
    updated_items: list[dict[str, Any] | None] = []

    for feats, item_kwargs in zip(embeddings, image_items):
        if item_kwargs is None:
            pruned.append(feats)
            updated_items.append(item_kwargs)
            continue

        item_kwargs = dict(item_kwargs)
        keep_spec = policy.build_keep_spec_from_embeddings(feats)
        item_kwargs["image_keep_indices"] = keep_spec.keep_indices.detach().cpu()
        item_kwargs["image_tokens_before"] = keep_spec.tokens_before
        item_kwargs["image_tokens_after"] = keep_spec.tokens_after
        item_kwargs["image_token_scores"] = keep_spec.scores.detach().cpu()
        item_kwargs["visual_pruning_deferred_to_model"] = True

        if hard_prune:
            pruned_feats = policy.prune_embeddings(feats, keep_spec.keep_indices)
        elif fallback_to_soft_mask:
            pruned_feats = policy.soft_mask_embeddings(feats, keep_spec.keep_indices)
            item_kwargs["image_tokens_after"] = keep_spec.tokens_before
        else:
            # Default to hard prune if no fallback behavior is requested.
            pruned_feats = policy.prune_embeddings(feats, keep_spec.keep_indices)

        pruned.append(pruned_feats)
        updated_items.append(item_kwargs)

    return pruned, updated_items


class Qwen3OmniPrunedForConditionalGeneration(
    Qwen3OmniForConditionalGeneration,
    SupportsMultiModal,
):
    """
    Keep Qwen3-Omni thinker unchanged.
    Compute pruning decisions from image embedding values in the multimodal
    embedding path, instead of from token positions/counts in the processor.
    """

    def get_multimodal_embeddings(self, **mm_kwargs: Any):
        parent_fn = getattr(super(), "get_multimodal_embeddings", None)
        if parent_fn is None:
            return None
        mm_embeds = parent_fn(**mm_kwargs)

        image_items = _coerce_image_items(mm_kwargs.get("image"))
        if isinstance(mm_embeds, dict) and "image" in mm_embeds:
            # Auto-generate tracking dicts if 'image' was stripped/missing
            if not image_items:
                 image_items = [{} for _ in range(len(mm_embeds["image"]))]

            pruned, updated_items = _apply_embedding_pruning(
                mm_embeds["image"],
                image_items,
                mm_kwargs,
            )
            mm_embeds["image"] = pruned
            mm_kwargs["image"] = updated_items
            return mm_embeds

        return mm_embeds

    def embed_multimodal(self, **kwargs: object):
        mm_embeds = super().embed_multimodal(**kwargs)
        if not mm_embeds:
            return mm_embeds

        image_items = _coerce_image_items(kwargs.get("image"))
        # Auto-generate tracking dicts if 'image' was stripped/missing
        if not image_items:
            image_items = [{} for _ in range(len(mm_embeds))]

        pruned, updated_items = _apply_embedding_pruning(
            list(mm_embeds),
            image_items,
            dict(kwargs),
        )
        kwargs["image"] = updated_items
        return tuple(pruned)
