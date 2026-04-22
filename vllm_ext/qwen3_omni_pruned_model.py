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


def _prune_single_image_features(
    feats: torch.Tensor,           # (T, D)
    keep_idx: torch.Tensor,        # (K,)
) -> torch.Tensor:
    keep_idx = keep_idx.to(device=feats.device, dtype=torch.long)
    return feats.index_select(0, keep_idx)


def _prune_embeddings_list(
    embeddings: list[torch.Tensor] | tuple[torch.Tensor, ...],
    image_items: list[dict[str, Any] | None],
) -> list[torch.Tensor]:
    pruned: list[torch.Tensor] = []
    for feats, item_kwargs in zip(embeddings, image_items):
        if not item_kwargs:
            pruned.append(feats)
            continue
        keep_idx = item_kwargs.get("image_keep_indices", None)
        if keep_idx is None:
            pruned.append(feats)
        else:
            pruned.append(_prune_single_image_features(feats, keep_idx))
    return pruned


def _coerce_keep_indices_list(raw_keep_indices: Any) -> list[torch.Tensor]:
    if raw_keep_indices is None:
        return []
    if isinstance(raw_keep_indices, torch.Tensor):
        if raw_keep_indices.ndim == 1:
            return [raw_keep_indices]
        return [row for row in raw_keep_indices]
    if isinstance(raw_keep_indices, (list, tuple)):
        out: list[torch.Tensor] = []
        for item in raw_keep_indices:
            if item is None:
                out.append(torch.empty(0, dtype=torch.long))
            elif isinstance(item, torch.Tensor):
                out.append(item)
            else:
                out.append(torch.as_tensor(item, dtype=torch.long))
        return out
    return []


class Qwen3OmniPrunedForConditionalGeneration(
    Qwen3OmniForConditionalGeneration,
    SupportsMultiModal,
):
    """
    Keep Qwen3-Omni thinker unchanged.
    Only prune image token embeddings in the multimodal embedding path.
    """

    def get_multimodal_embeddings(self, **mm_kwargs: Any):
        """
        Let the parent produce normal multimodal embeddings, then prune image embeddings.
        """
        parent_fn = getattr(super(), "get_multimodal_embeddings", None)
        if parent_fn is None:
            return None
        mm_embeds = parent_fn(**mm_kwargs)

        if (
            isinstance(mm_embeds, dict)
            and "image" in mm_embeds
            and "image" in mm_kwargs
        ):
            mm_embeds["image"] = _prune_embeddings_list(
                mm_embeds["image"],
                list(mm_kwargs["image"]),
            )
            return mm_embeds

        return mm_embeds

    def embed_multimodal(self, **kwargs: object):
        """
        Upstream vLLM path: prune image embeddings after parent vision encoding.
        """
        mm_embeds = super().embed_multimodal(**kwargs)
        if not mm_embeds:
            return mm_embeds

        embeds_list = list(mm_embeds)
        image_items = kwargs.get("image")
        if image_items:
            pruned = _prune_embeddings_list(embeds_list, list(image_items))
            return tuple(pruned)

        keep_indices_list = _coerce_keep_indices_list(kwargs.get("image_keep_indices"))
        if not keep_indices_list:
            return mm_embeds

        pruned: list[torch.Tensor] = []
        for idx, feats in enumerate(embeds_list):
            if idx >= len(keep_indices_list):
                pruned.append(feats)
                continue
            keep_idx = keep_indices_list[idx]
            if keep_idx.numel() == 0:
                pruned.append(feats)
                continue
            pruned.append(_prune_single_image_features(feats, keep_idx))
        return tuple(pruned)