from __future__ import annotations

from typing import Any

import torch

from vllm.multimodal.processing.processor import BaseMultiModalProcessor
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.processing.inputs import ProcessorInputs

from .visual_pruning_config import VisualPruningConfig
from .visual_pruning_policy import VisualTokenPruningPolicy


def _count_image_tokens_from_mm_kwargs(mm_item_kwargs: dict[str, Any]) -> int:
    """
    Estimate per-image visual token count from processor outputs.

    Adapt this function to your exact Qwen3-Omni processor outputs.
    In Qwen-family VLMs this is often derivable from image_grid_thw /
    patch grid metadata emitted by the HF processor.
    """
    if "image_embeds" in mm_item_kwargs:
        return int(mm_item_kwargs["image_embeds"].shape[0])

    if "image_grid_thw" in mm_item_kwargs:
        grid = mm_item_kwargs["image_grid_thw"]
        # handle tensor/list shape like (num_images, 3) or (3,)
        grid = torch.as_tensor(grid)
        if grid.ndim == 2:
            # per-item caller should pass one image at a time here
            t, h, w = grid[0].tolist()
        else:
            t, h, w = grid.tolist()
        return int(t * h * w)

    raise ValueError(
        "Cannot infer image token count from mm kwargs. "
        "Inspect the Qwen3-Omni processor output and update _count_image_tokens_from_mm_kwargs."
    )


class Qwen3OmniPrunedProcessor(BaseMultiModalProcessor):
    """
    Custom multimodal processor:
    - runs the normal HF multimodal processor
    - computes placeholder ranges
    - rewrites image placeholder lengths to the pruned token count
    - stores image_keep_indices in mm_kwargs for the model embedding path
    """

    def __init__(self, *args, visual_pruning: VisualPruningConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.visual_pruning = visual_pruning or VisualPruningConfig()
        self._policy = VisualTokenPruningPolicy(
            keep_ratio=self.visual_pruning.keep_ratio,
            min_keep=self.visual_pruning.min_keep,
        )

    def apply(self, inputs: ProcessorInputs, timing_ctx):
        mm_input = super().apply(inputs, timing_ctx)

        if not self.visual_pruning.enabled:
            return mm_input
        if self.visual_pruning.policy != "uniform":
            raise ValueError(
                f"Unsupported visual pruning policy: {self.visual_pruning.policy}. "
                "Only 'uniform' is supported in this rollout."
            )

        mm_kwargs = dict(mm_input["mm_kwargs"])
        mm_placeholders = dict(mm_input["mm_placeholders"])

        if "image" not in mm_kwargs or "image" not in mm_placeholders:
            return mm_input

        image_items = list(mm_kwargs["image"])
        image_ranges = list(mm_placeholders["image"])

        new_image_items = []
        new_image_ranges = []

        for item_kwargs, old_range in zip(image_items, image_ranges):
            if item_kwargs is None:
                new_image_items.append(item_kwargs)
                new_image_ranges.append(old_range)
                continue

            item_kwargs = dict(item_kwargs)

            num_tokens = _count_image_tokens_from_mm_kwargs(item_kwargs)
            keep_spec = self._policy.build_uniform_keep_spec(
                num_tokens=num_tokens,
                device=torch.device("cpu"),
            )
            keep = keep_spec.keep_indices.cpu()

            item_kwargs["image_keep_indices"] = keep
            item_kwargs["image_tokens_before"] = keep_spec.tokens_before
            item_kwargs["image_tokens_after"] = keep_spec.tokens_after

            new_image_items.append(item_kwargs)

            # Rewrite placeholder length so the downstream merge expects fewer image tokens.
            new_len = keep_spec.tokens_after
            new_range = PlaceholderRange(
                offset=old_range.offset,
                length=new_len,
                is_embed=old_range.is_embed,
            )
            new_image_ranges.append(new_range)

        mm_kwargs["image"] = new_image_items
        mm_placeholders["image"] = new_image_ranges
        mm_input["mm_kwargs"] = mm_kwargs
        mm_input["mm_placeholders"] = mm_placeholders
        return mm_input