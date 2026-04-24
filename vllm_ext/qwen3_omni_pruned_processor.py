from __future__ import annotations

from copy import deepcopy

from vllm.multimodal.processing.processor import BaseMultiModalProcessor
from vllm.multimodal.inputs import PlaceholderRange
from vllm.multimodal.processing import ProcessorInputs

from .visual_pruning_config import VisualPruningConfig
from .visual_pruning_policy import ImageKeepSpec, build_keep_spec, calculate_keep_n

try:
    from vllm_omni.model_executor.models.qwen3_omni import (
        Qwen3OmniMultiModalProcessor as BaseProcessor
    )
except ImportError:
    try:
        from vllm.model_executor.models.qwen3_omni_moe_thinker import (
            Qwen3OmniMoeThinkerMultiModalProcessor as BaseProcessor
        )
    except ImportError:
        # Fallback: Qwen3-Omni usually relies on Qwen2-VL's processor architecture
        from vllm.model_executor.models.qwen2_vl import (
            Qwen2VLMultiModalProcessor as BaseProcessor
        )


class Qwen3OmniPrunedProcessor(BaseProcessor):
    """
    Embedding-based pruning rollout.

    The processor is now intentionally lightweight:
    - runs the normal multimodal processor path
    - forwards image inputs and metadata unchanged
    - annotates mm_kwargs so the model-side embedding path knows pruning is enabled

    It no longer computes keep_indices or rewrites placeholder lengths. The pruning
    decision uses actual image embedding values later in qwen3_omni_pruned_model.py.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.visual_pruning = VisualPruningConfig()

    def apply(self, inputs: ProcessorInputs, timing_ctx):
        mm_input = super().apply(inputs, timing_ctx)

        if not self.visual_pruning.enabled:
            return mm_input

        # Copy so downstream runtime can mutate safely.
        mm_kwargs = deepcopy(mm_input.get("mm_kwargs", {}))
        if not isinstance(mm_kwargs, dict):
            return mm_input

        # mm_placeholders = dict(mm_input["mm_placeholders"])
        
        # if "image" not in mm_placeholders or "image_grid_thw" not in mm_kwargs:
        #     print("We got image but we denied that :)") # del
        #     return mm_input
        
        # image_ranges = mm_placeholders["image"]
        # grid_thw = mm_kwargs["image_grid_thw"]
        # for i, old_range in enumerate(image_ranges):
        #     num_tokens = grid_thw[i][1].item() * grid_thw[i][2].item()
        #     # spec = build_keep_spec(self.visual_pruning.min_keep,
        #     #                        self.visual_pruning.keep_ratio,
        #     #                        num_tokens,
        #     #                        False)
        #     old_range.length = calculate_keep_n(self.visual_pruning.min_keep, self.visual_pruning.keep_ratio, num_tokens)
        # mm_placeholders["image"] = image_ranges

        mm_kwargs["visual_pruning_enabled"] = True
        mm_kwargs["visual_pruning_policy"] = self.visual_pruning.policy
        mm_kwargs["visual_pruning_keep_ratio"] = self.visual_pruning.keep_ratio
        mm_kwargs["visual_pruning_min_keep"] = self.visual_pruning.min_keep
        mm_kwargs["visual_pruning_keep_cls_token"] = self.visual_pruning.keep_cls_token
        mm_kwargs["visual_pruning_normalize_scores"] = self.visual_pruning.normalize_scores
        mm_kwargs["visual_pruning_hard_prune"] = self.visual_pruning.hard_prune
        mm_kwargs["visual_pruning_fallback_to_soft_mask"] = self.visual_pruning.fallback_to_soft_mask
        mm_kwargs["visual_pruning_deferred_to_model"] = True

        mm_input["mm_kwargs"] = mm_kwargs
        return mm_input
