from __future__ import annotations

from copy import deepcopy

from vllm.multimodal.processing.processor import BaseMultiModalProcessor
from vllm.multimodal.processing.inputs import ProcessorInputs

from .visual_pruning_config import VisualPruningConfig


class Qwen3OmniPrunedProcessor(BaseMultiModalProcessor):
    """
    Embedding-based pruning rollout.

    The processor is now intentionally lightweight:
    - runs the normal multimodal processor path
    - forwards image inputs and metadata unchanged
    - annotates mm_kwargs so the model-side embedding path knows pruning is enabled

    It no longer computes keep_indices or rewrites placeholder lengths. The pruning
    decision uses actual image embedding values later in qwen3_omni_pruned_model.py.
    """

    def __init__(self, *args, visual_pruning: VisualPruningConfig | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.visual_pruning = visual_pruning or VisualPruningConfig()

    def apply(self, inputs: ProcessorInputs, timing_ctx):
        mm_input = super().apply(inputs, timing_ctx)

        if not self.visual_pruning.enabled:
            return mm_input

        # Copy so downstream runtime can mutate safely.
        mm_kwargs = deepcopy(mm_input.get("mm_kwargs", {}))
        if not isinstance(mm_kwargs, dict):
            return mm_input

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
