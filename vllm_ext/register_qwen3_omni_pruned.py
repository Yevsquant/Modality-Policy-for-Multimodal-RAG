from __future__ import annotations

from typing import Any

from vllm.model_executor.models import ModelRegistry

from .qwen3_omni_pruned_model import Qwen3OmniPrunedForConditionalGeneration
from .qwen3_omni_pruned_processor import Qwen3OmniPrunedProcessor


def build_pruned_processor(*args, **kwargs):
    return Qwen3OmniPrunedProcessor(
        *args,
        **kwargs,
    )


def _register_pruned_model() -> None:
    architecture_names = [
        "Qwen3OmniForConditionalGeneration",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
        "Qwen3OmniMoeForConditionalGeneration",
    ]
    for arch in architecture_names:
        ModelRegistry.register_model(arch, Qwen3OmniPrunedForConditionalGeneration)


def _register_pruned_processor() -> None:
    # Runtime-agnostic import chain:
    # - vLLM-Omni style path first
    # - upstream vLLM path fallback
    try:
        from vllm_omni.model_executor.models.qwen3_omni import (
            Qwen3OmniDummyInputsBuilder,
            Qwen3OmniProcessingInfo,
        )
    except ImportError:
        from vllm.model_executor.models.qwen3_omni_moe_thinker import (
            Qwen3OmniMoeThinkerDummyInputsBuilder as Qwen3OmniDummyInputsBuilder,
            Qwen3OmniMoeThinkerProcessingInfo as Qwen3OmniProcessingInfo,
        )

    from vllm.multimodal import MULTIMODAL_REGISTRY

    MULTIMODAL_REGISTRY.register_processor(
        Qwen3OmniPrunedProcessor,
        info=Qwen3OmniProcessingInfo,
        dummy_inputs=Qwen3OmniDummyInputsBuilder,
    )(Qwen3OmniPrunedForConditionalGeneration)


_register_pruned_model()
_register_pruned_processor()