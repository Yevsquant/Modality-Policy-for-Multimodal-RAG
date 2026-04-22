from __future__ import annotations

import os
from typing import Any

from vllm.model_executor.models import ModelRegistry

from .qwen3_omni_pruned_model import Qwen3OmniPrunedForConditionalGeneration
from .qwen3_omni_pruned_processor import Qwen3OmniPrunedProcessor
from .visual_pruning_config import VisualPruningConfig


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _resolve_visual_pruning_config(visual_pruning: Any) -> VisualPruningConfig:
    if isinstance(visual_pruning, VisualPruningConfig):
        return visual_pruning

    if isinstance(visual_pruning, dict):
        return VisualPruningConfig(**visual_pruning)

    return VisualPruningConfig(
        enabled=_env_bool("VLLM_VISUAL_PRUNING_ENABLED", False),
        keep_ratio=float(os.getenv("VLLM_VISUAL_PRUNING_KEEP_RATIO", "0.5")),
        min_keep=int(os.getenv("VLLM_VISUAL_PRUNING_MIN_KEEP", "4")),
        policy=os.getenv("VLLM_VISUAL_PRUNING_POLICY", "uniform"),
    )


def build_pruned_processor(*args, **kwargs):
    visual_pruning = _resolve_visual_pruning_config(kwargs.pop("visual_pruning", None))
    return Qwen3OmniPrunedProcessor(
        *args,
        visual_pruning=visual_pruning,
        **kwargs,
    )


def _register_pruned_model() -> None:
    architecture_names = [
        "Qwen3OmniForConditionalGeneration",
        "Qwen3OmniMoeThinkerForConditionalGeneration",
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