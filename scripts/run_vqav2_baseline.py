#!/usr/bin/env python3
"""
Run the multimodal pipeline on VQAv2-style data from Hugging Face.

Writes predictions and judged results under data/vqav2/outputs/ (separate from MMDocRAG).
Requires: pip install datasets (see requirement.txt).
"""
from pathlib import Path

from rag.config import RAGConfig
from rag.eval_baseline import run_baseline, run_offline_judge


def main() -> None:
    cfg = RAGConfig(
        benchmark="vqav2",
        output_dir=Path("data/vqav2/outputs"),
        pruned_image_dir=Path("data/vqav2/outputs/pruned_images"),
        image_prune_cache_path=Path("data/vqav2/outputs/image_prune_cache.json"),
        vqa_image_cache_dir=Path("data/vqav2/image_cache"),
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.pruned_image_dir.mkdir(parents=True, exist_ok=True)
    cfg.vqa_image_cache_dir.mkdir(parents=True, exist_ok=True)

    run_baseline(cfg)
    run_offline_judge(cfg)


if __name__ == "__main__":
    main()
