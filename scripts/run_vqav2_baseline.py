#!/usr/bin/env python3
"""
Run the multimodal pipeline on VQAv2 validation data.

Default: full official val split when JSON + COCO paths are provided; otherwise
full Hugging Face split (no max_examples cap). Use --max-examples N to cap.

Requires: pip install datasets (see requirement.txt).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from rag.config import RAGConfig
from rag.eval_baseline import run_baseline, run_offline_judge


def main() -> None:
    parser = argparse.ArgumentParser(description="VQAv2 baseline (HF or official JSON + COCO val2014).")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        metavar="N",
        help="Process at most N examples (default: no cap — full split / full official JSON).",
    )
    parser.add_argument(
        "--questions-json",
        type=Path,
        default=None,
        help="Official v2_OpenEnded_mscoco_val2014_questions.json (enables full ~214k val with annotations + images).",
    )
    parser.add_argument(
        "--annotations-json",
        type=Path,
        default=None,
        help="Official v2_mscoco_val2014_annotations.json",
    )
    parser.add_argument(
        "--coco-images-dir",
        type=Path,
        default=None,
        help="Directory containing COCO_val2014_*.jpg (e.g. .../val2014 after unzipping val2014.zip).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/vqav2/outputs"),
        help="Predictions and judged JSON output directory.",
    )
    args = parser.parse_args()

    official = bool(args.questions_json and args.annotations_json and args.coco_images_dir)
    if sum(bool(x) for x in (args.questions_json, args.annotations_json, args.coco_images_dir)) in (1, 2):
        parser.error("Provide all three of --questions-json, --annotations-json, and --coco-images-dir for official val, or none for Hugging Face.")

    cfg = RAGConfig(
        benchmark="vqav2",
        max_examples=args.max_examples,
        output_dir=args.output_dir,
        pruned_image_dir=args.output_dir / "pruned_images",
        image_prune_cache_path=args.output_dir / "image_prune_cache.json",
        vqa_image_cache_dir=Path("data/vqav2/image_cache"),
        vqa_questions_json=args.questions_json,
        vqa_annotations_json=args.annotations_json,
        vqa_coco_images_dir=args.coco_images_dir,
    )
    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    cfg.pruned_image_dir.mkdir(parents=True, exist_ok=True)
    cfg.vqa_image_cache_dir.mkdir(parents=True, exist_ok=True)

    run_baseline(cfg)
    run_offline_judge(cfg)


if __name__ == "__main__":
    main()
