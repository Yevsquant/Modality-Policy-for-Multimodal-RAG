#!/usr/bin/env python3
"""
Run the multimodal pipeline on VQAv2 validation data.

Default: full official val split when JSON + COCO paths are provided; otherwise
full Hugging Face split (no max_examples cap). Use --max-examples N to cap.

Requires: pip install datasets (see requirement.txt).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from rag.config import RAGConfig
from rag.eval_baseline import run_baseline, run_offline_judge
from rag.vllm_metrics import (
    default_metrics_url_from_vlm_api_base,
    diff_prometheus_metrics,
    scrape_prometheus_metrics,
)


def _merge_vllm_prometheus_into_predictions(
    predictions_path: Path,
    metrics_url: str,
    metrics_before: dict[str, float] | None,
    metrics_after: dict[str, float] | None,
    error: str | None,
) -> None:
    with predictions_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if metrics_before is not None and metrics_after is not None and not error:
        delta = diff_prometheus_metrics(metrics_before, metrics_after)
        data["vllm_prometheus"] = {
            "metrics_url": metrics_url,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "metrics_delta": delta,
        }
    else:
        data["vllm_prometheus"] = {
            "metrics_url": metrics_url,
            "metrics_before": metrics_before,
            "metrics_after": metrics_after,
            "error": error or "incomplete prometheus scrape",
        }
    with predictions_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="VQAv2 baseline (HF or official JSON + COCO val2014).")
    parser.add_argument(
        "--max-examples",
        type=int,
        default=500,
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
    parser.add_argument(
        "--metrics-url",
        type=str,
        default=None,
        help="vLLM Prometheus /metrics URL (default: same host/port as config vlm_api_base, path /metrics).",
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

    metrics_url = (args.metrics_url or "").strip() or default_metrics_url_from_vlm_api_base(cfg.vlm_api_base)

    scrape_errors: list[str] = []
    metrics_before: dict[str, float] | None = None
    metrics_after: dict[str, float] | None = None

    try:
        metrics_before = scrape_prometheus_metrics(metrics_url)
    except Exception as e:
        scrape_errors.append(f"metrics_before: {e}")

    run_baseline(cfg)

    try:
        metrics_after = scrape_prometheus_metrics(metrics_url)
    except Exception as e:
        scrape_errors.append(f"metrics_after: {e}")

    predictions_path = cfg.output_dir / "baseline_predictions.json"
    scrape_error = "; ".join(scrape_errors) if scrape_errors else None
    _merge_vllm_prometheus_into_predictions(
        predictions_path,
        metrics_url,
        metrics_before,
        metrics_after,
        scrape_error,
    )

    if metrics_before is not None and metrics_after is not None and not scrape_error:
        delta = diff_prometheus_metrics(metrics_before, metrics_after)
        pt = delta.get("vllm:prompt_tokens_total")
        gt = delta.get("vllm:generation_tokens_total")
        if pt is not None and gt is not None:
            total = pt + gt
            print(
                f"vLLM tokens (baseline delta, model={cfg.vlm_model_name!r}): "
                f"prompt={pt:.0f} generation={gt:.0f} total={total:.0f}"
            )
        else:
            print(
                "vLLM prometheus delta: prompt/generation totals not found in scraped metrics "
                f"(metrics_url={metrics_url!r}). See vllm_prometheus in baseline_predictions.json."
            )
    else:
        print(
            f"vLLM token metrics unavailable ({scrape_error or 'unknown'}). "
            f"metrics_url={metrics_url!r}. See vllm_prometheus.error in baseline_predictions.json."
        )

    run_offline_judge(cfg)


if __name__ == "__main__":
    main()
