"""Single-image VQA dataset loaders for the downscale stress test (Phase 0).

Each loader returns a list of unified records:
    {
      "id":         unique string,
      "image_path": absolute path to the image file on disk,
      "question":   prompt to send to the VLM (already includes any MC options),
      "task":       "mc" (multiple choice) or "anls" (DocVQA short answer),
      "gold":       letter, for task=="mc",
      "answers":    list[str] of gold answers, for task=="anls",
      "category":   optional grouping label,
    }

These are detail-sensitive, high-resolution benchmarks (V*Bench: small objects in
large scenes; DocVQA: dense document text) chosen because naive downscaling should
provably hurt — the precondition the plan's Phase-0 gate tests for.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional


def load_vstar(limit: Optional[int] = None) -> List[dict]:
    from datasets import load_dataset
    from huggingface_hub import snapshot_download

    repo = snapshot_download("craigwu/vstar_bench", repo_type="dataset")
    ds = load_dataset("craigwu/vstar_bench", split="test")
    records = []
    for ex in ds:
        records.append(
            {
                "id": f"vstar-{ex['category']}-{ex['question_id']}",
                "image_path": os.path.join(repo, ex["image"]),
                "question": ex["text"],
                "task": "mc",
                "gold": ex["label"].strip().upper(),
                "category": ex["category"],
            }
        )
    if limit:
        records = records[:limit]
    return records


def load_docvqa(limit: Optional[int] = 300, seed: int = 0) -> List[dict]:
    """DocVQA validation set. Images are materialized to disk once (the VLM client
    and token counter both consume file paths)."""
    import random

    from datasets import load_dataset

    img_dir = Path("data/vqa_stress/docvqa_images")
    img_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
    idx = list(range(len(ds)))
    random.Random(seed).shuffle(idx)
    if limit:
        idx = idx[:limit]

    records = []
    for i in idx:
        ex = ds[i]
        qid = str(ex.get("questionId", i))
        img_path = img_dir / f"{qid}.jpg"
        if not img_path.exists():
            ex["image"].convert("RGB").save(img_path, "JPEG", quality=95)
        records.append(
            {
                "id": f"docvqa-{qid}",
                "image_path": str(img_path),
                "question": ex["question"].strip()
                + "\nAnswer with the shortest span from the document.",
                "task": "anls",
                "answers": list(ex["answers"]),
                "category": "docvqa",
            }
        )
    return records


def load_dataset_by_name(name: str, limit: Optional[int] = None) -> List[dict]:
    if name == "vstar":
        return load_vstar(limit=limit)
    if name == "docvqa":
        return load_docvqa(limit=limit if limit else 300)
    raise ValueError(f"unknown dataset: {name}")
