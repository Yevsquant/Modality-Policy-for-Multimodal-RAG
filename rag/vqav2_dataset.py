from __future__ import annotations

import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from PIL import Image

from rag.config import RAGConfig

logger = logging.getLogger(__name__)


def _load_with_optional_slice(name: str, split: str, revision: str | None, limit: int | None) -> Any:
    kwargs: Dict[str, Any] = {}
    if revision:
        kwargs["revision"] = revision
    if limit is not None and "[" not in split and "]" not in split:
        split_arg = f"{split}[:{limit}]"
    else:
        split_arg = split
    return load_dataset(name, split=split_arg, **kwargs)  # type: ignore[return-value]


def _load_vqa_raw(cfg: RAGConfig) -> Tuple[Any, str]:
    try:
        ds = _load_with_optional_slice(
            cfg.vqa_hf_dataset,
            cfg.vqa_hf_split,
            cfg.vqa_hf_revision,
            cfg.max_examples,
        )
        return ds, cfg.vqa_hf_dataset
    except Exception as e:
        if (
            not cfg.vqa_hf_auto_fallback
            or cfg.vqa_hf_dataset == cfg.vqa_hf_fallback_dataset
        ):
            raise
        logger.warning(
            "Could not load primary VQA dataset %r (%s: %s). Using fallback %r split=%r.",
            cfg.vqa_hf_dataset,
            type(e).__name__,
            str(e)[:200],
            cfg.vqa_hf_fallback_dataset,
            cfg.vqa_hf_fallback_split,
        )
        ds = _load_with_optional_slice(
            cfg.vqa_hf_fallback_dataset,
            cfg.vqa_hf_fallback_split,
            None,
            cfg.max_examples,
        )
        return ds, f"{cfg.vqa_hf_fallback_dataset} (fallback)"


def _answers_from_row(row: Dict[str, Any]) -> List[str]:
    raw = row.get("answers")
    if isinstance(raw, list) and raw:
        if isinstance(raw[0], str):
            return [str(x) for x in raw]
        if isinstance(raw[0], dict) and "answer" in raw[0]:
            return [str(x["answer"]) for x in raw if isinstance(x, dict)]
    orig = row.get("answers_original")
    if isinstance(orig, list) and orig:
        out = []
        for x in orig:
            if isinstance(x, dict) and "answer" in x:
                out.append(str(x["answer"]))
        if out:
            return out
    mc = row.get("multiple_choice_answer")
    if isinstance(mc, str) and mc.strip():
        return [mc.strip()] * 10
    raise ValueError(
        "Could not parse reference answers from row; "
        f"keys include: {sorted(row.keys())[:40]}"
    )


def _majority_answer(answers: List[str]) -> str:
    counts = Counter(a.strip() for a in answers if a and str(a).strip())
    if not counts:
        return ""
    return counts.most_common(1)[0][0]


def _ensure_image_file(image: Any, dest: Path) -> None:
    if dest.exists():
        return
    dest.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(image, Image.Image):
        rgb = image.convert("RGB")
        rgb.save(dest, format="JPEG", quality=95)
        return
    raise TypeError(f"Unsupported image type: {type(image)}")


def load_vqav2_examples(cfg: RAGConfig) -> List[Dict[str, Any]]:
    if cfg.benchmark != "vqav2":
        raise ValueError("load_vqav2_examples requires cfg.benchmark == 'vqav2'")
    ds, resolved_name = _load_vqa_raw(cfg)
    logger.info("Loaded HF dataset %r with %d rows.", resolved_name, len(ds))

    cache_dir = Path(cfg.vqa_image_cache_dir)
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(ds):
        rowd = dict(row)
        qid = rowd.get("question_id")
        if qid is None:
            qid = rowd.get("id", idx)
        answers = _answers_from_row(rowd)
        answer_short = _majority_answer(answers)
        fname = f"{re.sub(r'[^0-9A-Za-z_.-]+', '_', str(qid))}.jpg"
        dest = cache_dir / fname
        img_obj = rowd.get("image")
        if img_obj is None:
            raise ValueError(f"Row {idx} has no 'image' field (dataset {resolved_name})")
        _ensure_image_file(img_obj, dest)

        img_quotes = [
            {
                "quote_id": "image1",
                "type": "image",
                "img_path": fname,
                "img_description": "",
                "local_img_path": str(dest.resolve()),
            }
        ]
        out.append(
            {
                "q_id": qid,
                "doc_name": str(qid),
                "domain": None,
                "question": str(rowd.get("question", "")).strip(),
                "question_type": rowd.get("question_type"),
                "evidence_modality_type": ["image"],
                "text_quotes": [],
                "img_quotes": img_quotes,
                "gold_quotes": ["image1"],
                "answer_short": answer_short,
                "vqa_answers": answers,
            }
        )
    return out
