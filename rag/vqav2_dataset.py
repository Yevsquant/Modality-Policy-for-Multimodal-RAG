from __future__ import annotations

import json
import logging
import re
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

from datasets import load_dataset
from PIL import Image

from rag.config import RAGConfig

logger = logging.getLogger(__name__)

_COCO_VAL_PREFIX = "COCO_val2014_"


def _official_vqa_configured(cfg: RAGConfig) -> bool:
    q = cfg.vqa_questions_json
    a = cfg.vqa_annotations_json
    img = cfg.vqa_coco_images_dir
    return bool(q and a and img)


def _coco_val_image_path(images_dir: Path, image_id: int) -> Path:
    return images_dir / f"{_COCO_VAL_PREFIX}{int(image_id):012d}.jpg"


def _build_vqa_example_dict(
    *,
    q_id: object,
    question: str,
    answers: List[str],
    local_image: Path,
    cache_basename: str,
    question_type: object | None = None,
) -> Dict[str, Any]:
    answer_short = _majority_answer(answers)
    img_quotes = [
        {
            "quote_id": "image1",
            "type": "image",
            "img_path": cache_basename,
            "img_description": "",
            "local_img_path": str(local_image.resolve()),
        }
    ]
    return {
        "q_id": q_id,
        "doc_name": str(q_id),
        "domain": None,
        "question": question.strip(),
        "question_type": question_type,
        "evidence_modality_type": ["image"],
        "text_quotes": [],
        "img_quotes": img_quotes,
        "gold_quotes": ["image1"],
        "answer": answer_short,
        "vqa_answers": answers,
    }


def load_vqav2_examples_from_official_json(cfg: RAGConfig) -> List[Dict[str, Any]]:
    if not _official_vqa_configured(cfg):
        raise ValueError("Official VQA paths are not fully set on config.")
    q_path = cfg.vqa_questions_json
    a_path = cfg.vqa_annotations_json
    coco_dir = cfg.vqa_coco_images_dir
    assert q_path is not None and a_path is not None and coco_dir is not None
    if not q_path.is_file():
        raise FileNotFoundError(f"Questions JSON not found: {q_path}")
    if not a_path.is_file():
        raise FileNotFoundError(f"Annotations JSON not found: {a_path}")
    if not coco_dir.is_dir():
        raise FileNotFoundError(f"COCO images directory not found: {coco_dir}")

    with q_path.open("r", encoding="utf-8") as f:
        qdata = json.load(f)
    with a_path.open("r", encoding="utf-8") as f:
        adata = json.load(f)

    questions = qdata.get("questions") or []
    qmap = {int(q["question_id"]): q for q in questions if "question_id" in q}

    annotations_all = adata.get("annotations") or []
    if not annotations_all:
        raise RuntimeError(f"No annotations found in {a_path}")
    annotations = annotations_all
    if cfg.max_examples is not None:
        annotations = annotations[: cfg.max_examples]

    out: List[Dict[str, Any]] = []
    skipped = 0
    for ann in annotations:
        qid = int(ann["question_id"])
        image_id = int(ann["image_id"])
        qrow = qmap.get(qid)
        if not qrow:
            skipped += 1
            continue
        question = str(qrow.get("question", "")).strip()
        raw_answers = ann.get("answers") or []
        answers = [str(x["answer"]) for x in raw_answers if isinstance(x, dict) and "answer" in x]
        if not answers:
            skipped += 1
            continue
        img_file = _coco_val_image_path(coco_dir, image_id)
        if not img_file.is_file():
            skipped += 1
            continue
        out.append(
            _build_vqa_example_dict(
                q_id=qid,
                question=question,
                answers=answers,
                local_image=img_file,
                cache_basename=img_file.name,
                question_type=ann.get("question_type"),
            )
        )

    if skipped:
        logger.warning("Skipped %d official VQA rows (missing question, answers, or image file).", skipped)
    if not out and annotations_all:
        raise RuntimeError(
            "No official VQA examples were built. Check that vqa_coco_images_dir contains "
            "COCO_val2014_*.jpg files matching the annotation image_ids."
        )
    logger.info("Loaded %d official VQAv2-style examples from JSON.", len(out))
    return out


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
    if _official_vqa_configured(cfg):
        return load_vqav2_examples_from_official_json(cfg)

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
        fname = f"{re.sub(r'[^0-9A-Za-z_.-]+', '_', str(qid))}.jpg"
        dest = cache_dir / fname
        img_obj = rowd.get("image")
        if img_obj is None:
            raise ValueError(f"Row {idx} has no 'image' field (dataset {resolved_name})")
        _ensure_image_file(img_obj, dest)

        out.append(
            _build_vqa_example_dict(
                q_id=qid,
                question=str(rowd.get("question", "")).strip(),
                answers=answers,
                local_image=dest,
                cache_basename=fname,
                question_type=rowd.get("question_type"),
            )
        )
    return out
