"""Phase 0 — downscale stress test (the go/no-go gate).

For a single-image VQA dataset, sweep a keep_ratio ladder, downscale each image to
each budget, send it to the target VLM, and score with the dataset-native metric.
The question the gate answers: does accuracy fall significantly as visual tokens
fall? If yes (unlike MMDocRAG's flat frontier), the dataset stresses visual tokens
and the adaptive-budget research is worth pursuing.

Output is an incremental, resumable JSONL of per-(example, keep_ratio) rows:
    {"id", "keep", "tokens", "score", "pred"}
Run `scripts/analyze_stress_test.py` on it for the accuracy/CI/oracle-budget table.

Usage:
    PYTHONPATH=. python scripts/downscale_stress_test.py \
        --dataset vstar --transform downscale \
        --keep-ratios 1.0,0.5,0.3,0.2,0.1 \
        --out data/vqa_stress/vstar_downscale.jsonl
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import time
from pathlib import Path

from openai import OpenAI
from PIL import Image

from rag.config import RAGConfig
from rag.image_ops import apply_transform
from rag.vqa_datasets import load_dataset_by_name
from rag.vqa_scoring import score_example
from rag.visual_token_counter import VisualTokenCounter


def _img_to_data_url(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="JPEG", quality=95)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


def _ask(client: OpenAI, model: str, image: Image.Image, question: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=0.0,
        max_tokens=64,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": _img_to_data_url(image)}},
                    {"type": "text", "text": question},
                ],
            }
        ],
    )
    return (resp.choices[0].message.content or "").strip()


def _load_done(out: Path) -> set:
    done = set()
    if out.exists():
        with out.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["id"], r["keep"]))
                except Exception:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", required=True,
                    choices=["vstar", "docvqa", "hrbench", "hrbench4k", "hrbench8k"])
    ap.add_argument("--transform", default="downscale", choices=["downscale", "trim_downscale"])
    ap.add_argument("--keep-ratios", default="1.0,0.5,0.3,0.2,0.1")
    ap.add_argument("--limit", type=int, default=0, help="0 = all examples")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    cfg = RAGConfig()
    keep_ratios = [float(x) for x in args.keep_ratios.split(",")]
    records = load_dataset_by_name(args.dataset, limit=args.limit or None)
    print(f"[load] {args.dataset}: {len(records)} examples, keep={keep_ratios}, "
          f"transform={args.transform}")

    counter = VisualTokenCounter(cfg.vlm_model_name)
    client = OpenAI(base_url=cfg.vlm_api_base, api_key="EMPTY")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out)
    if done:
        print(f"[resume] {len(done)} (id,keep) rows already present")

    t0 = time.time()
    n_new = 0
    with out.open("a") as f:
        for ri, rec in enumerate(records):
            try:
                base_img = Image.open(rec["image_path"]).convert("RGB")
            except Exception as e:
                print(f"[skip] {rec['id']}: cannot open image ({e})")
                continue
            for kr in keep_ratios:
                if (rec["id"], kr) in done:
                    continue
                img = apply_transform(base_img, args.transform, kr)
                tokens = counter.count(img)
                try:
                    pred = _ask(client, cfg.vlm_model_name, img, rec["question"])
                except Exception as e:
                    print(f"[err] {rec['id']} kr={kr}: {e}")
                    continue
                score = score_example(rec["task"], pred, rec)
                row = {"id": rec["id"], "keep": kr, "tokens": tokens,
                       "score": score, "pred": pred[:120], "category": rec.get("category")}
                f.write(json.dumps(row) + "\n")
                f.flush()
                n_new += 1
            if (ri + 1) % 20 == 0:
                dt = time.time() - t0
                print(f"[prog] {ri+1}/{len(records)} examples, {n_new} new rows, {dt:.0f}s")

    print(f"[done] wrote {n_new} new rows to {out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
