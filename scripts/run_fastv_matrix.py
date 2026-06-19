"""Phase 3 — input-level x in-model pruning composition on V*Bench.

Matrix: input resolution {full, downscale@0.3} x in-model {none, FastV r=0.5, r=0.25},
scored as multiple-choice on the Qwen2-VL-7B answer model (same model that produces
the FastV attention signal). Tests whether the two savings stack and whether
downscaling hurts FastV (the plan's Q2 hypotheses).

Resumable JSONL of {id, cond, gold, pred, score, img_before, img_after}.

Usage:
    PYTHONPATH=. python scripts/run_fastv_matrix.py --out data/vqa_stress/fastv_vstar.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from rag.fastv import FastVQwen2VL
from rag.vqa_datasets import load_vstar

CONDITIONS = [
    {"name": "full",            "input_keep": 1.0, "fastv_layer": None, "keep_ratio": 1.0},
    {"name": "full+fastv0.5",   "input_keep": 1.0, "fastv_layer": 3,    "keep_ratio": 0.5},
    {"name": "full+fastv0.25",  "input_keep": 1.0, "fastv_layer": 3,    "keep_ratio": 0.25},
    {"name": "ds0.3",           "input_keep": 0.3, "fastv_layer": None, "keep_ratio": 1.0},
    {"name": "ds0.3+fastv0.5",  "input_keep": 0.3, "fastv_layer": 3,    "keep_ratio": 0.5},
    {"name": "ds0.3+fastv0.25", "input_keep": 0.3, "fastv_layer": 3,    "keep_ratio": 0.25},
]


def _load_done(out: Path) -> set:
    done = set()
    if out.exists():
        with out.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                    done.add((r["id"], r["cond"]))
                except Exception:
                    continue
    return done


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = load_vstar(limit=args.limit or None)
    fv = FastVQwen2VL()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out)
    print(f"[run] {len(records)} examples x {len(CONDITIONS)} conditions; "
          f"{len(done)} already done")

    t0 = time.time()
    n_new = 0
    with out.open("a") as f:
        for ri, rec in enumerate(records):
            img0 = Image.open(rec["image_path"]).convert("RGB")
            for cond in CONDITIONS:
                if (rec["id"], cond["name"]) in done:
                    continue
                try:
                    res = fv.answer_mc(img0, rec["question"], num_choices=4,
                                       input_keep=cond["input_keep"],
                                       fastv_layer=cond["fastv_layer"],
                                       keep_ratio=cond["keep_ratio"])
                except Exception as e:
                    print(f"[err] {rec['id']} {cond['name']}: {e}")
                    continue
                score = float(res["pred"] == rec["gold"])
                f.write(json.dumps({
                    "id": rec["id"], "cond": cond["name"], "gold": rec["gold"],
                    "pred": res["pred"], "score": score,
                    "img_before": res["image_tokens_before"],
                    "img_after": res["image_tokens_after"],
                }) + "\n")
                f.flush()
                n_new += 1
            if (ri + 1) % 20 == 0:
                print(f"[prog] {ri+1}/{len(records)} examples, {n_new} new rows, "
                      f"{time.time()-t0:.0f}s")
    print(f"[done] {n_new} new rows in {time.time()-t0:.0f}s -> {out}")


if __name__ == "__main__":
    main()
