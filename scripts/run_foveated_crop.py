"""Angle 2 — query-conditioned foveated CROP vs FastV / uniform downscale on V*Bench.

For each example: localize the query-relevant region (CLIP heatmap or coarse-to-fine
7B attention), crop it from the full-res image, downscale the crop to a token budget
(~255 or ~509), and score the crop as multiple-choice on the SAME Qwen2-VL-7B answer
model (`rag.fastv.FastVQwen2VL.answer_mc`, input_keep=1.0, no in-model pruning — the
crop already reduced the tokens).

Writes a resumable JSONL {id, cond, gold, pred, score, img_tokens, category} that
`scripts/analyze_foveated_crop.py` pairs against the existing ds0.25/ds0.5 and
full+fastv0.25/0.5 columns in data/vqa_stress/fastv_vstar.jsonl at matched budgets.

Usage:
    PYTHONPATH=. python scripts/run_foveated_crop.py \
        --out data/vqa_stress/foveated_crop_vstar.jsonl
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from rag.fastv import FastVQwen2VL
from rag.foveated_crop import ClipLocalizer, c2f_relevance, make_foveated_image
from rag.vqa_datasets import load_vstar

# Budgets matched to the FastV/downscale columns already in fastv_vstar.jsonl.
BUDGETS = {"255": 255, "509": 509}
# (localizer, budget_key)
CONDITIONS = [
    ("crop_clip", "255"), ("crop_clip", "509"),
    ("crop_c2f", "255"), ("crop_c2f", "509"),
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
    ap.add_argument("--percentile", type=float, default=80.0)
    ap.add_argument("--margin", type=float, default=0.10)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    records = load_vstar(limit=args.limit or None)
    fv = FastVQwen2VL()
    clip_loc = ClipLocalizer()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    done = _load_done(out)
    print(f"[run] {len(records)} examples x {len(CONDITIONS)} conditions; "
          f"{len(done)} already done")

    t0 = time.time()
    n_new = 0
    with out.open("a") as f:
        for ri, rec in enumerate(records):
            cats = {c[0] for c in CONDITIONS if (rec["id"], _cond_name(*c)) not in done}
            if not cats:
                continue
            img0 = Image.open(rec["image_path"]).convert("RGB")

            # Compute each localizer's relevance map once (reused across budgets).
            rel_cache = {}
            if "crop_clip" in cats:
                rel_cache["crop_clip"] = (
                    clip_loc.relevance(img0, rec["question"]), clip_loc.grid_hw)
            if "crop_c2f" in cats:
                rel_cache["crop_c2f"] = c2f_relevance(fv, img0, rec["question"])

            for loc, bkey in CONDITIONS:
                cond = _cond_name(loc, bkey)
                if (rec["id"], cond) in done:
                    continue
                try:
                    rel, grid_hw = rel_cache[loc]
                    crop = make_foveated_image(
                        img0, rel, grid_hw, BUDGETS[bkey],
                        percentile=args.percentile, margin=args.margin)
                    res = fv.answer_mc(crop, rec["question"], num_choices=4,
                                       input_keep=1.0, fastv_layer=None)
                except Exception as e:
                    print(f"[err] {rec['id']} {cond}: {e}")
                    continue
                score = float(res["pred"] == rec["gold"])
                f.write(json.dumps({
                    "id": rec["id"], "cond": cond, "gold": rec["gold"],
                    "pred": res["pred"], "score": score,
                    "img_tokens": res["image_tokens_after"],
                    "category": rec.get("category", ""),
                }) + "\n")
                f.flush()
                n_new += 1
            if (ri + 1) % 20 == 0:
                print(f"[prog] {ri+1}/{len(records)} examples, {n_new} new rows, "
                      f"{time.time()-t0:.0f}s")
    print(f"[done] {n_new} new rows in {time.time()-t0:.0f}s -> {out}")


def _cond_name(loc: str, bkey: str) -> str:
    return f"{loc}{bkey}"


if __name__ == "__main__":
    main()
