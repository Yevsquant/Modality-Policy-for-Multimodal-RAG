"""Aggregate Angle-2 foveated-crop results and pair them against FastV / downscale.

Loads the crop JSONL (crop_clip/crop_c2f @ 255/509) plus the existing FastV+downscale
columns from fastv_vstar.jsonl, computes per-condition accuracy + bootstrap CI + avg
tokens, and the matched-budget paired tests:
  - crop vs uniform downscale (localization-quality proxy: does crop > downscale?)
  - crop vs full+FastV (does the input-side crop match/beat in-model pruning?)
Plus a per-category (direct_attributes / relative_position) accuracy breakdown.

Usage:
    PYTHONPATH=. python scripts/analyze_foveated_crop.py \
        --crop data/vqa_stress/foveated_crop_vstar.jsonl \
        --fastv data/vqa_stress/fastv_vstar.jsonl \
        --out data/vqa_stress/foveated_crop_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rag.metrics import bootstrap_ci, paired_diff_ci


def _load(path, score, tok, cat, ids, tok_key):
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            score[r["cond"]][r["id"]] = r["score"]
            tok[r["cond"]].append(r[tok_key])
            cat[r["id"]] = r.get("category") or r["id"].split("-")[1]
            ids.add(r["id"])


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--crop", required=True)
    ap.add_argument("--fastv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    score = defaultdict(dict)
    tok = defaultdict(list)
    cat = {}
    ids = set()
    _load(args.crop, score, tok, cat, ids, "img_tokens")
    _load(args.fastv, score, tok, cat, ids, "img_after")

    crop_conds = ["crop_clip255", "crop_clip509", "crop_c2f255", "crop_c2f509"]
    ref_conds = ["full", "full+fastv0.5", "full+fastv0.25", "ds0.5", "ds0.25"]
    conds = [c for c in crop_conds + ref_conds if c in score]
    common = sorted(i for i in ids if all(i in score[c] for c in conds))

    per_cond = {}
    for c in conds:
        s = [score[c][i] for i in common]
        m, lo, hi = bootstrap_ci(s)
        per_cond[c] = {"acc": m, "ci_low": lo, "ci_high": hi,
                       "avg_img_tokens": sum(tok[c]) / len(tok[c]) if tok[c] else None,
                       "n": len(s)}

    def paired(a, b):
        d, lo, hi = paired_diff_ci([score[a][i] for i in common],
                                   [score[b][i] for i in common])
        return {"a": a, "b": b, "delta": d, "ci_low": lo, "ci_high": hi,
                "significant": hi is not None and (hi < 0 or lo > 0)}

    tests = {}
    # Matched-budget pairings. crop_*255 ~ 255 tok (vs ds0.25, full+fastv0.25);
    # crop_*509 ~ 509 tok (vs ds0.5, full+fastv0.5).
    for loc in ("crop_clip", "crop_c2f"):
        for b, ds, fastv in (("255", "ds0.25", "full+fastv0.25"),
                             ("509", "ds0.5", "full+fastv0.5")):
            c = f"{loc}{b}"
            if c not in score:
                continue
            tests[f"{c}_vs_downscale"] = paired(c, ds)
            tests[f"{c}_vs_fastv"] = paired(c, fastv)

    # Per-category accuracy per condition.
    categories = sorted(set(cat[i] for i in common))
    per_category = {}
    for cval in categories:
        ids_c = [i for i in common if cat[i] == cval]
        per_category[cval] = {"n": len(ids_c)}
        for c in conds:
            per_category[cval][c] = sum(score[c][i] for i in ids_c) / len(ids_c)

    report = {"n_common": len(common), "per_cond": per_cond,
              "paired_tests": tests, "per_category": per_category}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"=== Angle 2 foveated crop (V*Bench, n={len(common)}) ===")
    for c in conds:
        p = per_cond[c]
        print(f"  {c:<16} acc={p['acc']:.3f} [{p['ci_low']:.3f},{p['ci_high']:.3f}]  "
              f"tok={p['avg_img_tokens']:.0f}")
    print("  paired tests (delta = a - b):")
    for name, t in tests.items():
        flag = "  SIG" if t["significant"] else ""
        print(f"    {name:<28} {t['delta']:+.3f} "
              f"[{t['ci_low']:+.3f},{t['ci_high']:+.3f}]{flag}")
    print("  per-category accuracy:")
    hdr = "    " + "category".ljust(20) + "".join(c[:14].ljust(15) for c in conds)
    print(hdr)
    for cval in categories:
        row = "    " + f"{cval}(n={per_category[cval]['n']})".ljust(20)
        row += "".join(f"{per_category[cval][c]:.3f}".ljust(15) for c in conds)
        print(row)
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
