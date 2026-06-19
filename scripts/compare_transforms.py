"""Paired trim_downscale-vs-downscale comparison at equal keep_ratio budgets.

Reads two stress-test JSONLs (same dataset, same keep ladder, different transform)
and, per shared keep_ratio, reports each transform's mean score + the paired
difference CI over the examples both ran. A CI excluding 0 = a real win at equal
token budget — the Phase-1 claim ("trim beats plain downscale on detail data").

Usage:
    PYTHONPATH=. python scripts/compare_transforms.py \
        --downscale data/vqa_stress/vstar_downscale.jsonl \
        --trim data/vqa_stress/vstar_trim.jsonl \
        --out data/vqa_stress/vstar_trim_vs_downscale.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rag.metrics import bootstrap_ci, paired_diff_ci


def _load(path):
    by_keep = defaultdict(dict)      # keep -> {id: score}
    tok = defaultdict(list)          # keep -> [tokens]
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            by_keep[r["keep"]][r["id"]] = r["score"]
            tok[r["keep"]].append(r["tokens"])
    return by_keep, tok


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--downscale", required=True)
    ap.add_argument("--trim", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    d_score, d_tok = _load(args.downscale)
    t_score, t_tok = _load(args.trim)
    keeps = sorted(set(d_score) & set(t_score), reverse=True)

    rows = []
    for k in keeps:
        common = sorted(set(d_score[k]) & set(t_score[k]))
        ds = [d_score[k][i] for i in common]
        ts = [t_score[k][i] for i in common]
        dm, _, _ = bootstrap_ci(ds)
        tm, _, _ = bootstrap_ci(ts)
        diff, lo, hi = paired_diff_ci(ts, ds)  # trim - downscale
        rows.append({
            "keep": k,
            "n": len(common),
            "downscale_score": dm,
            "trim_score": tm,
            "downscale_tokens": sum(d_tok[k]) / len(d_tok[k]),
            "trim_tokens": sum(t_tok[k]) / len(t_tok[k]),
            "delta_trim_minus_downscale": diff,
            "ci_low": lo,
            "ci_high": hi,
            "trim_significantly_better": lo is not None and lo > 0,
        })

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"downscale": args.downscale, "trim": args.trim, "per_keep": rows},
                  f, indent=2)

    print(f"=== trim vs downscale: {args.trim} ===")
    for r in rows:
        flag = "  <== trim SIG better" if r["trim_significantly_better"] else ""
        print(f"  keep={r['keep']:<4} n={r['n']:<4} "
              f"down={r['downscale_score']:.3f}@{r['downscale_tokens']:.0f}t  "
              f"trim={r['trim_score']:.3f}@{r['trim_tokens']:.0f}t  "
              f"d={r['delta_trim_minus_downscale']:+.3f} [{r['ci_low']:+.3f},{r['ci_high']:+.3f}]{flag}")
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
