"""Aggregate a downscale stress-test JSONL into the Phase-0 gate report.

Produces, per keep_ratio: mean score, bootstrap 95% CI, mean visual tokens.
Plus the GATE signal:
  - paired difference (full-res vs each lower budget), CI — does accuracy drop
    significantly as tokens fall?
  - oracle-budget distribution: per example, the smallest keep_ratio that is still
    correct. Spread here = budget variance = the precondition for a learned policy.

Usage:
    PYTHONPATH=. python scripts/analyze_stress_test.py \
        --in data/vqa_stress/vstar_downscale.jsonl --out data/vqa_stress/vstar_gate.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rag.metrics import bootstrap_ci, paired_diff_ci


def _correct(task_score: float) -> int:
    # MC scores are 0/1; ANLS is graded — treat >=0.5 as "correct" for the
    # oracle-budget definition (matches the ANLS thresholding).
    return int(task_score >= 0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = []
    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))

    # index: keep -> {id: score}, and keep -> [tokens]
    by_keep_score = defaultdict(dict)
    by_keep_tokens = defaultdict(list)
    ids = set()
    for r in rows:
        by_keep_score[r["keep"]][r["id"]] = r["score"]
        by_keep_tokens[r["keep"]].append(r["tokens"])
        ids.add(r["id"])

    keeps = sorted(by_keep_score.keys(), reverse=True)  # high res first
    full_keep = keeps[0]
    # examples present at every keep (for paired comparisons)
    common = [i for i in ids if all(i in by_keep_score[k] for k in keeps)]
    common.sort()

    per_keep = []
    for k in keeps:
        scores = [by_keep_score[k][i] for i in common]
        mean, lo, hi = bootstrap_ci(scores)
        toks = by_keep_tokens[k]
        per_keep.append({
            "keep": k,
            "mean_score": mean,
            "ci_low": lo,
            "ci_high": hi,
            "avg_tokens": sum(toks) / len(toks) if toks else None,
            "n": len(scores),
        })

    # paired drop vs full res
    full_scores = [by_keep_score[full_keep][i] for i in common]
    paired = []
    for k in keeps[1:]:
        s = [by_keep_score[k][i] for i in common]
        d, lo, hi = paired_diff_ci(s, full_scores)  # (lower - full); negative = worse
        paired.append({"keep": k, "delta_vs_full": d, "ci_low": lo, "ci_high": hi,
                       "significant_drop": hi is not None and hi < 0})

    # oracle budget per example: smallest keep still correct
    oracle = {}
    for i in common:
        ok = [k for k in keeps if _correct(by_keep_score[k][i])]
        oracle[i] = min(ok) if ok else None  # None = wrong even at full res
    budgets = [b for b in oracle.values() if b is not None]
    from collections import Counter
    dist = dict(sorted(Counter([b for b in oracle.values()]).items(),
                       key=lambda kv: (kv[0] is None, kv[0])))
    dist = {str(k): v for k, v in dist.items()}
    variance = len(set(budgets)) > 1

    # GATE verdict: any lower budget shows a significant accuracy drop vs full res
    gate_pass = any(p["significant_drop"] for p in paired)

    report = {
        "dataset_file": args.inp,
        "n_common": len(common),
        "keeps": keeps,
        "per_keep": per_keep,
        "paired_vs_full": paired,
        "oracle_budget_distribution": dist,
        "oracle_budget_has_variance": variance,
        "GATE_PASS_significant_downscale_drop": gate_pass,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"=== Phase-0 gate: {args.inp} (n={len(common)}) ===")
    for p in per_keep:
        print(f"  keep={p['keep']:<4} tokens={p['avg_tokens']:6.0f}  "
              f"score={p['mean_score']:.3f} [{p['ci_low']:.3f},{p['ci_high']:.3f}]")
    print("  paired drop vs full:")
    for p in paired:
        flag = "  <== SIGNIFICANT" if p["significant_drop"] else ""
        print(f"    keep={p['keep']:<4} d={p['delta_vs_full']:+.3f} "
              f"[{p['ci_low']:+.3f},{p['ci_high']:+.3f}]{flag}")
    print(f"  oracle-budget distribution: {dist}")
    print(f"  budget variance: {variance}")
    print(f"  GATE PASS (significant downscale drop): {gate_pass}")
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
