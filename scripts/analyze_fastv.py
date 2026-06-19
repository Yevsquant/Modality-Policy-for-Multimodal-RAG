"""Aggregate the Phase-3 FastV matrix into the composition report.

Per condition: accuracy + bootstrap CI + avg deep-layer image tokens. Plus the
paired tests for the plan's Q2 hypotheses:
  - FastV's marginal effect at full res vs after downscaling (diminishing overlap?)
  - whether downscale+FastV stacks below either alone at acceptable accuracy
  - negative interaction: is FastV's accuracy hit larger after downscaling?
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rag.metrics import bootstrap_ci, paired_diff_ci


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    score = defaultdict(dict)   # cond -> {id: score}
    tok = defaultdict(list)     # cond -> [img_after]
    ids = set()
    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            score[r["cond"]][r["id"]] = r["score"]
            tok[r["cond"]].append(r["img_after"])
            ids.add(r["id"])

    conds = ["full", "full+fastv0.5", "full+fastv0.25",
             "ds0.3", "ds0.3+fastv0.5", "ds0.3+fastv0.25",
             "ds0.5", "ds0.25"]
    conds = [c for c in conds if c in score]
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

    tests = {
        "fastv0.5_effect_at_full": paired("full+fastv0.5", "full"),
        "fastv0.25_effect_at_full": paired("full+fastv0.25", "full"),
        "fastv0.5_effect_after_ds": paired("ds0.3+fastv0.5", "ds0.3"),
        "fastv0.25_effect_after_ds": paired("ds0.3+fastv0.25", "ds0.3"),
        "downscale_effect_no_fastv": paired("ds0.3", "full"),
        "stacked_vs_full": paired("ds0.3+fastv0.5", "full"),
    }
    # Goal C: FastV vs input-downscale at MATCHED token budgets (the better lever?).
    # ~509 tok: full+fastv0.5 vs ds0.5 ; ~255 tok: full+fastv0.25 vs ds0.25.
    if "ds0.5" in score:
        tests["matched509_fastv_vs_downscale"] = paired("full+fastv0.5", "ds0.5")
    if "ds0.25" in score:
        tests["matched255_fastv_vs_downscale"] = paired("full+fastv0.25", "ds0.25")

    report = {"n_common": len(common), "per_cond": per_cond, "paired_tests": tests}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print(f"=== Phase 3 FastV composition (V*Bench, n={len(common)}) ===")
    for c in conds:
        p = per_cond[c]
        print(f"  {c:<18} acc={p['acc']:.3f} [{p['ci_low']:.3f},{p['ci_high']:.3f}]  "
              f"img_tokens={p['avg_img_tokens']:.0f}")
    print("  paired tests (delta = a - b):")
    for name, t in tests.items():
        flag = "  SIG" if t["significant"] else ""
        print(f"    {name:<28} {t['a']} - {t['b']}: "
              f"{t['delta']:+.3f} [{t['ci_low']:+.3f},{t['ci_high']:+.3f}]{flag}")
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
