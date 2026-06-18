"""Compare pruning methods with bootstrap confidence intervals.

Reads several `*_judged.json` files (each a {"summary", "rows"} produced by the
benchmark) and reports, per method: n, mean judge_correct with 95% CI, mean
visual tokens sent, and mean latency. Then reports the paired per-example
judge_correct difference (method - reference) with a 95% CI; a CI that excludes 0
means the accuracy gap is significant rather than judge noise.

Usage:
    PYTHONPATH=. python scripts/compare_methods_ci.py \
        runs/no_pruning_judged.json runs/clip_safecrop_judged.json \
        runs/downscale_baseline_judged.json runs/clip_safecrop_downscale_judged.json \
        --reference downscale_baseline --out data/mmdocrag/analysis/methods_ci.json
"""
import argparse
import json
from pathlib import Path

from rag.metrics import bootstrap_ci, paired_diff_ci


def load_method(path: str):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    name = data.get("summary", {}).get("pruning_mode") or Path(path).stem
    by_qid = {}
    for r in data["rows"]:
        qid = r.get("q_id")
        m = r.get("metrics", {})
        if qid is None or "judge_correct" not in m:
            continue
        by_qid[qid] = {
            "judge_correct": m["judge_correct"],
            "tokens_after": (r.get("visual_tokens") or {}).get("after"),
            "total_sec": (r.get("timing") or {}).get("total_sec"),
        }
    return name, by_qid


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("results", nargs="+", help="judged JSON files, one per method")
    ap.add_argument("--reference", default="downscale_baseline",
                    help="method name to compute paired differences against")
    ap.add_argument("--out", default="data/mmdocrag/analysis/methods_ci.json")
    args = ap.parse_args()

    methods = {}
    for path in args.results:
        name, by_qid = load_method(path)
        methods[name] = by_qid

    report = {"per_method": {}, "paired_vs_reference": {}, "reference": args.reference}

    print(f"{'method':<26} {'n':>4} {'correct[95% CI]':>22} {'tok_after':>10} {'sec':>6}")
    for name, by_qid in methods.items():
        jc = [v["judge_correct"] for v in by_qid.values()]
        toks = [v["tokens_after"] for v in by_qid.values() if v["tokens_after"] is not None]
        secs = [v["total_sec"] for v in by_qid.values() if v["total_sec"] is not None]
        mean, lo, hi = bootstrap_ci(jc)
        avg_tok = sum(toks) / len(toks) if toks else None
        avg_sec = sum(secs) / len(secs) if secs else None
        report["per_method"][name] = {
            "n": len(jc), "judge_correct": mean, "ci_low": lo, "ci_high": hi,
            "avg_tokens_after": avg_tok, "avg_total_sec": avg_sec,
        }
        ci = f"{mean:.3f} [{lo:.3f},{hi:.3f}]" if mean is not None else "NA"
        print(f"{name:<26} {len(jc):>4} {ci:>22} "
              f"{(avg_tok if avg_tok is not None else float('nan')):>10.0f} "
              f"{(avg_sec if avg_sec is not None else float('nan')):>6.1f}")

    ref = methods.get(args.reference)
    if ref is not None:
        print(f"\nPaired judge_correct difference vs {args.reference} (95% CI; excludes 0 = significant):")
        for name, by_qid in methods.items():
            if name == args.reference:
                continue
            shared = sorted(set(by_qid) & set(ref))
            if not shared:
                print(f"  {name:<26} no shared q_ids")
                continue
            a = [by_qid[q]["judge_correct"] for q in shared]
            b = [ref[q]["judge_correct"] for q in shared]
            mean, lo, hi = paired_diff_ci(a, b)
            sig = "" if (lo <= 0 <= hi) else "  *SIGNIFICANT*"
            report["paired_vs_reference"][name] = {
                "n_paired": len(shared), "mean_diff": mean, "ci_low": lo, "ci_high": hi,
                "significant": not (lo <= 0 <= hi),
            }
            print(f"  {name:<26} d={mean:+.3f} [{lo:+.3f},{hi:+.3f}] (n={len(shared)}){sig}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
