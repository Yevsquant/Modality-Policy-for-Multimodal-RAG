"""Angle 1 — re-score FastV vs input-downscale on a fair *total compute* axis.

Reads the Phase-3/4 condition results (`data/vqa_stress/fastv_vstar.jsonl`), computes
each condition's total forward FLOPs (vision encoder + LLM, with the FastV early/deep
split) via `rag.flops_model`, and reports accuracy vs total GFLOPs with the Pareto
frontier — does FastV still dominate input downscaling once its full-image encoder cost
is counted? No new model runs; all from existing data.

Usage:
    PYTHONPATH=. python scripts/angle1_flops_reframing.py \
        --in data/vqa_stress/fastv_vstar.jsonl \
        --out data/vqa_stress/angle1_flops.json --plot imgs/Angle1FlopsFrontier.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from rag.flops_model import condition_flops
from rag.metrics import bootstrap_ci

# prune layer K used in the FastV matrix (fastv_layer=3); None for non-FastV conditions
PRUNE_LAYER = {
    "full": None, "ds0.3": None, "ds0.5": None, "ds0.25": None,
    "full+fastv0.5": 3, "full+fastv0.25": 3,
    "ds0.3+fastv0.5": 3, "ds0.3+fastv0.25": 3,
}
GF = 1e9


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", default="data/vqa_stress/fastv_vstar.jsonl")
    ap.add_argument("--out", default="data/vqa_stress/angle1_flops.json")
    ap.add_argument("--plot", default="imgs/Angle1FlopsFrontier.png")
    args = ap.parse_args()

    agg = defaultdict(lambda: {"scores": [], "ib": 0, "ia": 0, "n": 0})
    with open(args.inp) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            a = agg[r["cond"]]
            a["scores"].append(r["score"])
            a["ib"] += r["img_before"]
            a["ia"] += r["img_after"]
            a["n"] += 1

    rows = []
    for cond, a in agg.items():
        if cond not in PRUNE_LAYER:
            continue
        ib = round(a["ib"] / a["n"])
        ia = round(a["ia"] / a["n"])
        total, vision, llm = condition_flops(ib, ia, PRUNE_LAYER[cond])
        acc, lo, hi = bootstrap_ci(a["scores"])
        rows.append({
            "cond": cond, "acc": acc, "ci_low": lo, "ci_high": hi,
            "deep_img_tokens": ia, "input_img_tokens": ib,
            "total_gflops": total / GF, "vision_gflops": vision / GF,
            "llm_gflops": llm / GF, "vision_frac": vision / total,
        })

    # Pareto frontier on (total_gflops asc, acc desc): keep points not dominated
    # (another point with <= flops AND >= acc).
    for r in rows:
        r["pareto"] = not any(
            (o["total_gflops"] <= r["total_gflops"] and o["acc"] > r["acc"]) or
            (o["total_gflops"] < r["total_gflops"] and o["acc"] >= r["acc"])
            for o in rows if o is not r
        )

    rows.sort(key=lambda r: r["total_gflops"])

    # Matched-FLOPs comparison: interpolate the downscale-ONLY frontier at each FastV
    # point's compute budget, to quantify FastV's advantage on the fair axis vs the
    # (inflated) deep-token framing.
    ds_only = sorted([r for r in rows if "fastv" not in r["cond"]],
                     key=lambda r: r["total_gflops"])

    def ds_acc_at(flops):
        if flops <= ds_only[0]["total_gflops"]:
            return ds_only[0]["acc"]
        if flops >= ds_only[-1]["total_gflops"]:
            return ds_only[-1]["acc"]
        for a, b in zip(ds_only, ds_only[1:]):
            if a["total_gflops"] <= flops <= b["total_gflops"]:
                t = (flops - a["total_gflops"]) / (b["total_gflops"] - a["total_gflops"])
                return a["acc"] + t * (b["acc"] - a["acc"])
        return ds_only[-1]["acc"]

    matched = []
    for r in rows:
        if "fastv" in r["cond"] and "ds0.3" not in r["cond"]:  # full+fastv points
            ds = ds_acc_at(r["total_gflops"])
            matched.append({"cond": r["cond"], "gflops": r["total_gflops"],
                            "fastv_acc": r["acc"], "downscale_acc_same_flops": ds,
                            "fastv_advantage_at_matched_flops": r["acc"] - ds})

    out = {"per_condition": rows, "matched_flops_comparison": matched}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)

    print("=== Angle 1: accuracy vs TOTAL compute (Qwen2-VL-7B, V*Bench n=191) ===")
    print(f"{'cond':<18}{'acc':>7}{'deepTok':>9}{'inTok':>7}{'GFLOPs':>9}{'vis%':>6}  frontier")
    for r in rows:
        flag = "  <= PARETO" if r["pareto"] else ""
        print(f"{r['cond']:<18}{r['acc']:>7.3f}{r['deep_img_tokens']:>9}"
              f"{r['input_img_tokens']:>7}{r['total_gflops']:>9.0f}{100*r['vision_frac']:>6.0f}{flag}")

    print("  matched-FLOPs: FastV advantage over downscale at the SAME total compute:")
    for m in matched:
        print(f"    {m['cond']:<16} {m['gflops']:.0f} GFLOPs: FastV {m['fastv_acc']:.3f} "
              f"vs downscale {m['downscale_acc_same_flops']:.3f} -> "
              f"+{m['fastv_advantage_at_matched_flops']:.3f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 5))
        for r in rows:
            fastv = "fastv" in r["cond"]
            ax.scatter(r["total_gflops"], r["acc"],
                       marker="^" if fastv else "o", s=90,
                       color="crimson" if fastv else "steelblue")
            ax.annotate(r["cond"], (r["total_gflops"], r["acc"]),
                        fontsize=7, xytext=(4, 4), textcoords="offset points")
        pareto = sorted([r for r in rows if r["pareto"]], key=lambda r: r["total_gflops"])
        ax.plot([r["total_gflops"] for r in pareto], [r["acc"] for r in pareto],
                "--", color="gray", lw=1, label="Pareto frontier")
        ax.set_xlabel("Total forward GFLOPs (vision + LLM)")
        ax.set_ylabel("V*Bench accuracy")
        ax.set_title("FastV (▲) vs input-downscale (●) on the fair compute axis")
        ax.legend()
        Path(args.plot).parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout(); fig.savefig(args.plot, dpi=130)
        print(f"[plot] {args.plot}")
    except Exception as e:
        print(f"[plot skipped] {e}")
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
