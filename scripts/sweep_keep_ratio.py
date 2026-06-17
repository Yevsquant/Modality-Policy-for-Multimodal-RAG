"""Sweep keep_ratio for clip_safecrop vs the downscale_baseline and plot the
real-token-vs-accuracy tradeoff curve.

For each keep_ratio and each mode this runs the online benchmark + offline judge
and records (avg_visual_tokens_after, avg_judge_correct, avg_total_sec) from the
judged summary. Results go to data/mmdocrag/analysis/keep_ratio_sweep.json and a
curve to imgs/TokenVsAccuracy.png.

Requires a running vLLM server (same as run_mmdocrag_benchmark.py).

Usage:
    PYTHONPATH=. python scripts/sweep_keep_ratio.py \
        --eval-slice-start 0 --eval-slice-stop 50
"""
import argparse
import json
from pathlib import Path

DEFAULT_KEEP_RATIOS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]
MODES = ["clip_safecrop", "downscale_baseline"]
SWEEP_PATH = Path("data/mmdocrag/analysis/keep_ratio_sweep.json")
PLOT_PATH = Path("imgs/TokenVsAccuracy.png")


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--eval-slice-start", type=int, default=0)
    p.add_argument("--eval-slice-stop", type=int, default=None)
    p.add_argument("--max-examples", type=int, default=0, help="0 = no cap")
    p.add_argument(
        "--keep-ratios",
        type=float,
        nargs="+",
        default=DEFAULT_KEEP_RATIOS,
    )
    p.add_argument("--modes", nargs="+", default=MODES)
    p.add_argument("--plot-only", action="store_true",
                   help="Skip benchmarking; just re-plot from the saved sweep JSON.")
    return p.parse_args()


def run_sweep(args):
    from rag.config import RAGConfig
    from rag.eval import run_rag_benchmark, run_rag_benchmark_offline_judge

    results = []
    for mode in args.modes:
        for keep_ratio in args.keep_ratios:
            cfg = RAGConfig()
            cfg.pruning_mode = mode
            cfg.pruning_keep_ratio = keep_ratio
            cfg.eval_slice_start = args.eval_slice_start
            cfg.eval_slice_stop = args.eval_slice_stop
            cfg.max_examples = None if args.max_examples == 0 else args.max_examples
            # Disable the reuse cache so every sweep point is measured fresh and
            # the two modes don't share stale pruned artifacts.
            cfg.image_prune_cache_enabled = False

            print(f"\n=== mode={mode} keep_ratio={keep_ratio} ===")
            run_rag_benchmark(cfg)
            judged = run_rag_benchmark_offline_judge(cfg)
            summary = judged["summary"]
            results.append({
                "mode": mode,
                "keep_ratio": keep_ratio,
                "avg_visual_tokens_after": summary.get("avg_visual_tokens_after"),
                "avg_visual_tokens_before": summary.get("avg_visual_tokens_before"),
                "avg_visual_tokens_reduction_pct": summary.get("avg_visual_tokens_reduction_pct"),
                "avg_judge_correct": summary.get("avg_judge_correct"),
                "avg_judge_score": summary.get("avg_judge_score"),
                "avg_total_sec": summary.get("avg_total_sec"),
            })

    SWEEP_PATH.parent.mkdir(parents=True, exist_ok=True)
    SWEEP_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nwrote {SWEEP_PATH}")
    return results


def plot(results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for mode in sorted({r["mode"] for r in results}):
        pts = [r for r in results
               if r["mode"] == mode
               and r["avg_visual_tokens_after"] is not None
               and r["avg_judge_correct"] is not None]
        pts.sort(key=lambda r: r["avg_visual_tokens_after"])
        if not pts:
            continue
        xs = [r["avg_visual_tokens_after"] for r in pts]
        ys = [r["avg_judge_correct"] for r in pts]
        ax.plot(xs, ys, marker="o", label=mode)
        for r in pts:
            ax.annotate(f"{r['keep_ratio']}",
                        (r["avg_visual_tokens_after"], r["avg_judge_correct"]),
                        fontsize=7, textcoords="offset points", xytext=(4, 4))

    ax.set_xlabel("Avg visual tokens after (target model)")
    ax.set_ylabel("Avg judge_correct")
    ax.set_title("Real visual tokens vs accuracy (keep_ratio sweep)")
    ax.grid(True)
    ax.legend()
    PLOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, bbox_inches="tight")
    print(f"wrote {PLOT_PATH}")


def main():
    args = parse_args()
    if args.plot_only:
        results = json.loads(SWEEP_PATH.read_text(encoding="utf-8"))
    else:
        results = run_sweep(args)
    plot(results)


if __name__ == "__main__":
    main()
