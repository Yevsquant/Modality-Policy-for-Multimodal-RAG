"""Run the benchmark + offline judge for one pruning mode and save its judged
JSON (with per-row data) to a chosen path. The prune cache is disabled by default
so cross-mode runs never reuse another mode's crops.

Usage:
    PYTHONPATH=. python scripts/run_method.py --mode clip_safecrop_downscale \
        --keep-ratio 0.3 --slice-start 0 --slice-stop 300 \
        --out data/mmdocrag/analysis/runs/clip_safecrop_downscale_judged.json
"""
import argparse
import shutil
from pathlib import Path

from rag.config import RAGConfig
from rag.eval import _judged_path, run_rag_benchmark, run_rag_benchmark_offline_judge


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", required=True)
    ap.add_argument("--keep-ratio", type=float, default=0.3)
    ap.add_argument("--slice-start", type=int, default=0)
    ap.add_argument("--slice-stop", type=int, default=300)
    ap.add_argument("--max-examples", type=int, default=0, help="0 = no cap")
    ap.add_argument("--cache", action="store_true", help="enable prune cache (off by default)")
    ap.add_argument("--out", required=True, help="destination path for the judged JSON")
    args = ap.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"[skip] {out} already exists ({args.mode}) — resume")
        return

    work_dir = Path("data/mmdocrag/analysis/runs") / f"_work_{args.mode}_{int(args.keep_ratio*100)}"
    cfg = RAGConfig(
        pruning_mode=args.mode,
        pruning_keep_ratio=args.keep_ratio,
        eval_slice_start=args.slice_start,
        eval_slice_stop=args.slice_stop,
        max_examples=None if args.max_examples == 0 else args.max_examples,
        image_prune_cache_enabled=args.cache,
        output_dir=work_dir,
    )
    print(f"[run] mode={args.mode} keep={args.keep_ratio} slice=[{args.slice_start},{args.slice_stop})")
    run_rag_benchmark(cfg)
    run_rag_benchmark_offline_judge(cfg)

    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(_judged_path(cfg), out)
    print(f"[done] wrote {out}")


if __name__ == "__main__":
    main()
