"""Plot average visual tokens before/after pruning, per method.

Reads one or more `baseline_results_judged.json` files (each carries a
`summary` with `pruning_mode`, `avg_visual_tokens_before`, and
`avg_visual_tokens_after`) and writes a grouped bar chart.

Usage:
    PYTHONPATH=. python scripts/plot_visual_tokens.py \
        data/mmdocrag/outputs/baseline_results_judged.json \
        --out imgs/VisualTokensByMethods.png
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "results", nargs="+", help="One or more baseline_results_judged.json files."
    )
    parser.add_argument("--out", default="imgs/VisualTokensByMethods.png")
    args = parser.parse_args()

    methods, before, after = [], [], []
    for path in args.results:
        summary = json.loads(Path(path).read_text(encoding="utf-8")).get("summary", {})
        if "avg_visual_tokens_before" not in summary:
            print(f"skip {path}: no avg_visual_tokens_before in summary")
            continue
        label = summary.get("pruning_mode", Path(path).stem)
        ratio = summary.get("pruning_keep_ratio")
        if ratio is not None:
            label = f"{label}\n(keep={ratio})"
        methods.append(label)
        before.append(summary["avg_visual_tokens_before"])
        after.append(summary["avg_visual_tokens_after"])

    if not methods:
        raise SystemExit("No usable summaries with visual-token fields found.")

    x = np.arange(len(methods))
    width = 0.38
    fig, ax = plt.subplots(figsize=(max(6, 2 * len(methods)), 5))
    ax.bar(x - width / 2, before, width, label="before", color="#9aa0a6")
    ax.bar(x + width / 2, after, width, label="after", color="#1a73e8")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Avg visual tokens (target model)")
    ax.set_title("Visual tokens before vs after pruning, by method")
    ax.legend()
    for i, (b, a) in enumerate(zip(before, after)):
        ax.text(i - width / 2, b, f"{b:.0f}", ha="center", va="bottom", fontsize=8)
        ax.text(i + width / 2, a, f"{a:.0f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
