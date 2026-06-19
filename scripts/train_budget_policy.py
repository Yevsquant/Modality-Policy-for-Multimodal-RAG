"""Phase 2: train + rigorously evaluate the learned query-conditioned budget policy.

No VLM server needed. Phase 0 already recorded, per (example, keep), the answer model's
score AND token count (`data/vqa_stress/vstar_downscale.jsonl`). We:

  1. Extract cheap CLIP features per (image, question) [rag/budget_features].
  2. For each budget k in the ladder, fit a logistic regression predicting
     P(correct @ k) from the features. Out-of-fold via leave-one-out (n=191 is small),
     so every example gets a held-out prediction for every budget.
  3. Build the policy: pick the cheapest budget whose held-out P(correct) >= threshold;
     sweep the threshold to trace the (accuracy vs avg-tokens) frontier.
  4. Compare that frontier against (a) the static uniform-downscale frontier and
     (b) the oracle upper bound; bootstrap + paired CIs at a matched operating point.

Usage:
    PYTHONPATH=. python scripts/train_budget_policy.py \
        --stress data/vqa_stress/vstar_downscale.jsonl \
        --out data/vqa_stress/vstar_policy.json \
        --plot imgs/Phase2LearnedBudgetFrontier.png
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

import numpy as np

from rag.budget_features import (
    SPATIAL_FEATURE_NAMES,
    ClipFeaturizer,
    SpatialClipFeaturizer,
)
from rag.budget_policy import (
    oracle_budget,
    oracle_frontier_point,
    policy_per_example,
    static_frontier,
)
from rag.metrics import bootstrap_ci, paired_diff_ci
from rag.vqa_datasets import load_dataset_by_name, load_vstar


def load_stress(path: str):
    """-> (ids, keeps_cheap_first, scores[id][k], tokens[id][k])."""
    scores: Dict[str, Dict[float, float]] = defaultdict(dict)
    tokens: Dict[str, Dict[float, float]] = defaultdict(dict)
    keeps = set()
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            scores[r["id"]][r["keep"]] = float(r["score"])
            tokens[r["id"]][r["keep"]] = float(r["tokens"])
            keeps.add(r["keep"])
    keeps_cheap_first = sorted(keeps)
    ids = [i for i in scores if len(scores[i]) == len(keeps_cheap_first)]
    ids.sort()
    return ids, keeps_cheap_first, scores, tokens


def out_of_fold_probs(
    X: np.ndarray, y_by_keep: Dict[float, np.ndarray], keeps: List[float]
) -> Dict[float, np.ndarray]:
    """Leave-one-out OOF P(correct@k) for each budget.

    Degenerate budgets (every example correct, or every example wrong in the training
    fold) get the constant base rate — logistic regression can't fit a single class.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import LeaveOneOut
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    n = X.shape[0]
    loo = LeaveOneOut()
    oof: Dict[float, np.ndarray] = {k: np.zeros(n) for k in keeps}
    for k in keeps:
        y = y_by_keep[k]
        for tr, te in loo.split(X):
            ytr = y[tr]
            if len(np.unique(ytr)) < 2:
                oof[k][te] = float(ytr.mean())  # base rate fallback
                continue
            clf = make_pipeline(
                StandardScaler(),
                LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
            )
            clf.fit(X[tr], ytr)
            oof[k][te] = clf.predict_proba(X[te])[:, 1]
    return oof


def out_of_fold_auc(oof: Dict[float, np.ndarray], y_by_keep, keeps):
    """OOF ROC-AUC per budget — the decisive "can features predict survival?" number.

    Degenerate budgets (one class only) have undefined AUC -> reported as None.
    """
    from sklearn.metrics import roc_auc_score

    out = {}
    for k in keeps:
        y = y_by_keep[k]
        base = float(y.mean())
        if len(np.unique(y)) < 2:
            out[k] = {"auc": None, "base_rate": base}
        else:
            out[k] = {"auc": float(roc_auc_score(y, oof[k])), "base_rate": base}
    return out


def ci_str(triple):
    m, lo, hi = triple
    return f"{m:.3f} [{lo:.3f}, {hi:.3f}]"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stress", default="data/vqa_stress/vstar_downscale.jsonl")
    ap.add_argument("--out", default="data/vqa_stress/vstar_policy.json")
    ap.add_argument("--plot", default="imgs/Phase2LearnedBudgetFrontier.png")
    ap.add_argument("--cache-features", default="data/vqa_stress/vstar_clip_feats.npz")
    ap.add_argument(
        "--featurizer", choices=["pooled", "spatial", "both"], default="pooled",
        help="pooled=Phase2 CLIP img+txt+cos; spatial=Phase4 patch-grid relevance "
             "+ detail + resolution; both=concat.",
    )
    ap.add_argument(
        "--dataset", default="vstar",
        help="dataset name for resolving image paths/questions (vstar|docvqa|hrbench).",
    )
    args = ap.parse_args()

    ids, keeps, scores, tokens = load_stress(args.stress)
    print(f"[data] {len(ids)} examples, keeps (cheap->expensive) = {keeps}")
    print(f"[featurizer] {args.featurizer}  dataset={args.dataset}")

    # --- features (cached; CLIP forward is the only GPU work and it's tiny) ---
    cache = Path(args.cache_features)
    recs = {r["id"]: r for r in load_dataset_by_name(args.dataset)}
    if cache.exists():
        npz = np.load(cache, allow_pickle=True)
        feat_ids = list(npz["ids"])
        X_all = npz["X"]
        feat_map = {i: X_all[j] for j, i in enumerate(feat_ids)}
        print(f"[features] loaded {len(feat_ids)} cached from {cache}")
    else:
        present = [i for i in ids if i in recs]
        paths = [recs[i]["image_path"] for i in present]
        qs = [recs[i]["question"] for i in present]
        print(f"[features] extracting {args.featurizer} CLIP for {len(present)} ...")
        parts = []
        if args.featurizer in ("pooled", "both"):
            parts.append(ClipFeaturizer().features(paths, qs))
        if args.featurizer in ("spatial", "both"):
            parts.append(SpatialClipFeaturizer().features(paths, qs))
        X = np.concatenate(parts, axis=1).astype("float32")
        feat_map = {i: X[j] for j, i in enumerate(present)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ids=np.array(present), X=X)
        print(f"[features] cached -> {cache}  dim={X.shape[1]}")

    ids = [i for i in ids if i in feat_map]
    X = np.stack([feat_map[i] for i in ids]).astype("float32")
    n = len(ids)
    print(f"[features] X = {X.shape}")

    y_by_keep = {k: np.array([scores[i][k] for i in ids]) for k in keeps}
    oof = out_of_fold_probs(X, y_by_keep, keeps)
    auc = out_of_fold_auc(oof, y_by_keep, keeps)
    print("\n=== Out-of-fold AUC per budget (does it beat ~0.50?) ===")
    for k in keeps:
        a = auc[k]["auc"]
        print(f"  keep={k:<4} base_rate={auc[k]['base_rate']:.3f}  "
              f"AUC={'n/a' if a is None else f'{a:.3f}'}")

    # --- assemble per-example records for the policy logic ---
    examples = [
        {
            "id": i,
            "scores": {k: scores[i][k] for k in keeps},
            "tokens": {k: tokens[i][k] for k in keeps},
            "probs": {k: float(oof[k][j]) for k in keeps},
        }
        for j, i in enumerate(ids)
    ]

    # --- static baseline frontier (per-keep) ---
    keeps_desc = sorted(keeps, reverse=True)
    static = static_frontier(examples, keeps)  # [(keep, acc, tok)]
    oracle_acc, oracle_tok = oracle_frontier_point(examples, keeps)

    # --- learned policy frontier: sweep thresholds ---
    thresholds = [round(t, 3) for t in np.linspace(0.0, 1.0, 51)]
    pol_points = []
    per_thr_scores = {}
    per_thr_tokens = {}
    for t in thresholds:
        s, tk = policy_per_example(examples, keeps, t)
        per_thr_scores[t] = s
        per_thr_tokens[t] = tk
        acc_m, acc_lo, acc_hi = bootstrap_ci(s)
        tok_m = float(np.mean(tk))
        pol_points.append(
            {"threshold": t, "acc": acc_m, "acc_ci_low": acc_lo,
             "acc_ci_high": acc_hi, "tokens": tok_m}
        )

    # --- headline paired comparison: matched-accuracy operating point ---
    # Static "full res" accuracy = the ceiling of plain downscaling without losing acc.
    # Find the policy threshold whose mean acc is closest to (>=) the best static acc
    # at the LOWEST tokens, and pair-test tokens saved. Also compare at matched tokens.
    static_by_keep = {k: (a, tk) for k, a, tk in static}
    full_keep = keeps_desc[0]
    full_acc = static_by_keep[full_keep][0]
    full_tok = static_by_keep[full_keep][1]

    # Operating point A: highest-accuracy policy point, compare to full-res static.
    best_acc_pt = max(pol_points, key=lambda p: (p["acc"], -p["tokens"]))
    tA = best_acc_pt["threshold"]
    polA_scores, polA_tokens = per_thr_scores[tA], per_thr_tokens[tA]
    full_scores = [examples[j]["scores"][full_keep] for j in range(n)]
    full_tokens = [examples[j]["tokens"][full_keep] for j in range(n)]
    accA_diff = paired_diff_ci(polA_scores, full_scores)
    tokA_diff = paired_diff_ci(polA_tokens, full_tokens)

    # Operating point B: policy point near a static keep's token budget, compare acc.
    # Use the static keep with the largest significant drop region (0.2) as a foil.
    foil_keep = 0.2 if 0.2 in keeps else keeps[1]
    foil_acc, foil_tok = static_by_keep[foil_keep]
    # nearest policy point by tokens to the foil's token budget
    nearest = min(pol_points, key=lambda p: abs(p["tokens"] - foil_tok))
    tB = nearest["threshold"]
    polB_scores, polB_tokens = per_thr_scores[tB], per_thr_tokens[tB]
    foil_scores = [examples[j]["scores"][foil_keep] for j in range(n)]
    foil_tokens = [examples[j]["tokens"][foil_keep] for j in range(n)]
    accB_diff = paired_diff_ci(polB_scores, foil_scores)
    tokB_diff = paired_diff_ci(polB_tokens, foil_tokens)

    # oracle budget distribution (sanity vs vstar_gate.json)
    obud = [oracle_budget(keeps_desc, examples[j]["scores"]) for j in range(n)]
    from collections import Counter
    obud_dist = {str(k): v for k, v in sorted(
        Counter(obud).items(), key=lambda kv: (kv[0] is None, kv[0]))}

    report = {
        "stress_file": args.stress,
        "featurizer": args.featurizer,
        "dataset": args.dataset,
        "n": n,
        "feature_dim": int(X.shape[1]),
        "keeps_cheap_first": keeps,
        "oof_auc_per_budget": {
            str(k): auc[k] for k in keeps
        },
        "static_frontier": [
            {"keep": k, "acc": a, "tokens": tk} for k, a, tk in static
        ],
        "oracle_point": {"acc": oracle_acc, "tokens": oracle_tok},
        "policy_frontier": pol_points,
        "oracle_budget_distribution": obud_dist,
        "operating_point_A_full_res": {
            "policy_threshold": tA,
            "policy_acc": best_acc_pt["acc"],
            "policy_tokens": best_acc_pt["tokens"],
            "static_full_acc": full_acc,
            "static_full_tokens": full_tok,
            "acc_diff_policy_minus_full": dict(zip(("mean", "lo", "hi"), accA_diff)),
            "token_diff_policy_minus_full": dict(zip(("mean", "lo", "hi"), tokA_diff)),
        },
        "operating_point_B_matched_tokens": {
            "foil_static_keep": foil_keep,
            "policy_threshold": tB,
            "policy_acc": nearest["acc"],
            "policy_tokens": nearest["tokens"],
            "static_foil_acc": foil_acc,
            "static_foil_tokens": foil_tok,
            "acc_diff_policy_minus_static": dict(zip(("mean", "lo", "hi"), accB_diff)),
            "token_diff_policy_minus_static": dict(zip(("mean", "lo", "hi"), tokB_diff)),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    # --- console summary ---
    print("\n=== Static uniform-downscale frontier ===")
    for k, a, tk in static:
        print(f"  keep={k:<4} acc={a:.3f}  tokens={tk:6.0f}")
    print(f"  ORACLE upper bound: acc={oracle_acc:.3f}  tokens={oracle_tok:6.0f}")
    print("\n=== Learned policy frontier (threshold sweep) ===")
    for p in pol_points[:: max(1, len(pol_points) // 12)]:
        print(f"  thr={p['threshold']:.2f}  acc={p['acc']:.3f}"
              f" [{p['acc_ci_low']:.3f},{p['acc_ci_high']:.3f}]  tokens={p['tokens']:6.0f}")
    print("\n=== Operating point A: best-accuracy policy vs full-res static ===")
    print(f"  policy thr={tA:.2f}: acc={best_acc_pt['acc']:.3f}, tokens={best_acc_pt['tokens']:.0f}")
    print(f"  static full:        acc={full_acc:.3f}, tokens={full_tok:.0f}")
    print(f"  acc  Δ(policy-full): {ci_str(accA_diff)}")
    print(f"  token Δ(policy-full): {accA_diff and ci_str(tokA_diff)}")
    print("\n=== Operating point B: policy vs static keep=%.1f at matched tokens ===" % foil_keep)
    print(f"  policy thr={tB:.2f}: acc={nearest['acc']:.3f}, tokens={nearest['tokens']:.0f}")
    print(f"  static k={foil_keep}:   acc={foil_acc:.3f}, tokens={foil_tok:.0f}")
    print(f"  acc  Δ(policy-static): {ci_str(accB_diff)}")
    print(f"  token Δ(policy-static): {ci_str(tokB_diff)}")
    print(f"\n[written] {args.out}")

    # --- plot ---
    try:
        make_plot(static, (oracle_acc, oracle_tok), pol_points, args.plot)
        print(f"[plot] {args.plot}")
    except Exception as e:  # plotting is non-essential
        print(f"[plot] skipped: {e}")


def make_plot(static, oracle_pt, pol_points, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    st = sorted(static, key=lambda x: x[2])
    ax.plot([t for _, _, t in st], [a for _, a, _ in st], "o-",
            color="tab:gray", label="static uniform downscale")
    for k, a, t in st:
        ax.annotate(f"k={k}", (t, a), textcoords="offset points", xytext=(4, -10),
                    fontsize=8, color="tab:gray")
    pp = sorted(pol_points, key=lambda p: p["tokens"])
    ax.plot([p["tokens"] for p in pp], [p["acc"] for p in pp], "-",
            color="tab:blue", label="learned policy (threshold sweep)")
    ax.scatter([oracle_pt[1]], [oracle_pt[0]], marker="*", s=180,
               color="tab:green", zorder=5, label="oracle upper bound")
    ax.set_xlabel("avg visual tokens")
    ax.set_ylabel("accuracy")
    ax.set_title("V*Bench: learned budget policy vs static downscale")
    ax.legend()
    ax.grid(True, alpha=0.3)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=130)


if __name__ == "__main__":
    main()
