"""Phase 4 (Goal B): retrain the budget policy on V*Bench + HR-Bench COMBINED.

The per-budget classifier predicts P(correct@keep_k) from CLIP features; the keep
ladder {0.1,0.2,0.3,0.5,1.0} is shared across datasets, and the features are
dataset-agnostic, so we can POOL training examples from both datasets to give the
head more data. We then evaluate the learned frontier on V*Bench — the only substrate
with real oracle headroom (HR-Bench's static frontier is nearly flat). The question:
does the extra HR-Bench training data let the policy finally beat static downscaling
on V*Bench?

Out-of-fold predictions on the V*Bench examples come from classifiers trained on
{all HR-Bench} + {V*Bench minus this fold} (5-fold over V*Bench). This is an honest
held-out evaluation: no V*Bench example is in its own training set.

Usage:
    PYTHONPATH=. python scripts/train_budget_policy_combined.py \
        --featurizer spatial --out data/vqa_stress/combined_policy_spatial.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rag.budget_policy import (
    oracle_budget,
    oracle_frontier_point,
    policy_per_example,
    static_frontier,
)
from rag.metrics import bootstrap_ci, paired_diff_ci
from scripts.train_budget_policy import load_stress, out_of_fold_auc


def feats_for(dataset, featurizer, stress, cache):
    """Load (ids, keeps, scores, tokens, X) for one dataset, reusing cached feats."""
    from rag.budget_features import ClipFeaturizer, SpatialClipFeaturizer
    from rag.vqa_datasets import load_dataset_by_name

    ids, keeps, scores, tokens = load_stress(stress)
    cache = Path(cache)
    if cache.exists():
        npz = np.load(cache, allow_pickle=True)
        feat_map = {i: npz["X"][j] for j, i in enumerate(list(npz["ids"]))}
    else:
        recs = {r["id"]: r for r in load_dataset_by_name(dataset)}
        present = [i for i in ids if i in recs]
        paths = [recs[i]["image_path"] for i in present]
        qs = [recs[i]["question"] for i in present]
        parts = []
        if featurizer in ("pooled", "both"):
            parts.append(ClipFeaturizer().features(paths, qs))
        if featurizer in ("spatial", "both"):
            parts.append(SpatialClipFeaturizer().features(paths, qs))
        X = np.concatenate(parts, axis=1).astype("float32")
        feat_map = {i: X[j] for j, i in enumerate(present)}
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, ids=np.array(present), X=X)
    ids = [i for i in ids if i in feat_map]
    X = np.stack([feat_map[i] for i in ids]).astype("float32")
    return ids, keeps, scores, tokens, X


def fit_predict(Xtr, ytr, Xte):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    if len(np.unique(ytr)) < 2:
        return np.full(Xte.shape[0], float(ytr.mean()))
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, C=1.0, class_weight="balanced"),
    )
    clf.fit(Xtr, ytr)
    return clf.predict_proba(Xte)[:, 1]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--featurizer", choices=["pooled", "spatial", "both"], default="spatial")
    ap.add_argument("--eval-dataset", default="vstar")
    ap.add_argument("--out", default="data/vqa_stress/combined_policy.json")
    args = ap.parse_args()

    suffix = {"pooled": "pooled", "spatial": "spatial", "both": "both"}[args.featurizer]
    v_ids, keeps, v_scores, v_tokens, Xv = feats_for(
        "vstar", args.featurizer, "data/vqa_stress/vstar_downscale.jsonl",
        f"data/vqa_stress/vstar_{suffix}_feats.npz")
    h_ids, hkeeps, h_scores, h_tokens, Xh = feats_for(
        "hrbench", args.featurizer, "data/vqa_stress/hrbench_downscale.jsonl",
        f"data/vqa_stress/hrbench_{suffix}_feats.npz")
    assert keeps == hkeeps, (keeps, hkeeps)
    print(f"[combined] vstar={len(v_ids)} hrbench={len(h_ids)} feat_dim={Xv.shape[1]} "
          f"featurizer={args.featurizer}")

    # Labels per budget.
    yv = {k: np.array([v_scores[i][k] for i in v_ids]) for k in keeps}
    yh = {k: np.array([h_scores[i][k] for i in h_ids]) for k in keeps}

    # 5-fold OOF on V*Bench; each training fold ALSO includes all HR-Bench rows.
    from sklearn.model_selection import KFold

    n = len(v_ids)
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    oof = {k: np.zeros(n) for k in keeps}
    for k in keeps:
        for tr, te in kf.split(Xv):
            Xtr = np.concatenate([Xv[tr], Xh], axis=0)
            ytr = np.concatenate([yv[k][tr], yh[k]], axis=0)
            oof[k][te] = fit_predict(Xtr, ytr, Xv[te])

    auc = out_of_fold_auc(oof, yv, keeps)
    print("\n=== OOF AUC on V*Bench (trained on V*Bench-fold + ALL HR-Bench) ===")
    for k in keeps:
        a = auc[k]["auc"]
        print(f"  keep={k:<4} base={auc[k]['base_rate']:.3f}  "
              f"AUC={'n/a' if a is None else f'{a:.3f}'}")

    examples = [
        {"id": i, "scores": {k: v_scores[i][k] for k in keeps},
         "tokens": {k: v_tokens[i][k] for k in keeps},
         "probs": {k: float(oof[k][j]) for k in keeps}}
        for j, i in enumerate(v_ids)
    ]
    static = static_frontier(examples, keeps)
    oracle_acc, oracle_tok = oracle_frontier_point(examples, keeps)
    static_by_keep = {k: (a, tk) for k, a, tk in static}

    # Frontier + matched-token paired test vs static keep=0.2 (the Phase 2 foil).
    thresholds = [round(t, 3) for t in np.linspace(0.0, 1.0, 51)]
    pol_points, per_s, per_t = [], {}, {}
    for t in thresholds:
        s, tk = policy_per_example(examples, keeps, t)
        per_s[t], per_t[t] = s, tk
        m, lo, hi = bootstrap_ci(s)
        pol_points.append({"threshold": t, "acc": m, "tokens": float(np.mean(tk))})

    foil = 0.2
    foil_acc, foil_tok = static_by_keep[foil]
    nearest = min(pol_points, key=lambda p: abs(p["tokens"] - foil_tok))
    tB = nearest["threshold"]
    foil_scores = [examples[j]["scores"][foil] for j in range(n)]
    accB = paired_diff_ci(per_s[tB], foil_scores)

    report = {
        "featurizer": args.featurizer, "eval_dataset": args.eval_dataset,
        "n_vstar": n, "n_hrbench": len(h_ids), "feat_dim": int(Xv.shape[1]),
        "oof_auc_per_budget": {str(k): auc[k] for k in keeps},
        "static_frontier": [{"keep": k, "acc": a, "tokens": tk} for k, a, tk in static],
        "oracle_point": {"acc": oracle_acc, "tokens": oracle_tok},
        "policy_frontier": pol_points,
        "matched_tokens_vs_static_keep0.2": {
            "policy_threshold": tB, "policy_acc": nearest["acc"],
            "policy_tokens": nearest["tokens"],
            "static_acc": foil_acc, "static_tokens": foil_tok,
            "acc_diff_policy_minus_static": dict(zip(("mean", "lo", "hi"), accB)),
        },
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(report, f, indent=2)

    print("\n=== V*Bench frontier (policy trained on combined data) ===")
    for k, a, tk in static:
        print(f"  static keep={k:<4} acc={a:.3f} tokens={tk:6.0f}")
    print(f"  ORACLE acc={oracle_acc:.3f} tokens={oracle_tok:6.0f}")
    print(f"\n  matched-token (~{foil_tok:.0f}) policy thr={tB:.2f}: "
          f"acc={nearest['acc']:.3f} tokens={nearest['tokens']:.0f}")
    print(f"  vs static keep=0.2 acc={foil_acc:.3f}: "
          f"Δ={accB[0]:+.3f} [{accB[1]:+.3f},{accB[2]:+.3f}]")
    print(f"[written] {args.out}")


if __name__ == "__main__":
    main()
