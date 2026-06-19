# Work Report: Phase 2 — Learned Query-Conditioned Budget Policy (V*Bench)

**Date:** 2026-06-19
**Branch:** `clip-safecrop-token-measurement`
**Plan:** [thoughts/plans/2026-06-19-adaptive-visual-tokenization-repositioning.md](../plans/2026-06-19-adaptive-visual-tokenization-repositioning.md) (Q3, output-space option 1: global scalar budget)
**Goal:** Train a small query-conditioned policy that predicts, per (image, question),
the smallest downscale `keep_ratio` that still answers correctly — spending visual
tokens only where the query needs resolution — and test whether its accuracy/token
frontier beats static uniform downscaling.

## Bottom line (honest, negative)

**The learned policy does NOT beat static downscaling. It is significantly worse at a
matched token budget.** The cause is upstream of the model: **CLIP image+text features
carry essentially no signal about per-example downscale sensitivity** (out-of-fold AUC
≈ 0.50 at every budget). The oracle is far above any static point (0.942 acc at 868
tokens vs the static curve's 0.770 at 1027), so the *opportunity* is real and large —
but cheap CLIP features cannot find it. This is a clean feature-informativeness ceiling,
not a modeling bug: it replicates across LOO and 5-fold CV and across logistic
regression and a small MLP.

## Method (no VLM calls — evaluation is free)

Phase 0 already recorded, per (example, keep), the answer model's (Qwen3-Omni-30B)
score AND token count for keep ∈ {1.0, 0.5, 0.3, 0.2, 0.1}, all 191 V*Bench examples at
every budget (`data/vqa_stress/vstar_downscale.jsonl`). So a budget *choice* is scored
by *looking up* the recorded (score, tokens) at the chosen keep — no server needed.

- **Features** (`rag/budget_features.py`): the retriever's `openai/clip-vit-base-patch32`.
  Per (image, question): `[ L2-norm CLIP image emb (512) | CLIP text emb (512) |
  cos(img,txt) (1) ] = 1025` dims. Cached to `data/vqa_stress/vstar_clip_feats.npz`.
- **Policy** (`rag/budget_policy.py`, pure logic, unit-tested): for each budget *k*,
  predict P(correct@k); at inference pick the **cheapest** budget whose predicted
  P(correct) ≥ threshold, else fall back to full res. Sweeping the threshold traces the
  (accuracy vs avg-tokens) frontier. Models: per-budget logistic regression
  (StandardScaler + balanced class weights).
- **Evaluation** (`scripts/train_budget_policy.py`): **leave-one-out CV** (n=191 is
  small) gives every example a held-out P(correct@k) for every budget. Frontier compared
  vs (a) the static per-keep frontier and (b) the oracle (each example at its own
  min-sufficient budget). Bootstrap + paired CIs via `rag/metrics.py`.

## Results

### Per-budget learnability — the decisive number

Out-of-fold AUC of CLIP features predicting "still correct at budget *k*":

| keep | base rate | LOO AUC | 5-fold AUC (LR) | 5-fold AUC (MLP) |
|---|---|---|---|---|
| 0.1 | 0.623 | 0.493 | 0.496 | 0.490 |
| 0.2 | 0.723 | 0.476 | 0.474 | 0.501 |
| 0.3 | 0.770 | 0.458 | 0.494 | 0.456 |
| 0.5 | 0.848 | 0.394 | 0.372 | 0.469 |
| 1.0 | 0.885 | 0.477 | 0.497 | 0.477 |

All ≈ 0.5 (chance); several below chance = pure noise. Pearson corr between the CLIP
cos-sim feature and the per-example oracle budget is **−0.066** — no relationship.
The features simply do not encode "this image needs high resolution for this question."

### Frontier comparison (n=191, V*Bench)

| frontier | acc | avg tokens |
|---|---|---|
| static keep=0.1 | 0.623 | 342 |
| static keep=0.2 | 0.723 | 677 |
| static keep=0.3 | 0.770 | 1027 |
| static keep=0.5 | 0.848 | 1702 |
| static keep=1.0 | 0.885 | 3406 |
| **oracle upper bound** | **0.942** | **868** |
| learned policy (best feasible point) | ≤ static at every token budget | — |

The learned-policy curve lies **below** the static curve everywhere (see
`imgs/Phase2LearnedBudgetFrontier.png`).

### Paired test at a matched token budget (the claim to earn — and it fails)

Policy operating point nearest static keep=0.2's token budget (~677):

| | acc | tokens |
|---|---|---|
| learned policy (thr=0.48) | 0.665 | 675 |
| static keep=0.2 | 0.723 | 677 |
| **paired Δ (policy − static)** | **−0.058 [−0.110, −0.005] SIG** | −2 [−92, +99] n.s. |

At equal tokens the policy is **significantly less accurate** than just downscaling
everything uniformly. The highest-accuracy policy point degenerates to threshold→1.0
(full res for everyone), i.e. it learns nothing better than "use the most expensive
budget" — paired Δ vs full-res static = 0.000.

## Why it fails (and why that is interesting, not just disappointing)

- The **gap is not in the framing**: the oracle (0.942 acc @ 868 tokens) crushes every
  static point, so there is large, exploitable budget variance — exactly the fat tail
  Phase 0 found (119/191 need only keep=0.1; ~30% need more). A perfect policy would be
  a major win.
- The **gap is in the features**: CLIP image+text embeddings describe global semantic
  content, not whether the *answer-bearing detail survives a 10×-area downscale*. V*Bench
  is "small object in a large scene"; whether a tiny target is still legible at low res
  depends on object size/contrast/clutter — fine spatial properties CLIP's pooled
  embedding discards. A query-agnostic CLIP image embedding is nearly constant across the
  budget axis, so it cannot separate "robust" from "fragile" examples.
- Consistent with [[vstar-is-the-stress-substrate-trim-doesnt-transfer]] (static
  query-agnostic tricks don't transfer) and [[fastv-composition-negative-interaction]]
  (the two redundancy axes overlap): the lever on detail data is genuinely *where to
  spend resolution*, and locating that needs spatial/object signal the pooled CLIP
  features do not provide.

## What would be needed to turn this positive (not done here)

A budget policy needs features sensitive to *answer-region resolution*, e.g.: object
detector / saliency stats on the queried target; multi-scale CLIP patch features rather
than the pooled embedding; or a cheap "does a low-res crop already answer" probe. That
is output-space option 2/3 territory (crop box + budget, foveated map) — a larger build
than the global-scalar v1 this phase scoped. The honest v1 conclusion: **global-scalar
budgeting on pooled CLIP features is not viable; the signal is real but lives at a
spatial granularity these features throw away.**

## Artifacts

- New code: `rag/budget_policy.py` (pure policy logic), `rag/budget_features.py`
  (CLIP featurizer), `scripts/train_budget_policy.py` (LOO-CV driver + frontier + CIs +
  plot).
- Tests: `tests/test_budget_policy.py` (10), `tests/test_budget_features.py` (1) — green.
- Outputs: `data/vqa_stress/vstar_policy.json` (full frontier + paired CIs),
  `data/vqa_stress/vstar_clip_feats.npz` (cached features),
  `imgs/Phase2LearnedBudgetFrontier.png`.
- Sanity: oracle-budget distribution reproduces `vstar_gate.json`
  (0.1→119, 0.2→26, 0.3→11, 0.5→14, 1.0→10, unsolved→11).
