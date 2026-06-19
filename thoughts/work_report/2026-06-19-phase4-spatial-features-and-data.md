# Work Report: Phase 4 — Spatial budget features, more stress data, FastV vs trim

**Date:** 2026-06-19
**Branch:** `clip-safecrop-token-measurement`
**Plan:** [thoughts/plans/2026-06-19-adaptive-visual-tokenization-repositioning.md](../plans/2026-06-19-adaptive-visual-tokenization-repositioning.md)
**Builds on:** Phase 2 ([[phase2-clip-features-cant-predict-budget]]), Phase 3
([[fastv-composition-negative-interaction]]), Phase 0/1
([[vstar-is-the-stress-substrate-trim-doesnt-transfer]]).

Three goals, done in order so only one model is GPU-resident at a time:
- **A.** Spatial CLIP features for the budget policy (no server) — gate.
- **B.** More detail-sensitive stress data (HR-Bench, 30B server) → retrain policy.
- **C.** FastV vs trim-downscale at matched token budget (7B HF, server down).

---

## Goal A — Spatial features for the budget policy (NEGATIVE)

### Hypothesis
Phase 2's global-scalar policy lost to static downscaling because **pooled** CLIP
carries ~0 signal about per-example downscale sensitivity (OOF AUC ≈ 0.50). Hypothesis:
sensitivity is driven by whether a *small query-relevant region* exists that
downscaling destroys — a spatial property the pooled embedding discards.

### Method
New `SpatialClipFeaturizer` in `rag/budget_features.py`: read the per-patch CLIP ViT
tokens (`vision_model.last_hidden_state[:, 1:]`, the 7×7=49 patches at 224px) from the
same `openai/clip-vit-base-patch32`, project them to the shared space, and compute
each patch's cosine to the question's CLIP text embedding → a **spatial query-relevance
map**. From that map, 13 scalar features (pure, unit-tested `spatial_features_from_map`):
peakiness (max, top-1/top-5 softmax mass, entropy), high-relevance region size
(active-fraction, mass-weighted spatial spread), peak distance from center, plus cheap
detail features (`pruner._detail_density` global + around the peak region) and native
resolution (log pixel area, aspect ratio).

Trained the existing per-budget logistic-regression head (`rag/budget_policy.py`,
`scripts/train_budget_policy.py` extended with `--featurizer {pooled,spatial,both}`)
and evaluated **for free** by lookup against Phase 0's recorded score+tokens per
(example, keep) in `data/vqa_stress/vstar_downscale.jsonl` (n=191 V*Bench). LOO CV;
bootstrap/paired CIs via `rag/metrics.py`. Added `out_of_fold_auc` (ROC-AUC per budget).

### Result — spatial features do NOT beat pooled CLIP or the static frontier

Out-of-fold AUC for "still correct at keep k" (chance = 0.50):

| keep | base rate | pooled (Phase 2) | **spatial** | both |
|---|---|---|---|---|
| 0.1 | 0.623 | 0.493 | 0.413 | 0.490 |
| 0.2 | 0.723 | 0.476 | 0.438 | 0.487 |
| 0.3 | 0.770 | 0.458 | **0.532** | 0.462 |
| 0.5 | 0.848 | 0.394 | 0.433 | 0.397 |
| 1.0 | 0.885 | 0.477 | 0.523 | 0.467 |

Spatial nudges two budgets just above chance (keep=0.3: 0.532, keep=1.0: 0.523) but the
rest sit at/below 0.5 — within noise, no usable signal. Concatenating pooled+spatial
("both") does not help.

Frontier (paired, matched-token operating point vs static keep=0.2 ≈ 677 tok):

| featurizer | policy acc @ ~677 tok | static keep=0.2 | paired Δ (policy − static) |
|---|---|---|---|
| pooled (Phase 2) | 0.665 | 0.723 | −0.058 [−0.110, −0.005] **SIG worse** |
| **spatial** | 0.670 @ 767 | 0.723 | −0.052 [−0.105, +0.000] n.s. (worse) |
| both | 0.660 @ 668 | 0.723 | −0.063 [−0.115, −0.016] **SIG worse** |

The learned frontier still tracks the static curve and never beats it. The only
"win" is the trivial free trim at full res (operating point A: −173 tokens at
identical accuracy), which any policy gets.

### Bottom line (A)
**Spatial patch-grid CLIP features do not rescue the budget policy.** Adding per-patch
query-relevance, peakiness, region-size, off-center, and detail-density features moves
OOF AUC from ≈0.48 to ≈0.47 average — still chance. The oracle headroom is real
(0.942 acc @ 868 tok, far above any static point), but even spatial CLIP cannot locate
which examples survive a 10×-area downscale. This is a second, stronger confirmation of
the Phase 2 feature-informativeness ceiling: the lever on detail data is genuinely
*where* to spend resolution, and CLIP-grade features — pooled or per-patch — don't
encode "does the answer-bearing detail stay legible after downscale." Per the brief,
proceeded to Goal B regardless (more data requested).

**Artifacts (A):** `rag/budget_features.py` (`SpatialClipFeaturizer`,
`spatial_features_from_map`, `SPATIAL_FEATURE_NAMES`), `scripts/train_budget_policy.py`
(`--featurizer`/`--dataset`, `out_of_fold_auc`), tests in `tests/test_budget_features.py`
(+2). Outputs: `data/vqa_stress/vstar_policy_spatial.json`, `..._both.json`,
`..._spatial_feats.npz`, `imgs/Phase4SpatialBudgetFrontier.png`.

---

## Goal B — More stress data (HR-Bench) + retrain (NEGATIVE for the policy)

### Dataset choice
HR-Bench unavailable at the brief's guessed repos; the real one is **`DreamMr/HR-Bench`**
(config `hrbench_version_split`, splits `hrbench_4k`/`hrbench_8k`, 800 ex each). Used
**hrbench_4k**: true 4032×4032 4K photos, multiple-choice A–D (base64-JPEG rows, gold
letter). Added `load_hrbench` to `rag/vqa_datasets.py` (materializes images to disk,
formats MC like V*Bench, scored by `vqa_scoring.mc_score`). n=300, keep ladder
{1.0,0.5,0.3,0.2,0.1}, target = Qwen3-Omni-30B via vLLM.

### Downscale stress — gate PASSES but WEAK (DocVQA-like, not V*Bench-like)

| keep | visual tokens | accuracy [95% CI] | paired Δ vs full |
|---|---|---|---|
| 1.0 | 11437 | 0.753 [0.703, 0.800] | — |
| 0.5 |  6832 | 0.733 [0.683, 0.783] | −0.020 n.s. |
| 0.3 |  4110 | 0.730 [0.680, 0.780] | −0.023 n.s. |
| 0.2 |  2709 | 0.713 [0.663, 0.763] | **−0.040 [−0.073, −0.010] SIG** |
| 0.1 |  1380 | 0.697 [0.643, 0.747] | **−0.057 [−0.093, −0.020] SIG** |

The drop is significant only at the two smallest budgets, and shallow: cutting tokens
88% (11437→1380) costs just −0.057. Oracle-budget tail: 0.1→209, 0.2→16, 0.3→4,
0.5→3, 1.0→5, unsolved→63 of 300 — **thin variance** (most need only keep=0.1; 21%
unsolved at any budget). This is the DocVQA pattern, **not** V*Bench's fat tail. Full-res
accuracy itself is low (0.753): the 30B finds 4K HR-Bench questions hard regardless of
resolution, so there is little *downscale-specific* headroom (oracle 0.790 vs static-full
0.753).

### Retrain the policy on V*Bench + HR-Bench combined

Per-budget classifier trained on **all HR-Bench + (V*Bench minus fold)**, 5-fold OOF on
V*Bench (`scripts/train_budget_policy_combined.py`). Evaluated on V*Bench (the only
substrate with real oracle headroom). Matched-token operating point vs static keep=0.2:

| training data / features | V*Bench OOF AUC (range) | matched-token Δ (policy − static k=0.2) |
|---|---|---|
| V*Bench only, pooled (Phase 2) | 0.39–0.50 | −0.058 [−0.110, −0.005] **SIG worse** |
| V*Bench only, spatial (Goal A) | 0.41–0.53 | −0.052 [−0.105, +0.000] n.s. worse |
| **+HR-Bench, pooled** | 0.47–0.53 | −0.084 [−0.141, −0.031] **SIG worse** |
| **+HR-Bench, spatial** | 0.50–0.57 | −0.042 [−0.099, +0.016] n.s. worse |

A caution on HR-Bench's *in-domain* AUC: predicting "correct@k" on HR-Bench itself gives
AUC **0.90 (pooled) / 0.74 (spatial)** — but this is an artifact of a flat frontier.
With accuracy nearly constant across budgets, "correct@k" ≈ "is this question solvable
at all," which pooled CLIP predicts well — it is **not** downscale-sensitivity. Proof:
despite AUC 0.90, the HR-Bench policy still ties static at matched tokens (Δ −0.003 n.s.)
— high AUC, zero frontier gain.

### Bottom line (B)
**More data did not make the policy beat static.** Combined-data + spatial features is
the best variant and moves V*Bench from "significantly worse" (Phase 2, −0.058 SIG) to
"statistically indistinguishable" from static (−0.042, CI includes 0) — a real but modest
improvement, and still not a win (point estimate worse). HR-Bench was a weak-stress
addition (flat frontier, thin oracle tail) and its high in-domain AUC is answerability,
not downscale-sensitivity, so it adds little signal of the kind the policy needs. The
V*Bench oracle headroom (0.942 @ 868 tok) remains unexploited by CLIP-grade features.

**Artifacts (B):** `rag/vqa_datasets.py` (`load_hrbench`),
`scripts/downscale_stress_test.py` (+hrbench choice),
`scripts/train_budget_policy_combined.py`. Outputs:
`data/vqa_stress/hrbench_downscale.jsonl`, `hrbench_gate.json`,
`hrbench_{spatial,pooled}_feats.npz`, `hrbench_policy_{spatial,pooled}.json`,
`combined_policy_{spatial,pooled}.json`.

---

## Goal C — FastV vs trim-downscale at MATCHED token budget (FastV wins)

### Method
On the **same Qwen2-VL-7B (GPTQ-Int4, HF)** answer model and V*Bench (n=191), compare
in-model pruning (FastV, `rag/fastv.py`) against input-level downscaling at *matched*
visual-token budgets. Phase 1 showed `trim_downscale` ≈ plain downscale on V*Bench
(natural photos have no uniform margins → `_trim_bbox` finds nothing), so here trim ≡
downscale; the comparison is **FastV vs input-downscale**. Reused the Phase 3 FastV data
(`data/vqa_stress/fastv_vstar.jsonl`) and added two downscale conditions at FastV's exact
token budgets: `ds0.5` (~507 tok, matches `full+fastv0.5`'s 509) and `ds0.25` (~258 tok,
matches `full+fastv0.25`'s 255). Paired per-example CIs via `scripts/analyze_fastv.py`.
fp16 GPTQ + eager → NaN gotcha avoided (`rag/fastv.py` uses sdpa for the stable forward).

### Result (n=191, V*Bench) — matched-budget table

| budget | lever | img tokens | accuracy [95% CI] |
|---|---|---|---|
| **~509 tok** | FastV r=0.5 | 509 | 0.602 [0.534, 0.670] |
|              | downscale (ds0.5) | 507 | 0.550 [0.482, 0.623] |
| **~255 tok** | FastV r=0.25 | 255 | 0.576 [0.508, 0.644] |
|              | downscale (ds0.25) | 258 | 0.476 [0.408, 0.550] |

Paired (Δ = FastV − downscale, same examples):

| matched budget | Δ (FastV − downscale) | verdict |
|---|---|---|
| ~509 tok | +0.052 [−0.010, +0.115] | FastV better, n.s. |
| ~255 tok | **+0.099 [+0.026, +0.173]** | **FastV SIG better** |

### Bottom line (C)
**At a matched visual-token budget, FastV (in-model pruning) is the better lever than
input-level downscaling — and its edge grows as the budget tightens.** At ~255 tokens
FastV is significantly more accurate (+0.099, CI excludes 0); at ~509 the advantage is
positive but not significant. The mechanism: FastV keeps *full-resolution* patches and
drops only the low-attention ones, preserving the small query-relevant detail V*Bench
depends on, whereas downscaling blurs that detail everywhere. This is consistent with
Phase 3 ([[fastv-composition-negative-interaction]]: FastV nearly free at full res,
harmful after downscale) and with the project pivot: spend the token budget by *keeping
the right tokens at full resolution*, not by lowering resolution uniformly. (On DocVQA,
where trim ≠ downscale, trim was already shown neutral in Phase 1; the optional DocVQA
matched-budget FastV run was not needed to settle the V*Bench verdict and was skipped
to respect the one-model-at-a-time GPU constraint.)

**Artifacts (C):** `scripts/run_fastv_matrix.py` (+ds0.5/ds0.25 conditions),
`scripts/analyze_fastv.py` (+matched-budget paired tests). Outputs:
`data/vqa_stress/fastv_vstar.jsonl` (extended), `fastv_vstar_report.json` (extended).

---

## Overall verdict

| goal | question | answer |
|---|---|---|
| A | spatial features beat pooled CLIP / static frontier? | **No** — OOF AUC still ≈0.5; policy ties but never beats static. |
| B | new data shows variance; does more data help the policy win? | HR-Bench passes the gate but is weak-stress (flat, thin tail). Combined+spatial improves V*Bench from SIG-worse to n.s., **still not a win**. |
| C | FastV or trim-downscale at matched budget? | **FastV** — significantly better at ~255 tok (+0.099), favorable at ~509. |

The consistent thread: on detail-sensitive data the lever is **which tokens to keep at
full resolution**, not uniform resolution reduction — and cheap CLIP features (pooled
or per-patch) cannot predict per-example downscale sensitivity well enough to drive an
adaptive input-downscale policy. In-model attention-based selection (FastV) achieves
what the input-downscale policy could not.
