# Work Report: Angle 2 — Query-conditioned foveated CROP vs FastV

**Date:** 2026-06-19
**Branch:** `clip-safecrop-token-measurement`
**Plan:** [thoughts/plans/2026-06-19-preprocess-vs-fastv.md](../plans/2026-06-19-preprocess-vs-fastv.md) (Angle 2)
**Builds on:** Phase 3 ([[fastv-composition-negative-interaction]]), Phase 4
([[phase2-clip-features-cant-predict-budget]],
[[vstar-is-the-stress-substrate-trim-doesnt-transfer]]).

## The question

Phase 3/4 found FastV (in-model attention pruning) beats **uniform input downscaling**
at a matched visual-token budget on V*Bench (+0.099 [+0.026,+0.173] @ ~255 tok), because
FastV keeps query-relevant patches at full resolution while downscaling blurs detail
everywhere. Angle 2 asks: can an **input-side** crop reproduce that spatial selectivity
*before* the model — localize the relevant region, crop it from the full-res image, and
downscale only the crop to the budget (foveation) — matching or beating FastV at the
same token budget?

## Method

`rag/foveated_crop.py`. Pure geometry (`heatmap_to_bbox`, `crop_box_to_budget`,
`make_foveated_image`; unit-tested in `tests/test_foveated_crop.py`): a per-patch
relevance map → percentile-thresholded bbox (+margin) → crop from full-res → downscale
to the token budget (tokens ≈ pixels / 28²) only if the crop still exceeds it.

Two localizers:
- **`crop_clip`** (cheap): per-patch CLIP cosine to the question text, reusing Phase-4
  `SpatialClipFeaturizer._patch_relevance` (7×7 grid, one CLIP forward).
- **`crop_c2f`** (coarse-to-fine): one cheap **low-res** pass through the 7B (~256 tok),
  read the last-query-token attention to image tokens at layer 3 (the same signal FastV
  uses), map high-attention tokens back to image coords via `image_grid_thw` (merged
  grid `grid_h//2 × grid_w//2`), take their bbox → crop the **full-res** image. Low-res
  locate → full-res crop, at a fraction of FastV's full-res-first cost.

The crop is scored as multiple-choice on the **same Qwen2-VL-7B (GPTQ-Int4, HF)** answer
model via `rag.fastv.FastVQwen2VL.answer_mc(crop, q, input_keep=1.0, fastv_layer=None)` —
the crop already reduced the tokens, so no in-model pruning. Conditions: `{crop_clip,
crop_c2f} × {~255, ~509 tok}`, n=191 V*Bench. Paired bootstrap CIs (`rag.metrics`)
against the existing `ds0.25/ds0.5` and `full+fastv0.25/0.5` columns in
`data/vqa_stress/fastv_vstar.jsonl` (`scripts/run_foveated_crop.py`,
`scripts/analyze_foveated_crop.py`).

## Result (n=191, V*Bench) — matched-budget table

| budget | method | img tokens | accuracy [95% CI] |
|---|---|---|---|
| **~255 tok** | full + FastV r=0.25 | 255 | 0.576 [0.508, 0.644] |
|              | crop_c2f            | 258 | 0.487 [0.414, 0.560] |
|              | crop_clip           | 257 | 0.487 [0.414, 0.555] |
|              | downscale (ds0.25)  | 258 | 0.476 [0.408, 0.550] |
| **~509 tok** | full + FastV r=0.5  | 509 | 0.602 [0.534, 0.670] |
|              | crop_c2f            | 505 | 0.571 [0.503, 0.639] |
|              | downscale (ds0.5)   | 507 | 0.550 [0.482, 0.623] |
|              | crop_clip           | 507 | 0.545 [0.476, 0.613] |
| reference    | full (no pruning)   | 1018 | 0.607 [0.534, 0.675] |

Paired tests (Δ = crop − baseline, same examples):

| budget | test | Δ [95% CI] | verdict |
|---|---|---|---|
| ~255 | crop_c2f − downscale | +0.010 [−0.021, +0.042] | tie (n.s.) |
| ~255 | crop_clip − downscale | +0.010 [−0.042, +0.058] | tie (n.s.) |
| ~255 | crop_c2f − **FastV** | **−0.089 [−0.162, −0.016]** | **crop SIG worse** |
| ~255 | crop_clip − **FastV** | **−0.089 [−0.168, −0.016]** | **crop SIG worse** |
| ~509 | crop_c2f − downscale | +0.021 [−0.016, +0.058] | tie (n.s., favorable) |
| ~509 | crop_clip − downscale | −0.005 [−0.052, +0.042] | tie (n.s.) |
| ~509 | crop_c2f − FastV | −0.031 [−0.094, +0.031] | tie (n.s., worse) |
| ~509 | crop_clip − FastV | −0.058 [−0.120, +0.005] | n.s. worse |
| both | crop_c2f − crop_clip | +0.026 [−0.016, +0.068] @509; +0.000 @255 | localizers tie |

## Per-category breakdown (where crop helps / hurts)

V*Bench has two categories: `direct_attributes` (n=115, find a small object and read an
attribute = hard visual search) and `relative_position` (n=76, spatial relation between
objects).

| category | crop_c2f255 | crop_c2f509 | full+fastv0.25 | full+fastv0.5 | ds0.25 | full |
|---|---|---|---|---|---|---|
| direct_attributes (115) | 0.417 | 0.557 | **0.600** | 0.626 | 0.417 | 0.617 |
| relative_position (76)  | 0.592 | 0.592 | 0.539 | 0.566 | 0.566 | 0.592 |

The split is the whole story: on **`relative_position`**, crop matches everything
(0.592 = full, ≥ FastV) — these questions need the global spatial layout, which a
generous crop preserves. On **`direct_attributes`** at the tight 255-tok budget, crop
collapses to ds-level (0.417 = exactly ds0.25) and is far below FastV (0.600). This is
precisely the small-object visual-search case the localizer must nail and doesn't: a
miss crops away the answer object, and a hit that's too small still downscales it.

## Bottom line — honest null

**The input-side query-conditioned crop neither beats uniform downscale nor matches
FastV at a matched token budget on V*Bench.**

1. **vs uniform downscale:** crop **ties** (Δ +0.010 to +0.021, all CIs include 0).
   c2f's +0.021 @509 hints the localizer occasionally finds the region, but the signal
   is within noise. The localization-quality proxy (crop > downscale) does **not** fire:
   the cheap localizers cannot reliably isolate the answer region well enough to beat
   blurring everything.
2. **vs FastV:** crop is **significantly worse at ~255 tok** (−0.089, CI excludes 0) and
   not-significantly worse at ~509 (−0.031 / −0.058). FastV's edge holds: it gets its
   localization from a **full-resolution** encode it already pays for, so it never crops
   away the answer; a one-shot input crop must localize *without* that full-res look and
   pays for every miss.
3. **Which localizer:** **c2f ≈ clip** (Δ +0.026 @509, +0.000 @255, both n.s.). The
   coarse-to-fine 7B-attention pass is *not* meaningfully better than the nearly-free
   CLIP heatmap — at low resolution the 7B's own attention is already too coarse to beat
   CLIP at finding the small object, consistent with Phase 4's finding that cheap
   features can't locate downscale-sensitive detail.

This is the expected outcome the brief flagged: V*Bench exists *because* single-shot
visual search is hard. Both localizers fail on exactly the category that requires it
(`direct_attributes`), which is itself the explanation for FastV's edge — FastV looks at
the full-resolution image first and prunes second, so it never has to localize blind. The
`relative_position` parity is a genuine (if narrow) positive: where the task is layout
rather than fine-detail search, a layout-preserving crop loses nothing.

A note for Angle 3: since crop ties downscale rather than beating it, the
"crop+FastV interacts positively (unlike downscale+FastV)" hypothesis is now weaker —
crop did not preserve more answer-relevant detail than downscale in aggregate. The one
place to look is `relative_position`, where crop = full.

## Artifacts

- `rag/foveated_crop.py` (`heatmap_to_bbox`, `crop_box_to_budget`, `make_foveated_image`,
  `ClipLocalizer`, `c2f_relevance`), `scripts/run_foveated_crop.py`,
  `scripts/analyze_foveated_crop.py`, `tests/test_foveated_crop.py` (+7).
- Outputs: `data/vqa_stress/foveated_crop_vstar.jsonl` (764 rows),
  `data/vqa_stress/foveated_crop_report.json`.
- Reused: `rag/fastv.py`, `rag/budget_features.SpatialClipFeaturizer`,
  `rag/vqa_datasets.load_vstar`, `rag/metrics.paired_diff_ci`,
  `data/vqa_stress/fastv_vstar.jsonl` (FastV/downscale reference columns).
