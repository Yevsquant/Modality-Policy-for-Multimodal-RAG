# Plan: Can input-side pre-processing beat FastV?

**Date:** 2026-06-19
**Status:** proposal — follow-up to Phase 3/4 (FastV beat uniform input-downscale at
matched tokens: +0.099 [+0.026,+0.173] @ ~255 tok on V\*Bench).
**Branch:** `clip-safecrop-token-measurement`

## Why FastV won, precisely (this dictates the strategy)

FastV beats **uniform downscaling** because it is **spatially selective at full
resolution**: it keeps the query-relevant patches at native res and drops the rest.
Uniform downscale degrades resolution *everywhere*, including the small object V\*Bench
asks about (V\* = visual *search*). Our Phase-2/4 learned policy lost because its action
space was a **global scalar budget** ("how much to downscale the whole image") — which
*cannot* preserve a small region. The fix is not a better feature; it is a **spatial
action space**: decide *where* to keep full resolution, at the input.

Two structural facts make a spatial pre-processor a real contender against FastV:

1. **FastV pays full early cost.** It encodes the *entire* image through the vision
   encoder and the first K LLM layers, then prunes. Its token saving is **deep-LLM
   only**. A pre-processor that drops regions *before* the encoder saves the encoder
   **and** all LLM layers. So at equal *deep-layer* tokens, pre-processing used strictly
   less total compute — the Phase-3/4 "matched token budget" axis was unfair to it.
2. **Crop ≠ downscale w.r.t. FastV.** Phase 3 found downscale+FastV interact
   *negatively* (downscale removes the detail FastV wants). A **crop** keeps the relevant
   region at full res, so crop+FastV should interact *positively* — a sharp, testable
   distinction.

## Three angles (ordered by risk; do the gates in order)

### Angle 1 — Fair-cost reframing (robust, ~free, do first)
Re-score the existing Phase-3/4 comparison on **accuracy vs total cost** (vision-encoder
FLOPs + LLM FLOPs, and wall-clock latency), not deep-layer tokens. For each condition in
`data/vqa_stress/fastv_vstar.jsonl` we already know input tokens (encoder load), the
prune layer K, and post-prune tokens, so total FLOPs are computable analytically; add a
direct latency measurement on the 7B for a few conditions.
- **Hypothesis:** at equal *total* FLOPs/latency, input-downscale closes much of the gap
  to FastV (FastV's full-image early cost is large).
- **Deliverable / GATE 1:** a corrected Pareto plot on the honest cost axis. Even if the
  method work below fails, this reframing is a legitimate, publishable correction.
- No new GPU runs needed beyond a small latency probe.

### Angle 2 — Query-conditioned coarse-to-fine foveated crop (the method)
Localize the query-relevant region cheaply, crop it from the **full-res** image, feed
only that (optionally + a low-res thumbnail of the whole for context = foveation). This
is the project's old `safecrop` idea, resurrected but (a) on V\*Bench where it should
*help* (unlike MMDocRAG where it didn't), (b) measured on the fair cost axis, (c)
explicitly targeting FastV.
Localizer options, cheapest first:
- **CLIP patch-relevance map** (reuse Phase-4 `SpatialClipFeaturizer`): per-patch CLIP
  similarity to the query → heatmap → bounding box. Nearly free (one CLIP pass).
  *Caveat:* Phase-4 showed CLIP can't predict downscale *sensitivity*; **localization is
  a different task** and CLIP patch-text similarity is an established weak localizer —
  but on small-object visual search it may still miss.
- **Coarse-to-fine ("zoom"):** a cheap **low-res** full-image pass (CLIP, or the 7B at
  low res) to locate, then crop the region from the full-res image. Informed by an actual
  (cheap) look at the whole image — closest to how FastV gets its signal, but at a
  fraction of the cost.
- **Tiny open-vocab grounder** (OWLv2-base / GroundingDINO-T): purpose-built localization,
  still far smaller than the answer model.
- **GATE 2:** crop-contains-answer rate, and crop-only accuracy vs full-image accuracy at
  matched tokens. If localization can't find the small object (the benchmark's whole
  difficulty), report it — that itself explains FastV's edge (FastV sees full-res first).

### Angle 3 — crop → FastV composition (likely the winner)
Cheap crop cuts encoder + early cost; FastV prunes within the cropped, still-full-res
region. Directly tests the "crop+FastV is positive (unlike downscale+FastV)" hypothesis.
- **GATE 3:** does crop+FastV dominate both FastV-alone and crop-alone on accuracy-vs-
  total-cost? A positive interaction here is the headline result.

## Success criteria
- **Minimum win (Angle 1):** on accuracy-vs-total-FLOPs, input-side methods are no longer
  dominated — the "FastV wins" claim is shown to be axis-dependent.
- **Method win (Angle 2/3):** a query-conditioned crop (or crop+FastV) matches or beats
  FastV's accuracy at equal deep tokens **and** strictly less total compute, on V\*Bench,
  n≥191, paired CIs (reuse `rag/metrics.py`).

## Honest risks
- **Localization is the crux and may fail.** V\*Bench exists *because* single-shot models
  fail at visual search; cheap localizers may not find the small object. Angle 1 is the
  fallback that wins regardless of localization quality.
- FastV's signal comes from a full-res encode it already pays for; a pre-processor must
  match that localization without that pass. Coarse-to-fine is the bridge but adds a
  (cheap) second pass — fold its cost into the fair-cost accounting.
- Keep it on **V\*Bench** (real budget variance); HR-Bench/DocVQA are too downscale-robust
  to discriminate (Phase 0/4).

## Reuse
`rag/fastv.py` (FastV + the 7B harness, matched-budget conditions already added),
`rag/image_ops.py`, `rag/budget_features.py` (`SpatialClipFeaturizer` for the heatmap),
`rag/pruner._safe_crop` (existing bbox crop), `rag/metrics.py`, `data/vqa_stress/fastv_vstar.jsonl`.
Sequence GPU work (one model at a time); heed the fp16-eager-NaN and processor-reclamp
gotchas in memory.
