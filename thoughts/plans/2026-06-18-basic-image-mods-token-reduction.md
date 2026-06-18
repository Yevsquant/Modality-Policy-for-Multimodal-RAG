# Plan: Basic Image-Modification Combinations for Visual-Token Reduction

## Premise (what the n=300 evaluation established)

- Visual tokens ∝ pixel area (`prod(image_grid_thw)//merge_size**2`). The lever is
  **resolution**, not content selection.
- **Uniform downscale wins:** 70% token cut (452 vs 1505) at no significant accuracy
  loss vs full images, and it's the fastest path.
- **Cropping hurts:** query-aware `clip_safecrop` is dominated; the hybrid
  `clip_safecrop_downscale` is *significantly worse* than plain downscale at equal
  budget — discarding peripheral content loses more than it saves.
- **CLIP/query-aware compute didn't pay off** and added ~2× latency (per-tile scoring).

**Design principle going forward:** preserve all real content; reduce pixel area by
resolution + removing only non-informative area; spend (near) zero query-aware compute
— and if any, put it at the very end and reuse signals we already computed.

`downscale_baseline` is the **new bar to beat**. A modification is only worth keeping
if, at equal-or-lower token budget, it is *not* significantly worse than downscale
(ideally: same accuracy at fewer tokens, or more accuracy at equal tokens).

## Hypotheses to test

1. **Margin/whitespace trim is free area.** Document images carry uniform borders that
   cost tokens but no information. Trimming them (query-agnostic, content-preserving)
   then downscaling spends the budget on actual content → holds accuracy at lower
   tokens than pure downscale.
2. **Token budget should follow information density.** Text/table-dense pages need
   more resolution than photos/logos. A query-agnostic per-image budget set by detail
   density preserves accuracy at a lower *average* token count than a flat budget.
3. **Query-awareness helps only as resolution allocation, done for free.** The
   retriever already computes a query↔image CLIP similarity to rank image quotes
   (`rag/retriever.py`). Reuse that existing score to give more tokens (higher res) to
   more-relevant images and fewer to marginal ones — no new model, no extra forward,
   no cropping. This is the "CLIP at the very end, do less work" idea taken to zero
   marginal cost.

## What we're NOT doing

- No cropping away real content (montage, bbox crop, cluster) — shown harmful.
- No 7B; no per-tile CLIP scoring; no new heavy model. Any query signal must be the
  retriever's already-computed score (or at most one cheap image-level small-CLIP call).
- No JPEG-quality / grayscale / binarization changes — they do **not** change token
  count (tokens depend on the resize grid, not file size or channels). Explicitly out.
- No seam-carving / foveated variable-resolution single images for now — speculative,
  and our evidence says global layout matters, so keep transforms uniform per image.

## Phases

Each phase: add one query-agnostic-or-free mode in `rag/pruner.py` (reusing
`_save_pruned_image`, `_area_budget_factor`, `_downscale_image` patterns), TDD the pure
geometry, then evaluate at n=300 with CIs.

### Phase 0 — Map the downscale frontier (defines the bar)
Run `downscale_baseline` at keep ∈ {0.1, 0.15, 0.2, 0.3} (n=300, target-model token
counts, CIs). Find the budget where pure downscaling *starts* to drop below the
full-image accuracy. That breakpoint is where smarter modifications have room to help;
above it, nothing can beat "just downscale."
- **Success:** a token-vs-accuracy curve for downscale with CIs; an identified knee.

### Phase 1 — `trim_downscale` (margin trim → downscale)
Auto-trim near-uniform borders (PIL: difference from a sampled border color → bbox of
non-background, with a conservative threshold and a "never trim >X% / revert if bbox
collapses" guard), then downscale the trimmed image to the keep_ratio budget.
- **Test:** at the Phase-0 knee budget, is `trim_downscale` accuracy ≥ downscale
  (paired diff CI vs downscale not negative)? Does it reach a *lower* token budget at
  equal accuracy?
- **Risk/guard:** trimming must be safe (unit-test that uniform-border images trim to
  content bbox and photos with full-bleed content are left unchanged).

### Phase 2 — `density_adaptive_downscale` (query-agnostic budget by detail)
Per image, estimate detail density (cheap: mean gradient magnitude / edge ratio on a
thumbnail). Allocate that image's token budget up or down around the global keep_ratio
within [min,max], so dense pages keep resolution and sparse images give tokens back.
- **Test:** at equal *average* token budget to downscale, is accuracy ≥ downscale? Does
  it push the average budget lower at equal accuracy?

### Phase 3 — `relevance_adaptive_downscale` (free query-awareness)
Reuse the retriever's existing query↔image similarity (already attached as the image
quote `tag`/score) to distribute the total budget across the retrieved images:
higher-relevance → higher resolution, lower → more downscaled. Pure downscaling, zero
extra model calls.
- **Test:** vs flat downscale at equal total tokens, paired diff CI on judge_correct.
  This is the cleanest test of whether *any* query signal helps once we stop cropping.
- **Stretch:** combine the winners (e.g. `trim` + `relevance_adaptive`) and re-measure.

## Measurement protocol (carry over the rigor)

- Tokens counted on the **target model processor** (`VisualTokenCounter`).
- n=300 (slice [0,300) — [0,50) is unrepresentatively hard), prune cache disabled.
- Per-method judged JSON via `scripts/run_method.py`; CIs + paired diffs vs
  `downscale_baseline` via `scripts/compare_methods_ci.py`; sweeps via
  `scripts/sweep_keep_ratio.py` (resumable). Detached server to survive the reaper.
- **Decision rule:** keep a modification only if its paired judge_correct diff vs
  downscale is non-negative (CI not entirely below 0) at an equal-or-lower token
  budget. Report the token-vs-accuracy Pareto frontier across all modes.

## Why this is the right bet

The expensive query-aware machinery has now been falsified twice (crop, hybrid). The
remaining upside is (a) squeezing the budget lower than flat downscale via free,
content-preserving area reduction (trim, density), and (b) a *zero-cost* query signal
(reuse retrieval score) for resolution allocation — not cropping. If even (b) shows no
significant gain, the honest, publishable conclusion is "uniform downscaling to a
detail-aware budget is the practical optimum for visual-token reduction on MMDocRAG,"
which is itself a clean result.
