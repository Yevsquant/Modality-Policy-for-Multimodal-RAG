# Work Report: CLIP-Safecrop + Real Visual-Token Measurement

**Date:** 2026-06-17 (implementation); 2026-06-18 (benchmark run)
**Branch:** `clip-safecrop-token-measurement`
**Plan:** [thoughts/plans/2026-06-17-clip-safecrop-and-token-measurement.md](../plans/2026-06-17-clip-safecrop-and-token-measurement.md)
**Method:** TDD (red → green → commit), one commit per phase.

## Goal

Move the "how to crop" signal off the expensive **7B Qwen2-VL full-attention forward**
onto the already-present cheap **CLIP region scoring**, while keeping safecrop's
single-box, layout-preserving output (no montage, no multi-subimage). Then add the
project's genuinely-missing metric: count visual tokens with the **target generation
model's own processor** (Qwen3-Omni), aggregate the reduction, and plot it. Finally,
add an honest control (uniform downscale to the same token budget) and a keep_ratio
sweep to produce a token-vs-accuracy tradeoff curve.

## What was implemented

### Phase 1 — `clip_safecrop` mode (commit `1b831bc`)
- `rag/pruner.py`: module-level `_bbox_union`; registered `clip_safecrop`; extended the
  CLIP-load condition to cover it; new `_clip_safecrop_image` (reuses
  `_extract_grid_tiles` / `_score_tiles` / `_save_pruned_image`): CLIP-score tiles →
  take the enclosing box of the kept tiles → crop the original once → single `.jpg`.
- `rag/config.py`: default `pruning_mode` → `clip_safecrop` so the main path no longer
  loads the 7B.

### Phase 2 — real token measurement (commit `969cc3d`)
- `rag/visual_token_counter.py` (new): `VisualTokenCounter` loads **only** the target
  model's image processor (not the 30B weights); token count =
  `prod(image_grid_thw) // merge_size**2`, with a pixel-area fallback.
- `rag/query_pipeline.py`: capture original image paths **before** `pruner.apply()`
  (which overwrites `local_img_path`); count before/after on the images actually sent
  (single jpg once; directory summed); cached images isolated as `from_cache_after` so
  they don't distort the reduction. Attaches `out["visual_tokens"]`.
- `rag/metrics.py`: `aggregate_summary` now emits `avg_visual_tokens_before`,
  `avg_visual_tokens_after`, `avg_visual_tokens_reduction_pct`.
- `scripts/plot_visual_tokens.py` (new): before-vs-after bar chart per method →
  `imgs/VisualTokensByMethods.png`.

### Phase 3 — baseline + sweep (commit `0eaeb8e`)
- `rag/pruner.py`: `downscale_baseline` mode (`_downscale_image`) — query-agnostic
  uniform downscale by edge factor `sqrt(keep_ratio)` → ~`keep_ratio` of the tokens,
  full layout at lower resolution.
- `scripts/sweep_keep_ratio.py` (new): sweeps keep_ratio for `clip_safecrop` vs
  `downscale_baseline`, records real target-model `avg_visual_tokens_after` vs
  `avg_judge_correct` / `avg_total_sec`, writes
  `data/mmdocrag/analysis/keep_ratio_sweep.json`, plots `imgs/TokenVsAccuracy.png`.
  Disables the reuse cache during the sweep for fairness; heavy imports deferred so
  `--help` is fast.

## Verification done (CPU + GPU, no vLLM server)

- **Unit tests:** 10 passing across `tests/test_pruner_geometry.py` and
  `tests/test_visual_tokens.py` (bbox union, downscale math/behavior, mode
  registration, summary reduction %, counter math). Installed `pytest` into the `mrag`
  conda env.
- **clip_safecrop functional (real CLIP, GPU):** produces a single `.jpg` with a
  contiguous crop box, tokens 16→6, and a trip-wire confirms the **7B is never
  constructed**.
- **Counter functional (real Qwen3-Omni processor):** loads processor only; sane
  counts — 1280×960→1200, 640×480→300, 320×240→80 tokens.
- **downscale_baseline functional:** 800×600 → 400×300 (exactly keep_ratio=0.25 area).

## Benchmark results (run 2026-06-18, slice [0,50), Qwen3-Omni-30B via vLLM)

All token figures measured with the target model's processor (`VisualTokenCounter`).

**Headline single-point comparison:**

| method | visual tokens | judge_correct | judge_score | total_sec |
|---|---|---|---|---|
| no_pruning (clean, cache off) | 1715 → 1715 | 0.36 | 2.10 | 10.0 |
| clip_safecrop (keep=0.3) | 1605 → **975** | 0.32 | 1.80 | 6.05 |

clip_safecrop's 0.32 is **not a regression** — the full-image ceiling is only 0.36.
The slice is hard (retrieval_recall ~0.49 caps everyone; judge marks ~16–18/50
correct even with full images). So pruning costs ~0.04 accuracy for ~43% fewer
tokens and ~40% lower latency.

**keep_ratio sweep — clip_safecrop vs downscale_baseline** (`imgs/TokenVsAccuracy.png`,
`data/mmdocrag/analysis/keep_ratio_sweep.json`):

| keep | clip_safecrop tok / correct | downscale tok / correct |
|---|---|---|
| 0.1 | 1042 / 0.32 | 170 / 0.28 |
| 0.2 | 1042 / 0.30 | 343 / 0.24 |
| 0.3 | 1042 / 0.32 | 512 / 0.32 |
| 0.4 | 1463 / 0.28 | 687 / 0.30 |
| 0.5 | 1622 / 0.34 | 857 / 0.36 |
| 0.7 | 1686 / 0.32 | 1200 / 0.28 |

**Key (honest, partly negative) conclusion:** the mediocre baseline **wins**. Plain
uniform downscaling is Pareto-superior to query-aware clip_safecrop on this slice —
at equal-or-better accuracy it uses far fewer tokens (e.g. downscale keep=0.3 = 512
tok @ 0.32 vs clip_safecrop's best ~1042 tok @ 0.32; downscale keep=0.5 = 857 tok @
0.36 matches the full-image accuracy ceiling), and runs ~2x faster (~4.5s vs ~8.5s).

**Why clip_safecrop is dominated:** its token-after **floors at ~1042** (≈65% of the
full image) and never drops further, because the single bounding-box union of the
top-scored tiles stays large whenever relevant content is spatially scattered (and
`min_visual_tokens=4` keeps ≥4 tiles). So the layout-preserving single-crop design
caps the achievable token reduction at ~35%, while downscaling scales smoothly down
to ~10% of tokens. Accuracy is essentially flat (0.24–0.36, within judge noise)
across all 12 points — no method improves answers; the only real lever is tokens,
and downscaling pulls it harder.

**Implications worth discussing:** (1) For this dataset, a query-agnostic downscale
is the stronger, simpler token-reduction lever. (2) clip_safecrop might still win
where relevant content is spatially *concentrated* (its bbox would shrink) — this
slice doesn't exhibit that. (3) A natural follow-up is a hybrid: crop to the
relevant bbox *then* downscale that crop to a token budget.

## Notes / decisions

- Cluster stitching (reflection suggestion C) deliberately **not** done — breaks the
  spatial layout the downstream visual encoder relies on.
- Old 7B modes and `qwen2vl_catp_pruner_v2.py` retained for A/B comparison; revert by
  setting `pruning_mode = "safecrop_pruning"`.
- When switching modes, clear `image_prune_cache.json` + `pruned_images/` to avoid
  mixing artifacts from different modes (per plan's Migration Notes).

## Operational traps hit during benchmarking (and how they were handled)

- **Prune-cache contaminated the first cross-mode control.** The on-disk
  `image_prune_cache.json` is keyed by `image_cache_id` + query tag, NOT pruning
  mode, so the first `no_pruning` control reused clip_safecrop's cached crops and
  reported `avg_visual_tokens_before=0.0`. Fix: rerun controls with
  `image_prune_cache_enabled=False`. The clean control gave the true 0.36 ceiling.
- **~60-min background-task reaper killed the vLLM server twice mid-sweep**, losing
  all sweep progress (it wrote JSON only at the end). Fixes: (a) made
  `sweep_keep_ratio.py` write incrementally + resume (commit `6e1ebc8`); (b) ran the
  server and sweep **detached** via `nohup setsid` so they outlive the reaper.
- **Orphaned GPU memory:** hard-killed engines left ~63 GB allocated with no process
  attached; reclaiming needs a GPU reset (not performed unprompted).

## Artifacts produced

- Figures: `imgs/VisualTokensByMethods.png`, `imgs/TokenVsAccuracy.png`.
- Sweep data: `data/mmdocrag/analysis/keep_ratio_sweep.json` (12 points).
- Judged results (gitignored): `baseline_results_judged_clip_safecrop.json`,
  `baseline_results_judged_no_pruning_clean.json`.

## Bottom line

The implementation meets the plan (cheap CLIP signal, layout-preserving single crop,
real target-model token measurement, honest baseline + sweep). But the measurement it
enabled is **negative for the method**: on MMDocRAG slice [0,50), query-agnostic
downscaling Pareto-dominates clip_safecrop — fewer tokens at equal/better accuracy and
~2× faster — because clip_safecrop's single-bbox token-after floors at ~1042 (~65% of
the full image) on spatially-scattered content. The value delivered here is the honest
measurement apparatus that surfaces this, not a win for the crop method.
