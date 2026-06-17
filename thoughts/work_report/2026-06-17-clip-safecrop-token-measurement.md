# Work Report: CLIP-Safecrop + Real Visual-Token Measurement

**Date:** 2026-06-17
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

## Not yet done — requires dataset + running vLLM server (the plan's Manual Verification)

`data/` is empty (download script not run) and no vLLM server is up. Remaining:
1. `bash scripts/download_mmdocrag.sh`.
2. Serve Qwen3-Omni-30B via vLLM (README command, Terminal 1).
3. Run `clip_safecrop` benchmark; confirm `avg_visual_tokens_reduction_pct > 0` and
   `avg_total_sec` below the 7B `safecrop_pruning` baseline.
4. Generate `imgs/VisualTokensByMethods.png` (plot script) and run
   `scripts/sweep_keep_ratio.py` → `imgs/TokenVsAccuracy.png`; read off the keep_ratio
   "knee".

## Notes / decisions

- Cluster stitching (reflection suggestion C) deliberately **not** done — breaks the
  spatial layout the downstream visual encoder relies on.
- Old 7B modes and `qwen2vl_catp_pruner_v2.py` retained for A/B comparison; revert by
  setting `pruning_mode = "safecrop_pruning"`.
- When switching modes, clear `image_prune_cache.json` + `pruned_images/` to avoid
  mixing artifacts from different modes (per plan's Migration Notes).
