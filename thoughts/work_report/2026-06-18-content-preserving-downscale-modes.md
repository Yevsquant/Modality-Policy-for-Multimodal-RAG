# Work Report: Content-Preserving Downscale Modes (trim / density / relevance)

**Date:** 2026-06-18
**Branch:** `clip-safecrop-token-measurement`
**Plan:** [thoughts/plans/2026-06-18-basic-image-mods-token-reduction.md](../plans/2026-06-18-basic-image-mods-token-reduction.md)
**Goal:** After plain downscaling beat all query-aware cropping, test whether
content-preserving tricks (trim margins, density-adaptive budget, free
relevance-based allocation) can beat plain downscaling at an **equal token budget**,
and map how low downscaling alone can go.

## Method

n=300 (slice [0,300)), keep_ratio=0.3, prune cache disabled, tokens counted on the
target model's processor. Per-method judged JSONs via `scripts/run_method.py`;
bootstrap 95% CIs + paired per-example differences via `scripts/compare_methods_ci.py`
(`data/mmdocrag/analysis/methods_ci_v2.json`). Reference = `downscale_baseline`.
Four prior n=300 baselines reused (no_pruning, downscale, clip_safecrop, hybrid).

## Result 1 — equal-budget comparison (keep=0.3)

| method | judge_correct [95% CI] | visual tokens | total_sec | paired Δ vs downscale [95% CI] |
|---|---|---|---|---|
| no_pruning (full) | 0.717 [0.667, 0.767] | 1505 | 5.1 | +0.007 [−0.023, +0.037] — n.s. |
| downscale_baseline | 0.710 [0.657, 0.760] | 452 | 4.7 | — (reference) |
| **trim_downscale** | **0.740 [0.690, 0.790]** | 452 | 4.6 | **+0.030 [+0.003, +0.057] — SIGNIFICANT** |
| relevance_adaptive_downscale | 0.723 [0.673, 0.773] | 452 | 5.1 | +0.013 [−0.020, +0.047] — n.s. |
| density_adaptive_downscale | 0.707 [0.653, 0.757] | 424 | 4.6 | −0.003 [−0.040, +0.030] — n.s. |
| clip_safecrop | 0.687 [0.633, 0.740] | 949 | 9.6 | −0.023 [−0.063, +0.013] — n.s. |
| clip_safecrop_downscale (hybrid) | 0.663 [0.610, 0.717] | 444 | 9.2 | −0.047 [−0.080, −0.013] — SIGNIFICANT (worse) |

**`trim_downscale` is a genuine, statistically significant win** over plain
downscaling at the same token budget (+0.030, CI excludes 0) — and it even edges out
full images (0.740 vs 0.717). Trimming blank margins before downscaling lets the
budget's pixels land on actual content at higher effective resolution. It is also as
fast as plain downscaling (no CLIP/model in the path).

The two query-aware-ish modes did **not** significantly help: `relevance_adaptive`
(reusing the retriever's free CLIP score to give more pixels to more-relevant images)
is +0.013 but not significant; `density_adaptive` is neutral. Consistent with the
earlier finding that query/content signals add little here — the win came from a
purely geometric, query-agnostic move (drop the margins).

## Result 2 — downscale frontier (how low can pure downscaling go?)

| keep_ratio | visual tokens | judge_correct [95% CI] |
|---|---|---|
| 0.10 | 150 | 0.710 [0.660, 0.760] |
| 0.15 | 223 | 0.690 [0.637, 0.740] |
| 0.20 | 301 | 0.700 [0.647, 0.750] |
| 0.30 | 452 | 0.710 [0.657, 0.760] |
| (full) | 1505 | 0.717 [0.667, 0.767] |

The frontier is **flat from 150 to 1505 tokens** — every CI overlaps. Downscaling to
keep=0.1 (**150 tokens, a 90% cut**) loses nothing statistically vs full images. The
task needs very little visual resolution; on MMDocRAG the retrieved text quotes carry
most of the answer signal and the image only needs to be legible, not high-res.

## Bottom line

Two clean, defensible findings:

1. **Trimming blank margins before downscaling significantly beats plain downscaling
   at equal token budget** (and matches full-image accuracy) — the first modification
   in this project to improve on the baseline, and it is query-agnostic and free.
2. **Uniform downscaling alone cuts visual tokens ~90% (1505→150) with no significant
   accuracy loss.** Query-aware cropping/allocation does not help; the levers that
   matter are resolution and not wasting pixels on margins.

Practical recommendation: **trim margins, then downscale to a low budget (~150–450
tokens).** The expensive query-conditioned machinery (7B attention, per-tile CLIP)
is unnecessary on this benchmark.

## Follow-ups

- **Combine the two winners:** `trim_downscale` at the low frontier budget
  (keep≈0.1) — likely pushes below 150 tokens at no accuracy cost. Highest-value next
  run.
- Confirm the trim win replicates on a second n=300 slice (guard against the
  significance being a single-slice artifact, p just under 0.05).
- The flat frontier suggests probing why images barely matter here (text-quote
  sufficiency); relevant to whether visual pruning is even the right lever for this
  dataset.

## Artifacts

- `data/mmdocrag/analysis/runs/*_judged.json` (10 methods/budgets, n=300)
- `data/mmdocrag/analysis/methods_ci_v2.json` (CI table + paired diffs)
- Modes implemented in `rag/pruner.py` (commit `b7e0085`); stats in `rag/metrics.py`.
