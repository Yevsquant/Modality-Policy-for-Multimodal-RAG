# Work Report: Larger-N Evaluation with Confidence Intervals

**Date:** 2026-06-18
**Branch:** `clip-safecrop-token-measurement`
**Goal:** Replace the n=50 point estimates (which were within judge noise) with an
n=300 evaluation carrying bootstrap 95% CIs and paired significance tests, so the
method comparison is defensible.

## Setup

- **Sample:** slice [0,300) of the 2000-example eval set. Running all 2000 × 4
  methods is ~13h+; n=300 tightens the 95% CI on judge_correct to ~±0.05 (vs ±0.13
  at n=50), enough to separate methods. keep_ratio=0.3, prune cache disabled.
- **Methods:** `no_pruning`, `clip_safecrop`, `downscale_baseline`,
  `clip_safecrop_downscale` (hybrid). Each run via `scripts/run_method.py` →
  per-method judged JSON in `data/mmdocrag/analysis/runs/`.
- **Stats:** `scripts/compare_methods_ci.py` (bootstrap CI per method; paired
  per-example difference vs a reference, CI excluding 0 ⇒ significant). Tokens
  measured on the target model's processor.

## Headline: the n=50 slice was unrepresentative

At n=50, full-image `no_pruning` scored only **0.36** correct and everything looked
identical within noise. At **n=300 it scores 0.717** — the [0,50) window was an
anomalously hard slice. The earlier "all methods ~0.32, indistinguishable" was a
small-sample artifact. With real headroom, the methods now separate.

## Results (n=300, keep_ratio=0.3)

| method | judge_correct [95% CI] | visual tokens | token cut | total_sec |
|---|---|---|---|---|
| no_pruning | 0.717 [0.667, 0.767] | 1505 | — | 5.1 |
| downscale_baseline | 0.710 [0.657, 0.760] | **452** | **−70%** | **4.7** |
| clip_safecrop | 0.687 [0.633, 0.740] | 949 | −37% | 9.6 |
| clip_safecrop_downscale (hybrid) | 0.663 [0.610, 0.717] | 444 | −71% | 9.2 |

**Paired judge_correct difference vs `downscale_baseline`** (95% CI; excludes 0 ⇒ significant):

| method | mean diff | 95% CI | significant? |
|---|---|---|---|
| no_pruning | +0.007 | [−0.023, +0.037] | no |
| clip_safecrop | −0.023 | [−0.063, +0.013] | no |
| clip_safecrop_downscale | −0.047 | [−0.080, −0.013] | **yes (worse)** |

## Conclusions

1. **Plain downscaling is the clear winner.** `downscale_baseline` cuts visual
   tokens **70%** (1505→452) with **no statistically detectable accuracy loss vs
   full images** (paired diff +0.007, CI includes 0), and it is the fastest method
   (4.7s). This is a clean, positive, defensible result.
2. **Query-aware cropping (`clip_safecrop`) is dominated.** It uses ~2× the tokens
   of downscaling (949 vs 452) and is ~2× slower (9.6s vs 4.7s, the CLIP tile-scoring
   tax) for no accuracy advantage (its CI overlaps everything).
3. **The hybrid is significantly worse.** At the same token budget as downscaling
   (444 vs 452), cropping-then-downscaling loses 4.7 accuracy points — the only
   *significant* gap in the study (CI excludes 0). See the hybrid report for why.
4. **Accuracy is flat among full-image / downscale at this budget; the real lever is
   token cost and latency**, where downscaling wins decisively.

## Takeaway for the project's north star

To reduce visual tokens without hurting answer quality, the evidence says: **just
downscale the retrieved images** — 70% fewer tokens at no measurable accuracy cost,
and faster than any query-aware path. The query-conditioned cropping machinery
(CLIP scoring, 7B attention) does not earn its place on this benchmark.

## Artifacts

- `data/mmdocrag/analysis/runs/*_judged.json` (per-method, n=300, with rows)
- `data/mmdocrag/analysis/methods_ci.json` (CI table + paired diffs)
- Stats: `rag/metrics.py:bootstrap_ci` / `paired_diff_ci`; tooling:
  `scripts/run_method.py`, `scripts/compare_methods_ci.py`

## Not done / follow-ups

- Did not scale to the full 2000 (n=300 was sufficient to separate methods; 2000
  would only narrow CIs already tight enough for the conclusions).
- A keep_ratio sweep for the hybrid was left optional (the keep=0.3 point is already
  decisive). See the hybrid report.
