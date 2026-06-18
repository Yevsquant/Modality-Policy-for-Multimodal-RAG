# Work Report: `clip_safecrop_downscale` Hybrid Mode

**Date:** 2026-06-18
**Branch:** `clip-safecrop-token-measurement`
**Goal:** Test whether combining query-aware cropping with downscaling beats plain
downscaling at an equal token budget — i.e. does focusing on the relevant region
add value once token cost is controlled?

## Motivation

Earlier sweep (slice [0,50)) showed `clip_safecrop`'s token count **floors at ~1042**
(~65% of the full image): its single bounding box stays large when relevant content
is spatially scattered, so it cannot reach a low token budget. The hypothesis: crop
to the query-relevant bbox to drop irrelevant regions, **then** downscale that crop
to the target budget — getting both query-focus and a low token count.

## Design (implemented in `rag/pruner.py`)

`clip_safecrop_downscale`:
1. CLIP-score the tile grid vs the query; take the bounding-box union of the top
   tiles (same as `clip_safecrop`) → query-relevant crop.
2. Downscale that crop by `_area_budget_factor(crop_area, full_area, keep_ratio)` =
   `sqrt(keep_ratio·full_area / crop_area)` (clamped ≤1), so the crop's area — hence
   its token count — matches the `keep_ratio` budget that `downscale_baseline` uses.

This makes hybrid and downscale directly comparable at the **same token budget**;
the only difference is hybrid first discards the non-relevant border via cropping.
Unit-tested (`_area_budget_factor`) and functionally verified with real CLIP
(16→5 tokens at keep=0.3, matching the downscale budget).

## Result (n=300, keep_ratio=0.3) — hypothesis REJECTED

| method | judge_correct [95% CI] | visual tokens | total_sec |
|---|---|---|---|
| downscale_baseline | 0.710 [0.657, 0.760] | 452 | 4.7 |
| clip_safecrop_downscale | 0.663 [0.610, 0.717] | 444 | 9.2 |

**Paired difference (hybrid − downscale): −0.047, 95% CI [−0.080, −0.013] →
SIGNIFICANT.** At essentially identical token budgets (444 vs 452), the hybrid is
**significantly worse** than plain downscaling — and slower (9.2s vs 4.7s, the CLIP
scoring tax). It is the only significant accuracy gap in the whole study.

## Interpretation

Cropping to the query-relevant bbox **actively removes context the model uses**. On
document images, peripheral content (surrounding text, table headers, captions,
layout cues) contributes to the answer; discarding it before downscaling loses more
than it saves. Plain downscaling keeps the *whole* page (global layout intact) at
lower resolution, which the VLM handles better than a tight high-detail crop. So
query-aware spatial selection is the wrong lever here — the bottleneck isn't "too
much irrelevant area," it's resolution, and uniform downscaling addresses that
without throwing away context.

## Verdict

The hybrid does not justify itself: at equal tokens it is significantly less accurate
than the trivial baseline and ~2× slower. Combined with the full-eval report, the
consistent message across `clip_safecrop`, `clip_safecrop_downscale`, and the 7B
`safecrop`/`cluster` modes is that **query-conditioned cropping does not help on
MMDocRAG; uniform downscaling is the better token-reduction lever.**

## Honest framing for a resume

This is a rigorous *negative* result with a *significant* finding and a strong
practical recommendation, backed by n=300 + bootstrap CIs + a paired significance
test. The contribution is the controlled, equal-token-budget comparison that
isolates the effect of query-aware cropping and shows it to be net-harmful — and the
measurement apparatus (target-model token counting, CI tooling) that makes the claim
trustworthy.

## Follow-ups (optional)

- A keep_ratio sweep (hybrid vs downscale across budgets) to confirm hybrid never
  wins — left undone since the equal-budget keep=0.3 point is already significant and
  decisive.
- If pursuing query-awareness further, the lever to test is *resolution allocation*
  (keep the whole page but spend more pixels on the relevant region), not *cropping*.
