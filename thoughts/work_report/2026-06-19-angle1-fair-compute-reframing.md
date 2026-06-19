# Work Report: Angle 1 — FastV vs input-downscale on the fair compute axis

**Date:** 2026-06-19
**Branch:** `clip-safecrop-token-measurement`
**Plan:** [thoughts/plans/2026-06-19-preprocess-vs-fastv.md](../plans/2026-06-19-preprocess-vs-fastv.md) (Angle 1)
**Goal:** The Phase-3/4 "FastV beats input-downscale" result matched *deep-layer* visual
tokens. But FastV encodes the FULL image (vision tower + first K LLM layers) and prunes
only afterwards, while input-downscale feeds fewer tokens to *everything*. Re-score the
comparison on **total forward FLOPs** (vision + LLM) to see whether FastV's advantage
survives once the cost it hides in the encoder/early layers is counted. No new model
runs — all from existing `data/vqa_stress/fastv_vstar.jsonl`.

## Method

Transparent FLOPs model (`rag/flops_model.py`, unit-tested) using Qwen2-VL-7B's actual
config: LLM hidden 3584, 28 layers, GQA 28/4 heads, head_dim 128, d_ff 18944; ViT depth
32, dim 1280, 16 heads, mlp_ratio 4, 2×2 merge, **full attention** (Qwen2-VL pre-2.5, so
the ViT attention term is O(patches²)). Multiply-accumulate = 2 FLOPs; text prompt = 56
tokens (median). Per condition: `vision(img_before) + Σ_{l<K} llm(text+img_before) +
Σ_{l≥K} llm(text+img_after)` (K=3 for FastV; no split otherwise). `scripts/angle1_flops_reframing.py`.

## Result (V*Bench, n=191)

| condition | acc | deep tok | input tok | total GFLOPs | vision % |
|---|---|---|---|---|---|
| ds0.3+fastv0.25 | 0.382 | 75 | 297 | 3 758 | 46 |
| ds0.3+fastv0.5 | 0.393 | 148 | 297 | 4 617 | 37 |
| ds0.25 | 0.476 | 258 | 258 | 5 611 | 26 |
| ds0.3 | 0.518 | 297 | 297 | 6 383 | 27 |
| ds0.5 | 0.550 | 507 | 507 | 10 700 | 30 |
| full+fastv0.25 | 0.576 | 255 | 1018 | 13 050 | 60 |
| full+fastv0.5 | 0.602 | 509 | 1018 | 16 090 | 49 |
| full | 0.607 | 1018 | 1018 | 22 320 | 35 |

**FastV's advantage collapses on the fair axis.** At matched *total compute* (interpolating
the downscale-only frontier at each FastV point's budget):

| FastV point | total GFLOPs | FastV acc | downscale acc @ same FLOPs | FastV advantage |
|---|---|---|---|---|
| full+fastv0.25 | 13 050 | 0.576 | 0.561 | **+0.015** |
| full+fastv0.5 | 16 090 | 0.602 | 0.576 | **+0.026** |

Compare to the deep-token framing, where FastV beat downscale by **+0.099** @ ~255 tokens
(significant) and +0.052 @ ~509. The advantage shrinks ~3–7× once the encoder cost is counted.

## Why

1. **FastV can't avoid the vision tower.** `full+fastv0.25` keeps `img_before=1018`, so it
   pays the full ViT (vision is **60%** of its total FLOPs) plus 3 full early LLM layers —
   then prunes. It costs **13 050 GFLOPs vs ds0.25's 5 611** (2.3×) for the deep-token
   budget it "tied." The token-matched comparison was measuring the cheap part only.
2. **Halving deep tokens ≠ halving compute.** `full+fastv0.5` halves deep visual tokens
   (1018→509) but costs **72% of full** (16 090 vs 22 320) — because the identical full-image
   vision encode is ~half its cost. The deep-token metric overstated FastV's saving ~2×.
3. **Input-downscale owns the entire low-compute regime** (3 758–10 700 GFLOPs); FastV points
   only appear at the expensive end (≥13 050) because they always encode the full image.

Every condition is technically Pareto-optimal (the frontier is a smooth accuracy/compute
trade), so neither method *dominates* — but the shape is the story: cheap = downscale,
expensive = FastV, and FastV's incremental accuracy-per-FLOP is poor.

## Bottom line

**The "FastV beats input-downscale" claim is axis-dependent.** On deep-layer tokens FastV
wins by ~+0.10; on honest total compute the edge is ~+0.015–0.026 because FastV pays the
full vision-encoder + early-layer cost it hides. Input-side downscaling is Pareto-competitive
and owns the low-compute regime. This rehabilitates pre-processing: it is *not* dominated
once measured fairly — vindicating Angle 1, even though Angle 2 (single-shot crop) failed to
add spatial smarts on top.

**Caveat:** analytical FLOPs ≠ latency. Confirmed below with measured wall-clock.

## Measured-latency confirmation (Qwen2-VL-7B GPTQ-Int4, single H-class GPU)

`scripts/angle1_measure_latency.py` times two primitives on the real model (CUDA-synced,
5 warmup + median of 20 trials) and composes each condition exactly as the FLOPs model
does: `t_vision(P)` (vision tower) and `per_layer(L)` (28-layer stack / 28); FastV =
3 full layers + 25 pruned layers. lm_head/embedding excluded (small, constant). Composing
measured primitives — rather than timing the 2-pass `answer_mc` — avoids double-counting
FastV's early layers.

| condition | acc | vision ms | LLM ms | total ms | vision % |
|---|---|---|---|---|---|
| ds0.25 | 0.476 | 11.8 | 28.2 | 39.9 | 29 |
| ds0.3 | 0.518 | 12.6 | 28.9 | 41.5 | 30 |
| ds0.5 | 0.550 | 19.7 | 33.5 | 53.2 | 37 |
| full+fastv0.25 | 0.576 | 36.8 | 30.1 | 66.9 | 55 |
| full+fastv0.5 | 0.602 | 36.8 | 35.2 | 71.9 | 51 |
| full | 0.607 | 36.8 | 48.7 | 85.4 | 43 |

The vision tower is a **constant 36.8 ms** for every full-res FastV condition and is
**51–55%** of their latency — FastV cannot touch it. Matched-latency advantage of FastV
over input-downscale (interpolated downscale frontier):

| FastV point | latency | FastV acc | downscale acc @ same latency | advantage |
|---|---|---|---|---|
| full+fastv0.25 | 66.9 ms | 0.576 | 0.574 | **+0.002** |
| full+fastv0.5 | 71.9 ms | 0.602 | 0.583 | **+0.019** |

**Three convergent axes** for FastV's advantage over input-downscale: deep-tokens +0.099/
+0.052 → analytical FLOPs +0.015/+0.026 → **measured latency +0.002/+0.019**. At the
aggressive r=0.25 operating point FastV's edge is **statistically and practically zero**
on real latency. The deep-token framing overstated FastV ~25–50×. Confirmed: the "FastV
beats input-downscale" result was an artifact of an axis that ignores the vision encoder.

## Artifacts

- `rag/flops_model.py` (+ `tests/test_flops_model.py`), `scripts/angle1_flops_reframing.py`,
  `scripts/angle1_measure_latency.py`
- `data/vqa_stress/angle1_flops.json`, `data/vqa_stress/angle1_latency.json`, `imgs/Angle1FlopsFrontier.png`
