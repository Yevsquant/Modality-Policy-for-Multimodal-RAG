# Project Summary: Reducing Visual Tokens After Multimodal Retrieval

**Date:** 2026-06-19
**Branch:** `clip-safecrop-token-measurement`
**Scope:** consolidates the per-phase reports in this folder (Phase 0–4, Angles 1–2)
into one narrative + the headline result.

## TL;DR (thesis)

The project's single goal is to **reduce the visual tokens a VLM consumes after
multimodal retrieval, without hurting answer quality or net latency.** Rigorously
measured, the central finding is:

> **On detail-sensitive tasks, in-model token pruning (FastV) and input-side resolution
> reduction trade along one accuracy/compute frontier. FastV's apparent superiority is
> an artifact of measuring *deep-layer tokens only*; on honest total compute (FLOPs and
> measured latency) its edge nearly vanishes, because it still encodes the full image.
> Cheap input-side reduction is the efficient lever — but no cheap signal we tried
> (pooled CLIP, per-patch CLIP, single-shot crop) is "spatially smart" enough to beat
> uniform downscaling. That gap is the open problem.**

## Why the pivot

The original system did query-conditioned image pruning + a disk cache on **MMDocRAG**
(document RAG). Once we built honest measurement (token counts on the *target* model;
n≥300; bootstrap + paired CIs), MMDocRAG turned out to be the **wrong benchmark to
prove a visual-token method**: its accuracy/token frontier is **flat** — downscaling to
150 tokens (a 90% cut) loses nothing, because the answer lives in the retrieved *text*
quotes and the image is nearly redundant. A method can't demonstrate value where the
signal it compresses is already free to throw away. So we repositioned to benchmarks
where **visual tokens are load-bearing** (detail-sensitive, high-resolution VQA), with a
hard go/no-go gate.

## Methodology (what makes the numbers trustworthy)

- **Token counts on the model that actually answers** (`rag/visual_token_counter.py`),
  not a proxy.
- **n≥191 (V*Bench) / 300 (DocVQA, HR-Bench); bootstrap 95% CIs and paired
  per-example significance** (`rag/metrics.py`).
- **Two target models, used deliberately:** the dataset stress tests run on the served
  answer model **Qwen3-Omni-30B** (vLLM); the in-model-pruning (FastV) study runs on
  **Qwen2-VL-7B** in HF, because vLLM can't expose early-layer attention — and this also
  fixes the old "7B signal vs 30B answer" mismatch by putting pruning on the model that
  answers.
- **Honest negatives reported as findings.** Most phases below are negative results;
  they are stated plainly with CIs.

## Findings (phase by phase)

| # | question | result |
|---|---|---|
| **0 gate** | Does any dataset stress visual tokens? | **V\*Bench: yes** (30B acc 0.885→0.623 as tokens 3406→342; significant; fat oracle-budget tail = real per-example variance). **DocVQA / HR-Bench: weak** (downscale-robust, flat). V\*Bench is the substrate. |
| **1** | Does the MMDocRAG winner `trim_downscale` transfer? | **No** — no-op on V\*Bench (natural photos have no blank margins), neutral on DocVQA. The trim win was margin-specific to documents. |
| **2** | Can a learned global-scalar budget policy beat uniform downscale? | **No** (−0.058 SIG at matched tokens). Pooled CLIP image+text features have **~0 signal** on per-example downscale sensitivity (out-of-fold AUC ≈ 0.50). The oracle headroom is real (0.942 @ 868 tok), but the features can't find it. |
| **3** | Does input downscaling compose with FastV? | **Negatively.** FastV is ~free at full res (halve image tokens, −0.005 n.s.) but **significantly harmful after downscaling** (−0.126/−0.136). The two target overlapping redundancy. At matched *deep tokens*, FastV beats input-downscale (+0.099 @255 tok, SIG). |
| **4** | Do *spatial* CLIP features fix Phase 2? More data? | **No** to both. Per-patch CLIP relevance features still AUC ≈ 0.5; HR-Bench has weak budget variance; combined data only lifts the policy from "SIG worse" to "n.s." vs static. |
| **A2** | Can a query-conditioned **crop** beat FastV at the input? | **No** — ties uniform downscale, SIG worse than FastV @255 tok. Cheap localizers miss small objects (the `direct_attributes` split collapses to downscale level); coarse-to-fine ≈ CLIP. Crop only ties on `relative_position` (layout). |
| **A1** | Is "FastV beats downscale" robust to a fair cost axis? | **No — it's an axis artifact.** See below. |

## The headline: Angle 1 (fair-cost reframing)

The Phase-3/4 comparison matched **deep-layer** visual tokens. But FastV encodes the
**full** image (vision tower + first K LLM layers) and prunes only afterwards, while
input-downscale feeds fewer tokens to *everything*. Re-scoring on honest total compute
(analytical FLOPs **and** measured wall-clock on the 7B):

| FastV advantage over input-downscale | r=0.25 | r=0.5 |
|---|---|---|
| deep-layer tokens (the original axis) | +0.099 | +0.052 |
| analytical total FLOPs (matched) | +0.015 | +0.026 |
| **measured latency (matched)** | **+0.002** | **+0.019** |

The vision tower is a **constant 36.8 ms = 51–55% of FastV's latency** — a cost FastV
cannot avoid. The deep-token framing overstated FastV's advantage **~25–50×**; at the
aggressive operating point its real-latency edge is **practically zero**, and input-side
downscaling **owns the entire low-compute regime**.

## Synthesis

1. **Measure visual-token methods on a cost axis that includes the vision encoder.**
   "Deep-layer tokens" flatters in-model pruning (FastV/CATP) and unfairly penalizes
   input-side reduction. This single methodological correction overturns the headline.
2. **Input-side reduction is the efficient lever** on these models, because it saves the
   encoder + all layers, not just deep layers.
3. **But making it adaptive is hard.** Every cheap controller failed: a global-scalar
   budget (Phase 2/4) because cheap CLIP can't predict downscale sensitivity; a
   query-conditioned crop (Angle 2) because single-shot localization misses small objects
   on visual-search data. Uniform downscaling is a stubbornly strong baseline.
4. **The open problem** is a *cheap, spatially-aware* input-side policy that recovers
   FastV's small remaining edge without paying FastV's full-encode cost.

## Limitations / honest caveats

- Stress tests are on the 30B; the FastV/crop/cost studies are on the 7B (GPTQ-Int4).
  Absolute accuracies differ by model; the *comparisons* are within-model.
- Angle 1's latency is measured on one GPU with composed primitives (vision tower +
  decoder-layer stack, lm_head/embedding excluded); FLOPs are analytical. Direction is
  robust across three axes; the exact crossover is hardware/kernel-dependent.
- V\*Bench n=191 is small; the learned-policy negatives use LOO/k-fold CV but a larger
  detail benchmark with genuine budget variance would strengthen them.
- Bugs caught and fixed en route (documented): fp16-GPTQ + eager attention → NaN logits
  (→ silent constant predictions); the Qwen2-VL processor re-clamping pre-downscaled
  images (→ the input-keep axis silently did nothing).

## What's next (if continued)

- A measured-latency-*aware* adaptive input policy (optimize accuracy per ms, not per
  token) — the cost axis that actually matters.
- Stronger-but-still-cheap localization (tiny open-vocab grounder) for the foveated crop,
  evaluated on the fair cost axis (Angle 2 × Angle 1).
- Angle 3 (crop→FastV) on the fair axis — weakened by Angle 2 but untested on cost.

## Artifact map

- **Reports (this folder):** `…phase0-downscale-stress-gate`, `…phase2-learned-budget-policy`,
  `…phase3-fastv-composition`, `…phase4-spatial-features-and-data`,
  `…angle2-foveated-crop-vs-fastv`, `…angle1-fair-compute-reframing`.
- **Plans:** `thoughts/plans/2026-06-19-adaptive-visual-tokenization-repositioning.md`,
  `…-preprocess-vs-fastv.md`.
- **Code:** `rag/{visual_token_counter,metrics,image_ops,vqa_scoring,vqa_datasets,fastv,
  budget_features,budget_policy,flops_model}.py`; `scripts/{downscale_stress_test,
  analyze_stress_test,run_fastv_matrix,analyze_fastv,train_budget_policy,
  run_foveated_crop,angle1_flops_reframing,angle1_measure_latency}.py`. 57 tests in `tests/`.
