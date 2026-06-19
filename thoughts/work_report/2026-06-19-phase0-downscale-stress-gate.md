# Work Report: Phase 0 — Downscale Stress-Test Gate (V*Bench + DocVQA)

**Date:** 2026-06-19
**Branch:** `clip-safecrop-token-measurement`
**Plan:** [thoughts/plans/2026-06-19-adaptive-visual-tokenization-repositioning.md](../plans/2026-06-19-adaptive-visual-tokenization-repositioning.md)
**Goal (Phase 0 gate):** Find a dataset where naive downscaling *provably hurts*
accuracy — the precondition (absent on MMDocRAG, whose frontier was flat to 150
tokens) for a learned adaptive-budget policy to have anything to learn.

## Method

Single-image VQA, no RAG retrieval. For each example, downscale the image to a
keep_ratio ladder {1.0, 0.5, 0.3, 0.2, 0.1}, send to the target model
(Qwen3-Omni-30B via vLLM), and score with the dataset-native metric. Visual tokens
counted on the target model's image processor (`rag/visual_token_counter.py`).
Bootstrap 95% CIs + paired per-example differences vs full-res
(`rag/metrics.py`). New harness:

- `rag/image_ops.py` — in-memory `downscale_to_keep` / `trim_downscale` (reuse the
  pruner's trim geometry); `rag/vqa_scoring.py` — MC exact-match (V*Bench) + ANLS
  (DocVQA); `rag/vqa_datasets.py` — V*Bench + DocVQA loaders.
- `scripts/downscale_stress_test.py` (resumable JSONL driver),
  `scripts/analyze_stress_test.py` (gate report). Unit tests: `tests/test_image_ops.py`,
  `tests/test_vqa_scoring.py` (9 tests, green).

## Result — GATE PASSES on both datasets

### V*Bench (n=191) — strong stress

| keep | visual tokens | accuracy [95% CI] | paired Δ vs full |
|---|---|---|---|
| 1.0 | 3406 | 0.885 [0.838, 0.927] | — |
| 0.5 | 1702 | 0.848 [0.796, 0.895] | −0.037 [−0.079, +0.005] n.s. |
| 0.3 | 1027 | 0.770 [0.707, 0.827] | **−0.115 [−0.173, −0.063] SIG** |
| 0.2 |  677 | 0.723 [0.660, 0.785] | **−0.162 [−0.225, −0.099] SIG** |
| 0.1 |  342 | 0.623 [0.555, 0.691] | **−0.262 [−0.335, −0.188] SIG** |

Monotonic, significant decline — the **exact opposite of MMDocRAG's flat frontier**.
Downscaling provably destroys accuracy here (small objects in large scenes need
resolution).

### DocVQA (n=300) — passes, but only at the extreme budget

| keep | visual tokens | accuracy (ANLS) [95% CI] | paired Δ vs full |
|---|---|---|---|
| 1.0 | 3720 | 0.949 [0.927, 0.970] | — |
| 0.5 | 2005 | 0.955 [0.934, 0.974] | +0.006 n.s. |
| 0.3 | 1286 | 0.964 [0.945, 0.980] | +0.015 [+0.002, +0.031] |
| 0.2 |  861 | 0.929 [0.902, 0.953] | −0.021 [−0.044, +0.002] n.s. |
| 0.1 |  431 | 0.860 [0.820, 0.895] | **−0.089 [−0.125, −0.056] SIG** |

Qwen3-Omni is robust to moderate document downscaling (flat/slightly-better to
keep=0.3) and only breaks **significantly at keep=0.1**. The gate passes, but the
stress is mild — consistent with the plan's risk note that DocVQA might barely fail.

## Budget variance (the precondition for a learned policy)

Per example, the smallest keep_ratio still correct ("oracle budget"):

- **V*Bench:** 0.1→119, 0.2→26, 0.3→11, 0.5→14, 1.0→10, unsolved→11. **Fat tail** —
  ~30% of examples need more than the minimum budget; a meaningful fraction need
  full resolution. Strong variance to exploit.
- **DocVQA:** 0.1→267, 0.2→22, 0.3→6, unsolved→5. Most examples need almost nothing;
  thin tail. Weak variance.

## Bottom line / decision

**Gate PASS.** The repositioning is validated: there exist benchmarks where visual
tokens are load-bearing, unlike MMDocRAG. **V*Bench is the primary substrate** — large
accuracy swings and a fat oracle-budget tail make it the right home for the adaptive
policy (Phase 2/3). **DocVQA is a weak-stress confirmation**: useful as a "dense-text"
contrast where this 30B model is already downscale-robust, but it offers little budget
variance to learn from. Phase 1 then showed the MMDocRAG trim win does not transfer
(no-op on V*Bench photos, neutral on DocVQA), redirecting effort to Phase 2 (oracle
labeling + learned query-conditioned policy), centered on V*Bench.

## Phase 1 — trim_downscale vs plain downscale at equal budget

Tested the MMDocRAG winner (`trim_downscale`: trim near-uniform borders, then
downscale to budget) at the budgets where plain downscale significantly hurt.
**The MMDocRAG trim win does NOT replicate on detail data.**

| dataset | keep | downscale | trim | paired Δ (trim − down) |
|---|---|---|---|---|
| V*Bench | 0.3 | 0.770 | 0.770 | +0.000 [+0.000, +0.000] |
| V*Bench | 0.2 | 0.723 | 0.723 | +0.000 |
| V*Bench | 0.1 | 0.623 | 0.623 | +0.000 |
| DocVQA | 0.3 | 0.964 | 0.958 | −0.006 [−0.014, +0.000] |
| DocVQA | 0.2 | 0.929 | 0.923 | −0.006 [−0.019, +0.006] n.s. |
| DocVQA | 0.1 | 0.860 | 0.858 | −0.002 [−0.017, +0.013] n.s. |

- **V*Bench: trim is a no-op** — natural photographic scenes have no near-uniform
  borders, so `_trim_bbox` finds nothing and `trim_downscale` degenerates *exactly*
  to plain downscale (identical tokens, identical scores).
- **DocVQA: trim is neutral-to-slightly-worse** (all CIs include or touch 0). Trimming
  document margins doesn't recover accuracy when the model is already downscale-robust,
  and occasionally clips a relevant edge.

**Interpretation:** the MMDocRAG trim win was specific to MMDocRAG's document images
having large blank margins; it is **not a general lever**. On detail-sensitive data the
lever is not "drop the margins" but "spend resolution where the query needs it" — i.e.
the query-conditioned adaptive policy (Phase 2/3). The V*Bench oracle-budget tail
(~30% of examples need >minimum) is where that signal lives. This *strengthens* the
case for the learned policy: static, query-agnostic tricks don't transfer.

## Artifacts

- `data/vqa_stress/{vstar,docvqa}_downscale.jsonl`, `..._trim.jsonl`
- `data/vqa_stress/{vstar,docvqa}_gate.json`, `..._trim_vs_downscale.json`
- New code: `rag/image_ops.py`, `rag/vqa_scoring.py`, `rag/vqa_datasets.py`,
  `scripts/downscale_stress_test.py`, `scripts/analyze_stress_test.py`,
  `scripts/compare_transforms.py`; tests `tests/test_image_ops.py`, `tests/test_vqa_scoring.py`.
- Server/run logs in `logs/phase0*.log`, `logs/phase1.log` (data/ and logs/ gitignored).
