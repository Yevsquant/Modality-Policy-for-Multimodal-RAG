# Work Report: Phase 3 — Input-level × In-model Pruning Composition (FastV)

**Date:** 2026-06-19
**Branch:** `clip-safecrop-token-measurement`
**Plan:** [thoughts/plans/2026-06-19-adaptive-visual-tokenization-repositioning.md](../plans/2026-06-19-adaptive-visual-tokenization-repositioning.md) (Q2)
**Goal:** Does input-level downscaling compose with in-model visual-token pruning
(FastV)? Do the savings stack, and does downscaling hurt FastV on detail-sensitive
data (V\*Bench)?

## Method

A faithful FastV implementation on **Qwen2-VL-7B (GPTQ-Int4, HF)** — the same model
that produces the answer *and* the pruning attention signal (fixing the old
30B-answer / 7B-signal mismatch). `rag/fastv.py`:

1. One full forward (eager attn) with `output_attentions` + `output_hidden_states`.
2. Rank image tokens by attention received from the last query token at layer K;
   keep all text + top-`keep_ratio` image tokens.
3. Re-run only `layers[K:]` (+ norm + lm_head) on the gathered token set — rotary
   `position_embeddings` captured via a hook and gathered by index, fresh causal mask.
   Faithful FastV: early layers see the full image, deep layers see the pruned set.
4. Score V\*Bench multiple-choice by the argmax over {A,B,C,D} option-letter logits at
   the last position (single prefill; no generation/KV-cache).

`select_keep_indices` is unit-tested (`tests/test_fastv_select.py`). Resolution is
controlled by fitting each image to a ~1024-token base then applying the input-level
keep ratio, so the input axis genuinely reduces tokens (an earlier bug where the
processor re-clamped both full and downscaled images to the same budget was fixed).

Matrix (K=3): input {full ~1020 tok, downscale@0.3 ~298 tok} × in-model {none, FastV
r=0.5, r=0.25}. n=191 V\*Bench, bootstrap 95% CIs + paired per-example diffs
(`scripts/analyze_fastv.py`).

## Result (n=191, V\*Bench)

| condition | img tokens | accuracy [95% CI] |
|---|---|---|
| full | 1018 | 0.607 [0.534, 0.675] |
| full + FastV r=0.5 | 509 | 0.602 [0.534, 0.670] |
| full + FastV r=0.25 | 255 | 0.576 [0.508, 0.644] |
| downscale@0.3 | 297 | 0.518 [0.450, 0.592] |
| downscale@0.3 + FastV r=0.5 | 148 | 0.393 [0.325, 0.461] |
| downscale@0.3 + FastV r=0.25 | 75 | 0.382 [0.314, 0.450] |

Paired tests (delta = a − b, 95% CI):

| test | Δ [CI] | verdict |
|---|---|---|
| FastV r=0.5 at full res | −0.005 [−0.042, +0.031] | **free** (n.s.) |
| FastV r=0.25 at full res | −0.031 [−0.089, +0.021] | ~free (n.s.) |
| FastV r=0.5 after downscale | −0.126 [−0.194, −0.058] | **SIG worse** |
| FastV r=0.25 after downscale | −0.136 [−0.215, −0.058] | **SIG worse** |
| downscale@0.3 (no FastV) vs full | −0.089 [−0.157, −0.026] | **SIG worse** |
| stacked (ds0.3+FastV0.5) vs full | −0.215 [−0.288, −0.136] | **SIG worse** |

## Bottom line

**The two compression axes do NOT stack — they interact negatively.**

1. **In-model pruning (FastV) is nearly free at full resolution.** Halving the image
   tokens (1018→509) costs nothing significant (−0.005), and quartering (→255) costs
   only −0.031 (n.s.). At native resolution the deep layers genuinely don't need most
   visual tokens — FastV is the cheaper, safer lever.
2. **The same FastV pruning becomes significantly harmful once you downscale**
   (−0.126 / −0.136, CIs exclude 0). Input downscaling already discarded the
   fine detail; dropping more visual tokens on top compounds the loss. The methods
   target overlapping redundancy, so combining them double-counts and destroys signal.
3. **Stacking both aggressively is the worst option** (ds0.3+FastV0.5 = −0.215 vs full):
   75–148 tokens at a large accuracy cost, strictly dominated by FastV-at-full-res
   (255 tokens, −0.031) on the accuracy/token frontier.

**Practical recommendation:** on detail-sensitive data, prefer **in-model pruning at
native resolution** over input-level downscaling, and do **not** combine the two
aggressively. This also validates the project's pivot direction: query-conditioned
*in-model* token selection (FastV-family) is a stronger lever than uniform input
downscaling — and an adaptive policy (Phase 2) that decides per-token keep at full
resolution is the right place to push next.

A methodological note: fp16 GPTQ + *eager* attention produces NaN logits over ~1k
tokens (silently → constant argmax). The fix was SDPA for the stable forward/deep
re-run, with eager used only for the shallow, finite layer-K ranking signal. An
earlier version that scored from NaN logits gave a spurious "all conditions identical"
result; the fix (and a token-budget fix where the processor re-clamped downscaled
images) was required before the numbers above were valid.

## Artifacts

- `rag/fastv.py`, `scripts/run_fastv_matrix.py`, `scripts/analyze_fastv.py`,
  `tests/test_fastv_select.py`.
- `data/vqa_stress/fastv_vstar.jsonl`, `data/vqa_stress/fastv_vstar_report.json`.
