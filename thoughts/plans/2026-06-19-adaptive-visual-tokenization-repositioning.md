# Plan: Repositioning toward query-conditioned adaptive visual tokenization

**Date:** 2026-06-19
**Status:** proposal (strategic pivot) — answers three questions raised by the user
**Context:** On MMDocRAG, the downscale frontier is **flat from 150 to 1505 tokens**
(every CI overlaps) and `trim_downscale` only edges the baseline by +0.030. The
project has hit a ceiling not because the method is good but because **the benchmark
does not stress visual tokens** — the answer is carried by retrieved text quotes, so
the image is nearly redundant and 90% of it can be discarded for free. See
[[trim-downscale-wins-images-barely-matter]] and [[eval-slice-0-50-is-unrepresentatively-hard]].

## Core diagnosis (why all three questions point the same way)

A visual-token-reduction method is only *interesting* on data where **naive downscaling
hurts** — i.e. where the optimal budget varies across (image, query) pairs and detail
is load-bearing. MMDocRAG is the opposite: a flat frontier means zero budget variance,
so:

- **Q3** (learn a dynamic detail-preserving policy) has nothing to learn — the oracle
  budget is "minimum, everywhere."
- **Q2** (compose with FastV/CATP) has nothing interesting to compose — there are no
  high-value tokens to protect.
- **Q1** (switch datasets) is therefore not optional; it is the **prerequisite** that
  makes Q2 and Q3 meaningful.

The whole plan hinges on one go/no-go gate: **find a dataset where downscaling
provably costs accuracy.** Everything else is downstream.

---

## Q1 — Discard the RAG framing? Switch dataset? (Decision)

**Recommendation: reframe the project, but do NOT pick VQAv2.**

### Reframe, don't discard
Make the project's identity **"query-conditioned adaptive visual token budgeting"**, not
"RAG + disk cache." Keep MMDocRAG as *one evaluation setting* — specifically the
"visual tokens are redundant" contrast point (a useful negative result: shows the
method correctly spends ~nothing when the image doesn't matter). Drop from the headline:
the disk-cache reuse index and the 7B CATP-crop proxy (both are off-goal and the
reflection already flagged the 7B-vs-30B mismatch). **Keep** the measurement apparatus
— that is the strongest asset:
- `rag/visual_token_counter.py` (target-model token counting),
- `rag/metrics.py` bootstrap/paired CIs,
- the `trim_downscale` / `downscale_baseline` modes and the resumable
  `run_method.py` / `compare_methods_ci.py` harness.

### Why not VQAv2
VQAv2 has the *same failure mode* as MMDocRAG for our purpose: it is 224px-tolerant,
saturated, and many answers follow from coarse gist or language priors. Downscaling
barely hurts → flat frontier again → nothing to learn. We would relearn the MMDocRAG
lesson on a different dataset.

### Pick detail-sensitive, high-resolution VQA (where downscaling provably hurts)
| dataset | why it stresses visual tokens | role |
|---|---|---|
| **V\*Bench (V-Star)** | small objects in large scenes; built to force high-res attention | primary — foveation showcase |
| **DocVQA / InfographicVQA** | dense small text; downscaling destroys legibility; answer IS in the image | primary — closest visual domain to MMDocRAG but vision is load-bearing |
| ChartQA | fine numeric/axis detail | extension |
| TextVQA | scene text needs resolution | extension |
| MMDocRAG (kept) | visual tokens redundant | contrast / negative control |

**Concrete pick:** **V\*Bench + DocVQA** as the two primaries (one "needle in a big
image," one "dense document"). They maximize budget variance → ideal substrate for the
learned policy (Q3) and the composition study (Q2).

### Decision gate (Phase 0 — must pass before anything else)
Run the **downscale stress test** on V\*Bench + DocVQA: sweep keep_ratio
{0.1,0.2,0.3,0.5,1.0}, n≥300, target-model token counts, bootstrap CIs.
- **PASS** if accuracy drops monotonically and significantly as tokens fall (the curve
  MMDocRAG failed to produce). → proceed to Q2/Q3.
- **FAIL** (flat frontier again) → the dataset is also redundant; do not invest in a
  learned policy there.

---

## Q2 — Does `trim_downscale` interact with in-model pruning (CATP / FastV)?

**Short answer: they are complementary and compose, but with a predictable diminishing
overlap — and on detail tasks a potential negative interaction. It is worth a clean
experiment, with one big engineering caveat.**

### Conceptual relationship
- `trim_downscale` acts at the **input / pixel** level: fewer patches ever reach the
  vision encoder. Cost saved includes the vision-encoder forward itself.
- FastV / CATP act **inside the LLM**: keep all input patches, then drop low-attention
  visual tokens at an intermediate layer k. Cost saved is later-layer LLM compute only;
  the encoder still runs on everything.
- They stack multiplicatively on token count: `tokens_final ≈ input_keep × inmodel_keep`.

### Hypotheses to test (not assume)
1. **Diminishing overlap:** trim removes blank margins up front — exactly the
   low-attention tokens FastV would also drop. So FastV's *marginal* benefit shrinks
   after trim, but trim is strictly cheaper for those tokens (never encoded). Expect
   trim+FastV < additive, but trim dominates on the encoder side.
2. **Negative interaction on detail:** global downscaling lowers resolution everywhere,
   including the high-attention region FastV wants to keep — so aggressive downscale +
   FastV may underperform FastV-alone on V\*Bench. This is the interesting failure mode
   and motivates Q3 (don't downscale the relevant region).
3. **Re-tuning:** FastV's prune layer/ratio likely needs re-tuning after trim (fewer
   redundant tokens remain).

### Engineering reality (flag up front)
vLLM (our Qwen3-Omni-30B serving path) does **not** expose per-token attention at an
early layer, so FastV/CATP cannot be cleanly instrumented on the answer model there.
Two options:
- **(recommended)** Use **HF transformers + Qwen2-VL-7B** (already loaded in this repo
  for the old CATP proxy) as the in-model-pruning testbed. FastV is ~a few lines:
  rank visual tokens by attention-to-text at layer k, keep top-r, drop the rest. This
  finally puts CATP/FastV on the *same* model that answers — fixing the 7B-vs-30B
  signal/answer mismatch the reflection complained about.
- Keep the 30B (vLLM) for input-level methods only.

### Experiment matrix
`{none, trim, trim+downscale}` × `{no in-model prune, FastV@layer3 r=0.5, CATP}` on
V\*Bench + DocVQA. **Metrics:** accuracy, *actual tokens processed* (encoder patches +
post-prune LLM tokens), and FLOPs/latency. Deliverable: a Pareto plot showing where
input-level vs in-model vs combined lands.

---

## Q3 — Train a small model for dynamic `trim_downscale` (the novel contribution)

**This is the strongest research direction and matches the user's earlier intent
("put the model at the very end, do less work, small model is enough"). It is viable
ONLY on a Phase-0-PASS dataset.**

### Problem framing
A lightweight **query-conditioned budgeting policy** `π(image, query) → resolution plan`
that preserves detail where the query needs it and downscales elsewhere. `trim_downscale`
is the static, query-agnostic special case; the learned policy generalizes it.

### Output-space options (start simple, extend)
1. **Global scalar budget** — predict one keep_ratio per (image,query). Easiest;
   regression head. Captures "this query needs high res, that one doesn't."
2. **Crop box + budget** — predict a bounding box (generalized trim) + a downscale
   factor. Layout-preserving (consistent with the project's "no stitching" rule).
3. **Foveated resolution map / multi-res tiling** — high-res tiles where relevant,
   low-res elsewhere. Most powerful and most novel ("query-conditioned foveated
   tokenization"), but needs care to keep positional layout valid for the encoder.

**Recommended path:** ship (1)→(2) first (clear win, low risk), then (3) as the
headline result if Phase 0 shows strong budget variance.

### Training signal (the crux — there is no ground-truth budget)
Build an **oracle by search**, then supervise the small model on it:
- For each (image, query): sweep budgets/crops, query the big VLM, record the
  **minimum-token plan that still answers correctly**. That tuple is the label.
- Train π to predict it. Loss = plan regression (+ optional RL fine-tune with reward
  `accuracy − λ·tokens`). Supervised-from-oracle is more stable than pure RL; keep RL
  as a later refinement.

### Small-model architecture ("less work at the end")
Reuse the **retriever's existing CLIP** image + text features (already computed,
`item["clip_relevance"]` is in `rag/retriever.py`) → a small MLP / ViT-tiny head that
emits the plan. No 7B, no full-attention capture. This is exactly the "small model is
enough, at the very end" design the user asked about, and it keeps the policy off the
latency-critical path.

### Evaluation
Pareto curve (accuracy vs tokens) of the learned policy vs: static `trim_downscale`,
plain downscale, FastV, and full image — on V\*Bench + DocVQA, n≥300 + CIs. The claim
to earn: **"matches full-image accuracy at materially fewer tokens than any static
budget, by spending resolution only where the query needs it."**

### Hard dependency / kill criterion
If Phase 0 budget variance is low even on the new dataset, the learned policy collapses
to a constant and adds nothing over static trim. The Phase-0 gate catches this before
any training investment.

---

## Phased roadmap (with gates)

| phase | work | gate / deliverable |
|---|---|---|
| **0** | Downscale stress test on V\*Bench + DocVQA (reuse current harness + token counter) | **GATE:** non-flat frontier, else stop |
| **1** | Port `trim_downscale` + downscale modes to the new dataset; replicate the trim win; establish static baselines | static Pareto baseline + n≥300 CIs |
| **2** | Oracle labeling (budget search) → train small policy (output spaces 1→2→3) | learned-policy Pareto beats static trim |
| **3** | In-model pruning testbed (Qwen2-VL-7B HF + FastV/CATP); composition matrix | input × in-model Pareto plot |

Sequencing note: **Q1+Q3 are the spine; Q2 is a strong add-on.** This is realistically
2–3 distinct results; do not attempt all in parallel.

## Risks & honest caveats
- **vLLM cannot host FastV cleanly** → Q2 must run on a 7B HF harness, not the 30B.
  Acknowledge the model differs from the input-level experiments.
- **Phase 0 could fail on DocVQA too** if the 30B is robust to downscaling — V\*Bench is
  the safer bet for guaranteeing variance.
- **Scope creep:** three papers' worth of ideas. Gate hard; ship Q1+static-trim first.
- **Oracle search cost:** labeling needs many VLM calls per example; budget compute or
  subsample.

## What we keep vs drop
- **Keep:** `visual_token_counter.py`, CI stats in `metrics.py`, `trim_downscale` /
  `downscale_baseline`, `run_method.py` / `compare_methods_ci.py` / `sweep_keep_ratio.py`.
- **Drop from headline:** disk-cache reuse index, 7B CATP-crop proxy, RAG retrieval as
  the project's identity (keep MMDocRAG only as the redundant-vision contrast point).
