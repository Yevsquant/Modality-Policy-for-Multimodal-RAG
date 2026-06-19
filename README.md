# Visual Token Reduction for Vision-Language Models — Input Preprocessing vs In-Model Pruning

> *The repository name is legacy.* This project is a **rigorous empirical study** of one
> question: **how do you cut the visual tokens a VLM consumes — and how do you measure that
> honestly?** The headline finding is methodological: how you *measure* visual-token
> reduction decides which method "wins," and the field's usual axis (deep-layer token
> count) systematically flatters in-model pruning over input-side preprocessing.

**Tested hardware:** NVIDIA H200 (your environment may differ; see
[requirements.txt](requirements.txt) for CUDA / vLLM-oriented pins).

Full write-up: **[thoughts/work_report/2026-06-19-project-summary-visual-token-reduction.md](thoughts/work_report/2026-06-19-project-summary-visual-token-reduction.md)**
(per-phase reports in [thoughts/work_report/](thoughts/work_report/); design decisions in
[thoughts/plans/](thoughts/plans/)).

## Headline findings

Measurement throughout: visual tokens counted on the **target generation model's**
processor; accuracy with bootstrap 95% CIs and paired per-example significance tests over
n = 191–300.

- **Pick a benchmark where image detail is actually load-bearing.** On a document-VQA
  benchmark the accuracy/token frontier is *flat* — downscaling to ~150 tokens (a 90% cut)
  loses nothing, because the answer is recoverable from text. The study uses **V\*Bench**
  (high-resolution visual search), where downscaling provably hurts, screened by a
  go/no-go "stress-test gate."
- **In-model pruning (FastV) only *looks* like it beats input-side downscaling.** At a
  matched *deep-layer token* budget FastV wins by +0.099 accuracy; but on honest **total
  compute** the edge collapses, because FastV still encodes the full image first:

  | FastV advantage over input-downscale | r=0.25 | r=0.5 |
  |---|---|---|
  | deep-layer tokens (the usual axis) | +0.099 | +0.052 |
  | analytical total FLOPs (matched) | +0.015 | +0.026 |
  | **measured latency (matched)** | **+0.002** | **+0.019** |

  The vision tower is a constant **51–55% of FastV's latency** — a cost it cannot avoid.
- **Cheap input-side reduction is the efficient lever, but making it *adaptive* is hard.**
  A learned per-image resolution-budget policy and a query-conditioned foveated crop both
  failed to beat plain uniform downscaling — cheap CLIP features can't predict per-example
  downscale sensitivity (out-of-fold AUC ≈ 0.50), and single-shot localization misses the
  small objects visual search depends on. **That gap is the open problem.**

**Methodological takeaway:** always measure visual-token-reduction methods on a cost axis
that **includes the vision encoder**; "deep-layer tokens" flatters in-model pruning and
unfairly penalizes input-side reduction.

## Repository map

A self-contained single-image VQA study (analysis needs no server; the stress test sends
images to a served VLM).

| Module / script | Purpose |
| --- | --- |
| `rag/visual_token_counter.py` | count visual tokens on the **target** model's processor |
| `rag/metrics.py` | bootstrap + paired-difference CIs |
| `rag/vqa_datasets.py` | loaders: V\*Bench, DocVQA, HR-Bench |
| `rag/image_ops.py` | `downscale_to_keep`, `trim_downscale` input transforms |
| `scripts/downscale_stress_test.py` + `analyze_stress_test.py` | the **gate**: does downscaling provably hurt accuracy? |
| `rag/fastv.py` + `scripts/run_fastv_matrix.py` + `analyze_fastv.py` | faithful **FastV** in-model pruning on Qwen2-VL-7B |
| `rag/budget_features.py` + `rag/budget_policy.py` + `scripts/train_budget_policy.py` | learned per-image budget policy (negative result) |
| `rag/foveated_crop.py` + `scripts/run_foveated_crop.py` | query-conditioned crop preprocessor (negative result) |
| `rag/flops_model.py` + `scripts/angle1_flops_reframing.py` + `scripts/angle1_measure_latency.py` | **fair-cost reframing**: FLOPs + measured latency |

## Setup

```bash
conda env create -f environment.yml   # creates env "mrag"
conda activate mrag
```

Run everything from the repo root with `PYTHONPATH=.`. Run tests with
`PYTHONPATH=. python -m pytest tests/ -q`.

## Reproducing the study

### 1. Stress-test gate — does downscaling provably hurt? (needs a served VLM)

The stress test sends images to an OpenAI-compatible VLM endpoint. Launch the target model
in one terminal:

```bash
PYTHONPATH=. python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --max-model-len 16384 --gpu_memory_utilization=0.5 --host 0.0.0.0 --port 8000
```

Then, in another terminal:

```bash
PYTHONPATH=. python scripts/downscale_stress_test.py \
  --dataset vstar --transform downscale --keep-ratios 1.0,0.5,0.3,0.2,0.1 \
  --out data/vqa_stress/vstar_downscale.jsonl
PYTHONPATH=. python scripts/analyze_stress_test.py \
  --in data/vqa_stress/vstar_downscale.jsonl --out data/vqa_stress/vstar_gate.json
```

Swap `--dataset docvqa|hrbench` to screen other benchmarks. The endpoint defaults come from
`rag/config.py:RAGConfig` (`vlm_api_base`, default `http://127.0.0.1:8000/v1`).

### 2. FastV vs input-downscale (Qwen2-VL-7B, loads itself — no server)

```bash
PYTHONPATH=. python scripts/run_fastv_matrix.py --out data/vqa_stress/fastv_vstar.jsonl
PYTHONPATH=. python scripts/analyze_fastv.py \
  --in data/vqa_stress/fastv_vstar.jsonl --out data/vqa_stress/fastv_vstar_report.json
```

### 3. Fair-cost reframing — the headline (no server)

```bash
PYTHONPATH=. python scripts/angle1_flops_reframing.py     # analytical FLOPs + Pareto plot
PYTHONPATH=. python scripts/angle1_measure_latency.py     # measured wall-clock on the 7B
```

## Figure

**Headline result — FastV (▲) vs input-downscale (●) on the fair compute axis.** FastV's
token-budget advantage shrinks to near-zero once the full-image vision-encoder cost it pays
is counted.

![FastV vs input-downscale on the fair compute axis](imgs/Angle1FlopsFrontier.png)
