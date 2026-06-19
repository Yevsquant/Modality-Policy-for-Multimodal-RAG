# Visual Token Reduction for Vision-Language Models — Input Preprocessing vs In-Model Pruning

> The repository name reflects the project's **origin** — a query-aware visual-token-pruning
> and disk-cache system for multimodal RAG. It became a **rigorous empirical study** of a
> single question: **how do you cut the visual tokens a VLM consumes — and how do you
> measure that honestly?** The headline finding is methodological: how you *measure*
> visual-token reduction decides which method "wins," and the field's usual axis
> (deep-layer token count) systematically flatters in-model pruning over input-side
> preprocessing.

**Tested hardware:** NVIDIA H200 (your environment may differ; see
[requirements.txt](requirements.txt) for CUDA / vLLM-oriented pins).

Full write-up: **[thoughts/work_report/2026-06-19-project-summary-visual-token-reduction.md](thoughts/work_report/2026-06-19-project-summary-visual-token-reduction.md)**
(per-phase reports in [thoughts/work_report/](thoughts/work_report/); design decisions in
[thoughts/plans/](thoughts/plans/)).

## Headline findings

Measurement throughout: visual tokens counted on the **target generation model's**
processor; accuracy with bootstrap 95% CIs and paired per-example significance tests over
n = 191–300.

- **The original RAG benchmark can't prove a visual-token method.** On MMDocRAG the
  accuracy/token frontier is *flat* — downscaling to ~150 tokens (a 90% cut) loses nothing,
  because the answer lives in the retrieved *text*. So the study moved to a benchmark where
  image detail is provably load-bearing: **V\*Bench** (high-resolution visual search).
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

## How the project evolved

1. **Origin — query-aware pruning + disk cache for multimodal RAG.** After hybrid
   text-and-image retrieval, retrieved document images were pruned in a query-conditioned
   way, written to disk, and reused when a later query was similar to a cached entry for the
   same image (to cut TTFT). This pipeline still lives in the repo (see
   [The original RAG pipeline](#the-original-rag-pipeline-origin)).
2. **Honest measurement exposed the problem.** Counting tokens on the *answer* model and
   adding CIs/significance showed MMDocRAG's frontier is flat — the image is nearly
   redundant, so no visual-token method can demonstrate value there.
3. **Repositioning.** The work refocused on **input preprocessing vs in-model pruning** on
   detail-sensitive VQA, with a go/no-go "stress-test gate," and on the right way to
   **measure** the trade-off (the fair-compute reframing above).

## Repository map

**The visual-token-reduction study** (self-contained single-image VQA; analysis needs no
RAG server):

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

**The original RAG pipeline:** `rag/{retriever,query_pipeline,pruner,prompt_builder,eval}.py`,
`scripts/run_mmdocrag_benchmark.py`. All tunable behavior is centralized in
`rag/config.py:RAGConfig`.

## Setup

```bash
conda env create -f environment.yml   # creates env "mrag"
conda activate mrag
```

Run everything from the repo root with `PYTHONPATH=.`. Run tests with
`PYTHONPATH=. python -m pytest tests/ -q`.

## Reproducing the study

### 1. Stress-test gate (needs the VLM server)

Does downscaling provably hurt on this dataset? Start the target model
([see server command](#serving-the-vlm)), then:

```bash
PYTHONPATH=. python scripts/downscale_stress_test.py \
  --dataset vstar --transform downscale --keep-ratios 1.0,0.5,0.3,0.2,0.1 \
  --out data/vqa_stress/vstar_downscale.jsonl
PYTHONPATH=. python scripts/analyze_stress_test.py \
  --in data/vqa_stress/vstar_downscale.jsonl --out data/vqa_stress/vstar_gate.json
```

Swap `--dataset docvqa|hrbench` to screen other benchmarks.

### 2. FastV vs input-downscale (Qwen2-VL-7B, loads itself — no server)

```bash
PYTHONPATH=. python scripts/run_fastv_matrix.py --out data/vqa_stress/fastv_vstar.jsonl
PYTHONPATH=. python scripts/analyze_fastv.py \
  --in data/vqa_stress/fastv_vstar.jsonl --out data/vqa_stress/fastv_vstar_report.json
```

### 3. Fair-cost reframing (the headline; no server)

```bash
PYTHONPATH=. python scripts/angle1_flops_reframing.py     # analytical FLOPs + Pareto plot
PYTHONPATH=. python scripts/angle1_measure_latency.py     # measured wall-clock on the 7B
```

## The original RAG pipeline (origin)

### Serving the VLM

The client uses OpenAI-compatible URLs from `RAGConfig` (`vlm_api_base`, default
`http://127.0.0.1:8000/v1`; offline judge uses `judge_api_base`, same default). Start the
server from the repo root in **one terminal**:

```bash
PYTHONPATH=$PYTHONPATH:. \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --max-model-len 16384 \
  --gpu_memory_utilization=0.5 \
  --host 0.0.0.0 \
  --port 8000
```

**Optional: vLLM with LMCache.** Point `LMCACHE_CONFIG_FILE` at your
[config.yaml](config.yaml) and align its storage location with `RAGConfig.lmcache_path` in
[rag/config.py](rag/config.py).

```bash
PYTHONPATH=$PYTHONPATH:. \
LMCACHE_CONFIG_FILE="config.yaml" \
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --max-model-len 16384 --gpu_memory_utilization=0.5 --host 0.0.0.0 --port 8000 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

### Running the MMDocRAG benchmark

In **another terminal** (server still running), from the repo root. (The script is
`run_mmdocrag_benchmark.py`; the older spelling `run_mmodcrag_benchmark.py` is a typo.)

```bash
# JSONL lines [0, 50), no extra cap on row count (--max-examples 0)
PYTHONPATH=$PYTHONPATH:. python scripts/run_mmdocrag_benchmark.py \
  --eval-slice-start 0 --eval-slice-stop 50 --max-examples 0

# From line index 100, at most 15 examples
PYTHONPATH=$PYTHONPATH:. python scripts/run_mmdocrag_benchmark.py \
  --eval-slice-start 100 --max-examples 15
```

It runs an **online** RAG pass ([rag/eval.py](rag/eval.py) `run_rag_benchmark`), then an
**offline** LLM-as-judge pass (`run_rag_benchmark_offline_judge`), and writes an aggregate
JSON including optional LMCache / host utilization.

### Outputs (default `data/mmdocrag/outputs/`)

| Artifact | Description |
| -------- | ----------- |
| `baseline_predictions.json` | Per-example predictions + lexical / retrieval metrics (online pass) |
| `baseline_results_judged.json` | Same rows with LLM-judge fields (+ optional `lmcache` block) |
| `image_prune_cache.json` | Disk cache index for reused pruned images (`image_prune_cache_path`) |
| `pruned_images/` | Pruned image files (`pruned_image_dir`) |
| `final_results_with_utilization.json` | Summary + rows + `lmcache` + system utilization |

## Figures

**Headline result — FastV (▲) vs input-downscale (●) on the fair compute axis.** FastV's
token-budget advantage shrinks to near-zero once the full-image vision-encoder cost it pays
is counted.

![FastV vs input-downscale on the fair compute axis](imgs/Angle1FlopsFrontier.png)

**Original RAG pipeline & pruning (origin):**

![Multimodal RAG pipeline overview](imgs/RAGPipeline.png)

![Pruning process overview](imgs/PruningProcess.png)

![Average latency by method](imgs/AvgLatencyByMethods.png)

![Correctness / score by method](imgs/CorrectScoreByMethods.png)
