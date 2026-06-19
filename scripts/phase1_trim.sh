#!/usr/bin/env bash
# Phase 1: after the DocVQA gate finishes, run trim_downscale on both datasets at
# the low-budget regime where plain downscale significantly hurt, and compare.
set -u
cd /home/runying2/Query-Aware-Visual-Token-Pruning-and-Disk-Cache-in-Multimodal-RAG
PY=/home/runying2/.conda/envs/mrag/bin/python
export PYTHONPATH=.
LOG=logs/phase1.log
echo "[$(date +%T)] waiting for DocVQA gate to finish..." >> "$LOG"
for i in $(seq 1 720); do
  grep -q "DOCVQA DONE" logs/phase0_docvqa.log 2>/dev/null && break
  sleep 10
done
echo "[$(date +%T)] V*Bench trim_downscale..." >> "$LOG"
$PY scripts/downscale_stress_test.py --dataset vstar --transform trim_downscale \
  --keep-ratios 0.3,0.2,0.1 --out data/vqa_stress/vstar_trim.jsonl >> "$LOG" 2>&1
echo "[$(date +%T)] DocVQA trim_downscale..." >> "$LOG"
$PY scripts/downscale_stress_test.py --dataset docvqa --transform trim_downscale \
  --keep-ratios 0.3,0.2,0.1 --limit 300 --out data/vqa_stress/docvqa_trim.jsonl >> "$LOG" 2>&1
echo "[$(date +%T)] comparisons..." >> "$LOG"
$PY scripts/compare_transforms.py --downscale data/vqa_stress/vstar_downscale.jsonl \
  --trim data/vqa_stress/vstar_trim.jsonl \
  --out data/vqa_stress/vstar_trim_vs_downscale.json >> "$LOG" 2>&1
$PY scripts/compare_transforms.py --downscale data/vqa_stress/docvqa_downscale.jsonl \
  --trim data/vqa_stress/docvqa_trim.jsonl \
  --out data/vqa_stress/docvqa_trim_vs_downscale.json >> "$LOG" 2>&1
echo "[$(date +%T)] PHASE1 DONE" >> "$LOG"
