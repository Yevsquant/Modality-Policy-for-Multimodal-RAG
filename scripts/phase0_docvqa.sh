#!/usr/bin/env bash
set -u
cd /home/runying2/Query-Aware-Visual-Token-Pruning-and-Disk-Cache-in-Multimodal-RAG
PY=/home/runying2/.conda/envs/mrag/bin/python
export PYTHONPATH=.
LOG=logs/phase0_docvqa.log
echo "[$(date +%T)] DocVQA downscale stress (n=300)..." >> "$LOG"
$PY scripts/downscale_stress_test.py --dataset docvqa --transform downscale \
  --keep-ratios 1.0,0.5,0.3,0.2,0.1 --limit 300 \
  --out data/vqa_stress/docvqa_downscale.jsonl >> "$LOG" 2>&1
echo "[$(date +%T)] analyzing DocVQA gate..." >> "$LOG"
$PY scripts/analyze_stress_test.py \
  --in data/vqa_stress/docvqa_downscale.jsonl \
  --out data/vqa_stress/docvqa_gate.json >> "$LOG" 2>&1
echo "[$(date +%T)] DOCVQA DONE" >> "$LOG"
