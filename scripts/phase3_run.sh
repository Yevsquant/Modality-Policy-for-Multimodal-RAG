#!/usr/bin/env bash
set -u
cd /home/runying2/Query-Aware-Visual-Token-Pruning-and-Disk-Cache-in-Multimodal-RAG
PY=/home/runying2/.conda/envs/mrag/bin/python
export PYTHONPATH=.
LOG=logs/phase3.log
echo "[$(date +%T)] FastV matrix on V*Bench..." >> "$LOG"
$PY scripts/run_fastv_matrix.py --out data/vqa_stress/fastv_vstar.jsonl >> "$LOG" 2>&1
echo "[$(date +%T)] analyzing..." >> "$LOG"
$PY scripts/analyze_fastv.py --in data/vqa_stress/fastv_vstar.jsonl \
  --out data/vqa_stress/fastv_vstar_report.json >> "$LOG" 2>&1
echo "[$(date +%T)] PHASE3 DONE" >> "$LOG"
