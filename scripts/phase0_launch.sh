#!/usr/bin/env bash
# Phase-0 launcher: wait for the vLLM server, run unit tests, then run the
# V*Bench downscale stress test (resumable) and analyze the gate. Detached so the
# ~60-min harness reaper can't kill it.
set -u
cd /home/runying2/Query-Aware-Visual-Token-Pruning-and-Disk-Cache-in-Multimodal-RAG
PY=/home/runying2/.conda/envs/mrag/bin/python
export PYTHONPATH=.
LOG=logs/phase0.log
mkdir -p logs data/vqa_stress

echo "[$(date +%T)] waiting for server..." >> "$LOG"
for i in $(seq 1 3600); do
  if curl -s http://127.0.0.1:8000/v1/models 2>/dev/null | grep -q Qwen; then
    echo "[$(date +%T)] server up after ${i}s" >> "$LOG"; break
  fi
  sleep 10
done

echo "[$(date +%T)] running unit tests..." >> "$LOG"
$PY -m pytest tests/test_image_ops.py tests/test_vqa_scoring.py -q >> "$LOG" 2>&1
if [ $? -ne 0 ]; then
  echo "[$(date +%T)] UNIT TESTS FAILED — aborting stress run" >> "$LOG"; exit 1
fi
echo "[$(date +%T)] tests green" >> "$LOG"

echo "[$(date +%T)] V*Bench downscale stress test..." >> "$LOG"
$PY scripts/downscale_stress_test.py --dataset vstar --transform downscale \
  --keep-ratios 1.0,0.5,0.3,0.2,0.1 \
  --out data/vqa_stress/vstar_downscale.jsonl >> "$LOG" 2>&1

echo "[$(date +%T)] analyzing gate..." >> "$LOG"
$PY scripts/analyze_stress_test.py \
  --in data/vqa_stress/vstar_downscale.jsonl \
  --out data/vqa_stress/vstar_gate.json >> "$LOG" 2>&1
echo "[$(date +%T)] PHASE0 DONE" >> "$LOG"
