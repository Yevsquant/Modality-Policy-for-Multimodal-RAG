#!/bin/bash
set -euo pipefail
export LMCACHE_CONFIG_FILE="config.yaml"

vllm serve Qwen/Qwen3-Omni-30B-A3B-Instruct \
  --max-model-len 16384 \
  --gpu_memory_utilization 0.5 \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}'