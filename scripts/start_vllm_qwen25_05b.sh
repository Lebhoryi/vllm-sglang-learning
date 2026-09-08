#!/usr/bin/env bash
# vLLM v0.28.0 起 Qwen2.5-0.5B-Instruct（标准注意力对照模型，默认端口 8003）
# 用法: MODEL_PATH=/path PORT=8003 ./start_vllm_qwen25_05b.sh
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SRC_DIR/set_env.sh" ] && source "$SRC_DIR/set_env.sh"

MODEL=${MODEL_05B:-${MODEL_PATH:?请通过 MODEL_05B 或 MODEL_PATH 指定模型目录}}
PORT=${PORT:-8003}

if [ ! -d "$MODEL" ]; then
  echo "模型目录不存在: $MODEL （用 MODEL_PATH=... 指定，或先跑 download_models.sh）" >&2
  exit 1
fi

mkdir -p logs
LOG="logs/vllm-qwen25-05b-$(date +%m%d-%H%M).log"
echo ">> vLLM 起 0.5B 对照: $MODEL :$PORT  -> $LOG"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 8192 \
  --max-num-seqs 32 \
  --gpu-memory-utilization 0.85 \
  --dtype bfloat16 \
  2>&1 | tee -a "$LOG"