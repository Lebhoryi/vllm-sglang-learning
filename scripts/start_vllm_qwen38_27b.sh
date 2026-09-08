#!/usr/bin/bash
# vLLM v0.28.0 起 Qwen3.8-27B HTTP 服务（默认端口 8002，限参配置内置）
# 用法: MODEL_PATH=/path PORT=8002 ./start_vllm_qwen38_27b.sh
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SRC_DIR/set_env.sh" ] && source "$SRC_DIR/set_env.sh"

MODEL=${MODEL_27B:-${MODEL_PATH:?请通过 MODEL_27B 或 MODEL_PATH 指定模型目录}}
PORT=${PORT:-8002}

if [ ! -d "$MODEL" ]; then
  echo "模型目录不存在: $MODEL （用 MODEL_PATH=... 指定）" >&2
  exit 1
fi

mkdir -p logs
LOG="logs/vllm-qwen38-27b-$(date +%m%d-%H%M).log"
echo ">> vLLM 起 27B: $MODEL :$PORT  -> $LOG"

vllm serve "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-model-len 16384 \
  --max-num-seqs 16 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  2>&1 | tee -a "$LOG"