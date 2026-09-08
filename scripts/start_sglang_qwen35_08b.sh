#!/usr/bin/env bash
# SGLang v0.5.19 起 Qwen3.5-0.8B（默认端口 30000）——阶段三（10/01 起）才用
# 用法: MODEL_PATH=/path/to/model PORT=30000 ./start_sglang_qwen35_08b.sh
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SRC_DIR/set_env.sh" ] && source "$SRC_DIR/set_env.sh"

MODEL=${MODEL_08B:-${MODEL_PATH:?请通过 MODEL_08B 或 MODEL_PATH 指定模型目录}}
PORT=${PORT:-30000}

if [ ! -d "$MODEL" ]; then
  echo "模型目录不存在: $MODEL （用 MODEL_PATH=... 指定，或先跑 download_models.sh）" >&2
  exit 1
fi

mkdir -p logs
LOG="logs/sglang-qwen35-08b-$(date +%m%d-%H%M).log"
echo ">> SGLang 起服务: $MODEL :$PORT  -> $LOG"

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --max-running-requests 32 \
  --mem-fraction-static 0.85 \
  2>&1 | tee -a "$LOG"