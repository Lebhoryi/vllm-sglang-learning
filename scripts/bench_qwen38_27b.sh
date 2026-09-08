#!/usr/bin/env bash
# vLLM v0.28 benchmark：vllm bench serve（旧 python -m vllm.benchmarks.benchmark_serving 已废弃）
# 前置：27B 服务已起（start_vllm_qwen38_27b.sh，默认 8002，用 PORT 对齐）
# 用法: PORT=8002 NUMPROMPTS=200 ./bench_qwen38_27b.sh
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SRC_DIR/set_env.sh" ] && source "$SRC_DIR/set_env.sh"

PORT=${PORT:-8002}
MODEL=${MODEL_27B:-${MODEL:?请通过 MODEL_27B 或 MODEL 环境变量指定 27B 模型目录}}
NUMPROMPTS=${NUMPROMPTS:-200}
INLEN=${INLEN:-512}
OUTLEN=${OUTLEN:-128}

mkdir -p logs
LOG="logs/bench-27b-vllm-$(date +%m%d-%H%M).log"
echo ">> bench serve: $MODEL @:$PORT prompts=$NUMPROMPTS in=$INLEN out=$OUTLEN  -> $LOG"

vllm bench serve \
  --model "$MODEL" \
  --port "$PORT" \
  --dataset-name random \
  --num-prompts "$NUMPROMPTS" \
  --random-input-len "$INLEN" \
  --random-output-len "$OUTLEN" \
  --max-concurrency 16 \
  --request-rate inf \
  --save-result \
  --result-dir logs \
  2>&1 | tee "$LOG"

echo ">> 完成：日志 $LOG；--save-result 的 json 在 logs/（相同时刻生成）"