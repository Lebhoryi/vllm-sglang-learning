#!/usr/bin/bash
# 下载 Qwen3.5-0.8B（主用） + Qwen2.5-0.5B-Instruct（对照） + Qwen3.8-27B（约 54GB）
# 用法: ./download_models.sh [目标目录]   （默认 $MODEL_DIR，由 set_env.sh 提供）
# 注意: Qwen3.5-0.8B-Instruct 在 hf-mirror 是 404，不要下；base 自带 chat template
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SRC_DIR/set_env.sh" ] && source "$SRC_DIR/set_env.sh"

export HF_ENDPOINT=https://hf-mirror.com
DEST=${1:-${MODEL_DIR:?请指定模型目标目录或先 source set_env.sh}}
mkdir -p "$DEST"

echo ">> 下载 Qwen/Qwen3.5-0.8B -> $DEST/Qwen3.5-0.8B"
huggingface-cli download Qwen/Qwen3.5-0.8B --local-dir "$DEST/Qwen3.5-0.8B"

echo ">> 下载 Qwen/Qwen2.5-0.5B-Instruct -> $DEST/Qwen2.5-0.5B-Instruct"
huggingface-cli download Qwen/Qwen2.5-0.5B-Instruct --local-dir "$DEST/Qwen2.5-0.5B-Instruct"

echo ">> 下载 Qwen/Qwen3.8-27B -> $DEST/Qwen3.8-27B"
huggingface-cli download Qwen/Qwen3.8-27B --local-dir "$DEST/Qwen3.8-27B"

echo "== 完成，模型位于 =="
du -sh "$DEST/Qwen3.5-0.8B" "$DEST/Qwen2.5-0.5B-Instruct" "$DEST/Qwen3.8-27B"