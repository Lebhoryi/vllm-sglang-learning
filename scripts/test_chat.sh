#!/usr/bin/bash
# 对话冒烟测试（vLLM / SGLang 均兼容 OpenAI 格式）
# 用法: ./test_chat.sh [端口] [模型名] ["提问"]
# 注意: model 字段必须等于服务启动时的 model 参数（默认就是本地路径）；
#       若服务用 --served-model-name 指定过别名，则传别名
set -euo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SRC_DIR/set_env.sh" ] && source "$SRC_DIR/set_env.sh"

PORT=${1:-8001}
MODEL=${2:-${MODEL_08B:?请传模型名（与启动服务时 model 参数一致）}}
PROMPT=${3:-"你好，请用一句话介绍你自己"}

echo ">> POST http://127.0.0.1:${PORT}/v1/chat/completions  model=${MODEL}"
curl -s "http://127.0.0.1:${PORT}/v1/chat/completions" \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"${PROMPT}\"}],\"max_tokens\":128}" \
  | python3 -m json.tool --no-ensure-ascii