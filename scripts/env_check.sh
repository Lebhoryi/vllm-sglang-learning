#!/usr/bin/bash
# 主力实验机环境核查：一键打印关键版本与显卡状态
set -uo pipefail
SRC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
[ -f "$SRC_DIR/set_env.sh" ] && source "$SRC_DIR/set_env.sh"

echo "== python =="
python -c "import sys; print(sys.version.split()[0])"

echo "== torch / cuda =="
python -c "import torch; print('torch', torch.__version__, '| cuda', torch.version.cuda, '| avx', torch.cuda.is_available())"

echo "== vllm =="
python -c "import vllm; print('vllm', vllm.__version__)" 2>/dev/null || echo "vllm 不可用"

echo "== sglang =="
python -c "import sglang; print('sglang', sglang.__version__)" 2>/dev/null || echo "sglang 未装"

echo "== flashinfer =="
python -c "import flashinfer; print('flashinfer', flashinfer.__version__)" 2>/dev/null || echo "flashinfer 不可用"

echo "== transformers =="
python -c "import transformers; print('transformers', transformers.__version__)" 2>/dev/null || echo "transformers 不可用"

echo "== GPU =="
nvidia-smi --query-gpu=index,name,memory.total,memory.used,utilization.gpu --format=csv 2>/dev/null || echo "nvidia-smi 不可用"