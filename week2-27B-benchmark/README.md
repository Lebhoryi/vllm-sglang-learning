# Week 2 · 27B benchmark 基线（09/15 - 09/21）

> 学习地图：阶段二前半（vLLM 实战：27B 起服务 + benchmark 基线）。纪律：仍然 **vLLM 专属**。
> 详细执行清单见 `docs/00-阶段一执行清单.md`（W2 部分 = D7-D13）。

## 本周目标

1. Qwen3.8-27B 下载（hf-mirror，54GB，后台拉；已在 `<model_dir>/Qwen3.8-27B` 则跳过）。
2. vLLM 起 27B 服务（**必须限参**：`--max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.9`），curl 冒烟。
3. **`vllm bench serve` 跑出基线**（v0.28 新 CLI；旧 `python -m vllm.benchmarks.benchmark_serving` 已废弃为报错 shim）。
4. 指标解读（TTFT / TPOT / 输出吞吐 / 请求吞吐 / 错误率），基线数字存档 + 笔记。

## 每日拆解（docs/00 D7-D13）

| 天 | 日期 | 任务 | 验收 |
|---|---|---|---|
| D7 | 09-15 | 下载 Qwen3.8-27B（54GB 后台拉；若已在 `<model_dir>` 跳过） | 目录完整 |
| D8 | 09-16 | vLLM 起 27B（限参 flags）→ 对话冒烟 | 回复正常 |
| D9 | 09-17 | `vllm bench serve` 小规模试跑 | 单个小规模跑通 |
| D10 | 09-18 | 正式 benchmark → 指标存档 | 基线数字落 `logs/` |
| D11 | 09-19 | 指标解读（TTFT/TPOT/吞吐）+ 记笔记 | 解读段落 |
| D12 | 09-20 | （机动）复跑/调参/补文档 | — |
| D13 | 09-21 | **检查点②自检** | 全勾 |

## 关键命令

```bash
# D7：下载（hf-mirror，54GB，建议后台；若已存在跳过）
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download Qwen/Qwen3.8-27B --local-dir <model_dir>/Qwen3.8-27B

# D8：起服务（27B 必须限参，A100-80G 单卡）
MODEL_PATH=<model_dir>/Qwen3.8-27B PORT=8002 \
  vllm serve <model_dir>/Qwen3.8-27B --host 0.0.0.0 --port 8002 \
  --max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.9 \
  --dtype bfloat16 2>&1 | tee -a logs/vllm-qwen38-27b.log

# D9-D10：benchmark（v0.28 新 CLI！）
vllm bench serve \
  --model <model_dir>/Qwen3.8-27B \
  --port 8002 \
  --dataset-name random \
  --num-prompts 200 \
  --random-input-len 512 \
  --random-output-len 128 \
  --max-concurrency 16 \
  --request-rate inf \
  --save-result --result-dir logs 2>&1 | tee logs/bench-27b-vllm-$(date +%m%d).log
```

参数速查：`--dataset-name` 默认 `random`（免外部文件）；random 默认 `--random-input-len 1024`、`--random-output-len 128`；`--num-prompts` 默认 1000；`--request-rate inf` = 打满；更多 `vllm bench serve --help`。

## 检查点（09/21）

- [ ] 27B vLLM 起服务成功（限参 flags 生效）
- [ ] `vllm bench serve` 基线 log + json 存档（`logs/`）
- [ ] TTFT / TPOT / 吞吐解读笔记

**本周要拿到的数字**：TTFT（平均/p50/p99）、TPOT、输出吞吐（tokens/s）、请求吞吐（req/s）、错误率。`--save-result` 的 json 一并保留。

## 输出产物 → 放这里

- `logs/vllm-qwen38-27b.log`（服务日志）
- `logs/bench-27b-vllm-*.log` + `logs/*.json`（`--save-result` 结果）
- `笔记-27B基准解读.md`（数字 + 解读：262K 上下文下 27B 并发仅 1.14x 的 KV 容量账）
- 检查点自检勾选结果

## 关联材料

| 材料 | 位置 |
|---|---|
| 执行清单（Step 4） | `docs/00-阶段一执行清单.md` |
| 基准解读卡 | `docs/07-基准解读卡.md` |
| 27B 一键脚本 | `scripts/start_vllm_qwen38_27b.sh`、`scripts/bench_qwen38_27b.sh` |
| 27B 离线 API 实测日志（已跑通） | `test_vllms/test_vllm_qwen3.8_27b.py` → `test_qwen3.8_27b.log` |
| 进度 | `进度.md` |