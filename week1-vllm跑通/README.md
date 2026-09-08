# Week 1 · vLLM 跑通（09/08 - 09/14）

> 学习地图：阶段一（vLLM 专属，跑通）。纪律：**只碰 vLLM，不装/不跑/不读 SGLang**。
> 详细执行清单见 `docs/00-阶段一执行清单.md`（W1 部分 = D0-D6）；走读材料 `docs/01-vLLM请求生命周期导览.md`。

## 本周目标

1. vLLM v0.28.0 跑 `Qwen3.5-0.8B`（base）——离线 LLM API 已 ✅（09-08），**补 `vllm serve` HTTP 模式 + curl 冒烟**（benchmark 的前置）。
2. 请求生命周期走读（docs/01）：entrypoint → AsyncLLM → EngineCore → Scheduler → Worker → AttentionBackend → KV → OutputProcessor/Detokenizer 全链路，产出**一张图 + 笔记**。
3. 通关问题 1-6 自答（docs/01 第 2 节），写进笔记。

## 每日拆解（docs/00 D0-D6）

| 天 | 日期 | 任务 | 验收 |
|---|---|---|---|
| D0 | 09-08 | 环境核查 + 下载 0.8B | 【已基本完成】env_check 全绿；模型在 `<model_dir>/Qwen3.5-0.8B` |
| D1 | 09-08/09 | **补：`vllm serve` 起 0.8B HTTP 服务 → curl 对话** | 有聊天回复，日志入 `logs/` |
| D2 | 09-10 | 走读 lifecycle（docs/01 第 2 节 a-c） | 链路图 v1（部分） |
| D3 | 09-11 | 继续走读（selector/KV 部分）+ 笔记 | 链路图 v1 完整 |
| D4 | 09-12 | 通关问题 1-6 自答；3 条「原来如此」 | 笔记成型 |
| D5 | 09-13 | （机动）补走读 / Qwen2.5-0.5B 对照 / 27B 下载预热 | — |
| D6 | 09-14 | **检查点①自检**（docs/00 文末清单） | 全勾 |

## 关键命令

```bash
# 起 0.8B HTTP 服务（主力实验机）
MODEL_PATH=<model_dir>/Qwen3.5-0.8B ./scripts/start_vllm_qwen35_08b.sh
# 等 "Application startup complete" / "Uvicorn running"
./scripts/test_chat.sh 8001 Qwen/Qwen3.5-0.8B "你好，请用一句话介绍你自己"
```

> 顺带观察：serve 模式下日志里的 `Available KV cache memory` 与 `GPU KV cache size` 两个数字——docs/02 专题 2 要用它们反推单 token KV 字节数。

## 检查点（09/14）

- [ ] `vllm.__version__` == 0.28.0
- [ ] vLLM 0.8B HTTP 服务 + curl 对话记录（`logs/` 有日志；离线 LLM API 已 ✅）
- [ ] 生命周期图（vLLM 全链路）
- [ ] 6 个通关问题有自答（笔记）
- [ ] 没有偷跑 SGLang 💪

## 输出产物 → 放这里

- `logs/vllm-qwen35-08b-*.log`（服务日志）
- `笔记-生命周期走读.md`（链路图 + 6 问答 + 3 条原来如此）
- 检查点自检勾选结果（可回填 `docs/00-阶段一执行清单.md` 文末清单）

## 关联材料

| 材料 | 位置 |
|---|---|
| 执行清单 | `docs/00-阶段一执行清单.md` |
| 生命周期走读导览 | `docs/01-vLLM请求生命周期导览.md` |
| 通关问题参考答案（agent 侧预建） | `docs/01B-参考答案.md` |
| 脚本 | `scripts/start_vllm_qwen35_08b.sh`、`scripts/test_chat.sh`、`scripts/env_check.sh` |
| 进度 | `进度.md` |