# Week 4 · SGLang 对照（10/01 - 10/08）

> 学习地图：阶段三（SGLang 对照学）。**10/08 = 一个月截止**。
> 方法：**不从头学概念**，用 vLLM 已掌握的框架逐项找 SGLang 对应物，填对照矩阵。
> 材料：`docs/03-SGLang对照导览.md`（安装 + 阅读地图 + 对照矩阵）、`docs/04-V4概念映射.md`（收尾）、`docs/05-核心自评清单.md`（自评）、`docs/06-issue候选池.md`（十月 issue 候选）。

## 本周目标

1. SGLang v0.5.19 安装（主力实验机，注意 torch 冲突预案）→ 0.8B 跑通（SGLang 侧）。
2. 按 vLLM 已掌握概念**逐项对照**：Radix 前缀缓存 / 调度策略 / tokenizer-detokenizer manager / 注意力后端（flashinfer 默认 + hybrid + deepseek_v4）/ 流式路径。
3. **填完对照矩阵**（docs/03 第一节）。
4. 收尾：V4 概念映射（docs/04）+ 核心自评清单全勾（docs/05）+ 十月 issue 候选池（docs/06）。
5. （stretch goal）27B SGLang 对照 benchmark（视机器可用情况）。

## 每日拆解（建议）

| 天 | 日期 | 任务 | 验收 |
|---|---|---|---|
| D23 | 10-01 | 装 SGLang 0.5.19（torch 冲突预案）→ 0.8B 起服务 + curl | 服务器 fired up；有回复 |
| D24 | 10-02 | 走读入口：launch_server → rust_server/server → entrypoints | 链路图 v2（SGLang 版） |
| D25 | 10-03 | 对照：scheduler + schedule_policy + cache_controller vs vLLM Scheduler | 矩阵行填充 |
| D26 | 10-04 | 对照：radix_cache vs vLLM prefix caching | 矩阵行填充 |
| D27 | 10-05 | 对照：注意力后端（flashinfer / hybrid / deepseek_v4_backend） | 矩阵行填充 |
| D28 | 10-06 | 对照：tokenizer/detokenizer manager + 流式路径 | 矩阵行填充 |
| D29 | 10-07 | 填 V4 映射（docs/04）+ 复核自评清单 | 映射表填完 |
| D30 | 10-08 | **收尾自检：对照矩阵 + 自评全勾 + issue 候选池** | 一个月目标达成 |

## 关键命令

```bash
# 10/01：安装（主力实验机）
pip install "sglang[all]==0.5.19"
# torch 冲突预案：pip install --dry-run 看依赖树；必要时 --no-deps 再补缺
python -c "import sglang; print(sglang.__version__)"   # 期望 0.5.19

# 起 0.8B 服务
MODEL_PATH=<model_dir>/Qwen3.5-0.8B ./scripts/start_sglang_qwen35_08b.sh
# 等 "The server is fired up and ready to roll!"
./scripts/test_chat.sh 30000 Qwen/Qwen3.5-0.8B "你好，请用一句话介绍你自己"
```

## 对照矩阵（docs/03 第 1 节，10/08 前填完）

| 维度 | vLLM v0.28.1rc0（已掌握） | SGLang v0.5.19+ | 差异本质 |
|---|---|---|---|
| 引擎进程模型 | EngineCore 独立进程 + core_client | tokenizer/detokenizer 独立 manager | |
| 调度器 | `v1/core/sched/scheduler.py` | `managers/scheduler.py` + `schedule_policy.py` | |
| 前缀缓存 | ？（W3 答案） | Radix tree（`mem_cache/radix_cache.py`） | |
| KV 分配 | block_pool + kv_cache_manager | memory pool + radix cache | |
| 注意力后端选择 | `v1/attention/selector.py` | `attention_registry.py` | |
| Qwen3.5 GDN 后端 | `backends/gdn_attn.py` | ？（hybrid/linear 后端） | |
| 流式输出路径 | async_llm → output_processor | tokenizer_manager → detokenizer_manager | |

## 检查点（10/08）

- [ ] SGLang 0.5.19 装好，0.8B 服务 + curl 跑通
- [ ] 对照矩阵填完（docs/03）
- [ ] V4 概念映射填完（docs/04）
- [ ] **核心自评清单全勾**（docs/05）
- [ ] 十月 issue 候选池 ≥3 条（docs/06）
- [ ] （stretch）27B SGLang 对照 benchmark

## 输出产物 → 放这里

- `笔记-SGLang对照.md`（或直接填 docs/03 对照矩阵）
- `SGLang链路图.md`（对照 vLLM 版）
- 自评结果（docs/05 全勾）
- issue 候选池清单（docs/06）

## 关联材料

| 材料 | 位置 |
|---|---|
| SGLang 对照导览（安装 + 阅读地图 + 矩阵） | `docs/03-SGLang对照导览.md` |
| V4 概念映射（收尾） | `docs/04-V4概念映射.md` |
| 核心自评清单（最后勾选） | `docs/05-核心自评清单.md` |
| issue 候选池 | `docs/06-issue候选池.md` |
| SGLang 0.8B 脚本 | `scripts/start_sglang_qwen35_08b.sh` |
| 进度 | `进度.md` |