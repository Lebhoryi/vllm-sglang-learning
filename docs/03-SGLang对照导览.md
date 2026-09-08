# 03 SGLang 对照导览（阶段三 W4，2026-09-08 版）

> **使用时间：10/01-10/08（阶段三）**。前置：阶段一/二完成——vLLM 的核心概念（生命周期/调度/KV/注意力后端）已成熟。
> 方法：**不从头学概念**，用 vLLM 已掌握的框架逐项找 SGLang 对应物，填对照矩阵。
> 路径基于本地源码实测：`share/2026Q4/sglang`（main 7d2d6624b1）。注意与经典教程不同：无 `srt/server.py`，核心在 `srt/rust_server/`。

## 安装与跑通（10/01 做）

```bash
# 主力实验机
pip install "sglang[all]==0.5.19"
# torch 冲突预案：pip install --dry-run 看依赖树；必要时 --no-deps 再补缺
python -c "import sglang; print(sglang.__version__)"   # 期望 0.5.19

MODEL_PATH=<model_dir>/Qwen3.5-0.8B ./scripts/start_sglang_qwen35_08b.sh
# 等 "The server is fired up and ready to roll!"
./scripts/test_chat.sh 30000 Qwen/Qwen3.5-0.8B "你好，请用一句话介绍你自己"
```

## SGLang 阅读地图（对照用）

```
HTTP 请求
  └─ python/sglang/launch_server.py               # 入口；参数见 srt/server_args.py
      └─ srt/rust_server/server.py                # ★ 服务器核心（新位置）
          └─ srt/entrypoints/http_server.py + openai/   # API 适配
              ├─ srt/managers/tokenizer_manager.py   # 请求接收/分词/组批
              ├─ srt/managers/scheduler.py           # ★ 调度器
              │     ├─ schedule_policy.py            # 调度策略（fcfs/权重等）
              │     ├─ cache_controller.py           # 前缀缓存控制
              │     └─ scheduler_components/         # 细分组件
              ├─ srt/mem_cache/radix_cache.py        # ★ Radix 前缀树缓存（SGLang 招牌）
              ├─ srt/model_executor/model_runner.py  # batch 前向
              │     └─ srt/layers/attention/         # 注意力后端（动态 import）
              │         ├─ flashinfer_backend.py     # 默认主力
              │         ├─ flashattention_backend.py
              │         ├─ hybrid_attn_backend.py    # 混合注意力
              │         ├─ deepseek_v4_backend.py    # V4 专用
              │         └─ radix_attention.py        # layers/ 下
              └─ srt/managers/detokenizer_manager.py # 解码回流传给 HTTP
```

模型定义：`srt/models/qwen3_5.py`（另有 `qwen3_5_mtp.py` MTP 头）。

## 对照矩阵（10/08 前填完）

左列 = 你在 vLLM 里已经掌握的答案；右列 = SGLang 对应物。

| 维度 | vLLM v0.28.1rc0（已掌握） | SGLang v0.5.19+ | 差异本质 |
|---|---|---|---|
| 引擎进程模型 | EngineCore 独立进程 + core_client | tokenizer/detokenizer 独立 manager | |
| 调度器 | `v1/core/sched/scheduler.py` | `managers/scheduler.py` + `schedule_policy.py` | |
| 前缀缓存 | ？（读 kv_cache_manager 找答案） | Radix tree（`mem_cache/radix_cache.py`） | |
| KV 分配 | block_pool + kv_cache_manager | memory pool + radix cache | |
| 注意力后端选择 | `v1/attention/selector.py` | `attention_registry.py` | |
| Qwen3.5 GDN 后端 | `backends/gdn_attn.py` | ？（找 hybrid/linear 后端） | |
| 流式输出路径 | async_llm → output_processor | tokenizer_manager → detokenizer_manager | |

## 通关问题（B 组）

1. tokenizer_manager 与 scheduler 之间怎么通信（队列/IO 对象）？一次请求的完整流转？
2. Radix tree 为什么能复用前缀？哪些请求能命中？（`radix_cache.py` 的 `CacheNode` 结构）
3. SGLang 的"连续批处理"中，新请求如何插入正在执行的 batch？
4. 前缀不命中/显存不足时，SGLang 会怎么做（淘汰请求？驱逐缓存节点）？
5. attention backend 的选择逻辑在哪？（`attention_registry.py`）
6. SGLang 哪些设计是 vLLM 没有的？哪些是相反的？（写成 3 条「原来如此」）

## Stretch goal（可选）

27B SGLang 对照 benchmark：若 主力实验机卡续期直接用；否则用备用实验机（A100-40G）+ 27B AWQ（≈16GB）跑 `benchmark_serving` 对照。**不是检查点**，时间不够就跳过，用 0.8B 数字讲清楚方法论即可。