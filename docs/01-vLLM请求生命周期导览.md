# 01 vLLM 请求生命周期走读导览（阶段一 W1，2026-09-08 版）

> 路径全部基于本地源码实测：`share/2026Q4/vllm`（main 869f78732b ≈ v0.28.1rc0-509）。
> **重要**：v0.28 目录结构与经典教程不同——没有 `vllm/core/`、`vllm/attention/`，引擎主体在 `vllm/v1/`。以下路径均为实测存在。
> SGLang 的对照在阶段三（docs/03），本阶段只读 vLLM。

## vLLM 链路

```
HTTP 请求
  └─ vllm/entrypoints/                      # OpenAI API 层（openai/api_server 等）
      └─ AsyncLLMEngine
          = vllm/v1/engine/async_llm.py     # 对外引擎门面（vllm/engine/ 下的是兼容封装）
              ├─ vllm/v1/engine/core_client.py   # 与 EngineCore 进程通信
              ├─ vllm/v1/engine/core.py          # EngineCore：引擎主循环（step）
              ├─ vllm/v1/engine/input_processor.py   # 请求 → 输入 tensors
              ├─ vllm/v1/core/sched/scheduler.py    # ★ 调度器（请求入批/抢占）
              │     ├─ async_scheduler.py / request_queue.py / interface.py
              ├─ vllm/v1/core/kv_cache_manager.py    # ★ KV cache 分配
              │     └─ block_pool.py / kv_cache_layout.py
              ├─ vllm/v1/worker/gpu_worker.py        # 执行器
              │     └─ gpu_model_runner.py           # batch 前向
              │           └─ vllm/v1/attention/selector.py   # ★ 注意力后端选择
              │                 └─ vllm/v1/attention/backends/
              │                     ├─ flash_attn.py       # 标准注意力
              │                     ├─ gdn_attn.py         # Gated DeltaNet（Qwen3.5 线性注意力）
              │                     ├─ linear_attn.py / mamba1_attn.py / mamba2_attn.py
              │                     └─ mla/               # MLA（V4 方向）
              ├─ vllm/v1/engine/output_processor.py     # 采样结果处理
              └─ vllm/v1/engine/detokenizer.py          # token → 文本回流
```

模型定义与 kernel：

```
vllm/model_executor/models/qwen3_5.py                     # Qwen3.5 模型（混合注意力装配）
vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py  # GDN 线性注意力 kernel 封装
```

## 通关问题（能自答即算过，答案写进笔记）

1. 一次 `POST /v1/chat/completions` 从 entrypoint 到 EngineCore 的调用链是什么？每跳传的是什么对象？
2. Scheduler 在什么时机把请求加入/移出 batch？它的输入输出结构（`vllm/v1/core/sched/interface.py`、`output.py`）？
3. 显存不够时 preemption 怎么触发？swap 还是 recompute？（`kv_cache_manager.py` 里找）
4. KV cache 的 block 怎么分配/释放？`kv_cache_layout.py` 在算什么？
5. attention backend 是怎么被选中的？（`selector.py` + `backends/registry.py`）
6. 生成完成后 token 如何回到 detokenizer 并流式吐出？

## 输出要求

1. 生命周期图 ×1（vLLM 全链路，手画/PPT/drawio 均可，放 docs/）
2. 6 个通关问题的自答
3. 3 条「原来如此」+ 3 个「存疑待查」写在下方附录

## 附录（我的笔记）

- 原来如此 1：
- 原来如此 2：
- 原来如此 3：
- 存疑 1：
- 存疑 2：