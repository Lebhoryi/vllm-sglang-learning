# Week 3 · 核心模块深读（09/22 - 09/30）

> 学习地图：阶段二后半（P3 核心模块：调度 / KV / 混合注意力深读）。**09/30 硬检查点**（主力实验机的 A100 当天到期，10 月起卡归属未知）。
> 材料：`docs/02-vLLM调度与KV深读.md`（三个专题）；组件文档 `docs/01B-参考答案.md`。

## 本周目标

把 docs/01 链路里带 ★ 的三块吃透，**最终能自己算出日志里的数字**（67.08 GiB / 5,788,118 tokens 怎么来）：

1. **专题 1 · 调度与批处理**：`scheduler.step()` 输入输出、preemption（swap vs recompute）、chunked prefill、prefix caching。
2. **专题 2 · KV cache 与显存**：单 token KV 字节数反推（0.8B/27B 两套日志数字）、block table / block_pool、`gpu_memory_utilization` / `max_num_seqs` / `max_model_len` 互咬、MLA 低秩 KV 原理（V4 铺垫）。
3. **专题 3 · 混合注意力（GDN）+ MLA**：`models/qwen3_5.py` 装配（GDN 线性注意力 + Gated Attention 排列）、`layers/mamba/gdn/` kernel 封装、`backends/gdn_attn.py`、attention page 与 mamba page 对齐（你日志里的 784 block size）、MLA 后端。

## 阅读地图（源码位置实测，vllm main 869f78732b）

```
调度：   vllm/v1/core/sched/{scheduler,async_scheduler,request_queue,interface,output}.py
配置侧： vllm/config/{scheduler,cache,vllm}.py
KV：     vllm/v1/core/{kv_cache_manager,block_pool,kv_cache_coordinator,single_type_kv_cache_manager}.py
         vllm/v1/{kv_cache_layout,kv_cache_interface,worker/block_table}.py
GDN：    vllm/model_executor/models/qwen3_5.py
         vllm/model_executor/layers/mamba/gdn/{base,qwen_gdn_linear_attn,...}.py
         vllm/v1/attention/backends/{gdn_attn,linear_attn,mamba1_attn,mamba2_attn}.py
         vllm/v1/attention/backends/mla/（V4 方向）
```

## 每日拆解（建议）

| 天 | 日期 | 任务 | 验收 |
|---|---|---|---|
| D14 | 09-22 | 专题 1：scheduler.step() 走读 | 通关问题 1-4 有答案 |
| D15 | 09-23 | 专题 1：preemption / chunked prefill / prefix caching | 笔记 + 图示 |
| D16 | 09-24 | 专题 2：KV layout + 反推 0.8B 数字 | **算数对上 67.08 GiB / 5,788,118** |
| D17 | 09-25 | 专题 2：27B 反推 + 三参数互咬 + MLA | **算数对上 18.61 GiB / 298,275** |
| D18 | 09-26 | 专题 3：qwen3_5.py 装配走读 | 层结构图（GDN+Gated 排列） |
| D19 | 09-27 | 专题 3：GDN kernel + attention/mamba page 对齐 | 讲清 784 block size 来源 |
| D20 | 09-28 | 专题 3：MLA + 双后端选择（selector） | 概念图 |
| D21 | 09-29 | （机动）补缺 / 复查 docs/05 前三模块 | — |
| D22 | 09-30 | **硬检查点自检**（docs/05 第 1-4 模块勾选） | 全勾 |

## 检查点（09/30，硬）

- [ ] 调度：能讲清一次 step 编排、preemption 主路径 = recompute（v0.28 无 CPU swap）
- [ ] KV：**能复现**两套日志数字（0.8B 与 27B）
- [ ] GDN：能画出 Qwen3.5 层结构图；讲清与标准 attention 的 kernel 差异
- [ ] 核心模块笔记/文档完成（这是 9/30 硬检查点的交付物）

## 输出产物 → 放这里

- `专题1-调度笔记.md` / `专题2-KV显存笔记.md` / `专题3-GDN笔记.md`
- `Qwen3.5结构图`（GDN + Gated Attention 排列）
- 硬检查点自检勾选结果（对照 `docs/05-核心自评清单.md` 第 1-4 模块）

## 关联材料

| 材料 | 位置 |
|---|---|
| 深读导览（三专题 + 通关问题） | `docs/02-vLLM调度与KV深读.md` |
| 通关问题参考答案 | `docs/01B-参考答案.md` |
| 核心自评清单（10/08 前勾完，本周至少勾 1-4） | `docs/05-核心自评清单.md` |
| 附件 A/B（配置默认值 / 0.8B 结构档案） | `docs/02` 附录 |
| V4 概念映射骨架 | `docs/04-V4概念映射.md` |
| 进度 | `进度.md` |