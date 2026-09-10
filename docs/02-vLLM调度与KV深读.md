# 02 vLLM 调度与 KV 深读（阶段二 / W3 用，2026-09-08 版）

> 前置：阶段一完成（docs/01 生命周期图 + 6 个通关问题自答）。
> 本导览把 docs/01 链路里带 ★ 的三块吃透：**调度、KV cache、注意力 kernel**。
> 路径全部实测自本地源码 `share/2026Q4/vllm`（main 869f78732b ≈ v0.28.1rc0-509）。

## 锚点：你的真实日志（test_qwen3.5_0.8b.log）

| 日志原文 | 源码位置 | 含义 |
|---|---|---|
| `Available KV cache memory: 67.08 GiB` | `vllm/v1/worker/gpu_worker.py` ~578 | KV 可用显存 |
| `GPU KV cache size: 5,788,118 tokens, Maximum concurrency for 262,144 tokens per request: 22.08x` | `vllm/v1/core/kv_cache_utils.py` ~1869 | 缓存容量与最大并发 |
| `Warming up Qwen Triton kernels for model_type=qwen3_5_text` | `qwen_triton_warmup.py` ~270 | 混合注意力 kernel 预热 |
| `Capturing CUDA graphs (mixed prefill-decode, PIECEWISE)` | `vllm/v1/worker/gpu_model_runner.py` | CUDA graph 捕获 |

**任务：把这三块读通，最后能自己算出日志里的数字**（67.08 GiB 怎么从显存规划来、5.7M tokens 怎么从 layout 来）。

## 专题 1：调度与批处理

文件（全部实测存在）：

- `vllm/v1/core/sched/scheduler.py` — 调度器主体（★）
- `vllm/v1/core/sched/async_scheduler.py` — 异步版（EngineCore 实际用哪个？）
- `vllm/v1/core/sched/request_queue.py` — 请求队列/优先级
- `vllm/v1/core/sched/interface.py` / `output.py` / `utils.py` — 数据结构
- 配置侧：`vllm/config/scheduler.py`（chunked prefill、调度策略）、`vllm/config/vllm.py`、`vllm/config/cache.py`（prefix caching）

通关问题：

1. `scheduler.step()` 的输入输出是什么？一次 step 里发生了什么？
2. preemption 在什么条件下触发？swap 与 recompute 怎么选？
3. chunked prefill 的 chunk 大小由谁定？默认开还是关？（`config/scheduler.py` 找）
4. prefix caching 命中判定在哪？（`config/cache.py` 参数 + scheduler 对应逻辑）

## 专题 2：KV cache 与显存

文件（全部实测存在）：

- `vllm/v1/core/kv_cache_manager.py` — KV 分配/释放（block 维度）（★）
- `vllm/v1/core/block_pool.py` — block 池
- `vllm/v1/core/kv_cache_coordinator.py` / `kv_cache_metrics.py` / `single_type_kv_cache_manager.py`
- `vllm/v1/kv_cache_layout.py` / `kv_cache_interface.py` / `kv_cache_spec_registry.py`
- `vllm/v1/worker/block_table.py` — PagedAttention 的 block table

通关问题：

1. 一个 token 的 KV 占多少字节？用日志反推：5,788,118 tokens ↔ 67.08 GiB，验证 `kv_cache_layout.py` 的计算（0.8B：24 层 / 8 头 / 2 KV 头 / head_dim？——先从 config.json 拿数字）。
2. block 大小（默认 16？）由谁决定？block_pool 与 kv_cache_manager 职责怎么分？
3. `gpu_memory_utilization` / `max_num_seqs` / `max_model_len` 三者如何互咬？在哪段代码算出来？（`gpu_worker.py` 启动段）
4. MLA 的 KV 为什么能更小？（为 V4 的 MQA 化 MLA 铺垫——读 `vllm/v1/attention/backends/mla/` 概念部分）

## 专题 3：混合注意力（GDN）+ MLA

文件（全部实测存在）：

- `vllm/model_executor/models/qwen3_5.py` — 模型装配（GDN 层与 gated attention 的排列）（★）
- `vllm/model_executor/layers/mamba/gdn/base.py` / `qwen_gdn_linear_attn.py` / `kimi_gdn_linear_attn.py` / `olmo_gdn_linear_attn.py`
- `vllm/v1/attention/backends/gdn_attn.py` / `linear_attn.py` / `mamba1_attn.py` / `mamba2_attn.py` / `short_conv_attn.py`
- `vllm/v1/attention/backends/mla/` — MLA 后端（V4 方向）
- 预热：`qwen_triton_warmup.py`（日志里 `model_type=qwen3_5_text` 的来源）

通关问题：

1. Qwen3.5 每层怎么混用 GDN 与 gated attention？（config.json 里的 layers 配置）
2. GDN 线性注意力与标准 attention 在 kernel 层差异？（linear_attn vs flash_attn）
3. selector 在什么条件下选 gdn_attn / mamba / mla？（`v1/attention/selector.py` + `backends/registry.py`）
4. MLA 低秩投影如何缩小 KV？与 V4 的「MQA 化 MLA」什么关系？

## 输出要求（9/30 前）

1. 三篇专题笔记（每篇含代码跳转记录 + ≥2 条「原来如此」）
2. 「显存规划小实验」：用日志数字（67.08 GiB / 5.7M tokens / 22.08x concurrency）自己算一遍并对上 = 通过
3. 一页「Qwen3.5 层结构图」（GDN / gated attention 交替排列）

衔接：阶段三（docs/03）将用这套概念对照 SGLang（Radix 前缀树 / schedule policy / hybrid 后端）。

## 附录 A：配置默认值速查（2026-09-08 源码核实，先自查再对）

| 配置 | 默认值 | 位置 |
|---|---|---|
| `enable_chunked_prefill` | **True（v0.28 默认开）** | `vllm/config/scheduler.py:116` |
| `max_num_batched_tokens`（chunk 上限由它定） | **2048**（`DEFAULT_MAX_NUM_BATCHED_TOKENS`） | `vllm/config/scheduler.py:42/49` |
| `enable_prefix_caching` | **True（v0.28 默认开）** | `vllm/config/cache.py:139` |
| `vllm serve` 全部参数入口 | `vllm/entrypoints/launchers/cli_args.py`（`make_arg_parser`）；浏览方式 `vllm serve --help=ModelConfig` / `--help=all` | cli/serve.py 只做启动 |

> 注意：chunked prefill 与 prefix caching **默认开启**是老版本教程的差异点（老版默认关/需显式开）。另外 scheduler.py:1035 附近有"chunked prefill has to be enabled explicitly"的提示——那是 spec decode / MTP 等特定场景的例外，别被误导。

## 附录 B：Qwen3.5-0.8B 结构档案（config.json 实测 2026-09-08）

> 来源：hf-mirror `Qwen/Qwen3.5-0.8B/config.json`（`text_config`）。**先按此档案自己算 KV 显存，再对日志数字**。

### 层结构（`layer_types`，24 层）= `[linear ×3 + full ×1] × 6`（`full_attention_interval=4`）

```
L00 L01 L02 L03     L04 L05 L06 L07     ... 共 6 组
lin lin lin FULL    lin lin lin FULL    → 18 线性层 + 6 full attention 层
```

### 关键参数

| 部分 | 参数 | 值 |
|---|---|---|
| full attention | `num_attention_heads` / `num_key_value_heads`（GQA）/ `head_dim` | 8 / **2** / **256** |
| linear (GDN) | `linear_num_key_heads` / `linear_num_value_heads` | **16 / 16** |
| linear (GDN) | `linear_key_head_dim` / `linear_value_head_dim` / `linear_conv_kernel_dim` | **128 / 128 / 4** |
| linear (GDN) | `mamba_ssm_dtype` | **float32**（状态精度） |
| 其他 | `max_position_embeddings` / `rope partial_rotary_factor` | 262144 / 0.25（mrope 11/11/10） |
| vision | `vision_config`：12 层塔，hidden 768→out 1024 | 多模态 |
| mtp | `mtp_num_hidden_layers: 1` | MTP 头 |

### KV 显存推导框架（专题 2 的"通关关卡"）

日志锚点（你的真实数字）：

- 0.8B：67.08 GiB ↔ **5,788,118 tokens ≈ 12.4 KB/token**
- 27B：18.61 GiB ↔ **298,275 tokens ≈ 65.5 KB/token**

公式骨架：

```
KV_bytes/token = Σ_层 (kv_heads × head_dim × dtype_bytes)  ×（该层是否按 token 计）
```

**矛盾引导**（先自己想，再翻源码）：

- Q1：仅 full attention 6 层：2 × 256 × 2B × 6 = 6 KB/token，只占 12.4 KB 的一半不到——**剩下一半是什么**？
- Q2：日志里 `attention block size 784` = mamba page → 线性层状态**不是每 token 一份**，而是按块存。那 784 意味着什么？摊到每 token 多少？（读 `vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py` 的 state shape + `vllm/v1/core/kv_cache_layout.py`）
- Q3：27B 的 65.5 KB/token ≈ 0.8B 的 5.3 倍——27B 是 64 层（0.8B 的 2.67 倍）……差的另外 2 倍从哪来？（head_dim？KV 头数？等 27B config.json 到手核对）

> 把这 3 问算通 = 专题 2 过关。「算出来的数对不上日志」正是你该追的线索，不是 bug。

## 附录 C：混合注意力 serving 专属 flag（2026-09-08 源码核实）

定义位置：`vllm/config/cache.py`（MambaCacheMode:71、字段 :181-215）。

| flag | 取值 | 默认 | 含义 |
|---|---|---|---|
| `--mamba-cache-mode` | `none` / `all` / `align` | `none` | mamba 状态缓存策略：none=不缓存（prefix caching 关时）；all=缓存所有 i×block_size 位置的 state；**align=只缓存每个 scheduler step 末尾且位于 i×block_size 的 state（prefix caching 开时的默认）** |
| `--mamba-cache-dtype` | auto/bfloat16/float32… | `auto` | conv + ssm state 的 dtype（auto = 从模型 config 推断，Qwen3.5 是 float32） |
| `--mamba-ssm-cache-dtype` | 同上 | `auto` | 只控制 ssm state（conv 仍归 mamba_cache_dtype） |
| `--mamba-block-size` | 8 的倍数 | — | 连续缓存块大小（仅 prefix caching 开启时可设；需 8 对齐 causal_conv1d kernel） |
| `--use-replayssm` | bool | False | Kimi-K3 ReplaySSM 解码 kernel（需 none/align + Triton/FlashInfer mamba 后端） |

**Qwen3.5 特例**：`mamba_cache_mode="all"` 直接 NotImplementedError（`models/qwen3_5.py:332-336`，让你改用 align）。

**和你日志的对应**：prefix caching 默认开 → mamba_cache_mode 默认走 **align** → 需要 attention page 与 mamba page 对齐 → 于是 `Setting attention block size to 784 tokens`（interface.py:911/935）。两个默认配置的交互，就是你日志里那行日志的完整因果链。

> **0.29.0 追加修复（2026-09-11 核实，全部不在 0.28.0）**：
> - `#55760` / `#55861`：Mamba/hybrid + EAGLE 的 `prefix_cache_retention_interval` 未设置时默认 **dense**（0.28.x 默认仅语义 checkpoint）→ 0.29.0 起 0.8B/27B 的 prefix cache 命中行为/显存占用可能微变
> - `#54044`：mamba align metadata 在 profiling teardown 时重置（修 stale 缓存）
> - `#52743`：GDN decode 的 Ampere preprocessor guard 修复——**A100 (sm_80) 直接受益**，0.28.0 无此修复
> - `#53663` / `#51358`：Mooncake mamba boundary state 修复（同 #43559 主题，暂不涉及）
> - 依赖连带：flashinfer-python 0.6.16 → **0.6.18**（0.29.0 硬依赖）；torch 2.13.0、transformers 5.16.1 均满足
> - `vllm bench serve` / serve CLI **无破坏性变更**（Rust bench 仅 flag parity 对齐）→ 9/21 脚本直接可用

### 附录 B2：Qwen3.8-27B 结构档案（对照用，config.json 实测同日）

| 维度 | 0.8B | 27B |
|---|---|---|
| 层数 / 结构 | 24 = [lin×3+full]×6 | **64 = [lin×3+full]×16**（全 4 间隔同） |
| full：KV 头 / head_dim | GQA 2 / 256 | **GQA 4 / 256** |
| linear：key 头 / value 头 / dim | 16 / 16 / 128 | **16 / 48 / 128**（value 头 3 倍） |
| `mamba_ssm_dtype` | float32 | float32 |
| vision 塔 | 12 层，hidden 768 | **27 层，hidden 1152** |
| tie_word_embeddings | true | false |

**Q3 升级提示（先自己算再对）**：full attention 部分的每 token KV——

- 0.8B：2 头 × 256 × 2B × 6 层 = 6,144 B
- 27B：4 头 × 256 × 2B × 16 层 = 32,768 B
- 比值 = **32,768 / 6,144 ≈ 5.33**
- 而日志 KV 容量比值 65.5 / 12.4 ≈ **5.28**——几乎一致！

这个"巧合"说明：**KV 大头由 full attention 层决定，线性层（按块存 state）摊薄后占比很小**——这正是混合注意力"省 KV"的本质，也是 V4 思路的预演。验证路径：把 linear 层按 784-token page 摊薄算一遍，看它是不是真的只占零头。