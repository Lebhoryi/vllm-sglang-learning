# 01 vLLM 请求生命周期走读导览（阶段一 W1，2026-09-08 版 / 09-24 仓库地图校正）

> 路径全部基于本地源码实测：`share/2026Q4/vllm`（main `00b7847c80` ≈ `v0.30.1rc0-60`，2026-09-24）。
> **版本演进**：`869f78732b`（0.28 初版）→ `8c34a37232`（v0.28.1rc0-626，09-11）→ **`00b7847c80`（v0.30.1rc0-60，09-24，跨 782 提交 / 核心目录 228 文件变更）**。
> **重要**：没有顶层 `vllm/core/`、`vllm/attention/`，引擎主体在 `vllm/v1/`。以下路径均为 `ls` 实测。
> SGLang 的对照在阶段三（docs/03），本阶段只读 vLLM。

## 仓库顶层地图（0.30，2026-09-24 实测）

常见教程那张树是简化过的，和本机仓库有几处实质性出入。走读以这张为准。

```
vllm/                              # 仓库根：share/2026Q4/vllm
├── vllm/                          # 核心 Python 包（from vllm import ...）
│   ├── v1/                        # ⭐ 当前引擎本体
│   │   ├── engine/                #    AsyncLLM / EngineCore / LLMEngine / core_client / admission_control
│   │   ├── core/                  #    Scheduler + KVCacheManager + BlockPool + kv_cache_coordinator
│   │   ├── worker/                #    gpu_worker + gpu_model_runner(V1)
│   │   │   └── gpu/               #  ⭐ Model Runner V2（默认路径，87 个 .py）
│   │   │                          #     model_runner / sample/ / spec_decode/ / mm/ / model_states/
│   │   ├── executor/              #    Uniproc / Multiproc / Ray
│   │   ├── attention/             #    ★ 后端选择：selector.py + backends/
│   │   ├── sample/                #    sampler / logits_processor
│   │   ├── structured_output/     #    结构化输出
│   │   ├── spec_decode/           #    投机解码
│   │   ├── metrics/               #    指标
│   │   ├── kv_offload/            #    KV 卸载（含 0.30 新增 tiering/ 分层卸载）
│   │   ├── simple_kv_offload/     #    简易卸载（disk/copy backend）
│   │   ├── hisparse/              #  🆕 0.30：host-resident sparse-MLA decode 热缓冲
│   │   ├── watermarking/          #  🆕 0.30：输出水印（强制 V2 runner）
│   │   ├── kv_hints/              #  🆕 0.30：编程式 KV 管理请求信封
│   │   ├── fault_tolerance/       #    容错
│   │   └── pool/                  #    pooling 模型
│   ├── engine/                    # 不是旧 V0 引擎：LLMEngine/AsyncLLMEngine 已是 v1 别名
│   │                              #    仍活着的是 arg_utils.py（EngineArgs，~3.1k 行）
│   ├── entrypoints/               # OpenAI/Anthropic/CLI/离线 LLM；llm.py 是离线门面
│   │                              #    api_server.py 仅 59 行，是 deprecation 壳；真入口在 launchers/
│   ├── config/                    # 35 个 .py；*Config 类约 37 个（不是 27）
│   ├── model_executor/            # 加载 / layers / models/（含 qwen3_5.py）
│   ├── models/                    # 新模型注册（deepseek_v4 等），和 model_executor/models 分开
│   ├── distributed/               # TP/PP/EP 通信
│   ├── multimodal/                # 图像 / 音频 / 视频
│   ├── platforms/                 # CUDA / ROCm / XPU / CPU 抽象
│   ├── lora/                      # LoRA
│   ├── compilation/               # torch.compile / CUDA graph 相关
│   ├── renderers/                 # prompt / chat 渲染（在线 + 离线都走这里）
│   ├── tokenizers/                # tokenizer 封装
│   ├── transformers_utils/        # HF 配置 / 权重工具
│   ├── sampling_params.py         # SamplingParams / BeamSearchParams
│   └── pooling_params.py          # pooling 模型参数
├── csrc/                          # C++/CUDA kernel（attention / moe / quantization）
├── rust/                          # Rust 辅助（tool parser 等，wheel 里是 .so）
├── cmake/  CMakeLists.txt         # 原生扩展构建
├── docker/ docs/ examples/        # 镜像、官方文档、示例
├── benchmarks/ tests/ tools/      # 基准 / 测试 / 开发工具
├── requirements/ scripts/         # 依赖清单 / 仓库脚本
└── pyproject.toml  setup.py
```

### 和常见树的出入

| 常见说法                                        | 本机 0.30 实测                                                                                                                                                                                                                                    |
| ----------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `v1/` 只有 engine / core / worker / executor  | 还有**attention / sample / structured_output / spec_decode / metrics / kv_offload / simple_kv_offload / hisparse / watermarking / kv_hints / fault_tolerance / pool**。走读 Q5 的 `selector.py` 就在 `v1/attention/`，不在那 4 个目录里 |
| `engine/` = 旧架构、正在被淘汰                | V0 引擎类已经没了。`llm_engine.py` / `async_llm_engine.py` 各 7 行，只是 `v1` 的别名；真正还在用的是 `arg_utils.py`（`EngineArgs`）                                                                                                     |
| `config/` 有 27 个配置类                      | `vllm/config/` **35 个 .py**（0.30 新增 `aux_output.py`、`watermarking.py`），`*Config` 约 **37** 个                                                                                                                          |
| 仓库根只有 csrc / benchmarks / tests / examples | 还有**cmake、docker、docs、rust、requirements、scripts、tools**。Rust 工具链是一等公民                                                                                                                                                      |
| 模型实现只在`model_executor/`                 | `model_executor/models/` 仍是主力（`qwen3_5.py`）；顶层 `vllm/models/` 另有一份新模型注册，不要搜错                                                                                                                                         |
| Worker =`gpu_model_runner.py`                 | **0.30 起 Model Runner V2 是默认**：`v1/worker/gpu/model_runner.py`（87 个 .py 的 V2 目录树）。`v1/worker/gpu_model_runner.py` 是**遗产路径**，仅在 V2 不支持某些特性时回退                                                       |
| `openai/api_server.py` 是服务入口             | 全文**59 行**，只是 `DeprecationWarning` + 转发。真入口 `entrypoints/cli/serve.py:46 ServeSubcommand` → `launchers/api_server/entry.py:67`                                                                                           |

已装环境：`site-packages/vllm` 的包内目录与上面一致，多出来的是编译产物（`*.abi3.so`、`vllm-rs/`），走读仍看源码仓库。

## vLLM 链路

```
HTTP 请求
  └─ vllm/entrypoints/                      # OpenAI API 层
      ├─ cli/serve.py:46 ServeSubcommand    # 真 CLI 入口
      └─ launchers/api_server/entry.py:67   # 组装 AsyncLLM
      └─ AsyncLLM
          = vllm/v1/engine/async_llm.py:80  # 对外引擎门面（vllm/engine/ 下的是兼容封装）
              ├─ vllm/v1/engine/core_client.py      # 与 EngineCore 进程通信（:80）
              ├─ vllm/v1/engine/core.py             # EngineCore：引擎主循环（step :633）
              ├─ vllm/v1/engine/input_processor.py  # 请求 → 输入 tensors（:41）
              ├─ vllm/v1/core/sched/scheduler.py    # ★ 调度器（请求入批/抢占，:79）
              │     ├─ async_scheduler.py / request_queue.py / interface.py / output.py
              ├─ vllm/v1/core/kv_cache_manager.py    # ★ KV cache 分配（:131）
              │     └─ block_pool.py / kv_cache_layout.py / kv_cache_coordinator.py
              ├─ vllm/v1/worker/gpu_worker.py        # 执行器（:183）
              │     └─ ⭐ 默认 V2: vllm/v1/worker/gpu/model_runner.py:183   # batch 前向
              │         回退 V1: vllm/v1/worker/gpu_model_runner.py:479
              │           └─ vllm/v1/attention/selector.py   # ★ 注意力后端选择（:105/:228）
              │                 └─ vllm/v1/attention/backends/
              │                     ├─ flash_attn.py       # 标准注意力
              │                     ├─ gdn_attn.py         # Gated DeltaNet（Qwen3.5 线性注意力）
              │                     ├─ linear_attn.py / mamba1_attn.py / mamba2_attn.py
              │                     └─ mla/               # MLA（V4 方向）
              ├─ vllm/v1/engine/output_processor.py     # 采样结果处理（:464）
              └─ vllm/v1/engine/detokenizer.py          # token → 文本回流（:166）
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

