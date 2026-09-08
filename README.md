# 一个月掌握 vLLM/SGLang 核心（2026-09-08 ~ 2026-10-08）

> 承接「模型适配岗 3 个月学习计划」（Documents id: e110b919）与「路线 v2」（id: 10d43e84）。
> 2026-09-08 用户重申：**学习窗口收紧为一个月，目标 = 掌握 vLLM/SGLang 核心**。
> 2026-09-08 节奏修订：**串行学习，vLLM 为主——不再双引擎并行**。

## 一、背景与口径

- 目标岗位：模型适配（DeepSeek V4 Flash / Qwen3.8-Flash / GLM-5.3-Flash 网络结构）；
- **vLLM / SGLang = 参考实现 + 性能标杆**，不是最终交付栈。
- **学习顺序（2026-09-08 修订）**：

  1. **阶段一/二（9/08-9/30）：全部精力给 vLLM**——跑通 → benchmark → 核心模块深读；
  2. **阶段三（10/01-10/08）：SGLang 对照学**——不从头学概念，按 vLLM 已掌握的框架逐项找对应物（Radix 前缀缓存、调度策略、管理器架构、注意力后端、V4 原生支持）。

  - 理由：vLLM 是主流参考实现，且 vllm-ascend（DeepSeek V4 的直接参考模板）同源 vLLM → **vLLM 权重最大**；SGLang 的差异化价值在概念成熟后一次对照吸收效率更高。
- 解剖主模型：**Qwen3.5-0.8B**（与 Qwen3.8-27B 同 `qwen3_5` 混合注意力架构，结果 1:1 迁移）；对照：Qwen2.5-0.5B-Instruct（标准注意力）。
- 实战模型：Qwen3.8-27B（BF16 ≈54GB，A100-80G TP1 需限参）。
- 硬约束：**主力实验机的 A100 可用至 2026-09-30** → vLLM 实验密集期压在 W1-W3；10 月 SGLang 用小模型对照，27B 对照 benchmark 为 stretch goal。

## 二、"核心"达成定义（10/08 可逐条自评）

1. **vLLM 请求生命周期**：从 API 请求到 token 流回的全链路——entrypoint → AsyncLLMEngine → EngineCore → Scheduler → Worker/ModelRunner → AttentionBackend → KV 管理器 → OutputProcessor/Detokenizer。能画图、能讲清每一跳。
2. **调度与批处理**：continuous batching、chunked prefill、preemption（swap vs recompute）、prefix caching，vLLM 的调度决策流程。
3. **KV cache 与显存规划**：PagedAttention block table、`gpu_memory_utilization` / `max_num_seqs` / `max_model_len` 三者如何互咬；MLA 低秩 KV cache 原理（为 V4 的 MQA 化 MLA 铺路）。
4. **注意力后端与 kernel**：FLASH_ATTN / 各后端的选择逻辑；GQA/MQA；**混合注意力**（Qwen3.5 的 Gated DeltaNet 线性注意力 + Gated Attention，vLLM `v1/attention/backends/gdn_attn.py` + `model_executor/layers/mamba/gdn/`）。
5. **vLLM 模型 executor 走读**：`models/qwen3_5.py` 装配逻辑，翻译成算子级词汇表（为 Lyngor 适配准备）。
6. **SGLang 对照（阶段三）**：能用 vLLM 的概念框架对照解释 SGLang 的 Radix prefix cache、schedule policy、tokenizer/detokenizer manager、flashinfer 后端、deepseek_v4 backend，填完对照矩阵。
7. **性能标杆方法论**：`benchmark_serving` 用法、TTFT / TPOT / 吞吐 / QPS 指标解读（vLLM 基线 9/21 完成；SGLang 对照为 stretch）。

## 三、三阶段排期与检查点

| 阶段                           | 日期        | 主线                                                                                                                                                                                                       | 检查点                                                                                |
| ------------------------------ | ----------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| **一 · vLLM 跑通**      | 09/08-09/14 | P1（vLLM 专属）：Qwen3.5-0.8B 起服务 → 请求生命周期走读（docs/01）                                                                                                                                        | **09/14**：vLLM 0.8B 跑通 + 生命周期图 + 走读笔记                               |
| **二 · vLLM 实战+核心** | 09/15-09/30 | P2：27B hf-mirror 下载 → vLLM 起服务（`--max-model-len 16384 --max-num-seqs 16 --gpu-memory-utilization 0.9`）→ `vllm bench serve` 基线；P3 核心模块：调度/KV-PagedAttention/GDN 混合注意力/MLA 深读 | **09/21**：27B vLLM benchmark 基线出炉；**09/30（硬）**：核心模块文档完成 |
| **三 · SGLang 对照**    | 10/01-10/08 | 装 SGLang → 0.8B 跑通 → 按 vLLM 已掌握概念逐项对照（docs/03）→ 填对照矩阵 → V4 概念映射 + 核心自评 + 十月 issue 候选池                                                                                 | **10/08**：对照矩阵完成 + 核心清单全勾                                          |

## 四、每日节奏（在职友好）

- 工作日：晚 2-3h = 1h 代码专题走读 + 1-1.5h 实验 + 10 分钟笔记。
- **纪律：同一时间只碰一个引擎**。阶段三之前不要起 SGLang 服务、不要读 SGLang 源码（装好环境除外）。
- 周末：集中实验（错峰用卡：深夜 / 午休 / 周末，先用 `nvidia-smi`）。
- 负载纪律：单卡实验先查卡；72B 级 / 多卡（A800 TP8）不在本月主线。

## 五、目录约定

```
README.md        本计划（串行三阶段版）
week1-vllm跑通/        W1（09/08-09/14）：0.8B 跑通 + 生命周期走读 → 检查点 09/14
week2-27B-benchmark/   W2（09/15-09/21）：27B 起服务 + benchmark 基线 → 检查点 09/21
week3-核心模块深读/    W3（09/22-09/30）：调度/KV/GDN 混合注意力深读 → 硬检查点 09/30
week4-sglang对照/      W4（10/01-10/08）：SGLang 对照 + V4 映射 + 自评 → 检查点 10/08
docs/00-阶段一执行清单.md   vLLM 专属执行清单（含 benchmark 步骤，覆盖 W1+W2）
docs/01-vLLM请求生命周期导览.md   走读地图 + 通关问题
docs/01B-参考答案.md   6 问答案（先自答再对，带源码行号）
docs/01C-生命周期参考图.md   mermaid+文本参考图（先画再对）
docs/02-vLLM调度与KV深读.md   深读三专题 + 附录 A/B/B2/C（W3 用）
docs/05-核心自评清单.md      （10/08 收尾自评）
docs/06-issue候选池.md       （P4 前置，模板+弱信号）
docs/07-基准解读卡.md        （9/21 用，v0.28 输出格式实测）
docs/08-学习材料索引.md      （三套材料 ↔ 三阶段映射）
docs/03-SGLang对照导览.md    （W4 用：对照矩阵模板 + SGLang 阅读地图）
docs/04-V4概念映射.md        （W4 收尾）
scripts/         起服务/下载/测试脚本（双引擎脚本都在，但按阶段使用；各文件用途见下文「scripts/ 脚本速查」）
logs/            服务与 benchmark 日志（不入 git）
test_vllms/      W1 历史实证（冻结目录）：2026-09-08 首跑的离线 LLM API 脚本与日志（进度.md 两个 ✅ 的原始证据，含 0.8B vs 27B 生命周期/显存对照素材）；不再往里放新东西，新日志一律进 logs/
models/          模型权重下载记录与软链（权重本体在主力实验机本地盘，不入 git）
```

### scripts/ 脚本速查

| 脚本                           | 用途                                                                                       | 默认端口 / 关键参数                                    | 使用阶段           |
| ------------------------------ | ------------------------------------------------------------------------------------------ | ------------------------------------------------------ | ------------------ |
| `set_env.sh`                | **个人环境配置**：模型路径 / 服务器 IP / 端口 / HF 镜像等；其它脚本自动加载，显式传入优先 | 变量见文件内注释 | 随时（先配）      |
| `env_check.sh`               | 一键环境核查：python / torch / vllm / sglang / flashinfer / transformers 版本 + nvidia-smi | —                                                     | 随时（W1 起）      |
| `download_models.sh`         | hf-mirror 下载 Qwen3.5-0.8B（主用）+ Qwen2.5-0.5B-Instruct（对照）到模型目录                    | 参数：目标目录（默认 `$MODEL_DIR`）                   | W1                 |
| `start_vllm_qwen35_08b.sh`   | vLLM 起 Qwen3.5-0.8B HTTP 服务                                                             | 8001；max-len 8192 / 32 seq / mem 0.85                 | W1                 |
| `start_vllm_qwen25_05b.sh`   | vLLM 起 Qwen2.5-0.5B-Instruct（标准注意力对照）                                            | 8003；max-len 8192 / 32 seq / mem 0.85                 | W1 机动 / 随时对照 |
| `start_vllm_qwen38_27b.sh`   | vLLM 起 Qwen3.8-27B HTTP 服务（限参内置）                                                  | 8002；max-len 16384 / 16 seq / mem 0.9                 | W2                 |
| `start_sglang_qwen35_08b.sh` | SGLang 起 0.8B 服务（阶段三才用）                                                          | 30000；mem-fraction 0.85                               | W4                 |
| `bench_qwen38_27b.sh`        | `vllm bench serve` 跑 27B 基线（v0.28 新 CLI，结果入 logs/）                             | 8002；200 prompts / in 512 / out 128，可用环境变量覆盖 | W2                 |
| `test_chat.sh`               | 通用对话冒烟：curl POST`/v1/chat/completions`（vLLM / SGLang 均兼容）                    | 参数：端口 / 模型名（默认 `$MODEL_08B`）/ 提问        | 随时               |

> 通用约定：模型目录默认来自 `set_env.sh`（`MODEL_08B` / `MODEL_05B` / `MODEL_27B`），可用 `MODEL_PATH`（bench 用 `MODEL`）显式覆盖，显式值优先；`PORT` 及 bench 的 `NUMPROMPTS` / `INLEN` / `OUTLEN` 可覆盖默认；服务日志自动 tee 到 `logs/`（时间戳命名）。⚠️ `set_env.sh` 含内网 IP 等隐私信息，勿提交 git / 勿外传。

## 六、V4 概念映射（W4 交付）

| V4 概念              | vLLM 对应实现 | SGLang 对应实现 | Lyngor 适配注意点 |
| -------------------- | ------------- | --------------- | ----------------- |
| CSA / HCA 混合注意力 |               |                 | 稀疏 mask 算子    |
| MQA 化 MLA           |               |                 | 低秩 KV 投影      |
| MoE + EP             |               |                 | EP 切分           |
| FP4/MXFP4 量化位点   |               |                 | 量化 kernel       |

## 七、环境速查（事实）

- 主力实验机：A100 80G sm_80，vllm 0.28.0 / torch 2.13.0+cu130 / transformers 5.16.1 / flashinfer 0.6.16 已装，**SGLang 未装（阶段三再装）**；卡可用至 09-30。
- 本地工作机：本机（DSH / NFS 共享盘 / 1080Ti×2 仅作 transformers 解剖台）。
- HF 下载一律 `HF_ENDPOINT=https://hf-mirror.com`（huggingface.co 不通）。
- git 操作在本地工作机做；主力实验机上对 NFS `.git` 执行 git 会超时。
- 模型权重下到主力实验机本地盘（NFS 读权重慢）。

