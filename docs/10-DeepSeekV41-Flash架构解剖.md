# 10 DeepSeek-V4.1-Flash 架构解剖（论文 ↔ 参考实现 ↔ 权重 ↔ vLLM 四方对照）

> 定位：**不提 PR**，把官方参考实现当解剖台。方法沿用 docs/09 的「测试/契约驱动深读」——
> 先拿到可验证的一手材料（源码 + 权重名 + 论文），再逐条闭环，而不是读二手解读。
> 与你现有文档的关系：docs/02 专题 2 是「KV cache 怎么按 block 分」，本文是**极端反例**——
> 一个把 KV 压到 890 bytes/token 的模型，KV 根本不按层来。

**材料来源（本机已落地，无需重下）**：`/home/lebhoryi/share/2026Q4/dsv41-ref/`

| 文件 | 大小 | 用途 |
|---|---|---|
| `inference_model.py` | 61549 B / 1309 行 | 官方参考实现主体 |
| `inference_kernel.py` | 23790 B / 591 行 | tilelang kernel（9 个） |
| `inference_engram.py` | 8138 B / 184 行 | n-gram 哈希布局与查表 |
| `inference_vision.py` | 4457 B / 119 行 | DeepSeek-ViT + Aligner |
| `inference_convert.py` | 9458 B / 205 行 | HF 权重 → TP 分片 |
| `inference_config.json` | 1982 B | **驱动 ModelArgs 的真实超参** |
| `config.json` | 3311 B | HF 格式 config（键名不同，见 §4） |
| `index.json` | 7.47 MB | 权重索引：96085 个张量名 ← **最硬的一手证据** |
| `Tech_Report.pdf` / `.txt` | 1.81 MB / 207638 B | 论文（pdftotext 已转文本） |

> 获取方式备忘：本机 `huggingface.co` 不通（HTTP 000），`hf-mirror.com` 通。
> **重拉材料用 `scripts/fetch_dsv41_ref.sh`**（已冒烟测试通过，含 index.json 与 PDF 转文本）；
> 或手动：`curl -sL https://hf-mirror.com/deepseek-ai/DeepSeek-V4.1-Flash/resolve/main/<path>`。
> 权重本体 **510.3 GB** 不拉 —— 结构分析只需要 `index.json`（7.47 MB）。

---

## 0. 结论先行：为什么这条线不该做 PR

| 目标 | 现状（2026-09-10 实测） | 判断 |
|---|---|---|
| transformers 接入 | main **无** `models/deepseek_v41`；本机 5.15.0 `CONFIG_MAPPING['deepseek_v41']` → KeyError | 但 config 写 `transformers_version: "5.6.0"` → **官方手里已有实现** |
| vLLM 接入 | 本地 main `8c34a37232` 里 `grep -ri v41` **完全为空** | 上游 1 小时内已开 15 个 PR，作者全是 vLLM member → **已被认领** |

**结论**：缺的不是「接入」，是「读懂」。参考实现是一份罕见的、官方亲自写的、可读的完整推理栈——
这是市面上唯一能把论文 §2 每一句话落到代码行的材料。本文就是把它拆开。

---

## 1. 一页纸：数据流与九大组件

```
input_ids ──► ParallelEmbedding ──► unsqueeze(2).repeat(hc_mult) ──► [B,S,4,5120]
                                          │  ← Hyper-Connections：残差流是 4 份并行拷贝
                                          ▼
   ┌──────────── 40 × Block（20 encoder + 20 decoder）────────────┐
   │  [可选] Engram 写入残差流（layer 1, 14）                      │
   │  hc_mixes ─► hc_pre ─► attn_norm ─► Attention ─► hc_post      │
   │  hc_mixes ─► hc_pre ─► ffn_norm  ─► MoE       ─► hc_post      │
   └───────────────────────────┬──────────────────────────────────┘
                               ▼
        hc_pre（塌回单份）─► RMSNorm ─► ParallelHead ─► sample
```

| # | 组件 | 论文 | 参考实现 | 一句话 |
|---|---|---|---|---|
| 1 | **CED** 因果编解码器 | §2.2 eq.(1) | `kv_source_layers` + `SharedAttentionRuntime` | 解码器的全局 KV 只造一次，20 层共用 |
| 2 | **CSA2** 三模式 | §2.3.1 图 4 | `is_kv_source` / `is_index_source` / 都不是 | Full / Reindex / Reuse 由两个 id 列表静态决定 |
| 3 | **两级稀疏 indexer** | §2.3.2 图 5 | `Indexer` + `select_candidate_blocks` | 先粗选 block，再精挑 Top-512 |
| 4 | **SWA** | §2.1 | `_window_kv` + 环形缓冲 | 每层 128 窗口，唯一「按层保留」的 KV |
| 5 | **FP4 主 KV** | §2.4.4 | `fp4_act_quant(latent, 16, ..., e4m3)` | E2M1 + 每 16 通道一个 E4M3 scale |
| 6 | **Engram** | §2.4.2 | `Engram` + `EngramLayout` + `NgramHashState` | 2/3/4-gram 哈希查表，196B 参数 |
| 7 | **Single-Pass mHC** | §2.4.1 eq.(6) | `hc_mixes` / `hc_pre` / `hc_post` | 混系系数**错位一层**使用 |
| 8 | **DSpark** | §2.4.3 | `DSparkBlock` / `DSparkMarkovHead` / `ConfidenceHead` | 半自回归草稿 + 置信度调度 |
| 9 | **MoE + VL 分流** | §2.1.1 | `Gate` + `bias_vl` | 图像/文本各一套校正偏置 |

### 关键数字（全部与 `inference_config.json` 对得上）

| 项 | 值 |
|---|---|
| backbone 参数 / 激活 | 552B / **8B prefill、16B decode** |
| 层数 | 40 = 20 causal encoder + 20 decoder |
| 全局 KV 占用 | **890 bytes/token**（≈ V4-Flash 的 1/4） |
| 持久化 KV（SSD/host） | ≈ V4-Flash 的 1/8（靠 SWA Bounded Replay，参考实现**不含**） |
| hidden / heads | 5120 / 64 Q : **1 KV**（shared-KV MQA） |
| head_dim / rope_head_dim | 512 / 64（q_lora 1280，o_lora 1024，o_groups 8） |
| MoE | 384 routed + 1 shared，激活 6/token，`sqrtsoftplus` |
| 窗口 / compress_ratios | 128 / `[0,0,2×18,1×20,0,0,0]`（43 项含 3 个 MTP 层） |
| indexer | 32 头 × head_dim 128，Top-512，候选池 2048 block × block_size 8 |
| Engram | 196B 参数，层 1/14，N-gram {2,3,4} × 8 头 × 256 dim |
| DSpark | 3 层草稿，block_size 5，markov_rank 256，目标层 37/38/39 |
| 上下文 | 1M（rope YaRN factor 16，original 65536，theta 10000；压缩 KV 用 theta 160000） |

---

## 2. 源码地图（`inference_model.py`，1309 行）

行号实测自本地文件，可直接跳转。

| 行 | 符号 | 职责 | 论文 |
|---|---|---|---|
| L24-30 | 全局常量 | `fp8_block_size=32`、`fp4_block_size=32`、`scale_fmt="ue8m0"` | §2.4.4 |
| L45 | `ModelArgs` | 字段名 = config JSON 键名。**默认值是小模型**（5 层/dim 1024/8 experts），可 `python model.py` 跑 | — |
| L152 | `ParallelEmbedding` | 词表维切分 + 越界零掩码后 all_reduce | — |
| L181 | `linear()` | 按权重 dtype 选 GEMM：fp4 / fp8 / bf16 | — |
| L210-278 | `Linear` / `ColumnParallelLinear` / `RowParallelLinear` | TP 三种切分 | — |
| L296 | `ParallelEngramEmbedding` | 表按行切分，**表存 fp8**，查表时反量化 | §2.4.2 |
| L328 | `Engram` | 查表 → wkv → 门控（点积 → signed sqrt → sigmoid） | §2.4.2 |
| L368 | `precompute_freqs_cis` | YaRN 频率（beta_fast/slow 线性斜坡） | §2.1 |
| L392 | `apply_rotary_emb` | `inverse=True` 是**共轭**——把 Q 的旋转从输出上撤掉，KV 保持单一旋转态 | — |
| L409 | `get_window_topk_idxs` | 环形缓冲的 slot 索引，-1 = 空槽 | §2.1 |
| **L429** | **`Compressor`** | 把 `compress_ratio` 个 token 池化成 1 个 KV latent（学习式 softmax 门控） | §2.3 |
| **L488** | **`Indexer`** | 侧路小注意力，为每个 query 选 `index_topk` 个压缩位置 | §2.3.2 |
| **L583** | **`select_candidate_blocks`** | 两级 top-k 的**第一级**：按 block 内最高分选 `topk_blocks` 个 block | §2.3.2 |
| **L613** | **`Attention`** | 两种 KV 源拼成**一次** `sparse_attn` 调用 | §2.3 |
| L700 | `Attention._window_kv` | SWA 的 K，fp8 量化（RoPE 尾部也在块内） | §2.4.4 |
| L722 | `Attention._compress_topk_idxs` | 非 index source 直接复用 `shared_attn.topk_idxs` | §2.3.1 |
| L739 | `Attention._compress_kv` | 只有 kv source 造 latent；indexer 必须在写缓存**之前**跑 | §2.3 |
| L792 | `Gate` | MoE 门控；`bias` 只影响选择、不影响权重；`bias_vl` 走图像分支 | §2.1.1 |
| L830 | `Expert` | SwiGLU；`swiglu_limit` 双向夹 up、单向夹 gate | — |
| L854 | `MoE` | 384 专家按 rank 切分，`self.experts[i]=None` 表示别人持有 | — |
| **L907** | **`Block`** | mHC：`hc_pre` 塌缩 / `hc_post` 展开 + `comb` 混残差 | §2.4.1 |
| L948 | `Block.hc_mixes` | 一次投影切出 pre/post/comb 三组系数，`comb` 经 Sinkhorn 双随机化 | §2.4.1 eq.(2) |
| L997 | `ParallelHead` | logits 直接 fp32，all_gather 拼词表 | — |
| L1032 | `DSparkAttention` | 草稿层注意力：窗口 KV 复用主模型，草稿 token 追加在后面 | §2.4.3 |
| L1077 | `DSparkMarkovHead` | 建模草稿 token 间的依赖 | §2.4.3 |
| L1089 | `DSparkConfidenceHead` | 逐位置条件接受概率 | §2.4.3 |
| L1100 | `DSparkBlock` | `mtp.*` 命名空间；stage 0 有 `main_proj`，末 stage 有 markov/confidence | §2.4.3 |
| L1159 | `make_identity_pre_mix` | 初始 one-hot（只用第 0 份拷贝） | — |
| **L1166** | **`SharedAttentionRuntime`** | **CED/CSA2 的枢纽**：4 个槽位跨层传值 | §2.2/§2.3.1 |
| L1183 | `Transformer` | 装配；建它时会设置文件顶部的全局变量 | — |
| L1241 | `Transformer.forward` | 主前向 | — |
| L1274 | `Transformer.forward_spec` | DSpark 草稿前向（**SD 调度循环不在本仓库**） | §2.4.3 |
| L1285 | `sample` | Gumbel-max，避免 GPU→CPU 同步 | — |
| L1295 | `__main__` | 自测：prefill 128 + 22 步 decode + 每次 DSpark | — |

**⚠ 参考实现的边界（官方自述）**：*"A readable reference implementation rather than a production serving engine."*
它覆盖**模型结构**，不覆盖**服务系统**：SWA Bounded Replay、持久化 KV/SSD offload、RDMA 预取、DSpark 调度器、长上下文优化全部不在内。
论文 §3.2 讲的那些，在这里找不到——这是读论文时最容易踩的坑。

---

## 3. 核心机制源码级拆解

### 3.1 CED：解码器的 KV 到底从哪来（★ 与论文有出入，已用权重定论）

论文 §2.2 eq.(1) 字面读法是：对解码器第 *l* 层（*l* > L/2），KV 由 H_{L/2} 乘**层相关的** W_l^KV 得到。

参考实现的实现方式完全不同，靠 `SharedAttentionRuntime`（L1166-1180）：

```python
class SharedAttentionRuntime:
    """Layers run in order and every source writes before its consumers read,
    so one slot each is enough and nothing needs resetting between forwards."""
    compress_kv: torch.Tensor | None = None   # ← 主 KV
    index_k:     torch.Tensor | None = None   # ← indexer 的 K
    topk_idxs:   torch.Tensor | None = None   # ← Top-K 选择结果
    candidates:  torch.Tensor | None = None   # ← 候选 block 池
shared_attn = SharedAttentionRuntime()        # L1180 进程级单例
```

**用权重索引定论**（`index.json`，96085 个张量名）：

| 现象 | 实测 |
|---|---|
| 带 `attn.compressor` 的层 | **只有 2, 8, 14, 20** |
| 层 2/8/14 的 compressor 模块 | `wkv` + `wgate` + `norm`（ratio=2，有池化门控） |
| 层 20 的 compressor 模块 | `wkv` + `norm`（**无 wgate** → ratio=1，不池化） |
| 带 `attn.indexer` 的层 | 2, 8, 14, 20（含 `wk`/`k_norm`）＋ 24, 28, 32, 36（**无** `wk`/`k_norm`） |
| 层 21、25 的 attn 模块 | 只有 `attn_sink/wkv/kv_norm/q_norm/wq_a/wq_b/wo_a/wo_b`，**没有 compressor 也没有 indexer** |
| 搜 `w_l` / `ced` / `kv_proj` | **一个都没有** |

→ **实物的解码器全局 KV 只造一次（层 20），21-39 层全部共用**。论文 eq.(1) 里的「layer-dependent W_l^KV」在权重里不存在 20 套。

**我的解释（待官方澄清）**：论文 eq.(1) 是 CED 的一般表述；本项目里 CED 与 CSA2 的跨层 KV 复用叠加后，
只有解码器的 Full-mode 层（层 20）真正拥有 W^KV，其余层走「复用」。
W_l^Z 之所以也没有，是因为解码器 ratio=1 → `Compressor.forward` 直接短路（L461-462：*"one token per group: nothing to pool, so no gate and no fp32"*），根本不需要压缩权重。

这也是为什么 `Attention.forward` 对两类层走同一条代码路径 —— 差别只在 `is_kv_source` 这一个布尔量。

### 3.2 CSA2 三模式：两个 id 列表决定一切

`inference_config.json`：

```json
"compress_ratios":      [0,0, 2×18, 1×20, 0,0,0]        // 43 项，含 3 个 MTP 层
"kv_source_layers":     [2, 8, 14, 20]                   // 谁造主 KV
"index_source_layers":  [2, 8, 14, 20, 24, 28, 32, 36]   // 谁跑 indexer
"candidate_source_layer": 20
```

按论文图 4 的三种模式归类（代码 L653-661 + L725-726）：

| 模式 | 判据（代码） | 层 | 行为 |
|---|---|---|---|
| **Full** | `is_kv_source ∧ is_index_source` | 2, 8, 14, 20 | 自己造主 KV + 自己的 indexer K + 全新 Top-K |
| **Reindex** | `is_index_source ∧ ¬is_kv_source` | 24, 28, 32, 36 | 复用主 KV 和 indexer K，用**自己的** Q 重打分 → 全新 Top-K |
| **Reuse** | 两者皆非且 `compress_ratio > 0` | 3-7, 9-13, 15-19, 21-23, 25-27, 29-31, 33-35, 37-39 | 直接吃 `shared_attn.topk_idxs`，不算 indexer |
| SWA-only | `compress_ratio == 0` | 0, 1（＋3 个 MTP 层） | 论文 §2.1：「前两层只用 SWA」✅ |

论文说「解码器第一个 Full Mode 层建立共享候选池」——代码里就是
`candidate_source_layer=20`，且 `uses_candidates = 0 <= candidate_source_layer < layer_id`（L503），
即**只有 20 之后的层**才吃候选池。完全对应。

### 3.3 两级稀疏 indexer（`Indexer` + `select_candidate_blocks`）

```
第一级 select_candidate_blocks（L583）：把 -inf 补齐到 block_size，
       unflatten(-1, (-1, block_size)).amax(-1)  → 每个 block 取最高分
       ★ 关键技巧 L604：pin 住「包含本 query 最新位置」的那个半满 block（置 +inf），
         否则它会被更老的、已填满的 block 挤掉 —— 而它装着最新的 token
       topk(candidate_topk_blocks=2048) → bool mask

第二级 Indexer.forward（L527）：index_score 在候选池内重算，再 topk(index_topk=512)
```

**为什么 indexer 必须在写缓存之前跑**（L529-531 注释，很值钱）：
indexer 的 K 是从「未加 RoPE 的 latent」推出来的，而 `Attention._compress_kv` 随后会用
「加过 RoPE、量化成 FP4」的值**覆盖同一块显存**。顺序错了就静默算错。

`Indexer.owns_k`（L500）只看 `kv_source_layers` —— 所以 Reindex 层（24/28/32/36）**没有自己的 K cache**，
它们读 `shared_attn.index_k`，权重索引已确认（无 `wk`/`k_norm`）。✅

### 3.4 890 bytes/token 是怎么来的（★ 从 config 就能自己推导，与论文数字精确吻合）

论文反复强调「890 bytes/token」。这个数字**不需要跑模型**，纯按 config 算就出来：

| 组成 | 谁 | 条目尺寸 | 每 token |
|---|---|---|---|
| 解码器主 KV | 层 20，ratio=1，1 份 | 512 dim × 0.5 B (FP4) = 256 B<br>+ 512/16 = 32 个 E4M3 scale = 32 B → **288 B/entry** | 288 × 1 = **288 B** |
| 编码器主 KV | 层 2, 8, 14，ratio=2，3 份 | 同上 288 B/entry | 288 ÷ 2 × 3 = **432 B** |
| indexer K | 层 2,8,14 (ratio 2) + 层 20 (ratio 1)，共 4 份 | 128 dim × 0.5 = 64 B<br>+ 128/32 = 4 个 E8M0 scale = 4 B → **68 B/entry** | 34×3 + 68 = **170 B** |
| | | | **合计 890 B** ✅ |

**代码里有一行注释直接确认这个算法**（L759）：

> `# Compressed KV uses groups of 16 with E4M3 scales; the indexer uses 32 with E8M0.`

对应 L760 `fp4_act_quant(latent, 16, True, scale_dtype=torch.float8_e4m3fn)` 与
L546 `fp4_act_quant(k, fp4_block_size, True)`（`fp4_block_size=32`，默认 E8M0）。

**SWA KV 不参与这个数**：它保持 FP8（L707），且被 `window_size=128` 限住，不随上下文线性增长。
论文 §2.4.4：「We retain FP8 for the SWA KV cache due to its sensitivity to quantization.」✅

> 顺带：**196B Engram 参数**同样可自证 —— `engram_num_embeddings = [384006168, 384016682]`，
> head_dim 256：
> 384,006,168 × 256 × 2 层 ≈ **196.6e9** = 论文说的「196B Engram parameters, evenly across two modules」✅
> 表的行长为什么是 384M？`EngramLayout.from_args`（engram.py L102）：3 个 n-gram 阶 × 8 个头 = 24 个
> **互不重复的质数**桶，从 `engram_vocab_size-1 = 16000000-1` 起 `find_next_prime` 累加。
> 本机实跑复现：24 个质数之和 = **384006168**，与 config 的 `engram_num_embeddings[0]` **逐位相同**（第 1 层 = 384016682）✅
> 两个层的质数序列不重用（`seen` 集合），所以第二层略大 —— 这个细节只有照代码跑才能对上。

#### 参数都去哪了（用 510.3 GB 反向核对）

HF API 实测：48 个分片合计 **510.3 GB**，其中**最大的两个分片各 101.54 GB**（`model-00047/48`）。
按组件算一遍，正好对上 —— 这两个巨无霸就是 **Engram 表**：

| 组件 | 参数量 | 精度 | 占用 |
|---|---|---|---|
| MoE 专家（384 × 40 层 × w1/w2/w3） | 384×40×3×(5120×2304) ≈ **543.6B** | FP4（0.5 B/elem） | ≈ 272 GB |
| ↳ 专家的 FP8 scale（每 32 元素） | 543.6e9/32 ≈ 17.0e9 | E8M0（1 B） | ≈ 17 GB |
| **Engram 表（2 层）** | 196.6e9 | FP8 | **≈ 203 GB** ← 2 × 101.54 GB ✅ |
| MTP/DSpark（128 专家 × 3 stage） | ≈ 13.6B | FP4 | ≈ 7 GB |
| attention + dense + shared expert（40 层） | ≈ 4.2B | FP8 | ≈ 4 GB |
| **合计** | | | **≈ 503 GB** ≈ 510.3 GB ✅ |

> 副产品结论：**MoE 专家占参数量的 ~96%（543.6B / 565B）**，但激活只有 6/384。
> 这正是 `# Backbone Params 552B` 与 `# Activated 8B/16B` 差距的来源，
> 也是 vLLM 那一串 PR 里 `--enable-expert-parallel` 是刚需的原因。

### 3.5 Single-Pass mHC：系数错位一层

论文 eq.(6)：`X_{l+1} = B_l X_l + C_l F_l(A_{l-1} X_l)` —— 注意是 **A_{l-1}**，不是 A_l。
这个「错位」让输入混合不再依赖本步算出的系数，从而消除 kernel 间的依赖。

代码 L968-994 精确实现了这一点，并且**注释直接点破**：

```python
# L976-978: "Each sub-block's own `hc_mixes` produces the mix for the *next* one,
#            so attention uses what the previous layer's FFN produced and
#            the FFN uses what this attention produced."
def forward(self, x, start_pos, pre_mix, image_mask, *attn_args):
    residual = x
    attn_pre, attn_post, attn_comb = self.hc_mixes(x, self.hc_attn_fn, ...)
    x = self.hc_pre(x, pre_mix)          # ← 用的是「上一层传进来的」pre_mix
    x = self.attn_norm(x); x = self.attn(x, start_pos, *attn_args)
    x = self.hc_post(x, residual, attn_post, attn_comb)

    residual = x
    ffn_pre, ffn_post, ffn_comb = self.hc_mixes(x, self.hc_ffn_fn, ...)
    x = self.hc_pre(x, attn_pre)         # ← 用本层 attn 算出的 attn_pre
    x = self.ffn_norm(x); x = self.ffn(x, image_mask)
    x = self.hc_post(x, residual, ffn_post, ffn_comb)
    return x, ffn_pre                     # ← 把 ffn_pre 交给下一层
```

入口处 `make_identity_pre_mix`（L1159）给第一层一个 one-hot 初值（只用第 0 份拷贝）。
`comb` 的 Sinkhorn 双随机化在 kernel 里：`hc_split_sinkhorn`（kernel.py L465，20 次迭代）。

### 3.6 Engram：哈希怎么算的

`NgramHashState.forward`（engram.py L160-184）三步：

1. `token_map[input_ids]` → 压缩词表 id；图像 span 位置写成 `DEAD = -1`
2. 取前 `max_ngram_size` 个位置（含自身），被 DEAD/越界截断的填 `pad_id`
3. 逐 lookback **XOR** 累乘结果，`% primes[阶][头]` 落到各自的质数桶，再加 `offsets` 变成全局行号

> `assert vocab_size == args.engram_compressed_vocab_size`（L146）—— 注释说得很直白：
> 「every hash multiplier derives from the compressed vocab size, so **a mismatch there would silently rehash the whole table**」。
> 这是重实现时最容易静默出错的地方。哈希乘子用 `np.random.default_rng(10007 * layer_id)` 生成（L75），
> 即**每个 layer 的哈希都不同**，且强制取奇数、限制上界防 int64 溢出。

`Engram.forward`（L350-365）的门控也很特别：`sigmoid(copysign(sqrt(|dot|), dot))` ——
**先取绝对值开方再带上符号**，注释说「matching the training kernel」。这类细节抄错就数值不对。

### 3.7 MoE 的 VL 分流

`Gate.forward`（L809-827）三件事值得记：

```python
bias = torch.where(image_mask.unsqueeze(-1), self.bias_vl, bias)   # L820
indices = (scores + bias).topk(self.topk, dim=-1)[1]               # 偏置只决定「选谁」
weights = scores.gather(1, indices)                                # 权重用「未加偏置」的分
weights /= weights.sum(dim=-1, keepdim=True) + 1e-20               # L825: 不是 norm_eps！
```

- 权重索引确认 `ffn.gate` + `ffn.gate.bias_vl` 两套偏置都在（论文 §2.1.1 的无辅助损失模态分流）✅
- L825 注释：分母用 `1e-20` 而**不是** `norm_eps`（1e-20 恰好同值，但注释特意声明是「matches training」，不是复用了 config）
- 专家循环（L895-900）是按 `bincount` 跳过空专家的朴素实现 —— 生产引擎必须换成 grouped GEMM

---

## 4. 两份 config 的键名映射（重实现时这就是 conversion mapping）

`config.json`（HF 格式，给 transformers 用）与 `inference_config.json`（给参考实现用）**键名不同**：

| HF `config.json` | 参考实现 | 备注 |
|---|---|---|
| `text_config.hidden_size` | `dim` | 5120 |
| `text_config.num_hidden_layers` | `n_layers` | 40 |
| `text_config.num_attention_heads` | `n_heads` | 64 |
| `text_config.num_key_value_heads` | — | =1，隐含在「shared-KV MQA」里 |
| `text_config.head_dim` | `head_dim` | 512 |
| `text_config.qk_rope_head_dim` | `rope_head_dim` | **改名** |
| `text_config.sliding_window` | `window_size` | **改名** |
| `text_config.scoring_func` | `score_func` | `sqrtsoftplus` **改名** |
| `text_config.routed_scaling_factor` | `route_scale` | **改名** |
| `text_config.moe_intermediate_size` | `moe_inter_dim` | **改名** |
| `text_config.num_experts_per_tok` | `n_activated_experts` | **改名** |
| `text_config.rms_norm_eps` | `norm_eps` | 1e-20 |
| `text_config.kv_source_layer_ids` | `kv_source_layers` | **改名** |
| `text_config.index_source_layer_ids` | `index_source_layers` | **改名** |
| `text_config.candidate_source_layer_id` | `candidate_source_layer` | **改名** |
| `text_config.candidate_topk_blocks` / `_block_size` | 同名 | 2048 / 8 |
| `text_config.compress_ratios` | `compress_ratios` | 同名，43 项 |
| `text_config.compress_rope_theta` | `compress_rope_theta` | 160000 |
| `text_config.rope_scaling.original_max_position_embeddings` | `original_seq_len` | **展平+改名** |
| `text_config.rope_scaling.factor` | `rope_factor` | **展平+改名** |
| `text_config.rope_scaling.beta_fast/slow` | `beta_fast/beta_slow` | **展平** |
| `text_config.max_position_embeddings` | （不读） | 1048576；参考实现用运行时 `max_seq_len`（默认 4096） |
| `text_config.num_nextn_predict_layers` | `n_mtp_layers` | **改名** |
| `text_config.dspark_num_experts_per_tok` | `dspark_n_activated_experts` | **改名** |
| `text_config.engram_pad_token_id` | `engram_pad_id` | **改名** |
| `text_config.engram_n_heads` / `_head_dim` | 同名 | 8 / 256 |
| `text_config.topk_method` | （不读） | `noaux_tc`；参考实现只认 score_func + bias |
| `vision_config.num_hidden_layers` | `vision_n_layers` | **加前缀** |
| `vision_config.hidden_size` | `vision_dim` | **加前缀** |
| `vision_config.max_image_tokens` | `vision_max_n_token` | **加前缀+改名** |
| `quantization_config.{quant_method,expert_dtype}` | `dtype="fp8"` + `expert_dtype="fp4"` | 拆成两个标量 |

**注意**：`convert.py` 只做机械改名（`self_attn`→`attn`、`mlp`→`ffn`、`weight_scale_inv`→`scale`、
`e_score_correction_bias`→`bias`，L105-126），**不做模块结构重排**。
所以 **HF 权重名 ≈ 参考实现名**，这意味着上面的表就是一次完整的「HF ↔ 引擎」名字账本。

### 权重布局总览（`index.json`，96085 张量 / 48 分片）

| 顶层前缀 | 张量数 | 说明 |
|---|---|---|
| `layers.*` | 93415 | 40 层 backbone（384 专家 × 3 矩阵 × 40 层占了绝大部分） |
| `mtp.*` | 2401 | 3 个 DSpark stage（128 专家，非 384） |
| `vision.*` | 259 | DeepSeek-ViT 32 层 |
| `aligner` | 4 | MLP projector |
| `embed` / `head` / `norm` / `image_start` / `image_end` / `image_newline` | 各 1 | 顶层 |

单层骨架（以层 0 为例，SWA-only 层）：
```
attn.{attn_sink, kv_norm, q_norm, wkv, wq_a, wq_b, wo_a, wo_b}
attn_norm
ffn.gate, ffn.gate.bias_vl, ffn.shared_experts.{w1,w2,w3}
ffn.experts.0..383.{w1,w2,w3}
ffn_norm
hc_{attn,ffn}_{fn,base,scale}          ← 6 个，43 层全有（258 = 43×6 ✅）
```

MTP（DSpark）三个 stage 的分工也从权重名看得一清二楚：

| stage | 独有模块 | 对应代码 |
|---|---|---|
| `mtp.0` | `main_proj` + `main_norm` | `DSparkBlock.__init__` 的 `stage_id == 0` 分支（L1111） |
| `mtp.1` | 无（纯 Block） | — |
| `mtp.2` | `markov_head.{embed,head}`、`confidence_head.proj`、`norm` | `stage_id == n_mtp_layers-1` 分支（L1115） |

---

## 5. 量化矩阵（三套精度并行，别混）

| 对象 | 精度 | scale | 代码位置 |
|---|---|---|---|
| Linear 权重（dense） | FP8 E4M3，32×32 block | E8M0，`scale_fmt="ue8m0"` | L226-233 |
| MoE 专家权重 | **FP4** E2M1（每字节 2 值） | E8M0，每 32 元素 | L219-224 |
| 主 KV（压缩 latent） | **FP4** E2M1 | **E4M3，每 16 通道**（NVFP4 风格，无二级全局 scale） | L760 |
| SWA KV | FP8 E4M3 | 整块 | L707 |
| indexer Q / K | FP4 E2M1 | E8M0，每 32 | L546, L552 |
| Engram 表 | FP8 E4M3（**存储态即 fp8**，查表反量化） | E8M0 | L309-310 |
| Compressor 权重 | ratio=1 → bf16；ratio>1 → **fp32**（池化在 fp32 里做） | — | L446-448 |
| `wo_a` | bf16；`convert.py` 会反量化 | einsum 实现（非 grouped GEMM） | L645-649, L786-787 |

`torch.float4_e2m1fn_x2` / `torch.float8_e8m0fnu` 这两个 dtype 是 torch ≥ 2.10 才有的 ——
**这就是 `requirements.txt` 写 `torch>=2.10.0` 的原因**。kernel 全在 `tilelang==0.1.8` 上写。

---

## 6. 上游进度追踪（当学习路径的「路标」，不是待办）

时效性说明：以下是 **2026-09-10** 的快照，PR 状态会变。查最新：
`https://github.com/vllm-project/vllm/pulls?q=is%3Apr+v4.1+deepseek`

| PR | 标题 | 作者 | 状态 |
|---|---|---|---|
| [#56214](https://github.com/vllm-project/vllm/pull/56214) | [Model] Support DeepSeek-V4.1-Flash（主 PR） | zyongye (Member) | open, needs-rebase |
| [#56228](https://github.com/vllm-project/vllm/pull/56228) | [Model] DeepSeek-V4.1-Flash **Model Definitions** | ywang96 (Member) | open, ready |
| [#56227](https://github.com/vllm-project/vllm/pull/56227) | SWA-bounded replay | ivanium (Collaborator) | open |
| #56224 / #56223 / #56222 / #56221 / #56220 / #56219 | Engram 跨 microbatch / PP 跨 stage KV+index / SP 边界 / 性能 overlap | 0z5a | 6 个 draft |
| [#56208](https://github.com/vllm-project/vllm/pull/56208) | Rust + Python 前端支持 | BugenZhao (Member) | **已 merge** |
| [#56215](https://github.com/vllm-project/vllm/pull/56215) | fused DSv4 MLA epilogue kernel | zyongye (Member) | **已 merge** |
| [#56235](https://github.com/vllm-project/vllm/pull/56235) | V4.1 工具参数 schema 约束 | abmfy (Member) | open |

**本地 vLLM main 的现状（对照用）**：

- `grep -ri v41 vllm/` → **空**
- 已有 V4 地基：`vllm/models/deepseek_v4/{nvidia,xpu,amd,common}/`，注册
  `DeepseekV4ForCausalLM` / `DeepseekV4ForConditionalGeneration` / `DSparkDraftModel`（`registry.py` L98/L371/L638）
- 命名对照（用 `grep -ril` 统计出现文件数）：

| V4.1 概念 | vLLM V4 里已有？ |
|---|---|
| `Compressor` / `Indexer` | ✅ 15 / 14 个文件 |
| `sparse_attn` / `attn_sink` | ✅ 5 / 15 个文件 |
| `hc_mult` / Sinkhorn | ✅ 10 / 3 个文件 |
| `DSpark` / MTP | ✅ 10 个文件 |
| **`Engram`** | ❌ **0 个文件** ← V4.1 的全新组件 |
| `candidate`（候选池） | 1 个文件 ← 基本是新的 |

→ **Engram 与候选池是 V4.1 相对 V4 的主要增量**，也是 PR #56219-#56224 那一串在做的。

**transformers 侧**：main 有 `models/deepseek_v4/`（PR #45643 引入，含 `DeepseekV4HCACache`/`DeepseekV4CSACache`、
`DeepseekV4GroupedLinear`、`DeepseekV4TopKRouter`/`HashRouter`、mHC 相关模块），
但**没有** `deepseek_v41`。公开 PR 检索 `is:pr deepseek v4.1` → 0 条。

---

## 7. 你能做什么实验（诚实版：先说不能做的）

### ❌ 不能做
- **真机部署**：552B FP8+FP4。A100 无 FP8/FP4 硬件（sm_80）；A800 40G×8 显存不够；1080Ti 直接出局。
- **跑官方自测** `python model.py`：需要 `torch>=2.10`（`float4_e2m1fn_x2` dtype）+ `tilelang==0.1.8`，
  且 `sparse_attn` 是 tilelang kernel 没有 torch 回退 → A100 上预期在 kernel 层失败。
- **`convert.py`**：要读 48 个分片、合计 **510.3 GB**（实测自 HF API `?blobs=true`），本地磁盘不够且 NFS 慢。

### ✅ 能做（按性价比排序）

**E1. 纯 config 推导（零依赖，立刻可做）**
把 §3.4 的 890 B/token 推导**自己重算一遍**，并回答：
1. 如果把解码器也改成 ratio=2，890 会变成多少？论文说 1/4 于 V4-Flash，那 V4-Flash 是多少？
2. SWA KV 单独占多少（按 window=128 计到每序列，而不是每 token）？
3. 为什么 indexer K 用 E8M0（每 32）而主 KV 用 E4M3（每 16）？精度差在哪？

**E2. 用纯 Python 复现哈希（零依赖）**
`inference_engram.py` 只依赖 `numpy` + 一个 tokenizer。把 `EngramLayout.from_args` 抠出来，
用 `inference_config.json` 跑一遍，验证「24 个质数之和 = 384006168」。
**这是全文最容易闭环、也最能证明你读懂了 Engram 的实验。**

**E3. 消融式阅读（改参数，看代码行为变化）**
不动 GPU，只改 `ModelArgs` 默认值 + 读代码路径：
- `compress_ratios` 全置 0 → 哪些模块不再被构造？（预期：Compressor/Indexer 全消失，退化成纯 SWA）
- `candidate_source_layer = -1` → `uses_candidates` 全 False，两级退化单级
- `hc_mult = 1` → `hc_pre/hc_post` 退化成恒等？`comb` 变 1×1？
- `window_size ≥ max_seq_len` → 环形缓冲的 `cutoff` 分支还能走到吗？

**E4.（进阶，需自己写参考 kernel）把 tilelang 换掉**
把 `sparse_attn` 用稠密 einsum 重写（按 `topk_idxs` gather KV 再 `softmax(q@k^T)`），
`linear()` 走 bf16 分支 → 让 `python model.py` 的自测在 **CPU 或 A100** 上跑通。
这是理解 `sparse_attn` 契约（尤其是 `-1` 空槽语义和 `attn_sink` 的角色）最短的路径，
也是你最熟悉的那套「docs/09 测试驱动深读」的升级版：**把 kernel 换成可读实现，验证语义等价**。

> E4 的产出价值最高：它能把一个「只能读」的模型变成「能跑通形状和前向」的沙盒。
> 注意 `-1` 空槽（L411 `get_window_topk_idxs` 的约定）+ `attn_sink`（每头一个学习偏置，FP32）
> 是两个必须正确处理的语义点。

---

## 8. 与学习计划的锚点

| 本文内容 | 接进哪块 | 怎么用 |
|---|---|---|
| §3.2 三模式静态分类 | docs/02 专题 1（调度） | 对照 vLLM 的 `AttentionMetadata`/后端选择：**层间异构**是 V4.1 给调度器出的新题 |
| §3.4 KV 890 B/token | docs/02 专题 2（KV） | 极端反例：KV 不由「层数 × head × dim」决定，而由 **source layer 集合** 决定 |
| §3.3 两级 indexer | docs/02 专题 3（注意力） | 对照 vLLM `vllm/models/deepseek_v4/nvidia/{flashmla,flashinfer_sparse}.py` 看生产实现 |
| §3.5 mHC 错位 | docs/09 测试驱动深读 | 找 `vllm/models/deepseek_v4/` 里 Sinkhorn 相关测试（3 个文件命中），读断言验契约 |
| §3.6 Engram 哈希 | 阶段三（SGLang 对照） | SGLang 侧 `deepseek_v4*.py` 存在但无 engram → 是对照矩阵里一个明确的空白格 |
| §6 上游 PR | 观察位 | #56228 落地后，把它的模块划分与 §2 的表格对一遍，看生产实现如何取舍 |

**一句话总结**：V4.1 的架构创新几乎全在「**层间共享**」这一个维度上——
共享 KV（CED）、共享 indexer K（CSA2）、共享 Top-K（Reuse）、共享候选池（层次化 indexer）、
共享残差流（mHC 的 4 份拷贝）、共享 n-gram 表（Engram）。
读懂它的关键是**别再按「每层独立」的直觉读**。

---

*证据分级：§3.1/§3.2/§3.4 的结论有**权重索引 + 代码 + 论文**三方交叉验证；
§3.3/§3.5/§3.6 为源码 + 论文两方；§6 为 2026-09-10 网页快照（时效性弱）。
本机 hf-mirror 可直连，材料已落地 `dsv41-ref/`，无需重下。*
