# 06 十月 issue 候选池（P4 前置，2026-09-08 建池）

> **来源纪律**：只收录实操中**真实遇到**的报错 / 行为差异 / 源码疑问（阶段一/二期间），不编造。
> 处理节奏：9/30 前只"观察登记"不投入；10 月按下方模板逐条升级为可复现条目 → 复现 → 定位 → 提 PR。

## 条目模板（每条照填）

```
- 标题：（一句话）
- 引擎/版本：vLLM v0.28.1rc0-509 / SGLang main 7d2d6624b1
- 现象：（日志/输出摘录）
- 复现步骤：（模型 / 参数 / 命令）
- 初步定位：（源码位置 file:line）
- 状态：观察中 / 可复现 / 已定位 / 已提 PR
```

## 观察入口（遇到就往这里记）

1. `test_vllms/` 已有两份离线日志（0.8B / 27B）
2. serve 模式日志（`logs/`，跑 `start_vllm_*` 之后）
3. benchmark 输出（`logs/`，跑 `bench_*` 之后）
4. 走读笔记里的「存疑」附录（docs/01、docs/02）

## 预登记（弱信号，待升级）

- **[观察中]** triton 弃用警告 `tl.make_block_ptr is deprecated. Use TensorDescriptor...`（0.8B/27B 日志均出现）——是 triton 上游弃用，vLLM 是否已跟进？后续可查 `vllm/` 内 make_block_ptr 使用点。
- **[观察中]** `Reducing Torch threads from 24 to 1 for serving`（信息级）——单实例正常；多实例同机时 OMP 线程行为可留意，暂无价值。
- **[观察中]** 27B 日志 `Maximum concurrency for 262,144 tokens per request: 1.14x` 的口径（是否含混合批）——与 docs/02 显存账核对后如有出入可深挖。

> 以上均为弱信号，仅"登记不投入"；10 月按模板升级。