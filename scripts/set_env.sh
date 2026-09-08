#!/usr/bin/env bash
# ============================================================
# 个人环境配置（set_env.sh）
# - 集中配置：模型路径 / 服务器 IP / 常用端口 / 通用环境变量
# - 其他脚本（start_*.sh / bench / download / test_chat）会自动加载本文件
# - 所有变量用 ${VAR:-默认} 语法：显式传入的环境变量优先，不会被本文件覆盖
# - ⚠️ 本文件含内网 IP 与服务器路径，属隐私信息：勿提交 git / 勿外传
#
# 用法：
#   手动加载：source scripts/set_env.sh
#   自动加载：直接运行 scripts/ 下其它脚本即可（内部自动 source 本文件）
# ============================================================

# ---------- 模型目录（主力实验机本地盘） ----------
export MODEL_DIR=${MODEL_DIR:-/data/hf_models}

# ---------- 常用模型路径 ----------
export MODEL_08B=${MODEL_08B:-$MODEL_DIR/Qwen3.5-0.8B}          # 解剖主模型（混合注意力，W1/W4）
export MODEL_05B=${MODEL_05B:-$MODEL_DIR/Qwen2.5-0.5B-Instruct}  # 标准注意力对照
export MODEL_27B=${MODEL_27B:-$MODEL_DIR/Qwen3.8-27B}            # 实战模型（W2 benchmark）
export MODEL_PATH=${MODEL_PATH:-$MODEL_08B}                      # 通用默认模型（显式传入优先）

# ---------- 服务器（内网，勿外传） ----------
export MAIN_SERVER_IP=${MAIN_SERVER_IP:-192.168.49.147}   # 主力实验机：A100 80G，卡可用至 09-30
export LOCAL_SERVER_IP=${LOCAL_SERVER_IP:-192.168.9.145}  # 本地工作机（DSH 所在，NFS 共享盘）

# ---------- 常用端口 ----------
export PORT_08B=${PORT_08B:-8001}        # vLLM 0.8B
export PORT_27B=${PORT_27B:-8002}        # vLLM 27B
export PORT_05B=${PORT_05B:-8003}        # vLLM 0.5B 对照
export PORT_SGLANG=${PORT_SGLANG:-30000} # SGLang 0.8B（W4）

# ---------- 通用 ----------
export HF_ENDPOINT=${HF_ENDPOINT:-https://hf-mirror.com}  # huggingface.co 不通，一律走镜像
export CONDA_ENV=${CONDA_ENV:-py312}                       # 主力实验机 conda 环境名（按实际改）

echo "[set_env] 已加载个人环境：MODEL_DIR=$MODEL_DIR | 主模型 MODEL_PATH=$MODEL_PATH | 主力机 $MAIN_SERVER_IP"