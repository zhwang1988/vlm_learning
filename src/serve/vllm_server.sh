#!/usr/bin/env bash
# vLLM 部署脚本（Day 29）
#
# 用法:
#   bash src/serve/vllm_server.sh Qwen/Qwen2.5-VL-3B-Instruct
#   bash src/serve/vllm_server.sh outputs/qwen25vl3b-cx-lora-v0 cx-vlm awq
#
# 参数: $1=模型路径  $2=服务名  $3=量化方式(可选: awq|gptq|fp8)

set -euo pipefail

MODEL="${1:-Qwen/Qwen2.5-VL-3B-Instruct}"
SERVED_NAME="${2:-cx-vlm}"
QUANT="${3:-}"

# 显存不够时调小这些
GPU_UTIL="${GPU_UTIL:-0.88}"
MAX_LEN="${MAX_LEN:-8192}"        # ⚠️ 不要设成模型的 128k 上限！会预留巨量 KV cache
MAX_SEQS="${MAX_SEQS:-8}"
LIMIT_MM="${LIMIT_MM:-3}"         # ⚠️ 必须设，否则一个恶意请求塞 100 张图打爆显存

QUANT_ARG=""
if [ -n "$QUANT" ]; then
  QUANT_ARG="--quantization $QUANT"
  echo "量化方式: $QUANT"
  echo "  ⚠️ VLM 的视觉塔对量化比 LLM 敏感得多！"
  echo "     表现：OCR 数字读错、颜色判断偏、小瑕疵看不见"
  echo "     建议用 modules_to_not_convert 排除 visual（见 src/serve/quantize.py）"
fi

echo "=============================================="
echo "vLLM 启动"
echo "  模型:     $MODEL"
echo "  服务名:   $SERVED_NAME"
echo "  GPU 利用率: $GPU_UTIL   最大长度: $MAX_LEN"
echo "  多模态上限: $LIMIT_MM 张/请求"
echo "=============================================="

python -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" \
  --served-model-name "$SERVED_NAME" \
  --host 0.0.0.0 \
  --port "${PORT:-8000}" \
  --tensor-parallel-size "${TP:-1}" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --max-model-len "$MAX_LEN" \
  --max-num-seqs "$MAX_SEQS" \
  --limit-mm-per-prompt "image=$LIMIT_MM" \
  --enable-prefix-caching \
  --trust-remote-code \
  $QUANT_ARG

# 关于 --enable-prefix-caching:
#   system prompt 固定时收益很大，能省掉重复的 prefill。
#   客服场景的 system prompt 是不变的，所以必开。
#
# 关于 --gpu-memory-utilization:
#   0.85-0.92 之间。太高会 OOM，太低浪费显存。
#   如果同时还要跑其他进程（比如 embedding 服务），降到 0.6-0.7。
