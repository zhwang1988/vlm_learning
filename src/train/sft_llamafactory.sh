#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# LLaMA-Factory 路线的 SFT 启动脚本（Day 15 备选路线）
#
#   bash src/train/sft_llamafactory.sh                # 用默认配置开训
#   bash src/train/sft_llamafactory.sh --dry-run      # 只打印命令，不执行
#
# 什么时候用这条路线：
#   · 你不想读原生 transformers 的 Trainer 代码
#   · 想用现成的数据格式校验 / 现成的 webui
#   · 需要快速试多个超参组合
#
# 什么时候用原生路线（src/train/sft_peft.py）：
#   · 你想真的看懂训练循环（这个项目的初衷）
#   · 需要自定义 loss / collator / 多模态特殊处理
# ---------------------------------------------------------------------------
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-3B-Instruct}"
DATASET_DIR="${DATASET_DIR:-data/processed}"
DATASET_INFO="${DATASET_INFO:-configs/llamafactory_dataset_info.json}"
OUTPUT_DIR="${OUTPUT_DIR:-outputs/qwen25vl3b-cx-lf-v0}"
CONFIG="${CONFIG:-configs/llamafactory_sft_3b.yaml}"
DRY_RUN=0

for arg in "$@"; do
  case "$arg" in
    --dry-run) DRY_RUN=1 ;;
    *) echo "未知参数: $arg" >&2; exit 2 ;;
  esac
done

PY="${PY:-python}"

info() { printf '  %s\n' "$*"; }
die()  { printf '\n✗ %s\n' "$*" >&2; exit 1; }

echo "======================================================================"
echo "  LLaMA-Factory · SFT"
echo "======================================================================"
info "模型      : $MODEL"
info "数据集目录: $DATASET_DIR"
info "配置      : $CONFIG"
info "输出      : $OUTPUT_DIR"

# ---- 1. 依赖检查 ----------------------------------------------------------
if [ "$DRY_RUN" = "0" ]; then
  "$PY" - <<'PYEOF' || die "依赖不全。先跑：pip install -r requirements-train.txt"
import importlib, sys
need = {"llamafactory": "llamafactory-cli", "torch": "pip install torch",
        "transformers": "pip install transformers", "peft": "pip install peft",
        "trl": "pip install trl", "datasets": "pip install datasets"}
missing = [k for k in need if importlib.util.find_spec(k) is None]
if missing:
    print("缺这些包:", ", ".join(missing)); sys.exit(1)
import torch
print(f"  torch {torch.__version__}  cuda={torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("  ⚠️ 没有 GPU —— 训练跑不起来，先租机器（见 docs/13-hardware-and-cost.md）")
    sys.exit(1)
p = torch.cuda.get_device_properties(0)
print(f"  gpu   {p.name}  {p.total_memory/1024**3:.0f} GB")
PYEOF
fi

# ---- 2. 数据检查 ----------------------------------------------------------
[ -f "$DATASET_DIR/sft_train.jsonl" ] || \
  die "找不到 $DATASET_DIR/sft_train.jsonl —— 先跑 Day 11 的数据打包"
[ -f "$DATASET_INFO" ] || \
  die "找不到 $DATASET_INFO —— LLaMA-Factory 需要它来知道字段映射"

if [ "$DRY_RUN" = "0" ]; then
  info "训练样本: $("$PY" -c "print(sum(1 for _ in open('$DATASET_DIR/sft_train.jsonl')))") 条"
fi

# ---- 3. 启动 --------------------------------------------------------------
CMD=(llamafactory-cli train "$CONFIG")

if [ "$DRY_RUN" = "1" ]; then
  echo
  echo "  [dry-run] 将要执行："
  printf '    %q ' "${CMD[@]}"; echo
  echo
  echo "  对应的配置文件内容："
  sed 's/^/    /' "$CONFIG" 2>/dev/null || echo "    （$CONFIG 不存在）"
  exit 0
fi

mkdir -p "$OUTPUT_DIR"
LOG="$OUTPUT_DIR/train.log"

echo
info "启动训练，日志 → $LOG"
info "（另开一个终端 watch -n 5 nvidia-smi 盯显存）"
echo

# nohup 让 SSH 断了训练还在
nohup "${CMD[@]}" > "$LOG" 2>&1 &
PID=$!
info "训练进程 PID = $PID"
echo
info "跟踪日志：  tail -f $LOG"
info "停止训练：  kill $PID"
echo
info "关注这三条：loss 缓降 / grad_norm 平稳 / lr 先升后降（warmup+cosine）"
info "训练结束后接到 Day 16：python -m src.train.monitor $OUTPUT_DIR"
echo "======================================================================"
