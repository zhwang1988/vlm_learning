#!/usr/bin/env bash
# =============================================================================
# cloud_bootstrap.sh —— 云 GPU 实例一键初始化
#
# 用法（在租好的实例上，JupyterLab Terminal 或 SSH）：
#   bash scripts/cloud_bootstrap.sh                 # 完整流程
#   bash scripts/cloud_bootstrap.sh --stage core    # 只装第 1 层（Day 1-10）
#   bash scripts/cloud_bootstrap.sh --stage train   # 装到训练层
#   bash scripts/cloud_bootstrap.sh --with-vllm     # 额外建独立的 vllm 环境
#   bash scripts/cloud_bootstrap.sh --skip-model    # 不下载模型（省时间/traffic）
#   bash scripts/cloud_bootstrap.sh --check         # 只体检，什么都不装
#
# 设计原则：
#   1. 幂等 —— 重复跑不会出错，已完成的步骤会跳过
#   2. 每步都检查，失败给出「怎么修」而不是一堆 traceback
#   3. 不污染系统 Python；不 pip install -g
#   4. 默认配好国内镜像（hf-mirror / 清华 pypi）—— 不配会被下载速度折磨死
# =============================================================================

set -o pipefail

# ---- 颜色 ----------------------------------------------------------------
if [ -t 1 ]; then
  C_R="\033[0m"; C_G="\033[32m"; C_Y="\033[33m"; C_RD="\033[31m"
  C_B="\033[36m"; C_BOLD="\033[1m"
else
  C_R=""; C_G=""; C_Y=""; C_RD=""; C_B=""; C_BOLD=""
fi
ok()    { printf "${C_G}[✓]${C_R} %s\n" "$*"; }
info()  { printf "${C_B}[·]${C_R} %s\n" "$*"; }
warn()  { printf "${C_Y}[!]${C_R} %s\n" "$*"; }
err()   { printf "${C_RD}[✗]${C_R} %s\n" "$*"; }
step()  { printf "\n${C_BOLD}==> %s${C_R}\n" "$*"; }
die()   { err "$*"; echo; echo "   卡住了？把上面这行连同报错一起发我，或者看 docs/13-hardware-and-cost.md 第六节「七个必踩的坑」。"; exit 1; }

# ---- 参数 ----------------------------------------------------------------
STAGE="train"        # core | train | all
SKIP_MODEL=0
WITH_VLLM=0
CHECK_ONLY=0
SKIP_CLONE=0
MODEL_ID="Qwen/Qwen2.5-VL-3B-Instruct"

while [ $# -gt 0 ]; do
  case "$1" in
    --stage)      STAGE="$2"; shift 2 ;;
    --skip-model) SKIP_MODEL=1; shift ;;
    --with-vllm)  WITH_VLLM=1; shift ;;
    --check)      CHECK_ONLY=1; shift ;;
    --skip-clone) SKIP_CLONE=1; shift ;;
    --model)      MODEL_ID="$2"; shift 2 ;;
    -h|--help)    sed -n '2,20p' "$0"; exit 0 ;;
    *) die "未知参数: $1（用 --help 看用法）" ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# 数据盘优先：AutoDL 上是 /root/autodl-tmp，其他平台用 $HOME/data
if [ -d "/root/autodl-tmp" ]; then
  DATA_ROOT="/root/autodl-tmp"
elif [ -n "$WORKSPACE_DATA" ]; then
  DATA_ROOT="$WORKSPACE_DATA"
else
  DATA_ROOT="$HOME/data"
fi
mkdir -p "$DATA_ROOT" 2>/dev/null || true

VENV="$DATA_ROOT/venvs/mmlab"

printf "${C_BOLD}"
cat <<'BANNER'
  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _
  multimodal-lab :: cloud bootstrap
  多模态客服模型 —— 云 GPU 一键初始化
  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _  _
BANNER
printf "${C_R}"

# =============================================================================
# 0. 机器信息 —— 先确认你没租错卡
# =============================================================================
step "0/6 机器信息"

info "主机      : $(hostname)"
info "系统      : $(uname -s -r -m)"
info "工作目录  : $REPO_ROOT"
info "数据目录  : $DATA_ROOT"

DISK_FREE_GB=$(df -BG "$DATA_ROOT" 2>/dev/null | awk 'NR==2 {gsub("G","",$4); print $4}')
[ -n "$DISK_FREE_GB" ] && info "数据目录剩余空间 : ${DISK_FREE_GB} GB"
if [ -n "$DISK_FREE_GB" ] && [ "$DISK_FREE_GB" -lt 40 ] 2>/dev/null; then
  warn "剩余空间不足 40 GB。模型 7 GB + 数据集 + checkpoint 很容易吃掉几十 GB。"
  warn "建议换更大的数据盘，或先跑 --skip-model。"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
  GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -1)
  ok "GPU : ${GPU_NAME}  (${GPU_MEM} MiB 显存, driver ${DRIVER})"

  if [ "${GPU_MEM:-0}" -lt 23000 ] 2>/dev/null; then
    warn "显存只有 ${GPU_MEM} MiB。"
    warn "  → 3B QLoRA 可以跑，但 max_length 要压到 1024、batch 保持 1"
    warn "  → 7B QLoRA 会在这一步 OOM，先专注 3B"
    if [ "${GPU_MEM:-0}" -lt 15000 ] 2>/dev/null; then
      warn "  → 这块卡偏小，建议升级到 24 GB 以上（见 docs/13-hardware-and-cost.md）"
    fi
  fi

  if [ "${GPU_MEM:-0}" -ge 70000 ] 2>/dev/null; then
    ok "显存充足（>=70 GB）→ 可以跑 7B LoRA bf16 甚至 3B 全参"
  fi
else
  warn "没有检测到 nvidia-smi。"
  warn "这可能意味着：① 你开的是「无卡模式」——正常，装环境不需要 GPU；"
  warn "              ② 或者租错了机器，没有 GPU。"
fi

# CUDA
if command -v nvcc >/dev/null 2>&1; then
  info "CUDA compiler : $(nvcc --version | grep -o 'release [0-9.]*' | head -1)"
fi
if [ -n "${CUDA_HOME:-}" ]; then info "CUDA_HOME : $CUDA_HOME"; fi

if [ "$CHECK_ONLY" = "1" ]; then
  step "只做体检（--check），退出"
  PY="$VENV/bin/python"
  if [ -x "$PY" ]; then
    "$PY" "$REPO_ROOT/scripts/env_check.py" || true
  else
    warn "虚拟环境还不存在（$VENV），先跑一次完整流程。"
  fi
  exit 0
fi

# =============================================================================
# 1. Python 环境
# =============================================================================
step "1/6 Python 虚拟环境"

if [ -x "$VENV/bin/python" ]; then
  ok "已存在，跳过：$VENV ($("$VENV/bin/python" -V 2>&1))"
else
  if command -v python3 >/dev/null 2>&1; then
    BASE_PY=python3
  elif command -v python >/dev/null 2>&1; then
    BASE_PY=python
  else
    die "找不到 python3。云镜像一般自带，如果没有：apt-get update && apt-get install -y python3 python3-venv python3-pip"
  fi
  info "基础解释器 : $BASE_PY ($($BASE_PY -V 2>&1))"

  # 优先用 conda（云镜像通常预装），因为它对 CUDA 更友好
  if command -v conda >/dev/null 2>&1; then
    info "检测到 conda，用 conda 建环境（对 CUDA 兼容性更好）"
    CONDA_PREFIX_PATH="$DATA_ROOT/venvs/mmlab"
    if conda env list 2>/dev/null | grep -q "mmlab"; then
      ok "conda 环境 mmlab 已存在"
    else
      conda create -y -p "$CONDA_PREFIX_PATH" python=3.11 >/dev/null 2>&1 \
        || die "conda create 失败。改用 venv：/usr/bin/python3 -m venv $VENV"
      ok "conda 环境建成：$CONDA_PREFIX_PATH"
    fi
    VENV="$CONDA_PREFIX_PATH"
  else
    info "没有 conda，用 venv"
    $BASE_PY -m venv "$VENV" 2>/dev/null \
      || die "venv 创建失败。试：apt-get install -y python3-venv"
    ok "venv 建成：$VENV"
  fi
fi

PY="$VENV/bin/python"
PIP="$VENV/bin/pip"
[ -x "$PIP" ] || PIP="$PY -m pip"

"$PY" -m pip install --upgrade pip setuptools wheel -q 2>&1 | tail -2 || true
ok "pip 已升级"

# =============================================================================
# 2. 国内镜像 —— 不配这一步你会怀疑人生
# =============================================================================
step "2/6 配置国内镜像"

# PyPI
"$PIP" config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple >/dev/null 2>&1 \
  && ok "PyPI  -> 清华镜像" || warn "PyPI 镜像配置失败（不影响，但下载会慢）"

# HuggingFace
if [ -n "${HF_ENDPOINT:-}" ]; then
  ok "HF_ENDPOINT 已设置：$HF_ENDPOINT"
else
  export HF_ENDPOINT="https://hf-mirror.com"
  ok "HF_ENDPOINT -> $HF_ENDPOINT"
fi

# 写进 shell rc，让新开的终端也生效
for RC in "$HOME/.bashrc" "$HOME/.zshrc"; do
  if [ -f "$RC" ] || [ "$RC" = "$HOME/.bashrc" ]; then
    if ! grep -q "HF_ENDPOINT" "$RC" 2>/dev/null; then
      {
        echo ''
        echo '# --- multimodal-lab ---'
        echo 'export HF_ENDPOINT=https://hf-mirror.com'
        echo "export HF_HOME=$DATA_ROOT/hf-cache"
        echo "export MMLAB_DATA_ROOT=$DATA_ROOT"
      } >> "$RC"
    fi
  fi
done
export HF_HOME="$DATA_ROOT/hf-cache"
mkdir -p "$HF_HOME"
ok "HF_HOME  -> $HF_HOME  （模型缓存放数据盘，实例删除也不丢）"

# 注意：HF_ENDPOINT 只对当前 shell 生效。脚本里后续步骤都 export 过了，
# 但你自己新开终端时如果没 source rc，记得手动 export。

# =============================================================================
# 3. 安装依赖
# =============================================================================
step "3/6 安装依赖 (stage=$STAGE)"

req_file=""
case "$STAGE" in
  core)  req_file="requirements-core.txt" ;;
  train) req_file="requirements-core.txt requirements-train.txt" ;;
  all)   req_file="requirements-core.txt requirements-train.txt requirements-serve.txt" ;;
  *)     die "--stage 只能是 core / train / all" ;;
esac

info "将安装：$req_file"
warn "这一步大概 3–8 分钟，取决于网络。torch 如果已预装会跳过（省 2 GB 下载）。"

for f in $req_file; do
  p="$REPO_ROOT/$f"
  [ -f "$p" ] || { warn "找不到 $f，跳过"; continue; }
  info "-> $f"
  # 先看 torch 在不在。云镜像一般预装了匹配 CUDA 的 torch，
  # 千万不要让 pip 把它换掉（换了就没 GPU 了）。
  if [ "$f" = "requirements-train.txt" ] && "$PY" -c "import torch" 2>/dev/null; then
    torch_ver=$("$PY" -c "import torch;print(torch.__version__)" 2>/dev/null)
    cuda_ok=$("$PY" -c "import torch;print(torch.cuda.is_available())" 2>/dev/null)
    ok "检测到已装 torch $torch_ver (cuda_available=$cuda_ok)，保留不动"
    grep -v -E '^\s*torch' "$p" > /tmp/_req_notorch.txt
    "$PIP" install -q -r /tmp/_req_notorch.txt 2>&1 | tail -3 || die "依赖安装失败（见上面报错）"
  else
    "$PIP" install -q -r "$p" 2>&1 | tail -3 || die "依赖安装失败（见上面报错）"
  fi
  ok "$f 装完"
done

# =============================================================================
# 4. 验证 PyTorch + CUDA
# =============================================================================
step "4/6 验证 PyTorch + CUDA"

"$PY" - <<'PYEOF' || die "PyTorch 自检失败。最常见原因：torch 是 CPU 版。解决：选 PyTorch 镜像重建实例，或 pip install torch --index-url https://download.pytorch.org/whl/cu121"
import sys
try:
    import torch
except ImportError:
    print("  torch 没装上")
    sys.exit(1)

print(f"  torch      : {torch.__version__}")
print(f"  cuda(编译) : {torch.version.cuda}")
print(f"  cuda(可用) : {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  设备       : {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability(0)
    print(f"  算力       : sm_{cap[0]}{cap[1]}")
    print(f"  显存       : {torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB")
    if cap[0] >= 8:
        print("  -> 支持 bf16，训练配置里 bf16=True 可用")
    else:
        print("  -> 不支持 bf16（T4/V100 等），必须用 fp16=True, bf16=False")
    if cap[0] >= 8 and torch.cuda.get_device_properties(0).total_memory > 70*1024**3:
        bf16_ok = True
    else:
        bf16_ok = cap[0] >= 8
    if not bf16_ok:
        print("  !! 注意：请在你的 config 里设置 bf16: false, fp16: true")

    # 真实跑一次矩阵乘法，确认不是「看得见卡但用不了」
    a = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    b = torch.randn(2048, 2048, device="cuda", dtype=torch.float16)
    for _ in range(3):
        c = a @ b
    torch.cuda.synchronize()
    print("  矩阵乘法   : 通过（卡是真能用的）")
else:
    print("  !! CUDA 不可用。如果你开的是无卡模式，这是正常的；")
    print("     否则请检查：是否选了 GPU 实例、torch 是否为 +cu 版本。")
    sys.exit(2)
PYEOF

[ $? -eq 2 ] && warn "CUDA 不可用 —— 无卡模式下属正常，可以继续（装代码/数据）"

# =============================================================================
# 5. 依赖完整性
# =============================================================================
step "5/6 关键依赖检查"

"$PY" - <<'PYEOF' || true
mods = [
    ("transformers", "AutoProcessor"),
    ("peft",         "LoraConfig"),
    ("accelerate",   "Accelerator"),
    ("PIL",          "Image"),
    ("numpy",        None),
    ("yaml",         None),
    ("datasets",     None),
]
missing = []
for m, attr in mods:
    try:
        mod = __import__(m)
        if attr and not hasattr(mod, attr):
            print(f"  [~] {m}: 装了但缺少 {attr}（版本可能太旧）")
            missing.append(m)
        else:
            v = getattr(mod, "__version__", "?")
            print(f"  [ok] {m:14s} {v}")
    except ImportError:
        print(f"  [!!] {m:14s} 缺失")
        missing.append(m)

try:
    import transformers
    from packaging.version import Version
    if Version(transformers.__version__) < Version("4.49.0"):
        print("  [!!] transformers < 4.49.0 —— Qwen2.5-VL 需要 >=4.49.0")
        print("       修: pip install -U 'transformers>=4.49.0'")
        missing.append("transformers-version")
except Exception:
    pass

if missing:
    print(f"\n  需要补装: {' '.join(x for x in missing if '-' not in x)}")
else:
    print("\n  依赖齐全")
PYEOF

# =============================================================================
# 6. 下载模型 + 冒烟测试
# =============================================================================
step "6/6 模型与冒烟测试"

if [ "$SKIP_MODEL" = "1" ]; then
  warn "--skip-model 已指定，跳过下载与冒烟测试"
else
  # 把脚本里的 --model key 翻译成 download_model.py 的 key
  case "$MODEL_ID" in
    *7B*|*7b*)     MODEL_KEY="7b-instruct" ;;
    *3B*|*3b*|*Instruct*) MODEL_KEY="3b-instruct" ;;
    *)             MODEL_KEY="3b-instruct" ;;
  esac

  info "模型 : $MODEL_ID  (key=$MODEL_KEY)"
  info "脚本 : scripts/download_model.py  ——  ModelScope 优先，失败自动切 HF 镜像"
  warn "下载 ~7 GB，一般 2–5 分钟（ModelScope），最坏 10–15 分钟（HF 镜像）。"

  # 用专门的下载脚本：自动选后端、显示进度、校验完整性、断点续传
  if ! "$PY" scripts/download_model.py --model "$MODEL_KEY" --root "${MODEL_ROOT:-/root/autodl-tmp/models}"; then
    die "模型下载失败。常见原因：网络/磁盘空间不足。可重跑本脚本（断点续传），或换 backend。"
  fi

  info "加载模型做一次真实推理（第一次会慢，要初始化 CUDA）..."

  # 先问 download_model.py 实际的缓存路径 —— 因为可能走 ModelScope 落到 modelscope/ 下
  MODEL_PATH=$("$PY" scripts/download_model.py --status --root "${MODEL_ROOT:-/root/autodl-tmp/models}" 2>/dev/null \
    | awk -v k="$MODEL_KEY" '$1==k && $2!="(未下载)" {print $2}' | head -1)
  if [ -z "$MODEL_PATH" ]; then
    die "找不到 $MODEL_KEY 的缓存路径。先看 download_model.py --status"
  fi
  info "缓存路径 : $MODEL_PATH"

  "$PY" - "$MODEL_PATH" <<'PYEOF' || die "冒烟测试失败。先看报错关键词：OOM=显存不够；CUDA=torch/driver 版本；shape=transformers 版本太旧"
import sys, os, time
mid = sys.argv[1]

import torch
from PIL import Image, ImageDraw
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# 造一张有明确内容的测试图 —— 不要用随机噪声，
# 那样模型答错了你分不清是"模型不行"还是"图本来就没内容"
img = Image.new("RGB", (448, 448), (245, 245, 245))
d = ImageDraw.Draw(img)
d.ellipse([70, 120, 200, 250], fill=(230, 140, 60))      # 橙色圆
d.rectangle([250, 150, 380, 300], fill=(60, 110, 200))    # 蓝色方块
d.text((80, 340), "MMLAB SMOKE TEST", fill=(20, 20, 20))

img_path = "/tmp/_mmlab_smoke.png"
img.save(img_path)

t0 = time.time()
proc = AutoProcessor.from_pretrained(mid)
model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
    mid, torch_dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
    device_map="auto",
)
model.eval()
print(f"  加载耗时 {time.time()-t0:.1f}s")

msgs = [{"role": "user", "content": [
    {"type": "image", "image": img_path},
    {"type": "text", "text": "这张图里有哪些形状？分别是什么颜色？用一句话回答。"},
]}]
text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
inputs = proc(text=[text], images=[img], return_tensors="pt").to(model.device)

with torch.no_grad():
    out = model.generate(**inputs, max_new_tokens=80, do_sample=False)

trimmed = out[:, inputs.input_ids.shape[1]:]
answer = proc.batch_decode(trimmed, skip_special_tokens=True)[0].strip()

n_vis = int((inputs.input_ids == model.config.image_token_id).sum())
print(f"  视觉 token : {n_vis}")
print(f"  回答       : {answer}")

peak = torch.cuda.max_memory_allocated() / 1024**3 if torch.cuda.is_available() else 0
total = torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
print(f"  峰值显存   : {peak:.1f} GB / {total:.1f} GB")

answer_l = answer.lower()
hit = any(k in answer_l for k in ["圆", "circle", "橙", "橘", "orange"]) and \
      any(k in answer_l for k in ["方", "square", "矩形", "rect", "蓝", "blue"])

print()
if hit:
    print("  [✓] 冒烟测试通过 —— 模型确实看得见图，不是瞎答")
else:
    print("  [!] 回答里没同时提到橙色圆形和蓝色方形。")
    print("      可能只是表述差异，但请人工确认一眼上面的回答。")
    print("      如果答得完全不着边 -> 检查 bitsandbytes / torch 版本 mismatch")

free = total - peak
print(f"  [i] 训练可用显存 ≈ {free:.1f} GB")
if free < 6:
    print("      -> 只够 3B QLoRA + max_length 1024")
elif free < 14:
    print("      -> 够 3B QLoRA (seq 2048, bs 2)，7B QLoRA 勉强")
else:
    print("      -> 够 3B LoRA bf16 或 7B QLoRA，配置可参考 configs/sft_lora_3b.yaml")
PYEOF
fi

# =============================================================================
# 可选：独立的 vLLM 环境
# =============================================================================
if [ "$WITH_VLLM" = "1" ]; then
  step "附加：独立 vLLM 环境"
  VLLM_VENV="$DATA_ROOT/venvs/vllm"
  warn "vLLM 会自带一套 torch。所以它必须独立建环境 ——"
  warn "在训练环境里装 vllm 会换掉 torch，然后你就 CUDA mismatch 了（第六节坑 2）。"
  if [ -x "$VLLM_VENV/bin/python" ]; then
    ok "已存在：$VLLM_VENV"
  else
    python3 -m venv "$VLLM_VENV" || die "vllm venv 创建失败"
    "$VLLM_VENV/bin/pip" install -q --upgrade pip
    info "安装 vllm（约 2–4 分钟）..."
    "$VLLM_VENV/bin/pip" install -q vllm 2>&1 | tail -3 \
      && ok "vllm 装好：$VLLM_VENV" \
      || warn "vllm 安装失败（vLLM 版本对 CUDA 很敏感）。可先跳过，Day 25 再处理。"
  fi
fi

# =============================================================================
# 完成
# =============================================================================
step "完成"

cat <<EOF

  ${C_G}一切就绪。接下来：${C_R}

  # 激活环境（每次新开终端都要先跑这行）
  source $VENV/bin/activate

  # Day 4 的验收测试：算出的视觉 token 数 == 官方的 <|image_pad|> 数量
  python -m src.minivlm.processor --check

  # 从零搭的 VLM 长什么样（打印各层形状）
  python -m src.minivlm.vision
  python -m src.minivlm.connector

  # 环境体检（随时可跑）
  python scripts/env_check.py

  # 显存估算：换模型/换方法时先算一下再下单
  python scripts/estimate_vram.py --model 3b --method qlora

  ${C_Y}花钱提醒：${C_R}
    1. 写完代码先关机，需要 GPU 了再开 —— 无卡模式 ¥0.1/h，GPU 模式 ¥1.88/h
    2. 训练命令后面加 && /usr/bin/shutdown 可以跑完自动关机
    3. checkpoint 记得拉回本地：bash scripts/sync_down.sh

  数据目录: $DATA_ROOT
  环境目录: $VENV
  模型缓存: $HF_HOME

EOF

ok "bootstrap 结束"
