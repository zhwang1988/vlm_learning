#!/usr/bin/env bash
# =============================================================================
# demo_up.sh —— 演示会话一键启动
#
# 解决一个具体的成本问题：
#
#   W6–W8 你需要一个「随时在线」的模型服务。
#   如果租着 4090 让它 7x24 空转，8 周要烧 ~¥2500 —— 比所有训练加起来贵 10 倍。
#
#   正确姿势：演示时才开，演示完就关。
#   这个脚本做的就是「开 -> 起服务 -> 自检 -> 告诉你一共花了多少」。
#
# 用法：
#   bash scripts/demo_up.sh                    # 起 vLLM + 应用层
#   bash scripts/demo_up.sh --no-vllm           # 只起应用层（模型走远端 API）
#   bash scripts/demo_up.sh --down              # 关掉一切（省钱）
#   bash scripts/demo_up.sh --cost              # 看看这次会话烧了多少钱
# =============================================================================

set -o pipefail

C_G="\033[32m"; C_Y="\033[33m"; C_RD="\033[31m"; C_B="\033[36m"; C_R="\033[0m"; C_BOLD="\033[1m"
ok()   { printf "${C_G}[✓]${C_R} %s\n" "$*"; }
info() { printf "${C_B}[·]${C_R} %s\n" "$*"; }
warn() { printf "${C_Y}[!]${C_R} %s\n" "$*"; }
err()  { printf "${C_RD}[✗]${C_R} %s\n" "$*"; }
step() { printf "\n${C_BOLD}==> %s${C_R}\n" "$*"; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT" || exit 1

# 计费参数（按需改）。默认按 4090 算。
RATE_PER_HOUR="${GPU_RATE:-1.88}"
START_MARK=".demo_session_start"
PID_DIR=".demo_pids"

MODEL_PATH="${SERVED_MODEL_PATH:-outputs/qwen25vl3b-cx-lora-v0/merged}"
VLLM_PORT="${VLLM_PORT:-8000}"
APP_PORT="${APP_PORT:-8080}"
NO_VLLM=0
DO_DOWN=0
DO_COST=0

while [ $# -gt 0 ]; do
  case "$1" in
    --no-vllm) NO_VLLM=1; shift ;;
    --down)    DO_DOWN=1; shift ;;
    --cost)    DO_COST=1; shift ;;
    --model)   MODEL_PATH="$2"; shift 2 ;;
    --port)    APP_PORT="$2"; shift 2 ;;
    -h|--help) sed -n '2,20p' "$0"; exit 0 ;;
    *) err "未知参数: $1"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# 成本
# ---------------------------------------------------------------------------
if [ "$DO_COST" = "1" ] || [ "$DO_DOWN" = "1" ]; then
  if [ -f "$START_MARK" ]; then
    start_epoch=$(cat "$START_MARK")
    now_epoch=$(date +%s)
    elapsed=$(( now_epoch - start_epoch ))
    hours=$(echo "scale=4; $elapsed / 3600" | bc 2>/dev/null || echo "0")
    cost=$(echo "scale=2; $hours * $RATE_PER_HOUR" | bc 2>/dev/null || echo "?")
    mins=$(( elapsed / 60 ))
    step "本次会话"
    info "已运行 : ${mins} 分钟"
    info "单价   : ¥${RATE_PER_HOUR}/小时"
    info "已花费 : ¥${cost}"
    echo
    if [ "$elapsed" -gt 7200 ]; then
      warn "超过 2 小时了。如果是忘了关，现在关还来得及 ——"
      warn "跑 --down 关机，或者直接去平台控制台关机。"
    fi
  else
    info "没有会话记录（还没跑过 demo_up.sh）"
  fi
  [ "$DO_COST" = "1" ] && [ "$DO_DOWN" = "0" ] && exit 0
fi

# ---------------------------------------------------------------------------
# 关闭
# ---------------------------------------------------------------------------
if [ "$DO_DOWN" = "1" ]; then
  step "关闭服务"
  if [ -d "$PID_DIR" ]; then
    for pf in "$PID_DIR"/*.pid; do
      [ -f "$pf" ] || continue
      pid=$(cat "$pf")
      name=$(basename "$pf" .pid)
      if kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null && ok "停掉 $name (pid $pid)"
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
      fi
      rm -f "$pf"
    done
    rmdir "$PID_DIR" 2>/dev/null || true
  else
    warn "没有找到进程记录，手动清理：pkill -f vllm; pkill -f uvicorn"
  fi

  warn "服务已停，但**实例还在计费**（只要 GPU 开着就在烧钱）"
  warn "去平台控制台关机，或者用平台提供的 shutdown 命令。"

  # 如果在 AutoDL 上，直接提示
  if [ -x /usr/bin/shutdown ]; then
    echo
    printf "${C_Y}现在就关机？(y/N) ${C_R}"
    read -r ans
    if [ "$ans" = "y" ] || [ "$ans" = "Y" ]; then
      ok "关机"
      /usr/bin/shutdown
    else
      warn "记得一会儿关。空转的 GPU 和训练的 GPU 一个价。"
    fi
  fi
  exit 0
fi

# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------
step "启动演示会话"
date +%s > "$START_MARK"
mkdir -p "$PID_DIR" logs
info "计费开始计时（¥${RATE_PER_HOUR}/小时）"

# --- 1. vLLM ---
if [ "$NO_VLLM" = "0" ]; then
  step "1/3 启动 vLLM"

  if ! command -v vllm >/dev/null 2>&1; then
    err "找不到 vllm 命令"
    echo "  vLLM 必须在独立环境里（它会换掉 torch，见 docs/13 第 6 节坑 2）"
    echo "  装：bash scripts/cloud_bootstrap.sh --with-vllm"
    echo "  或者先跳过：bash scripts/demo_up.sh --no-vllm"
    exit 1
  fi

  if [ ! -d "$MODEL_PATH" ]; then
    warn "找不到合并后的权重：$MODEL_PATH"
    echo "  你可能只训了 LoRA adapter，还没合并。先合并："
    echo "    python -m src.train.merge_lora --adapter outputs/qwen25vl3b-cx-lora-v0 \\"
    echo "        --base Qwen/Qwen2.5-VL-3B-Instruct --out $MODEL_PATH"
    echo "  或者临时用原始模型演示："
    echo "    bash scripts/demo_up.sh --model Qwen/Qwen2.5-VL-3B-Instruct"
    exit 1
  fi

  info "模型 : $MODEL_PATH"
  info "端口 : $VLLM_PORT"
  # 关键参数解释见 src/serve/vllm_server.sh
  nohup vllm serve "$MODEL_PATH" \
      --served-model-name "${SERVED_MODEL:-cx-vlm}" \
      --port "$VLLM_PORT" \
      --limit-mm-per-prompt image=3 \
      --enable-prefix-caching \
      --gpu-memory-utilization 0.88 \
      --max-model-len 8192 \
      > logs/vllm.log 2>&1 &
  echo $! > "$PID_DIR/vllm.pid"
  ok "vLLM 启动中 (pid $!) —— 日志: logs/vllm.log"

  info "等待就绪（首次加载权重需要 30–90 秒）..."
  for i in $(seq 1 60); do
    if curl -sf "http://localhost:$VLLM_PORT/health" >/dev/null 2>&1; then
      ok "vLLM 就绪（等了 ${i}s）"
      break
    fi
    if ! kill -0 "$(cat "$PID_DIR/vllm.pid")" 2>/dev/null; then
      err "vLLM 进程退出了。看 logs/vllm.log 最后 30 行："
      tail -30 logs/vllm.log
      exit 1
    fi
    sleep 2
    [ "$i" = "60" ] && { err "120 秒还没起来"; tail -20 logs/vllm.log; exit 1; }
  done
else
  step "1/3 跳过 vLLM（--no-vllm）"
  info "应用层会走 VLLM_BASE 指向的远端服务"
fi

# --- 2. 应用层 ---
step "2/3 启动应用层"
export VLLM_BASE="${VLLM_BASE:-http://localhost:$VLLM_PORT/v1}"
export APP_PORT

if [ -f "src/shopify/app.py" ]; then
  nohup python -m uvicorn src.shopify.app:app \
      --host 0.0.0.0 --port "$APP_PORT" \
      > logs/app.log 2>&1 &
  echo $! > "$PID_DIR/app.pid"
  ok "Shopify App 后端启动中 (pid $!) —— 日志: logs/app.log"
  sleep 3

  if curl -sf "http://localhost:$APP_PORT/health" >/dev/null 2>&1; then
    ok "应用层就绪"
  else
    warn "应用层还没起来，看 logs/app.log（可能是 DATABASE_URL 没配）"
  fi
else
  warn "还没写应用层（Day 39 之后才有），跳过"
fi

# --- 3. 自检 ---
step "3/3 自检"

# 用一张真实的测试图打一次完整链路
python - <<PYEOF
import base64, io, json, os, sys, urllib.request
from PIL import Image, ImageDraw

port = os.getenv("VLLM_PORT", "8000")
app_port = os.getenv("APP_PORT", "8080")

img = Image.new("RGB", (336, 336), (250, 250, 250))
d = ImageDraw.Draw(img)
d.ellipse([50, 90, 150, 190], fill=(220, 120, 50))
d.rectangle([190, 110, 290, 220], fill=(50, 100, 190))
buf = io.BytesIO(); img.save(buf, format="PNG")
b64 = base64.b64encode(buf.getvalue()).decode()

ok = True

# 1) vLLM 直连
if os.getenv("SKIP_VLLM_TEST") != "1":
    payload = {
        "model": os.getenv("SERVED_MODEL", "cx-vlm"),
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
            {"type": "text", "text": "图里有什么？一句话。"},
        ]}],
        "max_tokens": 60, "temperature": 0,
    }
    try:
        req = urllib.request.Request(
            f"http://localhost:{port}/v1/chat/completions",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        resp = json.loads(urllib.request.urlopen(req, timeout=60).read())
        ans = resp["choices"][0]["message"]["content"].strip()
        print(f"  [ok] vLLM 直接推理: {ans[:100]}")
        print(f"       usage: {resp.get('usage')}")
    except Exception as e:
        print(f"  [!!] vLLM 推理失败: {type(e).__name__}: {e}")
        ok = False

# 2) 应用层
try:
    urllib.request.urlopen(f"http://localhost:{app_port}/health", timeout=5)
    print(f"  [ok] 应用层 /health 可达")
except Exception as e:
    print(f"  [!!] 应用层不可达: {type(e).__name__}")
    ok = False

sys.exit(0 if ok else 1)
PYEOF

# ---------------------------------------------------------------------------
step "会话已启动"

IP=$(hostname -I 2>/dev/null | awk '{print $1}')
[ -z "$IP" ] && IP="<实例IP>"

cat <<EOF

  ${C_G}服务地址${C_R}
    vLLM  : http://localhost:$VLLM_PORT/v1       （OpenAI 兼容）
    应用  : http://$IP:$APP_PORT
    日志  : logs/vllm.log  /  logs/app.log

  ${C_Y}别忘了${C_R}
    · 平台控制台要把这两个端口开放（安全组 / 自定义服务）
    · 演示结束后 ${C_BOLD}立刻${C_R}跑：bash scripts/demo_up.sh --down
    · 想看这次烧了多少钱：bash scripts/demo_up.sh --cost

  ${C_Y}关于长期成本（重要）${C_R}
    按 ¥$RATE_PER_HOUR/h 算，7x24 开一周是 ¥$(echo "scale=0; 1.88*24*7" | bc 2>/dev/null)，
    8 周下来比所有训练加起来还贵 10 倍。
    如果这个服务要长期在线，正确做法是把合并后的权重部署到按 token 计费的
    推理服务（没有请求就不花钱），而不是让 GPU 空转。
    详见 docs/13-hardware-and-cost.md 第三节末尾。

EOF

ok "demo_up 结束"
