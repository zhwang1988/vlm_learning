#!/usr/bin/env bash
# =============================================================================
# sync_down.sh —— 把云上的东西拉回本地
#
# 这是整个项目里最重要的一条纪律：
#
#   云 GPU 平台的实例是「随时可能消失」的。
#   平台关停、实例被回收、手滑删数据盘 —— 8 周的成果就没了。
#   所以：**训练产物必须定期拉回本地。**
#
# 用法（在本地 Mac 上跑，不是云上）：
#   bash scripts/sync_down.sh --host <ip> --port <ssh端口>          # 拉全部
#   bash scripts/sync_down.sh --host <ip> --port 12345 --only outputs
#   bash scripts/sync_down.sh --dry-run                             # 先看要拉什么
#
# AutoDL 的 SSH 信息在「实例详情」里，格式是：
#   登录指令: ssh -p 12345 root@connect.xxx.seetacloud.com
#   → --host connect.xxx.seetacloud.com --port 12345
# =============================================================================

set -o pipefail

C_G="\033[32m"; C_Y="\033[33m"; C_RD="\033[31m"; C_B="\033[36m"; C_R="\033[0m"; C_BOLD="\033[1m"
ok()   { printf "${C_G}[✓]${C_R} %s\n" "$*"; }
info() { printf "${C_B}[·]${C_R} %s\n" "$*"; }
warn() { printf "${C_Y}[!]${C_R} %s\n" "$*"; }
err()  { printf "${C_RD}[✗]${C_R} %s\n" "$*"; }
step() { printf "\n${C_BOLD}==> %s${C_R}\n" "$*"; }

HOST=""
PORT="22"
USER_NAME="root"
REMOTE_ROOT="/root/autodl-tmp/multimodal-lab"
LOCAL_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ONLY=""
DRY_RUN=0

while [ $# -gt 0 ]; do
  case "$1" in
    --host)     HOST="$2"; shift 2 ;;
    --port)     PORT="$2"; shift 2 ;;
    --user)     USER_NAME="$2"; shift 2 ;;
    --remote)   REMOTE_ROOT="$2"; shift 2 ;;
    --local)    LOCAL_ROOT="$2"; shift 2 ;;
    --only)     ONLY="$2"; shift 2 ;;
    --dry-run)  DRY_RUN=1; shift ;;
    -h|--help)  sed -n '2,22p' "$0"; exit 0 ;;
    *) err "未知参数: $1"; exit 1 ;;
  esac
done

if [ -z "$HOST" ]; then
  err "必须给 --host"
  echo
  echo "  例: bash scripts/sync_down.sh --host connect.xxx.seetacloud.com --port 12345"
  echo
  echo "  不想用 SSH？还有两个办法："
  echo "    1) 在 JupyterLab 里把 outputs/ 打包下载（点右键 Download）"
  echo "    2) 平台一般有「文件下载」入口，直接下 zip"
  exit 1
fi

RSH="ssh -p $PORT -o ConnectTimeout=10 -o StrictHostKeyChecking=accept-new $USER_NAME@$HOST"

# 要拉的东西。按重要度排序 —— 前面的是「丢了会哭」的。
ITEMS="
outputs:训练产物（adapter / checkpoint / merged）—— 最重要
reports:评测报告、错误分析、bad cases
data/processed:清洗打包好的训练数据（花了很多合成成本）
logs:训练日志和 tensorboard
progress:进度追踪
"

step "检查连接"
if ! $RSH "echo ok" >/dev/null 2>&1; then
  err "连不上 $USER_NAME@$HOST:$PORT"
  echo "  排查："
  echo "    1) 实例是不是关机了？（关机后 SSH 连不上，得先开机）"
  echo "    2) 端口对不对？AutoDL 每个实例端口不同"
  echo "    3) 是不是要用「无卡模式开机」才能省着传文件"
  exit 1
fi
ok "连接正常"

if [ "$DRY_RUN" = "1" ]; then
  step "dry-run：只列出远端有什么"
  echo
  echo "$ITEMS" | while IFS=: read -r rel desc; do
    [ -z "$rel" ] && continue
    [ -n "$ONLY" ] && [ "$rel" != "$ONLY" ] && continue
    remote_path="$REMOTE_ROOT/$rel"
    if $RSH "test -d '$remote_path'" 2>/dev/null; then
      size=$($RSH "du -sh '$remote_path' 2>/dev/null | cut -f1" 2>/dev/null)
      count=$($RSH "find '$remote_path' -type f 2>/dev/null | wc -l" 2>/dev/null)
      printf "  %-20s %8s  %5s 个文件   %s\n" "$rel" "${size:-?}" "${count:-?}" "$desc"
    else
      printf "  %-20s %8s  %5s       %s\n" "$rel" "-" "-" "$desc"
    fi
  done
  echo
  info "确认无误后去掉 --dry-run 再跑一次"
  exit 0
fi

step "开始同步"

echo "$ITEMS" | while IFS=: read -r rel desc; do
  [ -z "$rel" ] && continue
  [ -n "$ONLY" ] && [ "$rel" != "$ONLY" ] && continue

  remote_path="$REMOTE_ROOT/$rel"
  if ! $RSH "test -d '$remote_path'" 2>/dev/null; then
    warn "远端没有 $rel，跳过"
    continue
  fi

  mkdir -p "$LOCAL_ROOT/$rel"
  info "拉 $rel  ($desc)"
  # -a 保留权限和时间；--info=progress2 显示总进度
  if rsync -az --info=progress2 \
       -e "ssh -p $PORT -o StrictHostKeyChecking=accept-new" \
       "$USER_NAME@$HOST:$remote_path/" "$LOCAL_ROOT/$rel/" 2>&1 | tail -1; then
    size=$(du -sh "$LOCAL_ROOT/$rel" 2>/dev/null | cut -f1)
    ok "$rel -> $LOCAL_ROOT/$rel  ($size)"
  else
    err "$rel 同步失败"
    echo "    如果本地没装 rsync，用 tar 走一遍："
    echo "      $RSH \"tar czf - -C $REMOTE_ROOT $rel\" | tar xzf - -C $LOCAL_ROOT"
  fi
done

step "完成"
cat <<EOF

  本地备份位置：$LOCAL_ROOT

  ${C_Y}建议：${C_R}
    · 每次训完一个 epoch 就跑一次这个脚本
    · outputs/ 里的 adapter 通常只有几十 MB，拉起来很快
    · 如果拉的是大 checkpoint，先想清楚哪个真的需要 —— 最后一个通常就够了

  下一步（可选）：推到对象存储/网盘做二次备份
    平台跑路 + 本地硬盘坏了同时发生的概率不高，但也不是零。

EOF
