#!/usr/bin/env bash
set -uo pipefail

WORKDIR="$HOME/.openclaw/workspace"
LOG_DIR="$WORKDIR/logs"
LOG_FILE="$LOG_DIR/daily-status-report.log"
USER_ID="${REPORT_USER_ID:-ou_b77c437e5ef8bcf9c7f3ffef86af2937}"

mkdir -p "$LOG_DIR"
exec >>"$LOG_FILE" 2>&1

echo "[$(date '+%F %T')] ===== daily-status-report start ====="

PATH="/home/oc/.local/share/pnpm:/home/oc/.nvm/versions/node/v24.13.1/bin:/home/oc/.local/bin:/home/oc/bin:/usr/local/bin:/usr/bin:/bin"
export PATH HOME

cd "$WORKDIR" || {
  echo "[$(date '+%F %T')] ERROR: cannot cd to $WORKDIR"
  exit 1
}

OPENCLAW_BIN="$(command -v openclaw || true)"
if [ -z "$OPENCLAW_BIN" ]; then
  for p in \
    /home/oc/.local/share/pnpm/openclaw \
    /home/oc/.nvm/versions/node/v24.13.1/bin/openclaw \
    /usr/local/bin/openclaw \
    /usr/bin/openclaw; do
    if [ -x "$p" ]; then
      OPENCLAW_BIN="$p"
      break
    fi
  done
fi

if [ -z "$OPENCLAW_BIN" ]; then
  echo "[$(date '+%F %T')] ERROR: openclaw not found in PATH=$PATH"
  exit 1
fi

echo "[$(date '+%F %T')] openclaw=$OPENCLAW_BIN"

check_http() {
  local url="$1"
  local token="$2"
  local code
  code="$(curl -sS --max-time 10 -o /dev/null -w '%{http_code}' "$url" -H "Authorization: Bearer $token" || echo 000)"
  if [ "$code" = "200" ]; then
    echo "✅ 正常 (HTTP 200)"
  else
    echo "❌ 异常 (HTTP $code)"
  fi
}

KIMI_CHECK="$(check_http 'https://api.moonshot.cn/v1/models' 'sk-AwkryXkjIne5LeO68usLJVZYPOBWatUai919gngMwVnj9mP2')"
GPT_CHECK="$(check_http 'https://gmn.chuangzuoli.com/v1/models' 'sk-52c125ab14a851c2d9a39a5f0e489cbe2bf58fa008960cecb04a2ea41d82bd6e')"

HOSTNAME="$(hostname)"
KERNEL="$(uname -sr)"
UPTIME_TXT="$(uptime | sed 's/.*up \([^,]*\),.*/\1/')"
LOAD_AVG="$(uptime | awk -F'load average:' '{print $2}' | xargs)"

MEM_TOTAL_KB="$(awk '/MemTotal/ {print $2}' /proc/meminfo)"
MEM_AVAIL_KB="$(awk '/MemAvailable/ {print $2}' /proc/meminfo)"
MEM_FREE_KB="$(awk '/MemFree/ {print $2}' /proc/meminfo)"
MEM_CACHED_KB="$(awk '/^Cached:/ {print $2}' /proc/meminfo)"
MEM_BUFFERS_KB="$(awk '/Buffers/ {print $2}' /proc/meminfo)"
MEM_USED_KB=$((MEM_TOTAL_KB - MEM_AVAIL_KB))
MEM_USAGE=$((MEM_USED_KB * 100 / MEM_TOTAL_KB))

to_mb() { echo $(( $1 / 1024 )); }

DISK_LINE="$(df -h / | awk 'NR==2 {print $3"/"$2" ("$5")"}')"
GW_ACTIVE="$(systemctl --user is-active openclaw-gateway.service 2>/dev/null || echo unknown)"
GW_PID="$(pgrep -o openclaw-gateway || echo -)"
VERSION="$(OPENCLAW_NO_UPDATE_NOTIFIER=1 timeout 8s "$OPENCLAW_BIN" --version 2>/dev/null || echo unknown)"
MODEL_LINES="$(timeout 15s "$OPENCLAW_BIN" models list 2>/dev/null | tail -n +2 || true)"
if [ -z "$MODEL_LINES" ]; then
  MODEL_LINES="(获取失败，可能是网关暂不可达)"
fi
TOP_PROC="$(ps aux --sort=-%mem | head -6 | tail -5 | awk '{printf "%-22s %6s %6s\n", $11, $4"%", int($6/1024)"MB"}')"

REPORT=$(cat <<EOF
📊 **每日系统状态报告**
时间：\`$(date '+%Y-%m-%d %H:%M:%S')\`
版本：\`$VERSION\`

**🤖 模型联通性**
- GPT 5.3 Codex: $GPT_CHECK
- Kimi K2.5: $KIMI_CHECK

**🧠 模型配置**
\`\`\`
$MODEL_LINES
\`\`\`

**🔌 网关状态**
- 服务：$GW_ACTIVE
- 进程 PID：$GW_PID
- 地址：127.0.0.1:18789
- Dashboard：http://127.0.0.1:18789/

**🖥️ ECS 主机状态**
- 主机：$HOSTNAME
- 系统：$KERNEL
- 运行时长：$UPTIME_TXT
- 负载：$LOAD_AVG
- 内存：$(to_mb "$MEM_USED_KB")MB/$(to_mb "$MEM_TOTAL_KB")MB (${MEM_USAGE}%)
- 可用内存：$(to_mb "$MEM_AVAIL_KB")MB
- 空闲内存：$(to_mb "$MEM_FREE_KB")MB
- 缓存/缓冲：$(to_mb "$MEM_CACHED_KB")MB / $(to_mb "$MEM_BUFFERS_KB")MB
- 磁盘：$DISK_LINE

**📌 Top 内存进程**
\`\`\`
$TOP_PROC
\`\`\`
EOF
)

if timeout 90s "$OPENCLAW_BIN" message send --channel feishu --target "user:$USER_ID" --message "$REPORT"; then
  echo "[$(date '+%F %T')] report sent"
else
  echo "[$(date '+%F %T')] ERROR: failed to send report"
  exit 1
fi

echo "[$(date '+%F %T')] ===== daily-status-report done ====="
