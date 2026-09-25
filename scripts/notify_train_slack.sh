#!/usr/bin/env bash
# Usage: notify_train_slack.sh <phase_label> <max_iterations> -- <train.py args...>
# Launches train.py in background, posts a Slack "started" message with server/PID/user,
# waits for it to finish, then posts a "finished" (or "failed") message.
set -uo pipefail

# Webhook URL is kept out of git: set SLACK_WEBHOOK_URL or put it in ~/.config/go2_slack_webhook
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-$(cat "$HOME/.config/go2_slack_webhook" 2>/dev/null)}"
REPO_DIR="/home/tanaka/isaacsim/unitree_rl_lab"
VENV_ACTIVATE="/home/tanaka/isaacsim/env_isaaclab/bin/activate"

PHASE_LABEL="$1"; shift
MAX_ITER="$1"; shift
if [ "$1" = "--" ]; then shift; fi
TRAIN_ARGS=("$@")

USER_NAME="$(whoami)"
HOST_NAME="$(hostname)"
SERVER_IP="$(hostname -I 2>/dev/null | tr ' ' '\n' | grep -E '^192\.168\.' | head -1)"
LOG_FILE="/tmp/go2_train_${PHASE_LABEL}.log"

slack_post() {
  local text="$1"
  curl -s -X POST -H 'Content-type: application/json' \
    --data "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1]}))' "$text")" \
    "$SLACK_WEBHOOK_URL" > /dev/null
}

cd "$REPO_DIR"
source "$VENV_ACTIVATE"

nohup python scripts/rsl_rl/train.py "${TRAIN_ARGS[@]}" > "$LOG_FILE" 2>&1 &
TRAIN_PID=$!

# Minimal notifications per user directive (2026-07-07): name + server + status only.
slack_post "▶️ ${PHASE_LABEL} 開始 (${HOST_NAME})"

wait "$TRAIN_PID"
EXIT_CODE=$?

if [ "$EXIT_CODE" -eq 0 ]; then
  slack_post "✅ ${PHASE_LABEL} 完了 (${HOST_NAME})"
else
  slack_post "❌ ${PHASE_LABEL} 失敗 (${HOST_NAME})"
fi

exit "$EXIT_CODE"
