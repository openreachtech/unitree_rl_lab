#!/usr/bin/env bash
# Runs curriculum_level 1 -> 2 -> 3 sequentially, automatically resuming each
# phase from the previous phase's final checkpoint. Stops (does not cascade)
# if a phase fails, and posts a Slack alert in that case.
set -uo pipefail

# Webhook URL is kept out of git: set SLACK_WEBHOOK_URL or put it in ~/.config/go2_slack_webhook
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-$(cat "$HOME/.config/go2_slack_webhook" 2>/dev/null)}"
REPO_DIR="/home/tanaka/isaacsim/unitree_rl_lab"
CFG_FILE="$REPO_DIR/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg_go2.py"
LOG_ROOT="$REPO_DIR/logs/rsl_rl/unitree_go2_velocity_v1"

cd "$REPO_DIR"

slack_post() {
  curl -s -X POST -H 'Content-type: application/json' \
    --data "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1]}))' "$1")" \
    "$SLACK_WEBHOOK_URL" > /dev/null
}

set_curriculum_level() {
  sed -i -E "s/curriculum_level: int = [0-9]+/curriculum_level: int = $1/" "$CFG_FILE"
}

latest_run_dir() {
  ls -dt "$LOG_ROOT"/2*/ 2>/dev/null | head -1 | sed 's:/$::'
}

final_checkpoint() {
  ls "$1"/model_*.pt 2>/dev/null \
    | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | awk '{print $2}' | xargs -n1 basename
}

# run_phase <label> <max_iterations> <curriculum_level> <train.py extra args...>
run_phase() {
  local label="$1" max_iter="$2" level="$3"
  shift 3
  set_curriculum_level "$level"
  echo "[INFO] === ${label}: curriculum_level=${level}, max_iterations=${max_iter} ==="
  if ! bash scripts/notify_train_slack.sh "$label" "$max_iter" -- "$@"; then
    slack_post "🛑 [Go2 RL] ${label} が失敗したため、以降のフェーズを自動起動せず停止しました。手動確認をお願いします。"
    exit 1
  fi
}

# --- Phase1: level=1, fresh ---
run_phase Phase1 3000 1 \
  --task Unitree-Go2-Velocity-v1 --headless --num_envs 4096 --max_iterations 3000 --deploy-keyboard-commands
P1_DIR="$(latest_run_dir)"; P1_RUN="$(basename "$P1_DIR")"; P1_CKPT="$(final_checkpoint "$P1_DIR")"
echo "[INFO] Phase1 done: run=$P1_RUN checkpoint=$P1_CKPT"

# --- Phase2: level=2, resume from Phase1 ---
run_phase Phase2 4000 2 \
  --task Unitree-Go2-Velocity-v1 --headless --num_envs 4096 --max_iterations 4000 --deploy-keyboard-commands \
  --resume --load_run "$P1_RUN" --checkpoint "$P1_CKPT"
P2_DIR="$(latest_run_dir)"; P2_RUN="$(basename "$P2_DIR")"; P2_CKPT="$(final_checkpoint "$P2_DIR")"
echo "[INFO] Phase2 done: run=$P2_RUN checkpoint=$P2_CKPT"

# --- Phase3: level=3, resume from Phase2 ---
run_phase Phase3 5000 3 \
  --task Unitree-Go2-Velocity-v1 --headless --num_envs 4096 --max_iterations 5000 --deploy-keyboard-commands \
  --resume --load_run "$P2_RUN" --checkpoint "$P2_CKPT"
P3_DIR="$(latest_run_dir)"; P3_RUN="$(basename "$P3_DIR")"; P3_CKPT="$(final_checkpoint "$P3_DIR")"
echo "[INFO] Phase3 done: run=$P3_RUN checkpoint=$P3_CKPT"

slack_post "🎉 [Go2 RL] 全フェーズ(1->2->3)完走しました。最終run: ${P3_RUN} / checkpoint: ${P3_CKPT}"
