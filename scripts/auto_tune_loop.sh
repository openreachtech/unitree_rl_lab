#!/usr/bin/env bash
# Fully unattended reward-tuning loop for Go2 blind-stair Phase3 (curriculum_level=3).
# Survives the Claude Code session/laptop closing entirely: no LLM calls, pure bash+python.
#
# Each cycle: restore the config to the current best-known-good snapshot, apply ONE
# reward-weight patch from the queue (with a documented evidentiary rationale -- never
# an arbitrary guess), train a short trial resuming from the best checkpoint, measure
# final Curriculum/terrain_levels, and keep the change only if it improves on the best.
#
# Confounding control (2026-07-05, per user request): a greedy "always branch from
# current best" search cannot tell a real standalone (main) effect apart from an
# effect that only exists in combination with the other already-stacked changes
# (interaction effect). So whenever a hypothesis is ADOPTED, we additionally run a
# confirmation trial applying that SAME single change to BASE_RUN (the common
# ancestor from before any of this reward-tuning stack began) and compare against
# BASE_LEVEL. This roughly doubles cost only for winners, not for every hypothesis,
# which keeps it tractable (full factorial would be 2^n trials).
#
# Never touches terrain/curriculum/command-range files (reward weights only, per
# project rule). Stops on: goal reached, queue exhausted, or a training failure
# (does not cascade past a crash).
set -uo pipefail

# Webhook URL is kept out of git: set SLACK_WEBHOOK_URL or put it in ~/.config/go2_slack_webhook
SLACK_WEBHOOK_URL="${SLACK_WEBHOOK_URL:-$(cat "$HOME/.config/go2_slack_webhook" 2>/dev/null)}"
REPO_DIR="/home/tanaka/isaacsim/unitree_rl_lab"
VENV_ACTIVATE="/home/tanaka/isaacsim/env_isaaclab/bin/activate"
CFG_FILE="$REPO_DIR/source/unitree_rl_lab/unitree_rl_lab/tasks/locomotion/robots/go2/velocity_env_cfg_go2.py"
LOG_ROOT="$REPO_DIR/logs/rsl_rl/unitree_go2_velocity_v1"
STATE_FILE="/tmp/go2_autotune_state.txt"
ANALYSIS_LOG="$REPO_DIR/AUTOTUNE_LOG.md"
TRIAL_ITERS=1500
GOAL_LEVEL=6.0

# Common ancestor before target_clearance/joint_pos tuning began (right after
# demote_fraction=0.25 was adopted). Fixed reference point for main-effect checks.
BASE_RUN="2026-07-04_23-10-53"
BASE_CKPT="model_16900.pt"

cd "$REPO_DIR"
# CRITICAL: without this, python3 below is the system interpreter with no
# tensorboard installed -> terrain_level_of/is_better/reached_goal all crash
# silently, and every hypothesis gets rejected regardless of actual result
# (this happened for real on 2026-07-05; all 5 queue items were discarded
# without ever being genuinely evaluated). Always activate the venv first.
source "$VENV_ACTIVATE"

slack_post() {
  curl -s -X POST -H 'Content-type: application/json' \
    --data "$(python3 -c 'import json,sys; print(json.dumps({"text": sys.argv[1]}))' "$1")" \
    "$SLACK_WEBHOOK_URL" > /dev/null
}

log_entry() {
  # Appends a structured markdown entry to the persistent analysis log.
  printf "%s\n\n" "$1" >> "$ANALYSIS_LOG"
}

latest_run_dir() { ls -dt "$LOG_ROOT"/2*/ 2>/dev/null | head -1 | sed 's:/$::'; }

final_checkpoint() {
  ls "$1"/model_*.pt 2>/dev/null \
    | sed -E 's/.*model_([0-9]+)\.pt/\1 &/' | sort -n | tail -1 | awk '{print $2}' | xargs -n1 basename
}

terrain_level_of() {
  # Mean of the last 15 Curriculum/terrain_levels points (smooths step-to-step noise).
  # Always prints exactly one line: a float, or "nan" on ANY failure (never empty
  # output), so callers can reliably detect a failed measurement instead of
  # silently mis-parsing empty stdout as a valid (rejecting) comparison.
  python3 - "$1" << 'EOF'
import sys
try:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(sys.argv[1], size_guidance={'scalars': 0}); ea.Reload()
    s = ea.Scalars("Curriculum/terrain_levels")
    vals = [e.value for e in s[-15:]]
    print(sum(vals) / len(vals) if vals else "nan")
except Exception:
    print("nan", file=sys.stderr)
    print("nan")
EOF
}

wait_for_training_to_finish() {
  while pgrep -f "python scripts/rsl_rl/train.py" > /dev/null 2>&1; do sleep 30; done
}

load_state() {
  if [ -f "$STATE_FILE" ]; then
    IFS=$'\t' read -r BEST_RUN BEST_CKPT BEST_LEVEL BASE_LEVEL < "$STATE_FILE"
  else
    # Corrected 2026-07-05 after the venv bug above caused every prior queue item
    # to be rejected without real evaluation. Manually re-measured with a working
    # venv: 2026-07-05_07-06-24 (target_clearance=0.15 + joint_pos=-0.2 stacked on
    # the demote_fraction=0.25 baseline) is the true best so far, terrain_levels=5.28
    # (mean of last 15 points), beating every other run measured (4.74-5.01).
    BEST_RUN="2026-07-05_07-06-24"; BEST_CKPT="model_19898.pt"; BEST_LEVEL="5.28"
    BASE_LEVEL="$(terrain_level_of "$LOG_ROOT/$BASE_RUN")"
  fi
}

save_state() {
  printf "%s\t%s\t%s\t%s\n" "$BEST_RUN" "$BEST_CKPT" "$BEST_LEVEL" "$BASE_LEVEL" > "$STATE_FILE"
}

evaluate_run() {
  # $1 = run_dir (already completed). Prints "level<TAB>checkpoint".
  local run_dir="$1"
  local level ckpt
  level="$(terrain_level_of "$run_dir")"
  ckpt="$(final_checkpoint "$run_dir")"
  printf "%s\t%s\n" "$level" "$ckpt"
}

is_better() {
  python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) > float(sys.argv[2]) else 1)" "$1" "$2"
}

reached_goal() {
  python3 -c "import sys; sys.exit(0 if float(sys.argv[1]) >= float(sys.argv[2]) else 1)" "$1" "$2"
}

# run_one_trial <label> <load_run> <load_ckpt> ; prints "level<TAB>checkpoint<TAB>rundir"
run_one_trial() {
  local label="$1" load_run="$2" load_ckpt="$3"
  if ! bash scripts/notify_train_slack.sh "AutoTune-${label}" "$TRIAL_ITERS" -- \
      --task Unitree-Go2-Velocity-v1 --headless --num_envs 4096 --max_iterations "$TRIAL_ITERS" --deploy-keyboard-commands \
      --resume --load_run "$load_run" --checkpoint "$load_ckpt"; then
    return 1
  fi
  local trial_dir level ckpt
  trial_dir="$(latest_run_dir)"
  IFS=$'\t' read -r level ckpt < <(evaluate_run "$trial_dir")
  printf "%s\t%s\t%s\n" "$level" "$ckpt" "$trial_dir"
}

load_state
wait_for_training_to_finish
slack_post "🤖 [Go2 RL] auto_tune_loop 起動(厳密化版: 採用時に単体確認runを追加)。best=${BEST_LEVEL}(${BEST_RUN}) base=${BASE_LEVEL}(${BASE_RUN})。目標 >= ${GOAL_LEVEL}"

if reached_goal "$BEST_LEVEL" "$GOAL_LEVEL"; then
  slack_post "🎉 [Go2 RL] 目標到達済み(terrain_levels=${BEST_LEVEL})。auto_tune_loop終了"
  exit 0
fi

# --- Hypothesis queue: <label>|<func_name>|<new_weight>|<rationale> ---
# Only touches existing RewTerm weight= values already present in RewardsCfgGo2.
# Never touches terrain/curriculum/command-range files. Every rationale below is
# grounded in logged Episode_Reward/* evidence or an existing documented design
# constraint (Fix comments in velocity_env_cfg_go2.py) -- not an arbitrary guess.
#
# forward_progress_2.2 intentionally omitted here: it was already launched under
# the pre-fix (round-1 buggy) script before this rewrite, correctly resumed from
# the true best (2026-07-05_07-06-24/model_19898.pt), and is being evaluated +
# confirmation-tested manually rather than re-run, to avoid wasting that trial.
QUEUE=(
  "flat_orientation_-0.5|flat_orientation_l2|-0.5|Already halved once (-2.5->-1.0, Fix3) because steep body pitch is required to climb. Episode_Reward/flat_orientation_l2 has stayed small (-0.03 to -0.05) in every run, i.e. rarely saturating -- testing whether the residual penalty still caps the more aggressive pitch needed on the hardest (0.15-0.23m) steps that the population has not yet conquered (plateau ~5, short of the top rows)."
  "lin_vel_z_-0.5|lin_vel_z_l2|-0.5|Same Fix-3 category: climbing requires vertical body motion, penalised by this term. Already halved once (-2.0->-1.0). Episode_Reward/base_linear_velocity has stayed small (-0.01 to -0.03) in every run, suggesting room to relax further without it having been a dominant term."
  "foot_clearance_0.8|foot_clearance_terrain_adaptive|0.8|Episode_Reward/foot_clearance_terrain_adaptive has logged 0.50-0.63 in every run, i.e. near its practical ceiling given weight=0.5 -- the robot reliably earns most of the available clearance credit already. Raising the weight (distinct from the already-adopted target_clearance=0.15) tests whether making confident stepping relatively MORE valuable than speed/tracking shifts behaviour toward the hardest rows."
  "air_time_var_-0.05|air_time_variance_penalty|-0.05|Already relaxed once (-1.0->-0.2, Fix6) for the inherently asymmetric stair gait. Episode_Reward/air_time_variance has stayed near-zero (-0.004 to -0.009) in every run, i.e. NOT currently a binding constraint at -0.2. Weakest-evidence hypothesis in this queue by design: included as a control -- an expected null result would corroborate this term isn't a bottleneck."
)

for item in "${QUEUE[@]}"; do
  IFS='|' read -r label func_name new_weight rationale <<< "$item"

  if reached_goal "$BEST_LEVEL" "$GOAL_LEVEL"; then
    slack_post "🎉 [Go2 RL] 目標到達(terrain_levels=${BEST_LEVEL})。auto_tune_loop終了"
    exit 0
  fi

  PREV_BEST_RUN="$BEST_RUN"; PREV_BEST_CKPT="$BEST_CKPT"; PREV_BEST_LEVEL="$BEST_LEVEL"

  # --- Main-line trial: patch on top of the current best ---
  cp "$LOG_ROOT/$BEST_RUN/params/velocity_env_cfg_go2.py" "$CFG_FILE"
  if ! python3 scripts/patch_reward_weight.py "$CFG_FILE" "$func_name" "$new_weight"; then
    slack_post "🛑 [Go2 RL] auto_tune_loop: ${label}のパッチ適用に失敗。手動確認をお願いします。停止します"
    exit 1
  fi
  sed -i -E "s/curriculum_level: int = [0-9]+/curriculum_level: int = 3/" "$CFG_FILE"

  slack_post "🧪 [Go2 RL] 仮説: ${label} (${func_name} -> ${new_weight})
根拠: ${rationale}
best(${BEST_RUN}/${BEST_CKPT}, level=${BEST_LEVEL})から${TRIAL_ITERS}iter開始"

  if ! MAIN_RESULT="$(run_one_trial "$label" "$BEST_RUN" "$BEST_CKPT")"; then
    slack_post "🛑 [Go2 RL] ${label} の学習が失敗しました。次の仮説へ暴走せず停止します。手動確認をお願いします"
    exit 1
  fi
  IFS=$'\t' read -r TRIAL_LEVEL TRIAL_CKPT TRIAL_DIR <<< "$MAIN_RESULT"
  TRIAL_RUN="$(basename "$TRIAL_DIR")"

  adopted="no"
  if [ "$TRIAL_LEVEL" != "nan" ] && is_better "$TRIAL_LEVEL" "$BEST_LEVEL"; then
    adopted="yes"
    slack_post "✅ [Go2 RL] ${label} 採用: terrain_levels ${TRIAL_LEVEL} > 従来best ${BEST_LEVEL}。単体確認runを追加実行します"
  else
    slack_post "➡️ [Go2 RL] ${label} 不採用: terrain_levels ${TRIAL_LEVEL} (best ${BEST_LEVEL}維持)。次の仮説へ"
  fi

  CONFIRM_LEVEL="(未実施)"
  interpretation="不採用のため単体確認は実施せず。"
  if [ "$adopted" = "yes" ]; then
    # --- Confirmation trial: the SAME single change applied to the bare BASE, to
    # separate "real main effect" from "only works combined with other changes". ---
    cp "$LOG_ROOT/$BASE_RUN/params/velocity_env_cfg_go2.py" "$CFG_FILE"
    if python3 scripts/patch_reward_weight.py "$CFG_FILE" "$func_name" "$new_weight"; then
      sed -i -E "s/curriculum_level: int = [0-9]+/curriculum_level: int = 3/" "$CFG_FILE"
      if CONFIRM_RESULT="$(run_one_trial "${label}-confirm" "$BASE_RUN" "$BASE_CKPT")"; then
        IFS=$'\t' read -r CONFIRM_LEVEL _ _ <<< "$CONFIRM_RESULT"
        if [ "$CONFIRM_LEVEL" != "nan" ] && is_better "$CONFIRM_LEVEL" "$BASE_LEVEL"; then
          interpretation="単体でもbase(${BASE_LEVEL})を上回った(${CONFIRM_LEVEL}) -> ${func_name}=${new_weight}は独立した主効果と判断できる。"
        else
          interpretation="単体ではbase(${BASE_LEVEL})を上回らなかった(${CONFIRM_LEVEL}) -> 主線での改善は${func_name}単体の効果ではなく、既存の採用済み変更(target_clearance=0.15, joint_pos=-0.2等)との組み合わせ(交互作用)に依存している可能性が高い。"
        fi
      else
        interpretation="確認runの学習が失敗。主効果の判定は保留。"
      fi
    else
      interpretation="base側にこの項の上書きが存在せず確認runを実施できなかった(anchor不一致)。主効果の判定は保留。"
    fi
    slack_post "🔬 [Go2 RL] ${label} 単体確認結果: terrain_levels ${CONFIRM_LEVEL} (base=${BASE_LEVEL})
${interpretation}"

    BEST_RUN="$TRIAL_RUN"; BEST_CKPT="$TRIAL_CKPT"; BEST_LEVEL="$TRIAL_LEVEL"
    save_state
  fi

  log_entry "## $(date '+%Y-%m-%d %H:%M') ${label}

- **変更**: \`${func_name}\` -> \`${new_weight}\`
- **根拠**: ${rationale}
- **主線結果** (分岐元: best ${PREV_BEST_RUN}/${PREV_BEST_CKPT}, level=${PREV_BEST_LEVEL}): terrain_levels = ${TRIAL_LEVEL} -> 採用=${adopted}
- **単体確認結果** (分岐元: base ${BASE_RUN}/${BASE_CKPT}, level=${BASE_LEVEL}): terrain_levels = ${CONFIRM_LEVEL}
- **考察**: ${interpretation}
- **現在のbest**: ${BEST_RUN}/${BEST_CKPT} (terrain_levels=${BEST_LEVEL})
"
done

slack_post "⏸️ [Go2 RL] auto_tune_loop: 用意した仮説キューを使い切りました。best=${BEST_LEVEL} (run=${BEST_RUN}/${BEST_CKPT})。詳細は ${ANALYSIS_LOG} 参照。新しい仮説の追加をお願いします"
