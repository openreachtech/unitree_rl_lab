#!/usr/bin/env bash
# 2026-09-16 深夜: 「着地の唯一の実測レバー」×「遅延DR」の組み合わせを1本だけ測る。
#
#   起点: 2026-09-16_18-28-38/model_13200
#         （遅延DR系列で遅延1下の最良。飛距離 1.015 m / 着地 25% / 成立 88%）
#   変更: jump_pitch_rate weight 0.0 -> -12.0（max_rate 3.0 は既定のまま）
#         実測の根拠: 2026-09-15 サイクル6 で着地 28% -> 68〜78%（ただし遅延0条件）
#   条件: SIM_DT=0.002 / ACTION_DELAY_STEPS=0,2 / 2048 env / +500 iter
#   評価: --action-delay 1（20 ms）・mujoco 40試行・助走コマンド 2.8（全て既存プロトコルと同じ）
#
# ITERS は「追加 iter 数」。絶対値を渡すと桁違いに長く回る（今日踏んだ）。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
EXP=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1
SCENE=/home/tanaka/isaacsim/unitree_mujoco/unitree_robots/go2/scene_flat.xml
PY=/home/tanaka/isaacsim/env_isaaclab/bin/python
MJPY=/home/tanaka/isaacsim/anaguma/venv/bin/python
OUT=${REPO}/eval_pitch_delay_20260916
REPORT=${REPO}/docs/go2_pitch_delay_20260916.md
cd "${REPO}"; export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}" "${OUT}"
export SIM_DT=0.002 ACTION_DELAY_STEPS=0,2
SRC_RUN=2026-09-16_18-28-38; SRC=13200
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }
note() { echo "$*" >> "${REPORT}"; }

note "# jump_pitch_rate −12 × 行動遅延DR（2026-09-16 深夜）"
note ""
note "起点 \`${SRC_RUN}/model_${SRC}\`（遅延1で 飛距離 1.015 m / 着地 25% / 成立 88%）。"
note "変更は \`jump_pitch_rate\` weight 0.0 → **−12.0** の1点のみ。+500 iter、seed 601。"
note "評価は全て \`--action-delay 1\`・mujoco 40試行・助走コマンド 2.8 m/s。開始 $(ts)。"
note ""

say "学習開始: ${SRC_RUN}/model_${SRC} +500 iter, jump_pitch_rate=-12.0"
JUMP_PITCH_RATE_W=-12.0 START_RUN=${SRC_RUN} START_CKPT=model_${SRC} ITERS=500 SEED=601 \
  bash scripts/run_go2_loop_cycle.sh > "${REPO}/go2_pitch_delay.log" 2>&1
RUN=$(ls -1t "${EXP}" | grep '^2026-' | head -1)
say "学習終了: ${RUN}"
note "run: \`${RUN}\`"
note ""
note "| ckpt | 追加iter | 跳躍成立 | 着地成功 | 飛距離 |"
note "|---|---|---|---|---|"

BEST_L=-1; BEST_C=""; BEST_D=0
for off in 100 200 300 400 500; do
  it=$((SRC + off))
  [ -s "${EXP}/${RUN}/model_${it}.pt" ] || continue
  onnx="${OUT}/pd_${it}.onnx"
  if [ ! -s "${onnx}" ]; then
    rm -f "${EXP}/${RUN}/exported/policy.onnx"
    ${PY} scripts/rsl_rl/play.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 1 \
      --checkpoint "${EXP}/${RUN}/model_${it}.pt" > "${OUT}/export_${it}.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 90); do sleep 5; [ -s "${EXP}/${RUN}/exported/policy.onnx" ] && break; done
    sleep 5; kill -9 "${pid}" 2>/dev/null; sleep 3
    [ -s "${EXP}/${RUN}/exported/policy.onnx" ] || continue
    cp "${EXP}/${RUN}/exported/policy.onnx" "${onnx}"
  fi
  ${MJPY} scripts/mujoco_jump_eval.py --run "${EXP}/${RUN}" --policy "${onnx}" --scene "${SCENE}" \
    --trials 40 --vx 2.8 --action-delay 1 > "${OUT}/mj_${it}.log" 2>&1
  read -r d j l <<< "$(awk '/飛距離 平均/{d=$3} /跳躍成立/{for(i=1;i<=NF;i++) if($i ~ /^\(/){gsub(/[()%]/,"",$i); if(j=="") {j=$i; continue}; l=$i}} END{printf "%s %s %s\n", (d==""?0:d), (j==""?0:j), (l==""?0:l)}' "${OUT}/mj_${it}.log")"
  say "  model_${it} (+${off}): 着地 ${l}%  飛距離 ${d} m  成立 ${j}%"
  note "| \`model_${it}\` | +${off} | ${j}% | ${l}% | ${d} m |"
  if awk -v a="${l}" -v b="${BEST_L}" 'BEGIN{exit !(a>b)}'; then BEST_L=${l}; BEST_C="model_${it}"; BEST_D=${d}; fi
done

note ""
note "**最良: \`${BEST_C}\`（着地 ${BEST_L}% / 飛距離 ${BEST_D} m）**"
note ""
note "比較:"
note "- 起点（pitch レバー無し・遅延DRのみ）: 着地 **25%** / 飛距離 1.015 m"
note "- 遅延ゼロ条件での着地記録（\`2026-09-15_16-35-23/model_12500\`）: 着地 78% / ただし遅延1では **0%**"
note ""
note "終了 $(ts)。"
say "完了: 最良 ${BEST_C} 着地 ${BEST_L}% 飛距離 ${BEST_D} m（起点は 25% / 1.015 m）"
echo DONE_PITCH_DELAY
