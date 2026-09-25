#!/usr/bin/env bash
# 2026-09-16: 行動遅延 DR での再学習と、遅延つき評価。
#
# 起点は D(着地重視) 2026-09-16_05-50-08/model_12700（遅延ゼロ条件で 1.338m/着地48%、
# ただし遅延1ステップで着地5%）。ここから ACTION_DELAY_STEPS=0,2 で回し、
# **評価は --action-delay 1 を主指標にする**（0 の数字は実機予測にならないため）。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
EXP=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1
SCENE=/home/tanaka/isaacsim/unitree_mujoco/unitree_robots/go2/scene_flat.xml
PY=/home/tanaka/isaacsim/env_isaaclab/bin/python
MJPY=/home/tanaka/isaacsim/anaguma/venv/bin/python
OUT=${REPO}/eval_delay_dr_20260916
REPORT=${REPO}/docs/go2_delay_dr_20260916.md
cd "${REPO}"; export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}" "${OUT}"
export SIM_DT=0.002 ACTION_DELAY_STEPS=${ACTION_DELAY_STEPS:-0,2}
SRC_RUN=${SRC_RUN:-2026-09-16_05-50-08}; SRC_CKPT=${SRC_CKPT:-model_12700}; SRC=12700
ITERS=${ITERS:-1200}
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }
note() { echo "$*" >> "${REPORT}"; }

note "# Go2 走り幅跳び 行動遅延 DR（2026-09-16）"
note ""
note "起点 \`${SRC_RUN}/${SRC_CKPT}\`、\`ACTION_DELAY_STEPS=${ACTION_DELAY_STEPS}\`、+${ITERS} iter。開始 $(ts)。"
note ""
note "**評価の主指標は \`--action-delay 1\`（20 ms）**。遅延0の数字は実機を予測しないので参考。"
note ""

say "学習開始: ${SRC_RUN}/${SRC_CKPT} +${ITERS} iter (delay ${ACTION_DELAY_STEPS})"
START_RUN=${SRC_RUN} START_CKPT=${SRC_CKPT} ITERS=${ITERS} SEED=${SEED:-301} \
  bash scripts/run_go2_loop_cycle.sh > "${REPO}/go2_delay_dr.log" 2>&1
RUN=$(ls -1t "${EXP}" | grep '^2026-' | head -1)
say "学習終了: ${RUN}"
note "run: \`${RUN}\`"
note ""
note "| ckpt | 遅延1 飛距離 | 遅延1 成立 | 遅延1 着地 | 遅延0 飛距離 | 遅延0 着地 |"
note "|---|---|---|---|---|---|"

eval_one() {  # dir ckpt tag delay -> "距離 成立 着地"
  local dir=$1 ckpt=$2 tag=$3 dl=$4
  local onnx="${OUT}/${tag}.onnx"
  if [ ! -s "${onnx}" ]; then
    rm -f "${dir}/exported/policy.onnx"
    ${PY} scripts/rsl_rl/play.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 1 \
      --checkpoint "${dir}/${ckpt}.pt" > "${OUT}/export_${tag}.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 90); do sleep 5; [ -s "${dir}/exported/policy.onnx" ] && break; done
    sleep 5; kill -9 "${pid}" 2>/dev/null; sleep 3
    [ -s "${dir}/exported/policy.onnx" ] || { echo "0 0 0"; return 1; }
    cp "${dir}/exported/policy.onnx" "${onnx}"
  fi
  ${MJPY} scripts/mujoco_jump_eval.py --run "${dir}" --policy "${onnx}" --scene "${SCENE}" \
    --trials 40 --vx 2.8 --action-delay "${dl}" > "${OUT}/mj_${tag}_d${dl}.log" 2>&1
  awk '/飛距離 平均/{d=$3} /跳躍成立/{for(i=1;i<=NF;i++) if($i ~ /^\(/){gsub(/[()%]/,"",$i); if(j=="") {j=$i; continue}; l=$i}} END{printf "%s %s %s\n", (d==""?0:d), (j==""?0:j), (l==""?0:l)}' "${OUT}/mj_${tag}_d${dl}.log"
}

BEST_D=0; BEST_C=""
for off in 200 400 600 800 1000 1200; do
  it=$((SRC + off))
  [ -s "${EXP}/${RUN}/model_${it}.pt" ] || continue
  read -r d1 j1 l1 <<< "$(eval_one "${EXP}/${RUN}" "model_${it}" "dr_${it}" 1)"
  read -r d0 j0 l0 <<< "$(eval_one "${EXP}/${RUN}" "model_${it}" "dr_${it}" 0)"
  say "  model_${it}: 遅延1 ${d1}m 成立${j1}% 着地${l1}%  /  遅延0 ${d0}m 着地${l0}%"
  note "| \`model_${it}\` | ${d1} m | ${j1}% | ${l1}% | ${d0} m | ${l0}% |"
  if awk -v a="${l1}" -v b="${BEST_D}" 'BEGIN{exit !(a>b)}'; then BEST_D=${l1}; BEST_C="model_${it}"; fi
done

note ""
note "**遅延1での着地が最良: ${BEST_C}（${BEST_D}%）**"
note ""
note "比較（学習前、遅延DRなしの D \`2026-09-16_05-50-08/model_12700\`）: 遅延0で着地48%、遅延1で着地5%・成立35%・飛距離0.359m。"
note ""
note "終了 $(ts)。"
say "完了: 最良 ${BEST_C} (遅延1で着地 ${BEST_D}%)"
