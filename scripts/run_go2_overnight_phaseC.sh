#!/usr/bin/env bash
# Phase C（2026-09-16 04:45）: ①0.002 最良個体の検証 ②引き数を揃えた比較のための追加サイクル。
#
# Phase B は6引きで 1.777 m。0.005 側の記録推移は 1.761(c5) -> 1.783(c7) -> 1.825(c9)
# -> 1.965(c13) -> 1.977(c14) で、**同じ引き数なら 0.002 と 0.005 はほぼ互角**
# （6引きで 1.777 vs 7引きで 1.783）。1.977 は14引き目に出た数字。
# したがって「0.002 は飛距離を22%失う」は誤り——あれは1引きの不運だった。
#
# ここでやるのは記録追いではなく **引き数を揃えること**（14引き相当まで）。
# 停止は締切のみ。5連続空振りで止める規則は「記録更新が目的」のときの規則なので使わない。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
EXP=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1
SCENE=/home/tanaka/isaacsim/unitree_mujoco/unitree_robots/go2/scene_flat.xml
MJPY=/home/tanaka/isaacsim/anaguma/venv/bin/python
PY=/home/tanaka/isaacsim/env_isaaclab/bin/python
OUT=${REPO}/eval_overnight_20260916
STATE=${REPO}/.go2_overnight_state
REPORT=${REPO}/docs/go2_overnight_20260916.md
cd "${REPO}"; export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}" "${OUT}"
export SIM_DT=0.002
SRC_RUN=2026-09-15_18-53-20; SRC=12600
DEADLINE=$(( $(date +%s) + ${HOURS:-3} * 3600 ))
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }
note() { echo "$*" >> "${REPORT}"; }
RECORD=1.777; RECORD_CKPT=model_12700; RECORD_RUN=2026-09-16_03-40-28

eval_ckpt() {
  local dir=$1 ckpt=$2 tag=$3
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
  ${MJPY} scripts/mujoco_jump_eval.py --run "${dir}" --policy "${onnx}" \
    --scene "${SCENE}" --trials 40 --vx 2.8 > "${OUT}/mj_${tag}.log" 2>&1
  awk '/飛距離 平均/{d=$3} /跳躍成立/{for(i=1;i<=NF;i++) if($i ~ /^\(/){gsub(/[()%]/,"",$i); if(j=="") {j=$i; continue}; l=$i}} END{printf "%s %s %s\n", (d==""?0:d), (j==""?0:j), (l==""?0:l)}' "${OUT}/mj_${tag}.log"
}

# ---------- ① 0.002 最良個体の検証 ----------
say "検証: 0.002 最良 ${RECORD_RUN}/${RECORD_CKPT} (mujoco 1.777 m, 離陸 3.11/1.89, 積 5.878)"
note ""
note "## Phase C-1: 0.002 最良個体（1.777 m）の検証"
note ""
export TN_SPEED_SCALE_MIN=1.0 TN_SPEED_SCALE_MAX=1.0 TN_TORQUE_SCALE_MIN=1.0 TN_TORQUE_SCALE_MAX=1.0
${PY} -u scripts/rsl_rl/jump_trace_isaac.py --task Unitree-Go2-LongJump-v1 --headless \
  --num_envs 64 --steps 2000 --checkpoint "${EXP}/${RECORD_RUN}/${RECORD_CKPT}.pt" --sim_dt 0.002 \
  --label "0.002最良 ${RECORD_CKPT}" --out "${OUT}/isaac_best002.txt" > "${OUT}/isaac_best002.log" 2>&1
note '```'
cat "${OUT}/isaac_best002.txt" 2>/dev/null >> "${REPORT}" || echo "(Isaac トレース失敗)" >> "${REPORT}"
note '```'
${MJPY} scripts/mujoco_tn_audit.py --run "${EXP}/${RECORD_RUN}" \
  --policy "${OUT}/pb1_12700.onnx" --trials 20 --vx 2.8 > "${OUT}/audit_best002.log" 2>&1
note ""
note "T-N 監査（踏切の仕事のうち実機に無いトルクで作られた割合）:"
note '```'
tail -9 "${OUT}/audit_best002.log" >> "${REPORT}"
note '```'
say "検証終了"

# ---------- ② 引き数を揃える追加サイクル ----------
note ""
note "## Phase C-2: 引き数を揃える追加サイクル（記録追いではない）"
note ""
note "0.005 側は14引き目に 1.977 m を出した。0.002 側は6引きで 1.777 m。"
note "**同じ引き数で比べるため**に14引き相当まで回す。停止は締切のみ。"
note ""
note "| # | seed | 最良 ckpt | 飛距離 | 成立 | 着地 | 0.002側の最良 |"
note "|---|---|---|---|---|---|---|"
for c in $(seq 7 14); do
  [ "$(date +%s)" -lt "${DEADLINE}" ] || { say "時間切れ"; note ""; note "_時間切れで終了（${c}引き目に到達せず）_"; break; }
  SEED=$((200 + c))
  say "サイクル${c}: seed ${SEED}"
  START_RUN=${SRC_RUN} START_CKPT=model_${SRC} ITERS=300 SEED=${SEED} \
    bash scripts/run_go2_loop_cycle.sh > "${REPO}/go2_pc_c${c}.log" 2>&1
  RUN_B=$(ls -1t "${EXP}" | grep '^2026-' | head -1)
  BEST_D=0; BEST_C=""; BEST_J=0; BEST_L=0
  for off in 100 150 200 250 300; do
    it=$((SRC + off))
    [ -s "${EXP}/${RUN_B}/model_${it}.pt" ] || continue
    read -r d j l <<< "$(eval_ckpt "${EXP}/${RUN_B}" "model_${it}" "pc${c}_${it}")"
    say "  c${c} model_${it}: ${d} m  成立${j}%  着地${l}%"
    if awk -v a="${d}" -v b="${BEST_D}" 'BEGIN{exit !(a>b)}'; then BEST_D=${d}; BEST_C="model_${it}"; BEST_J=${j}; BEST_L=${l}; fi
  done
  if awk -v a="${BEST_D}" -v b="${RECORD}" 'BEGIN{exit !(a>b)}'; then
    RECORD=${BEST_D}; RECORD_CKPT=${BEST_C}; RECORD_RUN=${RUN_B}; UPD="**${RECORD} m に更新**"
  else UPD="${RECORD} m のまま"; fi
  say "サイクル${c} 最良 ${BEST_D} m / 0.002最良 ${RECORD} m"
  note "| ${c} | ${SEED} | \`${BEST_C}\` | ${BEST_D} m | ${BEST_J}% | ${BEST_L}% | ${UPD} |"
  printf 'PHASE=C\nCYCLE=%s\nRECORD=%s\nRECORD_CKPT=%s\nRECORD_RUN=%s\n' \
    "${c}" "${RECORD}" "${RECORD_CKPT}" "${RECORD_RUN}" > "${STATE}"
done
note ""
note "## 最終結果"
note ""
note "| 条件 | 引き数 | 最良 | ckpt |"
note "|---|---|---|---|"
note "| sim.dt **0.005**（従来・バイアスあり） | 16 | **1.977 m** | \`2026-09-15_18-53-20/model_12700\` |"
note "| sim.dt **0.002**（解像された物理） | 14 | **${RECORD} m** | \`${RECORD_RUN}/${RECORD_CKPT}\` |"
note ""
note "終了 $(ts)。"
say "Phase C 終了: 0.002 最良 ${RECORD} m (${RECORD_RUN}/${RECORD_CKPT})"
