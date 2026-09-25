#!/usr/bin/env bash
# Go2 走り幅跳び: sim.dt=0.002 での基準確定 + 停止条件つき最適化（2026-09-16 夜）。
#
# 背景: sim.dt=0.005 は収束値より29%低い踏切しか作れず、記録1.977mはその条件下の測定だった
#       （docs/go2_loop_log.md、プロジェクト_学習の物理刻みが粗すぎた.md）。
#       0.002 で学習し直すと2つのシミュレータが一致する代わりに飛距離が1.543mに落ちる。
#       ただしその run は action_rate が -66〜-95 と荒れており、新しい力学に馴染む途中の疑いがある。
#
# Phase A: 同じ起点から 0.002 で **+1500 iter** 回して、時間を与えれば馴染むのかを見る。
#          seed も報酬も据え置きなので最初の300 iter は先の run とビット単位で同一（＝延長になる）。
#          100 iter 刻みで mujoco 40試行評価し、0.002 での正直な基準を確定する。
# Phase B: Phase A の最良から、収穫帯(std 0.27-0.34)の起点で 300 iter サイクルを回す。
#          **3サイクル連続で記録更新が無ければ終了**（前回16サイクル回して最後3つが空振りした反省）。
#
# やらないこと: 実機/Drive/社内docへの反映、既定値(sim.dt=0.005)の変更、報酬の変更。
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
MAX_CYCLES=${MAX_CYCLES:-12}
DEADLINE=$(( $(date +%s) + ${HOURS:-7} * 3600 ))
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

RECORD=0
RECORD_CKPT=""
RECORD_RUN=""

# ---- 候補1本を export して mujoco 40試行で測る。標準出力に "距離 成立率 着地率" を返す ----
eval_ckpt() {   # $1=run dir  $2=ckpt名  $3=タグ
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

note() { echo "$*" >> "${REPORT}"; }

cat > "${REPORT}" <<MD
# Go2 走り幅跳び 夜間作業ログ（2026-09-16）

sim.dt=0.002 での基準確定と、停止条件つきの最適化。開始 $(ts)。

**前提**: 記録 1.977 m は sim.dt=0.005（収束値より29%低い）で学習したポリシーの測定値。
0.002 で学習すると2つのシミュレータが一致する代わりに飛距離が落ちる（1.543 m、300 iter 時点）。
その run は action_rate が -66〜-95 と荒れていたので、**時間を与えれば馴染むのか**をまず見る。

MD

# ========================= Phase A: 長い適応 run =========================
say "Phase A 開始: 0.002 で +1500 iter"
note "## Phase A: 0.002 で +1500 iter（起点 2026-09-15_18-53-20/model_12600）"
note ""
START_RUN=2026-09-15_18-53-20 START_CKPT=model_12600 ITERS=1500 \
  bash scripts/run_go2_loop_cycle.sh > "${REPO}/go2_overnight_phaseA.log" 2>&1
RUN_A=$(ls -1t "${EXP}" | grep '^2026-' | head -1)
say "Phase A 学習終了: ${RUN_A}"
${PY} scripts/go2_std_dump.py "${EXP}/${RUN_A}" > "${OUT}/std_${RUN_A}.tsv" 2>/dev/null

note "run: \`${RUN_A}\`"
note ""
note "| ckpt | 飛距離 | 跳躍成立 | 着地 | std | action_rate |"
note "|---|---|---|---|---|---|"
for it in $(seq 12700 100 14100); do
  [ -s "${EXP}/${RUN_A}/model_${it}.pt" ] || continue
  read -r d j l <<< "$(eval_ckpt "${EXP}/${RUN_A}" "model_${it}" "A_${it}")"
  sa=$(awk -v i="${it}" '$1==i{printf "%.3f", $2}' "${OUT}/std_${RUN_A}.tsv")
  ra=$(awk -v i="${it}" '$1==i{printf "%.1f", $3}' "${OUT}/std_${RUN_A}.tsv")
  say "  A model_${it}: ${d} m  成立${j}%  着地${l}%  std ${sa}  action_rate ${ra}"
  note "| \`model_${it}\` | ${d} m | ${j}% | ${l}% | ${sa} | ${ra} |"
  if awk -v a="${d}" -v b="${RECORD}" 'BEGIN{exit !(a>b)}'; then
    RECORD=${d}; RECORD_CKPT="model_${it}"; RECORD_RUN=${RUN_A}
  fi
done
say "Phase A 最良: ${RECORD} m (${RECORD_RUN}/${RECORD_CKPT})"
note ""
note "**Phase A 最良: ${RECORD} m（\`${RECORD_RUN}/${RECORD_CKPT}\`）**"
note ""
printf 'PHASE=A_done\nRECORD=%s\nRECORD_CKPT=%s\nRECORD_RUN=%s\n' "${RECORD}" "${RECORD_CKPT}" "${RECORD_RUN}" > "${STATE}"

# ========================= Phase B: 停止条件つきサイクル =========================
note "## Phase B: 収穫帯の起点から 300 iter サイクル（3連続で更新なしなら終了）"
note ""
note "| # | 起点 | seed | 最良 ckpt | 飛距離 | 成立 | 着地 | 記録更新 |"
note "|---|---|---|---|---|---|---|---|"
MISS=0
for c in $(seq 1 ${MAX_CYCLES}); do
  [ "$(date +%s)" -lt "${DEADLINE}" ] || { say "時間切れ、終了"; note ""; note "_時間切れで終了_"; break; }
  # 収穫帯(std 0.27-0.34)で、記録に最も近い ckpt を起点にする
  SRC=$(awk '$2>=0.27 && $2<=0.34 {print $1}' "${OUT}/std_${RECORD_RUN}.tsv" | awk -v r="${RECORD_CKPT#model_}" '
    {d=($1>r)?$1-r:r-$1; if(best==""||d<bd){bd=d;best=$1}} END{print best}')
  [ -n "${SRC}" ] || { say "収穫帯の起点が無い、終了"; note ""; note "_収穫帯(std 0.27-0.34)の起点が見つからず終了_"; break; }
  SEED=$((100 + c))
  say "サイクル${c}: 起点 ${RECORD_RUN}/model_${SRC} (seed ${SEED})"
  START_RUN=${RECORD_RUN} START_CKPT=model_${SRC} ITERS=300 SEED=${SEED} \
    bash scripts/run_go2_loop_cycle.sh > "${REPO}/go2_overnight_c${c}.log" 2>&1
  RUN_B=$(ls -1t "${EXP}" | grep '^2026-' | head -1)
  ${PY} scripts/go2_std_dump.py "${EXP}/${RUN_B}" > "${OUT}/std_${RUN_B}.tsv" 2>/dev/null
  BEST_D=0; BEST_C=""; BEST_J=0; BEST_L=0
  for off in 100 150 200 250 300; do
    it=$((SRC + off))
    [ -s "${EXP}/${RUN_B}/model_${it}.pt" ] || continue
    read -r d j l <<< "$(eval_ckpt "${EXP}/${RUN_B}" "model_${it}" "c${c}_${it}")"
    say "  c${c} model_${it}: ${d} m  成立${j}%  着地${l}%"
    if awk -v a="${d}" -v b="${BEST_D}" 'BEGIN{exit !(a>b)}'; then BEST_D=${d}; BEST_C="model_${it}"; BEST_J=${j}; BEST_L=${l}; fi
  done
  if awk -v a="${BEST_D}" -v b="${RECORD}" 'BEGIN{exit !(a>b)}'; then
    RECORD=${BEST_D}; RECORD_CKPT=${BEST_C}; RECORD_RUN=${RUN_B}; MISS=0; UPD="**更新**"
  else
    MISS=$((MISS+1)); UPD="なし (${MISS}/3)"
  fi
  say "サイクル${c} 最良 ${BEST_D} m / 記録 ${RECORD} m / 空振り ${MISS}"
  note "| ${c} | \`${RUN_B%%_*}…/model_${SRC}\` | ${SEED} | \`${BEST_C}\` | ${BEST_D} m | ${BEST_J}% | ${BEST_L}% | ${UPD} |"
  printf 'PHASE=B\nCYCLE=%s\nRECORD=%s\nRECORD_CKPT=%s\nRECORD_RUN=%s\nMISS=%s\n' \
    "${c}" "${RECORD}" "${RECORD_CKPT}" "${RECORD_RUN}" "${MISS}" > "${STATE}"
  [ "${MISS}" -lt 3 ] || { say "3連続空振り、停止条件で終了"; note ""; note "_3サイクル連続で記録更新なし → 停止条件により終了_"; break; }
done

note ""
note "## 結果"
note ""
note "**sim.dt=0.002 での最良: ${RECORD} m（\`${RECORD_RUN}/${RECORD_CKPT}\`）**"
note ""
note "比較: sim.dt=0.005 の記録は 1.977 m（\`2026-09-15_18-53-20/model_12700\`）。"
note "ただし後者は踏切仕事の26%を実機に無いトルクで作っており、2つのシミュレータが1.81倍食い違う条件下の測定。"
note ""
note "終了 $(ts)。"
say "全終了: 記録 ${RECORD} m (${RECORD_RUN}/${RECORD_CKPT})"
