#!/usr/bin/env bash
# Phase B 仕切り直し（2026-09-16 03:45）。
#
# Phase A の結果: 0.002 で +1500 iter 回すと std が 0.340 -> 0.928 へ単調上昇し、
# iter 13100 以降は跳躍成立 0%。最良は +100 iter の 1.543 m。
# ただし **「run は300 iterで足りる、1500は無駄」は 0.005 でも既知の性質**
# （プロジェクト_自律ループ10サイクルの結果.md の運用則3）なので、
# **この崩壊は 0.002 固有の現象とは言えない**。Phase A は「時間を与えれば馴染むか」を
# 見るつもりで、実際には既知の劣化パターンをなぞっただけだった。
#
# したがって正しい比較は「同じ手順を刻みだけ変えて回す」こと:
#   起点を 2026-09-15_18-53-20/model_12600 に固定し、seed を変えて 300 iter を引き直す。
#   これは 0.005 で記録 1.977 m を出したサイクル13/14とまったく同じ手順。
#   0.002 で同じ回数の引きをして 1.977 m に届くのかを見る。
#
# 停止: 5サイクル連続で 0.002 側の記録更新が無ければ終了 / 最大15サイクル / 締切あり。
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
SRC_RUN=2026-09-15_18-53-20     # 記録 1.977 m を出したのと同じ起点
SRC=12600
MAX_CYCLES=${MAX_CYCLES:-15}
DEADLINE=$(( $(date +%s) + ${HOURS:-5} * 3600 ))
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }
note() { echo "$*" >> "${REPORT}"; }

RECORD=1.543
RECORD_CKPT=model_12700
RECORD_RUN=2026-09-16_02-52-12

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

note ""
note "## Phase B（仕切り直し $(ts)）"
note ""
note "Phase A の \"+1500 iter で崩壊\" は **0.002 固有ではない**。\"run は300 iterで足りる、1500は無駄\" は"
note "0.005 でも既知の運用則で、Phase A は既知の劣化パターンをなぞっただけだった（設計の誤り）。"
note ""
note "正しい比較は **同じ手順を刻みだけ変えて回す**こと。起点を \`${SRC_RUN}/model_${SRC}\`（記録 1.977 m を"
note "出したのと同じ起点）に固定し、seed を変えて 300 iter を引き直す。0.005 での記録はこの手順の seed 2 で出た。"
note ""
note "| # | seed | 最良 ckpt | 飛距離 | 成立 | 着地 | 0.002側の記録 |"
note "|---|---|---|---|---|---|---|"

MISS=0
for c in $(seq 1 ${MAX_CYCLES}); do
  [ "$(date +%s)" -lt "${DEADLINE}" ] || { say "時間切れ"; note ""; note "_時間切れで終了_"; break; }
  SEED=$((200 + c))
  say "サイクル${c}: 起点 ${SRC_RUN}/model_${SRC} seed ${SEED} (sim.dt 0.002)"
  START_RUN=${SRC_RUN} START_CKPT=model_${SRC} ITERS=300 SEED=${SEED} \
    bash scripts/run_go2_loop_cycle.sh > "${REPO}/go2_pb_c${c}.log" 2>&1
  RUN_B=$(ls -1t "${EXP}" | grep '^2026-' | head -1)
  BEST_D=0; BEST_C=""; BEST_J=0; BEST_L=0
  for off in 100 150 200 250 300; do
    it=$((SRC + off))
    [ -s "${EXP}/${RUN_B}/model_${it}.pt" ] || continue
    read -r d j l <<< "$(eval_ckpt "${EXP}/${RUN_B}" "model_${it}" "pb${c}_${it}")"
    say "  c${c} model_${it}: ${d} m  成立${j}%  着地${l}%"
    if awk -v a="${d}" -v b="${BEST_D}" 'BEGIN{exit !(a>b)}'; then BEST_D=${d}; BEST_C="model_${it}"; BEST_J=${j}; BEST_L=${l}; fi
  done
  if awk -v a="${BEST_D}" -v b="${RECORD}" 'BEGIN{exit !(a>b)}'; then
    RECORD=${BEST_D}; RECORD_CKPT=${BEST_C}; RECORD_RUN=${RUN_B}; MISS=0; UPD="**${RECORD} m に更新**"
  else
    MISS=$((MISS+1)); UPD="${RECORD} m のまま (${MISS}/5)"
  fi
  say "サイクル${c} 最良 ${BEST_D} m / 0.002記録 ${RECORD} m / 空振り ${MISS}"
  note "| ${c} | ${SEED} | \`${BEST_C}\` | ${BEST_D} m | ${BEST_J}% | ${BEST_L}% | ${UPD} |"
  printf 'PHASE=B\nCYCLE=%s\nRECORD=%s\nRECORD_CKPT=%s\nRECORD_RUN=%s\nMISS=%s\n' \
    "${c}" "${RECORD}" "${RECORD_CKPT}" "${RECORD_RUN}" "${MISS}" > "${STATE}"
  [ "${MISS}" -lt 5 ] || { say "5連続空振りで終了"; note ""; note "_5サイクル連続で更新なし → 停止条件により終了_"; break; }
done

note ""
note "## 結果"
note ""
note "| 条件 | 最良 | ckpt |"
note "|---|---|---|"
note "| sim.dt **0.005**（従来） | **1.977 m** | \`2026-09-15_18-53-20/model_12700\` |"
note "| sim.dt **0.002**（解像された物理） | **${RECORD} m** | \`${RECORD_RUN}/${RECORD_CKPT}\` |"
note ""
note "0.005 の記録は踏切仕事の26%を実機に無いトルクで作っており、2つのシミュレータが1.81倍食い違う条件下の測定。"
note "0.002 側は2つのシミュレータが一致する（比 1.00〜1.14）。"
note ""
note "終了 $(ts)。"
say "Phase B 終了: 0.002 記録 ${RECORD} m (${RECORD_RUN}/${RECORD_CKPT})"
