#!/usr/bin/env bash
# 2026-09-16 復旧用: 遅延DR run のチェックポイントを広く評価する。
#
# 経緯: `--resume` の ITERS は「追加 iter 数」なのに絶対値を渡し、1200 のつもりが
# 13900 iter 回してしまった（学習は iter 22000 まで進んだ）。ckpt は 50 iter ごとに
# 残っているので、学習量と着地率の関係を**実測で**見るために広く測る。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
RUN_DIR=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1/2026-09-16_18-28-38
SCENE=/home/tanaka/isaacsim/unitree_mujoco/unitree_robots/go2/scene_flat.xml
PY=/home/tanaka/isaacsim/env_isaaclab/bin/python
MJPY=/home/tanaka/isaacsim/anaguma/venv/bin/python
OUT=${REPO}/eval_delay_dr_20260916
REPORT=${REPO}/docs/go2_delay_dr_20260916.md
cd "${REPO}"; export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}" "${OUT}"
export SIM_DT=0.002 ACTION_DELAY_STEPS=0,2
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }
note() { echo "$*" >> "${REPORT}"; }

note "run: \`2026-09-16_18-28-38\`（iter 22000 まで回った。ITERS の渡し方を誤って +13900 iter になったため）"
note ""
note "| ckpt | 遅延1 飛距離 | 遅延1 成立 | 遅延1 着地 | 学習量(+iter) |"
note "|---|---|---|---|---|"

eval_one() {  # ckpt tag -> "距離 成立 着地"
  local ckpt=$1 tag=$2
  local onnx="${OUT}/${tag}.onnx"
  if [ ! -s "${onnx}" ]; then
    rm -f "${RUN_DIR}/exported/policy.onnx"
    ${PY} scripts/rsl_rl/play.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 1 \
      --checkpoint "${RUN_DIR}/${ckpt}.pt" > "${OUT}/export_${tag}.log" 2>&1 &
    local pid=$!
    for _ in $(seq 1 90); do sleep 5; [ -s "${RUN_DIR}/exported/policy.onnx" ] && break; done
    sleep 5; kill -9 "${pid}" 2>/dev/null; sleep 3
    [ -s "${RUN_DIR}/exported/policy.onnx" ] || { echo "0 0 0"; return 1; }
    cp "${RUN_DIR}/exported/policy.onnx" "${onnx}"
  fi
  ${MJPY} scripts/mujoco_jump_eval.py --run "${RUN_DIR}" --policy "${onnx}" --scene "${SCENE}" \
    --trials 40 --vx 2.8 --action-delay 1 > "${OUT}/mj_${tag}.log" 2>&1
  awk '/飛距離 平均/{d=$3} /跳躍成立/{for(i=1;i<=NF;i++) if($i ~ /^\(/){gsub(/[()%]/,"",$i); if(j=="") {j=$i; continue}; l=$i}} END{printf "%s %s %s\n", (d==""?0:d), (j==""?0:j), (l==""?0:l)}' "${OUT}/mj_${tag}.log"
}

BEST_L=-1; BEST_C=""; BEST_D=0
for it in ${ITER_LIST:-12900 13200 13700 14500 16000 18000 20000 22000}; do
  [ -s "${RUN_DIR}/model_${it}.pt" ] || continue
  read -r d j l <<< "$(eval_one "model_${it}" "sw_${it}")"
  say "  model_${it}: 着地 ${l}%  飛距離 ${d} m  成立 ${j}%"
  note "| \`model_${it}\` | ${d} m | ${j}% | ${l}% | +$((it - 12700)) |"
  if awk -v a="${l}" -v b="${BEST_L}" 'BEGIN{exit !(a>b)}'; then BEST_L=${l}; BEST_C="model_${it}"; BEST_D=${d}; fi
done

note ""
note "**遅延1での着地が最良: \`${BEST_C}\`（着地 ${BEST_L}% / 飛距離 ${BEST_D} m）**"
note ""
note "比較（遅延DRなしの D \`2026-09-16_05-50-08/model_12700\`）: 遅延1で着地 5%・成立 35%・飛距離 0.359 m。"
note ""
note "終了 $(ts)。"
printf 'BEST_RUN=2026-09-16_18-28-38\nBEST_CKPT=%s\nBEST_LAND=%s\nBEST_DIST=%s\n' "${BEST_C}" "${BEST_L}" "${BEST_D}" \
  > "${REPO}/.go2_delay_dr_best"
say "完了: 最良 ${BEST_C} 着地 ${BEST_L}% 飛距離 ${BEST_D} m"
