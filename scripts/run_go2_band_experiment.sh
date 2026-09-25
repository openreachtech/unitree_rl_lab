#!/usr/bin/env bash
# 助走帯を上げる実験（2026-09-16）。
#
# 問い: mujoco スイープで見えた「飛距離の山＝実測助走 3.37 m/s」は、学習分布の縁か物理限界か。
# 手: 帯を U(2.8,3.3) -> U(3.3,3.8) に上げて記録run から 300 iter resume し、
#     出てきた候補に同じ 6 点スイープをかけ直す。山が右に動けば分布の縁、動かなければ物理側。
# 起点: 2026-09-15_18-53-20/model_12600 (std 0.338 = 収穫帯、mujoco 1.921m)。
#       記録個体 model_12700 自体は std 0.372 で「0.37以上は全滅」ゾーンなので起点に使わない。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
EXP=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1
SCENE=/home/tanaka/isaacsim/unitree_mujoco/unitree_robots/go2/scene_flat.xml
MJPY=/home/tanaka/isaacsim/anaguma/venv/bin/python
OUT=${REPO}/eval_go2_band38
cd "${REPO}"
export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}" "${OUT}"
export APPROACH_SPEED_MIN_MS=3.3 APPROACH_SPEED_MS=3.8
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "=== [$(ts)] 学習開始: band U(3.3,3.8), 起点 2026-09-15_18-53-20/model_12600, +300 iter ==="
START_RUN=2026-09-15_18-53-20 START_CKPT=model_12600 ITERS=300 \
  bash scripts/run_go2_loop_cycle.sh > "${REPO}/go2_band38_train.log" 2>&1
echo "=== [$(ts)] 学習終了 ==="

RUN=$(ls -1t "${EXP}" | grep '^2026-' | head -1)
echo "=== 新 run: ${RUN} ==="
ls -1 "${EXP}/${RUN}"/model_*.pt | sort -t_ -k2 -n | tail -8

for ckpt in model_12700 model_12750 model_12800 model_12850 model_12900; do
  dir="${EXP}/${RUN}"
  [ -s "${dir}/${ckpt}.pt" ] || { echo "!!! ${ckpt}.pt 無し、skip"; continue; }
  onnx="${OUT}/${RUN}_${ckpt}.onnx"
  if [ ! -s "${onnx}" ]; then
    echo "=== [$(ts)] export ${ckpt} ==="
    rm -f "${dir}/exported/policy.onnx"
    /home/tanaka/isaacsim/env_isaaclab/bin/python scripts/rsl_rl/play.py \
      --task Unitree-Go2-LongJump-v1 --headless --num_envs 1 \
      --checkpoint "${dir}/${ckpt}.pt" > "${OUT}/export_${ckpt}.log" 2>&1 &
    pid=$!
    for _ in $(seq 1 90); do sleep 5; [ -s "${dir}/exported/policy.onnx" ] && break; done
    sleep 5; kill -9 "${pid}" 2>/dev/null; sleep 3
    [ -s "${dir}/exported/policy.onnx" ] || { echo "!!! export 失敗 ${ckpt}"; continue; }
    cp "${dir}/exported/policy.onnx" "${onnx}"
  fi
  for vx in 2.8 3.2 3.6 4.0 4.4; do
    "${MJPY}" scripts/mujoco_jump_eval.py --run "${dir}" --policy "${onnx}" \
      --scene "${SCENE}" --trials 40 --vx "${vx}" > "${OUT}/sweep_${ckpt}_vx${vx}.log" 2>&1
    printf "%s vx%-4s " "${ckpt}" "${vx}"
    grep -E "跳躍成立|飛距離 平均|離陸|助走 " "${OUT}/sweep_${ckpt}_vx${vx}.log" | tr '\n' ' '; echo
  done
done
echo "=== [$(ts)] done ==="
