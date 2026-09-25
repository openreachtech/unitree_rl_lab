#!/usr/bin/env bash
# 物理刻みを直して学習し直す（2026-09-16）。
#
# 問い: sim.dt=0.005 は収束値より 29% 低い踏切しか作れない（eval_vz_gap/）。
#       刻みを 0.002 にして学習し直すと、学習結果そのものも良くなるのか。
# 手: 記録個体と同じ起点・同じ報酬・同じ iter 数で、**刻みだけ**変えて 300 iter resume。
#     起点 2026-09-15_18-53-20/model_12600 (std 0.338 = 収穫帯)。
#     比較対象は同じ起点・同じ設定で刻み 0.005 のまま回した 2026-09-15_18-53-20 の続き
#     （＝記録個体 model_12700 が出た run そのもの）。
# 評価: mujoco（素のまま、deploy の step_dt 0.02 は不変）と Isaac 側トレースの両方。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
EXP=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1
SCENE=/home/tanaka/isaacsim/unitree_mujoco/unitree_robots/go2/scene_flat.xml
MJPY=/home/tanaka/isaacsim/anaguma/venv/bin/python
OUT=${REPO}/eval_simdt002
cd "${REPO}"; export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}" "${OUT}"
export SIM_DT=0.002
ts() { date '+%Y-%m-%d %H:%M:%S'; }

echo "=== [$(ts)] 学習開始 sim.dt=0.002, 起点 model_12600, +300 iter ==="
START_RUN=2026-09-15_18-53-20 START_CKPT=model_12600 ITERS=300 \
  bash scripts/run_go2_loop_cycle.sh > "${REPO}/go2_simdt002_train.log" 2>&1
echo "=== [$(ts)] 学習終了 ==="

RUN=$(ls -1t "${EXP}" | grep '^2026-' | head -1)
echo "=== 新 run: ${RUN} ==="
grep -m1 "sim.dt=" "${REPO}/go2_simdt002_train.log" || echo "!!! sim.dt のログが無い"

for ckpt in model_12700 model_12750 model_12800 model_12850; do
  dir="${EXP}/${RUN}"
  [ -s "${dir}/${ckpt}.pt" ] || { echo "!!! ${ckpt}.pt 無し"; continue; }
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
  "${MJPY}" scripts/mujoco_jump_eval.py --run "${dir}" --policy "${onnx}" \
    --scene "${SCENE}" --trials 40 --vx 2.8 > "${OUT}/mj_${ckpt}.log" 2>&1
  printf "mujoco %s  " "${ckpt}"
  grep -E "跳躍成立|飛距離 平均|離陸|助走 " "${OUT}/mj_${ckpt}.log" | tr '\n' ' '; echo
done

# Isaac 側も同じ判定でトレース（刻み 0.002 のまま＝学習条件と一致）
BEST=$(ls -1 "${OUT}"/*.onnx 2>/dev/null | head -1)
export TN_SPEED_SCALE_MIN=1.0 TN_SPEED_SCALE_MAX=1.0 TN_TORQUE_SCALE_MIN=1.0 TN_TORQUE_SCALE_MAX=1.0
for ckpt in model_12700 model_12750; do
  [ -s "${EXP}/${RUN}/${ckpt}.pt" ] || continue
  /home/tanaka/isaacsim/env_isaaclab/bin/python -u scripts/rsl_rl/jump_trace_isaac.py \
    --task Unitree-Go2-LongJump-v1 --headless --num_envs 64 --steps 2000 \
    --checkpoint "${EXP}/${RUN}/${ckpt}.pt" --sim_dt 0.002 \
    --label "新run ${ckpt} (sim.dt 0.002 で学習)" --out "${OUT}/isaac_${ckpt}.txt" \
    > "${OUT}/isaac_${ckpt}.log" 2>&1
  cat "${OUT}/isaac_${ckpt}.txt" 2>/dev/null || echo "!!! Isaac トレース失敗 ${ckpt}"
done
echo "=== [$(ts)] done ==="
