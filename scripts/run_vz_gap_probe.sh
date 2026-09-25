#!/usr/bin/env bash
# v_z ギャップの原因候補を Isaac 内の単一変数実験で潰す（2026-09-16）。
# 基準: speed_scale=1.0 / torque_scale=1.0 / sim.dt=0.005 で v_z 1.379、膝ピーク 13.6 rad/s。
# mujoco は同じ判定で v_z 2.18、膝ピーク 32.8 rad/s（制御レート間引き）。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
CKPT=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1/2026-09-15_18-53-20/model_12700.pt
OUT=${REPO}/eval_vz_gap
cd "${REPO}"; export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}" "${OUT}"
run() {
  local tag=$1 label=$2; shift 2
  echo "=== $(date '+%H:%M:%S') ${label} ==="
  /home/tanaka/isaacsim/env_isaaclab/bin/python -u scripts/rsl_rl/jump_trace_isaac.py \
    --task Unitree-Go2-LongJump-v1 --headless --num_envs 64 --steps 2000 \
    --checkpoint "${CKPT}" --label "${label}" --out "${OUT}/${tag}.txt" "$@" \
    > "${OUT}/${tag}.log" 2>&1
  cat "${OUT}/${tag}.txt" 2>/dev/null || echo "!!! 失敗: ${OUT}/${tag}.log"
}
export TN_SPEED_SCALE_MIN=1.0 TN_SPEED_SCALE_MAX=1.0
TN_TORQUE_SCALE_MIN=1.2 TN_TORQUE_SCALE_MAX=1.2 run torque12 "トルク上限 +20% (mujocoのctrlrangeに寄せる)"
TN_TORQUE_SCALE_MIN=1.0 TN_TORQUE_SCALE_MAX=1.0 run simdt002 "物理刻み 0.005 -> 0.002 (mujocoと同じ)" --sim_dt 0.002
TN_TORQUE_SCALE_MIN=1.2 TN_TORQUE_SCALE_MAX=1.2 run both "両方"  --sim_dt 0.002
echo "=== $(date '+%H:%M:%S') done ==="
