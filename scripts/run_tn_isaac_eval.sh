#!/usr/bin/env bash
# 記録個体を「実機相当(speed_scale 1.0)」「mujoco相当(2.5)」「学習時の分布(0.8-2.5)」で測る。
# Isaac 内の対照実験。トルク側は 1.0 に固定し、T-N 曲線の速度側だけを動かす。
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
CKPT=${REPO}/logs/rsl_rl/unitree_go2_longjump_v1/2026-09-15_18-53-20/model_12700.pt
OUT=${REPO}/eval_tn_isaac
cd "${REPO}"
export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}" "${OUT}"
export TN_TORQUE_SCALE_MIN=1.0 TN_TORQUE_SCALE_MAX=1.0
run() {
  local tag=$1 lo=$2 hi=$3 label=$4
  export TN_SPEED_SCALE_MIN=$lo TN_SPEED_SCALE_MAX=$hi
  echo "=== $(date '+%H:%M:%S') ${label} (speed_scale ${lo}-${hi}) ==="
  /home/tanaka/isaacsim/env_isaaclab/bin/python -u scripts/rsl_rl/eval_tn_isaac.py \
    --task Unitree-Go2-LongJump-v1 --headless --num_envs 256 --steps 2000 \
    --checkpoint "${CKPT}" --label "${label}" --out "${OUT}/${tag}.txt" \
    > "${OUT}/${tag}.log" 2>&1
  cat "${OUT}/${tag}.txt" 2>/dev/null || echo "!!! 失敗: ${OUT}/${tag}.log"
}
run real   1.0 1.0 "実機相当 X2=30rad/s"
run mjlike 2.5 2.5 "mujoco相当 X2=75rad/s"
run train  0.8 2.5 "学習時の分布"
echo "=== $(date '+%H:%M:%S') done ==="
