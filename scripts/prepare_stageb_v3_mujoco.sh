#!/usr/bin/env bash
# Wait for Stage B v3 training to finish, then get mujoco ready to test it:
# export the final checkpoint to ONNX, repoint the top-level exported/params
# symlinks go2_ctrl actually reads, and launch the simulator + controller.
#
# go2_ctrl reads policy_dir/exported/policy.onnx and policy_dir/params/deploy.yaml,
# and param::parser_policy_dir stops at the FIRST exported/ it finds directly under
# policy_dir -- it does NOT pick the newest run automatically. So the symlinks below
# are mandatory, not cosmetic (see プロジェクト_mujoco実機検証ワークフロー.md).

set -uo pipefail

REPO=/home/tanaka/isaacsim/unitree_rl_lab
EXP_DIR="${REPO}/logs/rsl_rl/unitree_go2_longjump_v1"
TRAIN_PID=${1:?usage: prepare_stageb_v3_mujoco.sh <train_pid>}

cd "${REPO}"
export TMPDIR="${REPO}/.tmp"
mkdir -p "${TMPDIR}"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "=== [$(ts)] waiting for training pid ${TRAIN_PID} to exit ==="
while kill -0 "${TRAIN_PID}" 2>/dev/null; do
  sleep 10
done
echo "=== [$(ts)] training process gone ==="

# Newest timestamped run directory, excluding the net2net transplant drops.
RUN_DIR=$(ls -td "${EXP_DIR}"/2*/ 2>/dev/null | grep -v net2net | head -1)
if [ -z "${RUN_DIR}" ]; then
  echo "!!! [$(ts)] no Stage B run directory found under ${EXP_DIR}"
  exit 1
fi
CKPT=$(ls -v "${RUN_DIR}"model_*.pt 2>/dev/null | tail -1)
if [ -z "${CKPT}" ]; then
  echo "!!! [$(ts)] no checkpoint in ${RUN_DIR}"
  exit 1
fi
RUN_NAME=$(basename "${RUN_DIR%/}")
echo "=== [$(ts)] exporting ${CKPT} ==="

# play.py exports early then enters an endless render loop, so start it detached and
# kill it by pid once exported/policy.onnx appears (RobotPlayEnvCfgLongJump also sets
# initial_assist_scale = 0.0, i.e. this evaluates the policy unaided).
python scripts/rsl_rl/play.py \
  --task Unitree-Go2-LongJump-v1 \
  --headless \
  --num_envs 1 \
  --checkpoint "${CKPT}" \
  > "${REPO}/EXPORT_stageB_v3.log" 2>&1 &
PLAY_PID=$!

for _ in $(seq 1 60); do
  sleep 5
  if [ -s "${RUN_DIR}exported/policy.onnx" ]; then break; fi
done
sleep 5
kill -9 "${PLAY_PID}" 2>/dev/null
sleep 3

if [ ! -s "${RUN_DIR}exported/policy.onnx" ]; then
  echo "!!! [$(ts)] export failed -- see EXPORT_stageB_v3.log"
  exit 1
fi
echo "=== [$(ts)] exported: $(ls -l "${RUN_DIR}exported/policy.onnx") ==="

ln -sfn "./${RUN_NAME}/exported" "${EXP_DIR}/exported"
ln -sfn "./${RUN_NAME}/params" "${EXP_DIR}/params"
echo "=== [$(ts)] symlinks -> $(readlink "${EXP_DIR}/exported") , $(readlink "${EXP_DIR}/params") ==="

# Fresh simulator + controller on tanaka's NoMachine display. pkill -x (exact process
# name) rather than -f: a -f pattern also matches this script's own command line.
pkill -x unitree_mujoco 2>/dev/null
pkill -x go2_ctrl 2>/dev/null
sleep 2

cd /home/tanaka/isaacsim/unitree_mujoco/simulate/build
DISPLAY=:1 setsid nohup ./unitree_mujoco > "${REPO}/MUJOCO_stageB_v3.log" 2>&1 < /dev/null &
sleep 6

cd "${REPO}/deploy/robots/go2/build"
DISPLAY=:1 setsid nohup gnome-terminal --title="go2_ctrl (Stage B v3 最終)" -- \
  bash -c "./go2_ctrl --network lo; echo; echo '=== 終了。Enterで閉じる ==='; read" \
  > "${REPO}/GO2_CTRL_stageB_v3.log" 2>&1 < /dev/null &
sleep 6

echo "=== [$(ts)] mujoco: $(pgrep -x unitree_mujoco | tr '\n' ' ') / go2_ctrl: $(pgrep -x go2_ctrl | tr '\n' ' ') ==="
echo "=== [$(ts)] READY -- operate in the go2_ctrl window: 1 -> Enter -> f -> j -> Space ==="
