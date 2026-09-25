#!/usr/bin/env bash
# Get one long-jump checkpoint ready for mujoco sim2sim and launch simulator + controller.
#
# Usage: prepare_longjump_mujoco.sh <RUN_NAME> <CKPT> <LABEL>
#   e.g. prepare_longjump_mujoco.sh 2026-09-04_02-37-20 model_7600.pt "Loop9 v8"
#
# Why each step is here (see プロジェクト_mujoco実機検証ワークフロー.md):
#   - go2_ctrl reads policy_dir/exported/policy.onnx + policy_dir/params/deploy.yaml and
#     stops at the FIRST exported/ directly under policy_dir. It does NOT pick the newest
#     run, so repointing the top-level symlinks is mandatory.
#   - play.py must do the export: it dispatches to export_actor_critic_ee for ActorCriticEE,
#     which embeds the Estimator. The generic exporter wraps policy.actor alone and emits a
#     wrong input width that OrtRunner accepts silently (feedback_actor_critic_ee_onnx_export_bug).
#   - The ONNX input width is checked against the sum of deploy.yaml observations because a
#     mismatch does not raise -- it over-reads the heap and the robot just crouches.
#   - scene_flat.xml, not the default scene.xml: the default has an 8cm step and a stair of
#     boxes at x=1.2..3.4m, which the run-up hits before any jump.
#   - TMPDIR is redirected: /tmp/isaaclab/logs is owned by another user on this shared box.
set -uo pipefail

RUN_NAME=${1:?usage: prepare_longjump_mujoco.sh <RUN_NAME> <CKPT> <LABEL>}
CKPT_NAME=${2:?usage: prepare_longjump_mujoco.sh <RUN_NAME> <CKPT> <LABEL>}
LABEL=${3:-${RUN_NAME}/${CKPT_NAME}}

REPO=/home/tanaka/isaacsim/unitree_rl_lab
EXP_DIR="${REPO}/logs/rsl_rl/unitree_go2_longjump_v1"
RUN_DIR="${EXP_DIR}/${RUN_NAME}"
CKPT="${RUN_DIR}/${CKPT_NAME}"
TAG=$(echo "${LABEL}" | tr -c 'A-Za-z0-9' '_')

cd "${REPO}"
export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate

ts() { date '+%H:%M:%S'; }
[ -s "${CKPT}" ] || { echo "!!! no checkpoint: ${CKPT}"; exit 1; }

echo "=== [$(ts)] exporting ${LABEL}: ${CKPT} ==="
python scripts/rsl_rl/play.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 1 \
  --checkpoint "${CKPT}" > "${REPO}/EXPORT_${TAG}.log" 2>&1 &
PLAY_PID=$!
for _ in $(seq 1 60); do
  sleep 5
  [ -s "${RUN_DIR}/exported/policy.onnx" ] && break
done
sleep 5; kill -9 "${PLAY_PID}" 2>/dev/null; sleep 3
[ -s "${RUN_DIR}/exported/policy.onnx" ] || { echo "!!! export failed -- see EXPORT_${TAG}.log"; exit 1; }
echo "=== [$(ts)] exported: $(ls -l "${RUN_DIR}/exported/policy.onnx") ==="

echo "=== [$(ts)] dimension cross-check ==="
python - "${RUN_DIR}" <<'PY' || exit 1
import sys, onnx, yaml
run = sys.argv[1]
m = onnx.load(f"{run}/exported/policy.onnx")
onnx_dim = [d.dim_value for d in m.graph.input[0].type.tensor_type.shape.dim][-1]
cfg = yaml.safe_load(open(f"{run}/params/deploy.yaml"))
total = 0
for name, o in cfg["observations"].items():
    n = len(o["scale"]) * o.get("history_length", 1)
    print(f"  {name:32s} {len(o['scale']):3d} x {o.get('history_length',1)} = {n}")
    total += n
print(f"  {'ONNX input':32s} = {onnx_dim}    deploy.yaml sum = {total}")
sys.exit(0 if onnx_dim == total else 1)
PY
echo "=== [$(ts)] dimensions match ==="

ln -sfn "./${RUN_NAME}/exported" "${EXP_DIR}/exported"
ln -sfn "./${RUN_NAME}/params"   "${EXP_DIR}/params"
echo "=== [$(ts)] symlinks -> $(readlink "${EXP_DIR}/exported") , $(readlink "${EXP_DIR}/params") ==="

pkill -x unitree_mujoco 2>/dev/null; pkill -x go2_ctrl 2>/dev/null; sleep 2

cd /home/tanaka/isaacsim/unitree_mujoco/simulate/build
DISPLAY=:1 setsid nohup ./unitree_mujoco -r go2 -s scene_flat.xml \
  > "${REPO}/MUJOCO_${TAG}.log" 2>&1 < /dev/null &
sleep 6

cd "${REPO}/deploy/robots/go2/build"
DISPLAY=:1 setsid nohup gnome-terminal --title="go2_ctrl -- ${LABEL} (${CKPT_NAME})" --geometry=100x30 -- \
  bash -c "./go2_ctrl --network lo; echo; echo '=== 終了。Enterで閉じる ==='; read" \
  > "${REPO}/GO2_CTRL_${TAG}.log" 2>&1 < /dev/null &
sleep 6

echo "=== [$(ts)] mujoco: $(pgrep -x unitree_mujoco | tr '\n' ' ')/ go2_ctrl: $(pgrep -x go2_ctrl | tr '\n' ' ') ==="
echo "=== [$(ts)] READY (${LABEL}) -- go2_ctrlウィンドウで  1 -> Enter -> f -> j -> Space ==="
