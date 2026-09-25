#!/usr/bin/env bash
# Stage A v3 -> Net2Net -> Stage B v3, with the 2026-09-02 collapse fixes applied.
#
# Why v3 exists: the v2 Stage B run completed 3000 iterations but collapsed at
# iteration ~1620 to "fall over immediately", because the reward budget made dying
# optimal (-39.7 return for running well vs +6.0 for dying) and three separate
# mechanisms let a fallen robot farm the jump rewards. Full analysis and the list of
# fixes: プロジェクト_StageB崩壊の原因分析.md.
#
# Fixes now in the code (all verified by smoke test 2026-09-02):
#   reward budget   alive +1.0 / upright +2.0 added; action_rate -0.1 -> -0.01;
#                   joint_torques -2e-4 -> -2e-5; joint_pos -0.7 -> -0.1
#   curriculum      lin_vel_cmd_levels can now DEMOTE (hysteresis 0.8 / 0.5)
#   jump command    min_air_time_s 0.05 -> 0.35; max_jump_duration_s 1.5 (new);
#                   landing requires a genuine airborne phase
#   jump rewards    sparse 250 -> 25 and only fires on a rising, upright, first
#                   lift-off; dense 2.5 -> 0.5 and only during the push-off
#   termination     orientation relaxation bounded to 1.0 s after the trigger
#   EFGCL assist    crouch pulse 150 N / 0.12 s then a ramped vertical launch sized
#                   for a 0.20 m apex, decayed by success (tak's Go2-Jump recipe)
#   observation     jump_time_encoding added (actor obs 46 -> 47, so the Net2Net
#                   transplant below rebuilds at 88 = 47 + 41 estimator outputs)
#   actuator        calf/knee 45.43 N*m peak, armature/friction matched to MuJoCo
#   domain rand.    body mass +/-10%, actuator gains +/-20%
#
# Iteration counts are deliberately lower than v2's 3000+3000 so the pair finishes
# overnight before the 2026-09-03 morning meeting.

set -uo pipefail

cd /home/tanaka/isaacsim/unitree_rl_lab

export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"

# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate

STAGE_A_ITERS=2000
STAGE_B_ITERS=2500

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "=== [$(ts)] Stage A v3 (forward-only, rebalanced rewards) FROM SCRATCH, max_iterations=${STAGE_A_ITERS} ==="

python scripts/rsl_rl/train.py \
  --task Unitree-Go2-LongJump-Base-v1 \
  --headless \
  --num_envs 4096 \
  --max_iterations "${STAGE_A_ITERS}" \
  --deploy-keyboard-commands
STAGE_A_EXIT=$?

echo "=== [$(ts)] Stage A v3 exited with code ${STAGE_A_EXIT} ==="

if [ "${STAGE_A_EXIT}" -ne 0 ]; then
  echo "!!! [$(ts)] Stage A v3 failed -- aborting, Stage B will NOT start."
  exit 1
fi

STAGE_A_RUN_DIR=$(ls -td logs/rsl_rl/unitree_go2_longjump_base_v1/*/ 2>/dev/null | head -1)
STAGE_A_CKPT=$(ls -v "${STAGE_A_RUN_DIR}"model_*.pt 2>/dev/null | tail -1)
if [ -z "${STAGE_A_CKPT}" ]; then
  echo "!!! [$(ts)] No Stage A checkpoint found in '${STAGE_A_RUN_DIR}' -- aborting."
  exit 1
fi
echo "=== [$(ts)] Stage A v3 final checkpoint: ${STAGE_A_CKPT} ==="

NET2NET_RUN_NAME="$(date '+%Y-%m-%d_%H-%M-%S')_net2net_v3"
NET2NET_OUT_DIR="logs/rsl_rl/unitree_go2_longjump_v1/${NET2NET_RUN_NAME}"

echo "=== [$(ts)] Net2Net transplant -> ${NET2NET_OUT_DIR} ==="

# The transplant does its work in the first couple of minutes and then can hang
# indefinitely inside simulation_app.close() (observed 47 min on 2026-09-02, with the
# checkpoint already correctly written -- verified by loading it). So: run it with a
# generous timeout, then judge success by whether model_0.pt exists, not by exit code.
timeout 900 python scripts/rsl_rl/longjump_net2net_transplant.py \
  --headless \
  --stage_a_checkpoint "${STAGE_A_CKPT}" \
  --output_dir "${NET2NET_OUT_DIR}"
NET2NET_EXIT=$?
echo "=== [$(ts)] Net2Net transplant returned ${NET2NET_EXIT} (124 = timed out in shutdown, which is tolerated) ==="

# Give a hung-in-shutdown process a moment to be reaped, then check the artefact.
pkill -9 -f "longjump_net2net_transplant.py" 2>/dev/null
sleep 5

if [ ! -s "${NET2NET_OUT_DIR}/model_0.pt" ]; then
  echo "!!! [$(ts)] ${NET2NET_OUT_DIR}/model_0.pt missing or empty -- aborting."
  exit 1
fi
echo "=== [$(ts)] Transplanted checkpoint present: $(ls -l "${NET2NET_OUT_DIR}/model_0.pt") ==="

echo "=== [$(ts)] Stage B v3 from ${NET2NET_RUN_NAME}, max_iterations=${STAGE_B_ITERS} ==="

python scripts/rsl_rl/train.py \
  --task Unitree-Go2-LongJump-v1 \
  --headless \
  --num_envs 4096 \
  --max_iterations "${STAGE_B_ITERS}" \
  --resume \
  --load_run "${NET2NET_RUN_NAME}" \
  --checkpoint model_0.pt \
  --deploy-keyboard-commands
STAGE_B_EXIT=$?

echo "=== [$(ts)] Stage B v3 exited with code ${STAGE_B_EXIT} ==="
echo "=== [$(ts)] Pipeline finished. ==="
exit "${STAGE_B_EXIT}"
