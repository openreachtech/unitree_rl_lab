#!/usr/bin/env bash
# Go2 long jump: Stage B from iteration 0 with the DISTANCE objective -- never done before.
#
# Why this run exists. Stage B was last started from 0 on 2026-09-03 (run 2026-09-03_01-40-31);
# every loop since is one unbroken resume chain, 30,699 iterations and 49 loops long. The
# distance reward (jump_takeoff_distance, weight 74) only entered that chain at Loop 47 =
# iteration 27,800, so today's "distance policy" is a policy that spent 16,400 iterations
# being optimised for HEIGHT with the distance objective bolted on for its last 2,900.
# The current reward set has never trained a Stage B from the start.
#
# The chain also has a documented failure mode: "resume から1800〜1900iter で崩壊", cause
# identified in Loop 26 as the action std carrying across resumes (docs/longjump_height.md).
# Loops 48 and 49 reproduced it on 2026-09-13 -- both peaked +200..400 iterations after the
# resume and were below their own starting point by +1400, Loop 49 ending at mean reward
# -60,834.
#
# Reward configuration = Loop 47's, which is the best result this project has ever measured
# on the machine (mujoco 1.929 m, 85% jump rate, 12% landing over 40 trials):
#   jump_takeoff_distance  74.0, target 2.06 m   -- the PRODUCT 2*v_x*v_z/g, not its factors.
#       Loops 8/9 rewarded a factor and the policy bought the cheap one; Loop 11 showed the
#       product moves take-off energy instead. Target stays 2.06: Loop 48 raised it to 2.75
#       and mujoco distance FELL 1.929 -> 1.511.
#   jump_takeoff_apex       0.0                  -- see the config; Loop 49's 40.0 bought 3%
#       more ballistic product and lost every landing in a 40-trial sample.
#   track_lin_vel_xy        1.5                  -- NOT Loop 44's 0.8. Dropping it released
#       the approach to 4.4 m/s and those individuals landed 24-36% of the time in mujoco.
#   EFGCL assist ON from iteration 0, fading to zero by iteration 1500. Mandatory here and
#       only here: every jump reward is gated on real_jump, so a policy that has never left
#       the ground earns nothing from any of them.
#   Domain randomisation is inherited from the Stage A base cfg (EventCfgLongJump: body mass,
#       actuator gains, torque-speed curve). Loop 29 measured it taking mujoco landings from
#       0/8 to 23/43 -- the single largest landing improvement on record -- by lowering the
#       STEADY pitch rotation in flight. It must be present from the start.
#
# Stage A start point: 2026-09-02_20-38-02/model_1999, chosen on merit over
# 2026-09-02_11-58-21/model_2999 -- episode length 975.6 vs 504.7 out of 1000 and a speed
# curriculum at 4.6 vs 3.2, which is what a 2.8-3.3 m/s approach needs.
#
# Judge on the unaided_* metrics (25% holdout gets no assist) and then on mujoco against
# model_29200 of 2026-09-13_14-19-07. A fresh run may well land below 1.929 m; 30,700
# iterations of accumulated skill is not free. This is a comparison, not a replacement.
set -uo pipefail
REPO=/home/tanaka/isaacsim/unitree_rl_lab
cd "${REPO}"
export TMPDIR="${REPO}/.tmp"; mkdir -p "${TMPDIR}"
PY=/home/tanaka/isaacsim/env_isaaclab/bin/python
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

STAGE_A_CKPT="${REPO}/logs/rsl_rl/unitree_go2_longjump_base_v1/2026-09-02_20-38-02/model_1999.pt"
NET2NET_RUN_NAME="$(date '+%Y-%m-%d_%H-%M-%S')_net2net_distance_scratch"
NET2NET_OUT_DIR="logs/rsl_rl/unitree_go2_longjump_v1/${NET2NET_RUN_NAME}"

echo "=== [$(ts)] Net2Net transplant from ${STAGE_A_CKPT} ==="
mkdir -p "${NET2NET_OUT_DIR}"
# Known quirk: the transplant can hang in simulation_app.close() with the checkpoint already
# written (15 min observed). Judged by whether model_0.pt exists, not by exit code.
timeout 900 "${PY}" scripts/rsl_rl/longjump_net2net_transplant.py --headless \
  --stage_a_checkpoint "${STAGE_A_CKPT}" --output_dir "${NET2NET_OUT_DIR}"
echo "=== [$(ts)] transplant returned $? (124 = hung in shutdown, tolerated) ==="
pkill -9 -f "longjump_net2net_transplan[t].py" 2>/dev/null
sleep 5
if [ ! -s "${NET2NET_OUT_DIR}/model_0.pt" ]; then
  echo "!!! [$(ts)] ${NET2NET_OUT_DIR}/model_0.pt missing -- aborting."
  exit 1
fi
touch "${REPO}/.go2_transplant_done"   # Anaguma's script waits on this so the two transplants do not overlap on the GPU

echo "=== [$(ts)] Go2 Stage B from scratch (distance objective, assist on): ${NET2NET_RUN_NAME}, 3000 iters, 4096 envs ==="
"${PY}" scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations 3000 --resume --load_run "${NET2NET_RUN_NAME}" \
  --checkpoint model_0.pt --deploy-keyboard-commands
echo "=== [$(ts)] Go2 Stage B from scratch exited with code $? ==="
