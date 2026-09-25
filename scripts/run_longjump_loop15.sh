#!/usr/bin/env bash
# Loop 15 (v14) -- Stage B from Loop 14's model_14100. ONE change: the landing detection.
#
# What Loop 14's diagnostics found
#   The "landing collapse" that ended Loops 12, 13 and 14 was never a landing failure. At
#   the collapse, unaided_timeout_fraction is 0.73-0.97 -- the jump windows were closing on
#   the 2.2 s timeout, not on a landing -- while unaided_upright_at_close is 0.93-0.99 and
#   bad_orientation is 0.01-0.06. The robot lands upright and keeps running; the condition
#   ``>= 2 feet in contact at the same instant`` simply stops being satisfiable when it
#   runs out of a jump at 1.3 m/s putting its feet down one at a time. Loop 5 already had
#   to relax this from 4 to 2 for exactly this reason.
#
#   It was not only a broken metric. jump_command is cleared by ``landed | timed_out``, so
#   a timing-out window keeps the policy in jump mode for 2.2 s instead of ~0.65 s, while
#   the deployed controller drops the bit after a fixed deploy_hold_time_s. Training and
#   deploy had been drifting apart at exactly the moment of landing -- which is the shape
#   of tanaka's "転んだり着地できたりって感じ".
#
# Change: min_feet_down_for_landing 2 -> 1. Smoke result, on Loop 14's own policy with no
#   retraining at all: unaided_timeout_fraction 0.86 -> 0.0000 and the window length lands
#   at 0.649 s. deploy_hold_time_s 0.70 -> 0.65 to match it.
#
# Nothing else changes. jump_takeoff_apex (9.3) still drives height and is still at 0.67 of
# its target; jump_landing_gear (-3.0) still closes the front/rear foot gap, which went
# 0.279 -> 0.103 m in Loop 14.
#
# Prediction
#   1. unaided_timeout_fraction stays near 0 and success_rate is usable again (> 0.9).
#      If success_rate is still low with timeouts at 0, then landings really are failing
#      and Loops 12-14 were right for the wrong reason.
#   2. Foot clearance passes Loop 14's 0.309 m. The height term never stopped working --
#      the run just kept ending its windows in a state that fed the policy 2.2 s of
#      jump-mode observations it will never see on the robot.
#   3. unaided_rear_first_fraction finally moves off 1.00. The gap is down to 0.103 m and
#      the landing-gear term keeps pulling; the order should flip once it crosses zero.
#   4. unaided_window_length stays near 0.65 s. If it drifts, deploy_hold_time_s has to be
#      re-derived from it before the next mujoco test -- that is what this metric is for.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_01-58-36
LOAD_CKPT=model_14100.pt
ITERS=2000   # resume semantics: total = 14100 + 2000 = 16100
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 15 (v14) start: landing detection >=2 feet -> >=1 foot, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 15 (v14) exited with code ${EXIT} ==="
exit "${EXIT}"
