#!/usr/bin/env bash
# Loop 19 (v18) -- Stage B from Loop 18's model_19100.
#
# THE HEIGHT LINE HAS HIT THE MACHINE. Measured this loop for the first time: the peak
# applied joint torque during the take-off is **45.47 N.m against the Go2's 45.43 N.m knee
# limit**. The actuators are saturated. v_z has read 2.62-2.65 across the whole of Loops 17
# and 18 while jump_takeoff_apex still had gradient (0.78 of its target) -- Loop 16's note
# said that if v_z stalled with the reward still pulling, the constraint would be
# mechanical, and it is. No reward change buys more vertical speed on this robot.
#
# Where that leaves the height line, against the targets set at Loop 12:
#   lift-off v_z    2.34 m/s target  ->  2.63 m/s   exceeded, at the actuator limit
#   trunk rise      0.28 m target    ->  0.291 m    reached
#   foot clearance  (no target)      ->  0.618 m    3.0x the 0.204 m at Loop 12
#
# So this loop is landing quality only. The height terms stay exactly as they are; they
# now serve to hold v_z at the ceiling rather than to push it.
#
# Change -- new mdp.jump_landing_order, weight -4.0. Loop 14's jump_landing_gear closed the
#   GEOMETRY (front feet 0.279 m above the rear at touchdown -> 0.000 m) and the order
#   followed it down to 0.246 at Loop 17's best, then drifted back to 0.53-0.79 through
#   Loop 18. With the geometry equalised, which end lands first is decided by swing timing
#   and nothing rewards that -- it has become a coin flip with nothing holding it.
#
#   Charged for a fixed 0.3 s after the recorded touchdown rather than while the window is
#   open: with min_feet_down_for_landing at 1 the window closes on the same step as the
#   touchdown, and a window-gated version measured -0.0003 against a rear-first rate of
#   0.82. It also sidesteps IsaacLab's reward-before-command ordering, which is what made
#   jump_landing_feet_first structurally unable to fire for all of v5 and v6.
#   Smoke: -0.0449, against jump_landing_feet_first's +0.0510.
#
# Prediction
#   1. unaided_rear_first_fraction falls below Loop 17's best of 0.246. This is the first
#      term that charges the order itself rather than a proxy for it.
#   2. unaided_peak_torque_frac stays pinned at ~45.4 N.m for the whole run. That is the
#      confirmation that the height ceiling is the machine; if it drops well below while
#      v_z holds, the take-off found a cheaper posture and height is open again.
#   3. Falls stay below 1% (model_19100 is at 0.84%) and success_rate above 0.95.
#   4. Foot clearance holds near 0.60 m. It can still improve without more v_z -- it is
#      trunk rise plus tuck -- but it must not be paid for out of the landing.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_04-50-01
LOAD_CKPT=model_19100.pt
ITERS=2500   # resume semantics: total = 19100 + 2500 = 21600
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 19 (v18) start: landing order penalty (-4.0) + take-off torque diagnostic, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 19 (v18) exited with code ${EXIT} ==="
exit "${EXIT}"
