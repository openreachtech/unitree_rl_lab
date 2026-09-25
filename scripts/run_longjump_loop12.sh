#!/usr/bin/env bash
# Loop 12 (v11) -- Stage B, resumed from Loop 11's model_11100 (the checkpoint verified
# in mujoco on 2026-09-05). THE OBJECTIVE CHANGES: apex height, not distance.
#
# Why the objective changes
#   The company asked for height, and height is also the number that has not moved:
#   0.098 -> 0.119 -> 0.103 -> 0.116 m across v7..v10 against a 0.28 m goal, while
#   distance went 0.285 -> 0.359 -> 0.640 -> 0.735 m. Loops 9, 10 and 11 each moved a
#   reward weight on one factor of 2*v_x*v_z/g and each ended with the split v_x/v_z
#   unchanged at 1.66-1.68; only the take-off ENERGY grew. Paying for the product always
#   buys the cheaper factor, and at a 2.1 m/s approach that is v_x, every time.
#
# Change 1 -- reward: jump_takeoff_distance (9.0 -> 0.0) replaced by jump_takeoff_apex
#   (9.3), which scores v_z^2/(2g) against the 0.28 m goal, dense in the jump window and
#   gated on real_jump. Weight 9.3 is an income-neutral handover at the Loop 11 end
#   state: 9.0 * (0.735/2.06) = 3.21 out, 9.3 * (0.0968/0.28) = 3.22 in. Removing v_x
#   from the objective is the point, not a side effect -- the approach is held by the
#   velocity-tracking reward against the U(2.8, 3.3) command, not by this term, and the
#   smoke confirms it (unaided v_x stayed 2.15-2.25 with the distance term at zero).
#   Raising jump_height's weight instead was rejected for the third time: it scores the
#   trunk's measured rise, which can be bought by braking or by extending the legs, and
#   its Gaussian pull weakens as the target approaches, where v_z^2 keeps growing.
#
# Change 2 -- take-off technique: EFGCL assist back on (scale 0 -> 1.0), which is the
#   lever Loop 11's own post-mortem named. Three loops of reward reweighting left the
#   split untouched, so the split is set by the take-off posture, and a posture is shown
#   physically rather than paid for. Two things had to be fixed first:
#
#   (a) The launch force was 1.6x its nominal size. It is sized as impulse = m*v0 over
#       assist_duration_s (0.10 s), but launch_active spans delay..delay+ramp+duration,
#       so the force is actually applied for ramp/2 + duration = 0.16 s. Measured: a
#       0.20 m setting produced a 0.40-0.61 m apex and flipped the robot on 35-41% of
#       episodes. Fixed in JumpCommand._apply_assistance.
#       This also invalidates the Loop 10 argument for switching the assist off, which
#       compared the policy's 1.62 m/s against the assist's NOMINAL 1.53 m/s; the force
#       delivered 2.46 m/s. The teacher was never weaker than the student.
#   (b) assist_apex_height_m is now calibrated by measurement, not by the formula:
#       0.10 gives a smoke-measured apex of ~0.15-0.20 m, safely under the reward's
#       0.28 m target (the v4 lesson: a teacher that reaches the target hands out full
#       marks and the policy learns nothing) and still 2x the unaided 0.067 m.
#
#   force_zero_at_step 30_000 -> 19_200 (= 800 iterations), so this run trains unaided
#   for its last 1700. Loop 9 is the counter-example: 1100 of its 1500 iterations still
#   had the assist fading and the unaided numbers fell monotonically through the rest.
#
# Prediction (checkable)
#   1. unaided_liftoff_vel_z leaves 1.378, where three loops of reward changes left it.
#      If it does NOT move once the assist is gone, the constraint is the ground contact
#      time available at a 2.1 m/s approach, and the next loop is about the approach or
#      about a longer take-off stance -- not about any reward.
#   2. unaided_max_height_gain rises above 0.119 m (v8's record, never beaten).
#   3. Distance falls from 0.735 m. That is the trade being asked for. v_x is expected to
#      hold near 2.2 (the velocity tracking still pays for it), so the loss should be
#      roughly proportional to whatever v_z gains -- if distance collapses instead, the
#      approach is not being held and that is a finding.
#   4. Watch bad_orientation: it rose 0 -> 0.070 over the 15 smoke iterations as the
#      assist perturbs a policy that has never been launched. It must come back down as
#      the assist decays; if it is still climbing past iteration 800, stop the run.
#
# Everything else identical to Loop 11: approach band U(2.8, 3.3), plane terrain,
# 25% permanently-unaided holdout so unaided_* keeps reporting the deployable policy.

set -uo pipefail

cd /home/tanaka/isaacsim/unitree_rl_lab

# Shared machine: /tmp/isaaclab/logs belongs to another user.
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"

# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate

LOAD_RUN=2026-09-05_07-03-56
LOAD_CKPT=model_11100.pt
# rsl_rl's --resume treats max_iterations as "total = resume_iter + this": 11100 -> 13600.
ITERS=2500

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "=== [$(ts)] Loop 12 (v11) start: objective distance -> apex height, EFGCL assist back on (calibrated), from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="

python scripts/rsl_rl/train.py \
  --task Unitree-Go2-LongJump-v1 \
  --headless \
  --num_envs 4096 \
  --max_iterations "${ITERS}" \
  --resume \
  --load_run "${LOAD_RUN}" \
  --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?

echo "=== [$(ts)] Loop 12 (v11) exited with code ${EXIT} ==="
exit "${EXIT}"
