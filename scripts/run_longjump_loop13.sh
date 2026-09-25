#!/usr/bin/env bash
# Loop 13 (v12) -- Stage B, resumed from Loop 12's model_12200. Height again, plus the
# landing attitude that Loop 12's mujoco run exposed.
#
# What Loop 12 established
#   Height 0.116 -> 0.174 m and v_z 1.378 -> 1.893 m/s, distance held at 0.743 m. It also
#   overturned Loop 11's conclusion: the v_x/v_z split is NOT fixed by take-off posture,
#   it went 1.65 -> 0.69 the moment v_x left the objective. And landings broke after
#   iteration 12500 (success_rate 0.945 -> 0.053).
#
# What the mujoco run added
#   "The front legs still think they are flying, the rear legs touch down at once, and the
#   front legs get slammed into the ground." Instrumenting that gives
#   unaided_rear_first_fraction = 1.00 and a landing attitude of -0.173 (about 10 degrees
#   nose-up). EVERY real jump lands rear-first.
#
#   Ungated, the same metric read 0.124: at a 2 m/s approach the first contact inside the
#   jump window is a running stride, not the jump's landing. Ninth instance of the failure
#   mode in プロジェクト_報酬項の名前と実測対象のズレ7例.md -- and the first where the wrong
#   number would have said "nothing to fix here".
#
# Change 1 (height) -- track_lin_vel_xy is switched off inside the jump window, the same
#   treatment lin_vel_z_l2 got in v4, applied to the other side of the same trade. The
#   policy was being paid to hold its forward speed at the exact moment the task needs it
#   converted into vertical speed. Measured: gating removes only 2.5% of the term's
#   income, because a robot converting speed is already tracking badly -- the point is the
#   gradient (dR/de = -4R pulling every m/s back), not the level.
#
# Change 2 (landing) -- new mdp.jump_flight_pitch, weight -4.0: a bounded penalty on the
#   nose-up attitude inside the jump window, charging only relu(-g_x) because that is the
#   sign the measurement says lands rear-first. At the measured -0.173 it costs 0.98/s
#   against the ~6.0/s the height term pays there, about 16%. flat_orientation_l2 does not
#   cover this -- it is symmetric, and at -0.0035 measured it has never been large enough
#   to shape a take-off.
#
#   The two changes are expected to pull the SAME way: a rear-dominant push-off is what
#   makes the nose-up rotation, and a symmetric one puts more of the same energy into
#   vertical translation.
#
# Change 3 (measurement) -- four new metrics, all holdout-only: peak_foot_clearance
#   (0.204 m measured -- what a person watching calls the height of the jump, and higher
#   than the 0.143 m trunk rise because the robot tucks), peak_pitch_rate (1.82 rad/s),
#   landing_pitch, rear_first_fraction.
#
# EFGCL assist stays OFF: the teacher has been overtaken again (assist apex 0.15-0.20 m
# measured against the policy's own 0.174 m), and the room between that and the reward's
# 0.28 m target is too small for it to teach anything.
#
# Prediction (checkable)
#   1. unaided_rear_first_fraction falls from 1.00, and unaided_landing_pitch from -0.173
#      toward 0.
#   2. unaided_liftoff_vel_z passes 1.893 and unaided_peak_foot_clearance passes 0.204 m.
#   3. Watch unaided_liftoff_vel_x. It is 1.23 at the start; below about 0.8 the run-up has
#      been abandoned and this is a standing vertical jump, which is a different task and
#      is covered by another team. That is the risk change 1 creates.
#   4. success_rate should hold above 0.9 for longer than Loop 12's 1400 iterations if the
#      rear-first landing was what broke it. If it collapses at the same point anyway, the
#      cause is elsewhere and the next loop is about the landing itself.

set -uo pipefail

cd /home/tanaka/isaacsim/unitree_rl_lab

# Shared machine: /tmp/isaaclab/logs belongs to another user.
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"

# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate

LOAD_RUN=2026-09-05_23-28-22
LOAD_CKPT=model_12200.pt
# rsl_rl's --resume treats max_iterations as "total = resume_iter + this": 12200 -> 14200.
ITERS=2000

ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }

echo "=== [$(ts)] Loop 13 (v12) start: velocity tracking gated off in the jump window + nose-up landing penalty, assist off, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="

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

echo "=== [$(ts)] Loop 13 (v12) exited with code ${EXIT} ==="
exit "${EXIT}"
