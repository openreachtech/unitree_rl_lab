#!/usr/bin/env bash
# Loop 14 (v13) -- Stage B, resumed from Loop 13's model_12400. Landing precision, so that
# height can keep going: Loops 12 and 13 both lost success_rate at the same CAPABILITY
# level (foot clearance ~0.25 m, v_z ~1.9) rather than after the same number of
# iterations, so landing is what is capping height, not the height objective.
#
# What Loop 13 established
#   The nose-up landing attitude was removed completely (landing pitch -0.173 -> -0.035)
#   and unaided_rear_first_fraction did not move off 1.00. The landing order is therefore
#   not the body's attitude -- it is where the legs are.
#
# Measured here for the first time: at the touchdown instant the front feet are 0.279 m
#   ABOVE the rear feet. The front pair is fully tucked while the rear pair arrives, which
#   is tanaka's "the front legs still think they are flying" taken literally.
#
# Change 1 -- new mdp.jump_landing_gear, weight -3.0, max_delta 0.35. Charges
#   relu(front_z - rear_z) while airborne AND DESCENDING only: on the way up a tuck is
#   correct, it is what buys the foot clearance the jump is measured by. No existing term
#   reaches this -- jump_flight_pitch is about the trunk, jump_trunk_clearance only keeps
#   the trunk off the floor, and jump_landing_feet_first does not care WHICH feet.
#   max_delta is 0.35 rather than the 0.12 of leg travel it was first sized at, because at
#   0.12 the measured 0.279 clips the term at 1.0 and it becomes a flat penalty with zero
#   gradient -- the dead-term failure Loop 11 diagnosed in jump_takeoff_speed.
#   Smoke: -0.019 against the height term's +0.249, about 8%.
#
# Change 2 -- diagnostics for the iteration-12500 wall. Loop 13 showed the collapse is a
#   capability wall, but not which clause of
#   ``landed = has_been_airborne & enough_feet_down & upright & min_air_time_elapsed``
#   stops being satisfiable. New holdout metrics: unaided_timeout_fraction,
#   unaided_upright_at_close, unaided_feet_down_at_close, unaided_front_rear_delta.
#   At the healthy model_12400 they read 0.00 / 0.881 / 0.881 / 0.279 -- so windows are
#   closing on a landing, not timing out. If the collapse is upright failing, that is a
#   tumble; if it is feet_down, the robot is not getting its legs under it in time.
#
# Height pressure is unchanged (jump_takeoff_apex 9.3 against the 0.28 m goal, currently
# at 0.672 of full and still climbing). The hypothesis of this loop is that landing is the
# binding constraint and that fixing it lets the existing height term keep working.
#
# Prediction (checkable)
#   1. unaided_front_rear_delta falls from 0.279 m, and unaided_rear_first_fraction
#      finally moves off 1.00 -- the first term aimed at the actual mechanism.
#   2. success_rate stays above 0.9 past the point where Loops 12 and 13 both broke
#      (foot clearance 0.25 m / v_z 1.9). If it breaks there anyway, the wall is not the
#      landing gear either, and the diagnostics say which clause failed.
#   3. Foot clearance passes Loop 13's 0.254 m, because the height term is unchanged and
#      the landing that was capping it is what this loop removes.
#   4. Watch unaided_liftoff_vel_x (1.17 at the start): below 0.8 the run-up is gone.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate

LOAD_RUN=2026-09-06_01-03-11
LOAD_CKPT=model_12400.pt
# rsl_rl's --resume treats max_iterations as "total = resume_iter + this": 12400 -> 14400.
ITERS=2000
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 14 (v13) start: landing gear (front feet down before touchdown) + landing diagnostics, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 14 (v13) exited with code ${EXIT} ==="
exit "${EXIT}"
