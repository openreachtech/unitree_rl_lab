#!/usr/bin/env bash
# Loop 20 (v19) -- Stage B from Loop 19's model_21500. Landing stability.
#
# Loop 19 solved the landing ORDER and paid for it in falls.
#   rear_first_fraction  0.580 -> 0.035-0.077   (the order term worked on the first try)
#   trunk rise           0.273 -> 0.310 m       (best of the project)
#   distance             0.693 -> 0.734 m
#   falls                0.84% -> 1.4-2.6%      <- the cost
#   take-off torque      44.86-45.45 N.m        pinned at the 45.43 N.m knee limit all run
#
# Why the falls: both landing-shaping terms charge ONE DIRECTION only, and the policy ran
#   off the far side of each. The front/rear foot gap has gone +0.279 m (front tucked,
#   lands rear-first) -> 0.000 -> **-0.10 to -0.15 m**: the front feet now arrive well
#   BELOW the rear ones, and the landing attitude has flipped from -0.173 (nose-up) to
#   +0.043..+0.080 (nose-down). Nothing charged that, so nothing stopped it.
#
# Change -- mdp.jump_landing_gear becomes two-sided with a 0.05 m deadband, and max_delta
#   0.35 -> 0.15 so the far side is actually charged rather than lost in the clip. The
#   deadband preserves the intent: a long jump SHOULD land slightly front-first, so 0 to
#   5 cm below the rear feet stays free. At the measured -0.125 m the new form reads
#   ((0.125-0.05)/0.15)^2 = 0.25, i.e. -0.75/s. Smoke: income -0.0092 against the
#   one-sided version's -0.0020.
#
# This is the same lesson twice in three loops: a one-sided penalty does not define a
#   target, it defines a direction to run in. jump_flight_pitch has the same shape and is
#   the next candidate if the attitude keeps drifting nose-down.
#
# The height side is unchanged and is expected to stay where it is -- the take-off torque
#   has been at the actuator limit for three loops.
#
# Prediction
#   1. The front/rear gap comes back into the -0.05..0.00 m band and falls return below 1%.
#   2. rear_first_fraction stays below 0.10. If it climbs back toward 0.5 the deadband is
#      too wide and the order term and the gear term are now fighting.
#   3. v_z, trunk rise and foot clearance hold (2.67 / 0.30 / 0.56). Any gain here would be
#      a surprise given the torque ceiling.
#   4. Watch unaided_landing_pitch (+0.066). If the gap comes back but the attitude stays
#      nose-down, the attitude needs its own two-sided form.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_05-39-29
LOAD_CKPT=model_21500.pt
ITERS=2500   # resume semantics: total = 21500 + 2500 = 24000
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 20 (v19) start: two-sided landing gear (deadband 0.05, max_delta 0.15), from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 20 (v19) exited with code ${EXIT} ==="
exit "${EXIT}"
