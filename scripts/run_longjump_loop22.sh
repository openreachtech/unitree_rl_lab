#!/usr/bin/env bash
# Loop 22 (v21) -- Stage B from Loop 20's model_22100, which is still the best policy of
# the project. Loop 21 regressed and its own prediction named the wrong culprit.
#
# Loop 21 result: the asymmetric landing band did move the geometry into place (front/rear
#   gap -0.10 -> -0.02..-0.07 m, inside the target band) and rear_first stayed at 0.04-0.15.
#   But v_z collapsed 2.70 -> 2.05, foot clearance 0.55 -> 0.35, trunk 0.29 -> 0.22, and
#   falls went to 2.4-9%.
#
# The hypothesis written into Loop 21's prediction was "the landing budget as a whole is
#   too large". **Measured, it is not.** At the healthy operating point the entire landing
#   side nets +0.066 (feet_first +0.137 against gear -0.018, order -0.011, flight_pitch
#   -0.041, trunk_clearance -0.002), against jump_takeoff_apex +1.77 and upright +1.73.
#   The landing terms are ~4% of the budget; they cannot squeeze anything.
#
# What actually regressed is the height term's own quantity: jump_takeoff_apex's income
#   fell 1.77 -> 1.05 because v_z fell. The take-off torque has been pinned at the 45.43
#   N.m knee limit since Loop 18, so **the reward was still pulling toward a vertical
#   speed the hardware cannot produce** -- target 0.45 m of ballistic apex against the
#   0.372 m that v_z = 2.70 m/s buys. A live gradient into an unreachable region does not
#   make the robot jump higher; it makes it wander off the ceiling looking for a way, and
#   two loops running it did exactly that.
#
# Change: point both height terms AT the ceiling instead of past it.
#   jump_takeoff_apex  target 0.45 -> 0.37 m, weight 30.0 -> 25.0
#   jump_height        target 0.38 -> 0.30 m, weight 7.5 -> 4.7
#   Both are income-neutral at the operating point (30*0.826 = 24.8 vs 25*1.0; 7.5*0.618 =
#   4.64 vs 4.7*0.984). The terms saturate exactly where the machine does, so they stop
#   pushing and start HOLDING -- flat at the limit, with a restoring pull the moment v_z
#   drops below it. This is the one case where saturation is the design rather than the
#   bug (contrast Loops 11, 16 and 18, where the target was still reachable).
#
# Prediction
#   1. v_z holds at 2.65-2.70 for the whole run instead of drifting down. That is the
#      whole point of the change and the thing Loops 20 and 21 both failed at.
#   2. Falls come below 1%. With the height term no longer demanding the impossible, the
#      landing terms are the only live gradient left in the jump window.
#   3. Foot clearance holds near 0.60 m and trunk rise near 0.29 m.
#   4. If v_z drifts down ANYWAY, the operating point at the actuator limit is intrinsically
#      unstable under resume, and the answer is to stop tuning and consolidate on
#      model_22100 rather than to keep spending loops.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_06-27-36
LOAD_CKPT=model_22100.pt
ITERS=2000   # resume semantics: total = 22100 + 2000 = 24100
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 22 (v21) start: height targets moved to the measured machine ceiling (apex 0.37 / trunk 0.30), from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 22 (v21) exited with code ${EXIT} ==="
exit "${EXIT}"
