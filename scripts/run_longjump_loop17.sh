#!/usr/bin/env bash
# Loop 17 (v16) -- Stage B from Loop 16's model_16600. Closes a reward exploit.
#
# Loop 16 result: best-ever up to iteration 16700, then a clean regression.
#   16600 (best): foot clearance 0.498 m, trunk 0.234, v_z 2.556, rear_first 0.321,
#                 success 0.968, timeout 0.000, window 0.808 s, falls 0.006
#   17600 (bad):  clearance 0.482 but trunk 0.208, success 0.147, timeout 0.880,
#                 window 2.085 s, airborne 1.249 s, falls 0.100
#
# The mechanism, from the per-term incomes:
#   jump_takeoff_apex   0.7527 -> 1.9309      jump_landing_feet_first  0.0543 -> 0.0011
#   jump_height         0.1364 -> 0.4059      every landing penalty stayed below 0.03
#
#   Both height terms score quantities LATCHED at the take-off -- liftoff_vel_z, and
#   max_height_gain which is a running maximum. Neither changes after the jump. But they
#   were paid every step the window stayed open, so **not landing multiplied the income
#   2.6x**. The policy did not get worse at jumping; it learned that the window is worth
#   more open than closed. Nothing in the landing budget was large enough to argue.
#
# Change: bound the payout to 0.6 s after the latched take-off (airborne time at the good
#   checkpoint is 0.52 s, so the flight is fully covered). Extending the window is now
#   worth exactly nothing. Measured on model_16600 with no retraining, the bound cuts the
#   apex income to 43% of before, so the weights go 15.0 -> 30.0 and 2.5 -> 5.0, restoring
#   about 85% -- deliberately not 100%, since Loop 16 showed this side of the budget was
#   already loud enough to drown every landing term.
#
# This is the same family as the seven "name vs measured" errors, one level up: not a
# metric measuring the wrong thing, but a correctly-measured quantity being PAID on a
# schedule that has nothing to do with it.
#
# Prediction
#   1. unaided_window_length stays near 0.8 s and timeout_fraction near 0 for the whole
#      run. That is the exploit being closed; if the window grows again the bound leaks.
#   2. v_z passes 2.556 and foot clearance passes 0.506 m (the Loop 16 peak).
#   3. success_rate holds above 0.95 and falls stay below 0.01.
#   4. rear_first_fraction keeps falling below 0.321.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_03-12-33
LOAD_CKPT=model_16600.pt
ITERS=2500   # resume semantics: total = 16600 + 2500 = 19100
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 17 (v16) start: bound dense height payout to 0.6 s after take-off, weights 30.0 / 5.0, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 17 (v16) exited with code ${EXIT} ==="
exit "${EXIT}"
