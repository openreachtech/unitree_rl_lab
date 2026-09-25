#!/usr/bin/env bash
# Loop 26 (v25) -- Stage B from model_23600. Reward the height a person actually sees.
#
# Where the height line stands: both quantities it has ever rewarded are finished.
#   lift-off v_z  2.73 m/s, with the knee actuators saturated at 45.43 N.m for eight loops
#   trunk rise    0.283 m against a 0.28 m goal
# Neither is what tanaka was looking at when he said the jump did not look very high. What
# a person watching sees is daylight under the feet, and that number has been MEASURED
# since Loop 13 (unaided_peak_foot_clearance) without ever being rewarded.
#
# It is also the one height quantity the actuators do not cap. model_23600 and model_22100
# have the same take-off -- v_z 2.731 vs 2.692 -- and clearances of 0.497 m and 0.603 m.
# That 0.11 m sits entirely in how the legs fold in flight, which costs the knees nothing.
#
# Change 1: new mdp.jump_foot_clearance, weight 6.0, target 0.65 m (above the 0.618 m
#   peak ever recorded), bounded to 0.6 s after the take-off like the other two dense
#   height terms -- peak_foot_clearance is a running maximum, so an unbounded payout would
#   re-open the Loop 16 exploit. Smoke: +0.153 against jump_height's +0.150.
#   The risk -- a harder tuck arriving at the landing with the legs not extended -- is what
#   jump_landing_gear already measures and charges on both sides since Loop 21, so the
#   guard is in place rather than being added alongside.
#
# Change 2: entropy_coef back to 0.002, the value that produced model_23600 (falls 0.05%,
#   v_z 2.731, success 97.4%). Loop 25's 0.005 held entropy in the target band for 1800
#   iterations but its best checkpoint was worse on every landing number.
#
# Run length 1500, not 2500. All three entropy values collapse 1800-1900 iterations after
#   resume and every best checkpoint since Loop 17 has come from the first 1500, so the run
#   is capped below the failure rather than tuned against it.
#
# Prediction
#   1. Foot clearance passes 0.618 m, the project peak, without v_z moving (it cannot).
#   2. Falls stay at or below 0.05% and rear_first at or below 0.012. If either degrades,
#      the tuck is being bought out of the landing and the clearance target is too high.
#   3. Trunk rise holds near 0.283 m -- clearance should come from the legs, not from
#      trading trunk height for it.
#   4. No collapse, because the run ends before the point where nine consecutive runs died.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_09-30-27
LOAD_CKPT=model_23600.pt
ITERS=1500   # resume semantics: total = 23600 + 1500 = 25100
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 26 (v25) start: reward peak foot clearance (w 6.0, target 0.65 m), entropy 0.002, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 26 (v25) exited with code ${EXIT} ==="
exit "${EXIT}"
