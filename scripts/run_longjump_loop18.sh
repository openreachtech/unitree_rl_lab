#!/usr/bin/env bash
# Loop 18 (v17) -- Stage B from Loop 17's model_17600.
#
# Loop 17 closed the Loop 16 exploit and held for the whole run: window_length 0.85 s and
# timeout_fraction 0.000 from start to finish, no regression. Best (model_17600):
#   foot clearance 0.573 m   trunk rise 0.265 m   v_z 2.623   distance 0.643 m
#   rear_first 0.246         success 0.958        falls 1.9%
#
# **Both original height targets are now met.** v_z 2.65 against the 2.34 m/s the 0.28 m
# apex needs, and a trunk rise of 0.288 m at the end of the run against the 0.28 m goal
# taken from the reference paper. Foot clearance -- what a person watching actually sees
# -- is 0.57 m, 2.8x what it was when the height line started (0.204 m at Loop 12).
#
# Change 1 (height) -- jump_height: target 0.28 -> 0.38 m, weight 5.0 -> 7.5. At a 0.288 m
#   trunk rise the Gaussian reads 0.997: a flat +5.0 with no gradient. That is the third
#   time a term has died this way (jump_takeoff_speed in Loop 11, jump_takeoff_apex nearly
#   in Loop 16), and by now suspecting saturation is the first move, not the last. 0.38
#   restores it to 0.653 and the weight keeps the handover neutral.
#
# Change 2 (landing) -- jump_landing_feet_first: weight 1.5 -> 4.0. Falls have crept from
#   0.4% (Loop 15, clearance 0.43 m) to 1.9% (Loop 17, 0.57 m) as the jump grew, and the
#   whole landing budget is under 3% of the height budget: landing_feet_first +0.050,
#   landing_gear -0.002 (it already closed the front/rear gap, so there is nothing left
#   for it to charge), trunk_clearance -0.002, flight_pitch -0.027, against apex 1.77.
#   This is the only landing term with an income worth scaling.
#
# Change 3 (deploy) -- deploy_hold_time_s 0.76 -> 0.85, re-derived from
#   unaided_window_length (0.850-0.867 s across Loop 17). No effect on training; it is the
#   number that decides whether the next mujoco test measures the policy or the handoff.
#
# Prediction
#   1. Trunk rise passes 0.288 m and foot clearance passes 0.580 m (the Loop 17 peak).
#   2. Falls come back below 1%. If they do not, the landing budget is not the constraint
#      and the next loop has to look at what the landing actually does -- the front/rear
#      gap is already zero, so the remaining failure is in timing rather than geometry.
#   3. window_length stays near 0.85 s and timeout_fraction at 0 (the Loop 16 exploit
#      stays closed under a re-weighted budget).
#   4. rear_first_fraction: Loop 17 reached 0.246 at its best but drifted back to 0.53 by
#      the end, with the front/rear gap at ~0 throughout. With the geometry equalised the
#      order is now decided by swing timing, which nothing rewards -- watch whether it is
#      simply a coin flip.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_04-01-35
LOAD_CKPT=model_17600.pt
ITERS=2500   # resume semantics: total = 17600 + 2500 = 20100
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 18 (v17) start: jump_height target 0.38 (w 7.5), landing_feet_first w 4.0, deploy hold 0.85, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 18 (v17) exited with code ${EXIT} ==="
exit "${EXIT}"
