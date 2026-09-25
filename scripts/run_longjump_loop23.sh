#!/usr/bin/env bash
# Loop 23 (v22) -- Stage B from Loop 20's model_22100. NOT a reward change.
#
# Loop 22 answered its own prediction 4. Pointing the height terms at the measured machine
#   ceiling did buy the best landing of the height line -- model_22700 at **0.81% falls**,
#   the lowest since Loop 15, with clearance 0.523 m, trunk 0.284 m, v_z 2.629,
#   rear_first 0.050 -- and it held the ceiling for 1100 iterations rather than drifting
#   off it immediately. But it collapsed anyway at iteration 23300: v_x 1.48 -> 0.86,
#   falls 2% -> 18%, v_z 2.69 -> 2.44.
#
# Prediction 4 said that if v_z fell anyway, the operating point is intrinsically unstable
#   under resume. It is, and the instrumentation says why:
#
#     Loss/learning_rate   iter 23000: 1e-05      iter 23300: 3.8e-04   (38x)
#     Loss/surrogate       iter 23000: -2.4e-04   iter 23300: -3.8e-03  (15x)
#     the collapse lands between them.
#
#   rsl_rl's "adaptive" schedule moves the LR by 1.5x per update between 1e-5 and 1e-2 to
#   hold a target KL. That is fine for a policy sitting in a basin. This one is not: the
#   take-off has had its knee actuators saturated at 45.43 N.m since Loop 18, so it sits
#   on a boundary, and a step 38x larger than the last one walks straight off it. Every
#   loop since 17 has run well for 700-1200 iterations and then collapsed -- one mechanism,
#   six occurrences.
#
# Change: schedule "adaptive" -> "fixed" at 1.0e-4, roughly the geometric middle of the
#   range the schedule actually used while the runs were healthy. No reward term is
#   touched; the Loop 22 configuration (height targets at the ceiling, asymmetric landing
#   band) is kept exactly as it is.
#
# Prediction
#   1. No mid-run collapse. This is the whole claim: v_x stays near 1.48 and falls stay
#      around 1% for all 2500 iterations, where six consecutive runs broke by 1200.
#   2. v_z holds 2.62-2.70 and foot clearance 0.52-0.60 m.
#   3. Falls settle at or below Loop 22's 0.81%. With a small fixed step the landing terms
#      -- the only live gradient left now that the height terms sit at the ceiling -- get
#      to act without being periodically thrown off.
#   4. If it collapses anyway, the cause is neither the reward nor the learning rate, and
#      the answer is to stop tuning and consolidate on the best checkpoint.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_06-27-36
LOAD_CKPT=model_22100.pt
ITERS=2500   # resume semantics: total = 22100 + 2500 = 24600
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 23 (v22) start: fixed learning rate 1e-4 (was adaptive), no reward changes, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 23 (v22) exited with code ${EXIT} ==="
exit "${EXIT}"
