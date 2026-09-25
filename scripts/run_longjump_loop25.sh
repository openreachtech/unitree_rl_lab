#!/usr/bin/env bash
# Loop 25 (v24) -- Stage B from Loop 24's model_23600, the best policy of the project.
#
# Loop 24 confirmed the mechanism and produced the best landing by a wide margin.
#   entropy_coef 0.01 -> 0.002:
#     entropy      20.7 -> 2.8 (fell, where seven previous runs had it climbing 21 -> 22.4)
#     held         1900 iterations, against 700-1200 for every run since Loop 17
#     falls        0.77% -> **0.05%** (model_23600), a 15x improvement
#     rear_first   0.040 -> **0.012**    success 96.5% -> **97.4%**
#     v_z          2.679 -> **2.731**, the highest of the project
#
#   So the collapses were the entropy bonus. Once both height terms were targeted at the
#   machine ceiling (Loop 22) the task reward went flat there, and a 0.01 entropy bonus was
#   the largest live gradient left -- it widened the action distribution of a policy
#   balanced on a saturated actuator until it fell off.
#
# It did still collapse, at iteration 24900, but from the OTHER side: entropy bottomed at
#   2.8 -- a nearly deterministic policy -- then turned back up and took the policy with it.
#   Too little exploration is its own instability.
#
# Change: entropy_coef 0.002 -> 0.005, between the runaway-up at 0.01 and the
#   collapse-to-2.8 at 0.002. Nothing else.
#
# Prediction
#   1. Entropy settles in the 8-15 band and stays there, rather than climbing past 21 (the
#      0.01 failure) or falling below 4 (the 0.002 failure).
#   2. No collapse in 2500 iterations.
#   3. Falls stay at or below 0.05% and rear_first at or below 0.012 -- these came from the
#      landing terms finally getting an uninterrupted run, so more of the same run should
#      hold them.
#   4. Foot clearance recovers toward 0.55-0.60 m. It is 0.497 m at model_23600 against
#      0.603 m at model_22100 while v_z is HIGHER (2.731 vs 2.692), so the difference is in
#      the tuck rather than in the jump -- there is room there that costs no take-off.

set -uo pipefail
cd /home/tanaka/isaacsim/unitree_rl_lab
export TMPDIR=/home/tanaka/isaacsim/unitree_rl_lab/.tmp
mkdir -p "$TMPDIR"
# shellcheck disable=SC1091
source /home/tanaka/isaacsim/env_isaaclab/bin/activate
LOAD_RUN=2026-09-06_09-30-27
LOAD_CKPT=model_23600.pt
ITERS=2500   # resume semantics: total = 23600 + 2500 = 26100
ts() { date '+%Y-%m-%d %H:%M:%S %Z'; }
echo "=== [$(ts)] Loop 25 (v24) start: entropy_coef 0.002 -> 0.005, from ${LOAD_RUN}/${LOAD_CKPT}, +${ITERS} iters ==="
python scripts/rsl_rl/train.py --task Unitree-Go2-LongJump-v1 --headless --num_envs 4096 \
  --max_iterations "${ITERS}" --resume --load_run "${LOAD_RUN}" --checkpoint "${LOAD_CKPT}" \
  --deploy-keyboard-commands
EXIT=$?
echo "=== [$(ts)] Loop 25 (v24) exited with code ${EXIT} ==="
exit "${EXIT}"
